import numpy as np
import scipy.sparse as sp
from collections import defaultdict, Counter
import zipfile
import urllib.request
import re
import os
import argparse
import math
import sqlite3
import tempfile

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet as wn

try:
    NLTK_STOPWORDS = set(nltk_stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    NLTK_STOPWORDS = set(nltk_stopwords.words("english"))
try:
    # Ensure WordNet is available if vocab projection is requested
    _ = wn.synsets("dog")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


# optional extra junk tokens you almost always want to drop
JUNK_TOKENS = {"s", "t", "http", "https", "www"}


class CooccurrenceMatrix:
    """
    Compute word co-occurrence matrices with custom window sizes.

    Supports:
    - enwik8 / enwik9 download + preprocessing
    - in-memory counting OR disk-backed counting (SQLite)
    - PPMI and shifted-PPMI transforms
    - optional WordNet-filtered vocabulary projection
    """

    def __init__(self, window_start=5, window_end=19, min_count=5):
        self.window_start = window_start
        self.window_end = window_end
        self.min_count = min_count
        self.vocab = None
        self.word2idx = None
        self.cooc_matrix = None
        self.word_counts = None
        self.total_cooccurrences = 0
        self.ppmi_matrix = None

    # ----------------------------
    # Data: enwik8 / enwik9
    # ----------------------------
    def download_wiki(self, corpus="enwik8", data_dir="./data"):
        """Download and extract enwik8 or enwik9 (Matt Mahoney)."""
        if corpus not in {"enwik8", "enwik9"}:
            raise ValueError("corpus must be 'enwik8' or 'enwik9'")

        os.makedirs(data_dir, exist_ok=True)

        zip_name = f"{corpus}.zip"
        zip_path = os.path.join(data_dir, zip_name)
        extract_path = os.path.join(data_dir, corpus)

        if not os.path.exists(extract_path):
            if not os.path.exists(zip_path):
                print(f"Downloading {corpus}...")
                url = f"http://mattmahoney.net/dc/{zip_name}"
                urllib.request.urlretrieve(url, zip_path)
                print("Download complete!")

            print("Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Extracted to {extract_path}")

        return extract_path

    def preprocess_wiki(self, filepath, max_chars=None):
        """
        Clean enwik8/enwik9 XML-ish text to extract readable tokens.

        Args:
            filepath: path to extracted corpus file (e.g., ./data/enwik8)
            max_chars: limit processing to first N characters (for testing)

        Returns:
            list[str]: overlapped chunks (~200 words) for co-occurrence windows
        """
        print(f"Reading {filepath}...")
        with open(filepath, "rb") as f:
            data = f.read()
            if max_chars:
                data = data[:max_chars]

        text = data.decode("utf-8", errors="ignore")

        print("Cleaning text...")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)  # [[link|text]] -> text
        text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)             # [[link]] -> link
        text = re.sub(r"\{\{[^\}]+\}\}", " ", text)                 # {{templates}}
        text = re.sub(r"&[a-z]+;", " ", text)                       # HTML entities

        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        words = text.split()

        chunks = []
        chunk_size = 200
        step = chunk_size // 2  # 50% overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) > 20:
                chunks.append(" ".join(chunk_words))

        print(f"Created {len(chunks)} text chunks")
        return chunks

    # ----------------------------
    # Vocabulary filtering
    # ----------------------------
    def _wordnet_ok(self, word, pos=None, cache=None):
        """
        True if word has at least one WordNet synset (optionally constrained by POS).
        Cached for speed.
        """
        if cache is None:
            cache = {}
        key = (word, pos)
        if key in cache:
            return cache[key]
        syns = wn.synsets(word, pos=pos) if pos else wn.synsets(word)
        ok = len(syns) > 0
        cache[key] = ok
        return ok

    def _build_vocab(self, tokenized_docs, verbose=True, wn_filter=False, wn_pos=None):
        # Count words
        all_tokens = []
        for tokens in tokenized_docs:
            all_tokens.extend(tokens)

        word_counts = Counter(all_tokens)
        self.word_counts = word_counts

        # min_count filter
        vocab = [w for w, c in word_counts.items() if c >= self.min_count]
        vocab = sorted(vocab)

        if wn_filter:
            if verbose:
                print(f"Applying WordNet vocab projection (pos={wn_pos or 'any'})...")
            cache = {}
            keep = []
            for i, w in enumerate(vocab):
                if verbose and i % 20000 == 0 and i > 0:
                    print(f"  WordNet check {i}/{len(vocab)}")
                if self._wordnet_ok(w, pos=wn_pos, cache=cache):
                    keep.append(w)
            vocab = keep
            if verbose:
                print(f"WordNet-filtered vocab size: {len(vocab):,}")

        self.vocab = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}

        if verbose:
            print(f"Vocabulary size: {len(vocab):,}")
            print(f"Total tokens: {len(all_tokens):,}")

        return vocab

    # ----------------------------
    # Disk-backed co-occurrence counting (SQLite)
    # ----------------------------
    def _sqlite_init(self, db_path):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA cache_size=-200000;")  # ~200MB cache if available
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cooc (
                i INTEGER NOT NULL,
                j INTEGER NOT NULL,
                c INTEGER NOT NULL,
                PRIMARY KEY (i, j)
            );
        """)
        con.commit()
        return con

    def _sqlite_upsert_counts(self, con, pair_counts):
        """
        pair_counts: dict[(i,j)] -> count
        """
        if not pair_counts:
            return
        cur = con.cursor()
        rows = [(i, j, int(c)) for (i, j), c in pair_counts.items()]
        cur.executemany(
            """
            INSERT INTO cooc(i, j, c) VALUES(?, ?, ?)
            ON CONFLICT(i, j) DO UPDATE SET c = c + excluded.c
            """,
            rows
        )
        con.commit()

    def _build_csr_from_sqlite(self, con, vocab_size, verbose=True):
        cur = con.cursor()
        # fetch all pairs; still ends in RAM as CSR eventually lives in RAM
        cur.execute("SELECT i, j, c FROM cooc;")

        rows, cols, data = [], [], []
        n = 0
        while True:
            batch = cur.fetchmany(200000)
            if not batch:
                break
            for i, j, c in batch:
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([c, c])
            n += len(batch)
            if verbose and n % 1000000 == 0:
                print(f"  Read {n:,} unique pairs from disk...")

        mat = sp.csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
        return mat

    # ----------------------------
    # Main fitting
    # ----------------------------
    def fit(
        self,
        corpus,
        verbose=True,
        disk_backed=False,
        sqlite_path=None,
        flush_every=2_000_000,
        wn_filter=False,
        wn_pos=None,
    ):
        """
        Build co-occurrence matrix from corpus.

        Args:
            corpus: list[str]
            disk_backed: if True, count co-occurrences in SQLite (lower RAM, slower)
            sqlite_path: optional path for sqlite db; default temp file
            flush_every: approx number of observed pairs before flushing batch to disk
            wn_filter: if True, keep only tokens that exist in WordNet
            wn_pos: WordNet POS in {'n','v','a','r','s'} or None for any
        """
        if verbose:
            print("Step 0: Tokenizing...")
        tokenized_docs = []
        for i, doc in enumerate(corpus):
            if verbose and i % 1000 == 0:
                print(f"  Processing document {i}/{len(corpus)}")
            tokenized_docs.append(doc.split())

        if verbose:
            print("Step 1: Counting words / building vocab...")
        self._build_vocab(tokenized_docs, verbose=verbose, wn_filter=wn_filter, wn_pos=wn_pos)
        vocab_size = len(self.vocab)

        if verbose:
            print(f"\nStep 2: Computing co-occurrences (window {self.window_start}-{self.window_end})...")

        if disk_backed:
            if sqlite_path is None:
                fd, tmp = tempfile.mkstemp(prefix="cooc_", suffix=".sqlite3")
                os.close(fd)
                sqlite_path = tmp
            if verbose:
                print(f"Using disk-backed counting (SQLite): {sqlite_path}")
            con = self._sqlite_init(sqlite_path)

            buf = defaultdict(int)
            seen = 0

            for doc_idx, tokens in enumerate(tokenized_docs):
                if verbose and doc_idx % 1000 == 0:
                    print(f"  Document {doc_idx}/{len(tokenized_docs)} (buffer {len(buf):,})")

                for i, target_word in enumerate(tokens):
                    ti = self.word2idx.get(target_word)
                    if ti is None:
                        continue

                    lo = max(0, i - self.window_end)
                    hi = min(len(tokens), i + self.window_end + 1)
                    for j in range(lo, hi):
                        if i == j:
                            continue
                        d = abs(i - j)
                        if d < self.window_start or d > self.window_end:
                            continue
                        cj = self.word2idx.get(tokens[j])
                        if cj is None:
                            continue
                        a, b = (ti, cj) if ti < cj else (cj, ti)
                        buf[(a, b)] += 1
                        seen += 1

                        if seen >= flush_every:
                            # aggregate already in buf; just upsert then clear
                            if verbose:
                                print(f"    Flushing {len(buf):,} pairs to disk...")
                            self._sqlite_upsert_counts(con, buf)
                            buf.clear()
                            seen = 0

            if buf:
                if verbose:
                    print(f"    Final flush {len(buf):,} pairs to disk...")
                self._sqlite_upsert_counts(con, buf)
                buf.clear()

            if verbose:
                print("Step 3: Building sparse matrix from SQLite...")
            self.cooc_matrix = self._build_csr_from_sqlite(con, vocab_size, verbose=verbose)
            con.close()

        else:
            cooc_counts = defaultdict(int)
            for doc_idx, tokens in enumerate(tokenized_docs):
                if verbose and doc_idx % 1000 == 0:
                    print(f"  Document {doc_idx}/{len(tokenized_docs)}, {len(cooc_counts):,} pairs found")

                for i, target_word in enumerate(tokens):
                    ti = self.word2idx.get(target_word)
                    if ti is None:
                        continue

                    lo = max(0, i - self.window_end)
                    hi = min(len(tokens), i + self.window_end + 1)
                    for j in range(lo, hi):
                        if i == j:
                            continue
                        d = abs(i - j)
                        if d < self.window_start or d > self.window_end:
                            continue
                        cj = self.word2idx.get(tokens[j])
                        if cj is None:
                            continue
                        a, b = (ti, cj) if ti < cj else (cj, ti)
                        cooc_counts[(a, b)] += 1

            if verbose:
                print(f"Total unique co-occurrence pairs: {len(cooc_counts):,}")

            if verbose:
                print("\nStep 3: Building sparse matrix...")
            rows, cols, data = [], [], []
            for (i, j), count in cooc_counts.items():
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([count, count])

            self.cooc_matrix = sp.csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))

        self.total_cooccurrences = int(self.cooc_matrix.sum())
        if verbose:
            print(f"Matrix shape: {self.cooc_matrix.shape}")
            print(f"Non-zero entries: {self.cooc_matrix.nnz:,}")
            print("Done!\n")
        return self

    # ----------------------------
    # PPMI / shifted PMI
    # ----------------------------
    def compute_ppmi(self, shift=1.0, eps=1e-12, verbose=True):
        """
        Compute PPMI or shifted-PPMI.

        PMI(i,j) = log( (X_ij * X..) / (X_i. * X_.j) )
        shifted PPMI = max(PMI - log(shift), 0), shift>=1 is common (e.g., negative sampling k)

        Args:
            shift: >=1.0; use 1.0 for plain PPMI, k for shifted-PPMI
            eps: numerical stabilizer
        """
        if self.cooc_matrix is None:
            raise ValueError("cooc_matrix is not built yet. Run fit() or load().")
        if shift <= 0:
            raise ValueError("shift must be > 0")

        if verbose:
            print(f"Computing {'shifted-' if shift != 1.0 else ''}PPMI (shift={shift})...")

        X = self.cooc_matrix.tocoo(copy=False)
        data = X.data.astype(np.float64)

        row_sum = np.asarray(self.cooc_matrix.sum(axis=1)).ravel().astype(np.float64)
        # symmetric -> col_sum same, but keep general formula
        col_sum = np.asarray(self.cooc_matrix.sum(axis=0)).ravel().astype(np.float64)
        total = float(row_sum.sum())

        # PMI on nonzeros only
        num = data * total
        denom = (row_sum[X.row] * col_sum[X.col]) + eps
        pmi = np.log((num + eps) / denom)

        if shift != 1.0:
            pmi = pmi - math.log(shift)

        ppmi = np.maximum(pmi, 0.0)
        self.ppmi_matrix = sp.csr_matrix((ppmi, (X.row, X.col)), shape=self.cooc_matrix.shape)
        if verbose:
            nnz = self.ppmi_matrix.nnz
            print(f"PPMI nnz: {nnz:,}")
        return self.ppmi_matrix

    # ----------------------------
    # Queries / helpers
    # ----------------------------
    def get_cooccurrences(
        self,
        word,
        top_k=10,
        remove_stopwords=True,
        min_token_len=2,
        exclude_self=True,
        use_ppmi=False,
    ):
        if self.word2idx is None or self.cooc_matrix is None:
            raise ValueError("Matrix not loaded/fitted.")

        if word not in self.word2idx:
            return []

        stop = NLTK_STOPWORDS
        idx = self.word2idx[word]

        mat = self.ppmi_matrix if use_ppmi else self.cooc_matrix
        if mat is None:
            raise ValueError("PPMI matrix not computed; call compute_ppmi() or set use_ppmi=False.")

        row = mat[idx].toarray().flatten()
        candidate_k = max(top_k * 50, 200)
        top_indices = np.argsort(row)[::-1][:candidate_k]

        results = []
        for i in top_indices:
            val = float(row[i])
            if val <= 0:
                break

            w = self.vocab[i]
            wl = w.lower()

            if exclude_self and wl == word.lower():
                continue
            if len(wl) < min_token_len:
                continue
            if wl in JUNK_TOKENS:
                continue
            if remove_stopwords and wl in stop:
                continue

            word_freq = int(self.word_counts.get(w, 0)) if self.word_counts is not None else 0
            results.append((w, val, word_freq))

            if len(results) >= top_k:
                break

        return results

    def get_frequency(self, word1, word2):
        """Get co-occurrence frequency between two words."""
        if self.word2idx is None or self.cooc_matrix is None:
            raise ValueError("Matrix not loaded/fitted.")

        if word1 not in self.word2idx or word2 not in self.word2idx:
            return 0

        idx1 = self.word2idx[word1]
        idx2 = self.word2idx[word2]
        return int(self.cooc_matrix[idx1, idx2])

    def query_pair(self, word1, word2, mode="cooc", prob_mode="global"):
        """
        Query a custom pair.

        mode:
        - "cooc" -> return raw co-occurrence count
        - "prob" -> return probability

        prob_mode (only used if mode="prob"):
        - "global": P(w1,w2) = cooc(w1,w2) / sum_all_pairs
        - "cond":   P(w2|w1) = cooc(w1,w2) / sum_row(w1)
        """
        if self.word2idx is None or self.cooc_matrix is None:
            raise ValueError("Matrix not loaded/fitted.")

        if word1 not in self.word2idx or word2 not in self.word2idx:
            return None

        i, j = self.word2idx[word1], self.word2idx[word2]
        c = int(self.cooc_matrix[i, j])

        if mode == "cooc":
            return c

        if mode != "prob":
            raise ValueError("mode must be 'cooc' or 'prob'.")

        if prob_mode == "global":
            total = self.total_cooccurrences if self.total_cooccurrences else int(self.cooc_matrix.sum())
            return float(c / total) if total > 0 else 0.0

        if prob_mode == "cond":
            row_sum = float(self.cooc_matrix.getrow(i).sum())
            return float(c / row_sum) if row_sum > 0 else 0.0

        raise ValueError("prob_mode must be 'global' or 'cond'.")

    # ----------------------------
    # Save / load
    # ----------------------------
    def save(self, filepath=None):
        """Save co-occurrence matrix and vocabulary."""
        if self.cooc_matrix is None:
            raise ValueError("Nothing to save; build/load a matrix first.")

        if filepath is None:
            filepath = f"wiki_cooc_w{self.window_start}-{self.window_end}_minfreq{self.min_count}.npz"

        np.savez_compressed(
            filepath,
            data=self.cooc_matrix.data,
            indices=self.cooc_matrix.indices,
            indptr=self.cooc_matrix.indptr,
            shape=self.cooc_matrix.shape,
            vocab=self.vocab,
            word_counts=dict(self.word_counts) if self.word_counts is not None else {},
            window_start=self.window_start,
            window_end=self.window_end,
            min_count=self.min_count,
            total_cooccurrences=int(self.cooc_matrix.sum()),
        )
        print(f"Saved to {filepath}")
        return filepath

    def load(self, filepath):
        """Load co-occurrence matrix and vocabulary."""
        npz = np.load(filepath, allow_pickle=True)
        self.cooc_matrix = sp.csr_matrix(
            (npz["data"], npz["indices"], npz["indptr"]),
            shape=tuple(npz["shape"]),
        )
        self.vocab = npz["vocab"].tolist()
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.word_counts = Counter(npz["word_counts"].item()) if "word_counts" in npz else Counter()
        self.window_start = int(npz["window_start"])
        self.window_end = int(npz["window_end"])
        self.min_count = int(npz["min_count"]) if "min_count" in npz else 5
        self.total_cooccurrences = int(npz["total_cooccurrences"]) if "total_cooccurrences" in npz else int(self.cooc_matrix.sum())

        print(f"Loaded from {filepath}")
        print(f"Window: {self.window_start}-{self.window_end}, Min frequency: {self.min_count}")
        print(f"Vocabulary size: {len(self.vocab):,}")
        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build word co-occurrence matrix from enwik8/enwik9 corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--corpus", type=str, choices=["enwik8", "enwik9"], default="enwik8",
                        help="Which Wikipedia dump to use")
    parser.add_argument("--window-start", type=int, default=5,
                        help="Minimum distance between words for co-occurrence")
    parser.add_argument("--window-end", type=int, default=19,
                        help="Maximum distance between words for co-occurrence")
    parser.add_argument("--min-freq", type=int, default=10,
                        help="Minimum word frequency to include in vocabulary")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Process only first N characters (for testing). Use None for full corpus")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory for downloading/storing enwik8/enwik9")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filepath (auto-generated if not specified)")
    parser.add_argument("--load", type=str, default=None,
                        help="Load existing matrix file and run queries")

    # New: disk-backed counting
    parser.add_argument("--disk-backed", action="store_true",
                        help="Use SQLite on-disk accumulator to reduce RAM during counting (slower)")
    parser.add_argument("--sqlite-path", type=str, default=None,
                        help="Path to sqlite db file (only if --disk-backed). Default: temp file")
    parser.add_argument("--flush-every", type=int, default=2_000_000,
                        help="Flush buffered pair updates to disk every N observed pairs")

    # New: WordNet vocab projection
    parser.add_argument("--wn-filter", action="store_true",
                        help="Filter vocabulary to tokens with WordNet synsets")
    parser.add_argument("--wn-pos", type=str, default=None, choices=[None, "n", "v", "a", "r", "s"],
                        help="WordNet POS for filtering (n,v,a,r,s). Default: any POS")

    # New: PPMI / shifted PPMI
    parser.add_argument("--ppmi", action="store_true",
                        help="Compute PPMI matrix after building/loading cooc matrix")
    parser.add_argument("--shift", type=float, default=1.0,
                        help="Shift for shifted-PPMI (k). Use 1.0 for plain PPMI")

    args = parser.parse_args()

    if args.load:
        print(f"Loading existing matrix from {args.load}")
        cooc = CooccurrenceMatrix()
        cooc.load(args.load)
    else:
        print("Building co-occurrence matrix:")
        print(f"  Corpus: {args.corpus}")
        print(f"  Window: {args.window_start}-{args.window_end}")
        print(f"  Minimum frequency: {args.min_freq}")
        print(f"  Max chars: {args.max_chars if args.max_chars else 'All'}")
        print(f"  Disk-backed: {args.disk_backed}")
        if args.wn_filter:
            print(f"  WordNet vocab filter: ON (pos={args.wn_pos or 'any'})")
        print()

        cooc = CooccurrenceMatrix(
            window_start=args.window_start,
            window_end=args.window_end,
            min_count=args.min_freq,
        )

        wiki_path = cooc.download_wiki(corpus=args.corpus, data_dir=args.data_dir)
        corpus = cooc.preprocess_wiki(wiki_path, max_chars=args.max_chars)

        cooc.fit(
            corpus,
            disk_backed=args.disk_backed,
            sqlite_path=args.sqlite_path,
            flush_every=args.flush_every,
            wn_filter=args.wn_filter,
            wn_pos=args.wn_pos,
        )

        output_file = cooc.save(args.output)

    if args.ppmi:
        cooc.compute_ppmi(shift=args.shift)

    print("\n" + "=" * 60)
    print("QUERY EXAMPLES")
    print("=" * 60)

    test_words = ["king", "queen", "computer", "science", "algorithm"]
    for word in test_words:
        print(f"\nTop 10 co-occurrences for '{word}':")
        results = cooc.get_cooccurrences(word, top_k=10, use_ppmi=False)
        if results:
            for coword, val, word_freq in results:
                print(f"  {coword:20s} - co-occur: {int(val):6d}, total freq: {word_freq:6d}")
        else:
            print(f"  '{word}' not in vocabulary")

    if args.ppmi and cooc.ppmi_matrix is not None:
        print("\n" + "=" * 60)
        print("PPMI QUERY EXAMPLES")
        print("=" * 60)
        for word in test_words:
            print(f"\nTop 10 PPMI associates for '{word}':")
            results = cooc.get_cooccurrences(word, top_k=10, use_ppmi=True, remove_stopwords=True)
            if results:
                for coword, val, word_freq in results:
                    print(f"  {coword:20s} - ppmi: {val:8.4f}, total freq: {word_freq:6d}")
            else:
                print(f"  '{word}' not in vocabulary")

    print("\n" + "=" * 60)
    print("WORD PAIR FREQUENCIES")
    print("=" * 60)
    pairs = [("king", "queen"), ("computer", "science"), ("data", "structure")]
    for w1, w2 in pairs:
        freq = cooc.get_frequency(w1, w2)
        print(f"{w1} <-> {w2}: {freq} co-occurrences")
