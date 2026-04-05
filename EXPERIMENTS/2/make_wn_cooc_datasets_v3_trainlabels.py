
"""
make_wn_cooc_datasets.py

Builds 4 feature datasets for (hypo, hyper) pairs:

1) X_freq_* : sparse (CSR) concatenation of
      [ cooc_row(hypo_token) , cooc_col(hyper_token) ]
   where values are raw co-occurrence frequencies from the enwik8 .npz matrix.

2) X_prob333_* : sparse (CSR) 2*n_bins interval-structured vector where
      first  n_bins  (hyper half) is scaled by conditional prob  P(hyper_token|hypo_token)
      second n_bins  (hypo  half) is scaled by joint prob       P(hypo_token,hyper_token)

   The *support* (which bins are nonzero) comes from the WordNet interval encoder.

3) X_int_* : interval features (dense float32 or CSR) from the WordNet encoder:
      use build_all_features() to get canonical interval features, then rebuild
      richer structure via build_interval_pair_features(hypo_mode="path", hyper_mode="children").

4) X_int_freq_* : same as (3) but intervals are computed with frequency-weighted
      subdivision using corpus-derived synset weights from the .npz.

This file depends on your attached modules:
- wn_cooc_feature_pipeline_v7.py
- WN_arithmetic_interval_encoding8_5.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterable, List

import numpy as np
import pandas as pd
import scipy.sparse as sp

from nltk.corpus import wordnet as wn

# ---- attached pipeline helpers (v7) ----
from wn_cooc_feature_pipeline_v7 import (
    CoocBackend,
    build_all_features,
    build_interval_pair_features,
    build_cooc_pair_features,
    resolve_pair_synsets,
    tokens_from_spec_or_synset,
)


# ---------------------------
# Train-derived test labels (graph over positive edges)
# ---------------------------

from collections import defaultdict, deque

def _build_pos_graph(train_pos_df: pd.DataFrame, *, hypo_col: str = "hypo", hyper_col: str = "hyper") -> Dict[str, set]:
    """
    Directed adjacency: hypo -> {hyper1, hyper2, ...} using ONLY positive edges.
    """
    adj: Dict[str, set] = defaultdict(set)
    for _, r in train_pos_df.iterrows():
        h = str(r[hypo_col])
        H = str(r[hyper_col])
        adj[h].add(H)
    return adj

def _ancestors_of(node: str, adj: Dict[str, set], *, max_depth: Optional[int] = None) -> set:
    """
    Return all reachable hypernyms from `node` via adj edges (node -> hyper ...).
    """
    seen = set()
    q = deque([(node, 0)])
    while q:
        cur, d = q.popleft()
        if max_depth is not None and d >= max_depth:
            continue
        for nxt in adj.get(cur, ()):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, d + 1))
    return seen

def label_test_train_direct(hypo: str, hyper: str, adj: Dict[str, set]) -> int:
    """1 iff direct positive edge hypo -> hyper exists in TRAIN."""
    return 1 if str(hyper) in adj.get(str(hypo), set()) else 0

def label_test_train_rowpath(hypo: str, hyper: str, adj: Dict[str, set], *, max_depth: Optional[int] = None) -> int:
    """1 iff hyper is reachable from hypo via TRAIN positive edges."""
    hypo = str(hypo); hyper = str(hyper)
    return 1 if hyper in _ancestors_of(hypo, adj, max_depth=max_depth) else 0

def label_test_train_colpath(hypo: str, hyper: str, adj: Dict[str, set], *, max_depth: Optional[int] = None) -> int:
    """
    Column-path match (your described mechanism):

      ColPath(hyper) = {hyper} ∪ Ancestors(hyper)   (computed from TRAIN graph)
      y=1 iff HyperEdges(hypo) ∩ ColPath(hyper) ≠ ∅
    """
    hypo = str(hypo); hyper = str(hyper)
    hypers_of_hypo = adj.get(hypo, set())
    if not hypers_of_hypo:
        return 0
    col_path = {hyper} | _ancestors_of(hyper, adj, max_depth=max_depth)
    return 1 if len(hypers_of_hypo.intersection(col_path)) > 0 else 0

def build_test_labels_from_train(
    test_df: pd.DataFrame,
    train_df_pos: pd.DataFrame,
    *,
    hypo_col: str = "hypo",
    hyper_col: str = "hyper",
    max_depth: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns multiple train-derived label variants for the TEST set.
    """
    adj = _build_pos_graph(train_df_pos, hypo_col=hypo_col, hyper_col=hyper_col)

    y_direct = []
    y_rowpath = []
    y_colpath = []
    for h, H in zip(test_df[hypo_col].astype(str), test_df[hyper_col].astype(str)):
        y_direct.append(label_test_train_direct(h, H, adj))
        y_rowpath.append(label_test_train_rowpath(h, H, adj, max_depth=max_depth))
        y_colpath.append(label_test_train_colpath(h, H, adj, max_depth=max_depth))

    y_direct = np.asarray(y_direct, dtype=np.int64)
    y_rowpath = np.asarray(y_rowpath, dtype=np.int64)
    y_colpath = np.asarray(y_colpath, dtype=np.int64)
    y_union = ((y_direct == 1) | (y_rowpath == 1) | (y_colpath == 1)).astype(np.int64)

    return {
        "y_test_train_direct": y_direct,
        "y_test_train_rowpath": y_rowpath,
        "y_test_train_colpath": y_colpath,
        "y_test_train_union": y_union,
    }

# ---- attached WordNet interval encoder ----
from WN_arithmetic_interval_encoding8_5 import WordNetIntervalEncoder


# -------------------------
# Small utilities
# -------------------------

def _safe_float(x: Optional[float]) -> float:
    return float(x) if x is not None else 0.0

def _safe_int(x: Optional[int]) -> int:
    return int(x) if x is not None else 0

def _alias_root_type(root_type: str) -> str:
    """
    Map user-friendly root types to WordNet POS letters.
      all -> "all"
      noun -> "n"
      verb -> "v"
      adj  -> "a"
      prep -> "r"   (WordNet doesn't have a dedicated preposition POS; treat as adverb bucket)
    """
    rt = (root_type or "all").strip().lower()
    mapping = {"all": "all", "noun": "n", "n": "n", "verb": "v", "v": "v",
               "adj": "a", "a": "a", "prep": "r", "adv": "r", "r": "r"}
    return mapping.get(rt, rt)


def _pick_token_pair_max(
    hypo_spec: str,
    hyper_spec: str,
    *,
    encoder: WordNetIntervalEncoder,
    cooc: CoocBackend,
    prefer_pos: str = "n",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Pick a (w1,w2) token pair to query the cooc matrix, trying all token candidates and
    choosing the pair with maximum raw cooc count.

    Returns (None,None) if no candidate token is in vocab.
    """
    hypo_syn, hyper_syn = resolve_pair_synsets(hypo_spec, hyper_spec, encoder, prefer_pos=prefer_pos)

    if hypo_syn is None or hyper_syn is None:
        htoks = [t for t in str(hypo_spec).strip().lower().split() if t]
        Htoks = [t for t in str(hyper_spec).strip().lower().split() if t]
    else:
        htoks = tokens_from_spec_or_synset(hypo_spec, hypo_syn)
        Htoks = tokens_from_spec_or_synset(hyper_spec, hyper_syn)

    if not htoks or not Htoks:
        return None, None

    best_pair = (None, None)
    best_f = -1
    for w1 in htoks:
        if w1 not in cooc.word2idx:
            continue
        for w2 in Htoks:
            if w2 not in cooc.word2idx:
                continue
            f = cooc.cooc_count(w1, w2)
            if f is None:
                continue
            if f > best_f:
                best_f = int(f)
                best_pair = (w1, w2)

    return best_pair


def _pick_single_token(
    spec: str,
    syn: Optional["wn.synset"],
    *,
    cooc: CoocBackend,
    banned: Optional[set] = None,
    strategy: str = "max_marginal",
    backoff: str = "wn_hypernym",
    max_backoff_steps: int = 5,
) -> Optional[str]:
    '''
    Choose ONE token in `cooc` vocab for a word/synset spec.

    Candidates are:
      - tokens_from_spec_or_synset(spec, syn)  (includes lemma tokens)
      - whitespace tokens from `spec`

    strategy:
      - "max_marginal": pick token with largest cooc row-sum (proxy frequency)
      - "first": first in-vocab token

    backoff:
      - "none": no backoff; return None if OOV
      - "wn_hypernym": if OOV and `syn` is available, walk up hypernyms and
        pick the first ancestor lemma token that exists in vocab (using `strategy`)
        up to `max_backoff_steps`.

    banned: tokens to avoid (e.g., don't pick the same token as the other side).
    '''
    banned = banned or set()
    spec = str(spec).strip()

    def _dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for t in seq:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    def _select(in_vocab: List[str]) -> Optional[str]:
        if not in_vocab:
            return None
        if strategy == "first":
            return in_vocab[0]
        if strategy == "max_marginal":
            best_t = None
            best_score = -1.0
            for t in in_vocab:
                i = cooc.word2idx[t]
                score = float(cooc.cooc.getrow(i).sum())
                if score > best_score:
                    best_score = score
                    best_t = t
            return best_t
        raise ValueError("strategy must be 'max_marginal' or 'first'")

    # ---- primary candidates (surface + lemma tokens) ----
    cands: List[str] = []
    if syn is not None:
        try:
            cands.extend(tokens_from_spec_or_synset(spec, syn))
        except Exception:
            pass
    cands.extend([t for t in spec.lower().replace("_", " ").split() if t])
    cands = _dedup(cands)

    in_vocab = [t for t in cands if (t in cooc.word2idx and t not in banned)]
    chosen = _select(in_vocab)
    if chosen is not None:
        return chosen

    # ---- backoff: walk up hypernyms (semantic backoff) ----
    if backoff == "wn_hypernym" and syn is not None:
        cur = syn
        steps = 0
        while steps < max_backoff_steps:
            hypers = cur.hypernyms() + cur.instance_hypernyms()
            if not hypers:
                break
            # deterministic: follow the first hypernym
            cur = hypers[0]
            steps += 1

            hcands: List[str] = []
            try:
                hcands.extend(tokens_from_spec_or_synset(cur.name(), cur))
            except Exception:
                pass
            lemma = cur.name().split(".")[0].replace("_", " ")
            hcands.extend([t for t in lemma.split() if t])
            hcands = _dedup([t.lower() for t in hcands])

            hin_vocab = [t for t in hcands if (t in cooc.word2idx and t not in banned)]
            chosen = _select(hin_vocab)
            if chosen is not None:
                return chosen

    return None


# -------------------------
# Dataset (1): raw frequency sparse embeddings
# -------------------------

def build_sparse_rawfreq_pair_embeddings(
    df_pairs: pd.DataFrame,
    *,
    encoder: WordNetIntervalEncoder,
    cooc: CoocBackend,
    prefer_pos: str = "n",
    hypo_col: str = "hypo",
    hyper_col: str = "hyper",
    token_strategy: str = "max_marginal",
) -> sp.csr_matrix:
    """
    For each (hypo,hyper) pair produce a sparse feature vector:
        [ row(hypo_token) , col(hyper_token) ]
    values = raw co-occurrence frequencies.

    IMPORTANT: Unlike v1, this DOES NOT require both tokens to be in-vocab.
    If only one side has an in-vocab token, you still get a partially nonzero vector.

    Output shape: (m, 2*|V|)
    """
    V = len(cooc.vocab)
    coocT = cooc.cooc.transpose().tocsr()

    rows: List[sp.csr_matrix] = []
    zero_row = sp.csr_matrix((1, V), dtype=np.float32)
    zero_full = sp.csr_matrix((1, 2 * V), dtype=np.float32)

    for row in df_pairs.itertuples(index=False):
        hypo_spec = getattr(row, hypo_col)
        hyper_spec = getattr(row, hyper_col)

        hypo_syn, hyper_syn = resolve_pair_synsets(str(hypo_spec), str(hyper_spec), encoder, prefer_pos=prefer_pos)

        # pick tokens independently (prevents "all-zero row" when one side is OOV)
        hyper_tok = _pick_single_token(str(hyper_spec), hyper_syn, cooc=cooc, banned=set(), strategy=token_strategy)
        hypo_tok = _pick_single_token(str(hypo_spec), hypo_syn, cooc=cooc, banned={hyper_tok} if hyper_tok else set(), strategy=token_strategy)

        if hypo_tok is None and hyper_tok is None:
            rows.append(zero_full)
            continue

        if hypo_tok is None:
            r_h = zero_row
        else:
            i = cooc.word2idx[hypo_tok]
            r_h = cooc.cooc.getrow(i).astype(np.float32)

        if hyper_tok is None:
            c_H = zero_row
        else:
            j = cooc.word2idx[hyper_tok]
            c_H = coocT.getrow(j).astype(np.float32)

        rows.append(sp.hstack([r_h, c_H], format="csr"))

    return sp.vstack(rows, format="csr")




def report_oov_rows(
    df_pairs: pd.DataFrame,
    *,
    encoder: WordNetIntervalEncoder,
    cooc: CoocBackend,
    prefer_pos: str = "n",
    hypo_col: str = "hypo",
    hyper_col: str = "hyper",
    token_strategy: str = "max_marginal",
    max_show: int = 20,
) -> pd.DataFrame:
    """
    Return a DataFrame listing rows where at least one side has no in-vocab token candidate.
    Useful to explain why some rows were all-zero in the old builder.
    """
    records = []
    for idx, row in enumerate(df_pairs.itertuples(index=False)):
        hypo_spec = str(getattr(row, hypo_col))
        hyper_spec = str(getattr(row, hyper_col))
        hs, Hs = resolve_pair_synsets(hypo_spec, hyper_spec, encoder, prefer_pos=prefer_pos)
        hyper_tok = _pick_single_token(hyper_spec, Hs, cooc=cooc, strategy=token_strategy)
        hypo_tok  = _pick_single_token(hypo_spec, hs, cooc=cooc, banned={hyper_tok} if hyper_tok else set(), strategy=token_strategy)

        if hypo_tok is None or hyper_tok is None:
            records.append({
                "row": idx,
                "hypo": hypo_spec,
                "hyper": hyper_spec,
                "hypo_tok": hypo_tok,
                "hyper_tok": hyper_tok,
                "hypo_in_vocab": hypo_tok is not None,
                "hyper_in_vocab": hyper_tok is not None,
            })
            if len(records) >= max_show:
                break
    return pd.DataFrame.from_records(records)
# -------------------------
# Dataset (2): interval-structured + corpus-prob weighted values
# -------------------------

def build_interval_features_probweighted(
    df_pairs: pd.DataFrame,
    *,
    encoder: WordNetIntervalEncoder,
    cooc: CoocBackend,
    n_bins: int = 333,
    vec_mode: str = "multiscale",
    prefer_pos: str = "n",
    hypo_mode: str = "path",
    hyper_mode: str = "children",
    children_k: int = 25,
    hypo_col: str = "hypo",
    hyper_col: str = "hyper",
) -> sp.csr_matrix:
    """
    Support comes from build_interval_pair_features(); values are scaled by corpus probs:
      - hyper half [:n_bins]   *= P(hyper_token | hypo_token)
      - hypo  half [n_bins:]   *= P(hypo_token, hyper_token)
    """
    # interval support (binary/weighted by interval width depending on vec_mode in pipeline)
    X = build_interval_pair_features(
        df_pairs,
        encoder,
        n_bins=n_bins,
        vec_mode=vec_mode,
        prefer_pos=prefer_pos,
        hypo_mode=hypo_mode,
        hyper_mode=hyper_mode,
        children_k=children_k,
        hypo_col=hypo_col,
        hyper_col=hyper_col,
    ).astype(np.float32)

    # cooc probabilities per pair
    Xcooc = build_cooc_pair_features(
        df_pairs, cooc, encoder, token_strategy="max", prefer_pos=prefer_pos,
        hypo_col=hypo_col, hyper_col=hyper_col
    )
    # Xcooc[:,1] = joint, Xcooc[:,2] = conditional
    jp = Xcooc[:, 1].astype(np.float32)
    cp = Xcooc[:, 2].astype(np.float32)

    X[:, :n_bins] *= cp[:, None]
    X[:, n_bins:] *= jp[:, None]

    return sp.csr_matrix(X)


# -------------------------
# Dataset (4): frequency-weighted intervals (synset weights from corpus)
# -------------------------

class SynsetFreqDict:
    """
    Dict-like object with a .get(name, default) API, computing a synset's weight on demand
    from corpus token frequencies.

    Weight heuristic:
      weight(synset) = max_{token in lemma_tokens} token_marginal(token)
    where token_marginal(token) is approximated as sum of its co-occurrence row.
    """
    def __init__(self, cooc: CoocBackend, word_counts: Optional[Dict[str, int]] = None):
        self.cooc = cooc
        self.word_counts = word_counts
        self._cache: Dict[str, float] = {}

    def get(self, synset_name: str, default: float = 1.0) -> float:
        if synset_name in self._cache:
            return self._cache[synset_name]

        try:
            syn = wn.synset(synset_name)
        except Exception:
            self._cache[synset_name] = float(default)
            return float(default)

        lemma = syn.name().split(".")[0]
        toks = [t for t in lemma.replace("_", " ").split() if t]
        if not toks:
            self._cache[synset_name] = float(default)
            return float(default)

        best = 0.0
        for t in toks:
            if self.word_counts is not None and t in self.word_counts:
                best = max(best, float(self.word_counts[t]))
                continue
            i = self.cooc.word2idx.get(t)
            if i is None:
                continue
            # marginal ~ row sum
            best = max(best, float(self.cooc.cooc.getrow(i).sum()))

        if best <= 0.0:
            best = float(default)

        self._cache[synset_name] = best
        return best


class WordNetIntervalEncoderFreq(WordNetIntervalEncoder):
    """
    Same encoder API but canonical + through intervals are computed via encode_with_probabilities().
    """
    def __init__(self, *args, freq_dict: SynsetFreqDict, **kwargs):
        super().__init__(*args, **kwargs)
        self._freq_dict = freq_dict

    def get_canonical_interval(self, synset):
        paths = synset.hypernym_paths()
        if not paths:
            base = self._base_interval(synset.pos())
            return base if base is not None else (0.0, 0.0)

        best_len = None
        best_iv = None
        for path in paths:
            iv = self._encode_path_with_freq(path, synset.pos(), self._freq_dict)
            if iv is None:
                continue
            L = len(path)
            if best_len is None or L < best_len:
                best_len = L
                best_iv = iv
        return best_iv if best_iv is not None else (0.0, 0.0)

    def _interval_for_path_through(self, synset, ancestor):
        paths = synset.hypernym_paths()
        intervals = self.encode_with_probabilities(synset, self._freq_dict)

        candidates = []
        for p, iv in zip(paths, intervals):
            if iv is None:
                continue
            if ancestor in p:
                if p[0] == synset:
                    depth = (len(p) - 1) - p.index(ancestor)
                else:
                    depth = p.index(ancestor)
                candidates.append((depth, iv))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


def load_word_counts_from_npz(npz_path: str) -> Optional[Dict[str, int]]:
    """
    If your .npz includes token counts (e.g., 'word_counts' or 'counts' arrays), load them.
    Returns None if not found.
    """
    npz = np.load(npz_path, allow_pickle=True)
    if "word_counts" in npz:
        wc = npz["word_counts"]
        # could be dict-like or aligned with vocab; handle both
        if isinstance(wc, np.ndarray) and wc.dtype == object:
            try:
                d = wc.item()
                if isinstance(d, dict):
                    return {str(k): int(v) for k, v in d.items()}
            except Exception:
                pass
        # if it's aligned with vocab
        if isinstance(wc, np.ndarray) and "vocab" in npz and len(wc) == len(npz["vocab"]):
            vocab = npz["vocab"].tolist()
            return {str(w): int(c) for w, c in zip(vocab, wc.tolist())}
    if "counts" in npz and "vocab" in npz and len(npz["counts"]) == len(npz["vocab"]):
        vocab = npz["vocab"].tolist()
        return {str(w): int(c) for w, c in zip(vocab, npz["counts"].tolist())}
    return None


# -------------------------
# End-to-end dataset build
# -------------------------

@dataclass
class DatasetBundle:
    X_freq_train: sp.csr_matrix
    X_freq_test: sp.csr_matrix
    X_prob333_train: sp.csr_matrix
    X_prob333_test: sp.csr_matrix
    X_int_train: np.ndarray
    X_int_test: np.ndarray
    X_int_freq_train: np.ndarray
    X_int_freq_test: np.ndarray
    y_train: np.ndarray
    y_test_train_direct: np.ndarray
    y_test_train_rowpath: np.ndarray
    y_test_train_colpath: np.ndarray
    y_test_train_union: np.ndarray


def build_all_datasets(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    neg_df: Optional[pd.DataFrame],
    cooc_npz_path: str,
    root_type: str = "all",
    prefer_pos: str = "n",
    n_bins: int = 333,
    vec_mode_base: str = "midpoint",
    vec_mode_rich: str = "multiscale",
    children_k: int = 25,
) -> DatasetBundle:
    """
    Produces datasets 1-4 for train/test.
    """
    # 1) Load cooc backend
    cooc = CoocBackend.load_npz(cooc_npz_path)

    # 2) Base encoder
    encoder = WordNetIntervalEncoder(root_type=_alias_root_type(root_type))

    # 3) Canonical interval + cooc triple features via build_all_features
    out = build_all_features(
        train_df=train_df,
        test_df=test_df,
        neg_df=neg_df,
        encoder=encoder,
        cooc=cooc,
        n_bins=n_bins,
        vec_mode=vec_mode_base,
        prefer_pos=prefer_pos,
        point_weight=1.0,
    )
    train_all = out["train_df"]
    y_train = out["y_train"]

    # --- Train-derived TEST labels (use TRAIN positives graph; no WordNet) ---
    train_pos_df = train_df.copy()
    if "label" in train_pos_df.columns:
        train_pos_df = train_pos_df[train_pos_df["label"] == 1]
    y_test_train = build_test_labels_from_train(test_df, train_pos_df, max_depth=None)

    # (1) raw cooc frequency embeddings
    X_freq_train = build_sparse_rawfreq_pair_embeddings(train_all, encoder=encoder, cooc=cooc, prefer_pos=prefer_pos)
    X_freq_test  = build_sparse_rawfreq_pair_embeddings(test_df,  encoder=encoder, cooc=cooc, prefer_pos=prefer_pos)

    # (2) interval support weighted by corpus probs
    X_prob333_train = build_interval_features_probweighted(
        train_all, encoder=encoder, cooc=cooc, n_bins=n_bins,
        vec_mode=vec_mode_rich, prefer_pos=prefer_pos,
        hypo_mode="path", hyper_mode="children", children_k=children_k
    )
    X_prob333_test = build_interval_features_probweighted(
        test_df, encoder=encoder, cooc=cooc, n_bins=n_bins,
        vec_mode=vec_mode_rich, prefer_pos=prefer_pos,
        hypo_mode="path", hyper_mode="children", children_k=children_k
    )

    # (3) richer interval-only features
    X_int_train = build_interval_pair_features(
        train_all, encoder,
        n_bins=n_bins, vec_mode=vec_mode_rich, prefer_pos=prefer_pos,
        hypo_mode="path", hyper_mode="children", children_k=children_k
    ).astype(np.float32)
    X_int_test = build_interval_pair_features(
        test_df, encoder,
        n_bins=n_bins, vec_mode=vec_mode_rich, prefer_pos=prefer_pos,
        hypo_mode="path", hyper_mode="children", children_k=children_k
    ).astype(np.float32)

    # (4) frequency-weighted intervals
    word_counts = load_word_counts_from_npz(cooc_npz_path)
    freq_dict = SynsetFreqDict(cooc, word_counts=word_counts)
    encoder_freq = WordNetIntervalEncoderFreq(root_type=_alias_root_type(root_type), freq_dict=freq_dict)

    X_int_freq_train = build_interval_pair_features(
        train_all, encoder_freq,
        n_bins=n_bins, vec_mode=vec_mode_rich, prefer_pos=prefer_pos,
        hypo_mode="path", hyper_mode="children", children_k=children_k
    ).astype(np.float32)
    X_int_freq_test = build_interval_pair_features(
        test_df, encoder_freq,
        n_bins=n_bins, vec_mode=vec_mode_rich, prefer_pos=prefer_pos,
        hypo_mode="path", hyper_mode="children", children_k=children_k
    ).astype(np.float32)

    return DatasetBundle(
        X_freq_train=X_freq_train,
        X_freq_test=X_freq_test,
        X_prob333_train=X_prob333_train,
        X_prob333_test=X_prob333_test,
        X_int_train=X_int_train,
        X_int_test=X_int_test,
        X_int_freq_train=X_int_freq_train,
        X_int_freq_test=X_int_freq_test,
        y_train=y_train,
        y_test_train_direct=y_test_train["y_test_train_direct"],
        y_test_train_rowpath=y_test_train["y_test_train_rowpath"],
        y_test_train_colpath=y_test_train["y_test_train_colpath"],
        y_test_train_union=y_test_train["y_test_train_union"],
    )


def save_bundle(bundle: DatasetBundle, out_dir: str) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    sp.save_npz(outp / "X_freq_train.npz", bundle.X_freq_train)
    sp.save_npz(outp / "X_freq_test.npz", bundle.X_freq_test)

    sp.save_npz(outp / "X_prob333_train.npz", bundle.X_prob333_train)
    sp.save_npz(outp / "X_prob333_test.npz", bundle.X_prob333_test)

    np.save(outp / "X_int_train.npy", bundle.X_int_train)
    np.save(outp / "X_int_test.npy", bundle.X_int_test)

    np.save(outp / "X_int_freq_train.npy", bundle.X_int_freq_train)
    np.save(outp / "X_int_freq_test.npy", bundle.X_int_freq_test)

    np.save(outp / "y_train.npy", bundle.y_train)

    np.save(outp / "y_test_train_direct.npy", bundle.y_test_train_direct)
    np.save(outp / "y_test_train_rowpath.npy", bundle.y_test_train_rowpath)
    np.save(outp / "y_test_train_colpath.npy", bundle.y_test_train_colpath)
    np.save(outp / "y_test_train_union.npy", bundle.y_test_train_union)


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--neg_csv", default=None)
    ap.add_argument("--cooc_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--root_type", default="all", help="all|noun|verb|adj|prep")
    ap.add_argument("--prefer_pos", default="n")
    ap.add_argument("--n_bins", type=int, default=333)
    ap.add_argument("--vec_mode_base", default="midpoint", help="midpoint|spread|multiscale")
    ap.add_argument("--vec_mode_rich", default="multiscale", help="midpoint|spread|multiscale")
    ap.add_argument("--children_k", type=int, default=25)
    args = ap.parse_args()

    train_df = _load_csv(args.train_csv)
    test_df = _load_csv(args.test_csv)
    neg_df = _load_csv(args.neg_csv) if args.neg_csv else None

    # labels if absent
    if "label" not in train_df.columns:
        train_df = train_df.copy()
        train_df["label"] = 1
    if neg_df is not None and "label" not in neg_df.columns:
        neg_df = neg_df.copy()
        neg_df["label"] = 0

    bundle = build_all_datasets(
        train_df=train_df,
        test_df=test_df,
        neg_df=neg_df,
        cooc_npz_path=args.cooc_npz,
        root_type=args.root_type,
        prefer_pos=args.prefer_pos,
        n_bins=args.n_bins,
        vec_mode_base=args.vec_mode_base,
        vec_mode_rich=args.vec_mode_rich,
        children_k=args.children_k,
    )
    save_bundle(bundle, args.out_dir)


if __name__ == "__main__":
    main()
