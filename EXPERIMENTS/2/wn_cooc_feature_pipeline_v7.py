
"""
wn_cooc_feature_pipeline_v4.py

Print-free pipeline for:
1) WordNet arithmetic-interval features (333-d + 333-d):
   - hyper part: canonical interval of hypernym synset
   - hypo part : interval of hyponym synset *through* the candidate hypernym (ancestor) if possible

2) Corpus co-occurrence features from an enwik8 sparse CSR .npz:
   - raw co-occurrence frequency
   - joint probability P(w1,w2)
   - conditional probability P(w2|w1) (row-normalized)

3) Diagnostics table comparing dataset paths vs WordNet paths:
   - dataset shortest path (via positive edges)
   - WordNet shortest hypernym path (BFS upward)
   - Wu–Palmer similarity + LCA
   - interval-through stats + overlaps + common ancestors

Key resolution behavior (for non-synset-id strings like "pike fish"):
- Try to resolve full lemma (spaces->underscore) first.
- If that fails, try BOTH head and tail tokens as candidates.
- When selecting (hypo_syn, hyper_syn), we explicitly avoid the trivial match hypo_syn == hyper_syn.
- Prefer pairs where hyper is in hypo's WordNet hypernym closure; otherwise pick highest WUP.

Dependencies:
- `encoder` must provide:
    - get_canonical_interval(synset) -> (low, high)
    - _interval_for_path_through(child_synset, ancestor_synset) -> (low, high) or None
    - wup_similarity(syn1, syn2, return_lca=False) -> float or (float, lca)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import deque, defaultdict

from nltk.corpus import wordnet as wn


# -------------------------
# Interval -> vector encoder
# -------------------------

def interval_width(iv: Tuple[float, float]) -> float:
    lo, hi = float(iv[0]), float(iv[1])
    return float(max(0.0, hi - lo))


def interval_to_vec(
    iv: Tuple[float, float],
    n_bins: int = 333,
    mode: str = "midpoint",
    normalize: bool = False,
    point_weight: float = 1.0,
    multiscale_splits: Tuple[int, ...] = (13, 26, 52, 104, 138),
    ms_mass: str = "binary",  # "binary" | "width"
) -> np.ndarray:
    """
    Convert interval [low, high) into a length-n_bins vector.

    mode:
      - "midpoint": quasi-one-hot at bin(midpoint).
          If width==0, still emits `point_weight` mass in the midpoint bin.
      - "spread": distribute mass by bin-overlap (requires width>0).
      - "multiscale": write 1 bin at multiple resolutions (coarse->fine) *within* 333 dims.
          This is designed for your setting where many codewords are extremely close, so
          a single 333-bin quantization collapses most nodes into the same bin.

          We split the vector into segments whose lengths sum to n_bins (default: 13+26+52+104+138=333).
          For each segment length s, we activate bin floor(mid*s) inside that segment.

    normalize:
      - if True, scales vector to sum to 1 (if nonzero)

    ms_mass:
      - "binary": each activated multiscale bin gets 1 (or 1/levels if normalize)
      - "width": each activated bin gets interval width (or point_weight if width==0)
    """
    lo, hi = float(iv[0]), float(iv[1])
    w = max(0.0, hi - lo)
    v = np.zeros(n_bins, dtype=np.float32)

    mode = str(mode).strip().lower()

    if mode == "midpoint":
        mid = 0.5 * (lo + hi)
        idx = int(np.clip(np.floor(mid * n_bins), 0, n_bins - 1))
        mass = (1.0 if normalize else (w if w > 0 else float(point_weight)))
        v[idx] = mass
        if normalize and v.sum() > 0:
            v /= v.sum()
        return v

    if mode == "spread":
        if w <= 0:
            return v
        edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
        lefts = np.maximum(edges[:-1], lo)
        rights = np.minimum(edges[1:], hi)
        overlaps = np.maximum(0.0, rights - lefts).astype(np.float32)
        if normalize and overlaps.sum() > 0:
            overlaps = overlaps / overlaps.sum()
        return overlaps

    if mode == "multiscale":
        splits = tuple(int(x) for x in multiscale_splits)
        if sum(splits) != n_bins:
            raise ValueError(f"multiscale_splits must sum to n_bins={n_bins}, got sum={sum(splits)}")
        mid = 0.5 * (lo + hi)
        levels = len(splits)
        if ms_mass not in {"binary", "width"}:
            raise ValueError("ms_mass must be 'binary' or 'width'")

        if ms_mass == "binary":
            mass = 1.0
        else:
            mass = (w if w > 0 else float(point_weight))

        off = 0
        for s in splits:
            idx = int(np.clip(np.floor(mid * s), 0, s - 1))
            v[off + idx] += mass
            off += s

        if normalize and v.sum() > 0:
            v /= v.sum()
        return v

    raise ValueError("mode must be 'midpoint', 'spread', or 'multiscale'.")



def normalized_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Symmetric normalized overlap in [0,1]."""
    a1, a2 = float(a[0]), float(a[1])
    b1, b2 = float(b[0]), float(b[1])
    lo = max(a1, b1)
    hi = min(a2, b2)
    if hi <= lo:
        return 0.0
    inter = hi - lo
    wa = max(0.0, a2 - a1)
    wb = max(0.0, b2 - b1)
    denom = wa + wb
    return float((2.0 * inter / denom) if denom > 0 else 0.0)


# -------------------------
# Co-occurrence matrix loader
# -------------------------

@dataclass
class CoocBackend:
    """Minimal interface around an enwik8 co-occurrence CSR matrix saved as .npz."""
    cooc: sp.csr_matrix
    vocab: List[str]
    word2idx: Dict[str, int]
    total: int

    @classmethod
    def load_npz(cls, npz_path: str) -> "CoocBackend":
        npz = np.load(npz_path, allow_pickle=True)
        cooc = sp.csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=tuple(npz["shape"]))
        if "vocab" not in npz:
            raise ValueError("NPZ has no vocab; cannot query by word strings.")
        vocab = npz["vocab"].tolist()
        word2idx = {w: i for i, w in enumerate(vocab)}
        total = int(npz["total_cooccurrences"]) if "total_cooccurrences" in npz else int(cooc.sum())
        return cls(cooc=cooc, vocab=vocab, word2idx=word2idx, total=total)

    def cooc_count(self, w1: str, w2: str) -> Optional[int]:
        i = self.word2idx.get(w1)
        j = self.word2idx.get(w2)
        if i is None or j is None:
            return None
        return int(self.cooc[i, j])

    def joint_prob(self, w1: str, w2: str) -> Optional[float]:
        c = self.cooc_count(w1, w2)
        if c is None:
            return None
        return float(c / self.total) if self.total > 0 else 0.0

    def cond_prob(self, w1: str, w2: str) -> Optional[float]:
        """P(w2|w1) via row-normalization."""
        i = self.word2idx.get(w1)
        j = self.word2idx.get(w2)
        if i is None or j is None:
            return None
        row_sum = float(self.cooc.getrow(i).sum())
        c = int(self.cooc[i, j])
        return float(c / row_sum) if row_sum > 0 else 0.0


# -------------------------
# WordNet string helpers
# -------------------------

def is_synset_id(s: str) -> bool:
    parts = str(s).split(".")
    return len(parts) >= 3 and parts[-1].isdigit()

def norm_lemma(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("-", "_")
    return s

def synset_candidates(spec: str, prefer_pos: Optional[str] = "n") -> List:
    """
    Candidates for a spec.
    - synset id -> [synset] (or [])
    - otherwise try full normalized lemma first.
    - if that fails and it's multiword: try BOTH head and tail tokens; union.
    """
    s = str(spec).strip()
    if is_synset_id(s):
        try:
            return [wn.synset(s)]
        except Exception:
            return []

    lemma = norm_lemma(s)
    syns = wn.synsets(lemma)

    if not syns and "_" in lemma:
        toks = [t for t in lemma.split("_") if t]
        cand = []
        if toks:
            cand.extend(wn.synsets(toks[0]))          # head
            if len(toks) > 1:
                cand.extend(wn.synsets(toks[-1]))     # tail
        # dedupe preserving order
        seen = set()
        syns = [x for x in cand if (x not in seen and not seen.add(x))]

    if prefer_pos is not None:
        syns = [x for x in syns if x.pos() == prefer_pos]
    return syns

def in_hypernym_closure(hypo_syn, hyper_syn) -> bool:
    for s in hypo_syn.closure(lambda s: s.hypernyms() + s.instance_hypernyms()):
        if s == hyper_syn:
            return True
    return False

def _wn_shortest_len(hypo_syn, hyper_syn, max_depth: int = 50) -> Optional[int]:
    """Shortest upward hypernym distance (edges) from hypo to hyper, or None."""
    if hypo_syn == hyper_syn:
        return 0
    q = deque([(hypo_syn, 0)])
    seen = {hypo_syn}
    while q:
        node, d = q.popleft()
        if d >= max_depth:
            continue
        for p in node.hypernyms() + node.instance_hypernyms():
            if p in seen:
                continue
            if p == hyper_syn:
                return d + 1
            seen.add(p)
            q.append((p, d + 1))
    return None


def _lex_bias(hypo_spec: str, hyper_spec: str, hypo_syn, origin_hint: Optional[str] = None) -> float:
    """
    Small bias to avoid degenerate multiword matches where the resolved hyponym collapses
    to the hypernym's lexical head (e.g., 'pike fish' -> 'fish').

    Returns a small penalty (>=0) applied when *ranking* candidate pairs.
    """
    hs = norm_lemma(hypo_spec)
    Hs = norm_lemma(hyper_spec)
    hypo_toks = [t for t in hs.split("_") if t]
    hyper_toks = {t for t in Hs.split("_") if t}

    lemma = hypo_syn.name().split(".")[0].lower()
    lemma_toks = set(lemma.split("_"))

    penalty = 0.0

    # Penalize if the chosen hypo lemma is lexically identical to any hyper token.
    if lemma_toks & hyper_toks:
        penalty += 0.05

    # Extra penalty if the phrase contains the hyper token (common 'X fish' pattern)
    # and the chosen hypo lemma matches that hyper token.
    if len(hypo_toks) >= 2 and (set(hypo_toks) & hyper_toks) and (lemma_toks & hyper_toks):
        penalty += 0.05

    # Tiny preference for picking the head token sense when the phrase is multiword.
    # (head token = first token)
    if len(hypo_toks) >= 2:
        head = hypo_toks[0]
        if head in lemma_toks:
            penalty -= 0.01

    # Origin hint from candidate generation (if you later decide to track it):
    if origin_hint == "tail":
        penalty += 0.01

    return float(max(0.0, penalty))


def resolve_pair_synsets(
    hypo_spec: str,
    hyper_spec: str,
    encoder,
    prefer_pos: str = "n",
    max_depth: int = 50,
) -> Tuple[Optional[object], Optional[object]]:
    """
    Resolve (hypo, hyper) to a synset pair.

    Selection:
      - Generate candidates (prefer nouns by default; fallback to any POS if empty).
      - Choose any pair where hyper is in hypo's hypernym closure, preferring:
            (shortest wn path length) + small lexical penalty
      - Else choose pair with highest:
            (Wu–Palmer similarity) - small lexical penalty
      - Always avoid trivial match hypo_syn == hyper_syn.
    """
    hs = synset_candidates(hypo_spec, prefer_pos=prefer_pos)
    Hs = synset_candidates(hyper_spec, prefer_pos=prefer_pos)

    if not hs:
        hs = synset_candidates(hypo_spec, prefer_pos=None)
    if not Hs:
        Hs = synset_candidates(hyper_spec, prefer_pos=None)

    if not hs or not Hs:
        return None, None

    # closure-satisfying candidates (best = shortest wn path + bias)
    best_pair = None
    best_score = None  # lower is better
    for h in hs:
        for H in Hs:
            if h == H:
                continue
            if in_hypernym_closure(h, H):
                plen = _wn_shortest_len(h, H, max_depth=max_depth)
                plen = plen if plen is not None else (10**9)
                score = float(plen) + _lex_bias(hypo_spec, hyper_spec, h)
                if best_pair is None or score < best_score:
                    best_pair = (h, H)
                    best_score = score

    if best_pair is not None:
        return best_pair

    # else best WUP - bias (avoid hypo==hyper)
    best_pair = None
    best_score = None  # higher is better
    for h in hs:
        for H in Hs:
            if h == H:
                continue
            w = float(encoder.wup_similarity(h, H))
            score = w - _lex_bias(hypo_spec, hyper_spec, h)
            if best_pair is None or score > best_score:
                best_pair = (h, H)
                best_score = score

    if best_pair is not None:
        return best_pair

    # worst case fallback
    return hs[0], Hs[0]


def tokens_from_spec_or_synset(spec: str, syn) -> List[str]:
    """
    For corpus cooc querying:
      - if spec is not a synset id, use its whitespace tokens directly
      - else derive from synset lemma tokens
    """
    if not is_synset_id(spec):
        return [t for t in str(spec).strip().lower().split() if t]
    lemma = syn.name().split(".")[0]
    return [t for t in lemma.replace("_", " ").split() if t]


# -------------------------
# Dataset graph + WordNet path compare
# -------------------------

def build_graph_from_edges(edges: Iterable[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Directed graph: hypo -> hyper."""
    g = defaultdict(list)
    for h, H in edges:
        g[h].append(H)
    return g

def shortest_path_graph(g: Dict[str, List[str]], start: str, goal: str, max_depth: int = 50) -> Optional[List[str]]:
    """BFS shortest path in dataset graph."""
    if start == goal:
        return [start]
    q = deque([(start, [start])])
    seen = {start}
    while q:
        node, path = q.popleft()
        if len(path) > max_depth:
            continue
        for nxt in g.get(node, []):
            if nxt in seen:
                continue
            npath = path + [nxt]
            if nxt == goal:
                return npath
            seen.add(nxt)
            q.append((nxt, npath))
    return None

def shortest_path_wordnet(hypo_syn, hyper_syn, max_depth: int = 50) -> Optional[List[str]]:
    """BFS upward in WordNet hypernym graph. Returns list of synset names."""
    if hypo_syn == hyper_syn:
        return [hypo_syn.name()]
    q = deque([(hypo_syn, [hypo_syn])])
    seen = {hypo_syn}
    while q:
        node, path = q.popleft()
        if len(path) > max_depth:
            continue
        for p in node.hypernyms() + node.instance_hypernyms():
            if p in seen:
                continue
            npath = path + [p]
            if p == hyper_syn:
                return [s.name() for s in npath]
            seen.add(p)
            q.append((p, npath))
    return None

def ancestor_set_wordnet(syn) -> set:
    return set(s.name() for s in syn.closure(lambda s: s.hypernyms() + s.instance_hypernyms()))


# -------------------------
# Feature builders
# -------------------------


def _sum_interval_vecs(
    intervals: List[Tuple[float, float]],
    n_bins: int,
    vec_mode: str,
    normalize: bool,
    point_weight: float,
) -> np.ndarray:
    v = np.zeros(n_bins, dtype=np.float32)
    for iv in intervals:
        if iv is None:
            continue
        v += interval_to_vec(iv, n_bins=n_bins, mode=vec_mode, normalize=False, point_weight=point_weight)
    if normalize and v.sum() > 0:
        v = v / v.sum()
    return v


def build_interval_pair_features(
    df_pairs: pd.DataFrame,
    encoder,
    n_bins: int = 333,
    vec_mode: str = "midpoint",
    normalize: bool = False,
    prefer_pos: str = "n",
    point_weight: float = 1.0,
    hypo_col: str = "hypo",
    hyper_col: str = "hyper",
    hypo_mode: str = "through",   # "through" (default) or "path"
    hyper_mode: str = "canonical",# "canonical" (default) or "children"
    children_k: int = 25,
) -> np.ndarray:
    """
    X_interval shape (m, 2*n_bins).

    hyper half ([:n_bins]):
      - "canonical": vectorize hyper canonical interval (1 bin in midpoint mode)
      - "children" : sum vectors for hyper + up to K direct hyponyms (more bins)

    hypo half ([n_bins:]):
      - "through": vectorize ONE interval of hypo through hyper (1 bin if exists)
      - "path"   : if a WordNet path exists hypo -> ... -> hyper, sum canonical vectors
                  of all nodes on that path (more bins; approximates the path itself)

    Note: In midpoint mode, each interval contributes exactly one bin; to get many bins,
    use hypo_mode="path" and/or hyper_mode="children".
    """
    m = len(df_pairs)
    X = np.zeros((m, 2 * n_bins), dtype=np.float32)

    hypo_mode = str(hypo_mode).strip().lower()
    hyper_mode = str(hyper_mode).strip().lower()
    if hypo_mode not in {"through", "path"}:
        raise ValueError("hypo_mode must be 'through' or 'path'")
    if hyper_mode not in {"canonical", "children"}:
        raise ValueError("hyper_mode must be 'canonical' or 'children'")

    for r, row in enumerate(df_pairs.itertuples(index=False)):
        hypo_spec = getattr(row, hypo_col)
        hyper_spec = getattr(row, hyper_col)

        hypo_syn, hyper_syn = resolve_pair_synsets(hypo_spec, hyper_spec, encoder, prefer_pos=prefer_pos)
        if hypo_syn is None or hyper_syn is None:
            continue

        # ---- hyper half ----
        if hyper_mode == "canonical":
            hyper_iv = encoder.get_canonical_interval(hyper_syn)
            X[r, :n_bins] = interval_to_vec(
                hyper_iv, n_bins=n_bins, mode=vec_mode, normalize=normalize, point_weight=point_weight
            )
        else:  # children
            ivs = [encoder.get_canonical_interval(hyper_syn)]
            # take up to K direct hyponyms (stable order not guaranteed)
            for child in hyper_syn.hyponyms()[: max(0, int(children_k))]:
                ivs.append(encoder.get_canonical_interval(child))
            X[r, :n_bins] = _sum_interval_vecs(
                ivs, n_bins=n_bins, vec_mode=vec_mode, normalize=normalize, point_weight=point_weight
            )

        # ---- hypo half ----
        if hypo_mode == "through":
            iv_through = encoder._interval_for_path_through(hypo_syn, hyper_syn)
            if iv_through is not None:
                X[r, n_bins:2 * n_bins] = interval_to_vec(
                    iv_through, n_bins=n_bins, mode=vec_mode, normalize=normalize, point_weight=point_weight
                )
        else:  # path
            wn_path = shortest_path_wordnet(hypo_syn, hyper_syn, max_depth=50)
            if wn_path:
                ivs = []
                for name in wn_path:
                    try:
                        s = wn.synset(name)
                        ivs.append(encoder.get_canonical_interval(s))
                    except Exception:
                        pass
                X[r, n_bins:2 * n_bins] = _sum_interval_vecs(
                    ivs, n_bins=n_bins, vec_mode=vec_mode, normalize=normalize, point_weight=point_weight
                )

    return X



def build_cooc_pair_features(
    df_pairs: pd.DataFrame,
    cooc: CoocBackend,
    encoder,
    token_strategy: str = "max",  # "max" or "first"
    prefer_pos: str = "n",
    hypo_col: str = "hypo",
    hyper_col: str = "hyper",
) -> np.ndarray:
    """
    X_cooc shape (m, 3): [freq, joint_prob, cond_prob]
    Uses raw tokens directly if inputs are not synset IDs; otherwise uses synset lemma tokens.
    """
    m = len(df_pairs)
    X = np.zeros((m, 3), dtype=np.float32)

    for r, row in enumerate(df_pairs.itertuples(index=False)):
        hypo_spec = getattr(row, hypo_col)
        hyper_spec = getattr(row, hyper_col)

        hypo_syn, hyper_syn = resolve_pair_synsets(hypo_spec, hyper_spec, encoder, prefer_pos=prefer_pos)

        # Build candidate token lists
        if hypo_syn is None or hyper_syn is None:
            htoks = [t for t in str(hypo_spec).strip().lower().split() if t]
            Htoks = [t for t in str(hyper_spec).strip().lower().split() if t]
        else:
            htoks = tokens_from_spec_or_synset(hypo_spec, hypo_syn)
            Htoks = tokens_from_spec_or_synset(hyper_spec, hyper_syn)

        if not htoks or not Htoks:
            continue

        if token_strategy == "first":
            w1, w2 = htoks[0], Htoks[0]
            freq = cooc.cooc_count(w1, w2) or 0
            jp = cooc.joint_prob(w1, w2) or 0.0
            cp = cooc.cond_prob(w1, w2) or 0.0
        elif token_strategy == "max":
            best_f, best_j, best_c = 0, 0.0, 0.0
            for w1 in htoks:
                for w2 in Htoks:
                    f = cooc.cooc_count(w1, w2)
                    if f is None:
                        continue
                    j = cooc.joint_prob(w1, w2) or 0.0
                    c = cooc.cond_prob(w1, w2) or 0.0
                    if f > best_f:
                        best_f, best_j, best_c = f, j, c
            freq, jp, cp = best_f, best_j, best_c
        else:
            raise ValueError("token_strategy must be 'first' or 'max'.")

        X[r, 0] = float(freq)
        X[r, 1] = float(jp)
        X[r, 2] = float(cp)

    return X


def diagnostics_table_paths(
    df_pairs: pd.DataFrame,
    encoder,
    pos_edges: Iterable[Tuple[str, str]],
    prefer_pos: str = "n",
    max_depth: int = 50,
    hypo_col: str = "hypo",
    hyper_col: str = "hyper",
) -> pd.DataFrame:
    """
    Diagnostics per pair:
      - dataset shortest path (via pos edges)
      - WordNet shortest hypernym path
      - endpoint WUP + LCA
      - common ancestors count + sample
      - interval-through stats + overlaps
      - parent-interval overlap diagnostic (DS parent vs WN parent) if available
    """
    g = build_graph_from_edges(pos_edges)

    rows = []
    for row in df_pairs.itertuples(index=False):
        hypo_spec = getattr(row, hypo_col)
        hyper_spec = getattr(row, hyper_col)

        hypo_syn, hyper_syn = resolve_pair_synsets(hypo_spec, hyper_spec, encoder, prefer_pos=prefer_pos)
        if hypo_syn is None or hyper_syn is None:
            rows.append({
                "hypo": str(hypo_spec), "hyper": str(hyper_spec),
                "ds_path": None, "wn_path": None,
                "ds_len": None, "wn_len": None,
                "wup": None, "lca": None,
                "common_anc_count": None, "common_anc_sample": None,
                "interval_through": None, "interval_through_width": None,
                "overlap_through_vs_hypercanon": None,
                "ds_parent": None, "wn_parent": None, "parent_interval_overlap": None,
            })
            continue

        ds_path = shortest_path_graph(g, hypo_syn.name(), hyper_syn.name(), max_depth=max_depth)
        wn_path = shortest_path_wordnet(hypo_syn, hyper_syn, max_depth=max_depth)

        wup, lca = encoder.wup_similarity(hypo_syn, hyper_syn, return_lca=True)

        common_anc = ancestor_set_wordnet(hypo_syn) & ancestor_set_wordnet(hyper_syn)

        iv_through = encoder._interval_for_path_through(hypo_syn, hyper_syn)
        iv_through_w = interval_width(iv_through) if iv_through is not None else 0.0
        hyper_canon = encoder.get_canonical_interval(hyper_syn)
        overlap_through_hyper = normalized_overlap(iv_through, hyper_canon) if iv_through is not None else 0.0

        ds_parent = ds_path[1] if ds_path and len(ds_path) > 1 else None
        wn_parent = wn_path[1] if wn_path and len(wn_path) > 1 else None

        parent_iv_overlap = 0.0
        if ds_parent and wn_parent:
            try:
                ds_parent_syn = wn.synset(ds_parent)
                wn_parent_syn = wn.synset(wn_parent)
                iv_ds = encoder._interval_for_path_through(hypo_syn, ds_parent_syn)
                iv_wn = encoder._interval_for_path_through(hypo_syn, wn_parent_syn)
                if iv_ds is not None and iv_wn is not None:
                    parent_iv_overlap = normalized_overlap(iv_ds, iv_wn)
            except Exception:
                parent_iv_overlap = 0.0

        rows.append({
            "hypo": hypo_syn.name(),
            "hyper": hyper_syn.name(),
            "ds_path": ds_path,
            "wn_path": wn_path,
            "ds_len": len(ds_path) if ds_path else None,
            "wn_len": len(wn_path) if wn_path else None,
            "wup": float(wup),
            "lca": lca.name() if lca else None,
            "common_anc_count": len(common_anc),
            "common_anc_sample": sorted(list(common_anc))[:10],
            "interval_through": iv_through,
            "interval_through_width": float(iv_through_w),
            "overlap_through_vs_hypercanon": float(overlap_through_hyper),
            "ds_parent": ds_parent,
            "wn_parent": wn_parent,
            "parent_interval_overlap": float(parent_iv_overlap),
        })

    return pd.DataFrame(rows)


# -------------------------
# End-to-end convenience
# -------------------------

def build_all_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    neg_df: Optional[pd.DataFrame],
    encoder,
    cooc: Optional[CoocBackend],
    n_bins: int = 333,
    vec_mode: str = "midpoint",
    normalize: bool = False,
    prefer_pos: str = "n",
    point_weight: float = 1.0,
) -> Dict[str, object]:
    """
    Builds:
      - interval features for train/test
      - cooc features for train/test (if cooc provided)
      - diagnostics table for test (dataset path vs WordNet)

    Assumes train_df has positives, neg_df has negatives with label=0,
    and test_df has hypo/hyper columns.
    """
    # labels
    train_pos = train_df.copy()
    if "label" not in train_pos.columns:
        train_pos["label"] = 1

    if neg_df is not None:
        train_neg = neg_df.copy()
        if "label" not in train_neg.columns:
            train_neg["label"] = 0
        train_all = pd.concat([train_pos, train_neg], ignore_index=True)
    else:
        train_all = train_pos

    # interval features
    Xint_train = build_interval_pair_features(
        train_all, encoder, n_bins=n_bins, vec_mode=vec_mode, normalize=normalize,
        prefer_pos=prefer_pos, point_weight=point_weight
    )
    Xint_test = build_interval_pair_features(
        test_df, encoder, n_bins=n_bins, vec_mode=vec_mode, normalize=normalize,
        prefer_pos=prefer_pos, point_weight=point_weight
    )
    y_train = train_all["label"].to_numpy(dtype=np.int64)

    # cooc features
    Xcooc_train = Xcooc_test = None
    if cooc is not None:
        Xcooc_train = build_cooc_pair_features(train_all, cooc, encoder, token_strategy="max", prefer_pos=prefer_pos)
        Xcooc_test = build_cooc_pair_features(test_df, cooc, encoder, token_strategy="max", prefer_pos=prefer_pos)

    # diagnostics graph from positive edges only (use raw strings; diagnostics resolves internally)
    pos_edges = list(zip(train_pos["hypo"].tolist(), train_pos["hyper"].tolist()))
    diag = diagnostics_table_paths(test_df, encoder, pos_edges=pos_edges, prefer_pos=prefer_pos)

    return {
        "train_df": train_all,
        "test_df": test_df,
        "X_interval_train": Xint_train,
        "X_interval_test": Xint_test,
        "y_train": y_train,
        "X_cooc_train": Xcooc_train,
        "X_cooc_test": Xcooc_test,
        "diagnostics": diag,
    }
