# build_wn_codeword_tensor_ds.py
#
# Build a dataset of padded "arithmetic codeword" tensors for WordNet synsets.
# - Codeword is extracted from the canonical (shortest) hypernym path (root -> synset).
# - Each step encodes the branching decision among hyponyms:
#       (child_index_1based, n_children, p_child)
#   where p_child is either uniform (1/n_children) or frequency-weighted using a COOC NPZ's word_counts.
#
# This version FIXES the earlier issue: your COOC NPZ does NOT contain 'freq_dict' but DOES contain
# 'word_counts' (token->count). We load word_counts and project it to synset weights on the fly.
# (Matches how your cooc builder saves the NPZ.) :contentReference[oaicite:0]{index=0}
#
# Output:
#   torch.save({
#       "X": (N, T_max, 3) float32,
#       "lengths": (N,) int64,
#       "synsets": list[str],
#       "intervals": (N,2) float32,
#       "meta": ...
#   }, out_path)
#
# Example:
#   python build_wn_codeword_tensor_ds.py --out wn_codewords.pt --root_type noun --with_freq 0
#   python build_wn_codeword_tensor_ds.py --out wn_codewords_freq.pt --root_type noun --with_freq 1 --cooc_npz path/to/cooc.npz

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from nltk.corpus import wordnet as wn

# Your attached encoder (used for consistent root/POS interval gating)
from WN_arithmetic_interval_encoding8_5 import WordNetIntervalEncoder  # :contentReference[oaicite:1]{index=1}


# -----------------------------
# Helpers: canonical path + codeword extraction
# -----------------------------

def _sorted_children(parent_syn) -> List:
    """
    Deterministic child ordering. This must be stable across runs.
    """
    return sorted(parent_syn.hyponyms(), key=lambda s: s.name())


def pick_canonical_path_root_to_syn(syn) -> Optional[List]:
    """
    Pick the shortest hypernym path and normalize to root -> syn ordering.

    WordNet returns paths as lists; depending on the API call, sometimes the synset appears first.
    We normalize so traversal is parent->child along the path.
    """
    paths = syn.hypernym_paths()
    if not paths:
        return None
    best = min(paths, key=len)

    # Normalize direction:
    # If synset appears at start, reverse to get root->syn.
    if best and best[0] == syn:
        best = list(reversed(best))
    return best


def load_word_counts_from_cooc_npz(npz_path: str) -> Counter:
    """
    Load token->count mapping from a COOC NPZ produced by your cooc script.
    That script stores: word_counts=dict(self.word_counts). :contentReference[oaicite:2]{index=2}
    """
    npz = np.load(npz_path, allow_pickle=True)
    if "word_counts" not in npz:
        keys = list(npz.keys())
        raise ValueError(
            f"COOC NPZ missing 'word_counts'. Found keys={keys}. "
            "Expected a COOC NPZ produced by your builder."
        )
    d = npz["word_counts"].item()
    if not isinstance(d, dict):
        raise ValueError(f"'word_counts' exists but is not a dict (got {type(d)}).")
    return Counter({str(k): int(v) for k, v in d.items()})


def make_synset_weight_fn(word_counts: Counter, mode: str = "max") -> Callable[[str], float]:
    """
    Project token word_counts into synset weights.

    For synset s:
      lemma tokens = syn.name().split('.')[0].replace('_',' ').split()
      weight(s) = max(count(tok))   if mode='max'
                = sum(count(tok))   if mode='sum'

    Returns weight(synset_name)->float with caching.
    """
    mode = mode.lower().strip()
    if mode not in ("max", "sum"):
        raise ValueError("mode must be 'max' or 'sum'")

    cache: Dict[str, float] = {}

    def w(synset_name: str) -> float:
        if synset_name in cache:
            return cache[synset_name]
        try:
            syn = wn.synset(synset_name)
        except Exception:
            cache[synset_name] = 1.0
            return 1.0

        lemma = syn.name().split(".")[0].replace("_", " ")
        toks = [t for t in lemma.split() if t]
        if not toks:
            cache[synset_name] = 1.0
            return 1.0

        vals = [float(word_counts.get(t, 0)) for t in toks]
        out = max(vals) if mode == "max" else sum(vals)
        if out <= 0.0:
            out = 1.0
        cache[synset_name] = float(out)
        return float(out)

    return w


def path_to_codeword(
    path_root_to_syn: List,
    *,
    synset_weight_fn: Optional[Callable[[str], float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Convert a root->...->syn path into a variable-length codeword of steps:
      step t: (child_index_1based, n_children, p_child)

    Also returns the implied arithmetic interval (low, high) from those choices,
    starting from [0,1).
    """
    low, high = (0.0, 1.0)
    steps: List[Tuple[float, float, float]] = []

    for i in range(1, len(path_root_to_syn)):
        parent = path_root_to_syn[i - 1]
        child = path_root_to_syn[i]

        children = _sorted_children(parent)
        if not children or child not in children:
            # Be robust to occasional path/children mismatches
            continue

        n = len(children)
        idx0 = children.index(child)     # 0-based
        idx1 = idx0 + 1                 # 1-based (0 reserved for PAD)

        if synset_weight_fn is None:
            p_child = 1.0 / n
            cumulative_low = idx0 * p_child
        else:
            weights = np.array([float(synset_weight_fn(c.name())) for c in children], dtype=np.float64)
            total = float(weights.sum())
            if total <= 0.0:
                weights = np.ones(n, dtype=np.float64)
                total = float(n)
            probs = weights / total
            p_child = float(probs[idx0])
            cumulative_low = float(probs[:idx0].sum())

        width = high - low
        new_low = low + cumulative_low * width
        new_high = new_low + p_child * width
        low, high = new_low, new_high

        steps.append((float(idx1), float(n), float(p_child)))

    arr = np.asarray(steps, dtype=np.float32)  # (L, 3)
    return arr, (float(low), float(high))


# -----------------------------
# Dataset builder
# -----------------------------

@dataclass
class TensorDatasetBundle:
    X: torch.Tensor          # (N, T_max, 3)
    lengths: torch.Tensor    # (N,)
    synsets: List[str]       # N synset names
    intervals: torch.Tensor  # (N, 2)


def _collect_synsets(root_type: str) -> List:
    rt = root_type.lower().strip()
    if rt in ("noun", "n"):
        return list(wn.all_synsets(pos="n"))
    if rt in ("verb", "v"):
        return list(wn.all_synsets(pos="v"))
    if rt in ("adj", "a", "adjective"):
        return list(wn.all_synsets(pos="a"))
    if rt in ("adv", "r", "adverb"):
        return list(wn.all_synsets(pos="r"))
    if rt == "all":
        return list(wn.all_synsets())
    raise ValueError(f"Unsupported root_type={root_type}")


def build_codeword_tensor_dataset(
    *,
    root_type: str = "noun",
    with_freq: bool = False,
    synset_weight_fn: Optional[Callable[[str], float]] = None,
    use_pos_base_interval: bool = False,
    precision: int = 10,
) -> TensorDatasetBundle:
    """
    Build padded tensors for WordNet synsets.

    - If with_freq=True, synset_weight_fn must be provided (weights used for child probabilities).
    - If use_pos_base_interval=True, final interval is mapped into the POS base bucket from the encoder.
      For noun-only you usually get [0,1) anyway, but it’s here if you need it.
    """
    # Keep encoder gating consistent with your interval encoder
    enc = WordNetIntervalEncoder(precision=precision, root_type=root_type)  # :contentReference[oaicite:3]{index=3}

    syns = _collect_synsets(root_type)

    codewords: List[np.ndarray] = []
    intervals: List[Tuple[float, float]] = []
    syn_names: List[str] = []

    for syn in syns:
        base = enc._base_interval(syn.pos())  # respects root_type filter :contentReference[oaicite:4]{index=4}
        if base is None:
            continue

        path = pick_canonical_path_root_to_syn(syn)
        if path is None:
            cw = np.zeros((0, 3), dtype=np.float32)
            iv = base if use_pos_base_interval else (0.0, 1.0)
        else:
            cw, iv01 = path_to_codeword(path, synset_weight_fn=synset_weight_fn if with_freq else None)

            if use_pos_base_interval:
                b0, b1 = base
                low01, high01 = iv01
                iv = (b0 + low01 * (b1 - b0), b0 + high01 * (b1 - b0))
            else:
                iv = iv01

        codewords.append(cw)
        intervals.append(iv)
        syn_names.append(syn.name())

    lengths = np.array([cw.shape[0] for cw in codewords], dtype=np.int64)
    T_max = int(lengths.max()) if len(lengths) else 0
    D = 3

    X = np.zeros((len(codewords), T_max, D), dtype=np.float32)
    for i, cw in enumerate(codewords):
        if cw.shape[0]:
            X[i, : cw.shape[0], :] = cw

    return TensorDatasetBundle(
        X=torch.from_numpy(X),
        lengths=torch.from_numpy(lengths),
        synsets=syn_names,
        intervals=torch.tensor(intervals, dtype=torch.float32),
    )


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument(
        "--root_type",
        default="noun",
        choices=["all", "noun", "n", "verb", "v", "adj", "a", "adv", "r"],
        help="Which WordNet synsets to include."
    )
    ap.add_argument("--with_freq", type=int, default=0, help="1 to use frequency-weighted branching; 0 for uniform.")
    ap.add_argument(
        "--cooc_npz",
        default=None,
        help="COOC NPZ path (must contain 'word_counts' dict). Required if --with_freq 1."
    )
    ap.add_argument(
        "--synset_weight_mode",
        default="max",
        choices=["max", "sum"],
        help="How to map token word_counts -> synset weight (max or sum over lemma tokens)."
    )
    ap.add_argument("--use_pos_base_interval", type=int, default=0, help="1 to map interval into POS bucket; 0=[0,1).")
    ap.add_argument("--precision", type=int, default=10, help="Encoder precision parameter (kept for consistency).")

    args = ap.parse_args()

    synset_weight_fn = None
    if args.with_freq == 1:
        if args.cooc_npz is None:
            raise ValueError("--with_freq 1 requires --cooc_npz pointing to your COOC NPZ file.")
        word_counts = load_word_counts_from_cooc_npz(args.cooc_npz)  # :contentReference[oaicite:5]{index=5}
        synset_weight_fn = make_synset_weight_fn(word_counts, mode=args.synset_weight_mode)

    ds = build_codeword_tensor_dataset(
        root_type=args.root_type,
        with_freq=bool(args.with_freq),
        synset_weight_fn=synset_weight_fn,
        use_pos_base_interval=bool(args.use_pos_base_interval),
        precision=int(args.precision),
    )

    torch.save(
        {
            "X": ds.X,                    # (N, T_max, 3)
            "lengths": ds.lengths,        # (N,)
            "synsets": ds.synsets,        # list[str]
            "intervals": ds.intervals,    # (N, 2)
            "meta": {
                "root_type": args.root_type,
                "with_freq": bool(args.with_freq),
                "cooc_npz": args.cooc_npz,
                "synset_weight_mode": args.synset_weight_mode if args.with_freq else None,
                "use_pos_base_interval": bool(args.use_pos_base_interval),
                "precision": int(args.precision),
                "feature_cols": ["child_idx_1based", "n_children", "p_child"],
                "pad_value": 0.0,
                "child_idx_note": "1-based; 0 reserved for padding",
            },
        },
        args.out,
    )

    print(f"Saved: {args.out}")
    print(f"N={ds.X.shape[0]}  T_max={ds.X.shape[1]}  D={ds.X.shape[2]}")


if __name__ == "__main__":
    main()
