# train_closure_completion.py

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Set, Optional, Iterable, Tuple, List

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _norm(x: str) -> str:
    return str(x).strip()


def build_train_pos_adj(
    train_df: pd.DataFrame,
    *,
    hypo_col: str = "hypo",
    hyper_col: str = "hyper",
    label_col: Optional[str] = "label",
    positive_value: int = 1,
) -> Dict[str, Set[str]]:
    """
    Directed TRAIN adjacency: hypo -> {hyper,...} using ONLY positive edges.
    Matches the idea in your v2_testlabels script. :contentReference[oaicite:3]{index=3}
    """
    df = train_df
    if label_col is not None and label_col in df.columns:
        df = df[df[label_col].astype(int) == int(positive_value)]

    adj: Dict[str, Set[str]] = defaultdict(set)
    for r in df.itertuples(index=False):
        h = _norm(getattr(r, hypo_col))
        H = _norm(getattr(r, hyper_col))
        if h and H:
            adj[h].add(H)
    return adj


def ancestors_of(
    node: str,
    adj: Dict[str, Set[str]],
    *,
    max_depth: Optional[int] = None,
) -> Set[str]:
    """
    All reachable hypernyms from node via TRAIN positive edges.
    (Boolean reachability / transitive closure membership.)
    """
    node = _norm(node)
    seen: Set[str] = set()
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


@dataclass
class TrainClosure:
    adj: Dict[str, Set[str]]
    _cache: Dict[str, Set[str]]

    def reachable(self, hypo: str, hyper: str, *, max_depth: Optional[int] = None) -> int:
        """
        1 iff hyper is reachable from hypo in TRAIN graph (boolean path existence).
        """
        hypo = _norm(hypo); hyper = _norm(hyper)
        if hypo not in self._cache:
            self._cache[hypo] = ancestors_of(hypo, self.adj, max_depth=max_depth)
        return 1 if hyper in self._cache[hypo] else 0

    def label_pairs(
        self,
        pairs_df: pd.DataFrame,
        *,
        hypo_col: str = "hypo",
        hyper_col: str = "hyper",
        max_depth: Optional[int] = None,
    ) -> np.ndarray:
        y = np.zeros(len(pairs_df), dtype=np.int64)
        for i, (h, H) in enumerate(zip(pairs_df[hypo_col].astype(str), pairs_df[hyper_col].astype(str))):
            y[i] = self.reachable(h, H, max_depth=max_depth)
        return y

    def to_closure_csr(self, nodes: Optional[List[str]] = None, *, max_depth: Optional[int] = None) -> sp.csr_matrix:
        """
        Build a sparse adjacency matrix of the boolean closure (TRAIN only).

        WARNING: For very large graphs, materializing full closure can be big (dense-ish).
        Consider using reachable() / label_pairs() instead.
        """
        if nodes is None:
            nodes = sorted(set(self.adj.keys()) | {x for s in self.adj.values() for x in s})
        idx = {n: i for i, n in enumerate(nodes)}
        rows, cols = [], []

        for h in nodes:
            A = ancestors_of(h, self.adj, max_depth=max_depth)
            hi = idx[h]
            for anc in A:
                if anc in idx:
                    rows.append(hi)
                    cols.append(idx[anc])

        data = np.ones(len(rows), dtype=np.int8)
        n = len(nodes)
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def make_train_closure(
    train_df: pd.DataFrame,
    *,
    hypo_col: str = "hypo",
    hyper_col: str = "hyper",
    label_col: Optional[str] = "label",
) -> TrainClosure:
    adj = build_train_pos_adj(train_df, hypo_col=hypo_col, hyper_col=hyper_col, label_col=label_col)
    return TrainClosure(adj=adj, _cache={})
