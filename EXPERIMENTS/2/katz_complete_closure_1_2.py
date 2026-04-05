# katz_complete_closure.py

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Set, Optional, Iterable, Tuple, List

import numpy as np
import pandas as pd
import scipy.sparse as sp

def _norm(x: str) -> str:
    return str(x).strip()


def build_csr_adjacency_from_edges(
    train_df: pd.DataFrame,
    *,
    nodes: Optional[List[str]] = None,
    hypo_col: str = "hypo",
    hyper_col: str = "hyper",
    label_col: Optional[str] = "label",
    positive_value: int = 1,
) -> Tuple[sp.csr_matrix, List[str], Dict[str, int]]:
    """
    CSR adjacency M where M[i,j]=1 iff edge node_i -> node_j exists in TRAIN positives.
    (This is the Katz base adjacency.)
    """
    df = train_df
    if label_col is not None and label_col in df.columns:
        df = df[df[label_col].astype(int) == int(positive_value)]

    if nodes is None:
        nodes = sorted(set(df[hypo_col].astype(str)) | set(df[hyper_col].astype(str)))
    idx = {n: i for i, n in enumerate(nodes)}

    r, c = [], []
    for h, H in zip(df[hypo_col].astype(str), df[hyper_col].astype(str)):
        h = _norm(h); H = _norm(H)
        if h in idx and H in idx:
            r.append(idx[h]); c.append(idx[H])

    data = np.ones(len(r), dtype=np.float32)
    n = len(nodes)
    M = sp.csr_matrix((data, (r, c)), shape=(n, n))
    M.eliminate_zeros()
    return M, nodes, idx


def katz_truncated_scores(M: sp.csr_matrix, *, alpha: float = 0.01, k_max: int = 5) -> sp.csr_matrix:
    """
    Truncated Katz: S = sum_{k=1..k_max} alpha^k * M^k
    Uses sparse multiplications.
    """
    assert sp.isspmatrix_csr(M)
    n = M.shape[0]
    S = sp.csr_matrix((n, n), dtype=np.float32)

    Mk = M.copy().astype(np.float32)
    a = float(alpha)

    for k in range(1, k_max + 1):
        S = S + (a ** k) * Mk
        Mk = Mk @ M  # next power
        Mk.eliminate_zeros()

    S.eliminate_zeros()
    return S


def katz_complete_graph(
    M: sp.csr_matrix,
    *,
    alpha: float = 0.01,
    k_max: int = 5,
    top_k_per_row: int = 10,
    score_threshold: Optional[float] = None,
    n_rounds: int = 1,
    forbid_self_loops: bool = True,
) -> sp.csr_matrix:
    """
    Iterative Katz-based completion (TRAIN-only):
      - compute truncated Katz scores on current M
      - add edges based on top-k or threshold
      - repeat n_rounds times

    Returns completed adjacency (binary 0/1 CSR).
    """
    assert sp.isspmatrix_csr(M)
    n = M.shape[0]
    M_bin = (M > 0).astype(np.int8).tocsr()

    for _ in range(int(n_rounds)):
        S = katz_truncated_scores(M_bin.astype(np.float32), alpha=alpha, k_max=k_max).tocsr()

        # Remove existing edges from candidates
        # (Keep only NEW edges suggested by Katz)
        S = S.tocsr(copy=True)
        S[M_bin.nonzero()] = 0
        
        if forbid_self_loops:
            S.setdiag(0)
        S.eliminate_zeros()

        new_r, new_c = [], []

        # For each row: choose top_k_per_row by score OR all above threshold
        for i in range(n):
            start, end = S.indptr[i], S.indptr[i + 1]
            if start == end:
                continue
            cols = S.indices[start:end]
            vals = S.data[start:end]

            if score_threshold is not None:
                mask = vals >= float(score_threshold)
                cols = cols[mask]
                vals = vals[mask]

            if cols.size == 0:
                continue

            if top_k_per_row is not None and top_k_per_row > 0 and cols.size > top_k_per_row:
                # partial top-k
                top_idx = np.argpartition(-vals, top_k_per_row - 1)[:top_k_per_row]
                cols = cols[top_idx]

            new_r.extend([i] * len(cols))
            new_c.extend(cols.tolist())

        if not new_r:
            break

        data = np.ones(len(new_r), dtype=np.int8)
        M_add = sp.csr_matrix((data, (new_r, new_c)), shape=(n, n), dtype=np.int8)

        # Update adjacency
        M_bin = (M_bin + M_add).astype(np.int8)
        M_bin.eliminate_zeros()

    return M_bin
