#!/usr/bin/env python3
# ============================================================
# train_wordnet_katz_directed_d25.py
#
# Directed Katz/Resolvent-style WordNet embeddings from CSV pairs
#   - train_hypernyms.csv defines directed edges (hypo -> hyper)
#   - non_hypernyms.csv used only to include vocab for evaluation coverage
#
# Pipeline (Saedi et al.-style):
# 1) Build directed adjacency M
# 2) Compute S = I + αM + α^2 M^2 + ... (truncated series, sparse-safe)
# 3) PMI+  (PPMI or shifted PPMI)
# 4) L2 row normalization
# 5) Truncated SVD to d dims (PCA-like but sparse-friendly)
#
# Outputs:
#   - embeddings_d25.npy
#   - vocab.json
#   - embeddings_d25.txt (word2vec text format)
# ============================================================

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


# -----------------------
# Helpers
# -----------------------
def build_vocab_from_pairs(dfs, cols=("hypo", "hyper")):
    vocab = set()
    for df in dfs:
        for c in cols:
            vocab.update(df[c].astype(str).tolist())
    return sorted(vocab)


def build_directed_adjacency(df, vocab_index, src_col="hypo", dst_col="hyper", binary=True):
    src = df[src_col].astype(str).map(vocab_index).to_numpy()
    dst = df[dst_col].astype(str).map(vocab_index).to_numpy()
    n = len(vocab_index)

    data = np.ones(len(df), dtype=np.float32)
    M = sparse.coo_matrix((data, (src, dst)), shape=(n, n), dtype=np.float32).tocsr()
    M.sum_duplicates()
    if binary:
        M.data[:] = 1.0
    M.eliminate_zeros()
    return M


def prune_csr_topk(A: sparse.csr_matrix, topk=400, min_val=0.0):
    """
    Keep only top-k entries per row by value.
    Prevents densification during M^k expansions.
    """
    A = A.tocsr()
    n_rows = A.shape[0]
    indptr, indices, data = A.indptr, A.indices, A.data

    new_indptr = np.zeros(n_rows + 1, dtype=np.int32)
    new_indices, new_data = [], []
    nnz = 0

    for i in range(n_rows):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            new_indptr[i + 1] = nnz
            continue

        row_idx = indices[start:end]
        row_dat = data[start:end]

        if min_val > 0:
            mask = row_dat >= min_val
            row_idx = row_idx[mask]
            row_dat = row_dat[mask]

        if row_dat.size > topk:
            keep = np.argpartition(row_dat, -topk)[-topk:]
            row_idx = row_idx[keep]
            row_dat = row_dat[keep]

        order = np.argsort(row_idx)
        row_idx = row_idx[order]
        row_dat = row_dat[order]

        new_indices.append(row_idx.astype(np.int32))
        new_data.append(row_dat.astype(np.float32))
        nnz += row_dat.size
        new_indptr[i + 1] = nnz

    if nnz == 0:
        return sparse.csr_matrix(A.shape, dtype=np.float32)

    new_indices = np.concatenate(new_indices)
    new_data = np.concatenate(new_data)
    return sparse.csr_matrix((new_data, new_indices, new_indptr), shape=A.shape, dtype=np.float32)


def katz_series(M, alpha=0.8, K=10, tol=1e-5, prune_topk=400):
    """
    Directed Katz-like cumulative matrix:
      S = I + sum_{k=1..K} alpha^k M^k
    """
    n = M.shape[0]
    I = sparse.identity(n, format="csr", dtype=np.float32)

    S = I.copy()
    P = I.copy()
    a_pow = 1.0

    for _k in range(1, K + 1):
        P = P.dot(M)  # P = M^k
        if P.nnz == 0:
            break

        if prune_topk is not None:
            P = prune_csr_topk(P, topk=prune_topk)

        a_pow *= alpha
        add = P.multiply(a_pow)
        S = S + add

        # stop when marginal contribution becomes tiny
        mass_S = float(S.sum())
        mass_add = float(add.sum())
        if mass_S > 0 and (mass_add / mass_S) < tol:
            break

    S = S.tocsr()
    S.sum_duplicates()
    S.eliminate_zeros()
    return S


def ppmi_from_sparse(S: sparse.csr_matrix, eps=1e-12):
    """
    Standard PPMI:
      PPMI(i,j) = max( log( (S_ij * total)/(row_i*col_j) ), 0 )
    computed on nonzeros only.
    """
    S = S.tocsr()
    S.sum_duplicates()
    S.eliminate_zeros()

    row_sum = np.array(S.sum(axis=1)).ravel().astype(np.float64)
    col_sum = np.array(S.sum(axis=0)).ravel().astype(np.float64)
    total = float(row_sum.sum())
    if total <= 0:
        return sparse.csr_matrix(S.shape, dtype=np.float32)

    indptr, indices = S.indptr, S.indices
    data = S.data.astype(np.float64, copy=False)
    new_data = np.empty_like(data, dtype=np.float64)

    for i in range(S.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue
        rs = row_sum[i]
        if rs <= 0:
            new_data[start:end] = 0.0
            continue

        cols = indices[start:end]
        denom = rs * col_sum[cols]
        numer = data[start:end] * total

        pmi = np.log((numer + eps) / (denom + eps))
        pmi[pmi < 0] = 0.0
        new_data[start:end] = pmi

    X = sparse.csr_matrix(
        (new_data.astype(np.float32), indices.copy(), indptr.copy()),
        shape=S.shape, dtype=np.float32
    )
    X.eliminate_zeros()
    return X


def shifted_ppmi_from_sparse(S: sparse.csr_matrix, shift_k=5.0, eps=1e-12):
    """
    Shifted PPMI ("PMI+" variant often used to reduce hub/sense bias):
      max( PMI(i,j) - log(shift_k), 0 )
    """
    X = ppmi_from_sparse(S, eps=eps)
    if shift_k and shift_k > 0:
        X.data = np.maximum(X.data - math.log(shift_k), 0.0).astype(np.float32)
        X.eliminate_zeros()
    return X


def l2_row_normalize(X: sparse.csr_matrix, eps=1e-12):
    """L2 normalize each row: x_i <- x_i / ||x_i||_2"""
    X = X.tocsr()
    row_norm = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel().astype(np.float64)
    row_norm[row_norm < eps] = 1.0
    Dinv = sparse.diags((1.0 / row_norm).astype(np.float32))
    return Dinv.dot(X)


def train_embeddings(
    train_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    alpha=0.8,
    K=10,
    prune_topk=400,
    tol=1e-5,
    use_shifted_ppmi=True,
    shift_k=5.0,
    d=25,
    random_state=0,
):
    # include vocab from both to ensure evaluation words have vectors
    vocab = build_vocab_from_pairs([train_df, neg_df], cols=("hypo", "hyper"))
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # 1) directed adjacency from TRAIN edges only
    M = build_directed_adjacency(train_df, vocab_index, src_col="hypo", dst_col="hyper", binary=True)

    # 2) Katz/resolvent-style cumulative matrix
    S = katz_series(M, alpha=alpha, K=K, tol=tol, prune_topk=prune_topk)

    # 3) PMI+ (PPMI or shifted PPMI)
    if use_shifted_ppmi:
        X = shifted_ppmi_from_sparse(S, shift_k=shift_k)
    else:
        X = ppmi_from_sparse(S)

    # 4) L2 normalize rows
    Xn = l2_row_normalize(X)

    # 5) Reduce to d dims (PCA-like on sparse matrix)
    svd = TruncatedSVD(n_components=d, random_state=random_state)
    E = svd.fit_transform(Xn).astype(np.float32)  # (|V|, d)

    return vocab, E, svd


# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="/mnt/data/train_hypernyms.csv")
    ap.add_argument("--neg_csv", type=str, default="/mnt/data/non_hypernyms.csv")
    ap.add_argument("--out_dir", type=str, default="/mnt/data/wordnet_katz_directed_d25")
    ap.add_argument("--dim", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--prune_topk", type=int, default=400)
    ap.add_argument("--use_shifted_ppmi", action="store_true")
    ap.add_argument("--shift_k", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    neg_df = pd.read_csv(args.neg_csv)

    vocab, E, svd = train_embeddings(
        train_df=train_df,
        neg_df=neg_df,
        alpha=args.alpha,
        K=args.K,
        prune_topk=args.prune_topk,
        tol=args.tol,
        use_shifted_ppmi=args.use_shifted_ppmi,
        shift_k=args.shift_k,
        d=args.dim,
        random_state=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"embeddings_d{args.dim}.npy", E)
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # word2vec text format (spaces replaced for compatibility)
    with open(out_dir / f"embeddings_d{args.dim}.txt", "w", encoding="utf-8") as f:
        f.write(f"{len(vocab)} {args.dim}\n")
        for w, vec in zip(vocab, E):
            f.write(w.replace(" ", "_") + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")

    print("Saved to:", out_dir)
    print("Explained variance (TruncatedSVD sum):", float(svd.explained_variance_ratio_.sum()))


if __name__ == "__main__":
    main()
