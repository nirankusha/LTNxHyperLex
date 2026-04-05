#!/usr/bin/env python3
"""
Evaluate all feature bundles (freq / prob333 / interval / interval+freq) against one or more test targets.

Expected files in --bundle_dir:
  X_freq_train.npz,      X_freq_test.npz
  X_prob333_train.npz,   X_prob333_test.npz
  X_int_train.npy,       X_int_test.npy
  X_int_freq_train.npy,  X_int_freq_test.npy
  y_train.npy
  y_test_*.npy   (any number of test targets, e.g. y_test_train_rowpath.npy, y_test_wordnet.npy, ...)

Outputs:
  - prints a metrics table to stdout
  - optionally writes CSV to --out_csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# Metrics (robust to single-class y)
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    return {
        "acc": acc,
        "prec_macro": float(np.mean(pr)),
        "rec_macro": float(np.mean(rc)),
        "f1_macro": float(np.mean(f1)),
        "prec_1": float(pr[1]),
        "rec_1": float(rc[1]),
        "f1_1": float(f1[1]),
        "sup_1": int(sup[1]),
        "sup_0": int(sup[0]),
        "pos_rate": float(np.mean(y_true == 1)),
        "pred_pos_rate": float(np.mean(y_pred == 1)),
    }


# -----------------------------
# Loaders
# -----------------------------
def load_matrix(train_path: Path, test_path: Path):
    if train_path.suffix == ".npz":
        Xtr = sp.load_npz(train_path).tocsr()
        Xte = sp.load_npz(test_path).tocsr()
        return Xtr, Xte, "sparse"
    elif train_path.suffix == ".npy":
        Xtr = np.load(train_path)
        Xte = np.load(test_path)
        return Xtr, Xte, "dense"
    else:
        raise ValueError(f"Unknown matrix format: {train_path}")


def find_targets(bundle_dir: Path, include_patterns=None):
    """
    Find y_test targets in bundle_dir. By default includes all y_test*.npy.
    """
    targets = {}
    pats = include_patterns or ["y_test*.npy"]
    for pat in pats:
        for p in bundle_dir.glob(pat):
            name = p.stem
            targets[name] = np.load(p).astype(np.int64)
    return targets


# -----------------------------
# Models
# -----------------------------
def run_logreg(X_train, y_train, X_test, y_test, *, max_iter=2000):
    # Choose solver that definitely supports sparse.
    solver = "liblinear" if sp.issparse(X_train) else "lbfgs"
    clf = LogisticRegression(max_iter=max_iter, solver=solver)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred)


class MLPBinary(nn.Module):
    def __init__(self, input_dim, hidden_ratio=0.8):
        super().__init__()
        hidden_dim = max(4, int(hidden_ratio * input_dim))
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.d1 = nn.Linear(input_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.elu(self.d1(x))
        x = self.elu(self.d2(x))
        x = self.sigmoid(self.d3(x))
        return x


def run_mlp(
    X_train, y_train, X_test, y_test,
    *,
    hidden_ratio=0.8,
    n_epochs=50,
    batch_size=64,
    lr=1e-3,
    max_dim=5000,
    device="cpu",
    seed=13
):
    # Skip huge dims by default (your X_freq is 51406 -> hidden ~41124; too big for typical RAM/VRAM)
    input_dim = X_train.shape[1]
    if input_dim > max_dim:
        return None, f"skipped (input_dim={input_dim} > max_dim={max_dim})"

    # Convert sparse to dense
    if sp.issparse(X_train):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

    X_train_t = torch.from_numpy(np.asarray(X_train)).float().to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1).to(device)
    X_test_t = torch.from_numpy(np.asarray(X_test)).float().to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    mlp = MLPBinary(input_dim=input_dim, hidden_ratio=hidden_ratio).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    mlp.train()
    for _ in range(n_epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            preds = mlp(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    mlp.eval()
    with torch.no_grad():
        y_proba = mlp(X_test_t).detach().cpu().numpy().ravel()
    y_pred = (y_proba >= 0.5).astype(np.int64)

    return compute_metrics(y_test, y_pred), None


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--targets_glob", type=str, default="y_test*.npy",
                    help="Glob to select which y_test targets to evaluate (default: y_test*.npy)")
    ap.add_argument("--mlp", action="store_true", help="Also run the MLP baseline where feasible.")
    ap.add_argument("--mlp_max_dim", type=int, default=5000, help="Skip MLP if input_dim is above this.")
    ap.add_argument("--mlp_epochs", type=int, default=50)
    ap.add_argument("--mlp_hidden_ratio", type=float, default=0.8)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir)
    y_train_path = bundle_dir / "y_train.npy"
    if not y_train_path.exists():
        raise FileNotFoundError(f"Missing {y_train_path}")

    y_train = np.load(y_train_path).astype(np.int64)

    targets = find_targets(bundle_dir, include_patterns=[args.targets_glob])
    if not targets:
        raise FileNotFoundError(f"No targets found for glob={args.targets_glob} in {bundle_dir}")

    feature_specs = [
        ("X_freq",      bundle_dir / "X_freq_train.npz",      bundle_dir / "X_freq_test.npz"),
        ("X_prob333",   bundle_dir / "X_prob333_train.npz",   bundle_dir / "X_prob333_test.npz"),
        ("X_int",       bundle_dir / "X_int_train.npy",       bundle_dir / "X_int_test.npy"),
        ("X_int_freq",  bundle_dir / "X_int_freq_train.npy",  bundle_dir / "X_int_freq_test.npy"),
    ]

    rows = []
    for feat_name, tr_path, te_path in feature_specs:
        if not tr_path.exists() or not te_path.exists():
            # quietly skip missing matrices
            continue

        Xtr, Xte, kind = load_matrix(tr_path, te_path)

        for tgt_name, y_test in targets.items():
            # logistic regression
            m = run_logreg(Xtr, y_train, Xte, y_test)
            rows.append({
                "matrix": feat_name,
                "target": tgt_name,
                "model": "LogReg",
                "kind": kind,
                **m
            })

            if args.mlp:
                m2, note = run_mlp(
                    Xtr, y_train, Xte, y_test,
                    hidden_ratio=args.mlp_hidden_ratio,
                    n_epochs=args.mlp_epochs,
                    batch_size=args.mlp_batch_size,
                    lr=args.mlp_lr,
                    max_dim=args.mlp_max_dim,
                    device=args.device,
                )
                rows.append({
                    "matrix": feat_name,
                    "target": tgt_name,
                    "model": "MLP(ELU-ELU-Sigmoid)",
                    "kind": kind,
                    "note": note or "",
                    **(m2 or {"acc": np.nan, "prec_macro": np.nan, "rec_macro": np.nan, "f1_macro": np.nan,
                              "prec_1": np.nan, "rec_1": np.nan, "f1_1": np.nan, "sup_1": int((y_test==1).sum()),
                              "sup_0": int((y_test==0).sum()),
                              "pos_rate": float(np.mean(y_test==1)), "pred_pos_rate": np.nan})
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No evaluations were run (missing matrices or targets?).")

    # nice order
    order = ["matrix", "target", "model", "kind", "acc", "f1_macro", "prec_macro", "rec_macro",
             "f1_1", "prec_1", "rec_1", "sup_1", "sup_0", "pos_rate", "pred_pos_rate"]
    if "note" in df.columns:
        order.insert(4, "note")
    df = df[[c for c in order if c in df.columns]]

    # print
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(df.sort_values(["matrix", "target", "model"]).to_string(index=False))

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
