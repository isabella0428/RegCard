#!/usr/bin/env python3
"""
Baseline comparison: Compare regularization approach with post-processing projection
for enforcing monotonicity. Uses constraint-based projection from the actual
predicate relationship in the workload (cmp / pairs): for each pair (left, right)
we enforce pred[left] >= pred[right]. No assumption is made that index order
corresponds to cardinality order (see workloads/job-cmp-light-pairs.csv).

Reads from logs/grid_search (see result_analysis.ipynb for data layout).
Outputs a comparison table and optional plots.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    IsotonicRegression = None


def _isotonic_pav(x, y, increasing=True):
    """Pool-adjacent-violators: project y to be monotonic in x (numpy-only)."""
    order = np.argsort(x)
    y_s = y[order].astype(np.float64).copy()
    if not increasing:
        y_s = -y_s
    n = len(y_s)
    # PAV: merge adjacent violators into blocks and set to block mean
    while True:
        diff = np.diff(y_s)
        violators = np.where(diff < -1e-12)[0]
        if len(violators) == 0:
            break
        i = violators[0]
        j = i + 1
        while j < n and y_s[j] < y_s[i] - 1e-12:
            j += 1
        block_mean = np.mean(y_s[i : j + 1])
        y_s[i : j + 1] = block_mean
    if not increasing:
        y_s = -y_s
    out = np.empty_like(y_s)
    out[order] = y_s
    return out


GS_DIR = os.path.join("logs", "grid_search")
WORKLOADS_DIR = "workloads"
EPS = 1e-12


def load_gs_results(gs_dir=GS_DIR):
    """Load all grid-search runs that have config, eval, test_qerror, test_monom."""
    rows = []
    for sub_dir in os.listdir(gs_dir):
        full_dir = os.path.join(gs_dir, sub_dir)
        if not os.path.isdir(full_dir):
            continue
        config_path = os.path.join(full_dir, "config.json")
        eval_path = os.path.join(full_dir, "eval.json")
        qerror_path = os.path.join(full_dir, "test_qerror.csv")
        monom_path = os.path.join(full_dir, "test_monom.csv")
        if not all(os.path.isfile(p) for p in [config_path, eval_path, qerror_path, monom_path]):
            continue
        with open(config_path) as f:
            config = json.load(f)
        with open(eval_path) as f:
            eval_dict = json.load(f)
        row = {**config, **eval_dict, "run_dir": full_dir, "timestamp": sub_dir}
        rows.append(row)
    return pd.DataFrame(rows)


def isotonic_projection(x, y, increasing=False):
    """
    Project y to be monotonic in x. x, y are 1d arrays (e.g. index and pred).
    increasing=False: higher x -> lower or equal y (for cmp order like 5026>5027).
    """
    if IsotonicRegression is not None:
        ir = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        ir.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return ir.predict(np.asarray(x, dtype=np.float64))
    return _isotonic_pav(np.asarray(x), np.asarray(y), increasing=increasing)


def compute_qerror(pred, label):
    """Per-query Q-error: max(pred/label, label/pred) with EPS guard."""
    pred = max(float(pred), EPS)
    label = max(float(label), EPS)
    return pred / label if pred > label else label / pred


def compute_monom_from_pairs(preds_by_idx, monom_df):
    """Recompute MonoM for each pair using given prediction array (index -> pred)."""
    monom = []
    for _, row in monom_df.iterrows():
        left, right = int(row["left"]), int(row["right"])
        if 0 <= left < len(preds_by_idx) and 0 <= right < len(preds_by_idx):
            monom.append(1 if preds_by_idx[left] >= preds_by_idx[right] else 0)
        else:
            monom.append(np.nan)
    return np.array(monom)


def _constraint_graph_components(constraint_pairs, n_nodes):
    """
    Build directed graph from (left, right): left >= right => edge left -> right.
    Return list of connected components (each component = list of nodes in one chain).
    """
    pairs = [(int(left), int(right)) for left, right in constraint_pairs if 0 <= int(left) < n_nodes and 0 <= int(right) < n_nodes]
    out_edges = {}
    in_degree = {}
    for left, right in pairs:
        out_edges.setdefault(left, []).append(right)
        in_degree[right] = in_degree.get(right, 0) + 1
        in_degree.setdefault(left, 0)
    # Undirected graph for connected components (same chain)
    undir = {}
    for left, right in pairs:
        undir.setdefault(left, set()).add(right)
        undir.setdefault(right, set()).add(left)
    visited = set()
    components = []
    for start in range(n_nodes):
        if start in visited:
            continue
        stack = [start]
        comp = set()
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            for v in undir.get(u, []):
                if v not in visited:
                    stack.append(v)
        if comp:
            components.append(comp)
    return components, out_edges, in_degree


def _topological_order_component(component, out_edges, in_degree):
    """Topological order (largest to smallest cardinality) for one component."""
    from collections import deque
    comp = set(component)
    deg = {i: in_degree.get(i, 0) for i in comp}
    queue = deque(i for i in comp if deg[i] == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in out_edges.get(u, []):
            if v not in comp:
                continue
            deg[v] -= 1
            if deg[v] == 0:
                queue.append(v)
    return order


def project_to_cmp_constraints(pred_by_idx, constraint_pairs):
    """
    Project predictions to satisfy all cmp constraints: for each (left, right)
    we need pred[left] >= pred[right]. Uses the constraint graph: for each
    connected component (chain), get topological order and run isotonic
    regression (PAV) so predictions are non-increasing along that chain.
    """
    n = len(pred_by_idx)
    components, out_edges, in_degree = _constraint_graph_components(constraint_pairs, n)
    out = pred_by_idx.copy()
    for comp in components:
        order = _topological_order_component(comp, out_edges, in_degree)
        if not order:
            continue
        preds_ordered = np.array([out[i] if not np.isnan(out[i]) else np.nan for i in order])
        valid = ~np.isnan(preds_ordered)
        if not np.any(valid):
            continue
        preds_valid = preds_ordered[valid]
        ranks_valid = np.arange(np.sum(valid), dtype=np.float64)
        proj_valid = isotonic_projection(ranks_valid, preds_valid, increasing=False)
        idx = 0
        for i, node in enumerate(order):
            if valid[i]:
                out[node] = proj_valid[idx]
                idx += 1
    return out


def run_postprocessing(qerror_path, monom_path):
    """
    Load unregularized test_qerror and test_monom, apply constraint-based
    projection using the actual (left, right) pairs from the cmp: enforce
    pred[left] >= pred[right] for each pair (no assumption on index vs cardinality).
    """
    qdf = pd.read_csv(qerror_path)
    mdf = pd.read_csv(monom_path)
    idx = qdf.idx.values
    pred = qdf.pred.values.astype(np.float64)
    label = qdf.label.values.astype(np.float64)

    # Build full prediction array by index (test set may not be 0..n-1)
    max_idx = int(max(idx))
    pred_by_idx = np.full(max_idx + 1, np.nan)
    pred_by_idx[idx] = pred

    # Constraint pairs from monom (cmp): we need pred[left] >= pred[right] for each row
    constraint_pairs = mdf[["left", "right"]].values

    # Project to satisfy all cmp constraints (no index-order assumption)
    pred_by_idx = project_to_cmp_constraints(pred_by_idx, constraint_pairs)

    # Per-query projected predictions (only for rows in qerror csv)
    proj_pred = pred_by_idx[idx]

    # Q-error on projected predictions
    qerrors_proj = np.array([compute_qerror(proj_pred[i], label[i]) for i in range(len(idx))])
    qerror_median = np.median(qerrors_proj)
    qerror_25 = np.percentile(qerrors_proj, 25)
    qerror_75 = np.percentile(qerrors_proj, 75)
    qerror_mean = np.mean(qerrors_proj)

    # MonoM on projected predictions
    monom_proj = compute_monom_from_pairs(pred_by_idx, mdf)
    monom_proj = monom_proj[~np.isnan(monom_proj)]
    monom_mean = float(np.mean(monom_proj))
    monom_std = float(np.std(monom_proj))

    return {
        "qerror_median": qerror_median,
        "qerror_mean": qerror_mean,
        "qerror_25": qerror_25,
        "qerror_75": qerror_75,
        "monom_mean": monom_mean,
        "monom_std": monom_std,
        "qerrors": qerrors_proj,
        "monom_scores": monom_proj,
        "proj_pred": proj_pred,
        "idx": idx,
        "label": label,
        "qerror_df": qdf,
        "monom_df": mdf,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare regularization vs post-processing baseline")
    parser.add_argument("--gs-dir", default=GS_DIR, help="Grid search logs directory")
    parser.add_argument("--testset", default="job-cmp-card", help="Test set name to filter runs")
    parser.add_argument("--hid", type=int, default=256, help="Number of hidden units to compare")
    parser.add_argument("--out", default=None, help="Output CSV path for comparison table")
    parser.add_argument("--verbose", action="store_true", help="Print per-run details")
    args = parser.parse_args()

    df = load_gs_results(args.gs_dir)
    if df.empty:
        print("No grid-search runs found with config, eval, test_qerror, test_monom.", file=sys.stderr)
        sys.exit(1)

    # Filter by testset if present in config
    if "testset" in df.columns:
        df = df[df.testset == args.testset]
    if df.empty:
        print("No runs for testset", args.testset, file=sys.stderr)
        sys.exit(1)

    # Pick one unregularized run (lbda=0, matching hid)
    unreg = df[(df.lbda == 0) & (df.num_hidden_units == args.hid)]
    if unreg.empty:
        unreg = df[df.lbda == 0]
    if unreg.empty:
        print("No unregularized (lbda=0) run found.", file=sys.stderr)
        sys.exit(1)
    unreg_dir = unreg.iloc[0].run_dir
    unreg_eval = unreg.iloc[0]

    # Run post-processing on unregularized predictions
    qerror_path = os.path.join(unreg_dir, "test_qerror.csv")
    monom_path = os.path.join(unreg_dir, "test_monom.csv")
    post = run_postprocessing(qerror_path, monom_path)

    # Build comparison table
    results = []
    # Row 1: Unregularized
    results.append({
        "method": "Unregularized",
        "lbda": 0,
        "qerror_median": unreg_eval["qerror_median"],
        "qerror_25": unreg_eval.get("qerror_25", np.nan),
        "qerror_75": unreg_eval.get("qerror_75", np.nan),
        "monom_mean": unreg_eval["monom_mean"],
        "monom_std": unreg_eval.get("monom_std", np.nan),
    })
    # Row 2: Post-processing (constraint-based from cmp pairs)
    results.append({
        "method": "Post-processing (cmp constraints)",
        "lbda": np.nan,
        "qerror_median": post["qerror_median"],
        "qerror_25": post["qerror_25"],
        "qerror_75": post["qerror_75"],
        "monom_mean": post["monom_mean"],
        "monom_std": post["monom_std"],
    })
    # Rows 3+: Best regularized for lbda in [0.1, 1, 10]
    regularized = df[(df.lbda > 0) & (df.num_hidden_units == args.hid)]
    if regularized.empty:
        regularized = df[df.lbda > 0]
    for lbda in [0.1, 1, 10]:
        sub = regularized[regularized.lbda == lbda]
        if sub.empty:
            continue
        # Best by monom_mean
        best = sub.loc[sub.monom_mean.idxmax()]
        results.append({
            "method": f"Regularization (λ={lbda})",
            "lbda": lbda,
            "qerror_median": best["qerror_median"],
            "qerror_25": best.get("qerror_25", np.nan),
            "qerror_75": best.get("qerror_75", np.nan),
            "monom_mean": best["monom_mean"],
            "monom_std": best.get("monom_std", np.nan),
        })

    table = pd.DataFrame(results)
    print("Baseline comparison (same test set, unreg run from:", unreg_dir, ")")
    print(table.to_string(index=False))

    if args.out:
        table.to_csv(args.out, index=False)
        print("Wrote", args.out)

    if args.verbose:
        print("\nPost-processing: constraint-based projection (pred[left] >= pred[right] for each cmp pair) applied to unregularized predictions.")
        print("MonoM after projection:", post["monom_mean"], "(perfect monotonicity => 1.0)")

    return table, post


if __name__ == "__main__":
    main()
