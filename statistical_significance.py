#!/usr/bin/env python3
"""
Statistical significance testing for Q-error and MonoM improvements.
Uses paired t-tests and confidence intervals (bootstrap and/or normal)
to ensure reported improvements are not due to random variation.

Reads from logs/grid_search; see result_analysis.ipynb for data layout.
"""

import argparse
import json
import os
import sys

import json
import math
import numpy as np
import pandas as pd
try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


GS_DIR = os.path.join("logs", "grid_search")


def load_run_metrics(run_dir):
    """Load per-query qerror and per-pair MonoM from a run directory."""
    qpath = os.path.join(run_dir, "test_qerror.csv")
    mpath = os.path.join(run_dir, "test_monom.csv")
    if not os.path.isfile(qpath) or not os.path.isfile(mpath):
        return None
    qdf = pd.read_csv(qpath)
    mdf = pd.read_csv(mpath)
    return {
        "qerror": qdf.qerror.values.astype(np.float64),
        "idx": qdf.idx.values,
        "pred": qdf.pred.values,
        "label": qdf.label.values,
        "monom": mdf.MonoM.values.astype(np.float64),
        "n_queries": len(qdf),
        "n_pairs": len(mdf),
    }


def _norm_cdf(x):
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * abs(x))
    y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return (1.0 + (1.0 if x >= 0 else -1.0) * y) / 2.0


def paired_ttest_and_ci(a, b, metric_name="", alpha=0.05):
    """
    Paired t-test and confidence interval for difference (a - b).
    Returns dict with statistic, p_value, ci_low, ci_high, mean_diff.
    Uses scipy if available, else normal approximation for large n.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("Paired test requires same length: {} vs {}".format(len(a), len(b)))
    n = len(a)
    diff = a - b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0 or n < 2:
        t_stat = 0.0 if mean_diff == 0 else (np.inf if mean_diff > 0 else -np.inf)
        p_value = 0.0 if mean_diff != 0 else 1.0
        ci_low = ci_high = mean_diff
    else:
        se = std_diff / math.sqrt(n)
        t_stat = mean_diff / se
        if scipy_stats is not None:
            p_value = float(scipy_stats.ttest_rel(a, b).pvalue)
            h = se * scipy_stats.t.ppf(1 - alpha / 2, n - 1)
        else:
            # Normal approximation for large n
            p_value = 2 * (1 - _norm_cdf(abs(t_stat)))
            z = 1.96 if alpha == 0.05 else (2.576 if alpha == 0.01 else 2.0)
            h = se * z
        ci_low = mean_diff - h
        ci_high = mean_diff + h
    return {
        "metric": metric_name,
        "mean_diff": mean_diff,
        "std_diff": float(std_diff),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "n": n,
    }


def bootstrap_ci(a, b, statistic="median", n_bootstrap=2000, alpha=0.05, seed=42):
    """
    Bootstrap CI for the difference in statistic (e.g. median of a - median of b).
    For Q-error we often care about median; for MonoM we use mean.
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    assert n == len(b)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if statistic == "median":
            diffs.append(np.median(a[idx]) - np.median(b[idx]))
        else:
            diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    diffs = np.array(diffs)
    ci_low = np.percentile(diffs, 100 * alpha / 2)
    ci_high = np.percentile(diffs, 100 * (1 - alpha / 2))
    return ci_low, ci_high


def run_significance_tests(run_a_dir, run_b_dir, name_a="A", name_b="B", alpha=0.05):
    """
    Load two runs and compute paired t-tests and CIs for Q-error and MonoM.
    For Q-error we compare per-query values (same idx order in both CSVs).
    For MonoM we compare per-pair values (same row order in both test_monom).
    """
    ma = load_run_metrics(run_a_dir)
    mb = load_run_metrics(run_b_dir)
    if ma is None or mb is None:
        print("Missing test_qerror.csv or test_monom.csv in one or both runs.", file=sys.stderr)
        return None

    # Align by query index for qerror (both CSVs should have same order if same test set)
    qa, qb = ma["qerror"], mb["qerror"]
    if len(qa) != len(qb):
        print("Different number of queries:", len(qa), "vs", len(qb), file=sys.stderr)
        # Try aligning by idx
        idf_a = pd.DataFrame({"idx": ma["idx"], "qerror": qa})
        idf_b = pd.DataFrame({"idx": mb["idx"], "qerror": qb})
        merged = idf_a.merge(idf_b, on="idx", suffixes=("_a", "_b"))
        qa = merged.qerror_a.values
        qb = merged.qerror_b.values
    if len(qa) == 0:
        print("No overlapping queries.", file=sys.stderr)
        return None

    # Q-error: paired t-test (lower is better; diff = A - B so positive => B better)
    qres = paired_ttest_and_ci(qa, qb, "Q-error (per-query)", alpha=alpha)
    qres["interpretation"] = "Positive mean_diff => {} has lower Q-error than {}.".format(name_b, name_a)
    qres_median_ci = bootstrap_ci(qa, qb, statistic="median", alpha=alpha)
    qres["bootstrap_ci_median_diff"] = qres_median_ci

    # MonoM: paired t-test (higher is better; diff = A - B so negative => B better)
    monom_a, monom_b = ma["monom"], mb["monom"]
    if len(monom_a) != len(monom_b):
        print("Different number of MonoM pairs:", len(monom_a), "vs", len(monom_b), file=sys.stderr)
        monom_n = min(len(monom_a), len(monom_b))
        monom_a, monom_b = monom_a[:monom_n], monom_b[:monom_n]
    mres = paired_ttest_and_ci(monom_a, monom_b, "MonoM (per-pair)", alpha=alpha)
    mres["interpretation"] = "Negative mean_diff => {} has higher MonoM than {}.".format(name_b, name_a)
    mres_mean_ci = bootstrap_ci(monom_a, monom_b, statistic="mean", alpha=alpha)
    mres["bootstrap_ci_mean_diff"] = mres_mean_ci

    return {
        "run_a": run_a_dir,
        "run_b": run_b_dir,
        "name_a": name_a,
        "name_b": name_b,
        "qerror": qres,
        "monom": mres,
        "n_queries": len(qa),
        "n_pairs": len(monom_a),
    }


def load_gs_and_pick_runs(gs_dir, testset="job-cmp-card", hid=256):
    """Load grid search dataframe and pick one unreg and one reg run."""
    rows = []
    for sub in os.listdir(gs_dir):
        full = os.path.join(gs_dir, sub)
        if not os.path.isdir(full):
            continue
        cfg = os.path.join(full, "config.json")
        ev = os.path.join(full, "eval.json")
        if not os.path.isfile(cfg) or not os.path.isfile(ev):
            continue
        with open(cfg) as f:
            c = json.load(f)
        with open(ev) as f:
            e = json.load(f)
        rows.append({**c, **e, "run_dir": full})
    df = pd.DataFrame(rows)
    if "testset" in df.columns:
        df = df[df.testset == testset]
    unreg = df[(df.lbda == 0) & (df.num_hidden_units == hid)]
    if unreg.empty:
        unreg = df[df.lbda == 0]
    reg = df[(df.lbda > 0) & (df.num_hidden_units == hid)]
    if reg.empty:
        reg = df[df.lbda > 0]
    return unreg, reg, df


def main():
    parser = argparse.ArgumentParser(description="Statistical significance for Q-error and MonoM")
    parser.add_argument("--gs-dir", default=GS_DIR, help="Grid search directory")
    parser.add_argument("--testset", default="job-cmp-card", help="Test set name")
    parser.add_argument("--hid", type=int, default=256, help="Hidden units")
    parser.add_argument("--run-a", default=None, help="Path to run A (baseline)")
    parser.add_argument("--run-b", default=None, help="Path to run B (e.g. regularized)")
    parser.add_argument("--lbda", type=float, default=0.1, help="Regularization λ to select (when using gs-dir)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for CIs")
    parser.add_argument("--out", default=None, help="Write summary to this JSON file")
    args = parser.parse_args()

    if args.run_a and args.run_b:
        run_a_dir = args.run_a
        run_b_dir = args.run_b
        name_a, name_b = "A", "B"
    else:
        unreg, reg, _ = load_gs_and_pick_runs(args.gs_dir, args.testset, args.hid)
        if unreg.empty:
            print("No unregularized run found.", file=sys.stderr)
            sys.exit(1)
        reg_at_lbda = reg[reg.lbda == args.lbda]
        if reg_at_lbda.empty:
            print("No regularized run with lbda={} found.".format(args.lbda), file=sys.stderr)
            sys.exit(1)
        run_a_dir = unreg.iloc[0].run_dir
        selected_reg = reg_at_lbda.iloc[0]
        run_b_dir = selected_reg.run_dir
        name_a = "Unregularized"
        name_b = "Regularized (lbda={})".format(selected_reg.lbda)

    result = run_significance_tests(
        run_a_dir, run_b_dir, name_a=name_a, name_b=name_b, alpha=args.alpha
    )
    if result is None:
        sys.exit(1)

    # Report
    print("Statistical significance: {} vs {}".format(name_a, name_b))
    print("  Run A:", run_a_dir)
    print("  Run B:", run_b_dir)
    print()
    q = result["qerror"]
    print("Q-error (per-query, paired):")
    print("  Mean difference (A - B): {:.6f}  (positive => B has lower Q-error)".format(q["mean_diff"]))
    print("  95% CI for mean diff: [{:.6f}, {:.6f}]".format(q["ci_95_low"], q["ci_95_high"]))
    print("  Paired t-test: t = {:.4f}, p = {:.4e}".format(q["t_statistic"], q["p_value"]))
    print("  Bootstrap 95% CI for median(Q_A) - median(Q_B):", q["bootstrap_ci_median_diff"])
    print()
    m = result["monom"]
    print("MonoM (per-pair, paired):")
    print("  Mean difference (A - B): {:.6f}  (negative => B has higher MonoM)".format(m["mean_diff"]))
    print("  95% CI for mean diff: [{:.6f}, {:.6f}]".format(m["ci_95_low"], m["ci_95_high"]))
    print("  Paired t-test: t = {:.4f}, p = {:.4e}".format(m["t_statistic"], m["p_value"]))
    print("  Bootstrap 95% CI for mean(MonoM_A) - mean(MonoM_B):", m["bootstrap_ci_mean_diff"])
    print()
    print("Conclusion:")
    if q["p_value"] < args.alpha:
        print("  Q-error: difference is statistically significant (p < {}).".format(args.alpha))
    else:
        print("  Q-error: difference is not statistically significant (p >= {}).".format(args.alpha))
    if m["p_value"] < args.alpha:
        print("  MonoM: difference is statistically significant (p < {}).".format(args.alpha))
    else:
        print("  MonoM: difference is not statistically significant (p >= {}).".format(args.alpha))

    # Serializable summary for --out
    def to_python(x):
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, tuple):
            return [float(t) for t in x]
        return x
    summary = {
        "run_a": result["run_a"],
        "run_b": result["run_b"],
        "name_a": result["name_a"],
        "name_b": result["name_b"],
        "n_queries": int(result["n_queries"]),
        "n_pairs": int(result["n_pairs"]),
        "qerror": {k: to_python(v) for k, v in q.items() if k != "interpretation"},
        "monom": {k: to_python(v) for k, v in m.items() if k != "interpretation"},
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print("Wrote", args.out)

    return result


if __name__ == "__main__":
    main()
