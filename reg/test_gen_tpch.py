"""
Generate TPC-H test workload for monotonicity evaluation.
Reads a TPC-H train-format CSV, expands queries by varying selected predicate columns,
and writes -card.csv (query templates) and -pairs.csv (comparison pairs), same format as test_gen.py.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


# Columns we can vary for monotonicity chains: (predicate_pattern, list of values or (min, max, step))
# Pattern is the column part that appears in predicates, e.g. "l.l_quantity" or "o.o_orderdate"
VARYABLE_NUMERIC = [
    ("l.l_quantity", list(range(5, 51, 5))),           # 5,10,...,50
    ("p.p_size", list(range(5, 51, 5))),
    ("n.n_nationkey", list(range(0, 25, 2))),        # 0,2,...,24
    ("r.r_regionkey", list(range(0, 5))),             # 0..4
    ("c.c_custkey", list(range(10000, 150000, 20000))),
    ("s.s_suppkey", list(range(10000, 100001, 15000))),
]
# Date columns: use a small set of dates (as string literals in predicates)
VARYABLE_DATES = [
    ("o.o_orderdate", ["'1992-01-01'", "'1994-01-01'", "'1995-06-01'", "'1996-12-01'", "'1998-12-01'"]),
    ("l.l_shipdate", ["'1992-01-01'", "'1994-01-01'", "'1995-06-01'", "'1996-12-01'", "'1998-12-01'"]),
    ("l.l_commitdate", ["'1992-01-01'", "'1994-01-01'", "'1996-01-01'", "'1998-01-01'"]),
]


def _match_predicate(parts, col_pattern):
    """Find triple (col, op, val) in parts where col matches col_pattern. parts = predicates.split(',')."""
    i = 0
    while i + 2 < len(parts):
        col, op, val = parts[i], parts[i + 1], parts[i + 2]
        if col_pattern in col or col == col_pattern:
            return i, col, op, val
        i += 3
    return None


def generate_new_line_tpch(row, base_idx):
    """Generate variant rows by varying one varyable column in predicates. Returns (rows, rows_with_meta)."""
    new_rows = []
    new_rows_with_meta = []
    predicates = row["predicates"]
    if not predicates or not str(predicates).strip():
        return new_rows, new_rows_with_meta

    parts = str(predicates).split(",")
    if len(parts) < 3:
        return new_rows, new_rows_with_meta

    # Check numeric columns
    for col_pattern, values in VARYABLE_NUMERIC:
        match = _match_predicate(parts, col_pattern)
        if match is None:
            continue
        idx, col, op, _ = match
        base_predicates = str(base_idx)
        if op == "=":
            for rel_ord, val in enumerate(values):
                new_parts = parts[:]
                new_parts[idx + 2] = str(val)
                new_pred = ",".join(new_parts)
                new_rows.append([row["tables"], row["joins"], new_pred, None])
                new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, "=", val, val, rel_ord])
        elif op == "<":
            for rel_ord, val in enumerate(values):
                new_parts = parts[:]
                new_parts[idx + 2] = str(val)
                new_pred = ",".join(new_parts)
                new_rows.append([row["tables"], row["joins"], new_pred, None])
                new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, "<", 0, val, rel_ord])
        elif op == ">":
            for rel_ord, val in enumerate(values):
                new_parts = parts[:]
                new_parts[idx + 2] = str(val)
                new_pred = ",".join(new_parts)
                new_rows.append([row["tables"], row["joins"], new_pred, None])
                new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, ">", val, 1000000, rel_ord])
        # Only vary first matching column per row
        break
    else:
        # Check date columns
        for col_pattern, values in VARYABLE_DATES:
            match = _match_predicate(parts, col_pattern)
            if match is None:
                continue
            idx, col, op, _ = match
            base_predicates = str(base_idx)
            if op == "=":
                for rel_ord, val in enumerate(values):
                    new_parts = parts[:]
                    new_parts[idx + 2] = val
                    new_pred = ",".join(new_parts)
                    new_rows.append([row["tables"], row["joins"], new_pred, None])
                    new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, "=", rel_ord, rel_ord, rel_ord])
            elif op == "<":
                for rel_ord, val in enumerate(values):
                    new_parts = parts[:]
                    new_parts[idx + 2] = val
                    new_pred = ",".join(new_parts)
                    new_rows.append([row["tables"], row["joins"], new_pred, None])
                    new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, "<", 0, rel_ord, rel_ord])
            elif op == ">":
                for rel_ord, val in enumerate(values):
                    new_parts = parts[:]
                    new_parts[idx + 2] = val
                    new_pred = ",".join(new_parts)
                    new_rows.append([row["tables"], row["joins"], new_pred, None])
                    new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, ">", rel_ord, 1000, rel_ord])
            break

    return new_rows, new_rows_with_meta


def get_cmp(df):
    """Build comparison pairs from expanded query metadata (same logic as test_gen.get_cmp)."""
    cmp = []
    df = df.sort_values(by=["base_predicates", "relative_order"])
    df["relative_order"] = df["relative_order"].astype(str)
    gp = df.groupby(["base_predicates", "type"])
    for (base_predicate, ptype), group in gp:
        if ptype in ("range", "<", ">"):
            for i in range(len(group.index)):
                if i == 0:
                    cmp.append(str(group.index[i]) + "=" + str(group.index[i]))
                    df.at[group.index[i], "relative_order"] = str(group.index[i]) + "=" + str(group.index[i])
                else:
                    cmp.append(str(group.index[i]) + ">" + str(group.index[i - 1]))
                    df.at[group.index[i], "relative_order"] = str(group.index[i]) + ">" + str(group.index[i - 1])
        if ptype == "=":
            for i in range(len(group.index) - 1, -1, -1):
                cmp.append(str(group.index[i]) + "=" + str(group.index[i]))
                df.at[group.index[i], "relative_order"] = str(group.index[i]) + "=" + str(group.index[i])
    return df, cmp


def generate_new_queries_tpch(input_path, output_path_prefix, include_passthrough=True):
    """
    input_path: path to CSV (train format: tables#joins#predicates#cardinality)
    output_path_prefix: e.g. data/tpch7k_final-cmp -> writes ...-cmp-card.csv and ...-cmp-pairs.csv
    include_passthrough: if True, also append rows that had no varyable column as single-query groups (self=self pair).
    """
    input_path = Path(input_path)
    prefix = Path(output_path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep="#", header=None, names=["tables", "joins", "predicates", "count"])
    # Drop cardinality for output
    df = df[["tables", "joins", "predicates"]]

    new_rows = []
    new_rows_with_meta = []
    passthrough_rows = []

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        rows, rows_meta = generate_new_line_tpch(row, base_idx=i)
        if rows:
            new_rows.extend(rows)
            new_rows_with_meta.extend(rows_meta)
        elif include_passthrough:
            passthrough_rows.append([row["tables"], row["joins"], row["predicates"]])

    # Queries only (no cardinality column)
    out_card = str(prefix) + "-card.csv"
    out_pairs = str(prefix) + "-pairs.csv"

    if new_rows_with_meta:
        new_df = pd.DataFrame(new_rows)[["tables", "joins", "predicates"]]
        meta_df = pd.DataFrame(
            new_rows_with_meta,
            columns=["tables", "joins", "predicates", "base_predicates", "type", "val1", "val2", "relative_order"],
        )
        _, cmp = get_cmp(meta_df.copy())
        new_df.to_csv(out_card, sep="#", index=False, header=False)
        pd.Series(cmp).to_csv(out_pairs, index=False, header=False)
        print(f"Wrote {len(new_rows)} variant queries -> {out_card}, {out_pairs} ({len(cmp)} pairs)")
    else:
        cmp = []

    if include_passthrough and passthrough_rows:
        pass_df = pd.DataFrame(passthrough_rows, columns=["tables", "joins", "predicates"])
        if new_rows:
            pass_df.to_csv(out_card, sep="#", index=False, header=False, mode="a")
            start_idx = len(new_rows)
            extra_pairs = [f"{start_idx + j}={start_idx + j}" for j in range(len(passthrough_rows))]
            pd.Series(cmp + extra_pairs).to_csv(out_pairs, index=False, header=False)
        else:
            pass_df.to_csv(out_card, sep="#", index=False, header=False)
            extra_pairs = [f"{j}={j}" for j in range(len(passthrough_rows))]
            pd.Series(extra_pairs).to_csv(out_pairs, index=False, header=False)
        print(f"Appended {len(passthrough_rows)} passthrough queries (no varyable column).")
    elif not new_rows:
        # No variants and no passthrough, or passthrough only
        if not include_passthrough or not passthrough_rows:
            print("No rows had a varyable predicate column. No output written.")
            return

    total_queries = len(new_rows) + (len(passthrough_rows) if include_passthrough else 0)
    print(f"Total queries: {total_queries}")


def main():
    parser = argparse.ArgumentParser(description="Generate TPC-H test workload (query variants + pairs) for monotonicity eval")
    parser.add_argument("-f", "--file", type=str, default="data/tpch7k_final.csv", help="Input CSV (train format, # sep)")
    parser.add_argument("-o", "--output", type=str, default="data/tpch7k_final-cmp", help="Output prefix: writes {prefix}-card.csv and {prefix}-pairs.csv")
    parser.add_argument("--no-passthrough", action="store_true", help="Do not include rows without a varyable column")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    input_path = Path(args.file) if os.path.isabs(args.file) else base / args.file
    output_prefix = Path(args.output) if os.path.isabs(args.output) else base / args.output

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return
    generate_new_queries_tpch(input_path, output_prefix, include_passthrough=not args.no_passthrough)


if __name__ == "__main__":
    main()
