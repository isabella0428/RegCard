#!/usr/bin/env python3
"""
Generate bitmaps and column_min_max_vals for a TPC-H workload CSV (train format).
Reads e.g. data/tpch7k_final.csv, connects to PostgreSQL (TPC-H), and produces:
  - data/tpch7k_final.bitmaps  (same format as data/train.bitmaps)
  - data/tpch7k_final_column_min_max_vals.csv  (same format as data/column_min_max_vals.csv)
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TPC-H schema: (table_name, alias) and predicate columns (alias, col_name, type)
TPCH_TABLES = [
    ("region", "r"),
    ("nation", "n"),
    ("supplier", "s"),
    ("customer", "c"),
    ("part", "p"),
    ("partsupp", "ps"),
    ("orders", "o"),
    ("lineitem", "l"),
]

# (alias, column_name, type) for columns that appear in predicates
PREDICATE_COLUMNS = [
    ("r", "r_regionkey", "int"),
    ("n", "n_nationkey", "int"),
    ("s", "s_suppkey", "int"),
    ("s", "s_acctbal", "decimal"),
    ("c", "c_custkey", "int"),
    ("c", "c_acctbal", "decimal"),
    ("p", "p_partkey", "int"),
    ("p", "p_size", "int"),
    ("p", "p_retailprice", "decimal"),
    ("ps", "ps_availqty", "int"),
    ("ps", "ps_supplycost", "decimal"),
    ("o", "o_orderkey", "int"),
    ("o", "o_totalprice", "decimal"),
    ("o", "o_orderdate", "date"),
    ("l", "l_quantity", "int"),
    ("l", "l_extendedprice", "decimal"),
    ("l", "l_discount", "decimal"),
    ("l", "l_shipdate", "date"),
    ("l", "l_commitdate", "date"),
]

ALIAS_TO_TABLE = {alias: tbl for tbl, alias in TPCH_TABLES}
# qualified name (alias.col) -> (table_name, column_name) for SQL
QUAL_TO_TABLE_COL = {}
for alias, col, _ in PREDICATE_COLUMNS:
    QUAL_TO_TABLE_COL[f"{alias}.{col}"] = (ALIAS_TO_TABLE[alias], col)


def connect_psql(db_user="postgres", db_host="localhost", db_port="5432", db_password="", db_name="tpch"):
    try:
        import psycopg2
    except ImportError:
        raise ImportError("Install psycopg2: pip install psycopg2-binary")
    conn = psycopg2.connect(
        user=db_user, host=db_host, port=db_port, password=db_password or None, database=db_name
    )
    conn.autocommit = True
    return conn


def ensure_materialized_views(cursor, num_samples=1000):
    """Create alias_view for each TPC-H table (sample of 1000 rows). Drop if exists first."""
    for table_name, alias in TPCH_TABLES:
        view_name = f"{alias}_view"
        try:
            cursor.execute(f"DROP MATERIALIZED VIEW IF EXISTS {view_name};")
        except Exception:
            pass
        sql = f"CREATE MATERIALIZED VIEW {view_name} AS SELECT * FROM {table_name} AS {alias} ORDER BY RANDOM() LIMIT {num_samples};"
        cursor.execute(sql)
    print("Materialized views created.")


def get_bitmap(row, cursor):
    """Same semantics as reg/data_gen.get_bitmap: one bitmap per table, from predicate evaluation on view."""
    tables = row[0].split(",")
    table_abbrs = [t.split()[1] for t in tables]
    predicates = row[2]
    all_bitmaps = np.zeros((len(table_abbrs), 1000), dtype=int)

    if not predicates or not predicates.strip():
        return np.packbits(all_bitmaps, axis=1)

    parts = predicates.split(",")
    num_predicates = len(parts) // 3
    for i in range(num_predicates):
        col, op, val = parts[3 * i], parts[3 * i + 1], parts[3 * i + 2]
        table_abbr = col.split(".")[0]
        pred_expr = f"{col}{op}{val}"  # e.g. l.l_quantity<18
        view_name = f"{table_abbr}_view"
        sql = f'SELECT CASE WHEN {pred_expr} THEN 1 ELSE 0 END AS bitmap FROM {view_name} AS {table_abbr};'
        try:
            cursor.execute(sql)
            record = np.array([r[0] for r in cursor.fetchall()], dtype=int)
            if len(record) < 1000:
                record = np.pad(record, (0, 1000 - len(record)), constant_values=0)
            idx = table_abbrs.index(table_abbr)
            all_bitmaps[idx] = record[:1000]
        except Exception as e:
            print(f"Bitmap predicate failed for {pred_expr}: {e}")
    return np.packbits(all_bitmaps, axis=1)


def generate_column_min_max_vals(cursor, output_path):
    """Query min, max, count, count(distinct) for each predicate column; write CSV like data/column_min_max_vals.csv."""
    rows = [["name", "min", "max", "cardinality", "num_unique_values"]]
    for alias, col, typ in PREDICATE_COLUMNS:
        table_name, col_name = ALIAS_TO_TABLE[alias], col
        qual = f"{alias}.{col}"
        if typ == "date":
            # Store as epoch for numeric normalization
            cursor.execute(
                f"SELECT MIN(EXTRACT(EPOCH FROM {col_name})), MAX(EXTRACT(EPOCH FROM {col_name})), "
                f"COUNT(*), COUNT(DISTINCT {col_name}) FROM {table_name};"
            )
        else:
            cursor.execute(
                f"SELECT MIN({col_name})::float, MAX({col_name})::float, COUNT(*), COUNT(DISTINCT {col_name}) FROM {table_name};"
            )
        r = cursor.fetchone()
        min_val, max_val, card, num_uniq = r[0], r[1], r[2], r[3]
        if min_val is None:
            min_val, max_val = 0, 0
        rows.append([qual, min_val, max_val, card, num_uniq])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"Wrote column min/max to {output_path}")


def generate_bitmaps(cursor, csv_path, bitmaps_path):
    """For each row in CSV, compute bitmap; save list of packed bitmaps to pickle."""
    with open(csv_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    bitmaps = []
    for i, line in enumerate(lines):
        parts = line.split("#")
        if len(parts) < 3:
            row = (parts[0] if parts else "", "", "")
        else:
            row = (parts[0], parts[1], parts[2])
        b = get_bitmap(row, cursor)
        bitmaps.append(b)

    bitmaps_path = Path(bitmaps_path)
    bitmaps_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bitmaps_path, "wb") as f:
        pickle.dump(bitmaps, f)
    print(f"Wrote {len(bitmaps)} bitmaps to {bitmaps_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate bitmaps and column_min_max_vals for TPC-H workload CSV")
    ap.add_argument("csv", type=str, default="data/tpch7k_final.csv", help="Input CSV (train format, # delimiter)")
    ap.add_argument("--db-user", default="postgres")
    ap.add_argument("--db-host", default="localhost")
    ap.add_argument("--db-port", default="5432")
    ap.add_argument("--db-password", default="")
    ap.add_argument("--db-name", default="tpch")
    ap.add_argument("--bitmaps-only", action="store_true", help="Only generate bitmaps")
    ap.add_argument("--column-stats-only", action="store_true", help="Only generate column_min_max_vals CSV")
    ap.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory (default: same dir as CSV)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent.parent / csv_path
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    base = csv_path.stem  # e.g. tpch7k_final
    column_min_max_path = out_dir / f"{base}_column_min_max_vals.csv"
    bitmaps_path = out_dir / f"{base}.bitmaps"

    conn = connect_psql(
        db_user=args.db_user,
        db_host=args.db_host,
        db_port=args.db_port,
        db_password=args.db_password,
        db_name=args.db_name,
    )
    cur = conn.cursor()

    try:
        ensure_materialized_views(cur)
    except Exception as e:
        print("Creating views failed (you may need to drop existing views first):", e)
        conn.close()
        sys.exit(1)

    if not args.column_stats_only:
        generate_bitmaps(cur, csv_path, bitmaps_path)
    if not args.bitmaps_only:
        generate_column_min_max_vals(cur, column_min_max_path)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
