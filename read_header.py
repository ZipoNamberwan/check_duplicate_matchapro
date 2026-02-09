from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CSV = Path("result") / "match_sbr_kdm.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, dtype=str)


def get_unique_values(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")
    # Drop NaN, strip whitespace, keep unique
    values = (
        df[column]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    return values


def get_rows_by_value(df: pd.DataFrame, column: str, value: str, *, contains: bool = False) -> pd.DataFrame:
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")
    series = df[column].fillna("").astype(str)
    if contains:
        mask = series.str.contains(value, case=False, na=False)
    else:
        mask = series.str.strip() == value
    return df[mask].copy()


def get_total_rows(df: pd.DataFrame) -> int:
    return int(len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Small CSV helpers")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to CSV")

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_headers = subparsers.add_parser("headers", help="List all column headers")

    p_unique = subparsers.add_parser("unique", help="Get unique values from a column")
    p_unique.add_argument("--column", required=True)

    p_filter = subparsers.add_parser("filter", help="Get rows by column and value")
    p_filter.add_argument("--column", required=True)
    p_filter.add_argument("--value", required=True)
    p_filter.add_argument("--contains", action="store_true", help="Use contains match (case-insensitive)")
    p_filter.add_argument("--head", type=int, default=20, help="How many rows to print")

    p_count = subparsers.add_parser("count", help="Get total row count")

    args = parser.parse_args()
    df = read_csv(args.csv)

    if args.cmd == "headers":
        cols = list(df.columns)
        print(f"column_count={len(cols)}")
        for c in cols:
            print(c)
        return

    if args.cmd == "unique":
        values = get_unique_values(df, args.column)
        print(f"unique_count={len(values)}")
        for v in values:
            print(v)
        return

    if args.cmd == "filter":
        rows = get_rows_by_value(df, args.column, args.value, contains=args.contains)
        print(f"matched_rows={len(rows)}")
        if len(rows) > 0:
            print(rows.head(args.head).to_string(index=False))
        return

    if args.cmd == "count":
        print(get_total_rows(df))
        return

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()

