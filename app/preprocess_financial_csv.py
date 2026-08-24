import argparse
from pathlib import Path

import pandas as pd


REQUIRED_OUTPUT_COLUMNS = ["user_id", "month", "spend", "bill", "balance", "credit_score", "income"]


def parse_column_map(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pair in value.split(","):
        if not pair.strip():
            continue
        output_column, source_column = pair.split("=", 1)
        mapping[output_column.strip()] = source_column.strip()
    missing = [column for column in REQUIRED_OUTPUT_COLUMNS if column not in mapping]
    if missing:
        raise ValueError(f"Missing mappings: {', '.join(missing)}")
    return mapping


def build_user_month_features(input_path: Path, output_path: Path, column_map: dict[str, str]) -> None:
    frame = pd.read_csv(input_path)
    missing_source_columns = [source for source in column_map.values() if source not in frame.columns]
    if missing_source_columns:
        raise ValueError(f"Source columns not found: {', '.join(missing_source_columns)}")

    result = pd.DataFrame({output: frame[source] for output, source in column_map.items()})
    result = result[REQUIRED_OUTPUT_COLUMNS].dropna(subset=["user_id", "month"])
    for column in ["spend", "bill", "balance", "credit_score", "income"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["spend", "bill"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert approved financial CSV data to service User-Month feature CSV.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("../Data/processed/user_month_features.csv"), type=Path)
    parser.add_argument(
        "--columns",
        required=True,
        help="Comma-separated mapping, e.g. user_id=CUST_ID,month=BAS_YM,spend=USE_AM,bill=BILL_AM,balance=BAL_AM,credit_score=CRDT_SCORE,income=INCOME_AM",
    )
    args = parser.parse_args()
    build_user_month_features(args.input, args.output, parse_column_map(args.columns))


if __name__ == "__main__":
    main()
