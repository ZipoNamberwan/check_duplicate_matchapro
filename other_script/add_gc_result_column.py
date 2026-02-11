from __future__ import annotations

from pathlib import Path

import pandas as pd


DUPLICATE_PATH = Path("result") / "duplicate_detection_all_filled.csv"
COMBINED_PATH = Path("source_matcha_pro_all") / "combined_data.csv"
OUTPUT_PATH = Path("result") / "duplicate_detection_all_filled_gc_result.csv"

NEARBY_ID_COLUMN = "nearby_business_id"
CENTER_ID_COLUMN = "center_business_id"


def resolve_gc_username_column(columns: list[str]) -> str:
	if "gc.username" in columns:
		return "gc.username"
	if "gc_username" in columns:
		return "gc_username"
	raise ValueError("Missing gc username column (gc.username or gc_username) in combined data.")


def main() -> None:
	if not DUPLICATE_PATH.exists():
		raise FileNotFoundError(f"File not found: {DUPLICATE_PATH}")
	if not COMBINED_PATH.exists():
		raise FileNotFoundError(f"File not found: {COMBINED_PATH}")

	combined_header = pd.read_csv(COMBINED_PATH, nrows=0, low_memory=False)
	combined_columns = list(combined_header.columns)

	if "idsbr" not in combined_columns:
		raise ValueError("Missing idsbr column in combined data.")
	if "gcs_result" not in combined_columns:
		raise ValueError("Missing gcs_result column in combined data.")

	gc_username_col = resolve_gc_username_column(combined_columns)
	combined_usecols = ["idsbr", "gcs_result", gc_username_col]
	combined = pd.read_csv(COMBINED_PATH, usecols=combined_usecols, dtype=str, low_memory=False)

	duplicates = pd.read_csv(DUPLICATE_PATH, dtype=str, low_memory=False)
	for required_col in [NEARBY_ID_COLUMN, CENTER_ID_COLUMN]:
		if required_col not in duplicates.columns:
			raise ValueError(f"Missing {required_col} column in duplicate data.")

	combined["idsbr"] = combined["idsbr"].fillna("").astype(str)
	combined["gcs_result"] = combined["gcs_result"].fillna("")
	combined[gc_username_col] = combined[gc_username_col].fillna("")

	gcs_map = combined.set_index("idsbr")["gcs_result"].to_dict()
	gc_user_map = combined.set_index("idsbr")[gc_username_col].to_dict()

	for prefix, id_col in [("nearby", NEARBY_ID_COLUMN), ("center", CENTER_ID_COLUMN)]:
		duplicates[f"{prefix}_gcs_result"] = duplicates[id_col].map(gcs_map)
		duplicates[f"{prefix}_gc_username"] = duplicates[id_col].map(gc_user_map)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	duplicates.to_csv(OUTPUT_PATH, index=False)
	print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
