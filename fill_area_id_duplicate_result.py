import argparse
import os
from typing import Dict, Optional

import pandas as pd


def _is_blank_series(series: pd.Series) -> pd.Series:
	# Treat NaN/None/empty/whitespace and literal 'nan'/'none' (string) as blank.
	# All operations are safe even if the series isn't strictly string typed.
	as_str = series.astype("string")
	stripped = as_str.str.strip()
	lowered = stripped.str.lower()
	return stripped.isna() | stripped.eq("") | lowered.isin(["nan", "none", "null"])


def _build_idsbr_to_kode_wilayah(combined_df: pd.DataFrame) -> Dict[str, str]:
	required = {"idsbr", "kode_wilayah"}
	missing = required - set(combined_df.columns)
	if missing:
		raise ValueError(f"combined_data is missing required columns: {sorted(missing)}")

	idsbr = combined_df["idsbr"].astype("string").str.strip()
	kode = combined_df["kode_wilayah"].astype("string").str.strip()

	valid = (~_is_blank_series(idsbr)) & (~_is_blank_series(kode))
	lookup_df = pd.DataFrame({"idsbr": idsbr[valid], "kode_wilayah": kode[valid]})

	# If there are duplicate idsbr values, keep the first non-empty kode_wilayah.
	lookup_df = lookup_df.drop_duplicates(subset=["idsbr"], keep="first")
	return dict(zip(lookup_df["idsbr"].tolist(), lookup_df["kode_wilayah"].tolist()))


def _normalize_sls_code_series(series: pd.Series) -> pd.Series:
	# Common case: codes written as floats in CSV become like "3501...1700.0".
	# Convert ONLY when the decimal part is all zeros.
	as_str = series.astype("string")
	stripped = as_str.str.strip()
	normalized = stripped.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
	return normalized


def _default_summary_path(output_path: str) -> str:
	base, ext = os.path.splitext(output_path)
	if not ext:
		ext = ".csv"
	return f"{base}_summary_by_regency{ext}"


def _write_summary_by_regency(dup_df: pd.DataFrame, summary_path: str) -> None:
	center_area = dup_df.get("center_area_id")
	if center_area is None:
		raise ValueError("Expected column 'center_area_id' to exist for summary")

	clean = center_area.astype("string").str.strip()
	clean = _normalize_sls_code_series(clean)
	regency = clean.str.slice(0, 4)
	regency = regency.where(~_is_blank_series(clean), other="UNKNOWN")

	summary_df = (
		pd.DataFrame({"regency": regency})
		.groupby("regency", dropna=False)
		.size()
		.reset_index(name="duplicate_count")
		.sort_values(["duplicate_count", "regency"], ascending=[False, True])
	)

	os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
	summary_df.to_csv(summary_path, index=False, encoding="utf-8")
	print(f"Summary (by first 4 of center_area_id) wrote: {summary_path}")
	print(summary_df.to_string(index=False))


def _resolve_combined_path(path: str) -> str:
	if os.path.exists(path):
		return path
	# Handle common typo from the prompt: commbined_data.csv
	alt = os.path.join(os.path.dirname(path), "commbined_data.csv")
	if os.path.exists(alt):
		return alt
	return path


def fill_missing_sls_ids(
	duplicate_path: str,
	combined_path: str,
	output_path: str,
	inplace: bool = False,
	summary_output_path: Optional[str] = None,
) -> None:
	combined_path = _resolve_combined_path(combined_path)
	if not os.path.exists(duplicate_path):
		raise FileNotFoundError(f"duplicate_detection_all not found: {duplicate_path}")
	if not os.path.exists(combined_path):
		raise FileNotFoundError(f"combined_data not found: {combined_path}")

	dup_df = pd.read_csv(duplicate_path, dtype="string", keep_default_na=False)
	combined_df = pd.read_csv(combined_path, dtype="string", keep_default_na=False)

	required_dup_cols = {"center_business_id", "nearby_business_id"}
	missing_dup = required_dup_cols - set(dup_df.columns)
	if missing_dup:
		raise ValueError(
			f"duplicate_detection_all is missing required columns: {sorted(missing_dup)}"
		)

	# Support both old and new column naming.
	center_col = "center_sls_id" if "center_sls_id" in dup_df.columns else "center_area_id"
	nearby_col = "nearby_sls_id" if "nearby_sls_id" in dup_df.columns else "nearby_area_id"
	if center_col not in dup_df.columns:
		dup_df[center_col] = ""
	if nearby_col not in dup_df.columns:
		dup_df[nearby_col] = ""

	lookup = _build_idsbr_to_kode_wilayah(combined_df)

	center_id = dup_df["center_business_id"].astype("string").str.strip()
	nearby_id = dup_df["nearby_business_id"].astype("string").str.strip()
	center_sls = dup_df[center_col].astype("string")
	nearby_sls = dup_df[nearby_col].astype("string")

	# Fill center_sls_id only where blank and lookup is available.
	center_blank = _is_blank_series(center_sls)
	center_mapped = center_id.map(lookup)
	center_can_fill = center_blank & (~_is_blank_series(center_id)) & (~_is_blank_series(center_mapped))
	dup_df.loc[center_can_fill, center_col] = center_mapped[center_can_fill]

	# Fill nearby_sls_id only where blank and lookup is available.
	nearby_blank = _is_blank_series(nearby_sls)
	nearby_mapped = nearby_id.map(lookup)
	nearby_can_fill = nearby_blank & (~_is_blank_series(nearby_id)) & (~_is_blank_series(nearby_mapped))
	dup_df.loc[nearby_can_fill, nearby_col] = nearby_mapped[nearby_can_fill]

	# Normalize SLS columns like 3501...1700.0 -> 3501...1700 (only if decimal is all zeros)
	dup_df[center_col] = _normalize_sls_code_series(dup_df[center_col])
	dup_df[nearby_col] = _normalize_sls_code_series(dup_df[nearby_col])

	# Rename headers to requested output names.
	# If the file already uses center_area_id/nearby_area_id, this is a no-op.
	if center_col != "center_area_id":
		dup_df = dup_df.rename(columns={center_col: "center_area_id"})
	if nearby_col != "nearby_area_id":
		dup_df = dup_df.rename(columns={nearby_col: "nearby_area_id"})

	if inplace:
		output_path = duplicate_path
	else:
		os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

	dup_df.to_csv(output_path, index=False, encoding="utf-8")

	filled_center = int(center_can_fill.sum())
	filled_nearby = int(nearby_can_fill.sum())
	print(f"Filled center_area_id: {filled_center} rows")
	print(f"Filled nearby_area_id: {filled_nearby} rows")
	print(f"Wrote: {output_path}")

	# Summary of duplicates (row count) by regency code (first 4 chars of center_area_id)
	if summary_output_path is None:
		summary_output_path = _default_summary_path(output_path)
	_write_summary_by_regency(dup_df, summary_output_path)


def main(argv: Optional[list[str]] = None) -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Fill empty center_sls_id / nearby_sls_id in duplicate_detection_all.csv using "
			"idsbr -> kode_wilayah mapping from combined_data.csv. Only empty cells are filled."
		)
	)
	parser.add_argument(
		"--duplicate",
		default=os.path.join("result", "duplicate_detection_all.csv"),
		help="Path to result/duplicate_detection_all.csv",
	)
	parser.add_argument(
		"--combined",
		default=os.path.join("source_matcha_pro_all", "combined_data.csv"),
		help="Path to source_matcha_pro_all/combined_data.csv",
	)
	parser.add_argument(
		"--output",
		default=os.path.join("result", "duplicate_detection_all_filled.csv"),
		help="Output CSV path (ignored if --inplace)",
	)
	parser.add_argument(
		"--inplace",
		action="store_true",
		help="Overwrite the duplicate_detection_all.csv in-place",
	)
	parser.add_argument(
		"--summary-output",
		default=None,
		help=(
			"Write summary CSV (count by first 4 chars of center_area_id) to this path. "
			"Defaults to <output>_summary_by_regency.csv"
		),
	)
	args = parser.parse_args(argv)

	fill_missing_sls_ids(
		duplicate_path=args.duplicate,
		combined_path=args.combined,
		output_path=args.output,
		inplace=args.inplace,
		summary_output_path=args.summary_output,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

