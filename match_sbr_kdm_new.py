#!/usr/bin/env python3
"""Match SBR (source_matcha_pro_all) to KDM (source_kdm_all).

Rules (per request):
- Base dataset is SBR.
- Output includes all SBR rows with non-empty latitude & longitude.
- Coordinates are validated/fixed by converting comma decimals to point.
- Matching is strict: kode_wilayah[:10] == sls_id[:10] AND
  nama_usaha == name AND latitude == latitude AND longitude == longitude.

The KDM `id` column is written to output as `idkendedes`.
"""

from __future__ import annotations

import glob
import hashlib
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


# =====================================================================
# GLOBAL PARAMETERS (COLUMNS)
# =====================================================================

SOURCE_1_FOLDER = "source_matcha_pro_all"
SOURCE_2_FOLDER = "source_kdm_all"
RESULT_FOLDER = "result"
RESULT_FILENAME = "match_sbr_kdm.csv"

# If True, write output into multiple CSVs under result/split_regency/
# named match_sbr_kdm_regency_<XXXX>.csv based on kode_wilayah[:4] (regency code).
# If False, write a single combined CSV at result/match_sbr_kdm.csv.
SPLIT_OUTPUT_BY_REGENCY = True

# If True, include SBR rows with invalid coordinates in the output.
# If False, only include SBR rows with valid (non-empty) coordinates.
INCLUDE_INVALID_COORDINATES = True


# Source 1 (SBR) expected output columns
SOURCE1_OUTPUT_COLUMNS: List[str] = [
	"idsbr",
	"nama_usaha",
	"kode_wilayah",
	"latitude",
	"longitude",
	"sumber_data",
	"is_sbr_coordinate_valid",
]

# Source 2 (KDM) expected output columns
# NOTE: `id` is renamed to `idkendedes` in output
SOURCE2_OUTPUT_COLUMNS: List[str] = [
	"idkendedes",
	"name",
	"sls_id",
	"owner",
	"user_id",
	"project_id",
	"type",
]

RESULT_COLUMNS: List[str] = SOURCE1_OUTPUT_COLUMNS + SOURCE2_OUTPUT_COLUMNS


# Source 1 required columns
SOURCE1_REQUIRED_COLUMNS: List[str] = [
	"idsbr",
	"nama_usaha",
	"kode_wilayah",
	"latitude",
	"longitude",
]

# Source 2 columns needed for matching and output
SOURCE2_NEEDED_COLUMNS: List[str] = [
	"id",
	"sls_id",
	"owner",
	"user_id",
	"project_id",
	"name",
	"latitude",
	"longitude",
]


# =====================================================================
# HELPERS
# =====================================================================


def _as_str(value: Any) -> str:
	if value is None or (isinstance(value, float) and pd.isna(value)):
		return ""
	return str(value)


def normalize_name(value: Any) -> str:
	"""Normalize business name for exact-ish matching (trim, collapse spaces, lowercase)."""
	text = _as_str(value).strip()
	if not text:
		return ""
	return " ".join(text.split()).lower()


def normalize_prefix10(value: Any) -> str:
	"""Get first 10 characters as a stable string key."""
	text = _as_str(value).strip()
	return text[:10]


def coerce_coordinate(value: Any) -> Tuple[str, Optional[float], bool]:
	"""Return (fixed_text, float_value, is_valid).

	- If value is a string with comma decimals, convert to dot.
	- is_valid is True if float parsing succeeds after optional fix.
	"""
	text = _as_str(value).strip()
	if not text:
		return "", None, False

	# First try as-is
	try:
		return text, float(text), True
	except Exception:
		pass

	# Try comma->dot fix
	fixed = text.replace(",", ".")
	try:
		return fixed, float(fixed), True
	except Exception:
		return fixed, None, False


def format_coord_key(value: Optional[float], decimals: int = 6) -> str:
	if value is None or (isinstance(value, float) and pd.isna(value)):
		return ""
	return f"{float(value):.{decimals}f}"


def read_csv_only_columns(path: str, wanted: Sequence[str]) -> pd.DataFrame:
	"""Read a CSV selecting only available columns from `wanted`."""
	header = pd.read_csv(path, nrows=0, low_memory=False)
	available = [c for c in wanted if c in header.columns]
	return pd.read_csv(path, usecols=available, low_memory=False)


def _hash64(text: str) -> int:
	"""Stable 64-bit hash for memory-efficient uniqueness tracking."""
	if not text:
		return 0
	digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
	return int.from_bytes(digest, byteorder="big", signed=False)


def iter_source1_chunks(base_dir: str, chunksize: int = 250_000) -> Iterable[pd.DataFrame]:
	"""Stream SBR CSVs in chunks, selecting only needed columns.

	This avoids loading very large SBR files fully into memory.
	"""
	folder = os.path.join(base_dir, SOURCE_1_FOLDER)
	files = sorted(glob.glob(os.path.join(folder, "*.csv")))
	if not files:
		raise FileNotFoundError(f"No CSV files found in {folder}")

	wanted = list(dict.fromkeys(SOURCE1_REQUIRED_COLUMNS + ["sumber_data"]))
	for path in files:
		header = pd.read_csv(path, nrows=0, low_memory=False)
		missing = [c for c in SOURCE1_REQUIRED_COLUMNS if c not in header.columns]
		if missing:
			raise ValueError(
				f"Source 1 file {os.path.basename(path)} missing required columns: {missing}. "
				f"Available: {list(header.columns)}"
			)

		available = [c for c in wanted if c in header.columns]
		for chunk in pd.read_csv(
			path,
			usecols=available,
			chunksize=chunksize,
			low_memory=False,
			dtype=str,
		):
			chunk["__source_file"] = os.path.basename(path)
			if "sumber_data" not in chunk.columns:
				chunk["sumber_data"] = "sbr"
			yield chunk


# =====================================================================
# LOADERS
# =====================================================================


def load_source1(base_dir: str) -> pd.DataFrame:
	folder = os.path.join(base_dir, SOURCE_1_FOLDER)
	files = glob.glob(os.path.join(folder, "*.csv"))

	if not files:
		raise FileNotFoundError(f"No CSV files found in {folder}")

	dfs: List[pd.DataFrame] = []
	for path in files:
		df = pd.read_csv(path, low_memory=False)
		df["__source_file"] = os.path.basename(path)
		dfs.append(df)

	df_all = pd.concat(dfs, ignore_index=True)

	missing = [c for c in SOURCE1_REQUIRED_COLUMNS if c not in df_all.columns]
	if missing:
		raise ValueError(
			f"Source 1 missing required columns: {missing}. Available: {list(df_all.columns)}"
		)

	# Ensure these exist (per expected output)
	if "sumber_data" not in df_all.columns:
		df_all["sumber_data"] = "sbr"

	return df_all


def iter_source2_filtered(base_dir: str, prefixes10: Set[str]) -> Iterable[pd.DataFrame]:
	"""Stream KDM CSVs in chunks, filtering early by sls_id prefix."""
	folder = os.path.join(base_dir, SOURCE_2_FOLDER)
	files = sorted(glob.glob(os.path.join(folder, "*.csv")))
	if not files:
		raise FileNotFoundError(f"No CSV files found in {folder}")

	for path in files:
		filename = os.path.basename(path)
		is_supplement = filename.lower().startswith("supplement")
		business_type = "supplement" if is_supplement else "market"

		# Determine available columns so we can use usecols safely with chunks
		header = pd.read_csv(path, nrows=0, low_memory=False)
		available = [c for c in SOURCE2_NEEDED_COLUMNS if c in header.columns]

		# Stream in chunks to avoid holding the entire KDM dataset in memory
		for chunk in pd.read_csv(path, usecols=available, chunksize=250_000, low_memory=False):
			chunk["__source_file"] = filename
			chunk["type"] = business_type

			# Owner only exists in supplement files; otherwise, force empty
			if "owner" not in chunk.columns:
				chunk["owner"] = ""
			elif not is_supplement:
				chunk["owner"] = ""

			# Ensure output columns exist even if absent
			for col in ["user_id", "project_id", "sls_id", "id", "name", "latitude", "longitude"]:
				if col not in chunk.columns:
					chunk[col] = ""

			prefix = chunk["sls_id"].map(normalize_prefix10)
			chunk = chunk[prefix.isin(prefixes10)].copy()
			if chunk.empty:
				continue

			yield chunk


def build_source2_index(base_dir: str, prefixes10: Set[str]) -> pd.DataFrame:
	"""Load KDM data needed for matching, filtered by sls_id prefix set."""
	dfs: List[pd.DataFrame] = []
	for chunk in iter_source2_filtered(base_dir, prefixes10):
		dfs.append(chunk)
	if not dfs:
		return pd.DataFrame(columns=SOURCE2_NEEDED_COLUMNS + ["type"])
	return pd.concat(dfs, ignore_index=True)


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
	base_dir = os.path.dirname(os.path.abspath(__file__))
	os.makedirs(os.path.join(base_dir, RESULT_FOLDER), exist_ok=True)
	output_path = os.path.join(base_dir, RESULT_FOLDER, RESULT_FILENAME)
	split_dir = os.path.join(base_dir, RESULT_FOLDER, "split_match_sbr_kdm")
	if SPLIT_OUTPUT_BY_REGENCY:
		os.makedirs(split_dir, exist_ok=True)
	summary_xlsx_path = os.path.join(base_dir, RESULT_FOLDER, "match_sbr_kdm_summary.xlsx")
	summary_fallback_csv_path = os.path.join(base_dir, RESULT_FOLDER, "match_sbr_kdm_summary.csv")

	t0 = time.time()
	print("🔄 Scanning Source 1 (SBR) in chunks...")
	rows_total = 0
	rows_with_coords = 0
	prefixes10: Set[str] = set()
	for chunk in iter_source1_chunks(base_dir):
		rows_total += len(chunk)
		lat_raw = chunk["latitude"].map(_as_str).str.strip()
		lon_raw = chunk["longitude"].map(_as_str).str.strip()
		filtered = chunk[(lat_raw != "") & (lon_raw != "")]
		rows_with_coords += len(filtered)
		if not filtered.empty:
			prefixes10.update(filtered["kode_wilayah"].map(normalize_prefix10).unique())

	prefixes10.discard("")
	print(f"✓ Source 1 scanned: {rows_total:,} rows")
	print(f"✓ Source 1 rows with non-empty lat/lon: {rows_with_coords:,}")
	print(f"🔎 Prefix10 values from Source 1: {len(prefixes10):,}")

	print("🔄 Loading Source 2 (KDM), filtered by prefix10...")
	kdm = build_source2_index(base_dir, prefixes10)
	print(f"✓ Source 2 loaded after prefix filter: {len(kdm):,} rows")

	if not kdm.empty:
		# Normalize coordinates in KDM too
		kdm_lat_fixed = kdm["latitude"].map(coerce_coordinate)
		kdm_lon_fixed = kdm["longitude"].map(coerce_coordinate)
		kdm["latitude"] = kdm_lat_fixed.map(lambda t: t[0])
		kdm["longitude"] = kdm_lon_fixed.map(lambda t: t[0])
		kdm["__lat_float"] = kdm_lat_fixed.map(lambda t: t[1])
		kdm["__lon_float"] = kdm_lon_fixed.map(lambda t: t[1])

		kdm["__prefix10"] = kdm["sls_id"].map(normalize_prefix10)
		kdm["__name_norm"] = kdm["name"].map(normalize_name)
		kdm["__lat_key"] = kdm["__lat_float"].map(format_coord_key)
		kdm["__lon_key"] = kdm["__lon_float"].map(format_coord_key)

		# Reduce to unique keys to prevent exploding the join
		kdm = kdm.dropna(subset=["sls_id", "id"]).copy()
		kdm = kdm.drop_duplicates(subset=["__prefix10", "__name_norm", "__lat_key", "__lon_key"], keep="first")

	# Prepare KDM join frame once (if any)
	if kdm.empty:
		kdm_for_join = None
	else:
		kdm_for_join = kdm[[
			"id",
			"sls_id",
			"owner",
			"user_id",
			"project_id",
			"type",
			"__prefix10",
			"__name_norm",
			"__lat_key",
			"__lon_key",
		]].copy()

	# 3) Second pass: stream SBR again, match, and append results
	print("🔁 Matching & writing output in chunks...")
	first_write = True
	first_write_by_regency: Dict[str, bool] = {}
	total_rows_written = 0
	sbr_rows_matched = 0
	sbr_rows_not_matched = 0
	matched_source2_id_hashes: Set[int] = set()
	# How many Source-2 ids exist in the KDM slice we considered
	if kdm_for_join is None:
		kdm_unique_ids_considered = 0
		kdm_rows_considered = 0
	else:
		kdm_rows_considered = len(kdm_for_join)
		kdm_unique_ids_considered = int(
			kdm_for_join["id"].fillna("").map(_as_str).str.strip().replace("", pd.NA).nunique(dropna=True)
		)

	for chunk in iter_source1_chunks(base_dir):
		lat_raw = chunk["latitude"].map(_as_str).str.strip()
		lon_raw = chunk["longitude"].map(_as_str).str.strip()
		
		if INCLUDE_INVALID_COORDINATES:
			sbr_filtered = chunk.copy()
		else:
			sbr_filtered = chunk[(lat_raw != "") & (lon_raw != "")].copy()
		
		if sbr_filtered.empty:
			continue

		# Validate/fix coordinates (comma->dot)
		lat_fixed = sbr_filtered["latitude"].map(coerce_coordinate)
		lon_fixed = sbr_filtered["longitude"].map(coerce_coordinate)
		sbr_filtered["latitude"] = lat_fixed.map(lambda t: t[0])
		sbr_filtered["longitude"] = lon_fixed.map(lambda t: t[0])
		sbr_filtered["__lat_float"] = lat_fixed.map(lambda t: t[1])
		sbr_filtered["__lon_float"] = lon_fixed.map(lambda t: t[1])
		sbr_filtered["is_sbr_coordinate_valid"] = lat_fixed.map(lambda t: t[2]) & lon_fixed.map(lambda t: t[2])

		# Split into valid and invalid coordinate rows for optimized matching
		valid_coords_mask = sbr_filtered["is_sbr_coordinate_valid"]
		sbr_valid = sbr_filtered[valid_coords_mask].copy()
		sbr_invalid = sbr_filtered[~valid_coords_mask].copy()

		merged_parts = []

		# Process valid coordinates: perform matching
		if not sbr_valid.empty:
			# Build match keys for valid SBR rows only
			sbr_valid["__prefix10"] = sbr_valid["kode_wilayah"].map(normalize_prefix10)
			sbr_valid["__name_norm"] = sbr_valid["nama_usaha"].map(normalize_name)
			sbr_valid["__lat_key"] = sbr_valid["__lat_float"].map(format_coord_key)
			sbr_valid["__lon_key"] = sbr_valid["__lon_float"].map(format_coord_key)

			# Match: prefix10 + name + lat + lon
			if kdm_for_join is None:
				merged_valid = sbr_valid.copy()
				for col in SOURCE2_OUTPUT_COLUMNS:
					merged_valid[col] = ""
			else:
				merged_valid = sbr_valid.merge(
					kdm_for_join,
					how="left",
					left_on=["__prefix10", "__name_norm", "__lat_key", "__lon_key"],
					right_on=["__prefix10", "__name_norm", "__lat_key", "__lon_key"],
					suffixes=("", "_kdm"),
				)

				merged_valid = merged_valid.rename(columns={"id": "idkendedes"})
				for col in SOURCE2_OUTPUT_COLUMNS:
					if col not in merged_valid.columns:
						merged_valid[col] = ""
			
			merged_parts.append(merged_valid)

		# Process invalid coordinates: skip matching, just add empty KDM columns
		if not sbr_invalid.empty:
			merged_invalid = sbr_invalid.copy()
			for col in SOURCE2_OUTPUT_COLUMNS:
				merged_invalid[col] = ""
			merged_parts.append(merged_invalid)

		# Combine valid and invalid results
		if len(merged_parts) == 0:
			continue
		elif len(merged_parts) == 1:
			merged = merged_parts[0]
		else:
			merged = pd.concat(merged_parts, ignore_index=True)

		# Ensure source1 output columns exist
		if "sumber_data" not in merged.columns:
			merged["sumber_data"] = "sbr"

		# Final projection and append
		for col in RESULT_COLUMNS:
			if col not in merged.columns:
				merged[col] = ""

		result = merged[RESULT_COLUMNS].copy()
		total_rows_written += len(result)
		id_series = result["idkendedes"].fillna("").map(_as_str).str.strip()
		matched_mask = id_series != ""
		m = int(matched_mask.sum())
		sbr_rows_matched += m
		sbr_rows_not_matched += (len(result) - m)
		if m:
			# Track unique Source-2 IDs matched without storing full strings
			for v in pd.unique(id_series[matched_mask]):
				h = _hash64(str(v))
				if h:
					matched_source2_id_hashes.add(h)

		if not SPLIT_OUTPUT_BY_REGENCY:
			result.to_csv(
				output_path,
				mode="w" if first_write else "a",
				header=first_write,
				index=False,
				encoding="utf-8",
			)
			first_write = False
		else:
			regency = result["kode_wilayah"].fillna("").map(_as_str).str.strip().str[:4]
			result = result.assign(__regency=regency)
			for regency_value, part in result.groupby("__regency", dropna=False):
				reg = _as_str(regency_value).strip()
				if not reg:
					reg = "____"
				part = part.drop(columns=["__regency"])
				part_path = os.path.join(split_dir, f"match_sbr_kdm_regency_{reg}.csv")
				is_first = first_write_by_regency.get(reg, True)
				part.to_csv(
					part_path,
					mode="w" if is_first else "a",
					header=is_first,
					index=False,
					encoding="utf-8",
				)
				first_write_by_regency[reg] = False

	if not SPLIT_OUTPUT_BY_REGENCY:
		print(f"✅ Saved: {output_path}")
	else:
		print(f"✅ Saved split outputs under: {split_dir}")
	print(f"✓ Output rows written: {total_rows_written:,}")
	print(f"⏱️  Done in {time.time() - t0:.1f}s")

	# Summary: Source 2 match vs not match
	unique_source2_ids_matched = len(matched_source2_id_hashes)
	unique_source2_ids_not_matched = max(kdm_unique_ids_considered - unique_source2_ids_matched, 0)

	# Source 2 matched/not matched grouped by sls_id[:4] (unique IDs)
	if kdm_for_join is None:
		source2_by_sls4_df = pd.DataFrame(
			columns=["sls4", "source2_unique_ids_total", "source2_unique_ids_matched", "source2_unique_ids_not_matched"]
		)
	else:
		kdm_id_map = kdm[["id", "sls_id"]].copy()
		kdm_id_map["id"] = kdm_id_map["id"].fillna("").map(_as_str).str.strip()
		kdm_id_map["sls_id"] = kdm_id_map["sls_id"].fillna("").map(_as_str).str.strip()
		kdm_id_map = kdm_id_map[(kdm_id_map["id"] != "") & (kdm_id_map["sls_id"] != "")].copy()
		kdm_id_map["sls4"] = kdm_id_map["sls_id"].str[:4]
		kdm_id_map = kdm_id_map[kdm_id_map["sls4"] != ""].copy()
		kdm_id_map = kdm_id_map.drop_duplicates(subset=["id"], keep="first")
		kdm_id_map["__id_hash"] = kdm_id_map["id"].map(lambda v: _hash64(str(v)))
		kdm_id_map["is_matched"] = kdm_id_map["__id_hash"].isin(matched_source2_id_hashes)

		grp = kdm_id_map.groupby("sls4", as_index=False).agg(
			source2_unique_ids_total=("id", "count"),
			source2_unique_ids_matched=("is_matched", "sum"),
		)
		grp["source2_unique_ids_not_matched"] = (
			grp["source2_unique_ids_total"] - grp["source2_unique_ids_matched"]
		)
		source2_by_sls4_df = grp.sort_values("sls4").reset_index(drop=True)

	print("\nMatch Summary")
	print("=============")
	print(f"SBR rows written (non-empty coords): {total_rows_written:,}")
	print(f"SBR matched to Source 2: {sbr_rows_matched:,}")
	print(f"SBR not matched to Source 2: {sbr_rows_not_matched:,}")
	print(f"Source 2 rows considered (after prefix filter & dedupe): {kdm_rows_considered:,}")
	print(f"Source 2 unique IDs considered: {kdm_unique_ids_considered:,}")
	print(f"Source 2 unique IDs matched: {unique_source2_ids_matched:,}")
	print(f"Source 2 unique IDs not matched: {unique_source2_ids_not_matched:,}")

	summary_df = pd.DataFrame(
		[
			{"metric": "split_output_by_regency", "value": bool(SPLIT_OUTPUT_BY_REGENCY)},
			{"metric": "output_csv", "value": output_path if not SPLIT_OUTPUT_BY_REGENCY else ""},
			{"metric": "output_split_dir", "value": split_dir if SPLIT_OUTPUT_BY_REGENCY else ""},
			{"metric": "sbr_rows_written_non_empty_coords", "value": total_rows_written},
			{"metric": "sbr_rows_matched", "value": sbr_rows_matched},
			{"metric": "sbr_rows_not_matched", "value": sbr_rows_not_matched},
			{"metric": "kdm_rows_considered", "value": kdm_rows_considered},
			{"metric": "kdm_unique_ids_considered", "value": kdm_unique_ids_considered},
			{"metric": "kdm_unique_ids_matched", "value": unique_source2_ids_matched},
			{"metric": "kdm_unique_ids_not_matched", "value": unique_source2_ids_not_matched},
		]
	)

	# Save summary to Excel (fallback to CSV if Excel engine isn't available)
	try:
		with pd.ExcelWriter(summary_xlsx_path, engine="openpyxl") as writer:
			summary_df.to_excel(writer, index=False, sheet_name="summary")
			source2_by_sls4_df.to_excel(writer, index=False, sheet_name="source2_by_sls4")
		print(f"✅ Saved summary Excel: {summary_xlsx_path}")
	except Exception as e:
		summary_df.to_csv(summary_fallback_csv_path, index=False, encoding="utf-8")
		print(f"⚠️  Could not write Excel summary ({e}). Saved CSV instead: {summary_fallback_csv_path}")


if __name__ == "__main__":
	main()

