"""Deteksi duplikasi nama pemilik UMKM terhadap nama usaha OSS di desa yang sama."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


DATA_PATH = Path("source_matcha_pro_all") / "combined_data.csv"
OUTPUT_PATH = Path("result") / "duplicate_oss_umkm.csv"


OWNER_PATTERN = re.compile(
	r"<\s*([^<>]+?)\s*>|\(\s*([^()]+?)\s*\)|\{\s*([^{}]+?)\s*\}"
)


def extract_owner(nama_usaha: str | float | None) -> str | None:
	if not isinstance(nama_usaha, str):
		return None
	match = OWNER_PATTERN.search(nama_usaha)
	if not match:
		return None
	owner = next((grp for grp in match.groups() if grp), "")
	owner = owner.strip()
	return owner or None


def normalize_text(value: str | float | None) -> str | None:
	if not isinstance(value, str):
		return None
	cleaned = re.sub(r"\s+", " ", value).strip().lower()
	return cleaned or None


def text_similarity(left: str | float | None, right: str | float | None) -> float:
	left_norm = normalize_text(left)
	right_norm = normalize_text(right)
	if not left_norm or not right_norm:
		return 0.0
	return SequenceMatcher(None, left_norm, right_norm).ratio()


def main() -> None:
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Data tidak ditemukan: {DATA_PATH}")

	data = pd.read_csv(DATA_PATH, dtype=str)

	data["sumber_data_clean"] = data["sumber_data"].fillna("").str.strip()
	mask_umkm = data["sumber_data_clean"] == "PL-KUMKM 2023"
	mask_oss = data["sumber_data_clean"].str.contains("OSS", na=False)
	mask_blank = data["sumber_data_clean"] == ""

	umkm = data[mask_umkm].copy()
	umkm["owner"] = umkm["nama_usaha"].apply(extract_owner)
	umkm["owner_norm"] = umkm["owner"].apply(normalize_text)
	umkm = umkm[umkm["owner_norm"].notna()].copy()

	blank = data[mask_blank].copy()
	blank["owner"] = blank["nama_usaha"].apply(extract_owner)
	blank["owner_norm"] = blank["owner"].apply(normalize_text)
	blank = blank[blank["owner_norm"].notna()].copy()

	oss = data[mask_oss].copy()
	oss["nama_usaha_norm"] = oss["nama_usaha"].apply(normalize_text)
	oss["owner"] = None
	oss = oss[oss["nama_usaha_norm"].notna()].copy()

	def build_pair(
		left_df: pd.DataFrame,
		right_df: pd.DataFrame,
		left_key: str,
		right_key: str,
		pair_type: str,
	) -> pd.DataFrame:
		merged_pair = left_df.merge(
			right_df,
			left_on=["kode_wilayah", left_key],
			right_on=["kode_wilayah", right_key],
			suffixes=("_a", "_b"),
			how="inner",
		)
		merged_pair["pair_type"] = pair_type
		return merged_pair

	merged = pd.concat(
		[
			build_pair(umkm, oss, "owner_norm", "nama_usaha_norm", "UMKM_vs_OSS"),
			build_pair(blank, oss, "owner_norm", "nama_usaha_norm", "BLANK_vs_OSS"),
			build_pair(umkm, blank, "owner_norm", "owner_norm", "UMKM_vs_BLANK"),
		],
		ignore_index=True,
	)

	merged["alamat_similarity"] = merged.apply(
		lambda row: text_similarity(row.get("alamat_usaha_a"), row.get("alamat_usaha_b")),
		axis=1,
	)

	columns = [
		"pair_type",
		"kode_wilayah",
		"idsbr_a",
		"idsbr_b",
		"nama_usaha_a",
		"owner_a",
		"alamat_usaha_a",
		"sumber_data_a",
		"nama_usaha_b",
		"owner_b",
		"alamat_usaha_b",
		"sumber_data_b",
		"alamat_similarity",
	]
	output = merged.reindex(columns=columns)

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	output.to_csv(OUTPUT_PATH, index=False)

	print(f"Data duplikasi tersimpan: {OUTPUT_PATH} (jumlah: {len(output)})")


if __name__ == "__main__":
	main()