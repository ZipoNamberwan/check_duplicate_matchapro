
from __future__ import annotations

from pathlib import Path
import time

import pandas as pd


DEFAULT_INPUT = Path("source_matcha_pro_all") / "combined_data.csv"
DEFAULT_OUTPUT = Path("result") / "combined_all_filtered.csv"
DEFAULT_ALLOWED = ["A", "O", "T"]
DEFAULT_KBLI_PREFIXES: list[str] = ['01', '02', '03', '87', '88', '92', '96999', '9412', '942', '949']
DEFAULT_CHUNKSIZE = 250_000
DEFAULT_PROGRESS_EVERY = 1


def read_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"CSV not found: {path}")
	return pd.read_csv(path, dtype=str, low_memory=False)


def iter_csv_batches(path: Path, *, chunksize: int) -> pd.io.parsers.TextFileReader:
	if not path.exists():
		raise FileNotFoundError(f"CSV not found: {path}")
	return pd.read_csv(path, dtype=str, chunksize=chunksize, low_memory=False)


def extract_kategori_from_kegiatan_usaha(series: pd.Series) -> pd.Series:
	# Example blob includes: "Kategori: Q" -> we want "Q".
	# Be tolerant to whitespace and casing.
	pattern = r"Kategori\s*:\s*([A-Za-z])\b"
	extracted = series.fillna("").astype(str).str.extract(pattern, expand=False)
	return extracted.astype("string").str.upper()


def extract_kbli_from_kegiatan_usaha(series: pd.Series) -> pd.Series:
	# Example blob includes: "KBLI: 56102" -> we want "56102".
	pattern = r"KBLI\s*:\s*([0-9]+)\b"
	extracted = series.fillna("").astype(str).str.extract(pattern, expand=False)
	return extracted.astype("string").str.strip()


def process_in_batches(
	input_path: Path,
	output_path: Path,
	*,
	allowed: list[str],
	kbli_prefixes: list[str],
	chunksize: int,
) -> tuple[int, int]:
	t0 = time.perf_counter()
	allowed_set = {str(x).strip().upper() for x in allowed if str(x).strip()}
	kbli_prefix_tuple = tuple(str(x).strip() for x in kbli_prefixes if str(x).strip())
	output_path.parent.mkdir(parents=True, exist_ok=True)

	total_in = 0
	total_out = 0
	wrote_header = False
	columns_with_kategori: list[str] | None = None

	print(f"reading={input_path}")
	print(f"writing={output_path}")
	print(f"chunksize={chunksize}")
	print(f"allowed={sorted(allowed_set)}")
	print(f"kbli_prefixes={list(kbli_prefix_tuple)}")

	for chunk_idx, chunk in enumerate(iter_csv_batches(input_path, chunksize=chunksize), start=1):
		total_in += len(chunk)
		if "kegiatan_usaha" not in chunk.columns:
			raise KeyError("Column not found: kegiatan_usaha")

		chunk["kategori"] = extract_kategori_from_kegiatan_usaha(chunk["kegiatan_usaha"])
		chunk["kbli"] = extract_kbli_from_kegiatan_usaha(chunk["kegiatan_usaha"])
		if columns_with_kategori is None:
			columns_with_kategori = list(chunk.columns)

		kategori_is_allowed = chunk["kategori"].isin(allowed_set)
		kategori_blank_or_dash = chunk["kategori"].isna() | chunk["kategori"].astype("string").str.strip().isin(["", "-"])
		kbli_matches_prefix = (
			chunk["kbli"].fillna("").astype(str).str.startswith(kbli_prefix_tuple)
			if kbli_prefix_tuple
			else pd.Series(False, index=chunk.index)
		)
		filtered = chunk[kategori_is_allowed | (kategori_blank_or_dash & kbli_matches_prefix)].copy()
		chunk_out = len(filtered)
		if len(filtered) == 0:
			if chunk_idx % DEFAULT_PROGRESS_EVERY == 0:
				elapsed = time.perf_counter() - t0
				print(
					f"chunk={chunk_idx} read={len(chunk)} kept=0 total_in={total_in} total_out={total_out} elapsed_s={elapsed:.1f}",
					flush=True,
				)
			continue

		mode = "w" if not wrote_header else "a"
		filtered.to_csv(output_path, index=False, mode=mode, header=not wrote_header)
		wrote_header = True
		total_out += chunk_out

		if chunk_idx % DEFAULT_PROGRESS_EVERY == 0:
			elapsed = time.perf_counter() - t0
			print(
				f"chunk={chunk_idx} read={len(chunk)} kept={chunk_out} total_in={total_in} total_out={total_out} elapsed_s={elapsed:.1f}",
				flush=True,
			)

	if not wrote_header:
		# Ensure the output file exists even if no rows match.
		empty_cols = columns_with_kategori or ["kategori"]
		pd.DataFrame(columns=empty_cols).to_csv(output_path, index=False)

	elapsed = time.perf_counter() - t0
	print(f"done total_in={total_in} total_out={total_out} elapsed_s={elapsed:.1f}")
	return total_in, total_out


def main() -> None:
	total_in, total_out = process_in_batches(
		DEFAULT_INPUT,
		DEFAULT_OUTPUT,
		allowed=DEFAULT_ALLOWED,
		kbli_prefixes=DEFAULT_KBLI_PREFIXES,
		chunksize=DEFAULT_CHUNKSIZE,
	)
	print(f"input_rows={total_in}")
	print(f"output_rows={total_out}")
	print(f"saved={DEFAULT_OUTPUT}")


if __name__ == "__main__":
	main()
