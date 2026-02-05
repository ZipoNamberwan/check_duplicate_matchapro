"""Extract unique words from KDM business names.

What it does:
1) Loads all CSV files in the `source_kdm_all/` folder
2) Splits all words from the `name` column
3) Saves unique words to `result/unique_business_name_kdm.csv`

Run:
  python get_unique_word_business_name_kdm.py
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "source_kdm_all")
OUTPUT_PATH = os.path.join(BASE_DIR, "result", "unique_business_name_kdm.csv")


@dataclass(frozen=True)
class UniqueBusinessNameWordExtractor:
	source_dir: str
	output_path: str
	name_column: str = "name"
	output_column: str = "word"
	word_pattern: str = r"[0-9A-Za-z]+"
	market_filename_prefix: str = "market"

	def _iter_words(self, text: str) -> list[str]:
		if not text:
			return []
		word_re = re.compile(self.word_pattern, re.UNICODE)
		return word_re.findall(text.lower())

	def _strip_bracketed_text(self, text: str) -> str:
		if not text:
			return text
		# Remove any text inside <>, (), {} (including the brackets themselves).
		text = re.sub(r"<[^>]*>", " ", text)
		text = re.sub(r"\([^)]*\)", " ", text)
		text = re.sub(r"\{[^}]*\}", " ", text)
		return " ".join(text.split())

	def _read_name_column(self, csv_path: str) -> pd.Series | None:
		# Only load the 'name' column (when present) to keep it fast/memory-light.
		try:
			df = pd.read_csv(
				csv_path,
				usecols=[self.name_column],
				dtype={self.name_column: "string"},
				encoding="utf-8",
				engine="c",
			)
		except UnicodeDecodeError:
			df = pd.read_csv(
				csv_path,
				usecols=[self.name_column],
				dtype={self.name_column: "string"},
				encoding="utf-8-sig",
				engine="python",
			)
		except ValueError:
			# Missing column in this file.
			return None
		return df[self.name_column]

	def extract_unique_words(self) -> tuple[set[str], dict[str, int]]:
		if not os.path.isdir(self.source_dir):
			raise FileNotFoundError(f"Folder not found: {self.source_dir}")

		csv_paths = sorted(glob.glob(os.path.join(self.source_dir, "*.csv")))
		if not csv_paths:
			raise FileNotFoundError(f"No CSV files found in: {self.source_dir}")

		unique_words: set[str] = set()
		processed_files = 0
		processed_rows = 0
		total_csvs = len(csv_paths)

		for csv_path in csv_paths:
			is_market_file = os.path.basename(csv_path).lower().startswith(self.market_filename_prefix)
			name_series = self._read_name_column(csv_path)
			if name_series is None:
				continue

			processed_files += 1
			processed_rows += int(name_series.shape[0])

			for name in name_series.fillna("").astype(str):
				if is_market_file:
					name = self._strip_bracketed_text(name)
				unique_words.update(self._iter_words(name))

		stats = {
			"processed_files": processed_files,
			"total_csvs": total_csvs,
			"processed_rows": processed_rows,
			"unique_words": len([w for w in unique_words if w]),
		}
		return unique_words, stats

	def write_unique_words_csv(self, unique_words: set[str]) -> None:
		output_dir = os.path.dirname(self.output_path)
		os.makedirs(output_dir, exist_ok=True)
		out_df = pd.DataFrame({self.output_column: sorted(w for w in unique_words if w)})
		out_df.to_csv(self.output_path, index=False, encoding="utf-8")

	def run(self) -> dict[str, int]:
		unique_words, stats = self.extract_unique_words()
		self.write_unique_words_csv(unique_words)
		return stats


def main() -> int:
	extractor = UniqueBusinessNameWordExtractor(source_dir=SOURCE_DIR, output_path=OUTPUT_PATH)
	stats = extractor.run()
	print(f"Read files: {stats['processed_files']} (of {stats['total_csvs']} total CSVs)")
	print(f"Rows scanned: {stats['processed_rows']}")
	print(f"Unique words: {stats['unique_words']}")
	print(f"Wrote: {extractor.output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
