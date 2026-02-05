"""Extract RT/RW from alamat_usaha.

What it does:
1) Loads `source_identify_sls/source.csv`
2) Extracts RT and RW numbers from `alamat_usaha` (supports leading zeros like 005 -> 5)
3) Writes a new file with two additional columns: RT, RW

Run:
  python identify_sls.py
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd


try:
	import requests  # type: ignore
except Exception:  # pragma: no cover
	requests = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CSV = os.path.join(BASE_DIR, "source_identify_sls", "source.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "source_identify_sls", "source_with_rt_rw.csv")

GEOCODED_OUTPUT_CSV = os.path.join(
	BASE_DIR, "source_identify_sls", "source_with_rt_rw_geocoded.csv"
)
CACHE_PATH = os.path.join(BASE_DIR, "source_identify_sls", "geocode_cache.json")
ENV_PATH = os.path.join(BASE_DIR, ".env")

# ----------------------------
# Geocoding configuration
# ----------------------------
# Set these globals for testing/production.
ENABLE_GEOCODING = True

# Debug/testing: limit geocoding to first N eligible rows (0 = no limit).
DEBUG_FIRST_N = 0

# Geocoding request limits
GEOCODE_MAX_REQUESTS = 2000
GEOCODE_SLEEP_SECONDS = 0.1

# If True, ignore cache hits and always call the Google API
# for eligible rows (still writes results to cache).
FORCE_API_CALL = True

# If False, don't write extra metadata columns like geocode_place_id.
INCLUDE_GEOCODE_METADATA = False

# If True, keep a `geocode_status` column in the output CSV.
INCLUDE_GEOCODE_STATUS_COLUMN = False

# Output flag: 1 when latitude/longitude were filled by geocoding, else 0.
IS_GEOCODE_COLUMN = "is_geocode"

# Logging
LOG_GEOCODE_RESULTS = True
LOG_GEOCODE_QUERY = False
LOG_QUERY_MAX_CHARS = 140

# ----------------------------
# Input filtering
# ----------------------------
# Only keep rows where the source column contains one of these keywords.
SOURCE_FILTER_KEYWORDS = ("Pajak", "OSS")
# User said the column name is 'sumber'. If it doesn't exist, we'll fall back to 'sumber_data'.
SOURCE_FILTER_COLUMN_PRIMARY = "sumber"
SOURCE_FILTER_COLUMN_FALLBACK = "sumber_data"


def _parse_dotenv(path: str) -> dict[str, str]:
	if not os.path.isfile(path):
		return {}

	out: dict[str, str] = {}
	with open(path, "r", encoding="utf-8") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line or line.startswith("#"):
				continue
			if "=" not in line:
				continue
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip().strip('"').strip("'")
			if key:
				out[key] = value
	return out


def _get_google_api_key() -> str:
	# 1) environment variables
	for key in ("google_api_key", "GOOGLE_API_KEY"):
		val = os.environ.get(key)
		if val:
			return val.strip()

	# 2) .env file (if present)
	env = _parse_dotenv(ENV_PATH)
	for key in ("google_api_key", "GOOGLE_API_KEY"):
		val = env.get(key)
		if val:
			return val.strip()

	return ""


@dataclass(frozen=True)
class RtRwExtractor:
	source_csv: str
	output_csv: str
	alamat_column: str = "alamat_usaha"
	rt_column: str = "RT"
	rw_column: str = "RW"

	# Matches: RT 1, RT01, RT.003, RT/07, RT: 5 (case-insensitive)
	_rt_re: re.Pattern[str] = re.compile(r"(?i)\bRT\s*[:./\-]?\s*(\d{1,3})\b")
	_rw_re: re.Pattern[str] = re.compile(r"(?i)\bRW\s*[:./\-]?\s*(\d{1,3})\b")
	# Matches combined pattern: RT 01/08, Rt.01/08, RT. 003/005
	_rt_rw_slash_re: re.Pattern[str] = re.compile(
		r"(?i)\bRT\s*[:.\-]?\s*(\d{1,3})\s*/\s*(\d{1,3})\b"
	)

	def _read_csv(self) -> pd.DataFrame:
		if not os.path.isfile(self.source_csv):
			raise FileNotFoundError(f"CSV not found: {self.source_csv}")

		try:
			return pd.read_csv(
				self.source_csv,
				dtype=str,
				keep_default_na=False,
				low_memory=False,
				encoding="utf-8",
				engine="c",
			)
		except UnicodeDecodeError:
			return pd.read_csv(
				self.source_csv,
				dtype=str,
				keep_default_na=False,
				low_memory=False,
				encoding="utf-8-sig",
				engine="python",
			)

	def _extract_first_number(self, text: str, pattern: re.Pattern[str]) -> str:
		if not text:
			return ""
		match = pattern.search(text)
		if not match:
			return ""
		num_str = match.group(1)
		# Convert to int to remove leading zeros (e.g. "005" -> "5")
		try:
			return str(int(num_str))
		except ValueError:
			return ""

	def _extract_rt_rw_slash(self, text: str) -> tuple[str, str]:
		if not text:
			return "", ""
		match = self._rt_rw_slash_re.search(text)
		if not match:
			return "", ""
		rt_raw, rw_raw = match.group(1), match.group(2)
		try:
			return str(int(rt_raw)), str(int(rw_raw))
		except ValueError:
			return "", ""

	def add_rt_rw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
		if self.alamat_column not in df.columns:
			raise KeyError(f"Missing column '{self.alamat_column}' in {self.source_csv}")

		alamat = df[self.alamat_column].fillna("").astype(str)
		out = df.copy()

		rt_values = alamat.apply(lambda s: self._extract_first_number(s, self._rt_re))
		rw_values = alamat.apply(lambda s: self._extract_first_number(s, self._rw_re))

		# If RW is not present as a token, try the combined RT xx/yy pattern.
		rt_rw_slash = alamat.apply(self._extract_rt_rw_slash)
		rt_from_slash = rt_rw_slash.apply(lambda t: t[0])
		rw_from_slash = rt_rw_slash.apply(lambda t: t[1])
		need_slash = (rw_values == "") & (rw_from_slash != "")
		rt_values = rt_values.where(~need_slash, rt_from_slash)
		rw_values = rw_values.where(~need_slash, rw_from_slash)

		# Put RT and RW right next to alamat_usaha.
		insert_at = int(out.columns.get_loc(self.alamat_column)) + 1
		if self.rt_column in out.columns:
			out.drop(columns=[self.rt_column], inplace=True)
		if self.rw_column in out.columns:
			out.drop(columns=[self.rw_column], inplace=True)
		out.insert(insert_at, self.rt_column, rt_values)
		out.insert(insert_at + 1, self.rw_column, rw_values)
		return out

	def run(self) -> tuple[str, int]:
		df = self._read_csv()

		# Filter before any extraction/geocoding.
		source_col = None
		if SOURCE_FILTER_COLUMN_PRIMARY in df.columns:
			source_col = SOURCE_FILTER_COLUMN_PRIMARY
		elif SOURCE_FILTER_COLUMN_FALLBACK in df.columns:
			source_col = SOURCE_FILTER_COLUMN_FALLBACK
		else:
			raise KeyError(
				"Missing source column for filtering. "
				f"Expected '{SOURCE_FILTER_COLUMN_PRIMARY}' (or fallback '{SOURCE_FILTER_COLUMN_FALLBACK}')."
			)

		pattern = "|".join(re.escape(k) for k in SOURCE_FILTER_KEYWORDS)
		mask = df[source_col].fillna("").astype(str).str.contains(pattern, case=False, regex=True)
		filtered = df.loc[mask].copy()
		print(
			f"Filter by {source_col} contains {SOURCE_FILTER_KEYWORDS}: "
			f"{int(filtered.shape[0])}/{int(df.shape[0])} rows kept"
		)

		df = filtered
		out = self.add_rt_rw_columns(df)
		os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
		out.to_csv(self.output_csv, index=False, encoding="utf-8")
		return self.output_csv, int(out.shape[0])


@dataclass
class GeocodeResult:
	status: str
	lat: str
	lng: str
	formatted_address: str = ""
	place_id: str = ""

	def to_dict(self) -> dict[str, str]:
		return {
			"status": self.status,
			"lat": self.lat,
			"lng": self.lng,
			"formatted_address": self.formatted_address,
			"place_id": self.place_id,
		}


@dataclass
class MissingRtRwGeocoder:
	input_csv: str
	output_csv: str
	cache_path: str
	alamat_col: str = "alamat_usaha"
	rt_col: str = "RT"
	rw_col: str = "RW"
	# We write results directly into these existing columns.
	lat_col: str = "latitude"
	lng_col: str = "longitude"
	geocode_status_col: str = "geocode_status"
	geocode_query_col: str = "geocode_query"
	formatted_address_col: str = "geocode_formatted_address"
	place_id_col: str = "geocode_place_id"

	def _load_cache(self) -> dict[str, dict[str, str]]:
		if not os.path.isfile(self.cache_path):
			return {}
		try:
			with open(self.cache_path, "r", encoding="utf-8") as f:
				obj = json.load(f)
				if isinstance(obj, dict):
					return obj  # type: ignore[return-value]
		except Exception:
			return {}
		return {}

	def _save_cache(self, cache: dict[str, dict[str, str]]) -> None:
		os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
		with open(self.cache_path, "w", encoding="utf-8") as f:
			json.dump(cache, f, ensure_ascii=False, indent=2)

	def _build_query(self, row: pd.Series) -> str:
		alamat = str(row.get(self.alamat_col, "") or "").strip()
		parts = [alamat]
		for col in ("nmdesa", "nmkec", "nmkab", "nmprov"):
			val = str(row.get(col, "") or "").strip()
			if val:
				parts.append(val)
		parts.append("Indonesia")

		unique_parts: list[str] = []
		seen = set()
		for p in parts:
			p2 = " ".join(p.split())
			if not p2:
				continue
			key = p2.lower()
			if key in seen:
				continue
			seen.add(key)
			unique_parts.append(p2)
		return ", ".join(unique_parts)

	def _geocode(self, session: Any, api_key: str, query: str) -> GeocodeResult:
		url = "https://maps.googleapis.com/maps/api/geocode/json"
		params = {
			"address": query,
			"key": api_key,
			"language": "id",
			"region": "id",
		}
		try:
			resp = session.get(url, params=params, timeout=30)
			data = resp.json()
		except Exception as e:
			return GeocodeResult(status=f"ERROR: {type(e).__name__}", lat="", lng="")

		status = str(data.get("status", ""))
		if status != "OK":
			return GeocodeResult(status=status, lat="", lng="")

		results = data.get("results") or []
		if not results:
			return GeocodeResult(status="NO_RESULTS", lat="", lng="")

		best = results[0]
		loc = (best.get("geometry") or {}).get("location") or {}
		lat = loc.get("lat")
		lng = loc.get("lng")
		return GeocodeResult(
			status="OK",
			lat=str(lat) if lat is not None else "",
			lng=str(lng) if lng is not None else "",
			formatted_address=str(best.get("formatted_address") or ""),
			place_id=str(best.get("place_id") or ""),
		)

	def run(self) -> dict[str, int]:
		if requests is None:
			raise RuntimeError("Missing dependency: 'requests'. Install it with: pip install requests")

		api_key = _get_google_api_key()
		if not api_key:
			raise RuntimeError(
				"Google API key not found. Set 'google_api_key' in environment variables or create a .env file."
			)

		if not os.path.isfile(self.input_csv):
			raise FileNotFoundError(f"Input CSV not found: {self.input_csv}")

		max_requests = int(GEOCODE_MAX_REQUESTS)
		sleep_seconds = float(GEOCODE_SLEEP_SECONDS)

		df = pd.read_csv(self.input_csv, dtype=str, keep_default_na=False, low_memory=False)
		# Ensure base columns exist.
		for col in (self.lat_col, self.lng_col, IS_GEOCODE_COLUMN):
			if col not in df.columns:
				df[col] = ""
		# Default is_geocode to 0 for all rows.
		df[IS_GEOCODE_COLUMN] = df[IS_GEOCODE_COLUMN].replace("", "0")
		if INCLUDE_GEOCODE_STATUS_COLUMN and self.geocode_status_col not in df.columns:
			df[self.geocode_status_col] = ""
		if INCLUDE_GEOCODE_METADATA:
			for col in (self.geocode_query_col, self.formatted_address_col, self.place_id_col):
				if col not in df.columns:
					df[col] = ""

		need_geocode = (df[self.rt_col] == "") & (df[self.rw_col] == "")
		# Don't call the API if lat/long are already identified.
		# Only geocode when either latitude or longitude is missing.
		need_geocode = need_geocode & ((df[self.lat_col] == "") | (df[self.lng_col] == ""))
		remaining = int(df.loc[need_geocode].shape[0])

		cache = self._load_cache()
		requests_made = 0
		cache_hits = 0
		ok_results = 0
		non_ok_results = 0

		geocode_indices = list(df.index[need_geocode])
		if DEBUG_FIRST_N > 0:
			geocode_indices = geocode_indices[:DEBUG_FIRST_N]

		with requests.Session() as session:
			for idx in geocode_indices:
				if requests_made >= max_requests:
					break

				# Skip if already geocoded (only when we keep status column)
				if INCLUDE_GEOCODE_STATUS_COLUMN and df.at[idx, self.geocode_status_col]:
					continue

				query = self._build_query(df.loc[idx])
				if INCLUDE_GEOCODE_METADATA:
					df.at[idx, self.geocode_query_col] = query

				cached = None if FORCE_API_CALL else cache.get(query)
				if cached and cached.get("status") == "OK" and cached.get("lat") and cached.get("lng"):
					cache_hits += 1
					ok_results += 1
					if INCLUDE_GEOCODE_STATUS_COLUMN:
						df.at[idx, self.geocode_status_col] = cached.get("status", "")
					df.at[idx, self.lat_col] = cached.get("lat", "")
					df.at[idx, self.lng_col] = cached.get("lng", "")
					df.at[idx, IS_GEOCODE_COLUMN] = "1"
					if INCLUDE_GEOCODE_METADATA:
						df.at[idx, self.formatted_address_col] = cached.get("formatted_address", "")
						df.at[idx, self.place_id_col] = cached.get("place_id", "")
					if LOG_GEOCODE_RESULTS:
						idsbr = str(df.at[idx, "idsbr"]) if "idsbr" in df.columns else ""
						alamat = str(df.at[idx, self.alamat_col]) if self.alamat_col in df.columns else ""
						msg = (
							f"[GEOCODE][CACHE] idx={idx} idsbr={idsbr} status=OK "
							f"lat={df.at[idx, self.lat_col]} lng={df.at[idx, self.lng_col]}"
						)
						if LOG_GEOCODE_QUERY:
							msg += f" query={query[:LOG_QUERY_MAX_CHARS]}"
						print(msg)
						if alamat:
							print(f"  alamat: {alamat[:200]}")
					continue

				result = self._geocode(session=session, api_key=api_key, query=query)
				requests_made += 1
				if result.status == "OK" and result.lat and result.lng:
					ok_results += 1
				else:
					non_ok_results += 1

				if INCLUDE_GEOCODE_STATUS_COLUMN:
					df.at[idx, self.geocode_status_col] = result.status
				if result.status == "OK" and result.lat and result.lng:
					df.at[idx, self.lat_col] = result.lat
					df.at[idx, self.lng_col] = result.lng
					df.at[idx, IS_GEOCODE_COLUMN] = "1"
				if INCLUDE_GEOCODE_METADATA:
					df.at[idx, self.formatted_address_col] = result.formatted_address
					df.at[idx, self.place_id_col] = result.place_id

				if LOG_GEOCODE_RESULTS:
					idsbr = str(df.at[idx, "idsbr"]) if "idsbr" in df.columns else ""
					alamat = str(df.at[idx, self.alamat_col]) if self.alamat_col in df.columns else ""
					msg = (
						f"[GEOCODE][API] idx={idx} idsbr={idsbr} status={result.status} "
						f"lat={result.lat} lng={result.lng}"
					)
					if LOG_GEOCODE_QUERY:
						msg += f" query={query[:LOG_QUERY_MAX_CHARS]}"
					print(msg)
					if alamat:
						print(f"  alamat: {alamat[:200]}")

				cache[query] = result.to_dict()
				if sleep_seconds > 0:
					time.sleep(sleep_seconds)

				if requests_made % 25 == 0:
					self._save_cache(cache)

		self._save_cache(cache)
		if LOG_GEOCODE_RESULTS:
			print(
				"[GEOCODE][SUMMARY] "
				f"ok={ok_results} non_ok={non_ok_results} "
				f"api_calls={requests_made} cache_hits={cache_hits}"
			)

		# Keep output focused unless metadata/status is requested.
		if not INCLUDE_GEOCODE_METADATA:
			for c in (self.geocode_query_col, self.formatted_address_col, self.place_id_col):
				if c in df.columns:
					df.drop(columns=[c], inplace=True)
		if not INCLUDE_GEOCODE_STATUS_COLUMN and self.geocode_status_col in df.columns:
			df.drop(columns=[self.geocode_status_col], inplace=True)

		# Reorder: put latitude, longitude, is_geocode right after RT and RW.
		if self.rw_col in df.columns:
			insert_at = int(df.columns.get_loc(self.rw_col)) + 1
			for col in (self.lat_col, self.lng_col, IS_GEOCODE_COLUMN):
				if col in df.columns:
					series = df.pop(col)
					df.insert(insert_at, col, series)
					insert_at += 1

		os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
		df.to_csv(self.output_csv, index=False, encoding="utf-8")

		return {
			"rows_total": int(df.shape[0]),
			"rows_need_geocode": int(remaining),
			"requests_made": int(requests_made),
			"cache_hits": int(cache_hits),
			"max_requests": int(max_requests),
			"ok_results": int(ok_results),
			"non_ok_results": int(non_ok_results),
		}


def main() -> int:
	extractor = RtRwExtractor(source_csv=SOURCE_CSV, output_csv=OUTPUT_CSV)
	output_path, rows = extractor.run()
	print(f"Rows processed: {rows}")
	print(f"Wrote: {output_path}")

	# Optional: geocode rows where RT and RW are still empty.
	if ENABLE_GEOCODING:
		geocoder = MissingRtRwGeocoder(
			input_csv=OUTPUT_CSV,
			output_csv=GEOCODED_OUTPUT_CSV,
			cache_path=CACHE_PATH,
		)
		stats = geocoder.run()
		print(
			"Geocode done: "
			f"requests={stats['requests_made']}/{stats['max_requests']}, "
			f"cache_hits={stats['cache_hits']}, "
			f"rows_need={stats['rows_need_geocode']}"
		)
		print(f"Wrote: {geocoder.output_csv}")
		print(f"Cache: {geocoder.cache_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
