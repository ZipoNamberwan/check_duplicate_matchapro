"""Match business coordinates to SLS polygons (GeoJSON) and enrich with SLS name.

Inputs:
  - source_identify_sls/source_with_rt_rw_geocoded.csv
  - source_geojson/ (folder containing .geojson/.json polygon files)
	- source_identify_sls/master.xlsx (must contain columns: idsls, nmsls)

Output:
  - source_identify_sls/source_with_sls.csv

What it does:
1) For each row with valid latitude/longitude, finds the GeoJSON polygon that contains the point
2) Uses the matched GeoJSON filename (stem) as idsls
3) Looks up idsls in master.xlsx to get nmsls
	- If master.xlsx also contains RT/RW columns, lookup prefers (idsls, RT, RW)
	- Otherwise it falls back to idsls-only

Additional master matching (requested):
- If the input row has `kode_wilayah`, `RT`, and `RW`, we will try to match master by:
	- Filtering master candidates where `idsls` first 10 chars match `kode_wilayah` first 10 chars
  - Searching by name `RT 001 RW 001` (RT/RW are zero-padded to 3 digits)
- If found, this master match overrides the polygon-based idsls.

Run:
  python a2_coordinate_to_sls.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV = os.path.join(BASE_DIR, "source_identify_sls", "source_with_rt_rw_geocoded.csv")
MASTER_XLSX = os.path.join(BASE_DIR, "source_identify_sls", "master.xlsx")
GEOJSON_DIR = os.path.join(BASE_DIR, "source_geojson")

OUTPUT_CSV = os.path.join(BASE_DIR, "source_identify_sls", "source_with_sls.csv")


def _parse_float(value: Any) -> float | None:
	if value is None:
		return None
	s = str(value).strip()
	if not s:
		return None
	# Common in your data: comma decimal separator
	s = s.replace(",", ".")
	try:
		return float(s)
	except ValueError:
		return None


def _normalize_int_str(value: Any) -> str:
	"""Normalize numeric strings like '003' -> '3'. Returns '' if missing/invalid."""
	if value is None:
		return ""
	s = str(value).strip()
	if not s:
		return ""
	try:
		return str(int(s))
	except ValueError:
		try:
			return str(int(float(s)))
		except Exception:
			return ""


def _normalize_name(value: Any) -> str:
	if value is None:
		return ""
	s = str(value).strip().upper()
	if not s:
		return ""
	return " ".join(s.split())


KODE_PREFIX_LEN = 10


def _kode_prefix(value: Any) -> str:
	"""Return first 10 chars of kode_wilayah, preferring digits if available."""
	if value is None:
		return ""
	s = str(value).strip()
	if not s:
		return ""
	digits = "".join(ch for ch in s if ch.isdigit())
	if len(digits) >= KODE_PREFIX_LEN:
		return digits[:KODE_PREFIX_LEN]
	return s[:KODE_PREFIX_LEN]


def _format_rt_rw_name(rt_value: Any, rw_value: Any) -> str:
	"""RT=1,RW=1 -> 'RT 001 RW 001'. Returns '' if missing/invalid."""
	rt_norm = _normalize_int_str(rt_value)
	rw_norm = _normalize_int_str(rw_value)
	if not rt_norm or not rw_norm:
		return ""
	try:
		rt_i = int(rt_norm)
		rw_i = int(rw_norm)
	except ValueError:
		return ""
	return f"RT {rt_i:03d} RW {rw_i:03d}".upper()


def _point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
	"""Ray casting algorithm for a single ring."""
	inside = False
	if len(ring) < 4:
		return False
	# Ensure ring closed for safety
	if ring[0] != ring[-1]:
		ring = ring + [ring[0]]

	for i in range(len(ring) - 1):
		x1, y1 = ring[i]
		x2, y2 = ring[i + 1]
		# Check if edge crosses horizontal ray
		if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1):
			inside = not inside
	return inside


def _point_in_polygon(x: float, y: float, polygon: list[list[list[float]]]) -> bool:
	"""Polygon = [exterior_ring, hole_ring1, hole_ring2, ...]."""
	if not polygon:
		return False
	exterior = polygon[0]
	if not _point_in_ring(x, y, exterior):
		return False
	# Exclude holes
	for hole in polygon[1:]:
		if _point_in_ring(x, y, hole):
			return False
	return True


def _polygon_bbox(polygon: list[list[list[float]]]) -> tuple[float, float, float, float]:
	# Only exterior ring bbox
	exterior = polygon[0] if polygon else []
	xs = [pt[0] for pt in exterior] if exterior else [0.0]
	ys = [pt[1] for pt in exterior] if exterior else [0.0]
	return min(xs), min(ys), max(xs), max(ys)


def _iter_polygons_from_geometry(geometry: dict[str, Any]) -> Iterable[list[list[list[float]]]]:
	"""Yield polygons as list-of-rings from a GeoJSON geometry dict."""
	if not geometry:
		return
	geom_type = geometry.get("type")
	coords = geometry.get("coordinates")
	if geom_type == "Polygon" and isinstance(coords, list):
		# coords: [ring1, ring2, ...]
		yield coords
	elif geom_type == "MultiPolygon" and isinstance(coords, list):
		# coords: [[poly1_rings], [poly2_rings], ...]
		for poly in coords:
			if isinstance(poly, list):
				yield poly
	# ignore other types


def _iter_polygons_from_geojson(obj: dict[str, Any]) -> Iterable[list[list[list[float]]]]:
	"""Yield polygons from GeoJSON root object."""
	obj_type = obj.get("type")
	if obj_type == "FeatureCollection":
		for feat in obj.get("features") or []:
			if isinstance(feat, dict):
				geom = feat.get("geometry") or {}
				yield from _iter_polygons_from_geometry(geom)
	elif obj_type == "Feature":
		geom = obj.get("geometry") or {}
		yield from _iter_polygons_from_geometry(geom)
	else:
		# Might be raw geometry
		yield from _iter_polygons_from_geometry(obj)


@dataclass(frozen=True)
class GeojsonPolygonIndexItem:
	idsls: str
	filename: str
	bbox: tuple[float, float, float, float]
	polygon: list[list[list[float]]]


def _load_geojson_index(geojson_dir: str) -> list[GeojsonPolygonIndexItem]:
	if not os.path.isdir(geojson_dir):
		raise FileNotFoundError(f"GeoJSON folder not found: {geojson_dir}")

	paths: list[str] = []
	for name in os.listdir(geojson_dir):
		lower = name.lower()
		if lower.endswith(".geojson") or lower.endswith(".json"):
			paths.append(os.path.join(geojson_dir, name))
	paths.sort()

	if not paths:
		raise FileNotFoundError(
			f"No .geojson/.json files found in: {geojson_dir}. "
			"(If the folder looks empty in VS Code, double-check the path.)"
		)

	items: list[GeojsonPolygonIndexItem] = []
	for path in paths:
		filename = os.path.basename(path)
		idsls = os.path.splitext(filename)[0]
		with open(path, "r", encoding="utf-8") as f:
			obj = json.load(f)
		for polygon in _iter_polygons_from_geojson(obj):
			bbox = _polygon_bbox(polygon)
			items.append(
				GeojsonPolygonIndexItem(idsls=idsls, filename=filename, bbox=bbox, polygon=polygon)
			)

	return items


def _load_master_maps(
	master_xlsx: str,
) -> tuple[
	dict[str, str],
	dict[tuple[str, str, str], str],
	dict[tuple[str, str], str],
	dict[tuple[str, str, str], tuple[str, str]],
]:
	if not os.path.isfile(master_xlsx):
		raise FileNotFoundError(f"master.xlsx not found: {master_xlsx}")

	try:
		df = pd.read_excel(master_xlsx, dtype=str, engine="openpyxl")
	except ImportError as e:
		raise RuntimeError("Reading .xlsx requires openpyxl. Install with: pip install openpyxl") from e

	# Normalize columns (handle different casing)
	col_map = {c.lower(): c for c in df.columns}
	if "idsls" not in col_map or "nmsls" not in col_map:
		raise KeyError("master.xlsx missing required columns: idsls, nmsls")

	idsls_col = col_map["idsls"]
	nmsls_col = col_map["nmsls"]
	rt_col = col_map.get("rt")
	rw_col = col_map.get("rw")

	ids = df[idsls_col].fillna("").astype(str).str.strip()
	nms = df[nmsls_col].fillna("").astype(str)
	idsls_only: dict[str, str] = {k: v for k, v in zip(ids, nms) if k}

	idsls_rt_rw: dict[tuple[str, str, str], str] = {}
	if rt_col and rw_col:
		rt_norm = df[rt_col].apply(_normalize_int_str)
		rw_norm = df[rw_col].apply(_normalize_int_str)
		for k, rt, rw, v in zip(ids, rt_norm, rw_norm, nms):
			if not k or rt == "" or rw == "":
				continue
			idsls_rt_rw[(k, rt, rw)] = v

	# Index for: (kode_wilayah_prefix10, name) -> idsls
	# where `name` is assumed to be in the `nmsls` column (often like 'RT 001 RW 001').
	prefix_name_to_idsls: dict[tuple[str, str], str] = {}
	for k, v in zip(ids, nms):
		k = str(k).strip()
		if not k:
			continue
		prefix = _kode_prefix(k)
		name_norm = _normalize_name(v)
		if not prefix or not name_norm:
			continue
		# Keep the first occurrence if duplicates exist.
		prefix_name_to_idsls.setdefault((prefix, name_norm), k)

	# Index for: (kode_wilayah_prefix10, rt, rw) -> (idsls, nmsls)
	prefix_rt_rw_to_master: dict[tuple[str, str, str], tuple[str, str]] = {}
	if rt_col and rw_col:
		rt_norm = df[rt_col].apply(_normalize_int_str)
		rw_norm = df[rw_col].apply(_normalize_int_str)
		for k, rt, rw, v in zip(ids, rt_norm, rw_norm, nms):
			k = str(k).strip()
			if not k or rt == "" or rw == "":
				continue
			prefix = _kode_prefix(k)
			if not prefix:
				continue
			prefix_rt_rw_to_master.setdefault((prefix, rt, rw), (k, v))

	return idsls_only, idsls_rt_rw, prefix_name_to_idsls, prefix_rt_rw_to_master


def _match_point_to_idsls(
	lon: float,
	lat: float,
	index_items: list[GeojsonPolygonIndexItem],
) -> tuple[str, str] | None:
	"""Returns (idsls, filename) if match found."""
	for item in index_items:
		minx, miny, maxx, maxy = item.bbox
		if lon < minx or lon > maxx or lat < miny or lat > maxy:
			continue
		if _point_in_polygon(lon, lat, item.polygon):
			return item.idsls, item.filename
	return None


def main() -> int:
	if not os.path.isfile(INPUT_CSV):
		raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

	print("Loading master.xlsx mapping...")
	idsls_to_nmsls, idsls_rt_rw_to_nmsls, prefix_name_to_idsls, prefix_rt_rw_to_master = _load_master_maps(
		MASTER_XLSX
	)
	print(f"Master idsls rows: {len(idsls_to_nmsls)}")
	if idsls_rt_rw_to_nmsls:
		print(f"Master (idsls,rt,rw) rows: {len(idsls_rt_rw_to_nmsls)}")
	if prefix_name_to_idsls:
		print(f"Master (prefix10,name)->idsls rows: {len(prefix_name_to_idsls)}")
	if prefix_rt_rw_to_master:
		print(f"Master (prefix10,rt,rw)->(idsls,nmsls) rows: {len(prefix_rt_rw_to_master)}")

	print("Loading GeoJSON polygons...")
	index_items = _load_geojson_index(GEOJSON_DIR)
	print(f"GeoJSON polygons indexed: {len(index_items)}")

	df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False, low_memory=False)
	for col in ("latitude", "longitude"):
		if col not in df.columns:
			raise KeyError(f"Input CSV missing required column: {col}")

	has_rt = "RT" in df.columns
	has_rw = "RW" in df.columns

	# Find kode_wilayah column (case-insensitive)
	kode_wilayah_col = None
	for c in df.columns:
		if c.lower() == "kode_wilayah":
			kode_wilayah_col = c
			break

	# Output columns
	if "matched_geojson" not in df.columns:
		df["matched_geojson"] = ""
	if "idsls" not in df.columns:
		df["idsls"] = ""
	if "nmsls" not in df.columns:
		df["nmsls"] = ""

	matched_polygon = 0
	processed = 0
	invalid_coords = 0
	matched_master_name = 0
	matched_any = 0

	for idx in df.index:
		# Always attempt polygon match if coords exist
		lat = _parse_float(df.at[idx, "latitude"])
		lon = _parse_float(df.at[idx, "longitude"])
		polygon_idsls = ""
		polygon_filename = ""
		prefix10 = _kode_prefix(df.at[idx, kode_wilayah_col]) if kode_wilayah_col else ""
		if lat is None or lon is None:
			invalid_coords += 1
		else:
			processed += 1
			match = _match_point_to_idsls(lon=lon, lat=lat, index_items=index_items)
			if match:
				polygon_idsls, polygon_filename = match
				# Enforce prefix10 filter: idsls[:10] must match kode_wilayah[:10] when kode_wilayah exists.
				if prefix10 and _kode_prefix(polygon_idsls) != prefix10:
					polygon_idsls = ""
					polygon_filename = ""
				else:
					df.at[idx, "matched_geojson"] = polygon_filename
					matched_polygon += 1

		# Prefer master match by kode_wilayah prefix + RT/RW name if possible
		idsls_final = polygon_idsls
		nmsls_final = ""
		target_name = ""
		if has_rt and has_rw:
			target_name = _format_rt_rw_name(df.at[idx, "RT"], df.at[idx, "RW"])

		if prefix10 and target_name and prefix_name_to_idsls:
			master_idsls = prefix_name_to_idsls.get((prefix10, _normalize_name(target_name)), "")
			if master_idsls:
				idsls_final = master_idsls
				nmsls_final = idsls_to_nmsls.get(master_idsls, "")
				matched_master_name += 1

		# If master has RT/RW columns, try (prefix10, RT, RW) as a secondary master match.
		if not nmsls_final and prefix10 and has_rt and has_rw and prefix_rt_rw_to_master:
			rt_val = _normalize_int_str(df.at[idx, "RT"])
			rw_val = _normalize_int_str(df.at[idx, "RW"])
			if rt_val and rw_val:
				master_row = prefix_rt_rw_to_master.get((prefix10, rt_val, rw_val))
				if master_row:
					idsls_final, nmsls_final = master_row

		# Fallbacks using polygon idsls mapping
		if not nmsls_final and idsls_final:
			# Enforce prefix10 filter for any idsls-based lookup.
			if prefix10 and _kode_prefix(idsls_final) != prefix10:
				idsls_final = ""
				nmsls_final = ""
			else:
				rt_val = _normalize_int_str(df.at[idx, "RT"]) if has_rt else ""
				rw_val = _normalize_int_str(df.at[idx, "RW"]) if has_rw else ""
				if rt_val and rw_val and idsls_rt_rw_to_nmsls:
					nmsls_final = idsls_rt_rw_to_nmsls.get(
						(idsls_final, rt_val, rw_val),
						idsls_to_nmsls.get(idsls_final, ""),
					)
				else:
					nmsls_final = idsls_to_nmsls.get(idsls_final, "")

		if idsls_final:
			df.at[idx, "idsls"] = idsls_final
			df.at[idx, "nmsls"] = nmsls_final
			matched_any += 1

	print(f"Rows with valid coords processed: {processed}")
	print(f"Rows invalid/missing coords: {invalid_coords}")
	print(f"Rows matched to polygon: {matched_polygon}")
	print(f"Rows matched to master by (kode_prefix10,name): {matched_master_name}")
	print(f"Rows with idsls filled (any method): {matched_any}")

	# Result shaping (requested):
	# 1) remove RT/RW columns
	for col in ("RT", "RW"):
		if col in df.columns:
			df.drop(columns=[col], inplace=True)

	# 2) move matched_geojson, idsls, nmsls next to is_geocode
	move_cols = ["matched_geojson", "idsls", "nmsls"]
	for c in move_cols:
		if c not in df.columns:
			df[c] = ""

	cols = [c for c in df.columns if c not in move_cols]
	insert_after = None
	if "is_geocode" in cols:
		insert_after = "is_geocode"
	elif "longitude" in cols:
		insert_after = "longitude"
	elif "latitude" in cols:
		insert_after = "latitude"

	if insert_after and insert_after in cols:
		pos = cols.index(insert_after) + 1
		cols = cols[:pos] + move_cols + cols[pos:]
	else:
		cols = cols + move_cols

	df = df.loc[:, cols]

	os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
	df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
	print(f"Wrote: {OUTPUT_CSV}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

