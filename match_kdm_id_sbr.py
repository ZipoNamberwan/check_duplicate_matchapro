#!/usr/bin/env python3
"""
KDM ID to SBR ID Matching Script

This script matches businesses from two sources:
- Source 1 (folder: source_matcha_pro_all): CSV files with idsbr, nama_usaha, kode_wilayah, latitude, longitude
- Source 2 (folder: source_kdm_all): CSV files with id, name, village_id, latitude, longitude

Matching Algorithm:
1. Combine all files from each source into arrays
2. Filter only rows with valid latitude and longitude in source 1
3. Convert comma to point in latitude/longitude values (e.g., "-6,1234" to "-6.1234")
4. Derive village_id key from source 1: first 10 characters of kode_wilayah
5. Filter by matching village_id (derived from kode_wilayah in source 1, village_id in source 2)
6. Match by IDENTICAL name (nama_usaha = name) and EXACT same latitude/longitude coordinates

Output:
1) result/result1.csv: ALL rows from source 2, plus column 'idsbr' (filled when matched, empty when not)
2) result/result2.csv: ALL rows from source 1 with valid coordinates that did not match source 2
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import glob
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# GLOBAL CONFIGURATION
# =====================================================================

# Coordinate matching
RADIUS_METERS = 100

# File patterns
SOURCE1_PATTERN = "*.csv"
SOURCE2_PATTERN = "*.csv"

# Debug mode
DEBUG_MODE = False
DEBUG_ROW_LIMIT = 10000

# Streaming save mode (write results incrementally while processing)
STREAM_SAVE = False
STREAM_CHUNK_SIZE = 200_000

# Output configuration
OUTPUT_FILENAME_ALL_SOURCE2 = 'result1.csv'
OUTPUT_FILENAME_UNMATCHED_SOURCE1 = 'result2.csv'

# Enable/disable outputs
# - result1: ALL rows from source2 + idsbr when matched
# - result2: source1 rows (valid coords) that did not match source2
WRITE_RESULT1 = False
WRITE_RESULT2 = True

# Optional: split result1 into multiple CSVs by regency_id to avoid extremely large single files.
# Requires 'regency_id' column to exist in source2 data (it will be carried into result1).
SPLIT_RESULT1_BY_REGENCY = True
RESULT1_SPLIT_COLUMN = 'regency_id'

# When splitting, optionally log per-regency counts (can be noisy if many regencies)
SPLIT_RESULT1_LOG_PER_REGENCY = False
SPLIT_RESULT1_LOG_PER_REGENCY_MAX = 50

# Optional: split result2 into multiple CSVs by kode_wilayah prefix.
# This helps when result2 is very large and you want to process per-area.
SPLIT_RESULT2_BY_KODE_PREFIX = True
RESULT2_SPLIT_COLUMN = 'kode_wilayah'
RESULT2_KODE_PREFIX_LEN = 4
RESULT2_SPLIT_DIRNAME = 'split_result2'


class LatLongConverter:
    """Converter for latitude/longitude coordinates"""
    
    @staticmethod
    def normalize_coord(value):
        """
        Normalize coordinate value by converting comma to point.
        Handles various input types: string, int, float, NaN
        
        Args:
            value: Coordinate value to normalize
            
        Returns:
            float: Normalized coordinate value, or NaN if invalid
        """
        # Handle NaN/None values
        if pd.isna(value):
            return np.nan
        
        # Convert to string and clean
        str_val = str(value).strip()
        
        # Replace comma with point
        str_val = str_val.replace(',', '.')
        
        try:
            return float(str_val)
        except ValueError:
            return np.nan

    @staticmethod
    def normalize_coord_series(series: pd.Series) -> pd.Series:
        """Vectorized coordinate normalization for pandas Series."""
        # Convert to string, strip, replace comma with dot, then coerce to float
        s = series.astype(str).str.strip().str.replace(',', '.', regex=False)
        return pd.to_numeric(s, errors='coerce')
    
    @staticmethod
    def is_valid_coordinate(lat, lon):
        """
        Check if coordinates are valid.
        Valid range: latitude [-90, 90], longitude [-180, 180]
        
        Args:
            lat: Latitude value
            lon: Longitude value
            
        Returns:
            bool: True if both coordinates are valid
        """
        if pd.isna(lat) or pd.isna(lon):
            return False
        
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            return -90 <= lat_f <= 90 and -180 <= lon_f <= 180
        except (ValueError, TypeError):
            return False


class BusinessMatcher:
    """Matches businesses between two data sources"""
    
    def __init__(self, radius_meters: float = None):
        """
        Initialize the matcher.
        
        Args:
            radius_meters: Distance threshold in meters for coordinate matching.
                          If None, uses RADIUS_METERS global variable
        """
        self.radius_meters = radius_meters if radius_meters is not None else RADIUS_METERS
        self.converter = LatLongConverter()
        
    def load_source_files(self, folder_path: str, file_pattern: str) -> pd.DataFrame:
        """
        Load and combine all files from a folder.
        
        For source_kdm_all CSV files:
        - market_* files don't have 'owner' and 'project_id' columns
        - supplement_* files have 'owner' and 'project_id' columns
        Missing columns are added with NaN values to maintain consistency
        
        Args:
            folder_path: Path to folder containing source files
            file_pattern: File extension pattern (e.g., "*.xlsx", "*.csv")
            
        Returns:
            pd.DataFrame: Combined data from all files
        """
        logger.info(f"Loading files from {folder_path} with pattern {file_pattern}")
        
        files = glob.glob(os.path.join(folder_path, file_pattern))
        logger.info(f"Found {len(files)} files")
        
        if not files:
            logger.warning(f"No files found matching {file_pattern} in {folder_path}")
            return pd.DataFrame()
        
        dfs = []
        for file_path in files:
            try:
                logger.info(f"Reading {os.path.basename(file_path)}")
                if file_pattern.endswith(".xlsx"):
                    df = pd.read_excel(file_path)
                else:
                    # Read as strings to preserve codes (e.g., leading zeros) and keep matching stable.
                    df = pd.read_csv(file_path, dtype=str, low_memory=False)
                
                logger.info(f"  - Shape: {df.shape}")
                dfs.append(df)
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {str(e)}")
                continue
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Handle missing columns for source_kdm_all files
            # market_* files don't have 'owner' and 'project_id' columns
            if file_pattern == SOURCE2_PATTERN:  # CSV files from source_kdm_all
                if 'owner' not in combined_df.columns:
                    logger.info("Adding missing 'owner' column")
                    combined_df['owner'] = np.nan
                if 'project_id' not in combined_df.columns:
                    logger.info("Adding missing 'project_id' column")
                    combined_df['project_id'] = np.nan

            # Source 2 soft-delete filter: keep only rows where deleted_at is null/empty.
            # Applied only for source_kdm* folders to avoid unintended filtering of other datasets.
            folder_name = Path(folder_path).name.lower()
            if folder_name.startswith('source_kdm') and 'deleted_at' in combined_df.columns:
                deleted_at = combined_df['deleted_at']
                deleted_at_str = deleted_at.astype(str).str.strip().str.lower()
                active_mask = (
                    deleted_at.isna()
                    | (deleted_at_str == '')
                    | (deleted_at_str.isin(['nan', 'none', 'null']))
                )
                before_count = len(combined_df)
                combined_df = combined_df[active_mask].copy()
                after_count = len(combined_df)
                logger.info(f"Filtered source2 by deleted_at is null: {before_count} -> {after_count} rows")
            elif folder_name.startswith('source_kdm'):
                logger.info("Source2 has no 'deleted_at' column; skipping deleted_at filter")
            
            logger.info(f"Combined shape: {combined_df.shape}")
            return combined_df
        
        return pd.DataFrame()

    @staticmethod
    def normalize_village_key_series(series: pd.Series) -> pd.Series:
        """Normalize village key values to stable strings.

        - Strips whitespace
        - Converts NaN to empty string
        - Removes trailing '.0' that often appears when numeric codes are parsed as floats
        """
        s = series.fillna('').astype(str).str.strip()
        s = s.str.replace(r'\.0$', '', regex=True)
        return s
    
    def prepare_source1_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare source 1 data (CSV files from 'source_matcha_pro_all' folder).
        
        Steps:
        1. Filter rows with valid latitude and longitude
        2. Normalize latitude and longitude (convert comma to point)
        3. Standardize name column
        
        Args:
            df: Raw source 1 dataframe
            
        Returns:
            pd.DataFrame: Cleaned source 1 dataframe
        """
        logger.info("Preparing Source 1 data...")
        
        # Check required columns
        required_cols = ['idsbr', 'nama_usaha', 'kode_wilayah', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in source 1: {missing_cols}")
            return pd.DataFrame()
        
        # Create working copy
        df = df.copy()
        
        # Derive village_id key (iddesa) from kode_wilayah (first 10 chars)
        df['iddesa'] = self.normalize_village_key_series(df['kode_wilayah']).str.slice(0, 10)

        # Normalize coordinates
        df['latitude'] = self.converter.normalize_coord_series(df['latitude'])
        df['longitude'] = self.converter.normalize_coord_series(df['longitude'])
        
        # Filter valid coordinates
        initial_count = len(df)
        df['valid_coord'] = df.apply(
            lambda row: self.converter.is_valid_coordinate(row['latitude'], row['longitude']),
            axis=1
        )
        df = df[df['valid_coord']].copy()
        filtered_count = len(df)
        
        logger.info(f"Filtered from {initial_count} to {filtered_count} rows with valid coordinates")
        logger.info(f"Filtered {initial_count - filtered_count} rows with invalid/missing coordinates")
        
        # Standardize name
        df['nama_usaha'] = df['nama_usaha'].fillna('').astype(str).str.strip().str.lower()
        
        # Keep only required columns (iddesa derived from kode_wilayah)
        df = df[['idsbr', 'nama_usaha', 'iddesa', 'latitude', 'longitude']]
        
        logger.info(f"Source 1 ready: {df.shape}")
        return df

    def prepare_source1_raw_with_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare source 1 raw data while preserving ALL columns.

        Adds helper key columns used for matching:
        - _village_key: first 10 chars of kode_wilayah
        - _name_key: normalized nama_usaha
        - _lat_key / _lon_key: normalized coordinates

        Keeps only rows with valid coordinates.
        """
        required_cols = ['idsbr', 'nama_usaha', 'kode_wilayah', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in source 1: {missing_cols}")
            return pd.DataFrame()

        out = df.copy()
        out['_village_key'] = self.normalize_village_key_series(out['kode_wilayah']).str.slice(0, 10)
        out['_name_key'] = out['nama_usaha'].fillna('').astype(str).str.strip().str.lower()
        out['_lat_key'] = self.converter.normalize_coord_series(out['latitude'])
        out['_lon_key'] = self.converter.normalize_coord_series(out['longitude'])

        out['valid_coord'] = out.apply(
            lambda row: self.converter.is_valid_coordinate(row['_lat_key'], row['_lon_key']),
            axis=1,
        )
        out = out[out['valid_coord']].copy()
        return out
    
    def prepare_source2_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare source 2 data (CSV files from 'source_kdm_all' folder).
        
        Steps:
        1. Normalize latitude and longitude
        2. Standardize name column
        
        Args:
            df: Raw source 2 dataframe
            
        Returns:
            pd.DataFrame: Cleaned source 2 dataframe
        """
        logger.info("Preparing Source 2 data...")
        
        # Check required columns
        required_cols = ['id', 'name', 'village_id', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in source 2: {missing_cols}")
            return pd.DataFrame()
        
        # Create working copy
        df = df.copy()
        
        # Normalize coordinates
        df['latitude'] = self.converter.normalize_coord_series(df['latitude'])
        df['longitude'] = self.converter.normalize_coord_series(df['longitude'])
        
        # Standardize name
        df['name'] = df['name'].fillna('').astype(str).str.strip().str.lower()
        
        # Keep only required columns
        df = df[['id', 'name', 'village_id', 'latitude', 'longitude']]
        
        logger.info(f"Source 2 ready: {df.shape}")
        return df

    def prepare_source2_raw_with_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare source 2 raw data while preserving ALL columns and adding match keys."""
        required_cols = ['id', 'name', 'village_id', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in source 2: {missing_cols}")
            return pd.DataFrame()

        out = df.copy()
        out['_village_key'] = self.normalize_village_key_series(out['village_id'])
        out['_name_key'] = out['name'].fillna('').astype(str).str.strip().str.lower()
        out['_lat_key'] = self.converter.normalize_coord_series(out['latitude'])
        out['_lon_key'] = self.converter.normalize_coord_series(out['longitude'])
        return out
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates in meters using Haversine formula.
        
        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate
            
        Returns:
            float: Distance in meters
        """
        from math import radians, cos, sin, asin, sqrt
        
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371000  # Radius of earth in meters
        
        return c * r
    
    def is_coordinate_match(self, lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """
        Check if two coordinates are close enough to be considered a match.
        
        Args:
            lat1, lon1: Source 1 coordinate
            lat2, lon2: Source 2 coordinate
            
        Returns:
            bool: True if distance is within radius threshold
        """
        distance = self.calculate_distance(lat1, lon1, lat2, lon2)
        return distance <= self.radius_meters
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            str: Normalized text
        """
        # Remove extra spaces, convert to lowercase
        return ' '.join(str(text).lower().split())
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using SequenceMatcher ratio.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity ratio between 0 and 1
        """
        from difflib import SequenceMatcher
        
        text1 = self.normalize_text(text1)
        text2 = self.normalize_text(text2)
        
        return SequenceMatcher(None, text1, text2).ratio()
    
    def match_businesses(self, source1: pd.DataFrame, source2_raw: pd.DataFrame,
                        source2: pd.DataFrame) -> pd.DataFrame:
        """
        Match businesses between source 1 and source 2.
        
        Algorithm:
        1. For each row in source 1:
            a. Filter source 2 by matching village_id (iddesa = village_id)
            b. Check if nama_usaha and name are IDENTICAL
            c. Check if coordinates are EXACTLY equivalent
        
        Args:
            source1: Prepared source 1 dataframe
            source2_raw: Raw source 2 dataframe with all columns
            source2: Prepared source 2 dataframe
            
        Returns:
            pd.DataFrame: Matched results with idsbr, id, and all source2 columns
        """
        logger.info("Starting matching process - exact name and coordinate match (vectorized)")

        # Build match keys without mutating raw output columns
        s1_keys = source1[['idsbr', 'iddesa', 'nama_usaha', 'latitude', 'longitude']].copy()
        s1_keys.rename(
            columns={
                'iddesa': '_village_key',
                'nama_usaha': '_name_key',
                'latitude': '_lat_key',
                'longitude': '_lon_key',
            },
            inplace=True,
        )

        s2_keys = source2_raw.copy()
        required_cols = ['id', 'name', 'village_id', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in s2_keys.columns]
        if missing_cols:
            logger.error(f"Missing columns in source 2 raw: {missing_cols}")
            return pd.DataFrame()

        s2_keys['_village_key'] = self.normalize_village_key_series(s2_keys['village_id'])
        s2_keys['_name_key'] = s2_keys['name'].fillna('').astype(str).str.strip().str.lower()
        s2_keys['_lat_key'] = self.converter.normalize_coord_series(s2_keys['latitude'])
        s2_keys['_lon_key'] = self.converter.normalize_coord_series(s2_keys['longitude'])

        # Inner join on the exact-match keys
        merged = pd.merge(
            s2_keys,
            s1_keys[['idsbr', '_village_key', '_name_key', '_lat_key', '_lon_key']],
            on=['_village_key', '_name_key', '_lat_key', '_lon_key'],
            how='inner',
            sort=False,
            copy=False,
        )

        # Drop helper key columns
        merged.drop(columns=['_village_key', '_name_key', '_lat_key', '_lon_key'], inplace=True)

        logger.info(f"Found {len(merged)} exact matches")

        # Reorder columns to have idsbr first
        if not merged.empty:
            cols = merged.columns.tolist()
            cols.remove('idsbr')
            merged = merged[['idsbr'] + cols]

        return merged

    def build_all_source2_with_idsbr(self, source1_prepared: pd.DataFrame, source2_raw: pd.DataFrame) -> pd.DataFrame:
        """Return ALL source2 rows + idsbr column (blank if not matched).

        Matching is exact on: village_id, normalized name, normalized lat, normalized lon.
        If multiple idsbr match the same source2 key, idsbr values are joined with ';'.
        """
        logger.info("Building result_1: all source2 rows with idsbr filled when matched")

        if source1_prepared.empty or source2_raw.empty:
            return pd.DataFrame()

        s1 = source1_prepared[['idsbr', 'iddesa', 'nama_usaha', 'latitude', 'longitude']].copy()
        s1['_village_key'] = self.normalize_village_key_series(s1['iddesa'])
        s1['_name_key'] = s1['nama_usaha'].fillna('').astype(str).str.strip().str.lower()
        s1['_lat_key'] = self.converter.normalize_coord_series(s1['latitude'])
        s1['_lon_key'] = self.converter.normalize_coord_series(s1['longitude'])

        # Aggregate idsbr per key (to keep output 1 row per source2 row)
        idsbr_map = (
            s1.groupby(['_village_key', '_name_key', '_lat_key', '_lon_key'], dropna=False)['idsbr']
            .apply(lambda x: ';'.join([str(v) for v in x.dropna().astype(str).unique() if str(v).strip() != '']))
            .reset_index()
        )

        s2 = self.prepare_source2_raw_with_keys(source2_raw)
        if s2.empty:
            return pd.DataFrame()

        out = pd.merge(
            s2,
            idsbr_map,
            on=['_village_key', '_name_key', '_lat_key', '_lon_key'],
            how='left',
            sort=False,
            copy=False,
        )

        # Ensure idsbr exists and blank when no match
        out['idsbr'] = out['idsbr'].fillna('')

        # Drop helper keys, keep original source2 columns + idsbr
        out.drop(columns=['_village_key', '_name_key', '_lat_key', '_lon_key'], inplace=True)

        cols = out.columns.tolist()
        if 'idsbr' in cols:
            cols.remove('idsbr')
            out = out[['idsbr'] + cols]

        logger.info(f"result_1 rows: {len(out)} (matched idsbr non-empty: {(out['idsbr'] != '').sum()})")
        return out

    def build_unmatched_source1_with_coords(self, source1_raw: pd.DataFrame, source2_raw: pd.DataFrame) -> pd.DataFrame:
        """Return source1 rows with valid coords that did NOT match any source2 row."""
        logger.info("Building result_2: source1 rows with valid coords that are unmatched")

        s1 = self.prepare_source1_raw_with_keys(source1_raw)
        s2 = self.prepare_source2_raw_with_keys(source2_raw)

        if s1.empty:
            logger.warning("No valid-coordinate rows in source 1")
            return pd.DataFrame()
        if s2.empty:
            logger.warning("Source 2 missing required columns; treating all valid source 1 as unmatched")
            out = s1.copy()
            out.drop(columns=['valid_coord', '_village_key', '_name_key', '_lat_key', '_lon_key'], inplace=True, errors='ignore')
            return out

        s2_unique_keys = s2[['_village_key', '_name_key', '_lat_key', '_lon_key']].drop_duplicates()

        merged = pd.merge(
            s1,
            s2_unique_keys,
            on=['_village_key', '_name_key', '_lat_key', '_lon_key'],
            how='left',
            indicator=True,
            sort=False,
            copy=False,
        )

        out = merged[merged['_merge'] == 'left_only'].copy()
        out.drop(columns=['_merge'], inplace=True)

        # Drop helper columns (preserve original columns)
        out.drop(columns=['valid_coord', '_village_key', '_name_key', '_lat_key', '_lon_key'], inplace=True, errors='ignore')
        logger.info(f"result_2 rows (unmatched with coords): {len(out)}")
        return out

    def match_businesses_to_csv(
        self,
        source1: pd.DataFrame,
        source2_raw: pd.DataFrame,
        output_path: str,
        chunk_size: int = STREAM_CHUNK_SIZE,
    ) -> int:
        """Stream exact matches to CSV while processing in chunks.

        This avoids holding all matches in memory and lets you see results
        immediately on disk.

        Returns:
            int: Total matches written.
        """
        logger.info("Starting matching process - exact name and coordinate match (streaming)")

        required_cols = ['id', 'name', 'village_id', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in source2_raw.columns]
        if missing_cols:
            logger.error(f"Missing columns in source 2 raw: {missing_cols}")
            return 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare source2 indexed table once
        s2 = source2_raw.copy()
        s2['_village_key'] = s2['village_id']
        s2['_name_key'] = s2['name'].fillna('').astype(str).str.strip().str.lower()
        s2['_lat_key'] = self.converter.normalize_coord_series(s2['latitude'])
        s2['_lon_key'] = self.converter.normalize_coord_series(s2['longitude'])

        s2_indexed = s2.set_index(['_village_key', '_name_key', '_lat_key', '_lon_key'])

        # Prepare source1 keys
        s1 = source1[['idsbr', 'iddesa', 'nama_usaha', 'latitude', 'longitude']].copy()
        s1.rename(
            columns={
                'iddesa': '_village_key',
                'nama_usaha': '_name_key',
                'latitude': '_lat_key',
                'longitude': '_lon_key',
            },
            inplace=True,
        )

        total_rows = len(s1)
        total_written = 0
        wrote_header = os.path.exists(output_path)

        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = s1.iloc[start:end].copy()
            chunk_indexed = chunk.set_index(['_village_key', '_name_key', '_lat_key', '_lon_key'])

            # Join: keeps idsbr from chunk and all columns from source2
            joined = chunk_indexed.join(s2_indexed, how='inner', rsuffix='_s2')
            if joined.empty:
                if end % (chunk_size * 5) == 0 or end == total_rows:
                    logger.info(f"Processed {end} / {total_rows} rows (matches written: {total_written})")
                continue

            joined.reset_index(drop=True, inplace=True)

            # Ensure idsbr first
            cols = joined.columns.tolist()
            if 'idsbr' in cols:
                cols.remove('idsbr')
                joined = joined[['idsbr'] + cols]

            joined.to_csv(output_path, index=False, mode='a', header=not wrote_header)
            wrote_header = True

            total_written += len(joined)
            logger.info(
                f"Processed {end} / {total_rows} rows (chunk matches: {len(joined)}, total written: {total_written})"
            )

        if total_written == 0:
            logger.warning("No exact matches found")
        else:
            logger.info(f"Streaming results saved to {output_path}")

        return total_written
    
    def save_results(self, results: pd.DataFrame, output_path: str):
        """
        Save matching results to CSV file.
        
        Args:
            results: Dataframe with matching results
            output_path: Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    def save_results_csv(self, results: pd.DataFrame, output_path: str) -> str:
        """Save results to CSV.

        CSV avoids Excel row limits and avoids the large intermediate allocations that
        pandas may perform when exporting to .xlsx.

        Returns:
            str: The actual file path written.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        return output_path


def main():
    """
    Main execution function.
    Uses global DEBUG_MODE variable to process only first 10k rows from source 1 (all source 2 data loaded)
    """
    
    # Setup paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    source1_path = os.path.join(base_path, 'source_matcha_pro_all')
    source2_path = os.path.join(base_path, 'source_kdm_all')
    result_dir = os.path.join(base_path, 'result')
    result1_dir = os.path.join(result_dir, 'split') if SPLIT_RESULT1_BY_REGENCY else result_dir

    output_path_1 = os.path.join(result1_dir, OUTPUT_FILENAME_ALL_SOURCE2)
    output_path_2 = os.path.join(result_dir, OUTPUT_FILENAME_UNMATCHED_SOURCE1)
    
    logger.info("=" * 80)
    logger.info("KDM ID to SBR ID Matching Script")
    if DEBUG_MODE:
        logger.info(f"DEBUG MODE ENABLED - Processing first {DEBUG_ROW_LIMIT} rows from source 1")
    logger.info(f"WRITE_RESULT1={WRITE_RESULT1}, WRITE_RESULT2={WRITE_RESULT2}")
    logger.info("=" * 80)

    if not WRITE_RESULT1 and not WRITE_RESULT2:
        logger.warning("Both result outputs are disabled (WRITE_RESULT1 and WRITE_RESULT2 are False). Nothing to do.")
        return
    
    # Initialize matcher
    matcher = BusinessMatcher()
    
    # Load data
    source1_raw = matcher.load_source_files(source1_path, SOURCE1_PATTERN)
    source2_raw = matcher.load_source_files(source2_path, SOURCE2_PATTERN)
    
    if source1_raw.empty or source2_raw.empty:
        logger.error("Failed to load source data")
        return
    
    # Prepare data
    source1 = matcher.prepare_source1_data(source1_raw)
    source2 = matcher.prepare_source2_data(source2_raw)
    
    if source1.empty or source2.empty:
        logger.error("Failed to prepare source data")
        return
    
    # Debug mode: limit source 1 to first 10k rows
    if DEBUG_MODE:
        original_count = len(source1)
        source1 = source1.head(DEBUG_ROW_LIMIT)
        source1_raw = source1_raw.head(DEBUG_ROW_LIMIT)
        logger.info(f"DEBUG MODE: Limited source 1 from {original_count} to {len(source1)} rows")
    
    # Build + save outputs (based on toggles)
    result_1 = pd.DataFrame()
    result_2 = pd.DataFrame()

    result1_split_written = False
    result1_split_dir = None

    if WRITE_RESULT1:
        result_1 = matcher.build_all_source2_with_idsbr(source1_prepared=source1, source2_raw=source2_raw)
        if SPLIT_RESULT1_BY_REGENCY and not result_1.empty:
            if RESULT1_SPLIT_COLUMN not in result_1.columns:
                logger.warning(
                    f"SPLIT_RESULT1_BY_REGENCY=True but column '{RESULT1_SPLIT_COLUMN}' not found in result1. "
                    "Writing a single CSV instead."
                )
                matcher.save_results_csv(result_1, output_path_1)
            else:
                out_dir = os.path.dirname(output_path_1)
                os.makedirs(out_dir, exist_ok=True)

                stem = Path(output_path_1).stem
                suffix = Path(output_path_1).suffix or '.csv'

                def _safe_part(value) -> str:
                    if pd.isna(value):
                        value_str = 'NA'
                    else:
                        value_str = str(value).strip()
                        if value_str == '':
                            value_str = 'NA'
                    return ''.join(ch if (ch.isalnum() or ch in ['-', '_']) else '_' for ch in value_str)

                file_count = 0
                total_rows = len(result_1)
                matched_rows_total = 0
                unmatched_rows_total = 0
                per_regency_counts = []
                summary_records = []
                for regency_value, group in result_1.groupby(RESULT1_SPLIT_COLUMN, dropna=False, sort=False):
                    part = _safe_part(regency_value)
                    split_path = os.path.join(out_dir, f"{stem}_{RESULT1_SPLIT_COLUMN}_{part}{suffix}")
                    group.to_csv(split_path, index=False)

                    matched_in_group = int((group['idsbr'] != '').sum()) if 'idsbr' in group.columns else 0
                    unmatched_in_group = int(len(group) - matched_in_group)
                    matched_rows_total += matched_in_group
                    unmatched_rows_total += unmatched_in_group

                    summary_records.append(
                        {
                            RESULT1_SPLIT_COLUMN: regency_value,
                            'total_rows': int(len(group)),
                            'matched_rows': matched_in_group,
                            'not_matched_rows': unmatched_in_group,
                        }
                    )
                    if SPLIT_RESULT1_LOG_PER_REGENCY and len(per_regency_counts) < SPLIT_RESULT1_LOG_PER_REGENCY_MAX:
                        per_regency_counts.append((regency_value, len(group), matched_in_group, unmatched_in_group))

                    file_count += 1

                result1_split_written = True
                result1_split_dir = out_dir

                logger.info(
                    f"Result1 split complete: wrote {file_count} files to {out_dir} "
                    f"(total rows: {total_rows}, matched: {matched_rows_total}, not matched: {unmatched_rows_total})"
                )

                if summary_records:
                    summary_path = os.path.join(out_dir, 'summary.csv')
                    pd.DataFrame(summary_records).to_csv(summary_path, index=False)
                    logger.info(f"Result1 split summary saved to {summary_path}")

                if SPLIT_RESULT1_LOG_PER_REGENCY and per_regency_counts:
                    for regency_value, total_in_group, matched_in_group, unmatched_in_group in per_regency_counts:
                        logger.info(
                            f"Split summary {RESULT1_SPLIT_COLUMN}={regency_value}: "
                            f"total={total_in_group}, matched={matched_in_group}, not matched={unmatched_in_group}"
                        )
        else:
            matcher.save_results_csv(result_1, output_path_1)
    else:
        logger.info("Skipping result1 generation (WRITE_RESULT1=False)")

    if WRITE_RESULT2:
        result_2 = matcher.build_unmatched_source1_with_coords(source1_raw=source1_raw, source2_raw=source2_raw)
        if SPLIT_RESULT2_BY_KODE_PREFIX and not result_2.empty:
            if RESULT2_SPLIT_COLUMN not in result_2.columns:
                logger.warning(
                    f"SPLIT_RESULT2_BY_KODE_PREFIX=True but column '{RESULT2_SPLIT_COLUMN}' not found in result2. "
                    "Writing a single CSV instead."
                )
                matcher.save_results_csv(result_2, output_path_2)
            else:
                out_dir = os.path.join(os.path.dirname(output_path_2), RESULT2_SPLIT_DIRNAME)
                os.makedirs(out_dir, exist_ok=True)

                stem = Path(output_path_2).stem
                suffix = Path(output_path_2).suffix or '.csv'

                def _safe_part(value) -> str:
                    if pd.isna(value):
                        value_str = 'NA'
                    else:
                        value_str = str(value).strip()
                        if value_str == '':
                            value_str = 'NA'
                    return ''.join(ch if (ch.isalnum() or ch in ['-', '_']) else '_' for ch in value_str)

                kode_series = BusinessMatcher.normalize_village_key_series(result_2[RESULT2_SPLIT_COLUMN])
                prefix_series = kode_series.str.slice(0, int(RESULT2_KODE_PREFIX_LEN)).fillna('')
                prefix_series = prefix_series.mask(prefix_series.str.strip() == '', 'NA')

                tmp = result_2.copy()
                tmp['_kode_prefix'] = prefix_series

                file_count = 0
                total_rows = len(tmp)
                summary_records = []

                for prefix_value, group in tmp.groupby('_kode_prefix', dropna=False, sort=False):
                    part = _safe_part(prefix_value)
                    split_path = os.path.join(out_dir, f"{stem}_{RESULT2_SPLIT_COLUMN[:12]}_{RESULT2_KODE_PREFIX_LEN}_{part}{suffix}")
                    group.drop(columns=['_kode_prefix'], inplace=False).to_csv(split_path, index=False)
                    summary_records.append(
                        {
                            'kode_prefix': prefix_value,
                            'total_rows': int(len(group)),
                        }
                    )
                    file_count += 1

                logger.info(
                    f"Result2 split complete: wrote {file_count} files to {out_dir} (total rows: {total_rows})"
                )

                if summary_records:
                    summary_path = os.path.join(out_dir, 'summary_result2.csv')
                    pd.DataFrame(summary_records).to_csv(summary_path, index=False)
                    logger.info(f"Result2 split summary saved to {summary_path}")
        else:
            matcher.save_results_csv(result_2, output_path_2)
    else:
        logger.info("Skipping result2 generation (WRITE_RESULT2=False)")

    logger.info("=" * 80)
    logger.info("MATCHING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total source 1 records (valid coords): {len(source1)}")
    logger.info(f"Total source 2 records: {len(source2)}")
    if WRITE_RESULT1:
        if not result_1.empty and 'idsbr' in result_1.columns:
            matched_count = int((result_1['idsbr'] != '').sum())
            not_matched_count = int((result_1['idsbr'] == '').sum())
            logger.info(f"Total source 2 rows with idsbr match: {matched_count}")
            logger.info(f"Total source 2 rows NOT matched (idsbr empty): {not_matched_count}")
        if result1_split_written and result1_split_dir:
            logger.info(f"Output folder 1: {result1_split_dir}")
        else:
            logger.info(f"Output file 1: {output_path_1}")
    else:
        logger.info("Output file 1: (skipped)")

    if WRITE_RESULT2:
        logger.info(f"Total unmatched source 1 rows (valid coords): {len(result_2)}")
        if SPLIT_RESULT2_BY_KODE_PREFIX and not result_2.empty and RESULT2_SPLIT_COLUMN in result_2.columns:
            logger.info(f"Output folder 2: {os.path.join(os.path.dirname(output_path_2), RESULT2_SPLIT_DIRNAME)}")
        else:
            logger.info(f"Output file 2: {output_path_2}")
    else:
        logger.info("Output file 2: (skipped)")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
