#!/usr/bin/env python3
"""
KDM ID to SBR ID Matching Script

This script matches businesses from two sources:
- Source 1 (folder: source): XLSX files with idsbr, nama_usaha, iddesa, latitude, longitude
- Source 2 (folder: source_kdm_all): CSV files with id, name, village_id, latitude, longitude

Matching Algorithm:
1. Combine all files from each source into arrays
2. Filter only rows with valid latitude and longitude in source 1
3. Convert comma to point in latitude/longitude values (e.g., "-6,1234" to "-6.1234")
4. Filter by matching village_id (iddesa in source 1, village_id in source 2)
5. Match by IDENTICAL name (nama_usaha = name) and EXACT same latitude/longitude coordinates

Output:
Saves matched results to result/ folder with columns: idsbr, id, and ALL columns from source 2
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
SOURCE1_PATTERN = "*.xlsx"
SOURCE2_PATTERN = "*.csv"

# Debug mode
DEBUG_MODE = False
DEBUG_ROW_LIMIT = 10000

# Output configuration
OUTPUT_FILENAME = 'matched_businesses.csv'
OUTPUT_FILENAME_DEBUG = 'matched_businesses_debug.csv'


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
                    df = pd.read_csv(file_path)
                
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
            
            logger.info(f"Combined shape: {combined_df.shape}")
            return combined_df
        
        return pd.DataFrame()
    
    def prepare_source1_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare source 1 data (XLSX files from 'source' folder).
        
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
        required_cols = ['idsbr', 'nama_usaha', 'iddesa', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in source 1: {missing_cols}")
            return pd.DataFrame()
        
        # Create working copy
        df = df.copy()
        
        # Normalize coordinates
        df['latitude'] = df['latitude'].apply(self.converter.normalize_coord)
        df['longitude'] = df['longitude'].apply(self.converter.normalize_coord)
        
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
        
        # Keep only required columns
        df = df[['idsbr', 'nama_usaha', 'iddesa', 'latitude', 'longitude']]
        
        logger.info(f"Source 1 ready: {df.shape}")
        return df
    
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
        df['latitude'] = df['latitude'].apply(self.converter.normalize_coord)
        df['longitude'] = df['longitude'].apply(self.converter.normalize_coord)
        
        # Standardize name
        df['name'] = df['name'].fillna('').astype(str).str.strip().str.lower()
        
        # Keep only required columns
        df = df[['id', 'name', 'village_id', 'latitude', 'longitude']]
        
        logger.info(f"Source 2 ready: {df.shape}")
        return df
    
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
        logger.info("Starting matching process - exact name and coordinate match")
        
        matches = []
        total_rows = len(source1)
        
        for idx, row1 in source1.iterrows():
            if (idx + 1) % 5000 == 0:
                logger.info(f"Processing row {idx + 1} / {total_rows}")
            
            # Get source 1 data
            idsbr = row1['idsbr']
            name1 = row1['nama_usaha']
            village_id1 = row1['iddesa']
            lat1 = row1['latitude']
            lon1 = row1['longitude']
            
            # Filter source 2 by village_id
            source2_filtered = source2[source2['village_id'] == village_id1]
            
            if source2_filtered.empty:
                continue
            
            # Find exact matches in filtered source 2
            for idx2, row2 in source2_filtered.iterrows():
                name2 = row2['name']
                lat2 = row2['latitude']
                lon2 = row2['longitude']
                
                # Check name is IDENTICAL
                if name1 != name2:
                    continue
                
                # Check coordinates are EXACTLY equivalent
                if lat1 != lat2 or lon1 != lon2:
                    continue
                
                # Get the raw row with all columns
                raw_row = source2_raw.iloc[idx2].to_dict()
                
                # Match found - add idsbr to the raw data
                raw_row['idsbr'] = idsbr
                matches.append(raw_row)
        
        result_df = pd.DataFrame(matches)
        logger.info(f"Found {len(result_df)} exact matches")
        
        # Reorder columns to have idsbr first
        if not result_df.empty:
            cols = result_df.columns.tolist()
            if 'idsbr' in cols:
                cols.remove('idsbr')
            result_df = result_df[['idsbr'] + cols]
        
        return result_df
    
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


def main():
    """
    Main execution function.
    Uses global DEBUG_MODE variable to process only first 10k rows from source 1 (all source 2 data loaded)
    """
    
    # Setup paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    source1_path = os.path.join(base_path, 'source')
    source2_path = os.path.join(base_path, 'source_kdm_all')
    output_filename = OUTPUT_FILENAME_DEBUG if DEBUG_MODE else OUTPUT_FILENAME
    output_path = os.path.join(base_path, 'result', output_filename)
    
    logger.info("=" * 80)
    logger.info("KDM ID to SBR ID Matching Script")
    if DEBUG_MODE:
        logger.info(f"DEBUG MODE ENABLED - Processing first {DEBUG_ROW_LIMIT} rows from source 1")
    logger.info("=" * 80)
    
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
        logger.info(f"DEBUG MODE: Limited source 1 from {original_count} to {len(source1)} rows")
    
    # Match businesses - pass raw source2 to include all columns
    results = matcher.match_businesses(source1, source2_raw, source2)
    
    if results.empty:
        logger.warning("No exact matches found")
    else:
        # Save results
        matcher.save_results(results, output_path)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("MATCHING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total source 1 records: {len(source1)}")
        logger.info(f"Total source 2 records: {len(source2)}")
        logger.info(f"Total exact matches found: {len(results)}")
        logger.info(f"Output file: {output_path}")
        logger.info("=" * 80)


if __name__ == '__main__':
    main()
