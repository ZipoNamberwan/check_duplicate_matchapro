#!/usr/bin/env python3
"""
Business Duplicate Detector with Combined Data Sources (SBR + KDM)

This script is an enhanced version of duplicate_detection.py that supports multiple data sources:
1. SBR Source (source/ folder) - Original source with Sumber filter
2. KDM Source (source_kdm/ folder) - New source without filters

Features:
- Loads businesses from multiple sources
- Applies source-specific filtering rules
- Assigns business_type based on source (sbr/kdm)
- Uses R-tree spatial indexing for efficient duplicate detection
- Text similarity analysis with configurable rules
- Common words filtering support

Configuration:
- RADIUS_METERS: Search radius (default: 70 meters)
- SIMILARITY_THRESHOLD: Similarity score threshold (default: 0.75)
- USE_COMMON_WORDS_FILTERING: Enable/disable common words filtering (default: True)

Data Source Mappings:
SBR Source (source/ folder):
  - Business name: nama_usaha
  - Owner: extracted from nama_usaha <owner> or (owner)
  - Latitude: latitude
  - Longitude: longitude
  - Business Type: sbr
  - Filtering: Sumber Data = 'PL-KUMKM 2023'

KDM Source (source_kdm/ folder):
  - Business name: Nama_Usaha
  - Owner: Pemilik
  - Latitude: Latitude
  - Longitude: Longitude
  - Business Type: kdm
  - Filtering: No filtering (use all rows)
"""

import os
import time
import string
import csv
import difflib
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pytz
import glob
import re

import pandas as pd
from dotenv import load_dotenv
from rtree import index
from shapely.geometry import Point

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Timezone configuration
JAKARTA_TIMEZONE = pytz.timezone('Asia/Jakarta')

def get_jakarta_now():
    """Get current timestamp in Jakarta timezone"""
    return datetime.now(JAKARTA_TIMEZONE)

# =====================================================================
# CONFIGURATION
# =====================================================================

# Search radius in meters
RADIUS_METERS = 70

# Excel data configuration
EXCEL_SBR_FOLDER = 'source'  # Folder containing SBR Excel files (original source)
EXCEL_KDM_FOLDER = 'source_kdm'  # Folder containing KDM Excel files (new source)
EXCEL_RESULT_FOLDER = 'result'  # Folder to save results
EXCEL_RESULT_FILENAME = 'result.xlsx'  # Output filename

# SBR Source filter configuration
FILTER_SUMBER_VALUE = 'PL-KUMKM 2023'  # Filter rows where Sumber column = this value

# Debug mode - set to True to limit businesses for testing
DEBUG_MODE = False
DEBUG_LIMIT = 100000  # Number of businesses to process in debug mode

# Processing limit configuration
LIMIT_PROCESSING = False  # Set to True to limit the number of businesses to process
PROCESSING_LIMIT = 1500000  # Number of businesses to process when LIMIT_PROCESSING = True (set to None for no limit)

# Duplicate detection settings
SIMILARITY_THRESHOLD = 0.75  # Minimum similarity score (0.0 - 1.0) to consider as similar

# Common words filtering configuration
USE_COMMON_WORDS_FILTERING = True  # Set to False to disable common words filtering in text comparison

# Ignore names configuration
USE_IGNORE_NAMES = True  # Set to True to exclude specific business names from duplicate checking
IGNORE_NAMES_FILE = 'ignore_business_names.csv'  # CSV file containing business names to exclude from duplicate checking

# Duplicate detection rule configuration
DUPLICATE_RULES = {
    'both_high_similarity': 'strong_duplicate',
    'name_high_owner_low': 'not_duplicate',
    'name_low_owner_high': 'not_duplicate',
    'name_high_one_owner_empty': 'weak_duplicate',
    'both_owners_empty_name_high': 'weak_duplicate',
    'default': 'not_duplicate'
}

# Distance calculation configuration
CALCULATE_PRECISE_DISTANCE = True  # Set to True to calculate and store precise distances

# =====================================================================
# DATA MODELS
# =====================================================================

@dataclass
class Business:
    """Business data model"""
    id: str
    name: str
    owner: str
    latitude: float
    longitude: float
    user_id: str
    sls_id: str
    business_type: str  # 'sbr', 'kdm', etc.
    address: str = ""
    project_id: str = ""
    source_row: int = 0
    iddesa: str = ""
    
    def __post_init__(self):
        # Normalize text fields
        self.name = self.name or ""
        self.owner = self.owner or ""
        self.address = self.address or ""
        self.project_id = self.project_id or ""

@dataclass
class DuplicateComparison:
    """Result of duplicate comparison between two businesses"""
    business_a: Business
    business_b: Business
    name_similarity: float
    owner_similarity: float
    duplicate_type: str
    confidence_score: float
    distance_meters: Optional[float] = None

@dataclass
class NearbyBusinessResult:
    """Result for nearby business search with duplicate analysis"""
    source_business: Business
    nearby_businesses: List[Business]
    duplicate_comparisons: List[DuplicateComparison]

# =====================================================================
# BUSINESS DATA UTILITIES
# =====================================================================

def extract_owner_from_name(name: str) -> tuple:
    """
    Extract owner from business name for SBR businesses.
    
    Rules:
    - If name contains <owner> or (owner), extract owner and clean name
    - If no brackets/parentheses, owner is empty
    
    Returns:
        tuple: (cleaned_name, extracted_owner)
    """
    if not name:
        return "", ""
    
    import re
    
    # Look for owner in angle brackets <owner>
    angle_match = re.search(r'<([^>]+)>', name)
    if angle_match:
        owner = angle_match.group(1).strip()
        cleaned_name = re.sub(r'\s*<[^>]+>\s*', ' ', name).strip()
        return cleaned_name, owner
    
    # Look for owner in parentheses (owner)
    paren_match = re.search(r'\(([^)]+)\)', name)
    if paren_match:
        owner = paren_match.group(1).strip()
        cleaned_name = re.sub(r'\s*\([^)]+\)\s*', ' ', name).strip()
        return cleaned_name, owner
    
    # No owner found
    return name.strip(), ""

# =====================================================================
# TEXT NORMALIZATION AND SIMILARITY UTILITIES
# =====================================================================

class CommonWordsManager:
    """Manages common words filtering from CSV file"""
    
    _common_words = None
    
    @classmethod
    def load_common_words(cls) -> set:
        """Load common words from CSV file, with caching"""
        if cls._common_words is not None:
            return cls._common_words
        
        cls._common_words = set()
        common_words_file = os.path.join(os.path.dirname(__file__), 'common_words.csv')
        
        try:
            with open(common_words_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        for word in row:
                            if word.strip():
                                cls._common_words.add(word.strip().lower())
            
            print(f"✓ Loaded {len(cls._common_words)} common words from {common_words_file}")
        except FileNotFoundError:
            print(f"⚠️ Common words file not found: {common_words_file}")
            default_words = ['jual', 'toko', 'warung', 'usaha', 'dagang', 'depot', 'kios', 'stan', 'lapak', 'counter']
            cls._common_words = set(default_words)
        except Exception as e:
            print(f"⚠️ Error loading common words: {e}")
            cls._common_words = {'jual', 'toko', 'warung', 'usaha', 'dagang', 'depot', 'kios', 'stan', 'lapak', 'counter'}
        
        return cls._common_words
    
    @classmethod
    def filter_common_words(cls, text: str) -> str:
        """Remove common words from text, but keep original if result would be empty"""
        if not text:
            return ""
        
        common_words = cls.load_common_words()
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in common_words]
        
        if not filtered_words:
            return text
        
        return ' '.join(filtered_words)

class IgnoreNamesManager:
    """Manages ignore names list from CSV file"""
    
    _ignore_names = None
    
    @classmethod
    def load_ignore_names(cls) -> set:
        """Load ignore names from CSV file, with caching"""
        if cls._ignore_names is not None:
            return cls._ignore_names
        
        cls._ignore_names = set()
        ignore_names_file = os.path.join(os.path.dirname(__file__), IGNORE_NAMES_FILE)
        
        try:
            with open(ignore_names_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row:
                        for name in row:
                            if name.strip():
                                normalized_name = TextUtils.normalize_text(name.strip())
                                if normalized_name:
                                    cls._ignore_names.add(normalized_name)
            
            print(f"✓ Loaded {len(cls._ignore_names)} ignore names from {ignore_names_file}")
        except FileNotFoundError:
            print(f"⚠️ Ignore names file not found: {ignore_names_file}")
            cls._ignore_names = set()
        except Exception as e:
            print(f"⚠️ Error loading ignore names: {e}")
            cls._ignore_names = set()
        
        return cls._ignore_names
    
    @classmethod
    def should_ignore_business(cls, business_name: str) -> bool:
        """Check if a business name should be ignored from duplicate checking"""
        if not USE_IGNORE_NAMES:
            return False
        
        if not business_name:
            return False
        
        ignore_names = cls.load_ignore_names()
        normalized_name = TextUtils.normalize_text(business_name)
        
        return normalized_name in ignore_names

class ExclusionRulesManager:
    """Manages exclusion rules for business name pairs"""
    
    _exclusion_rules = None
    
    @classmethod
    def load_exclusion_rules(cls) -> set:
        """Load exclusion rules from CSV file, with caching"""
        if cls._exclusion_rules is not None:
            return cls._exclusion_rules
        
        cls._exclusion_rules = set()
        exclusion_rules_file = os.path.join(os.path.dirname(__file__), 'exclusion_rules.csv')
        
        try:
            with open(exclusion_rules_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                if reader.fieldnames is None or 'name1' not in reader.fieldnames:
                    print(f"⚠️ Warning: exclusion_rules.csv has no proper headers")
                    return cls._exclusion_rules
                
                for row in reader:
                    name1 = (row.get('name1') or '').strip().lower()
                    name2 = (row.get('name2') or '').strip().lower()
                    
                    if name1 and name2:
                        pair = tuple(sorted([name1, name2]))
                        cls._exclusion_rules.add(pair)
            
            print(f"✓ Loaded {len(cls._exclusion_rules)} exclusion rule pairs")
        except FileNotFoundError:
            print(f"ℹ️ exclusion_rules.csv not found. Skipping exclusion rules.")
        except Exception as e:
            print(f"⚠️ Error loading exclusion rules: {e}")
        
        return cls._exclusion_rules
    
    @classmethod
    def is_excluded_pair(cls, name1: str, name2: str) -> bool:
        """Check if a business name pair matches an exclusion rule"""
        if not name1 or not name2:
            return False
        
        exclusion_rules = cls.load_exclusion_rules()
        normalized_name1 = name1.strip().lower()
        normalized_name2 = name2.strip().lower()
        pair = tuple(sorted([normalized_name1, normalized_name2]))
        
        return pair in exclusion_rules

class TextUtils:
    """Text processing utilities for duplicate detection"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison:
        1. Convert to lowercase
        2. Remove punctuation and extra spaces
        3. Strip leading/trailing whitespace
        """
        if not text:
            return ""
        
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings using SequenceMatcher
        Optionally filters out common words before comparison based on USE_COMMON_WORDS_FILTERING setting
        """
        if USE_COMMON_WORDS_FILTERING:
            return TextUtils.calculate_similarity_with_filtering(text1, text2)
        else:
            return TextUtils.calculate_similarity_without_filtering(text1, text2)
    
    @staticmethod
    def calculate_similarity_without_filtering(text1: str, text2: str) -> float:
        """Calculate similarity between two text strings without common words filtering"""
        if not text1 and not text2:
            return 1.0
        
        if not text1 or not text2:
            return 0.0
        
        norm_text1 = TextUtils.normalize_text(text1)
        norm_text2 = TextUtils.normalize_text(text2)
        
        similarity = difflib.SequenceMatcher(None, norm_text1, norm_text2).ratio()
        
        return similarity
    
    @staticmethod
    def calculate_similarity_with_filtering(text1: str, text2: str) -> float:
        """Calculate similarity between two text strings with common words filtering"""
        if not text1 and not text2:
            return 1.0
        
        if not text1 or not text2:
            return 0.0
        
        norm_text1 = TextUtils.normalize_text(text1)
        norm_text2 = TextUtils.normalize_text(text2)
        
        filtered_text1 = CommonWordsManager.filter_common_words(norm_text1)
        filtered_text2 = CommonWordsManager.filter_common_words(norm_text2)
        
        similarity = difflib.SequenceMatcher(None, filtered_text1, filtered_text2).ratio()
        
        return similarity
    
    @staticmethod
    def is_empty_or_whitespace(text: str) -> bool:
        """Check if text is empty or contains only whitespace"""
        return not text or text.strip() == ""

class DuplicateDetector:
    """Main duplicate detection logic with new algorithm"""
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
    
    def compare_businesses(self, business_a: Business, business_b: Business, 
                          distance_meters: Optional[float] = None) -> DuplicateComparison:
        """
        Compare two businesses and determine if they are duplicates using configurable rules:
        
        Algorithm Rules (configurable via DUPLICATE_RULES):
        1. If name and owner have similarity higher than threshold → Use 'both_high_similarity' rule
        2. If name is high similarity but owner is low similarity → Use 'name_high_owner_low' rule
        3. If name is low similarity but owner is high similarity → Use 'name_low_owner_high' rule
        4. If name is high similarity and one owner is empty → Use 'name_high_one_owner_empty' rule
        5. If name is high similarity and both owners are empty → Use 'both_owners_empty_name_high' rule
        6. All other cases → Use 'default' rule
        """
        
        # Check if either business should be ignored
        if IgnoreNamesManager.should_ignore_business(business_a.name) or \
           IgnoreNamesManager.should_ignore_business(business_b.name):
            return DuplicateComparison(
                business_a=business_a,
                business_b=business_b,
                name_similarity=0.0,
                owner_similarity=0.0,
                duplicate_type='not_duplicate',
                confidence_score=0.0,
                distance_meters=distance_meters
            )
        
        # Check if this business pair is in the exclusion rules
        if ExclusionRulesManager.is_excluded_pair(business_a.name, business_b.name):
            return DuplicateComparison(
                business_a=business_a,
                business_b=business_b,
                name_similarity=0.0,
                owner_similarity=0.0,
                duplicate_type='not_duplicate',
                confidence_score=0.0,
                distance_meters=distance_meters
            )
        
        # Calculate initial similarities (without common words filtering for first pass)
        name_similarity = TextUtils.calculate_similarity_without_filtering(business_a.name, business_b.name)
        owner_similarity = TextUtils.calculate_similarity_without_filtering(business_a.owner, business_b.owner)
        
        # Check if owners are empty
        owner_a_empty = TextUtils.is_empty_or_whitespace(business_a.owner)
        owner_b_empty = TextUtils.is_empty_or_whitespace(business_b.owner)
        both_owners_empty = owner_a_empty and owner_b_empty
        one_owner_empty = (owner_a_empty or owner_b_empty) and not both_owners_empty
        
        # Apply configurable rules from DUPLICATE_RULES
        duplicate_type = DUPLICATE_RULES['default']
        confidence_score = 0.0
        
        if name_similarity >= self.similarity_threshold:
            if both_owners_empty or one_owner_empty:
                # Rule 4: Name is high similarity but owner is empty → advanced step
                # Use common words filtering for advanced comparison
                advanced_name_similarity = TextUtils.calculate_similarity_with_filtering(business_a.name, business_b.name)
                
                if advanced_name_similarity >= self.similarity_threshold:
                    # Still high similarity after filtering common words
                    if both_owners_empty:
                        duplicate_type = DUPLICATE_RULES['both_owners_empty_name_high']
                    else:
                        duplicate_type = DUPLICATE_RULES['name_high_one_owner_empty']
                    confidence_score = advanced_name_similarity
                else:
                    # Similarity dropped below threshold after filtering
                    duplicate_type = DUPLICATE_RULES['default']
                    confidence_score = advanced_name_similarity * 0.5
            elif owner_similarity >= self.similarity_threshold:
                # Rule 1: Name and owner both have high similarity
                duplicate_type = DUPLICATE_RULES['both_high_similarity']
                confidence_score = (name_similarity + owner_similarity) / 2
            else:
                # Rule 2: Name is high similarity but owner is low similarity
                duplicate_type = DUPLICATE_RULES['name_high_owner_low']
                confidence_score = max(name_similarity, owner_similarity) * 0.5
        else:
            if not (owner_a_empty or owner_b_empty) and owner_similarity >= self.similarity_threshold:
                # Rule 3: Name is low similarity but owner is high similarity (both owners not empty)
                duplicate_type = DUPLICATE_RULES['name_low_owner_high']
                confidence_score = max(name_similarity, owner_similarity) * 0.5
            else:
                # Default: All other cases
                duplicate_type = DUPLICATE_RULES['default']
                confidence_score = max(name_similarity, owner_similarity) * 0.3
        
        return DuplicateComparison(
            business_a=business_a,
            business_b=business_b,
            name_similarity=name_similarity,
            owner_similarity=owner_similarity,
            duplicate_type=duplicate_type,
            confidence_score=confidence_score,
            distance_meters=distance_meters
        )

# =====================================================================
# SPATIAL INDEX MANAGER
# =====================================================================

class SpatialIndex:
    """Manages R-tree spatial index for fast geographic queries"""
    
    def __init__(self):
        self.idx = index.Index()
        self.businesses = {}
        self.id_mapping = {}
        self.reverse_mapping = {}
        self.next_index = 0
        
    def insert_business(self, business: Business):
        """Insert a business into the spatial index"""
        if business.id not in self.id_mapping:
            self.id_mapping[business.id] = self.next_index
            self.reverse_mapping[self.next_index] = business.id
            self.next_index += 1
        
        idx = self.id_mapping[business.id]
        self.businesses[business.id] = business
        self.idx.insert(idx, (business.longitude, business.latitude, business.longitude, business.latitude))
    
    def find_nearby(self, business: Business, radius_meters: float) -> List[Business]:
        """Find all businesses within radius of given business"""
        # Convert radius from meters to approximate degrees
        radius_degrees = radius_meters / 111000  # Rough approximation
        
        bbox = (
            business.longitude - radius_degrees,
            business.latitude - radius_degrees,
            business.longitude + radius_degrees,
            business.latitude + radius_degrees
        )
        
        # Get indices from spatial index
        nearby_indices = list(self.idx.intersection(bbox))
        
        # Get business objects
        nearby_businesses = []
        for idx in nearby_indices:
            business_id = self.reverse_mapping[idx]
            biz = self.businesses[business_id]
            
            # Filter to only businesses from different users
            if biz.user_id != business.user_id:
                nearby_businesses.append(biz)
        
        return nearby_businesses

# =====================================================================
# COMBINED DATA MANAGER
# =====================================================================

class CombinedExcelDataManager:
    """Handles loading data from multiple Excel sources (SBR and KDM)"""
    
    def __init__(self):
        self.sbr_folder = EXCEL_SBR_FOLDER
        self.kdm_folder = EXCEL_KDM_FOLDER
        self.result_folder = EXCEL_RESULT_FOLDER
        os.makedirs(self.result_folder, exist_ok=True)
    
    def load_excel_files(self, folder: str) -> pd.DataFrame:
        """Load all Excel files from a folder and combine them"""
        print(f"📁 Loading Excel files from '{folder}' folder...")
        
        # Find all Excel files
        excel_files = glob.glob(os.path.join(folder, '*.xlsx')) + \
                     glob.glob(os.path.join(folder, '*.xls'))
        
        if not excel_files:
            print(f"⚠️ No Excel files found in '{folder}' folder")
            return pd.DataFrame()
        
        print(f"✓ Found {len(excel_files)} Excel file(s):")
        for file in excel_files:
            print(f"  - {os.path.basename(file)}")
        
        # Load and combine all Excel files
        dfs = []
        total_rows = 0
        
        for file in excel_files:
            try:
                print(f"  Loading {os.path.basename(file)}...", end=" ")
                df = pd.read_excel(file)
                dfs.append(df)
                total_rows += len(df)
                print(f"({len(df)} rows)")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
        
        if not dfs:
            print("⚠️ No data loaded from Excel files")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"✓ Combined {len(dfs)} Excel file(s) with {total_rows} total rows")
        
        return combined_df
    
    def load_csv_files(self, folder: str) -> pd.DataFrame:
        """Load all CSV files from a folder and combine them"""
        print(f"📁 Loading CSV files from '{folder}' folder...")
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(folder, '*.csv'))
        
        if not csv_files:
            print(f"⚠️ No CSV files found in '{folder}' folder")
            return pd.DataFrame()
        
        print(f"✓ Found {len(csv_files)} CSV file(s):")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")
        
        # Load and combine all CSV files
        dfs = []
        total_rows = 0
        
        for file in csv_files:
            try:
                print(f"  Loading {os.path.basename(file)}...", end=" ")
                df = pd.read_csv(file)
                dfs.append(df)
                total_rows += len(df)
                print(f"({len(df)} rows)")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
        
        if not dfs:
            print("⚠️ No data loaded from CSV files")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"✓ Combined {len(dfs)} CSV file(s) with {total_rows} total rows")
        
        return combined_df
    
    def get_sbr_businesses(self) -> List[Business]:
        """Load and process SBR Excel data into Business objects"""
        print("\n" + "="*70)
        print("LOADING SBR SOURCE (source/ folder)")
        print("="*70)
        
        # Load Excel files
        df = self.load_excel_files(self.sbr_folder)
        
        if df.empty:
            print("⚠️ No SBR data to process")
            return []
        
        # Check required columns
        required_columns = ['idsbr', 'nama_usaha', 'Sumber Data', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required SBR columns: {missing_columns}")
            print(f"   Available columns: {list(df.columns)}")
            return []
        
        # Filter by Sumber Data value
        print(f"\n🔍 Filtering rows where Sumber Data = '{FILTER_SUMBER_VALUE}'...")
        filtered_df = df[df['Sumber Data'] == FILTER_SUMBER_VALUE].copy()
        
        if filtered_df.empty:
            print(f"⚠️ No rows found with Sumber = '{FILTER_SUMBER_VALUE}'")
            return []
        
        print(f"✓ Found {len(filtered_df)} matching rows")
        
        # Remove rows with missing coordinates
        filtered_df = filtered_df.dropna(subset=['latitude', 'longitude'])
        print(f"✓ {len(filtered_df)} rows have valid coordinates")
        
        # Apply debug limit if enabled
        if DEBUG_MODE:
            filtered_df = filtered_df.iloc[:DEBUG_LIMIT]
            print(f"🐛 DEBUG MODE: Limited to {len(filtered_df)} rows")
        
        # Convert to Business objects
        businesses = []
        for idx, row in filtered_df.iterrows():
            try:
                nama_usaha = str(row['nama_usaha']) if pd.notna(row['nama_usaha']) else ""
                cleaned_name, extracted_owner = extract_owner_from_name(nama_usaha)
                
                lat_str = str(row['latitude']).replace(',', '.')
                lng_str = str(row['longitude']).replace(',', '.')
                
                business = Business(
                    id=str(row['idsbr']),
                    name=cleaned_name,
                    owner=extracted_owner,
                    latitude=float(lat_str),
                    longitude=float(lng_str),
                    user_id=str(row['idsbr']),
                    sls_id=str(row.get('sls_id', '')),
                    business_type='sbr',  # SBR source
                    address=str(row.get('alamat', '')),
                    project_id=str(row['idsbr']),
                    source_row=idx,
                    iddesa=str(row.get('iddesa', ''))
                )
                
                businesses.append(business)
            except Exception as e:
                print(f"⚠️ Error processing SBR row {idx}: {e}")
                continue
        
        print(f"✓ Loaded {len(businesses)} SBR businesses")
        return businesses
    
    def get_kdm_businesses(self) -> List[Business]:
        """Load and process KDM CSV data into Business objects"""
        print("\n" + "="*70)
        print("LOADING KDM SOURCE (source_kdm/ folder)")
        print("="*70)
        
        # Load CSV files
        df = self.load_csv_files(self.kdm_folder)
        
        if df.empty:
            print("⚠️ No KDM data to process")
            return []
        
        # Check required columns for KDM
        required_columns = ['id', 'Nama_Usaha', 'Pemilik', 'Latitude', 'Longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required KDM columns: {missing_columns}")
            print(f"   Available columns: {list(df.columns)}")
            return []
        
        # No filtering for KDM source - use all rows
        print(f"\n✓ Using all {len(df)} rows (no filtering)")
        
        # Remove rows with missing coordinates
        df = df.dropna(subset=['Latitude', 'Longitude'])
        print(f"✓ {len(df)} rows have valid coordinates")
        
        # Apply debug limit if enabled
        if DEBUG_MODE:
            df = df.iloc[:DEBUG_LIMIT]
            print(f"🐛 DEBUG MODE: Limited to {len(df)} rows")
        
        # Convert to Business objects
        businesses = []
        for idx, row in df.iterrows():
            try:
                nama_usaha = str(row['Nama_Usaha']) if pd.notna(row['Nama_Usaha']) else ""
                pemilik = str(row['Pemilik']) if pd.notna(row['Pemilik']) else ""
                
                lat_str = str(row['Latitude']).replace(',', '.')
                lng_str = str(row['Longitude']).replace(',', '.')
                
                # Use id column from CSV as business ID
                business_id = str(row['id'])
                
                # Extract SLS ID from Kabupaten, Kecamatan, Desa, SLS columns
                # Format: [code]text -> extract code from brackets
                sls_id = ""
                try:
                    
                    # Extract codes from each column
                    kabupaten_code = ""
                    kecamatan_code = ""
                    desa_code = ""
                    sls_code = ""
                    
                    if pd.notna(row.get('Kabupaten')):
                        match = re.search(r'\[(\d+)\]', str(row['Kabupaten']))
                        if match:
                            kabupaten_code = match.group(1)
                    
                    if pd.notna(row.get('Kecamatan')):
                        match = re.search(r'\[(\d+)\]', str(row['Kecamatan']))
                        if match:
                            kecamatan_code = match.group(1)
                    
                    if pd.notna(row.get('Desa')):
                        match = re.search(r'\[(\d+)\]', str(row['Desa']))
                        if match:
                            desa_code = match.group(1)
                    
                    if pd.notna(row.get('SLS')):
                        match = re.search(r'\[(\d+)\]', str(row['SLS']))
                        if match:
                            sls_code = match.group(1)
                    
                    # Combine codes: Kabupaten + Kecamatan + Desa + SLS
                    sls_id = kabupaten_code + kecamatan_code + desa_code + sls_code
                except Exception as e:
                    print(f"⚠️ Error extracting SLS ID from row {idx}: {e}")
                    sls_id = ""
                
                business = Business(
                    id=business_id,
                    name=nama_usaha,
                    owner=pemilik,
                    latitude=float(lat_str),
                    longitude=float(lng_str),
                    user_id=pemilik,  # Use owner as user_id for KDM
                    sls_id=sls_id,  # Combined code from Kabupaten + Kecamatan + Desa + SLS
                    business_type='kdm',  # KDM source
                    address=str(row.get('Alamat', '')),
                    project_id=business_id,
                    source_row=idx,
                    iddesa=sls_id  # Use sls_id for iddesa
                )
                
                businesses.append(business)
            except Exception as e:
                print(f"⚠️ Error processing KDM row {idx}: {e}")
                continue
        
        print(f"✓ Loaded {len(businesses)} KDM businesses")
        return businesses
    
    def get_all_businesses(self) -> Tuple[List[Business], List[Business]]:
        """Load all businesses from both sources"""
        sbr_businesses = self.get_sbr_businesses()
        kdm_businesses = self.get_kdm_businesses()
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"SBR businesses: {len(sbr_businesses)}")
        print(f"KDM businesses: {len(kdm_businesses)}")
        print(f"Total businesses: {len(sbr_businesses) + len(kdm_businesses)}")
        print("="*70)
        
        return sbr_businesses, kdm_businesses
    
    def save_results_to_excel(self, results: List[Dict[str, Any]]) -> str:
        """Save results to Excel file"""
        output_path = os.path.join(self.result_folder, EXCEL_RESULT_FILENAME)
        
        if not results:
            print(f"⚠️ No results to save")
            return output_path
        
        print(f"💾 Preparing to save {len(results)} results to Excel...")
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Create result folder if it doesn't exist
        os.makedirs(self.result_folder, exist_ok=True)
        
        # Save to Excel
        try:
            print(f"📝 Writing to file: {os.path.abspath(output_path)}")
            df_results.to_excel(output_path, index=False, engine='openpyxl')
            
            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✅ Results saved successfully to: {output_path}")
                print(f"   File size: {file_size:,} bytes")
                return output_path
            else:
                print(f"❌ Error: File was not created at {output_path}")
                return output_path
        except Exception as e:
            print(f"❌ Error saving results to Excel: {e}")
            import traceback
            traceback.print_exc()
            return output_path

# =====================================================================
# MAIN FINDER ENGINE
# =====================================================================

class NearbyBusinessFinder:
    """Main engine for finding nearby businesses and detecting duplicates"""
    
    def __init__(self, radius_meters: float = RADIUS_METERS, 
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.radius_meters = radius_meters
        self.data_manager = CombinedExcelDataManager()
        self.spatial_index = SpatialIndex()
        self.duplicate_detector = DuplicateDetector(similarity_threshold)
    
    def run_search(self):
        """Execute the complete nearby business search"""
        start_time = time.time()
        
        try:
            print("🔍 Starting Fast Nearby Business Search with Combined Sources (SBR + KDM)")
            print("=" * 60)
            print(f"Configuration:")
            print(f"  - SBR source folder: {EXCEL_SBR_FOLDER}")
            print(f"  - KDM source folder: {EXCEL_KDM_FOLDER}")
            print(f"  - Result file: {os.path.join(EXCEL_RESULT_FOLDER, EXCEL_RESULT_FILENAME)}")
            print(f"  - SBR filter value: Sumber = '{FILTER_SUMBER_VALUE}'")
            print(f"  - Search radius: {self.radius_meters} meters")
            print("-" * 60)
            
            # Load businesses from both sources
            print("📊 Loading data from both sources...")
            sbr_businesses, kdm_businesses = self.data_manager.get_all_businesses()
            all_businesses = sbr_businesses + kdm_businesses
            
            if not all_businesses:
                print("⚠️ No businesses found. Exiting.")
                return
            
            print(f"✓ Total businesses loaded: {len(all_businesses):,}")
            
            # Apply processing limit if enabled
            original_count = len(all_businesses)
            if LIMIT_PROCESSING and PROCESSING_LIMIT is not None and len(all_businesses) > PROCESSING_LIMIT:
                all_businesses = all_businesses[:PROCESSING_LIMIT]
                print(f"⚠️  Processing limit applied: {len(all_businesses):,} of {original_count:,} businesses will be processed")
            elif LIMIT_PROCESSING:
                print(f"✓ Processing limit enabled but not reached: processing all {len(all_businesses):,} businesses")
            
            # Build spatial index with all businesses
            print("🏗️ Building spatial index with all businesses...")
            for business in all_businesses:
                self.spatial_index.insert_business(business)
            print("✓ Spatial index built successfully")
            
            # Find nearby businesses and detect duplicates for each business
            print(f"\n🔍 Searching for nearby businesses and detecting duplicates...")
            search_start_time = time.time()
            
            total_matches = 0
            businesses_with_matches = 0
            total_duplicates = {'strong': 0, 'weak': 0, 'not_duplicate': 0}
            unique_businesses_with_duplicates = set()
            validation_data = []
            seen_pairs = set()
            
            for i, business in enumerate(all_businesses):
                if i % 100 == 0:
                    elapsed = time.time() - search_start_time
                    print(f"  Progress: {i:,}/{len(all_businesses):,} businesses processed ({elapsed:.1f}s)")
                
                # Find nearby businesses
                nearby_businesses = self.spatial_index.find_nearby(
                    business, self.radius_meters
                )
                
                if nearby_businesses:
                    businesses_with_matches += 1
                    total_matches += len(nearby_businesses)
                    
                    # Perform duplicate detection for each nearby business
                    duplicate_comparisons = []
                    
                    for nearby_business in nearby_businesses:
                        # Skip comparison if both businesses are from KDM source
                        if business.business_type == 'kdm' and nearby_business.business_type == 'kdm':
                            continue
                        
                        # Create a pair identifier to avoid duplicate comparisons
                        pair_id = tuple(sorted([business.id, nearby_business.id]))
                        
                        # Calculate precise distance for comparison if enabled
                        distance = calculate_precise_distance(
                            business.latitude, business.longitude,
                            nearby_business.latitude, nearby_business.longitude
                        ) if CALCULATE_PRECISE_DISTANCE else None
                        
                        # Detect duplicates
                        comparison = self.duplicate_detector.compare_businesses(
                            business, nearby_business, distance
                        )
                        duplicate_comparisons.append(comparison)
                        
                        # Count duplicate types
                        if comparison.duplicate_type == 'strong_duplicate':
                            total_duplicates['strong'] += 1
                            unique_businesses_with_duplicates.add(business.id)
                            unique_businesses_with_duplicates.add(nearby_business.id)
                        elif comparison.duplicate_type == 'weak_duplicate':
                            total_duplicates['weak'] += 1
                            unique_businesses_with_duplicates.add(business.id)
                            unique_businesses_with_duplicates.add(nearby_business.id)
                        else:
                            total_duplicates['not_duplicate'] += 1
                    
                    # Add detailed results to export list (only duplicates)
                    for comparison in duplicate_comparisons:
                        if comparison.duplicate_type in ['strong_duplicate', 'weak_duplicate']:
                            pair_key = tuple(sorted([comparison.business_a.id, comparison.business_b.id]))
                            
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                validation_data.append({
                                    'id_center': comparison.business_a.id,
                                    'id_nearby': comparison.business_b.id,
                                    'business_type_center': comparison.business_a.business_type,
                                    'business_type_nearby': comparison.business_b.business_type,
                                    'iddesa_center': comparison.business_a.iddesa,
                                    'iddesa_nearby': comparison.business_b.iddesa,
                                    'nama_usaha_center': comparison.business_a.name,
                                    'nama_usaha_nearby': comparison.business_b.name,
                                    'pemilik_center': comparison.business_a.owner,
                                    'pemilik_nearby': comparison.business_b.owner,
                                    'name_similarity': round(comparison.name_similarity, 3),
                                    'owner_similarity': round(comparison.owner_similarity, 3),
                                    'duplicate_type': comparison.duplicate_type,
                                    'confidence_score': round(comparison.confidence_score, 3),
                                    'distance_meters': comparison.distance_meters,
                                    'latitude_center': comparison.business_a.latitude,
                                    'longitude_center': comparison.business_a.longitude,
                                    'latitude_nearby': comparison.business_b.latitude,
                                    'longitude_nearby': comparison.business_b.longitude
                                })
                    
                    # Print results with duplicate information
                    strong_dupes = sum(1 for c in duplicate_comparisons if c.duplicate_type == 'strong_duplicate')
                    weak_dupes = sum(1 for c in duplicate_comparisons if c.duplicate_type == 'weak_duplicate')
                    
                    if strong_dupes > 0 or weak_dupes > 0:
                        progress_pct = ((i + 1) / len(all_businesses)) * 100
                        print(f"📍 [{progress_pct:.1f}%] {business.name} ({business.business_type}) → 🔴 {strong_dupes} strong, 🟡 {weak_dupes} weak duplicates")
            
            search_end_time = time.time()
            
            # Save results to Excel
            if validation_data:
                self.data_manager.save_results_to_excel(validation_data)
            
            total_end_time = time.time()
            
            print(f"\n" + "=" * 60)
            print("✅ Duplicate Detection Search completed successfully!")
            print(f"📊 Summary:")
            print(f"  - Total businesses analyzed: {len(all_businesses):,}")
            print(f"  - SBR businesses: {len(sbr_businesses):,}")
            print(f"  - KDM businesses: {len(kdm_businesses):,}")
            print(f"  - Businesses with nearby matches: {businesses_with_matches:,}")
            print(f"  - Total nearby business pairs found: {total_matches:,}")
            print(f"  - Average matches per business: {total_matches / len(all_businesses):.2f}" if len(all_businesses) > 0 else "  - Average matches per business: 0")
            
            print(f"\n🔍 Duplicate Detection Results:")
            print(f"  - Strong duplicate pairs found: {total_duplicates['strong']:,}")
            print(f"  - Weak duplicate pairs found: {total_duplicates['weak']:,}")
            print(f"  - Not duplicate pairs: {total_duplicates['not_duplicate']:,}")
            print(f"  - Total comparison pairs: {total_matches:,}")
            print(f"  - Unique businesses with duplicates: {len(unique_businesses_with_duplicates):,}")
            print(f"  - Business duplicate rate: {len(unique_businesses_with_duplicates) / len(all_businesses) * 100:.1f}% ({len(unique_businesses_with_duplicates)} of {len(all_businesses)} businesses)" if len(all_businesses) > 0 else "  - Business duplicate rate: 0%")
            if total_matches > 0:
                print(f"  - Pair duplicate rate: {(total_duplicates['strong'] + total_duplicates['weak']) / total_matches * 100:.1f}% (pairs that are duplicates)")
            
            print(f"\n📊 Output:")
            print(f"  - Results saved to: {os.path.join(EXCEL_RESULT_FOLDER, EXCEL_RESULT_FILENAME)}")
            print(f"  - Total rows in result: {len(validation_data):,}")
            
            print(f"\n⏱️  Performance:")
            print(f"  - Total execution time: {total_end_time - start_time:.1f} seconds")
            print(f"  - Search phase time: {search_end_time - search_start_time:.1f} seconds")
            print(f"  - Businesses processed per second: {len(all_businesses) / (search_end_time - search_start_time):.0f}" if (search_end_time - search_start_time) > 0 and len(all_businesses) > 0 else "  - Businesses processed per second: 0")
            
        except Exception as e:
            print(f"\n❌ Error during search: {e}")
            import traceback
            traceback.print_exc()
            raise

def calculate_precise_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate precise distance between two points in meters using Haversine formula
    """
    import math
    
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    r = 6371000
    
    return c * r

# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main function"""
    try:
        print("🚀 Business Duplicate Detector (Combined Sources: SBR + KDM)")
        print(f"🎯 Searching for businesses within {RADIUS_METERS}m radius")
        print(f"🔍 Detecting duplicates with {SIMILARITY_THRESHOLD} similarity threshold")
        print(f"⚡ Using R-tree spatial indexing for performance")
        
        # Display common words configuration
        if USE_COMMON_WORDS_FILTERING:
            common_words = CommonWordsManager.load_common_words()
            print(f"📝 Common words filtering enabled ({len(common_words)} words loaded)")
        else:
            print(f"📝 Common words filtering disabled (using full text comparison)")
        
        # Display ignore names configuration
        if USE_IGNORE_NAMES:
            ignore_names = IgnoreNamesManager.load_ignore_names()
            if len(ignore_names) > 0:
                print(f"🚫 Ignore names enabled ({len(ignore_names)} business names will be excluded)")
            else:
                print(f"🚫 Ignore names enabled but no names loaded (all businesses will be checked)")
        else:
            print(f"🚫 Ignore names disabled (all businesses will be checked)")
        
        if DEBUG_MODE:
            print(f"🐛 DEBUG MODE: Processing only {DEBUG_LIMIT:,} businesses for testing")
        
        # Display current algorithm configuration
        print(f"\n📋 Duplicate Detection Algorithm:")
        print(f"  1. Name & owner both high similarity → strong_duplicate")
        print(f"  2. Name high, owner low similarity → not_duplicate")
        print(f"  3. Name low, owner high similarity → not_duplicate")
        print(f"  4. Name high, owner empty → advanced step (common words filtering)")
        print(f"  5. Name low, owner empty → not_duplicate")
        print(f"  📝 Similarity threshold: {SIMILARITY_THRESHOLD}")
        
        # Display distance calculation configuration
        if CALCULATE_PRECISE_DISTANCE:
            print(f"  📏 Precise distance calculation: ✅ (slower but more accurate)")
        else:
            print(f"  📏 Precise distance calculation: ❌ (faster, using spatial approximation)")
        
        # Display processing limit configuration
        if LIMIT_PROCESSING:
            if PROCESSING_LIMIT is not None:
                print(f"  🔢 Processing limit: ✅ Limited to {PROCESSING_LIMIT:,} businesses")
            else:
                print(f"  🔢 Processing limit: ✅ Enabled but no limit set (will process all)")
        else:
            print(f"  🔢 Processing limit: ❌ Disabled (will process all businesses)")
        
        print(f"\n📋 Data Sources:")
        print(f"  📂 SBR (source folder): Excel files with Sumber = '{FILTER_SUMBER_VALUE}'")
        print(f"  📂 KDM (source_kdm folder): CSV files with no filtering")
        
        print(f"\n📋 Output Configuration:")
        print(f"  💾 Save to Excel file: ✅")
        print(f"  📄 Output file: {os.path.join(EXCEL_RESULT_FOLDER, EXCEL_RESULT_FILENAME)}")
        print(f"  📊 Content: All comparison results (duplicates)")
        
        print("")
        
        finder = NearbyBusinessFinder(RADIUS_METERS, SIMILARITY_THRESHOLD)
        finder.run_search()
        
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
