#!/usr/bin/env python3
"""
Business Duplicate Detector using Spatial Search

This script efficiently finds businesses within a specified radius using R-tree spatial indexing
and then analyzes them for potential duplicates using text similarity comparison with configurable rules.

It loads all businesses from a CSV file and for each business,
finds all other businesses within a configured radius that belong to different users, then compares them
to detect potential duplicate businesses using customizable detection rules.

Usage:
- Set SLS_ID_FILTER = "123456" to filter businesses by specific SLS ID
- Set SLS_ID_FILTER = None to process all businesses (default)
- Set USE_COMMON_WORDS_FILTERING = True to filter common words during comparison (default)
- Set USE_COMMON_WORDS_FILTERING = False to use full text comparison without filtering
- Customize common words in 'common_words.csv' file to improve comparison accuracy
- Set USE_IGNORE_NAMES = True to exclude specific business names from duplicate checking
- Add business names to 'ignore_business_names.csv' file (one name per row) to exclude them

Features:
- R-tree spatial indexing for O(log n) spatial queries instead of O(n²)
- Support for both supplement and market business tables
- Configurable radius (default: 50 meters)
- Text similarity analysis for duplicate detection
- String normalization for better comparison accuracy
- **Common words filtering** - ignores common business words (jual, toko, warung, etc.) during comparison
- **Ignore names list** - excludes specific business names from duplicate checking entirely
- Configurable similarity thresholds
- Classification of duplicates (Strong, Weak, Not duplicate)

Duplicate Detection Algorithm:
The script uses a refined algorithm for detecting business duplicates:
1. If name and owner have similarity higher than threshold → strong_duplicate
2. If name is high similarity but owner is low similarity → not_duplicate
3. If name is low similarity but owner is high similarity → not_duplicate
4. If name is high similarity but owner is empty → advanced step (common words filtering applied)
5. If name is low similarity but owner is empty → not_duplicate

Advanced Step (Rule 4):
When names have high similarity but owner information is missing, the system applies common words
filtering to remove generic business terms (toko, warung, jual, etc.) and re-evaluates similarity.
This helps distinguish between truly similar businesses and those that only share common prefixes.

Configurable Rules:
You can customize how the following conditions are classified:
1. Both name & owner similarity >= threshold → Configurable result
2. Name similarity >= threshold but owner similarity < threshold → Configurable result  
3. Name similarity < threshold but owner similarity >= threshold → Configurable result
4. Name similarity >= threshold and one owner empty → Configurable result
5. Both owners empty and name similarity >= threshold → Configurable result
6. All other cases → Configurable result

The current implementation uses a hardcoded balanced approach for duplicate detection.

Requirements:
    - rtree
    - shapely
    - difflib (built-in)
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
from rtree import index

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

# CSV input configuration
CSV_INPUT_FILE = os.path.join(os.path.dirname(__file__), 'result', 'match_sbr_kdm.csv')

# SLS ID Filter - set to specific SLS ID to filter businesses, or None to process all
SLS_ID_FILTER = None  # Example: "123456" to filter by specific SLS ID

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
    # Rule 1: Name similarity >= TH AND Owner similarity >= TH
    'both_high_similarity': 'strong_duplicate',  # Options: 'strong_duplicate', 'weak_duplicate', 'not_duplicate'
    
    # Rule 2: Name similarity >= TH but Owner similarity < TH (owner not empty)
    'name_high_owner_low': 'not_duplicate',  # Options: 'strong_duplicate', 'weak_duplicate', 'not_duplicate'
    
    # Rule 3: Name similarity < TH but Owner similarity >= TH (owner not empty)
    'name_low_owner_high': 'not_duplicate',  # Options: 'strong_duplicate', 'weak_duplicate', 'not_duplicate'
    
    # Rule 4: Name similarity >= TH and one owner empty
    'name_high_one_owner_empty': 'weak_duplicate',  # Options: 'strong_duplicate', 'weak_duplicate', 'not_duplicate'
    
    # Rule 5: Both owners empty, name similarity >= TH
    'both_owners_empty_name_high': 'weak_duplicate',  # Options: 'strong_duplicate', 'weak_duplicate', 'not_duplicate'
    
    # Default rule: All other cases
    'default': 'not_duplicate'  # Options: 'strong_duplicate', 'weak_duplicate', 'not_duplicate'
}

# Distance calculation configuration
CALCULATE_PRECISE_DISTANCE = True  # Set to True to calculate and store precise distances (slower but more accurate)

# Output mode configuration - choose where to save results
SAVE_RESULTS_TO_FILE = True  # Set to True to save detailed results to CSV file for manual inspection

# Output file configuration (only used when SAVE_RESULTS_TO_FILE = True)
# Writes into the workspace `result/` folder.
OUTPUT_FILENAME = "duplicate_detection_all.csv"  # Constant filename (overwrites previous results)
USE_TIMESTAMP_IN_FILENAME = False  # Set to True to append timestamp to filename
INCLUDE_NOT_DUPLICATES_IN_OUTPUT = False  # Set to True to include "not_duplicate" results in CSV output

# =====================================================================
# DATA MODELS
# =====================================================================

@dataclass
class Business:
    """Business data model"""
    id: str  # Changed from int to str for UUID support
    name: str
    owner: str
    latitude: float
    longitude: float
    user_id: str  # Changed from int to str for UUID support
    area_id: str  # Area ID for the business
    business_type: str  # 'supplement' or 'market'
    project_id: str = ""  # Project ID for supplement businesses
    
    def __post_init__(self):
        # Normalize text fields
        self.name = self.name or ""
        self.owner = self.owner or ""
        self.project_id = self.project_id or ""


class CSVBusinessLoader:
    """Loads businesses from a CSV file."""

    REQUIRED_COLUMNS = {
        'idsbr',
        'nama_usaha',
        'kode_wilayah',
        'latitude',
        'longitude',
        'sumber_data',
        'is_coordinate_valid',
        'idkendedes',
        'sls_id',
        'owner',
        'user_id',
        'project_id',
        'type',
    }

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @classmethod
    def load_businesses(cls, csv_path: str) -> List[Business]:
        """Load businesses from CSV using the requested mapping and rules."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV input file not found: {csv_path}")

        businesses: List[Business] = []
        skipped_empty_idsbr = 0
        kdm_rows = 0
        sbr_rows = 0
        skipped_missing_coords = 0
        skipped_invalid_coords = 0

        print(f"📄 Loading businesses from CSV: {csv_path}")
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")

            header = {h.strip() for h in reader.fieldnames if h}
            missing = cls.REQUIRED_COLUMNS - header
            if missing:
                raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

            for row in reader:
                # Business.id comes from idsbr
                raw_idsbr = (row.get('idsbr') or '').strip()
                if raw_idsbr == "":
                    skipped_empty_idsbr += 1
                    continue

                raw_idkendedes = (row.get('idkendedes') or '').strip()

                lat = cls._to_float(row.get('latitude'))
                lng = cls._to_float(row.get('longitude'))
                if lat is None or lng is None:
                    skipped_missing_coords += 1
                    continue
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
                    skipped_invalid_coords += 1
                    continue

                # Conditional mapping:
                # - If idkendedes exists => use KDM-style mapping (owner/user_id/project_id/type from CSV)
                # - If not => use SBR-style mapping (owner/user_id/project_id empty, business_type='sbr')
                if raw_idkendedes:
                    kdm_rows += 1
                    name_value = (row.get('name') or '').strip() or (row.get('nama_usaha') or '').strip()
                    business_type = str((row.get('type') or '')).strip()
                    owner_value = (row.get('owner') or '').strip()
                    if business_type in {"market", "sbr"}:
                        cleaned_name, extracted_owner = extract_owner_from_name(name_value)
                        name_value = cleaned_name
                        if extracted_owner:
                            owner_value = extracted_owner

                    businesses.append(
                        Business(
                            id=raw_idsbr,
                            name=name_value,
                            owner=owner_value,
                            latitude=lat,
                            longitude=lng,
                            user_id=str((row.get('user_id') or '')).strip(),
                            project_id=str((row.get('project_id') or '')).strip(),
                            area_id=str((row.get('sls_id') or '')).strip(),
                            business_type=business_type,
                        )
                    )
                else:
                    sbr_rows += 1
                    name_value = (row.get('nama_usaha') or '').strip()
                    cleaned_name, extracted_owner = extract_owner_from_name(name_value)
                    businesses.append(
                        Business(
                            id=raw_idsbr,
                            name=cleaned_name,
                            owner=extracted_owner,
                            latitude=lat,
                            longitude=lng,
                            user_id="",
                            project_id="",
                            area_id=str((row.get('kode_wilayah') or '')).strip(),
                            business_type="sbr",
                        )
                    )

        print(f"✓ Loaded {len(businesses):,} businesses from CSV")
        print(f"  - KDM-mapped rows (idkendedes present): {kdm_rows:,}")
        print(f"  - SBR-mapped rows (idkendedes empty): {sbr_rows:,}")
        if skipped_empty_idsbr:
            print(f"  - Skipped {skipped_empty_idsbr:,} rows with empty idsbr")
        if skipped_missing_coords:
            print(f"  - Skipped {skipped_missing_coords:,} rows with missing latitude/longitude")
        if skipped_invalid_coords:
            print(f"  - Skipped {skipped_invalid_coords:,} rows with invalid coordinates")

        return businesses

@dataclass
class DuplicateComparison:
    """Result of duplicate comparison between two businesses"""
    business_a: Business
    business_b: Business
    name_similarity: float
    owner_similarity: float
    duplicate_type: str  # "strong_duplicate", "weak_duplicate", "not_duplicate"
    confidence_score: float
    distance_meters: Optional[float] = None

@dataclass
class NearbyBusinessResult:
    """Result for nearby business search with duplicate analysis"""
    source_business: Business
    nearby_businesses: List[Business]
    duplicate_comparisons: List[DuplicateComparison]

# =====================================================================
# GEOGRAPHIC UTILITIES
# =====================================================================

class GeoUtils:
    """Geographic utility functions"""
    
    @staticmethod
    def meters_to_degrees_approx(lat: float, meters: float) -> float:
        """
        Approximate conversion from meters to degrees at given latitude
        Used for creating bounding boxes for R-tree queries
        """
        # At equator: 1 degree ≈ 111,320 meters
        lat_rad = lat * 3.14159 / 180
        meters_per_degree_lat = 111320
        meters_per_degree_lng = 111320 * abs(cos(lat_rad))
        
        # Use the smaller value to ensure we capture all points within radius
        return meters / min(meters_per_degree_lat, meters_per_degree_lng)

def cos(x):
    """Simple cosine approximation"""
    import math
    return math.cos(x)

# =====================================================================
# BUSINESS DATA UTILITIES
# =====================================================================

def extract_owner_from_name(name: str) -> tuple[str, str]:
    """
    Extract owner from business name for market businesses.
    
    Rules:
    - If name contains <owner> or (owner), extract owner and clean name
    - If no brackets/parentheses, owner is empty
    
    Examples:
    - "toko sembako <budi>" -> ("toko sembako", "budi")
    - "warung makan (siti)" -> ("warung makan", "siti")
    - "toko abc" -> ("toko abc", "")
    
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


def make_unordered_pair_key(id1: str, id2: str) -> frozenset[str]:
    """Return an order-insensitive key so (A,B) == (B,A)."""
    return frozenset((id1, id2))

# =====================================================================
# TEXT NORMALIZATION AND SIMILARITY UTILITIES
# =====================================================================

class CommonWordsManager:
    """Manages common words filtering from CSV file"""
    
    _common_words = None  # Class variable to cache loaded words
    
    @classmethod
    def load_common_words(cls) -> set:
        """Load common words from CSV file, with caching"""
        if cls._common_words is not None:
            return cls._common_words
        
        cls._common_words = set()
        common_words_file = os.path.join(os.path.dirname(__file__), 'common_words.csv')
        
        try:
            import csv
            with open(common_words_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:  # Skip empty rows
                        # Each row might contain multiple words, or just one word per row
                        for word in row:
                            if word.strip():  # Skip empty cells
                                cls._common_words.add(word.strip().lower())
            
            print(f"✓ Loaded {len(cls._common_words)} common words from {common_words_file}")
        except FileNotFoundError:
            print(f"⚠️ Common words file not found: {common_words_file}")
            print("   Creating default common words...")
            # Create default common words file
            default_words = ['jual', 'toko', 'warung', 'usaha', 'dagang', 'depot', 'kios', 'stan', 'lapak', 'counter']
            cls._create_default_common_words_file(common_words_file, default_words)
            cls._common_words = set(default_words)
        except Exception as e:
            print(f"⚠️ Error loading common words: {e}")
            print("   Using default common words...")
            cls._common_words = {'jual', 'toko', 'warung', 'usaha', 'dagang', 'depot', 'kios', 'stan', 'lapak', 'counter'}
        
        return cls._common_words
    
    @classmethod
    def _create_default_common_words_file(cls, filepath: str, words: List[str]):
        """Create default common words CSV file"""
        try:
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['common_word'])  # Header
                for word in words:
                    writer.writerow([word])
            print(f"✓ Created default common words file: {filepath}")
        except Exception as e:
            print(f"⚠️ Failed to create default common words file: {e}")
    
    @classmethod
    def filter_common_words(cls, text: str) -> str:
        """
        Remove common words from text, but keep original if result would be empty
        
        Args:
            text: Input text to filter
            
        Returns:
            Filtered text, or original text if filtering would result in empty string
        """
        if not text:
            return ""
        
        common_words = cls.load_common_words()
        
        # Split text into words
        words = text.split()
        
        # Filter out common words
        filtered_words = [word for word in words if word.lower() not in common_words]
        
        # If all words were common words, return original text (fallback)
        if not filtered_words:
            return text
        
        # Return filtered text
        return ' '.join(filtered_words)

class IgnoreNamesManager:
    """Manages ignore names list from CSV file"""
    
    _ignore_names = None  # Class variable to cache loaded names
    
    @classmethod
    def load_ignore_names(cls) -> set:
        """Load ignore names from CSV file, with caching"""
        if cls._ignore_names is not None:
            return cls._ignore_names
        
        cls._ignore_names = set()
        ignore_names_file = os.path.join(os.path.dirname(__file__), IGNORE_NAMES_FILE)
        
        try:
            import csv
            with open(ignore_names_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header row if exists
                for row in reader:
                    if row:  # Skip empty rows
                        # Each row might contain multiple names, or just one name per row
                        for name in row:
                            if name.strip():  # Skip empty cells
                                # Store normalized names for comparison
                                normalized_name = TextUtils.normalize_text(name.strip())
                                if normalized_name:
                                    cls._ignore_names.add(normalized_name)
            
            print(f"✓ Loaded {len(cls._ignore_names)} ignore names from {ignore_names_file}")
        except FileNotFoundError:
            print(f"⚠️ Ignore names file not found: {ignore_names_file}")
            print("   No business names will be excluded from duplicate checking")
            cls._ignore_names = set()
        except Exception as e:
            print(f"⚠️ Error loading ignore names: {e}")
            print("   No business names will be excluded from duplicate checking")
            cls._ignore_names = set()
        
        return cls._ignore_names
    
    @classmethod
    def should_ignore_business(cls, business_name: str) -> bool:
        """
        Check if a business name should be ignored from duplicate checking
        
        Args:
            business_name: Business name to check
            
        Returns:
            True if the business should be ignored, False otherwise
        """
        if not USE_IGNORE_NAMES:
            return False
        
        if not business_name:
            return False
        
        ignore_names = cls.load_ignore_names()
        
        # Normalize the business name for comparison
        normalized_name = TextUtils.normalize_text(business_name)
        
        # Check if normalized name is in the ignore list
        return normalized_name in ignore_names

class ExclusionRulesManager:
    """Manages exclusion rules for business name pairs that should never be marked as duplicates"""
    
    _exclusion_rules = None  # Class variable to cache loaded rules
    
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
                    print(f"⚠️ Warning: exclusion_rules.csv has no proper headers (expected 'name1', 'name2')")
                    return cls._exclusion_rules
                
                for row in reader:
                    name1 = (row.get('name1') or '').strip().lower()
                    name2 = (row.get('name2') or '').strip().lower()
                    
                    if name1 and name2:
                        # Store pairs as sorted tuples to handle both directions (A,B) and (B,A)
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
        """
        Check if a business name pair matches an exclusion rule
        
        Args:
            name1: First business name
            name2: Second business name
            
        Returns:
            bool: True if the pair matches an exclusion rule, False otherwise
        """
        if not name1 or not name2:
            return False
        
        exclusion_rules = cls.load_exclusion_rules()
        
        # Normalize names for comparison
        normalized_name1 = name1.strip().lower()
        normalized_name2 = name2.strip().lower()
        
        # Create sorted pair to handle both directions
        pair = tuple(sorted([normalized_name1, normalized_name2]))
        
        # Check if this pair is in the exclusion rules
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
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Replace multiple spaces with single space and strip
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings using SequenceMatcher
        Optionally filters out common words before comparison based on USE_COMMON_WORDS_FILTERING setting
        Returns a float between 0.0 (no similarity) and 1.0 (identical)
        """
        if USE_COMMON_WORDS_FILTERING:
            return TextUtils.calculate_similarity_with_filtering(text1, text2)
        else:
            return TextUtils.calculate_similarity_without_filtering(text1, text2)
    
    @staticmethod
    def calculate_similarity_without_filtering(text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings without common words filtering
        Returns a float between 0.0 (no similarity) and 1.0 (identical)
        """
        if not text1 and not text2:
            return 1.0  # Both empty strings are considered identical
        
        if not text1 or not text2:
            return 0.0  # One empty, one not empty
        
        # Normalize both texts
        norm_text1 = TextUtils.normalize_text(text1)
        norm_text2 = TextUtils.normalize_text(text2)
        
        # Use difflib.SequenceMatcher for similarity calculation
        similarity = difflib.SequenceMatcher(None, norm_text1, norm_text2).ratio()
        
        return similarity
    
    @staticmethod
    def calculate_similarity_with_filtering(text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings with common words filtering
        Returns a float between 0.0 (no similarity) and 1.0 (identical)
        """
        if not text1 and not text2:
            return 1.0  # Both empty strings are considered identical
        
        if not text1 or not text2:
            return 0.0  # One empty, one not empty
        
        # Normalize both texts
        norm_text1 = TextUtils.normalize_text(text1)
        norm_text2 = TextUtils.normalize_text(text2)
        
        # Filter out common words
        filtered_text1 = CommonWordsManager.filter_common_words(norm_text1)
        filtered_text2 = CommonWordsManager.filter_common_words(norm_text2)
        
        # Use difflib.SequenceMatcher for similarity calculation
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
        
        Note: If either business name is in the ignore list, returns 'not_duplicate' immediately
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
        
        # Check if this business pair is in the exclusion rules (e.g., kost putra vs kost putri)
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
# VALIDATION UTILITIES
# =====================================================================

def calculate_precise_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate precise distance between two points in meters using Haversine formula
    Used for validation only
    """
    import math
    
    # Convert decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    
    return c * r

def save_results_to_csv(results: List[Dict[str, Any]], filename: str):
    """Save results to CSV file for manual inspection"""
    import csv
    
    output_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"📁 Results saved to: {output_path}")
    return output_path

# =====================================================================
# SPATIAL INDEX MANAGER
# =====================================================================

class SpatialIndex:
    """Manages R-tree spatial index for fast geographic queries"""
    
    def __init__(self):
        self.idx = index.Index()
        self.businesses = {}  # string_id -> Business object
        self.id_mapping = {}  # string_id -> integer_index
        self.reverse_mapping = {}  # integer_index -> string_id
        self.next_index = 0
        
    def insert_business(self, business: Business):
        """Insert a business into the spatial index"""
        # Convert string ID to integer index for R-tree
        if business.id not in self.id_mapping:
            self.id_mapping[business.id] = self.next_index
            self.reverse_mapping[self.next_index] = business.id
            self.next_index += 1
        
        int_id = self.id_mapping[business.id]
        
        # Store business
        self.businesses[business.id] = business
        
        # Insert into R-tree index using integer ID
        # R-tree expects (minx, miny, maxx, maxy) bounding box
        # For points, min and max are the same
        self.idx.insert(
            int_id, 
            (business.longitude, business.latitude, business.longitude, business.latitude)
        )
    
    def find_nearby_businesses(self, center_business: Business, radius_meters: float) -> List[Business]:
        """
        Find all businesses within radius of center business
        Returns list of nearby businesses (no distance calculation)
        """
        # Convert radius to approximate degrees for bounding box
        radius_degrees = GeoUtils.meters_to_degrees_approx(center_business.latitude, radius_meters)
        
        # Create bounding box
        min_lat = center_business.latitude - radius_degrees
        max_lat = center_business.latitude + radius_degrees
        min_lng = center_business.longitude - radius_degrees
        max_lng = center_business.longitude + radius_degrees
        
        # Query R-tree for candidates within bounding box
        candidate_int_ids = list(self.idx.intersection((min_lng, min_lat, max_lng, max_lat)))
        
        # Filter candidates by different user_id (no distance calculation)
        nearby_businesses = []
        
        for int_id in candidate_int_ids:
            # Convert integer ID back to string ID
            business_id = self.reverse_mapping[int_id]
            candidate_business = self.businesses[business_id]
            
            # Skip if same business
            if candidate_business.id == center_business.id:
                continue
                
            # Skip if same user ONLY when both user_id values are present.
            # If one/both user_id are empty/null, we still compare the businesses.
            if (
                candidate_business.user_id
                and center_business.user_id
                and candidate_business.user_id == center_business.user_id
            ):
                # Additional check for supplement businesses: if same user and same project_id, skip
                if (candidate_business.business_type == 'supplement' and 
                    center_business.business_type == 'supplement' and
                    candidate_business.project_id and 
                    center_business.project_id and
                    candidate_business.project_id == center_business.project_id):
                    continue
                # If users are same but different project_ids (or one/both project_ids are empty), continue processing
                elif candidate_business.business_type == 'supplement' and center_business.business_type == 'supplement':
                    pass  # Continue processing - same user but different projects
                else:
                    continue  # For non-supplement businesses, skip if same user
            
            # Add to nearby businesses (assuming R-tree bounding box is accurate enough)
            nearby_businesses.append(candidate_business)
        
        return nearby_businesses

# =====================================================================
# MAIN FINDER ENGINE
# =====================================================================

class NearbyBusinessFinder:
    """Main engine for finding nearby businesses and detecting duplicates"""
    
    def __init__(self, radius_meters: float = RADIUS_METERS, 
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.radius_meters = radius_meters
        self.spatial_index = SpatialIndex()
        self.duplicate_detector = DuplicateDetector(similarity_threshold)
    
    def run_search(self):
        """Execute the complete nearby business search"""
        start_time = time.time()
        
        try:
            print("🔍 Starting Fast Nearby Business Search")
            print("=" * 60)
            print(f"Configuration:")
            print(f"  - Search radius: {self.radius_meters} meters")
            print(f"  - Data source: CSV ({CSV_INPUT_FILE})")
            if SLS_ID_FILTER:
                print(f"  - SLS ID filter: {SLS_ID_FILTER}")
            print("-" * 60)

            # Load businesses from CSV for spatial indexing + processing
            all_businesses = CSVBusinessLoader.load_businesses(CSV_INPUT_FILE)

            # Apply SLS ID filter if specified
            if SLS_ID_FILTER:
                all_businesses = [b for b in all_businesses if b.area_id == SLS_ID_FILTER]
            
            if not all_businesses:
                print("⚠️ No businesses found. Exiting.")
                return
            
            print(f"✓ Total businesses loaded for spatial index: {len(all_businesses)}")

            # CSV mode: treat all rows as "unprocessed".
            unprocessed_businesses = list(all_businesses)
            print(f"✓ Businesses to check: {len(unprocessed_businesses)}")
            
            # Apply processing limit if enabled
            original_count = len(unprocessed_businesses)
            if LIMIT_PROCESSING and PROCESSING_LIMIT is not None and len(unprocessed_businesses) > PROCESSING_LIMIT:
                unprocessed_businesses = unprocessed_businesses[:PROCESSING_LIMIT]
                print(f"⚠️  Processing limit applied: {len(unprocessed_businesses):,} of {original_count:,} businesses will be processed")
            elif LIMIT_PROCESSING:
                print(f"✓ Processing limit enabled but not reached: processing all {len(unprocessed_businesses):,} businesses")
            
            # Build spatial index with ALL businesses
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
            unique_businesses_with_duplicates = set()  # Track unique businesses that have duplicates
            
            # CSV mode: start fresh each run (no DB state)
            compared_pairs = set()
            
            skipped_existing_pairs = 0  # Track how many pairs were skipped due to existing database records
            validation_data = []
  
            for i, business in enumerate(unprocessed_businesses):
                if i % 1000 == 0:
                    elapsed = time.time() - search_start_time
                    print(f"  Progress: {i:,}/{len(unprocessed_businesses):,} businesses processed ({elapsed:.1f}s)")
                
                # Find nearby businesses
                nearby_businesses = self.spatial_index.find_nearby_businesses(
                    business, self.radius_meters
                )
                
                if nearby_businesses:
                    businesses_with_matches += 1
                    total_matches += len(nearby_businesses)
                    
                    # Perform duplicate detection for each nearby business
                    duplicate_comparisons = []
                    
                    for nearby_business in nearby_businesses:
                        # Create a pair identifier to avoid duplicate comparisons
                        # Order-insensitive key ensures (A,B) and (B,A) are treated as the same pair
                        pair_id = make_unordered_pair_key(business.id, nearby_business.id)
                        
                        # Skip if this pair has already been compared (either in this run or previous runs)
                        if pair_id in compared_pairs:
                            skipped_existing_pairs += 1
                            continue
                        
                        # Mark this pair as compared
                        compared_pairs.add(pair_id)
                        
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
                            # Track both businesses involved in the duplicate
                            unique_businesses_with_duplicates.add(business.id)
                            unique_businesses_with_duplicates.add(nearby_business.id)
                        elif comparison.duplicate_type == 'weak_duplicate':
                            total_duplicates['weak'] += 1
                            # Track both businesses involved in the duplicate
                            unique_businesses_with_duplicates.add(business.id)
                            unique_businesses_with_duplicates.add(nearby_business.id)
                        else:
                            total_duplicates['not_duplicate'] += 1
                    
                    # Save detailed results to CSV if enabled
                    if SAVE_RESULTS_TO_FILE:
                        for comparison in duplicate_comparisons:
                            # Skip not_duplicate results if configured to exclude them
                            if not INCLUDE_NOT_DUPLICATES_IN_OUTPUT and comparison.duplicate_type == 'not_duplicate':
                                continue
                                
                            validation_data.append({
                                'center_business_id': comparison.business_a.id,
                                'nearby_business_id': comparison.business_b.id,
                                'center_area_id': comparison.business_a.area_id,
                                'nearby_area_id': comparison.business_b.area_id,
                                'center_business_source': comparison.business_a.business_type,
                                'nearby_business_source': comparison.business_b.business_type,
                                'center_business_name': comparison.business_a.name,
                                'nearby_business_name': comparison.business_b.name,
                                'center_business_owner': comparison.business_a.owner,
                                'nearby_business_owner': comparison.business_b.owner,
                                'name_similarity': round(comparison.name_similarity, 3),
                                'owner_similarity': round(comparison.owner_similarity, 3),
                                'confidence_score': round(comparison.confidence_score, 3),
                                'distance_meters': comparison.distance_meters,
                                'center_business_user': comparison.business_a.user_id,
                                'nearby_business_user': comparison.business_b.user_id,
                                'center_lat': comparison.business_a.latitude,
                                'center_lng': comparison.business_a.longitude,
                                'nearby_lat': comparison.business_b.latitude,
                                'nearby_lng': comparison.business_b.longitude
                            })
                    
                    # Print results with duplicate information (only when duplicates found)
                    strong_dupes = sum(1 for c in duplicate_comparisons if c.duplicate_type == 'strong_duplicate')
                    weak_dupes = sum(1 for c in duplicate_comparisons if c.duplicate_type == 'weak_duplicate')
                    
                    if strong_dupes > 0 or weak_dupes > 0:
                        progress_pct = ((i + 1) / len(unprocessed_businesses)) * 100
                        print(f"📍 [{progress_pct:.1f}%] {business.name} → 🔴 {strong_dupes} strong, 🟡 {weak_dupes} weak duplicates")
                
                # No per-row persistence in CSV mode
            
            search_end_time = time.time()

            print(f"\n✅ Processing complete:")
            print(f"  - CSV mode")
            
            # Save results to file if enabled
            if SAVE_RESULTS_TO_FILE and validation_data:
                if USE_TIMESTAMP_IN_FILENAME:
                    timestamp = get_jakarta_now().strftime("%Y%m%d_%H%M%S")
                    filename = f"business_duplicate_detection_results_{timestamp}.csv"
                else:
                    filename = OUTPUT_FILENAME
                
                save_results_to_csv(validation_data, filename)
                
                # Display what was saved to CSV
                if INCLUDE_NOT_DUPLICATES_IN_OUTPUT:
                    print(f"💾 Saved {len(validation_data):,} total comparison results to CSV (including not duplicates)")
                else:
                    print(f"💾 Saved {len(validation_data):,} duplicate results to CSV only (excluded not duplicates)")
            
            # Display output mode summary
            if SAVE_RESULTS_TO_FILE:
                print(f"📤 Results saved to CSV file only")
            else:
                print(f"📤 No results saved (SAVE_RESULTS_TO_FILE is disabled)")
            
            total_end_time = time.time()
            
            print(f"\n" + "=" * 60)
            print("✅ Duplicate Detection Search completed successfully!")
            print(f"📊 Summary:")
            print(f"  - Total businesses in spatial index: {len(all_businesses):,}")
            print(f"  - Unprocessed businesses analyzed: {len(unprocessed_businesses):,}")
            print(f"  - Businesses with nearby matches: {businesses_with_matches:,}")
            print(f"  - Total nearby business pairs found: {total_matches:,}")
            print(f"  - Average matches per business: {total_matches / len(unprocessed_businesses):.2f}" if len(unprocessed_businesses) > 0 else "  - Average matches per business: 0")
            
            print(f"\n🔍 Duplicate Detection Results:")
            print(f"  - Strong duplicate pairs found: {total_duplicates['strong']:,}")
            print(f"  - Weak duplicate pairs found: {total_duplicates['weak']:,}")
            print(f"  - Not duplicate pairs: {total_duplicates['not_duplicate']:,}")
            print(f"  - Total comparison pairs: {total_matches:,}")
            print(f"  - Skipped existing database pairs: {skipped_existing_pairs:,}")
            print(f"  - Unique businesses with duplicates: {len(unique_businesses_with_duplicates):,}")
            print(f"  - Business duplicate rate: {len(unique_businesses_with_duplicates) / len(unprocessed_businesses) * 100:.1f}% ({len(unique_businesses_with_duplicates)} of {len(unprocessed_businesses)} unprocessed businesses)" if len(unprocessed_businesses) > 0 else "  - Business duplicate rate: 0%")
            if total_matches > 0:
                print(f"  - Pair duplicate rate: {(total_duplicates['strong'] + total_duplicates['weak']) / total_matches * 100:.1f}% (pairs that are duplicates)")
            
            print(f"\n⚡ Optimization:")
            total_skipped = skipped_existing_pairs
            if total_skipped > 0:
                total_potential_comparisons = total_matches + total_skipped
                efficiency_gain = (total_skipped / total_potential_comparisons) * 100
                print(f"  - Efficiency gain: {efficiency_gain:.1f}% (avoided {total_skipped:,} redundant/duplicate comparisons)")
                print(f"  - Skipped duplicate pairs: {skipped_existing_pairs:,}")
                print(f"  - Total potential comparisons: {total_potential_comparisons:,}")
            else:
                print(f"  - No redundant comparisons found (optimal case)")
            
            print(f"\n⏱️  Performance:")
            print(f"  - Total execution time: {total_end_time - start_time:.1f} seconds")
            print(f"  - Search phase time: {search_end_time - search_start_time:.1f} seconds")
            print(f"  - Businesses processed per second: {len(unprocessed_businesses) / (search_end_time - search_start_time):.0f}" if (search_end_time - search_start_time) > 0 and len(unprocessed_businesses) > 0 else "  - Businesses processed per second: 0")
            
        except Exception as e:
            print(f"\n❌ Error during search: {e}")
            raise
        finally:
            pass

# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main function"""
    try:
        print("🚀 Business Duplicate Detector")
        print(f"🎯 Searching for businesses within {RADIUS_METERS}m radius")
        print(f"🔍 Detecting duplicates with {SIMILARITY_THRESHOLD} similarity threshold")
        print(f"⚡ Using R-tree spatial indexing for performance")
        print(f"📄 Input CSV: {CSV_INPUT_FILE}")
        
        # Display common words configuration
        if USE_COMMON_WORDS_FILTERING:
            # Load common words (this will display loading info)
            common_words = CommonWordsManager.load_common_words()
            print(f"📝 Common words filtering enabled ({len(common_words)} words loaded)")
        else:
            print(f"📝 Common words filtering disabled (using full text comparison)")
        
        # Display ignore names configuration
        if USE_IGNORE_NAMES:
            # Load ignore names (this will display loading info)
            ignore_names = IgnoreNamesManager.load_ignore_names()
            if len(ignore_names) > 0:
                print(f"🚫 Ignore names enabled ({len(ignore_names)} business names will be excluded)")
            else:
                print(f"🚫 Ignore names enabled but no names loaded (all businesses will be checked)")
        else:
            print(f"🚫 Ignore names disabled (all businesses will be checked)")
        
        if SLS_ID_FILTER:
            print(f"🎯 Filtering by SLS ID: {SLS_ID_FILTER}")
        
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
            print(f"  🔢 Processing limit: ❌ Disabled (will process all unprocessed businesses)")
        
        # Display output configuration
        print(f"\n📁 Output Configuration:")
        if SAVE_RESULTS_TO_FILE:
            print(f"  📄 Save to CSV file: ✅ ({OUTPUT_FILENAME})")
        else:
            print(f"  📄 Save to CSV file: ❌")
            print(f"  ⚠️  No output configured - results will not be saved!")
        
        if SAVE_RESULTS_TO_FILE:
            if USE_TIMESTAMP_IN_FILENAME:
                print(f"  📝 CSV filename: With timestamp")
            else:
                print(f"  📝 CSV filename: Fixed (overwrites previous)")
            
        if SAVE_RESULTS_TO_FILE:
            if INCLUDE_NOT_DUPLICATES_IN_OUTPUT:
                print(f"  Content: All results (duplicates + not duplicates)")
            else:
                print(f"  Content: Only duplicates (strong + weak duplicates)")
        
        print("")
        
        finder = NearbyBusinessFinder(RADIUS_METERS, SIMILARITY_THRESHOLD)
        finder.run_search()
        
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())