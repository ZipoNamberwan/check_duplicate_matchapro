#!/usr/bin/env python3
"""
Business Duplicate Detector using Spatial Search

This script efficiently finds businesses within a specified radius using R-tree spatial indexing
and then analyzes them for potential duplicates using text similarity comparison with configurable rules.

It loads all businesses from supplement_business and market_business tables and for each business,
finds all other businesses within 50 meters that belong to different users, then compares them
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
    - mysql-connector-python
    - python-dotenv
    - geopy
    - rtree
    - shapely
    - difflib (built-in)

Install with: pip install mysql-connector-python python-dotenv geopy rtree shapely
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

import pandas as pd
from dotenv import load_dotenv
from rtree import index

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
EXCEL_SOURCE_FOLDER = 'source'  # Folder containing Excel files to process
EXCEL_RESULT_FOLDER = 'result'  # Folder to save results
EXCEL_RESULT_FILENAME = 'result.xlsx'  # Output filename

# Filter configuration
FILTER_SUMBER_VALUE = 'PL-KUMKM 2023'  # Filter rows where Sumber column = this value

# Debug mode - set to True to limit businesses for testing
DEBUG_MODE = False
DEBUG_LIMIT = 10000  # Number of businesses to process in debug mode

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

# =====================================================================
# DATABASE CONNECTION SETTINGS (DEPRECATED - NOW USING EXCEL)
# =====================================================================

# These are kept for reference but no longer used
DB_CONFIG = {}

# Business tables configuration (DEPRECATED - NOW USING EXCEL)
BUSINESS_TABLES = []

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
    user_id: str  # User/source identifier
    sls_id: str  # SLS ID for the business
    business_type: str  # 'excel' for Excel-based businesses
    address: str = ""
    project_id: str = ""  # Project ID (unique per row in Excel)
    source_row: int = 0  # Track which row in Excel this came from
    iddesa: str = ""  # IDDESA identifier
    
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
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'backup', 'validation')
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
                
            # Skip if same user
            if candidate_business.user_id == center_business.user_id:
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
# EXCEL DATA MANAGER
# =====================================================================

class ExcelDataManager:
    """Handles loading data from Excel files"""
    
    def __init__(self):
        self.source_folder = EXCEL_SOURCE_FOLDER
        self.result_folder = EXCEL_RESULT_FOLDER
        os.makedirs(self.result_folder, exist_ok=True)
    
    def load_excel_files(self) -> pd.DataFrame:
        """Load all Excel files from source folder and combine them"""
        print(f"📁 Loading Excel files from '{self.source_folder}' folder...")
        
        # Find all Excel files
        excel_files = glob.glob(os.path.join(self.source_folder, '*.xlsx')) + \
                     glob.glob(os.path.join(self.source_folder, '*.xls'))
        
        if not excel_files:
            print(f"⚠️ No Excel files found in '{self.source_folder}' folder")
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
    
    def get_businesses(self) -> List[Business]:
        """Load and process Excel data into Business objects"""
        # Load Excel files
        df = self.load_excel_files()
        
        if df.empty:
            print("⚠️ No data to process")
            return []
        
        # Check required columns
        required_columns = ['idsbr', 'nama_usaha', 'Sumber Data', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
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
                # Extract owner from nama_usaha column
                nama_usaha = str(row['nama_usaha']) if pd.notna(row['nama_usaha']) else ""
                cleaned_name, extracted_owner = extract_owner_from_name(nama_usaha)
                
                # Handle both comma and period as decimal separator for coordinates
                # Convert comma to period to normalize decimal format
                lat_str = str(row['latitude']).replace(',', '.')
                lng_str = str(row['longitude']).replace(',', '.')
                
                # Create Business object
                # Each row has a unique project_id since every row is treated as different project
                business = Business(
                    id=str(row['idsbr']),
                    name=cleaned_name,
                    owner=extracted_owner,
                    latitude=float(lat_str),
                    longitude=float(lng_str),
                    user_id=str(row['idsbr']),  # Use idsbr as user_id (each row is different)
                    sls_id=str(row.get('sls_id', '')),
                    business_type='excel',
                    address=str(row.get('alamat', '')),
                    project_id=str(row['idsbr']),  # Each row gets unique project_id
                    source_row=idx,
                    iddesa=str(row.get('iddesa', ''))
                )
                
                businesses.append(business)
            except Exception as e:
                print(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        print(f"✓ Loaded {len(businesses)} businesses from Excel")
        return businesses
    
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
        self.data_manager = ExcelDataManager()
        self.spatial_index = SpatialIndex()
        self.duplicate_detector = DuplicateDetector(similarity_threshold)
    
    def run_search(self):
        """Execute the complete nearby business search"""
        start_time = time.time()
        
        try:
            print("🔍 Starting Fast Nearby Business Search with Excel Files")
            print("=" * 60)
            print(f"Configuration:")
            print(f"  - Source folder: {EXCEL_SOURCE_FOLDER}")
            print(f"  - Result file: {os.path.join(EXCEL_RESULT_FOLDER, EXCEL_RESULT_FILENAME)}")
            print(f"  - Filter value: Sumber = '{FILTER_SUMBER_VALUE}'")
            print(f"  - Search radius: {self.radius_meters} meters")
            print("-" * 60)
            
            # Load businesses from Excel
            print("📊 Loading data from Excel files...")
            all_businesses = self.data_manager.get_businesses()
            
            if not all_businesses:
                print("⚠️ No businesses found. Exiting.")
                return
            
            print(f"✓ Total businesses loaded: {len(all_businesses)}")
            
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
            unique_businesses_with_duplicates = set()  # Track unique businesses that have duplicates
            
            skipped_comparisons = 0  # Track how many duplicate comparisons were avoided
            validation_data = []  # Collect results to save
            seen_pairs = set()  # Track pairs already added to avoid duplicates (A,B) and (B,A)
            
            import random
            random.seed(42)  # For reproducible results
            
            for i, business in enumerate(all_businesses):
                if i % 100 == 0:
                    elapsed = time.time() - search_start_time
                    print(f"  Progress: {i:,}/{len(all_businesses):,} businesses processed ({elapsed:.1f}s)")
                
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
                        # Use sorted tuple to ensure (A,B) and (B,A) are treated as the same pair
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
                    
                    # Add detailed results to export list (only duplicates)
                    for comparison in duplicate_comparisons:
                        # Only save if it's a duplicate (strong or weak)
                        if comparison.duplicate_type in ['strong_duplicate', 'weak_duplicate']:
                            # Create a sorted pair to avoid duplicate (A,B) and (B,A)
                            pair_key = tuple(sorted([comparison.business_a.id, comparison.business_b.id]))
                            
                            # Only add if we haven't seen this pair before
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                validation_data.append({
                                    'idsbr_center': comparison.business_a.id,
                                    'idsbr_nearby': comparison.business_b.id,
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
                    
                    # Print results with duplicate information (only when duplicates found)
                    strong_dupes = sum(1 for c in duplicate_comparisons if c.duplicate_type == 'strong_duplicate')
                    weak_dupes = sum(1 for c in duplicate_comparisons if c.duplicate_type == 'weak_duplicate')
                    
                    if strong_dupes > 0 or weak_dupes > 0:
                        progress_pct = ((i + 1) / len(all_businesses)) * 100
                        print(f"📍 [{progress_pct:.1f}%] {business.name} → 🔴 {strong_dupes} strong, 🟡 {weak_dupes} weak duplicates")
            
            search_end_time = time.time()
            
            # Save results to Excel
            if validation_data:
                self.data_manager.save_results_to_excel(validation_data)
            
            total_end_time = time.time()
            
            print(f"\n" + "=" * 60)
            print("✅ Duplicate Detection Search completed successfully!")
            print(f"📊 Summary:")
            print(f"  - Total businesses analyzed: {len(all_businesses):,}")
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

# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main function"""
    try:
        print("🚀 Business Duplicate Detector (Excel Edition)")
        print(f"🎯 Searching for businesses within {RADIUS_METERS}m radius")
        print(f"🔍 Detecting duplicates with {SIMILARITY_THRESHOLD} similarity threshold")
        print(f"⚡ Using R-tree spatial indexing for performance")
        
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
        
        print(f"\n📋 Output Configuration:")
        print(f"  💾 Save to Excel file: ✅")
        print(f"  📄 Output file: {os.path.join(EXCEL_RESULT_FOLDER, EXCEL_RESULT_FILENAME)}")
        print(f"  📊 Content: All comparison results (duplicates + not duplicates)")
        
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