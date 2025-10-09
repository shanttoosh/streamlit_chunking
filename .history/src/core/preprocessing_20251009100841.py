# Data Preprocessing Module
import pandas as pd
import numpy as np
import re
import os
import warnings
from typing import Dict, Any, Optional, Tuple, List
import logging

# Optional dependencies
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def remove_html(text):
    """Remove HTML tags from text"""
    if not BEAUTIFULSOUP_AVAILABLE:
        # Fallback to regex if BeautifulSoup not available
        return re.sub(r'<[^>]+>', '', str(text))
    
    if pd.isna(text) or text == '':
        return text
    
    try:
        soup = BeautifulSoup(str(text), 'html.parser')
        return soup.get_text()
    except:
        return re.sub(r'<[^>]+>', '', str(text))

def validate_and_normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize column headers"""
    new_columns = []
    for i, col in enumerate(df.columns):
        if col is None or str(col).strip() == "":
            new_col = f"column_{i+1}"
        else:
            new_col = str(col).strip().lower()
        new_columns.append(new_col)
    df.columns = new_columns
    return df

def normalize_text_column(s: pd.Series, lowercase=True, strip=True, remove_html_flag=True):
    """Normalize text column with HTML removal, lowercase, and strip"""
    if remove_html_flag:
        s = s.apply(remove_html)
    if lowercase:
        s = s.str.lower()
    if strip:
        s = s.str.strip()
    return s

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names"""
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col).strip()) for col in df.columns]
    return df

def preprocess_basic(df: pd.DataFrame, null_handling="keep", fill_value=None):
    """Basic preprocessing"""
    if null_handling == "fill" and fill_value is not None:
        df = df.fillna(fill_value)
    elif null_handling == "drop":
        df = df.dropna()
    return df

def preprocess_auto_fast(df: pd.DataFrame):
    """Automatic preprocessing for Fast Mode"""
    # Clean column names
    df = clean_column_names(df)
    
    # Fill null values with empty string for text columns
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].fillna('')
    
    return df

def preprocess_optimized_fast(df: pd.DataFrame):
    """Optimized preprocessing for Fast Mode with turbo"""
    # Clean column names
    df = clean_column_names(df)
    
    # Fill null values with empty string for text columns
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].fillna('')
    
    # Optimize data types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    return df

def preprocess_auto_config1(df: pd.DataFrame, null_handling="keep", fill_value=None):
    """Automatic preprocessing for Config-1 Mode"""
    # Clean column names
    df = clean_column_names(df)
    
    # Handle null values
    if null_handling == "fill" and fill_value is not None:
        df = df.fillna(fill_value)
    elif null_handling == "drop":
        df = df.dropna()
    elif null_handling == "keep":
        # Fill text columns with empty string
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            df[col] = df[col].fillna('')
    
    return df

def clean_text_advanced(text_series: pd.Series, lowercase: bool = True, remove_delimiters: bool = True, 
                       remove_html_flag: bool = True):
    """Advanced text cleaning with multiple options"""
    if remove_html_flag:
        text_series = text_series.apply(remove_html)
    
    if lowercase:
        text_series = text_series.str.lower()
    
    if remove_delimiters:
        text_series = text_series.str.replace(r'[^\w\s]', ' ', regex=True)
        text_series = text_series.str.replace(r'\s+', ' ', regex=True)
        text_series = text_series.str.strip()
    
    return text_series

def estimate_token_count(text: str) -> int:
    """Estimate token count for text"""
    if pd.isna(text) or text == '':
        return 0
    # Rough estimation: 1 token ≈ 4 characters
    return len(str(text)) // 4

def convert_column_types(df: pd.DataFrame, column_types: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
    """Convert column types based on configuration"""
    conversion_results = {}
    
    for col, target_type in column_types.items():
        if col not in df.columns:
            conversion_results[col] = {"status": "error", "message": "Column not found"}
            continue
        
        try:
            if target_type == "int":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif target_type == "float":
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif target_type == "bool":
                def _to_bool(v):
                    if pd.isna(v):
                        return pd.NA
                    v_str = str(v).lower()
                    if v_str in ['true', '1', 'yes', 'y', 'on']:
                        return True
                    elif v_str in ['false', '0', 'no', 'n', 'off']:
                        return False
                    else:
                        return pd.NA
                df[col] = df[col].apply(_to_bool)
            elif target_type == "datetime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif target_type == "category":
                df[col] = df[col].astype('category')
            
            conversion_results[col] = {"status": "success", "message": f"Converted to {target_type}"}
        except Exception as e:
            conversion_results[col] = {"status": "error", "message": str(e)}
    
    return df, conversion_results

def profile_nulls_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced null profiling"""
    null_profile = []
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        
        null_profile.append({
            'column': col,
            'null_count': null_count,
            'null_percentage': null_percentage,
            'data_type': str(df[col].dtype),
            'suggestion': suggest_null_strategy_enhanced(col, df[col])
        })
    
    return pd.DataFrame(null_profile)

def suggest_null_strategy_enhanced(col_name: str, s: pd.Series) -> str:
    """Suggest null handling strategy"""
    null_count = s.isnull().sum()
    total_count = len(s)
    null_percentage = (null_count / total_count) * 100
    
    if null_percentage == 0:
        return "no_action"
    elif null_percentage < 5:
        return "drop"
    elif null_percentage < 30:
        if s.dtype == 'object':
            return "fill_empty"
        else:
            return "fill_mean" if s.dtype in ['int64', 'float64'] else "fill_mode"
    else:
        return "keep"

def apply_null_strategies_enhanced(df: pd.DataFrame, strategies: dict, add_flags: bool = True) -> pd.DataFrame:
    """Apply null handling strategies"""
    df_result = df.copy()
    
    for col, strategy in strategies.items():
        if col not in df.columns:
            continue
        
        if strategy == "drop":
            df_result = df_result.dropna(subset=[col])
        elif strategy == "fill_empty":
            df_result[col] = df_result[col].fillna('')
        elif strategy == "fill_mean":
            if df_result[col].dtype in ['int64', 'float64']:
                df_result[col] = df_result[col].fillna(df_result[col].mean())
        elif strategy == "fill_mode":
            mode_value = df_result[col].mode()
            if len(mode_value) > 0:
                df_result[col] = df_result[col].fillna(mode_value[0])
        elif strategy == "keep":
            pass  # Keep nulls as is
        
        # Add null flag column if requested
        if add_flags and strategy != "drop":
            df_result[f"{col}_is_null"] = df[col].isnull().astype(int)
    
    return df_result

def analyze_duplicates_enhanced(df: pd.DataFrame) -> dict:
    """Enhanced duplicate analysis"""
    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    duplicate_percentage = (duplicate_rows / total_rows) * 100
    
    # Analyze duplicate patterns
    duplicate_subset = df[df.duplicated(keep=False)]
    
    return {
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_percentage,
        "unique_rows": total_rows - duplicate_rows,
        "duplicate_subset_size": len(duplicate_subset)
    }

def remove_duplicates_enhanced(df: pd.DataFrame, strategy: str = 'keep_first') -> pd.DataFrame:
    """Enhanced duplicate removal"""
    if strategy == 'keep_first':
        return df.drop_duplicates(keep='first')
    elif strategy == 'keep_last':
        return df.drop_duplicates(keep='last')
    elif strategy == 'drop_all':
        return df.drop_duplicates(keep=False)
    else:
        return df

def remove_stopwords_from_text_column_enhanced(df, remove_stopwords=True):
    """Remove stopwords from text columns"""
    if not remove_stopwords or not SPACY_AVAILABLE:
        return df
    
    try:
        nlp = spacy.load("en_core_web_sm")
        stop_words = nlp.Defaults.stop_words
        
        def process_text(text):
            if pd.isna(text):
                return text
            words = str(text).split()
            filtered_words = [word for word in words if word.lower() not in stop_words]
            return ' '.join(filtered_words)
        
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            df[col] = df[col].apply(process_text)
    except Exception as e:
        logger.warning(f"Stopword removal failed: {e}")
    
    return df

def process_text_enhanced(df, method):
    """Enhanced text processing"""
    if method == "none":
        return df
    
    text_cols = df.select_dtypes(include=['object']).columns
    
    if method == "lemmatize":
        def lemmatize_text(text):
            # Placeholder for lemmatization
            return text
        for col in text_cols:
            df[col] = df[col].apply(lemmatize_text)
    
    elif method == "stem":
        def stem_text(text):
            # Placeholder for stemming
            return text
        for col in text_cols:
            df[col] = df[col].apply(stem_text)
    
    return df

def preprocess_csv_enhanced(input_obj, fill_null_strategy=None, type_conversions=None, remove_stopwords_flag=False):
    """Enhanced CSV preprocessing"""
    # Load CSV
    if hasattr(input_obj, 'read'):
        df = pd.read_csv(input_obj)
    else:
        df = pd.read_csv(input_obj)
    
    # Apply type conversions
    if type_conversions:
        df, _ = convert_column_types(df, type_conversions)
    
    # Apply null strategies
    if fill_null_strategy:
        df = apply_null_strategies_enhanced(df, fill_null_strategy)
    
    # Remove stopwords
    if remove_stopwords_flag:
        df = remove_stopwords_from_text_column_enhanced(df)
    
    return df

def estimate_token_count_enhanced(text: str) -> int:
    """Enhanced token count estimation"""
    if pd.isna(text) or text == '':
        return 0
    # More accurate estimation
    return len(str(text).split())

def validate_and_normalize_headers_enhanced(columns):
    """Enhanced header validation and normalization"""
    new_columns = []
    for i, col in enumerate(columns):
        if col is None or str(col).strip() == "":
            new_col = f"column_{i+1}"
        else:
            new_col = str(col).strip().lower()
            # Replace special characters with underscores
            new_col = re.sub(r'[^a-z0-9_]', '_', new_col)
            # Remove multiple consecutive underscores
            new_col = re.sub(r'_+', '_', new_col)
            # Remove leading/trailing underscores
            new_col = new_col.strip('_')
            if not new_col:
                new_col = f"column_{i+1}"
        new_columns.append(new_col)
    return new_columns

def normalize_text_column_enhanced(s: pd.Series, lowercase=True, strip=True, remove_html_flag=True):
    """Enhanced text column normalization"""
    if remove_html_flag:
        s = s.apply(remove_html)
    if lowercase:
        s = s.str.lower()
    if strip:
        s = s.str.strip()
    return s

def apply_type_conversion_enhanced(df: pd.DataFrame, conversion: dict):
    """Enhanced type conversion"""
    df_result = df.copy()
    conversion_results = {}
    
    for col, target_type in conversion.items():
        if col not in df_result.columns:
            conversion_results[col] = {"status": "error", "message": "Column not found"}
            continue
        
        try:
            if target_type == "int":
                df_result[col] = pd.to_numeric(df_result[col], errors='coerce').astype('Int64')
            elif target_type == "float":
                df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
            elif target_type == "bool":
                def _to_bool(v):
                    if pd.isna(v):
                        return pd.NA
                    v_str = str(v).lower()
                    if v_str in ['true', '1', 'yes', 'y', 'on']:
                        return True
                    elif v_str in ['false', '0', 'no', 'n', 'off']:
                        return False
                    else:
                        return pd.NA
                df_result[col] = df_result[col].apply(_to_bool)
            elif target_type == "datetime":
                df_result[col] = pd.to_datetime(df_result[col], errors='coerce')
            elif target_type == "category":
                df_result[col] = df_result[col].astype('category')
            
            conversion_results[col] = {"status": "success", "message": f"Converted to {target_type}"}
        except Exception as e:
            conversion_results[col] = {"status": "error", "message": str(e)}
    
    return df_result, conversion_results