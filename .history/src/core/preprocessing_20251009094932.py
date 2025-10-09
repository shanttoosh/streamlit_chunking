# Preprocessing Functions
import pandas as pd
import re
import time
import logging
from typing import Dict, Any, Tuple
from ..utils.text_utils import clean_text, normalize_text_column, remove_html_tags, validate_and_normalize_headers

logger = logging.getLogger(__name__)

def preprocess_basic(df: pd.DataFrame, null_handling="keep", fill_value=None):
    """Basic preprocessing function"""
    start_time = time.time()
    if null_handling == "drop":
        df = df.dropna().reset_index(drop=True)
    elif null_handling == "fill" and fill_value is not None:
        df = df.fillna(fill_value)
    logger.info(f"Basic preprocessing completed in {time.time() - start_time:.2f}s")
    return df

def preprocess_auto_fast(df: pd.DataFrame):
    """Auto preprocessing for Fast Mode: lowercase + remove delimiters + remove whitespace"""
    start_time = time.time()
    
    # Handle nulls by dropping
    df = df.dropna().reset_index(drop=True)
    
    # Apply text cleaning to all string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = clean_text_advanced(df[col], lowercase=True, remove_delimiters=True, remove_whitespace=True)
    
    logger.info(f"Auto Fast preprocessing completed in {time.time() - start_time:.2f}s")
    return df

def preprocess_optimized_fast(df: pd.DataFrame):
    """OPTIMIZED: Faster preprocessing for large files - minimal operations"""
    start_time = time.time()
    
    # Clean column names
    df = clean_column_names(df)
    
    # Only essential operations - drop nulls
    df = df.dropna().reset_index(drop=True)
    
    # Fast text cleaning - only lowercase, skip other operations for speed
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.lower()
    
    logger.info(f"Optimized fast preprocessing completed in {time.time() - start_time:.2f}s")
    return df

def preprocess_auto_config1(df: pd.DataFrame, null_handling="keep", fill_value=None):
    """Auto preprocessing for Config1 Mode: lowercase + remove delimiters + remove whitespace + null handling"""
    start_time = time.time()
    
    # Handle nulls based on user choice
    if null_handling == "drop":
        df = df.dropna().reset_index(drop=True)
    elif null_handling == "fill" and fill_value is not None:
        df = df.fillna(fill_value)
    
    # Apply text cleaning to all string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = clean_text_advanced(df[col], lowercase=True, remove_delimiters=True, remove_whitespace=True)
    
    logger.info(f"Auto Config1 preprocessing completed in {time.time() - start_time:.2f}s")
    return df

def clean_text_advanced(text_series: pd.Series, lowercase: bool = True, remove_delimiters: bool = True, 
                       remove_whitespace: bool = True) -> pd.Series:
    """Advanced text cleaning for string columns"""
    cleaned_series = text_series.astype(str)
    
    if lowercase:
        cleaned_series = cleaned_series.str.lower()
    
    if remove_delimiters:
        # Remove common delimiters and special characters
        cleaned_series = cleaned_series.str.replace(r'[^\w\s]', ' ', regex=True)
    
    if remove_whitespace:
        # Remove extra whitespace
        cleaned_series = cleaned_series.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return cleaned_series

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize column names"""
    new_columns = []
    for i, col in enumerate(df.columns):
        if pd.isna(col) or str(col).strip() == "":
            new_col = f"column_{i+1}"
        else:
            # Clean column name: lowercase, replace spaces/special chars with underscores
            new_col = str(col).strip().lower()
            new_col = re.sub(r'[^a-z0-9_]', '_', new_col)
            new_col = re.sub(r'_+', '_', new_col)  # Replace multiple underscores with single
            new_col = new_col.strip('_')  # Remove leading/trailing underscores
            if not new_col or new_col.startswith('_'):
                new_col = f"column_{i+1}"
        new_columns.append(new_col)
    df.columns = new_columns
    return df

def convert_column_types(df: pd.DataFrame, column_types: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
    """Convert column data types based on user specification"""
    start_time = time.time()
    df_converted = df.copy()
    
    conversion_results = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    for column, target_type in column_types.items():
        if column not in df_converted.columns:
            conversion_results['skipped'].append(f"Column '{column}' not found")
            continue
            
        try:
            if target_type == 'string':
                df_converted[column] = df_converted[column].astype(str)
                conversion_results['successful'].append(f"{column} -> string")
                
            elif target_type == 'numeric':
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                conversion_results['successful'].append(f"{column} -> numeric")
                
            elif target_type == 'integer':
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce').fillna(0).astype(int)
                conversion_results['successful'].append(f"{column} -> integer")
                
            elif target_type == 'float':
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce').astype(float)
                conversion_results['successful'].append(f"{column} -> float")
                
            elif target_type == 'datetime':
                df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
                conversion_results['successful'].append(f"{column} -> datetime")
                
            elif target_type == 'boolean':
                # Try to convert common boolean representations
                if df_converted[column].dtype == 'object':
                    true_values = ['true', 'yes', '1', 't', 'y']
                    false_values = ['false', 'no', '0', 'f', 'n']
                    df_converted[column] = df_converted[column].astype(str).str.lower().isin(true_values)
                conversion_results['successful'].append(f"{column} -> boolean")
                
            elif target_type == 'category':
                df_converted[column] = df_converted[column].astype('category')
                conversion_results['successful'].append(f"{column} -> category")
                
            else:
                conversion_results['skipped'].append(f"Unknown type '{target_type}' for column '{column}'")
                
        except Exception as e:
            conversion_results['failed'].append(f"{column} -> {target_type}: {str(e)}")
    
    logger.info(f"Column type conversion completed in {time.time() - start_time:.2f}s")
    logger.info(f"Conversion results: {len(conversion_results['successful'])} successful, "
                f"{len(conversion_results['failed'])} failed, "
                f"{len(conversion_results['skipped'])} skipped")
    
    return df_converted, conversion_results
