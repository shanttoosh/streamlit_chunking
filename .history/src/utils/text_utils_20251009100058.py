# Text Processing Utilities
import re
import pandas as pd
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean text by removing special characters"""
    if not isinstance(text, str):
        return str(text)
    return re.sub(r'[^\w\s]', ' ', text)

def normalize_text_column(series: pd.Series, lowercase: bool = True, strip: bool = True) -> pd.Series:
    """Normalize text column"""
    series = series.fillna('')
    if lowercase:
        series = series.str.lower()
    if strip:
        series = series.str.strip()
    return series

def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text"""
    if not isinstance(text, str):
        return str(text)
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(text, "lxml").get_text(separator=' ')
    except ImportError:
        # Fallback without BeautifulSoup
        return re.sub('<[^<]+?>', ' ', text)
    except:
        return re.sub('<[^<]+?>', ' ', text)

def validate_and_normalize_headers(columns: List[str]) -> List[str]:
    """Validate and normalize column headers"""
    new_columns = []
    for i, col in enumerate(columns):
        if col is None or str(col).strip() == "":
            new_col = f"column_{i+1}"
        else:
            new_col = str(col).strip().lower()
        new_columns.append(new_col)
    return new_columns

def estimate_token_count(text: str) -> int:
    """Estimate token count for text"""
    if not text:
        return 0
    # More accurate token estimation: words + punctuation
    words = len(text.split())
    punctuation = len([c for c in text if c in '.,!?;:()[]{}"\''])
    # Average of 1.3 tokens per word + punctuation as separate tokens
    return int(words * 1.3 + punctuation)

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

def remove_stopwords_from_text(text: str) -> str:
    """Remove stopwords from text"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        filtered_tokens = [token.text for token in doc if not token.is_stop]
        return " ".join(filtered_tokens)
    except ImportError:
        # Fallback without spaCy
        return text

def lemmatize_text(text: str) -> str:
    """Lemmatize text"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        doc = nlp(text)
        return " ".join([token.text if token.lemma_ == '-PRON-' else token.lemma_ for token in doc])
    except ImportError:
        return text

def stem_text(text: str) -> str:
    """Stem text"""
    try:
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        stemmer = PorterStemmer()
        words = word_tokenize(text)
        return " ".join([stemmer.stem(word) for word in words])
    except ImportError:
        return text
