"""
Strategy pattern implementations for product matching
"""

from enum import Enum

# Enums for strategy types
class QueryStrategyType(Enum):
    BRAND_MODEL = "brand_model"
    KEY_TERMS = "key_terms"
    CATEGORY = "category"
    DETAILS = "details"
    HYBRID = "hybrid"

class ScorerStrategyType(Enum):
    TITLE_SIMILARITY = "title_similarity" 
    HYBRID = "hybrid"