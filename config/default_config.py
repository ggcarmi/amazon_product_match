"""
Default configuration settings for product matching system
"""

from src.strategies import QueryStrategyType, ScorerStrategyType
from typing import Dict, Any

DEFAULT_CONFIG = {
    # Strategy settings
    'query_strategy_type': QueryStrategyType.HYBRID.value,
    'confidence_scorer_type': ScorerStrategyType.HYBRID.value,
    
    # Matching parameters
    'confidence_threshold': 0.55,
    'save_all_candidates': True,
    'max_results_per_product': 50,
    'max_products': 374,  # Default to 5 products for debugging
    
    # File paths
    'input_file': 'data/amazon_items.json',
    'output_dir': 'data/matches',
    'output_file': 'data/matches/product_matches.json',
    'log_file': 'product_matching.log',
    
    # Logging
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}

def get_config() -> Dict[str, Any]:
    """Get the default configuration settings
    
    Returns:
        Dict[str, Any]: The configuration dictionary
    """
    return DEFAULT_CONFIG.copy()