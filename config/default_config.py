"""
Default configuration settings for product matching system
"""

from src.strategies import QueryStrategyType, ScorerStrategyType

DEFAULT_CONFIG = {
    # Strategy settings
    'query_strategy_type': QueryStrategyType.HYBRID.value,
    'confidence_scorer_type': ScorerStrategyType.HYBRID.value,
    
    # Matching parameters
    'confidence_threshold': 0.6,
    'save_all_candidates': True,
    'max_results_per_product': 50,
    'max_products': 5,  # Default to 5 products for debugging
    
    # File paths
    'amazon_items_file': 'data/amazon_items.json',  # Changed from input_file to match main.py
    'output_file': 'data/matches/amazon_alibaba_matches.json',  # Added output_file path
    'log_file': 'product_matching.log',
    
    # Logging
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}

def get_config():
    """Get the configuration settings"""
    return DEFAULT_CONFIG