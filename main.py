"""
Main entry point for the product matching service
"""

import sys
import json

from config.default_config import get_config
from src.text.processor import TextProcessor
from src.strategies import QueryStrategyFactory, ConfidenceScorerFactory, ScorerStrategyType
from src.services.matcher import ProductMatchingService
from src.services.search import AlibabaSearchService
from src.utils.logging_config import setup_logging

def create_matching_service(config):
    """Create and configure the product matching service"""
    text_processor = TextProcessor()
    query_strategy = QueryStrategyFactory.create_strategy(
        config['query_strategy_type'],
        text_processor
    )
    confidence_scorer = ConfidenceScorerFactory.create_scorer(
        config['confidence_scorer_type'],
        text_processor
    )
    search_service = AlibabaSearchService()
    
    return ProductMatchingService(
        text_processor,
        query_strategy,
        confidence_scorer,
        search_service
    )

def main():
    """Main function"""
    # Load configuration
    config = get_config()
    
    # Set up logging
    logger = setup_logging(config)
    
    # Create matching service
    matcher = create_matching_service(config)
    
    # Find matches
    results = matcher.find_matches(
        config['amazon_items_file'],
        confidence_threshold=config['confidence_threshold'],
        max_products=config['max_products']
    )
    
    if results:
        # Save matches
        matcher.save_matches(results, config['output_file'])
        
        # Log summary statistics
        matcher.log_summary_statistics()
        
        # Analyze threshold impact
        matcher.analyze_threshold_impact(results)
    else:
        logger.error("Failed to find matches")
        sys.exit(1)

if __name__ == "__main__":
    main()