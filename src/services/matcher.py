"""
Product matching service for Amazon-Alibaba product matching
"""

import json
import time
import logging
from typing import Dict, List, Any

from src.text.processor import TextProcessor
from src.strategies.query import QueryStrategy, QueryStrategyFactory
from src.strategies.scorer import ConfidenceScorer
from src.services.search import AlibabaSearchService

logger = logging.getLogger("ProductMatcher")

class ProductMatchingService:
    """Main service class for matching Amazon products with Alibaba products"""
    
    def __init__(
        self, 
        text_processor: TextProcessor,
        query_strategy: QueryStrategy,
        confidence_scorer: ConfidenceScorer,
        search_service: AlibabaSearchService
    ):
        self.text_processor = text_processor
        self.query_strategy = query_strategy
        self.confidence_scorer = confidence_scorer
        self.search_service = search_service
        
        # Initialize results tracking
        self.match_stats = {
            'total_products': 0,
            'products_with_matches': 0,
            'products_with_high_confidence': 0,
            'products_with_medium_confidence': 0,
            'products_with_low_confidence': 0,
            'products_with_no_matches': 0,
            'total_alibaba_results': 0,
            'confidence_distribution': {},
            'query_success_rate': 0
        }
        
    def find_matches(
        self, 
        amazon_items_file: str,
        confidence_threshold: float = 0.6,
        save_all_candidates: bool = True,
        max_results_per_product: int = 10,
        max_products: int = 5  # Default to 5 products for debugging
    ) -> Dict[str, Any]:
        """
        Find matches between Amazon and Alibaba products
        
        Parameters:
        - amazon_items_file: Path to the JSON file with Amazon product data
        - confidence_threshold: Minimum confidence score required to consider a match
        - save_all_candidates: Whether to save all matching candidates
        - max_results_per_product: Maximum number of candidates to save per product
        - max_products: Maximum number of Amazon products to process (default: 5 for easier debugging)
        
        Returns:
        - Dictionary with matches, all product data, and statistics
        """
        # Load Amazon items
        try:
            with open(amazon_items_file, 'r', encoding='utf-8') as f:
                amazon_items = json.load(f)
        except Exception as e:
            logger.error(f"Error loading Amazon items: {str(e)}")
            return None
        
        # Initialize results
        all_matches = []
        all_products_data = []
        successful_queries = 0
        
        # Process each Amazon item
        self.match_stats['total_products'] = len(amazon_items)
        
        # Get the items to process (limited by max_products if specified)
        amazon_items_to_process = list(amazon_items.items())
        if max_products > 0:
            amazon_items_to_process = amazon_items_to_process[:max_products]
            logger.info(f"Limited to processing {max_products} out of {len(amazon_items)} Amazon products")
        
        logger.info(f"Starting to process {len(amazon_items_to_process)} Amazon products")
        
        # Process each Amazon item
        for idx, (asin, amazon_item) in enumerate(amazon_items_to_process, 1):
            logger.info(f"\n\n Processing item {idx}/{len(amazon_items_to_process)}: {asin} - {amazon_item.get('title', '')[:50]}...")
            
            # Track product-specific results
            product_data = {
                'amazon_asin': asin,
                'amazon_title': amazon_item.get('title', ''),
                'amazon_price': amazon_item.get('price', ''),
                'amazon_category': amazon_item.get('catagories', [''])[0],
                'query': '',
                'match_found': False,
                'match_confidence': 0,
                'match_count': 0,
                'all_candidates': []
            }
            
            # Generate search query for Alibaba
            search_query = self.query_strategy.generate_query(amazon_item)
            product_data['query'] = search_query
            logger.info(f"Search query: {search_query}")
            
            # Search on Alibaba
            alibaba_results = self.search_service.search(search_query)
            
            # If no results, try a fallback query strategy
            if not alibaba_results:
                fallback_strategy = QueryStrategyFactory.create_strategy('key_terms', self.text_processor)
                fallback_query = fallback_strategy.generate_query(amazon_item, 40)
                logger.info(f"No results found. Trying fallback query: {fallback_query}")
                product_data['fallback_query'] = fallback_query
                alibaba_results = self.search_service.search(fallback_query)
            
            # Process search results
            if alibaba_results:
                successful_queries += 1
                self.match_stats['total_alibaba_results'] += len(alibaba_results)
                
                logger.info(f"Found {len(alibaba_results)} potential matches")
                product_data['match_count'] = len(alibaba_results)
                
                # Calculate confidence for each result
                best_match = None
                highest_confidence = 0
                all_candidates = []
                
                for alibaba_id, alibaba_title in alibaba_results.items():
                    # Calculate confidence score
                    confidence = self.confidence_scorer.calculate_confidence(amazon_item, alibaba_title)
                    confidence_category = self.confidence_scorer.categorize_confidence(confidence)
                    
                    # Track in confidence distribution
                    conf_rounded = round(confidence, 1)
                    self.match_stats['confidence_distribution'][conf_rounded] = \
                        self.match_stats['confidence_distribution'].get(conf_rounded, 0) + 1
                    
                    # Log match candidate
                    logger.info(f"  Alibaba item: {alibaba_id}, Confidence: {confidence:.2f} - {confidence_category}")
                    
                    # Save match candidate
                    candidate = {
                        'alibaba_id': alibaba_id,
                        'alibaba_title': alibaba_title,
                        'confidence': confidence,
                        'confidence_category': confidence_category
                    }
                    all_candidates.append(candidate)
                    
                    # Update best match if higher confidence
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_match = candidate
                
                # Sort candidates by confidence
                all_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Save top candidates
                if save_all_candidates:
                    product_data['all_candidates'] = all_candidates[:max_results_per_product]
                
                # If we found a match with sufficient confidence
                if best_match and best_match['confidence'] >= confidence_threshold:
                    match_data = {
                        'amazon_asin': asin,
                        'amazon_title': amazon_item.get('title', ''),
                        'amazon_price': amazon_item.get('price', ''),
                        'amazon_category': amazon_item.get('catagories', [''])[0],
                        'alibaba_id': best_match['alibaba_id'],
                        'alibaba_title': best_match['alibaba_title'],
                        'confidence': best_match['confidence'],
                        'confidence_category': best_match['confidence_category']
                    }
                    all_matches.append(match_data)
                    
                    product_data['match_found'] = True
                    product_data['match_confidence'] = best_match['confidence']
                    product_data['top_match'] = best_match
                    
                    logger.info(f"\n\n ✓ Match found with confidence {best_match['confidence']:.2f}")
                    
                    # Update stats for matched products
                    self.match_stats['products_with_matches'] += 1
                    if best_match['confidence_category'] == 'high':
                        self.match_stats['products_with_high_confidence'] += 1
                    elif best_match['confidence_category'] == 'medium':
                        self.match_stats['products_with_medium_confidence'] += 1
                    else:
                        self.match_stats['products_with_low_confidence'] += 1
                else:
                    logger.info(f"\n\n ✗ No match with sufficient confidence (threshold: {confidence_threshold})")
                    if all_candidates:
                        product_data['top_candidate'] = all_candidates[0]
                
                # Log highest confidence and match status
                if best_match and best_match['confidence'] >= confidence_threshold:
                    logger.info(f"✓ Match found! Highest confidence: {highest_confidence:.3f}")
                else:
                    logger.info(f"✗ No suitable match found. Highest confidence: {highest_confidence:.3f}")
                logger.info("\n")

            else:
                logger.info("✗ No matches found")
                self.match_stats['products_with_no_matches'] += 1
            
            # Save all product data
            all_products_data.append(product_data)
            
            # Respect rate limits for the Alibaba API
            time.sleep(1)
        
        # Calculate final statistics
        if self.match_stats['total_products'] > 0:
            self.match_stats['query_success_rate'] = successful_queries / self.match_stats['total_products']
            
        # Prepare final results
        results = {
            'matches': all_matches,
            'all_products_data': all_products_data,
            'match_stats': self.match_stats
        }
        
        return results

    def save_matches(self, results: Dict[str, Any], output_file: str) -> None:
        """Save matches to a JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def log_summary_statistics(self) -> None:
        """Log summary statistics of the matching process"""
        stats = self.match_stats
        
        logger.info("\n" + "=" * 60)
        logger.info("MATCHING RESULTS SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Total Amazon products processed: {stats['total_products']}")
        logger.info(f"Products with matches above threshold: {stats['products_with_matches']} ({stats['products_with_matches']/max(stats['total_products'], 1):.1%})")
        logger.info(f"  - High confidence matches (≥0.8): {stats['products_with_high_confidence']}")
        logger.info(f"  - Medium confidence matches (≥0.6): {stats['products_with_medium_confidence']}")
        logger.info(f"  - Low confidence matches (<0.6): {stats['products_with_low_confidence']}")
        logger.info(f"Products with no matches: {stats['products_with_no_matches']}")
        logger.info(f"Total Alibaba results found: {stats['total_alibaba_results']}")
        logger.info(f"Query success rate: {stats['query_success_rate']:.1%}")
        
        logger.info("\nConfidence Score Distribution:")
        
        # Sort confidence scores for better readability
        conf_dist = sorted(stats['confidence_distribution'].items())
        for score, count in conf_dist:
            logger.info(f"  {score:.1f}: {count} matches")
            
        logger.info("=" * 60)
    
    def analyze_threshold_impact(self, results: Dict[str, Any], thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) -> None:
        """Analyze the impact of different confidence thresholds"""
        logger.info("\nTHRESHOLD ANALYSIS")
        logger.info("=" * 60)
        
        all_products = results['all_products_data']
        total_products = len(all_products)
        
        # Count matches at different thresholds
        for threshold in thresholds:
            matches_above_threshold = 0
            
            for product in all_products:
                # Check if any candidate is above threshold
                candidates = product.get('all_candidates', [])
                if candidates and candidates[0]['confidence'] >= threshold:
                    matches_above_threshold += 1
            
            percentage = matches_above_threshold / max(total_products, 1) * 100
            logger.info(f"Threshold {threshold:.1f}: {matches_above_threshold} matches ({percentage:.1f}%)")
        
        logger.info("=" * 60)