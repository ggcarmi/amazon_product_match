"""
Alibaba search API interface for product search (Development Mock Version)
"""
import logging
import json
import requests
import time
from typing import Dict, Optional
import random

logger = logging.getLogger("ProductMatcher")

class AlibabaAPIError(Exception):
    """Custom exception for Alibaba API errors"""
    pass


def alibaba_get_search_result_titles(search_string):
    # Alibaba string search is limited to 50 char max
    search_text = search_string.replace(" ","+")[:50]
    url = f"https://open-s.alibaba.com/openservice/galleryProductOfferResultViewService?appName=magellan&appKey=a5m1ismomeptugvfmkkjnwwqnwyrhpb1&searchweb=Y&fsb=y&IndexArea=product_en&CatId=&SearchText={search_text}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            response_json = json.loads(response.text)
            return {str(x['information']['id']):x['information']['puretitle'] for x in response_json['data']['offerList']}
        except:
            pass


# def alibaba_get_search_result_titles(search_string: str) -> Optional[Dict[str, str]]:
#     """
#     Mock search function that returns simulated Alibaba product results
    
#     Args:
#         search_string: The search query to use
        
#     Returns:
#         Dictionary mapping product IDs to their titles
#     """
#     # Simulate API delay
#     time.sleep(0.5)
    
#     # Generate a deterministic but varied number of results
#     num_results = random.randint(3, 10)
    
#     # Create mock results based on the search string
#     results = {}
#     search_terms = search_string.split()
    
#     for i in range(num_results):
#         # Create a varied mock product ID
#         product_id = f"{random.randint(1600000000000, 1601999999999)}"
        
#         # Create a mock title using some of the search terms
#         used_terms = random.sample(search_terms, min(len(search_terms), 3))
#         additional_words = [
#             "Premium", "Professional", "High Quality", "New", "2024",
#             "Wholesale", "Custom", "Original", "Best"
#         ]
        
#         title_parts = (
#             random.sample(additional_words, 2) +
#             used_terms +
#             [["Product", "Item", "Solution"][random.randint(0, 2)]]
#         )
        
#         mock_title = " ".join(title_parts)
#         results[product_id] = mock_title
        
#     logger.info(f"Mock search found {len(results)} results for query: {search_string}")
#     return results

def enum_amazon_items(items_file_path: str):
    """Utility function to enumerate Amazon items from a JSON file"""
    try:
        with open(items_file_path, "r", encoding='utf-8') as f:
            items_dict = json.load(f)
            for asin, item in items_dict.items():
                yield item['title']
    except Exception as e:
        logger.error(f"Error reading Amazon items file: {str(e)}")
        raise