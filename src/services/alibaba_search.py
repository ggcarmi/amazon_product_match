"""
Alibaba search API interface for product search
"""

import logging
import json
import time
from typing import Dict, Optional
import requests
from requests.exceptions import RequestException

logger = logging.getLogger("ProductMatcher")

class AlibabaAPIError(Exception):
    """Custom exception for Alibaba API errors"""
    pass

def alibaba_get_search_result_titles(search_string: str) -> Optional[Dict[str, str]]:
    """
    Search for products on Alibaba using their open API
    
    Args:
        search_string: The search query to use
        
    Returns:
        Dictionary mapping product IDs to their titles, or None if the request fails
        
    Raises:
        AlibabaAPIError: If there's an error with the API request
    """
    # Constants
    BASE_URL = "https://open-s.alibaba.com/openservice/galleryProductOfferResultViewService"
    APP_KEY = "a5m1ismomeptugvfmkkjnwwqnwyrhpb1"  # TODO: Move to config
    MAX_QUERY_LENGTH = 50
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    try:
        # Clean and format search query
        search_text = search_string.replace(" ", "+")[:MAX_QUERY_LENGTH]
        logger.debug(f"Formatted search query: {search_text}")
        
        # Construct request URL with parameters
        params = {
            'appName': 'magellan',
            'appKey': APP_KEY,
            'searchweb': 'Y',
            'fsb': 'y',
            'IndexArea': 'product_en',
            'CatId': '',
            'SearchText': search_text
        }
        
        # Implement retry logic
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Searching Alibaba API (attempt {attempt + 1}/{MAX_RETRIES})")
                response = requests.get(BASE_URL, params=params, timeout=10)
                
                if response.status_code == 200:
                    response_json = response.json()
                    
                    # Check if the response contains the expected data structure
                    if 'data' in response_json and 'offerList' in response_json['data']:
                        results = {
                            str(item['information']['id']): item['information']['puretitle']
                            for item in response_json['data']['offerList']
                            if 'information' in item and 'id' in item['information'] and 'puretitle' in item['information']
                        }
                        
                        logger.info(f"Found {len(results)} results for query: {search_text}")
                        return results
                    else:
                        logger.warning(f"Unexpected API response structure: {response_json}")
                        return {}
                        
                elif response.status_code == 429:  # Too Many Requests
                    logger.warning("Rate limit exceeded, retrying after delay")
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    continue
                    
                else:
                    logger.error(f"API request failed with status code: {response.status_code}")
                    response.raise_for_status()
                    
            except requests.Timeout:
                logger.warning("Request timed out, retrying...")
                time.sleep(RETRY_DELAY)
                continue
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse API response: {str(e)}")
                raise AlibabaAPIError(f"Invalid JSON response: {str(e)}")
                
            except RequestException as e:
                logger.error(f"Request error: {str(e)}")
                time.sleep(RETRY_DELAY)
                continue
                
        logger.error("Max retries exceeded")
        return {}
        
    except Exception as e:
        logger.error(f"Unexpected error in Alibaba search: {str(e)}")
        raise AlibabaAPIError(f"Search failed: {str(e)}")

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