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