"""
Search service for Alibaba product search
"""

import logging
from typing import Dict

from src.services.alibaba_search import alibaba_get_search_result_titles

logger = logging.getLogger("ProductMatcher")

class AlibabaSearchService:
    """Service class for interacting with Alibaba search"""
    
    def search(self, query: str) -> Dict[str, str]:
        """
        Search for products on Alibaba
        
        Args:
            query: The search query to use
            
        Returns:
            Dictionary mapping Alibaba product IDs to their titles
        """
        try:
            results = alibaba_get_search_result_titles(query)
            return results
        except Exception as e:
            logger.error(f"Error searching Alibaba: {str(e)}")
            return {}