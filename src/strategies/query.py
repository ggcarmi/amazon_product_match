"""
Query generation strategies for product matching
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from src.text.processor import TextProcessor

class QueryStrategy(ABC):
    """Abstract base class for query generation strategies"""
    
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
    
    @abstractmethod
    def generate_query(self, amazon_item: Dict[str, Any], max_length: int = 50) -> str:
        """Generate a search query based on the strategy"""
        pass


class BrandModelQueryStrategy(QueryStrategy):
    """Strategy that prioritizes brand and model information"""
    
    def generate_query(self, amazon_item: Dict[str, Any], max_length: int = 50) -> str:
        title = amazon_item.get('title', '')
        brand_model = self.text_processor.extract_brand_and_model(title)
        
        # Ensure max length
        if len(brand_model) > max_length:
            brand_model = brand_model[:max_length]
            
        return brand_model


class KeyTermsQueryStrategy(QueryStrategy):
    """Strategy that uses key terms extraction"""
    
    def generate_query(self, amazon_item: Dict[str, Any], max_length: int = 50) -> str:
        title = amazon_item.get('title', '')
        key_terms = self.text_processor.extract_key_terms(title, max_length)
        return key_terms


class CategoryEnhancedQueryStrategy(QueryStrategy):
    """Strategy that incorporates category information"""
    
    def generate_query(self, amazon_item: Dict[str, Any], max_length: int = 50) -> str:
        title = amazon_item.get('title', '')
        categories = amazon_item.get('catagories', [])
        main_category = categories[0] if categories else ""
        
        # Extract key terms with space for the category
        remaining_length = max_length - len(main_category) - 1  # -1 for space
        if remaining_length > 0:
            key_terms = self.text_processor.extract_key_terms(title, remaining_length)
            query = f"{key_terms} {main_category}".strip()
        else:
            query = self.text_processor.extract_key_terms(title, max_length)
            
        return query


class DetailsEnhancedQueryStrategy(QueryStrategy):
    """Strategy that incorporates product specification details"""
    
    def generate_query(self, amazon_item: Dict[str, Any], max_length: int = 50) -> str:
        title = amazon_item.get('title', '')
        details = amazon_item.get('details', {})
        
        # Extract key specs
        specs = ""
        for key, value in details.items():
            if key in ['Brand', 'Model', 'Part Number'] and isinstance(value, str):
                specs += f" {value}"
        
        specs = specs.strip()
        
        # Combine specs with title-based key terms
        remaining_length = max_length - len(specs) - 1  # -1 for space
        if specs and remaining_length > 0:
            key_terms = self.text_processor.extract_key_terms(title, remaining_length)
            query = f"{specs} {key_terms}".strip()
        else:
            query = self.text_processor.extract_key_terms(title, max_length)
        
        # Ensure max length
        if len(query) > max_length:
            query = query[:max_length]
            
        return query


class HybridQueryStrategy(QueryStrategy):
    """Strategy that combines multiple approaches for optimal query generation"""
    
    def generate_query(self, amazon_item: Dict[str, Any], max_length: int = 50) -> str:
        title = amazon_item.get('title', '')
        
        # 1. Extract brand and specific model number if available
        brand_model = self.text_processor.extract_brand_and_model(title)
        
        # 2. Get product specs from details if available
        details = amazon_item.get('details', {})
        specs = ""
        for key, value in details.items():
            if key in ['Brand', 'Model', 'Part Number'] and isinstance(value, str):
                specs += f" {value}"
        
        # 3. Use category for context if needed
        categories = amazon_item.get('catagories', [])
        main_category = categories[0] if categories else ""
        
        # Combine key information
        initial_query = f"{brand_model} {specs}".strip()
        remaining_length = max_length - len(initial_query) - 1  # -1 for space
        
        # Add key terms if there's space
        if remaining_length > 10:  # Only add if there's meaningful space
            key_terms = self.text_processor.extract_key_terms(
                title, 
                remaining_length
            )
            query = f"{initial_query} {key_terms}".strip()
        else:
            query = initial_query
        
        # Ensure max length
        if len(query) > max_length:
            query = query[:max_length]
            
        return query


class QueryStrategyFactory:
    """Factory class for creating query generation strategies"""
    
    @staticmethod
    def create_strategy(strategy_type: str, text_processor: TextProcessor) -> QueryStrategy:
        """Create a strategy instance based on the provided type"""
        from src.strategies import QueryStrategyType
        
        if strategy_type == QueryStrategyType.BRAND_MODEL.value:
            return BrandModelQueryStrategy(text_processor)
        elif strategy_type == QueryStrategyType.KEY_TERMS.value:
            return KeyTermsQueryStrategy(text_processor)
        elif strategy_type == QueryStrategyType.CATEGORY.value:
            return CategoryEnhancedQueryStrategy(text_processor)
        elif strategy_type == QueryStrategyType.DETAILS.value:
            return DetailsEnhancedQueryStrategy(text_processor)
        elif strategy_type == QueryStrategyType.HYBRID.value:
            return HybridQueryStrategy(text_processor)
        else:
            # Default to hybrid
            return HybridQueryStrategy(text_processor)