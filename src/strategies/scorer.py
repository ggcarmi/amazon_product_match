"""
Confidence scoring strategies for product matching
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from src.text.processor import TextProcessor

logger = logging.getLogger("ProductMatcher")

class ConfidenceScorer(ABC):
    """Abstract base class for confidence scoring strategies"""
    
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
    
    @abstractmethod
    def calculate_confidence(self, amazon_item: Dict[str, Any], alibaba_title: str) -> float:
        """Calculate confidence score for a potential match"""
        pass
    
    def categorize_confidence(self, score: float) -> str:
        """Categorize confidence score into high/medium/low"""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"


class TitleSimilarityScorer(ConfidenceScorer):
    """Confidence scorer based on title similarity"""
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two text strings using TF-IDF and cosine similarity"""
        if not text1 or not text2:
            return 0.0
            
        # Clean texts
        clean_text1 = self.text_processor.clean_text(text1)
        clean_text2 = self.text_processor.clean_text(text2)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([clean_text1, clean_text2])
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def calculate_confidence(self, amazon_item: Dict[str, Any], alibaba_title: str) -> float:
        amazon_title = amazon_item.get('title', '')
        
        # Calculate title similarity
        title_similarity = self.calculate_text_similarity(amazon_title, alibaba_title)
        
        return title_similarity


class HybridScorer(ConfidenceScorer):
    """Confidence scorer combining multiple metrics including product details"""
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two text strings using TF-IDF and cosine similarity"""
        if not text1 or not text2:
            return 0.0
            
        # Clean texts
        clean_text1 = self.text_processor.clean_text(text1)
        clean_text2 = self.text_processor.clean_text(text2)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([clean_text1, clean_text2])
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating similarity: {str(e)}")
            return 0.0

    def extract_dimensions(self, dimension_str: str) -> tuple:
        """
        Extract numerical dimensions from string
        Example inputs: "5.5 x 2.8 x 0.3 inches", "140 x 71 x 7.6 mm"
        Returns tuple of (length, width, height) in default unit (inches)
        """
        try:
            # Remove units and split by x
            nums = [float(n) for n in dimension_str.lower().replace('inches', '').replace('mm', '').replace('cm', '').split('x')]
            if len(nums) >= 3:
                return tuple(nums[:3])
        except Exception:
            pass
        return (0, 0, 0)

    def calculate_dimension_similarity(self, amazon_details: Dict[str, Any]) -> float:
        """
        Calculate similarity score based on product dimensions
        TODO: Implement proper unit conversion (mm to inches, cm to inches, etc.)
        TODO: Handle different dimension formats and units
        TODO: Consider volume comparison instead of individual dimensions
        """
        if not amazon_details:
            return 0.5  # Default middle value if no details

        # Look for dimensions in various possible fields
        dimension_fields = ['dimensions', 'size', 'measurements', 'product dimensions']
        dimensions = None
        
        for field in dimension_fields:
            if field in amazon_details:
                dimensions = self.extract_dimensions(str(amazon_details[field]))
                if dimensions != (0, 0, 0):
                    break
        
        if not dimensions or dimensions == (0, 0, 0):
            return 0.5  # Default middle value if no valid dimensions
            
        # For now, just return a high confidence if we found valid dimensions
        # In a full implementation, we would:
        # 1. Convert all dimensions to same unit (e.g., inches)
        # 2. Compare dimensions considering different orientations
        # 3. Calculate volume and density if weight is available
        # 4. Handle ranges and tolerances
        return 0.8 if all(d > 0 for d in dimensions) else 0.5

    def calculate_details_similarity(self, amazon_details: Dict[str, Any]) -> float:
        """Calculate similarity score based on product details"""
        if not amazon_details:
            return 0.5  # Default middle value if no details available
        
        # Initialize score components
        detail_scores = []
        
        # Check for important product attributes
        if 'brand' in amazon_details:
            detail_scores.append(0.8)  # High confidence if brand is specified
        
        if 'model' in amazon_details:
            detail_scores.append(0.7)  # Good confidence if model is specified
            
        if 'specifications' in amazon_details:
            detail_scores.append(0.6)  # Medium confidence for specifications

        # Add dimension similarity
        dimension_score = self.calculate_dimension_similarity(amazon_details)
        detail_scores.append(dimension_score)
            
        # Calculate average score from available details
        if detail_scores:
            return sum(detail_scores) / len(detail_scores)
        return 0.5  # Default middle value
    
    def extract_price(self, price_str: str) -> float:
        """Extract numerical price value from string"""
        try:
            # Remove currency symbols and convert to float
            cleaned_price = ''.join(c for c in price_str if c.isdigit() or c == '.')
            return float(cleaned_price)
        except (ValueError, TypeError):
            return 0.0
    
    def calculate_price_similarity(self, amazon_price: str, alibaba_price: str = None) -> float:
        """Calculate similarity score based on price comparison"""
        if not amazon_price or not alibaba_price:
            return 0.5  # Default middle value if prices not available
            
        try:
            amazon_value = self.extract_price(amazon_price)
            alibaba_value = self.extract_price(alibaba_price)
            
            if amazon_value == 0 or alibaba_value == 0:
                return 0.5
                
            # Calculate price ratio
            ratio = min(amazon_value, alibaba_value) / max(amazon_value, alibaba_value)
            
            # Convert ratio to similarity score
            # 1.0 means identical prices
            # 0.0 means very different prices
            return ratio
        except Exception as e:
            logger.warning(f"Error calculating price similarity: {str(e)}")
            return 0.5
    
    def calculate_confidence(self, amazon_item: Dict[str, Any], alibaba_title: str) -> float:
        amazon_title = amazon_item.get('title', '')
        amazon_details = amazon_item.get('details', {})
        amazon_price = amazon_item.get('price')
        
        # 1. Base score from title similarity
        title_similarity = self.calculate_text_similarity(amazon_title, alibaba_title)
        
        # 2. Details similarity (now includes dimensions)
        details_similarity = self.calculate_details_similarity(amazon_details)
        
        # 3. Price similarity (simplified for now since we don't have Alibaba price)
        price_similarity = 0.5  # Default middle value
        
        # 4. Title length comparison
        title_length_factor = min(1.0, len(alibaba_title) / (len(amazon_title) + 1))
        
        # 5. Category matching (simplified)
        category_factor = 0.5  # Default middle value
        
        # Calculate final confidence score with weights
        confidence = (
            title_similarity * 0.35 +      # Title importance slightly reduced
            details_similarity * 0.30 +    # Details importance increased (includes dimensions)
            price_similarity * 0.15 +      # Price comparison
            title_length_factor * 0.1 +    # Length comparison less important
            category_factor * 0.1          # Category matching less important
        )
        
        return confidence


class ConfidenceScorerFactory:
    """Factory class for creating confidence scoring strategies"""
    
    @staticmethod
    def create_scorer(scorer_type: str, text_processor: TextProcessor) -> ConfidenceScorer:
        """Create a scorer instance based on the provided type"""
        from src.strategies import ScorerStrategyType
        
        if scorer_type == ScorerStrategyType.TITLE_SIMILARITY.value:
            return TitleSimilarityScorer(text_processor)
        elif scorer_type == ScorerStrategyType.HYBRID.value:
            return HybridScorer(text_processor)
        else:
            # Default to hybrid
            return HybridScorer(text_processor)