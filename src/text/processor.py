"""
Text processing operations for product matching
"""

import string
from collections import Counter
from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure required NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


class TextProcessor:
    """Class responsible for text processing operations"""
    
    def __init__(self):
        """Initialize text processing tools"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_key_terms(self, text: str, max_length: int = 50) -> str:
        """Extract the most important terms from the text"""
        if not text:
            return ""
            
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Extract important words
        words = cleaned_text.split()
        
        # Calculate word frequencies
        word_freq = Counter(words)
        
        # Sort words by frequency (higher frequency = more important)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Get top words to fit within length constraint
        important_words = []
        current_length = 0
        
        for word, freq in sorted_words:
            # +1 for space
            if current_length + len(word) + 1 <= max_length:
                important_words.append(word)
                current_length += len(word) + 1
            else:
                break
        
        return ' '.join(important_words)
    
    def extract_brand_and_model(self, text: str) -> str:
        """Try to extract brand and model information"""
        if not text:
            return ""
        
        # This is a simplified approach - in a full solution we would implement
        # more sophisticated entity recognition
        words = text.split()
        # Take first 2-3 words as they often represent brand and model
        brand_model = ' '.join(words[:3])
        return brand_model