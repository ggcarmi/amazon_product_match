# Harmonya Data Task - Product Matching Solution

## Overview
This project develops a solution to match Amazon products with their Alibaba counterparts using text-based analysis, and provides a confidence score for each match. The goal is to determine if an Amazon product can be found on Alibaba and with what degree of certainty.

## Solution Architecture

### Key Components:
1. **Query Generation**: Creating optimized search strings for Alibaba
2. **Matching**: Finding potential counterparts on Alibaba
3. **Confidence Scoring**: Evaluating match quality
4. **Threshold Decision**: Determining whether a match is valid

## Object-Oriented Design

The solution implements several design patterns to ensure a maintainable, extensible architecture:

### 1. Strategy Pattern
Used in two key areas:
- **Query Generation**: Different strategies for constructing search queries
- **Confidence Scoring**: Various approaches to evaluate match quality

```
┌─────────────────┐          ┌───────────────────┐
│                 │          │                   │
│  Client Code    │◄─────────┤  Strategy         │
│                 │          │  Interface        │
└────────┬────────┘          └─────────┬─────────┘
         │                             ▲
         │                             │
         │                   ┌─────────┼─────────┐
         │                   │         │         │
         │           ┌───────┴───┐ ┌───┴─────┐ ┌─┴────────┐
         │           │           │ │         │ │          │
         └───────────► Strategy A │ │Strategy B│ │Strategy C│
                     │           │ │         │ │          │
                     └───────────┘ └─────────┘ └──────────┘
```

### 2. Factory Pattern
Responsible for creating strategy objects without exposing implementation details:
- **QueryStrategyFactory**: Creates query generation strategies
- **ConfidenceScorerFactory**: Creates confidence scoring strategies

```
┌────────────┐     creates      ┌───────────┐
│            │  ─────────────►  │           │
│   Factory  │                  │  Product  │
│            │  ◄─────────────  │           │
└────────────┘     returns      └───────────┘
```

### 3. Facade Pattern
Simplifies interaction with the complex subsystem:
- **ProductMatchingFacade**: Provides a simple interface to the entire matching system

```
       ┌───────────────┐
       │  Client Code  │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐         ┌─────────────┐
       │               │         │             │
       │    Facade     │───────► │ Subsystem A │
       │               │         │             │
       └───────┬───────┘         └─────────────┘
               │                        ▲
               │                        │
               │                 ┌──────┴──────┐
               │                 │             │
               └────────────────►│ Subsystem B │
                                 │             │
                                 └─────────────┘
```

### 4. Service Layer Pattern
Encapsulates business logic and application services:
- **ProductMatchingService**: Orchestrates the product matching workflow
- **AlibabaSearchService**: Handles interaction with external search API

### Class Diagram

```
┌──────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│                  │     │                   │     │                     │
│  TextProcessor   │◄────┤  QueryStrategy    │◄────┤  QueryStrategyFactory│
│                  │     │    (abstract)     │     │                     │
└──────────────────┘     └───────────────────┘     └─────────────────────┘
         ▲                        ▲
         │                        │
         │                ┌───────┴─────────────────────┐
         │                │                             │
         │        ┌───────────────┐           ┌─────────────────┐
         │        │               │           │                 │
         │        │BrandModel     │           │CategoryEnhanced │
         │        │QueryStrategy  │           │QueryStrategy    │
         │        │               │           │                 │
         │        └───────────────┘           └─────────────────┘
┌────────┴───────┐
│                │
│ConfidenceScorer│
│  (abstract)    │◄──────────┐
│                │           │
└────────────────┘           │
        ▲           ┌────────┴───────────┐
        │           │                    │
        │           │ConfidenceScorer    │
┌───────┴────────┐  │Factory             │
│                │  │                    │
│HybridScorer    │  └────────────────────┘
│                │
└────────────────┘
        ▲
        │           ┌────────────────────┐
        │           │                    │
        │           │ProductMatching     │
        │           │Facade              │
        └───────────┤                    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │                    │
                    │ProductMatching     │
                    │Service             │
                    │                    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │                    │
                    │AlibabaSearch       │
                    │Service             │
                    │                    │
                    └────────────────────┘
```

## Approach and Methodology

### 1. Query Generation
Due to Alibaba's 50-character search limit, generating an effective query is critical. Our approach:

- **Brand & Model Extraction**: Identify and prioritize brand names and model numbers
- **Key Term Selection**: Use TF-IDF and frequency analysis to extract the most distinctive product terms
- **Multi-query Strategy**: If initial search yields poor results, try alternative queries
- **Detail Enrichment**: Incorporate product specifications from structured data when available

#### Query Strategy Implementations:

1. **BrandModelQueryStrategy**: Focuses on extracting and using brand and model information
2. **KeyTermsQueryStrategy**: Uses frequency analysis to identify the most important terms
3. **CategoryEnhancedQueryStrategy**: Incorporates category information into the query
4. **DetailsEnhancedQueryStrategy**: Uses product specification details from structured data
5. **HybridQueryStrategy**: Combines multiple approaches for optimal results

### 2. Text Processing
- **Cleaning**: Remove punctuation, convert to lowercase
- **Normalization**: Apply lemmatization to standardize word forms
- **Stopword Removal**: Filter out common non-informative words
- **Tokenization**: Break text into meaningful units for analysis

### 3. Confidence Scoring Model
The confidence score uses multiple factors:

- **Title Similarity**: TF-IDF vectorization with cosine similarity (70% weight)
- **Length Factor**: Comparison of title lengths (20% weight)
- **Category Factor**: Relevance of product categories (10% weight)
- **Visual Similarity**: Image-based confirmation (future enhancement)

#### Confidence Scorer Implementations:

1. **TitleSimilarityScorer**: Focuses solely on title similarity metrics
2. **HybridScorer**: Combines multiple metrics for a more robust scoring

### 4. Match Decision
- Confidence threshold of 0.6 (adjustable via configuration)
- Results saved to structured JSON for further analysis

## Confidence Threshold Determination

The confidence threshold is a critical parameter in the product matching system that directly impacts the tradeoff between precision and recall:

- **Higher threshold** → More precise matches with fewer false positives, but potentially missing valid matches
- **Lower threshold** → More comprehensive coverage (higher recall), but potentially including incorrect matches

### Threshold Selection Methodology

We use an empirical, data-driven approach to determine the optimal threshold:

1. **Distribution Analysis**: Analyzing the distribution of confidence scores across all potential matches
2. **Manual Verification**: Sampling matches at different confidence levels and manually verifying accuracy
3. **Precision-Recall Tradeoff**: Plotting precision vs. recall at different thresholds
4. **Business Impact Assessment**: Evaluating the cost of false positives vs. false negatives

### Our Recommended Thresholds

Based on our analysis, we recommend these threshold levels for different business needs:

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.8+      | Very High | Low    | When accuracy is critical (e.g., automatic purchasing) |
| 0.6-0.79  | High      | Medium | Balanced approach for most use cases |
| 0.4-0.59  | Medium    | High   | When comprehensive coverage is important (e.g., research) |

### How Confidence Scores Are Calculated

Our confidence score combines multiple factors with specific weights:

1. **Title Similarity (70%)**: TF-IDF vectorization with cosine similarity
   - Measures semantic similarity between product titles
   - Handles variations in word order and minor differences

2. **Length Factor (20%)**: Comparison of title lengths
   - Penalizes significant differences in title length
   - Rewards completeness of information

3. **Category Factor (10%)**: Relevance of product categories
   - Ensures products belong to similar categories
   - Prevents cross-category mismatches

The weighted combination of these factors produces a score between 0 and 1, where:
- **0.8-1.0**: Very high confidence (likely the same product)
- **0.6-0.79**: High confidence (probably the same product with variations)
- **0.4-0.59**: Medium confidence (possibly related products)
- **<0.4**: Low confidence (likely different products)

### Threshold Visualization

```
Confidence Score Distribution
────────────────────────────────────
│                                  │
│                                  │
│                                  │
│         ┌───┐                    │
│         │   │                    │
│         │   │                    │
│     ┌───┤   │                    │
│     │   │   │                    │
│     │   │   │   ┌───┐            │
│     │   │   │   │   │            │
│ ┌───┤   │   │   │   │   ┌───┐    │
│ │   │   │   │   │   │   │   │    │
│ │   │   │   │   │   │   │   │    │
└─┴───┴───┴───┴───┴───┴───┴───┴────┘
  0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
                ▲
           Our default
           threshold
```

### Adjusting the Threshold

The threshold can be easily adjusted in the configuration:

```python
matcher = ProductMatcher({
    'confidence_threshold': 0.6,  # Adjust based on your needs
    # Other configuration options
})
```

## Processing Flow Diagram

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Amazon       │     │   Query        │     │   Alibaba      │
│   Product Data │────▶│   Generation   │────▶│   Search API   │
└────────────────┘     └────────────────┘     └───────┬────────┘
                                                     │
┌────────────────┐     ┌────────────────┐     ┌──────▼────────┐
│ Final Matching │     │   Threshold    │     │ Confidence    │
│ Results        │◀────│   Application  │◀────│ Calculation   │
└────────────────┘     └────────────────┘     └────────────────┘
```

## Detailed User Workflow

```
User                  ProductMatchingFacade         ProductMatchingService         QueryStrategy            AlibabaSearchService       ConfidenceScorer
 |                           |                              |                           |                           |                       |
 |--match_products()-------->|                              |                           |                           |                       |
 |                           |                              |                           |                           |                       |
 |                           |--find_matches()------------->|                           |                           |                       |
 |                           |                              |--Load Amazon items        |                           |                       |
 |                           |                              |                           |                           |                       |
 |                           |                              | [For each Amazon item]    |                           |                       |
 |                           |                              |--generate_query()-------->|                           |                       |
 |                           |                              |<----optimized query-------|                           |                       |
 |                           |                              |                           |                           |                       |
 |                           |                              |--search(query)--------------------------->|           |                       |
 |                           |                              |<----search results------------------------|           |                       |
 |                           |                              |                           |                           |                       |
 |                           |                              | [If no results found]     |                           |                       |
 |                           |                              |--generate alt query------>|                           |                       |
 |                           |                              |<----simplified query------|                           |                       |
 |                           |                              |--search(query)--------------------------->|           |                       |
 |                           |                              |<----search results------------------------|           |                       |
 |                           |                              |                           |                           |                       |
 |                           |                              | [For each result]         |                           |                       |
 |                           |                              |--calculate_confidence-------------------------------->|                       |
 |                           |                              |<----confidence score----------------------------------|                       |
 |                           |                              |                           |                           |                       |
 |                           |                              |--Apply threshold          |                           |                       |
 |                           |                              |                           |                           |                       |
 |                           |<-----matches list------------|                           |                           |                       |
 |                           |                              |                           |                           |                       |
 |                           |--save_matches()              |                           |                           |                       |
 |<-----matches--------------|                              |                           |                           |                       |
 |                           |                              |                           |                           |                       |
```

## Implementation Details

### Dependencies
- `nltk`: For text processing and tokenization
- `sklearn`: For TF-IDF vectorization and cosine similarity calculation
- `numpy`: For numerical operations
- `typing`: For type hints and code clarity
- `abc`: For abstract base classes and interfaces

### Key Classes

1. **TextProcessor**
   - Handles all text normalization and processing
   - Extracts key terms and important product identifiers

2. **QueryStrategy** (Abstract Base Class)
   - Defines interface for query generation strategies
   - Multiple concrete implementations for different approaches

3. **ConfidenceScorer** (Abstract Base Class)
   - Defines interface for scoring match quality
   - Concrete implementations with different scoring algorithms

4. **Factory Classes**
   - Create appropriate strategy instances based on configuration
   - Allow for runtime selection of algorithms

5. **Service Classes**
   - Encapsulate business logic
   - Provide clean interfaces for subsystem operations

6. **ProductMatchingFacade**
   - Simplifies usage of the entire system
   - Configurable through a single configuration dictionary

## Production Considerations

### Scaling Challenges
1. **Rate Limiting**: External APIs may impose request limits
   - Solution: Implement request queuing and backoff strategies
   
2. **Processing Volume**: Large catalogs require significant processing power
   - Solution: Distributed processing using task queues (Celery, AWS SQS)
   
3. **Data Updates**: Product catalogs change frequently
   - Solution: Incremental processing with change detection

### Performance Optimization
1. **Caching**: Store previous search results and matches
2. **Preprocessing**: Maintain cleaned and vectorized text representations
3. **Batch Processing**: Group operations for efficiency
4. **Database Indexing**: Optimize for fast lookups of previous matches

### Reliability
1. **Error Handling**: Robust exception management for API failures
2. **Logging**: Comprehensive activity tracking
3. **Monitoring**: Real-time performance metrics
4. **Testing**: Regular validation of match quality

### Architecture for Production

![Production Architecture](https://i.imgur.com/8WaSePf.png)

#### Containerized Deployment

```
┌─────────────────────────────────────────────────┐
│                   Kubernetes Cluster            │
│                                                 │
│  ┌───────────────┐       ┌─────────────────┐    │
│  │               │       │                 │    │
│  │  API Service  │◄─────►│ Matching Worker │    │
│  │               │       │                 │    │
│  └───────┬───────┘       └────────┬────────┘    │
│          │                        │             │
│          ▼                        ▼             │
│  ┌───────────────┐       ┌─────────────────┐    │
│  │               │       │                 │    │
│  │  Redis Cache  │       │ Results DB      │    │
│  │               │       │                 │    │
│  └───────────────┘       └─────────────────┘    │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Future Enhancements

### 1. Machine Learning Integration
- **Feature Engineering**: Transform product attributes into feature vectors
   ```python
   def extract_features(product_data):
       # Extract numerical features
       price = float(product_data.get('price', 0))
       title_length = len(product_data.get('title', ''))
       
       # Extract text features using TF-IDF
       text_features = vectorizer.transform([product_data.get('title', '')])
       
       # Combine features
       return np.concatenate([
           [price, title_length], 
           text_features.toarray()[0]
       ])
   ```

- **Model Training**: Example model pipeline
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('classifier', RandomForestClassifier(n_estimators=100))
   ])
   
   # Train model on labeled matches
   pipeline.fit(X_train, y_train)
   
   # Predict match probability
   match_probability = pipeline.predict_proba(X_test)[:, 1]
   ```

### 2. Visual Similarity Analysis
- Extract image features using pre-trained CNNs
- Compare product images for visual similarity
- Example architecture:

```
┌─────────────────┐     ┌────────────────┐     ┌────────────────┐
│                 │     │                │     │                │
│  Product Image  │────►│  CNN Feature   │────►│  Cosine        │
│                 │     │  Extractor     │     │  Similarity    │
└─────────────────┘     └────────────────┘     └────────────────┘
```

### 3. Price Analysis
- Implement price normalization across currencies
- Detect statistical outliers in price comparisons
- Example implementation:

```python
def analyze_price_correlation(amazon_price, alibaba_price, exchange_rate=1.0):
    normalized_alibaba_price = alibaba_price * exchange_rate
    
    # Calculate price ratio
    if amazon_price > 0:
        price_ratio = normalized_alibaba_price / amazon_price
        
        # Higher score for closer prices
        if 0.5 <= price_ratio <= 1.5:
            return 1.0 - abs(1.0 - price_ratio) / 2.0
        else:
            return 0.0
    return 0.0
```

### 4. Automated Feedback Loop

```
┌───────────────┐     ┌─────────────┐     ┌─────────────┐
│               │     │             │     │             │
│  Predictions  │────►│  User       │────►│  Feedback   │
│               │     │  Validation │     │  Collection │
└─────┬─────────┘     └─────────────┘     └──────┬──────┘
      │                                          │
      │                                          ▼
┌─────▼─────────┐     ┌─────────────┐     ┌──────────────┐
│               │     │             │     │              │
│  Model        │◄────┤  Model      │◄────┤  Training    │
│  Deployment   │     │  Retraining │     │  Dataset     │
│               │     │             │     │              │
└───────────────┘     └─────────────┘     └──────────────┘
```

## Similar Kaggle Competitions & Datasets

### 1. [Walmart Trip Type Classification](https://www.kaggle.com/c/walmart-recruiting-trip-type-classification)
- **Relevance**: Similar challenge of matching items across different categorization systems
- **Key Techniques**: Feature engineering from product descriptions and categories
- **Lessons**: Effective balancing of text features and structured data

### 2. [Home Depot Product Search Relevance](https://www.kaggle.com/c/home-depot-product-search-relevance)
- **Relevance**: Focused on search relevance, similar to our query generation challenge
- **Key Techniques**: Advanced text preprocessing and TF-IDF weighting
- **Lessons**: Importance of handling misspellings and variations in product descriptions

### 3. [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge)
- **Relevance**: Involves product matching across different descriptions
- **Key Techniques**: Text embeddings and feature extraction from product titles
- **Winning Solutions**: Combined LSTMs with gradient boosting models

### 4. [Avito Duplicate Ads Detection](https://www.kaggle.com/c/avito-duplicate-ads-detection)
- **Relevance**: Directly applicable problem of identifying duplicate products
- **Key Techniques**: Similarity metrics for text and images
- **Top Approaches**: Ensemble models combining visual and textual features

## Implementation Plan

### Phase 1: Core Functionality (Week 1)
- [x] Text processing utilities
- [x] Query generation strategies
- [x] Basic matching algorithm
- [x] Confidence scoring

### Phase 2: Code Structure & Design Patterns (Week 2)
- [x] Implement Strategy pattern for query generation
- [x] Implement Strategy pattern for confidence scoring
- [x] Add Factory pattern for strategy creation
- [x] Create Service layer architecture
- [x] Implement Facade pattern for simplified usage

### Phase 3: Enhancements (Weeks 3-4)
- [ ] Improve brand/model extraction with NER models
  - [ ] Research suitable NER models for product entities
  - [ ] Train or fine-tune for product domain
  - [ ] Integrate with query generation strategies
- [ ] Add category-sensitive matching
  - [ ] Build category mapping between platforms
  - [ ] Implement category extraction from Alibaba titles
  - [ ] Adjust confidence scoring with category weights
- [ ] Performance optimization
  - [ ] Implement caching mechanisms
  - [ ] Optimize vector operations
  - [ ] Add batch processing capabilities

### Phase 4: Production Preparation (Weeks 5-6)
- [ ] API rate limiting and backoff
  - [ ] Implement rate limiting middleware
  - [ ] Add exponential backoff for failed requests
  - [ ] Create request queue system
- [ ] Error handling and logging
  - [ ] Add comprehensive exception handling
  - [ ] Implement structured logging
  - [ ] Create monitoring alerts
- [ ] Containerization (Docker)
  - [ ] Create Dockerfile for application
  - [ ] Set up Docker Compose for local testing
  - [ ] Prepare Kubernetes manifests

### Phase 5: Advanced Features (Weeks 7-10)
- [ ] Machine learning model development
  - [ ] Feature engineering pipeline
  - [ ] Model selection and hyperparameter tuning
  - [ ] Training and evaluation framework
- [ ] Image-based similarity
  - [ ] Image scraping and processing pipeline
  - [ ] CNN feature extractor implementation
  - [ ] Visual similarity scoring integration
- [ ] Analytics dashboard
  - [ ] Metrics collection system
  - [ ] Visualization components
  - [ ] User feedback interface

## Usage

### Basic Usage

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the matching process with default settings:
```
python solution.py
```

3. Results will be saved to `amazon_alibaba_matches.json`

### Advanced Configuration

You can configure the matching process by modifying the config dictionary:

```python
config = {
    'query_strategy_type': 'hybrid',  # Options: 'brand_model', 'key_terms', 'category', 'details', 'hybrid'
    'confidence_scorer_type': 'hybrid',  # Options: 'title_similarity', 'hybrid'
    'confidence_threshold': 0.6  # Adjust based on precision/recall needs
}

matcher = ProductMatchingFacade(config)
matches = matcher.match_products('amazon_items.json', 'matches.json')
```

### Extending the System

To add a new query generation strategy:

1. Create a new class that inherits from `QueryStrategy`
2. Implement the `generate_query` method
3. Register the new strategy in `QueryStrategyFactory`

```python
class MyCustomQueryStrategy(QueryStrategy):
    def generate_query(self, amazon_item, max_length=50):
        # Custom query generation logic
        return query

# Add to factory
@staticmethod
def create_strategy(strategy_type, text_processor):
    # ... existing code ...
    elif strategy_type == 'my_custom':
        return MyCustomQueryStrategy(text_processor)
```

## Conclusion

This solution provides a robust framework for matching Amazon products with Alibaba counterparts. The object-oriented design with strategy and factory patterns allows for flexible expansion of functionality and easy maintenance.

Key advantages of this implementation:

1. **Modularity**: Components can be updated or replaced independently
2. **Extensibility**: New strategies can be added without modifying existing code
3. **Configurability**: System behavior can be adjusted through configuration
4. **Maintainability**: Clear separation of concerns and well-defined interfaces

The confidence scoring system allows for flexible thresholding to balance precision and recall based on business requirements, and the architecture is designed to accommodate future machine learning enhancements.

