# Harmonya Data Task - Product Matching Solution

## TLDR
An intelligent product matching system that finds equivalent products between Amazon and Alibaba using advanced text analysis and similarity scoring. The system provides confidence scores for each match to help identify the most reliable product pairs.

### Quick Start with Docker
```bash
docker build -t product-matcher . && docker run product-matcher
```

## Overview
This project develops a solution to match Amazon products with their Alibaba counterparts using text-based analysis, and provides a confidence score for each match. The goal is to determine if an Amazon product can be found on Alibaba and with what degree of certainty.

## Solution Architecture

### Key Components:
1. **Query Generation**: Creating optimized search strings for Alibaba
2. **Matching**: Finding potential counterparts on Alibaba
3. **Confidence Scoring**: Evaluating match quality
4. **Threshold Decision**: Determining whether a match is valid


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

## Production Deployment Guide

### Serving in Production

1. **Containerized Deployment**
   - Utilize Docker containers for consistent environments
   - Implement container orchestration (e.g., Kubernetes) for scalability
   - Use rolling updates for zero-downtime deployments

2. **Microservices Architecture**
   - Split components into independent services:
     - Query Generation Service
     - Search Service
     - Matching Service
     - Confidence Scoring Service
   - Use API Gateway for request routing and load balancing

3. **Infrastructure**
   - Deploy across multiple regions for lower latency
   - Implement auto-scaling based on load
   - Use managed services where possible (e.g., managed Kubernetes)

### Production Challenges and Solutions

1. **API Rate Limits**
   - Challenge: Alibaba API has strict rate limits
   - Solutions:
     - Implement intelligent rate limiting
     - Use request queuing and batch processing
     - Cache frequently requested products
     - Consider multiple API keys rotation

2. **Data Consistency**
   - Challenge: Product information changes frequently
   - Solutions:
     - Implement regular data synchronization
     - Use event-driven updates
     - Maintain version control for product data
     - Set up data validation pipelines

3. **Performance at Scale**
   - Challenge: Processing large product catalogs
   - Solutions:
     - Implement caching layers (Redis/Memcached)
     - Use database indexing strategies
     - Optimize query patterns
     - Consider data partitioning

4. **Error Handling**
   - Challenge: External service failures
   - Solutions:
     - Implement circuit breakers
     - Use retry mechanisms with exponential backoff
     - Maintain fallback options
     - Set up comprehensive error logging

### Customer Satisfaction Strategies

1. **Reliability**
   - Implement health checks and monitoring
   - Set up automated failover mechanisms
   - Maintain backup systems
   - Regular disaster recovery testing

2. **Performance Optimization**
   - Monitor and optimize response times
   - Implement request prioritization
   - Use CDN for static content
   - Regular performance audits

3. **Quality Assurance**
   - Continuous monitoring of match quality
   - Regular validation of confidence scores
   - A/B testing for matching algorithms
   - Customer feedback integration

4. **Support and Monitoring**
   - 24/7 system monitoring
   - Real-time alerting system
   - Comprehensive logging
   - Regular reporting and analytics

## Future Enhancements

### 1. Machine Learning Integration
- **Feature Engineering**: Transform product attributes into feature vectors
- **Model Training**: Train a model to predict match quality

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
- [ ] Regenerate new query in case Amazon product not exists on Alibaba (empty result)
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

This solution provides a robust framework for matching Amazon products with Alibaba counterparts. The 
design allows for flexible expansion of functionality and easy maintenance.

Key advantages of this implementation:

1. **Modularity**: Components can be updated or replaced independently
2. **Extensibility**: New strategies can be added without modifying existing code
3. **Configurability**: System behavior can be adjusted through configuration
4. **Maintainability**: Clear separation of concerns and well-defined interfaces

The confidence scoring system allows for flexible thresholding to balance precision and recall based on business requirements, and the architecture is designed to accommodate future machine learning enhancements.

