"""Visualization utilities for product matching analysis"""

import os
import logging
from datetime import datetime
import plotly.graph_objects as go
from typing import Dict, List, Any

logger = logging.getLogger("ProductMatcher")

def plot_confidence_distributions(results: Dict[str, Any]) -> None:
    """Plot confidence score distributions for all matches, successful matches, and non-matches"""
    all_products = results['all_products_data']
    
    # Initialize lists to store confidence scores
    all_scores = []
    match_scores = []
    non_match_scores = []
    
    # Collect scores from all products
    for product in all_products:
        candidates = product.get('all_candidates', [])
        if not candidates:
            continue
            
        # Get the highest confidence score for this product
        top_score = candidates[0]['confidence']
        all_scores.append(top_score)
        
        # Separate scores based on match status
        if product['match_found']:
            match_scores.append(top_score)
        else:
            non_match_scores.append(top_score)
    
    # Create figure
    fig = go.Figure()
    
    # Add distribution layers
    fig.add_trace(go.Histogram(
        x=all_scores,
        name='All Scores',
        opacity=0.75,
        nbinsx=20
    ))
    
    fig.add_trace(go.Histogram(
        x=match_scores,
        name='Matched Products',
        opacity=0.75,
        nbinsx=20
    ))
    
    fig.add_trace(go.Histogram(
        x=non_match_scores,
        name='Non-Matched Products',
        opacity=0.75,
        nbinsx=20
    ))
    
    # Update layout
    fig.update_layout(
        title='Confidence Score Distributions',
        xaxis_title='Confidence Score',
        yaxis_title='Count',
        barmode='overlay'
    )
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'confidence_distributions_{timestamp}.html'
    plot_path = os.path.join(plots_dir, plot_filename)
    
    # Save the plot
    try:
        fig.write_html(plot_path)
        logger.info(f"Successfully generated confidence distribution plot: {plot_path}")
        logger.info(f"Plot statistics: {len(all_scores)} total scores, {len(match_scores)} matches, {len(non_match_scores)} non-matches")
    except Exception as e:
        logger.error(f"Error saving confidence distribution plot: {str(e)}")

def plot_threshold_impact(results: Dict[str, Any], thresholds: List[float] = None) -> None:
    """Plot the impact of different confidence thresholds on match counts"""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    all_products = results['all_products_data']
    total_products = len(all_products)
    
    # Calculate matches at each threshold
    threshold_matches = []
    for threshold in thresholds:
        matches = sum(1 for product in all_products
                     if product.get('all_candidates') 
                     and product['all_candidates'][0]['confidence'] >= threshold)
        threshold_matches.append(matches)
    
    # Create figure
    fig = go.Figure()
    
    # Add line plot
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=threshold_matches,
        mode='lines+markers',
        name='Number of Matches',
        hovertemplate='Threshold: %{x:.2f}<br>Matches: %{y}<br>Match Rate: %{text:.1%}<extra></extra>',
        text=[count/total_products for count in threshold_matches]
    ))
    
    # Update layout
    fig.update_layout(
        title='Impact of Confidence Threshold on Match Count',
        xaxis_title='Confidence Threshold',
        yaxis_title='Number of Matches',
        showlegend=True
    )
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'threshold_impact_{timestamp}.html'
    plot_path = os.path.join(plots_dir, plot_filename)
    
    # Save the plot
    try:
        fig.write_html(plot_path)
        logger.info(f"Successfully generated threshold impact plot: {plot_path}")
        logger.info(f"Analyzed {len(thresholds)} threshold values from {min(thresholds)} to {max(thresholds)}")
    except Exception as e:
        logger.error(f"Error saving threshold impact plot: {str(e)}")