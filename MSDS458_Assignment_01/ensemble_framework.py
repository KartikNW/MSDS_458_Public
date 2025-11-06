# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: .jupytext-sync-ipynb//ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
"""
Diamond Price Prediction Ensemble Framework

This module contains the complete ensemble framework for diamond price prediction using:
1. A classifier to predict price tiers (low/medium/high)
2. Tier-specific regression models for accurate price prediction
3. Automatic routing of new diamonds to appropriate models

Usage:
1. Train your tier-specific models and classifier in your main notebook
2. Import this ensemble framework
3. Create ensemble instance with your trained models
4. Use for prediction on new diamond data
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf

# %%
# Import the utility module
import diamonds_utils as du


# %%
class DiamondPriceEnsemble:
    """
    Ensemble model that combines a classifier and tier-specific regression models.
    
    Workflow:
    1. Classifier predicts price tier (low/medium/high)
    2. Routes to appropriate regression model based on predicted tier
    3. Returns price prediction from the tier-specific model
    
    Args:
        classifier_model: Trained classifier model for tier prediction
        regression_models (dict): Dictionary with keys 'low', 'medium', 'high' containing trained regression models
        preprocessors (dict): Dictionary with keys 'low', 'medium', 'high', 'classifier' containing preprocessors
        tier_encoder: LabelEncoder fitted on tier labels
    """
    
    def __init__(self, classifier_model, regression_models, preprocessors, tier_encoder):
        self.classifier_model = classifier_model
        self.regression_models = regression_models  # dict with keys: 'low', 'medium', 'high'
        self.preprocessors = preprocessors  # dict with keys: 'low', 'medium', 'high', 'classifier'
        self.tier_encoder = tier_encoder  # LabelEncoder for tier labels
        self.tier_names = ['low', 'medium', 'high']
        
    def predict_tier(self, X):
        """
        Predict price tier for given features.
        
        Args:
            X (pandas.DataFrame): Diamond features
            
        Returns:
            tuple: (tier_names, tier_probabilities)
        """
        # Preprocess features using classifier preprocessor
        X_processed = self.preprocessors['classifier'].transform(X)
        
        # Get tier predictions
        tier_probs = self.classifier_model.predict(X_processed)
        tier_preds = np.argmax(tier_probs, axis=1)
        
        # Convert back to tier names
        tier_names = self.tier_encoder.inverse_transform(tier_preds)
        return tier_names, tier_probs
    
    def predict_price(self, X):
        """
        Predict price using ensemble approach for single or few samples.
        
        Args:
            X (pandas.DataFrame): Diamond features
            
        Returns:
            tuple: (price_predictions, tier_names, confidence_scores)
        """
        # Get tier predictions
        tier_names, tier_probs = self.predict_tier(X)
        
        # Initialize price predictions
        price_predictions = np.zeros(len(X))
        confidence_scores = np.zeros(len(X))
        
        # Route each sample to appropriate regression model
        for i, tier in enumerate(tier_names):
            # Preprocess features using tier-specific preprocessor
            X_tier = X.iloc[[i]]  # Single row DataFrame
            X_tier_processed = self.preprocessors[tier].transform(X_tier)
            
            # Get price prediction from tier-specific model
            price_pred = self.regression_models[tier].predict(X_tier_processed)[0]
            price_predictions[i] = price_pred
            
            # Use classifier confidence as confidence score
            confidence_scores[i] = np.max(tier_probs[i])
        
        return price_predictions, tier_names, confidence_scores
    
    def predict_batch(self, X):
        """
        Efficiently predict prices for multiple samples.
        
        Args:
            X (pandas.DataFrame): Diamond features
            
        Returns:
            tuple: (price_predictions, tier_names, confidence_scores)
        """
        # Get tier predictions for all samples
        tier_names, tier_probs = self.predict_tier(X)
        
        # Group samples by predicted tier for efficient processing
        tier_groups = {}
        for i, tier in enumerate(tier_names):
            if tier not in tier_groups:
                tier_groups[tier] = []
            tier_groups[tier].append(i)
        
        # Initialize results
        price_predictions = np.zeros(len(X))
        confidence_scores = np.zeros(len(X))
        
        # Process each tier group
        for tier, indices in tier_groups.items():
            if len(indices) > 0:
                # Get samples for this tier
                X_tier = X.iloc[indices]
                
                # Preprocess using tier-specific preprocessor
                X_tier_processed = self.preprocessors[tier].transform(X_tier)
                
                # Get predictions from tier-specific model
                tier_predictions = self.regression_models[tier].predict(X_tier_processed)
                
                # Store results
                for j, idx in enumerate(indices):
                    price_predictions[idx] = tier_predictions[j]
                    confidence_scores[idx] = np.max(tier_probs[idx])
        
        return price_predictions, tier_names, confidence_scores
    
    def evaluate_ensemble(self, X_test, y_test, low_threshold=2500, high_threshold=6000):
        """
        Evaluate the ensemble model performance.
        
        Args:
            X_test (pandas.DataFrame): Test features
            y_test (pandas.Series or array): Test prices
            low_threshold (int): Low tier threshold
            high_threshold (int): High tier threshold
            
        Returns:
            dict: Performance metrics and predictions including:
                - mae: Mean Absolute Error
                - r2: R-squared score
                - mape: Mean Absolute Percentage Error (%)
                - tier_accuracy: Tier classification accuracy
                - avg_confidence: Average prediction confidence
                - predictions: Price predictions
                - tier_predictions: Predicted tiers
                - confidence: Confidence scores
        """
        # Get predictions
        y_pred, tier_preds, confidence = self.predict_batch(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Calculate tier accuracy
        actual_tiers = du.create_price_tiers(
            pd.DataFrame({'price': y_test}), 
            low_threshold=low_threshold, 
            high_threshold=high_threshold
        )['price_tier'].values
        
        tier_accuracy = np.mean(tier_preds == actual_tiers)
        
        print(f"Ensemble Performance:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Tier Classification Accuracy: {tier_accuracy:.4f}")
        print(f"  Average Confidence: {np.mean(confidence):.4f}")
        
        return {
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'tier_accuracy': tier_accuracy,
            'avg_confidence': np.mean(confidence),
            'predictions': y_pred,
            'tier_predictions': tier_preds,
            'confidence': confidence
        }
    
    def predict_single_diamond(self, carat, cut, color, clarity, depth, table, x, y, z):
        """
        Predict price for a single diamond using the ensemble model.
        
        Args:
            carat, cut, color, clarity, depth, table, x, y, z: Diamond features
        
        Returns:
            dict: Prediction results including price, tier, and confidence
        """
        # Create DataFrame with single diamond
        diamond_data = pd.DataFrame({
            'carat': [carat],
            'cut': [cut],
            'color': [color],
            'clarity': [clarity],
            'depth': [depth],
            'table': [table],
            'x': [x],
            'y': [y],
            'z': [z]
        })
        
        # Get prediction
        price_pred, tier_pred, confidence = self.predict_price(diamond_data)
        
        return {
            'predicted_price': price_pred[0],
            'predicted_tier': tier_pred[0],
            'confidence': confidence[0],
            'features': {
                'carat': carat,
                'cut': cut,
                'color': color,
                'clarity': clarity,
                'depth': depth,
                'table': table,
                'x': x,
                'y': y,
                'z': z
            }
        }


# %%
def plot_ensemble_performance(y_test, y_pred, tier_preds, actual_tiers, confidence):
    """
    Create comprehensive visualization of ensemble performance.
    
    Args:
        y_test: Actual prices
        y_pred: Predicted prices
        tier_preds: Predicted tiers
        actual_tiers: Actual tiers
        confidence: Prediction confidence scores
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Actual vs Predicted Prices
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price')
    axes[0, 0].set_ylabel('Predicted Price')
    axes[0, 0].set_title('Actual vs Predicted Prices')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Prediction Errors
    errors = np.abs(y_test - y_pred)
    axes[0, 1].hist(errors, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Absolute Error ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Prediction Errors')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Tier Prediction Accuracy
    tier_counts = pd.Series(tier_preds).value_counts()
    actual_tier_counts = pd.Series(actual_tiers).value_counts()
    tier_comparison = pd.DataFrame({
        'Predicted': tier_counts,
        'Actual': actual_tier_counts
    }).fillna(0)
    tier_comparison.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Tier Prediction vs Actual Distribution')
    axes[1, 0].set_xlabel('Price Tier')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Confidence vs Error
    axes[1, 1].scatter(confidence, errors, alpha=0.6)
    axes[1, 1].set_xlabel('Prediction Confidence')
    axes[1, 1].set_ylabel('Absolute Error ($)')
    axes[1, 1].set_title('Confidence vs Prediction Error')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# %%
def create_results_dataframe(y_test, y_pred, tier_preds, confidence):
    """
    Create a results DataFrame for analysis.
    
    Args:
        y_test: Actual prices
        y_pred: Predicted prices
        tier_preds: Predicted tiers
        confidence: Prediction confidence scores
        
    Returns:
        pandas.DataFrame: Results summary
    """
    return pd.DataFrame({
        'Actual_Price': y_test,
        'Predicted_Price': y_pred,
        'Predicted_Tier': tier_preds,
        'Confidence': confidence,
        'Error': np.abs(y_test - y_pred),
        'Percentage_Error': np.abs((y_test - y_pred) / y_test) * 100
    })


# %%
def create_ensemble_from_trained_models(classifier_model, regression_models, preprocessors, 
                                       classifier_preprocessor, low_threshold=2500, high_threshold=6000):
    """
    Convenience function to create ensemble from trained models.
    
    Args:
        classifier_model: Trained classifier model
        regression_models (dict): Dictionary with regression models for each tier
        preprocessors (dict): Dictionary with preprocessors for each tier
        classifier_preprocessor: Preprocessor used for classifier
        low_threshold (int): Low tier threshold
        high_threshold (int): High tier threshold
        
    Returns:
        DiamondPriceEnsemble: Configured ensemble model
    """
    # Create tier encoder
    tier_encoder = LabelEncoder()
    tier_encoder.fit(['low', 'medium', 'high'])
    
    # Prepare preprocessors dictionary
    ensemble_preprocessors = {
        'low': preprocessors['low'],
        'medium': preprocessors['medium'], 
        'high': preprocessors['high'],
        'classifier': classifier_preprocessor
    }
    
    # Create and return ensemble
    return DiamondPriceEnsemble(
        classifier_model=classifier_model,
        regression_models=regression_models,
        preprocessors=ensemble_preprocessors,
        tier_encoder=tier_encoder
    )


# %%
# Example usage function
def example_usage():
    """
    Example of how to use the ensemble framework in your main notebook.
    """
    print("""
    # Example usage in your main notebook:
    
    # 1. Import the ensemble framework
    from ensemble_framework import DiamondPriceEnsemble, create_ensemble_from_trained_models
    
    # 2. Create ensemble from your trained models
    ensemble = create_ensemble_from_trained_models(
        classifier_model=classifier_model,
        regression_models=models,  # Your trained regression models dict
        preprocessors=preprocessors,  # Your trained preprocessors dict
        classifier_preprocessor=preprocessor,  # Classifier preprocessor
        low_threshold=2500,
        high_threshold=6000
    )
    
    # 3. Predict single diamond
    result = ensemble.predict_single_diamond(
        carat=1.0, cut='Ideal', color='G', clarity='VS1',
        depth=62.0, table=55, x=6.0, y=6.0, z=3.7
    )
    print(f"Predicted Price: ${result['predicted_price']:.2f}")
    print(f"Predicted Tier: {result['predicted_tier']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # 4. Batch prediction
    price_preds, tier_preds, confidence = ensemble.predict_batch(X_test)
    
    # 5. Evaluate performance
    results = ensemble.evaluate_ensemble(X_test, y_test)
    """)
