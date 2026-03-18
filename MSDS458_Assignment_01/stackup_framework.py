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
Diamond Price Prediction Stackup Framework

This module contains the complete stackup framework for diamond price prediction using:
1. A classifier to predict price tiers (low/medium/high)
2. Tier-specific regression models for accurate price prediction
3. Automatic routing of new diamonds to appropriate models

Usage:
1. Train your tier-specific models and classifier in your main notebook
2. Import this stackup framework
3. Create stackup instance with your trained models
4. Use for prediction on new diamond data
"""

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf

# %%
# Import the utility module
import diamonds_utils as du


# %%
class DiamondPriceStackup:
    """
    Stackup model that combines a classifier and tier-specific regression models.
    
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
        Predict price using stackup approach for single or few samples.
        
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
    
    def evaluate_stackup(self, X_test, y_test, low_threshold=2500, high_threshold=6000):
        """
        Evaluate the stackup model performance.
        
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
        
        print(f"Stackup Performance:")
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
        Predict price for a single diamond using the stackup model.
        
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
def train_tier_regression_models(
    diamonds_train,
    tier_configs,
    numerical_features,
    categorical_features,
    low_threshold=2500,
    high_threshold=6000,
    random_state=42,
):
    """
    Train tier-specific regression models using config dicts.

    Args:
        diamonds_train: Training DataFrame (already outlier-cleaned)
        tier_configs: Dict of model configs per tier, e.g.
            {
                'low':    {'layers': [128, 64, 32], 'lr': 0.001, 'epochs': 50, 'batch_size': 32, 'patience': 3},
                'medium': {'layers': [128, 64, 32], 'lr': 0.001, 'epochs': 50, 'batch_size': 32, 'patience': 3},
                'high':   {'layers': [128, 64, 32], 'lr': 0.001, 'epochs': 50, 'batch_size': 32, 'patience': 3},
            }
        numerical_features: List of numerical feature column names
        categorical_features: List of categorical feature column names
        low_threshold: Price boundary between low and medium tiers
        high_threshold: Price boundary between medium and high tiers
        random_state: Random seed for train/val split

    Returns:
        tuple: (models, preprocessors, histories)
            - models: dict with keys 'low', 'medium', 'high'
            - preprocessors: dict with keys 'low', 'medium', 'high'
            - histories: dict with keys 'low', 'medium', 'high'
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input

    low_data, medium_data, high_data = du.split_data(
        diamonds_train, low_threshold=low_threshold, high_threshold=high_threshold
    )

    models = {}
    preprocessors = {}
    histories = {}

    for tier, data in [('low', low_data), ('medium', medium_data), ('high', high_data)]:
        cfg = tier_configs[tier]
        print(f"\n=== Training {tier.upper()} tier model ===")

        X_processed, y_processed, preprocessor = du.prepare_data_without_split(
            data, numerical_features, categorical_features, target='price'
        )
        preprocessors[tier] = preprocessor

        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=random_state
        )

        layers_list = []
        layers_list.append(Input(shape=(X_train.shape[1],)))
        for units in cfg['layers']:
            layers_list.append(Dense(units, activation='relu'))
        layers_list.append(Dense(1))

        tier_model = Sequential(layers_list)
        tier_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.get('lr', 0.001)),
            loss='mse',
            metrics=['mae', 'mape'],
        )

        history = tier_model.fit(
            X_train, y_train,
            epochs=cfg.get('epochs', 50),
            batch_size=cfg.get('batch_size', 32),
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=cfg.get('patience', 3),
                restore_best_weights=True,
            )],
        )

        models[tier] = tier_model
        histories[tier] = history

        du.evaluate_model(tier_model, X_val, y_val)
        du.plot_training_history(history)
        print(f"=== {tier.upper()} tier complete ===\n")

    print("All tier models trained successfully!")
    return models, preprocessors, histories


# %%
def train_tier_classifier(
    diamonds_train,
    classifier_config,
    numerical_features,
    categorical_features,
    low_threshold=2500,
    high_threshold=6000,
):
    """
    Train a neural network classifier to predict price tiers.

    Args:
        diamonds_train: Training DataFrame (already outlier-cleaned)
        classifier_config: Model config dict, e.g.
            {
                'layers': [256, 128, 64],
                'dropout': 0.2,
                'lr': 0.001,
                'epochs': 50,
                'batch_size': 32,
                'patience': 10,
            }
        numerical_features: List of numerical feature column names
        categorical_features: List of categorical feature column names
        low_threshold: Price boundary between low and medium tiers
        high_threshold: Price boundary between medium and high tiers

    Returns:
        tuple: (classifier_model, preprocessor, history)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input, Dropout

    diamonds_tiers_train = du.create_price_tiers(
        diamonds_train, low_threshold=low_threshold, high_threshold=high_threshold
    )
    diamonds_tiers_train = diamonds_tiers_train.drop(columns=['price'])

    X_train, X_test, y_train, y_test, preprocessor = du.prepare_data(
        diamonds_tiers_train, numerical_features, categorical_features,
        target='price_tier', encode_target=True,
    )

    cfg = classifier_config
    dropout_rate = cfg.get('dropout', 0.0)

    layers_list = []
    layers_list.append(Input(shape=(X_train.shape[1],)))
    for units in cfg['layers']:
        layers_list.append(Dense(units, activation='relu'))
        if dropout_rate > 0:
            layers_list.append(Dropout(dropout_rate))
    layers_list.append(Dense(3, activation='softmax'))

    classifier_model = Sequential(layers_list)
    classifier_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.get('lr', 0.001)),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    classifier_model.summary()

    history = classifier_model.fit(
        X_train, y_train,
        epochs=cfg.get('epochs', 50),
        batch_size=cfg.get('batch_size', 32),
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=cfg.get('patience', 10),
            restore_best_weights=True,
        )],
    )

    loss, accuracy = classifier_model.evaluate(X_test, y_test)
    print(f"Classifier — Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return classifier_model, preprocessor, history


# %%
def create_stackup_from_trained_models(classifier_model, regression_models, preprocessors,
                                       classifier_preprocessor, low_threshold=2500, high_threshold=6000):
    """
    Convenience function to create stackup from trained models.
    
    Args:
        classifier_model: Trained classifier model
        regression_models (dict): Dictionary with regression models for each tier
        preprocessors (dict): Dictionary with preprocessors for each tier
        classifier_preprocessor: Preprocessor used for classifier
        low_threshold (int): Low tier threshold
        high_threshold (int): High tier threshold
        
    Returns:
        DiamondPriceStackup: Configured stackup model
    """
    # Create tier encoder
    tier_encoder = LabelEncoder()
    tier_encoder.fit(['low', 'medium', 'high'])
    
    # Prepare preprocessors dictionary
    stackup_preprocessors = {
        'low': preprocessors['low'],
        'medium': preprocessors['medium'], 
        'high': preprocessors['high'],
        'classifier': classifier_preprocessor
    }
    
    # Create and return stackup
    return DiamondPriceStackup(
        classifier_model=classifier_model,
        regression_models=regression_models,
        preprocessors=stackup_preprocessors,
        tier_encoder=tier_encoder
    )

