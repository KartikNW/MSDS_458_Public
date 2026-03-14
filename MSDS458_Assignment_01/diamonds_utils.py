"""
Diamonds Analysis and Modeling Utility Library

This module contains utility functions for:
- Data loading and exploration
- Data visualization and analysis
- Data preparation and preprocessing
- Model creation and training
- Model evaluation and visualization

Author: MSDS 458 Collaboration
"""

import os
# Configure environment BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU toggle via environment variables (default: ENABLE GPU)
# Preferred: set DIAMONDS_DISABLE_GPU to: 1, true, yes, on (disables GPU)
_disable_gpu = os.getenv('DIAMONDS_DISABLE_GPU', '').strip().lower() in {"1", "true", "yes", "on"}
if (not _disable_gpu) and ('DIAMONDS_USE_GPU' in os.environ):
    _use_gpu_flag = os.getenv('DIAMONDS_USE_GPU', '').strip().lower()
    _disable_gpu = _use_gpu_flag in {"0", "false", "no", "off"}

if _disable_gpu:
    # Hide all CUDA/Metal GPUs from TensorFlow before import
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import seaborn as sns
import sys
from packaging import version
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout

# Enforce device visibility after import as well
try:
    if _disable_gpu:
        tf.config.set_visible_devices([], 'GPU')
except Exception:
    # Safe to ignore if no GPU devices are present or already initialized
    pass

def load_data():
    """Load the diamonds dataset from Seaborn's built-in datasets.

    Returns:
        pandas.DataFrame: The diamonds dataset
    """
    diamonds = sns.load_dataset('diamonds')
    return diamonds

def display_data_info(diamonds):
    """Display basic information about the dataset including its structure and summary statistics.

    Args:
        diamonds (pandas.DataFrame): The diamonds dataset
    """
    print("\nDataset Info:")
    print(diamonds.info())
    print("\nFirst 5 rows of the dataset:")
    print(diamonds.head())

    print("\nBasic Statistics:")
    print(diamonds.describe())

def analyze_categorical_variables(diamonds):
    """Analyze and visualize the distribution of categorical variables (cut, color, clarity) in the dataset.

    Args:
        diamonds (pandas.DataFrame): The diamonds dataset
    """
    categorical_cols = ['cut', 'color', 'clarity']

    for col in categorical_cols:
        print(f"\n{col.upper()} Distribution:")
        print(diamonds[col].value_counts())

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=diamonds, x=col)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def analyze_price_distribution(diamonds):
    """Analyze and visualize the distribution of diamond prices in the dataset.

    Args:
        diamonds (pandas.DataFrame): The diamonds dataset
    """
    plt.figure(figsize=(12, 6))

    # Create a histogram with KDE
    sns.histplot(data=diamonds, x='price', bins=50, kde=True)
    plt.title('Distribution of Diamond Prices')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Print price statistics
    print("\nPrice Statistics:")
    print(diamonds['price'].describe())

def analyze_carat_price_relationship(diamonds):
    """Analyze and visualize the relationship between carat and price of diamonds.

    Args:
        diamonds (pandas.DataFrame): The diamonds dataset
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=diamonds, x='carat', y='price', alpha=0.5)
    plt.title('Carat vs Price')
    plt.tight_layout()
    plt.show()

    # Calculate correlation
    correlation = diamonds['carat'].corr(diamonds['price'])
    print(f"\nCorrelation between carat and price: {correlation:.3f}")

def analyze_cut_impact(diamonds):
    """Analyze how the cut quality affects the price of diamonds.

    Args:
        diamonds (pandas.DataFrame): The diamonds dataset
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=diamonds, x='cut', y='price')
    plt.title('Price Distribution by Cut Quality')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print average price by cut
    print("\nAverage Price by Cut:")
    print(diamonds.groupby('cut')['price'].mean().sort_values(ascending=False))

def analyze_correlations(diamonds):
    """Analyze correlations between numerical variables in the dataset.

    Args:
        diamonds (pandas.DataFrame): The diamonds dataset
    """
    numerical_cols = diamonds.select_dtypes(include=[np.number]).columns
    correlation_matrix = diamonds[numerical_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.show()

def create_model(input_dim):
    """Create and compile the neural network model for price prediction.

    Args:
        input_dim (int): Number of input features

    Returns:
        tensorflow.keras.Model: Compiled neural network model
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae','mape']
    )

    return model

def create_classifier_model(input_dim, num_classes):
    """Create and compile a DNN classifier for price tier prediction.

    Args:
        input_dim (int): Number of input features
        num_classes (int): Number of output classes

    Returns:
        tensorflow.keras.Model: Compiled classifier model
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def prepare_data(diamonds, numerical_features, categorical_features, target='price', encode_target=False):
    """Prepare the data for model training by handling categorical variables and scaling numerical features.

    Args:
        diamonds (pandas.DataFrame): The diamonds dataset
        numerical_features (list): List of numerical feature column names
        categorical_features (list): List of categorical feature column names
        target (str): Target variable column name, defaults to 'price'
        encode_target (bool): Whether to encode the target variable, defaults to False

    Returns:
        tuple: (X_train_processed, X_test_processed, y_train_processed, y_test_processed, preprocessor)
    """
    # Separate features and target
    X = diamonds.drop(target, axis=1)
    y = diamonds[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # if encode_target is True, encode the target variable
    if encode_target:
        y_train_processed = y_train.astype(str)
        y_test_processed = y_test.astype(str)
        encoder = LabelEncoder()
        y_train_processed = encoder.fit_transform(y_train_processed)
        y_test_processed = encoder.transform(y_test_processed)
    else:
        y_train_processed = y_train
        y_test_processed = y_test

    return X_train_processed, X_test_processed, y_train_processed, y_test_processed, preprocessor


def prepare_data_without_split(diamonds, numerical_features, categorical_features, target='price', encode_target=False, test_size=0.2, random_state=42):
    """Prepare the data for model training by handling categorical variables and scaling numerical features.
    This version performs the train/test split internally.

    Args:
        diamonds (pandas.DataFrame): The diamonds dataset
        numerical_features (list): List of numerical feature column names
        categorical_features (list): List of categorical feature column names
        target (str): Target variable column name, defaults to 'price'
        encode_target (bool): Whether to encode the target variable, defaults to False
        test_size (float): Proportion of data for testing, defaults to 0.2
        random_state (int): Random seed for reproducibility, defaults to 42

    Returns:
        tuple: (X_processed, y_processed, preprocessor)
    """
    # Separate features and target
    X = diamonds.drop(target, axis=1)
    y = diamonds[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    # if encode_target is True, encode the target variable
    if encode_target:
        y_processed = y.astype(str)
        encoder = LabelEncoder()
        y_processed = encoder.fit_transform(y_processed)
    else:
        y_processed = y

    return X_processed, y_processed, preprocessor

def train_model(X_train, y_train, patience=10, epochs=50):
    """Train the neural network model on the prepared data.

    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        patience (int): Early stopping patience, defaults to 10
        epochs (int): Maximum number of training epochs, defaults to 50

    Returns:
        tuple: (trained_model, training_history)
    """
    model = create_model(X_train.shape[1])

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )]
    )

    return model, history

def plot_training_history(history):
    """Plot the training history of the model, showing loss and MAE over epochs.

    Args:
        history (tensorflow.keras.callbacks.History): Training history object
    """
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance using various metrics and visualizations.

    Args:
        model: Trained model (TensorFlow or scikit-learn)
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets

    Returns:
        numpy.ndarray: Predicted values
    """
    # Evaluate the model
    if hasattr(model, 'evaluate'):  # TensorFlow model
        evaluation_results = model.evaluate(X_test, y_test)
        if len(evaluation_results) == 3:  # loss, mae, mape
            test_loss, test_mae, test_mape = evaluation_results
            print(f"\nTest MAE: ${test_mae:.2f}")
            print(f"Test MAPE: {test_mape:.2f}%")
        elif len(evaluation_results) == 2:  # loss, mae (fallback)
            test_loss, test_mae = evaluation_results
            print(f"\nTest MAE: ${test_mae:.2f}")
        y_pred = model.predict(X_test)
    else:  # Scikit-learn model
        y_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        print(f"\nTest MAE: ${test_mae:.2f}")

    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared Score: {r2:.4f}")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Diamond Prices')
    plt.tight_layout()
    plt.show()

    return y_pred.ravel()

def residual_plot(y_test, y_pred):
    """Create a residual plot to analyze prediction errors.

    Args:
        y_test (numpy.ndarray): Actual test values
        y_pred (numpy.ndarray): Predicted values
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.show()

def plot_percentage_error_by_price(y_test, y_pred, price_bins=10):
    """Plot how percentage error varies based on diamond price.

    Args:
        y_test (numpy.ndarray): Actual test values
        y_pred (numpy.ndarray): Predicted values
        price_bins (int): Number of price bins to group diamonds, defaults to 10
    """
    # Calculate percentage errors
    percentage_errors = np.abs((y_test - y_pred) / y_test) * 100

    # Create price bins
    price_ranges = pd.cut(y_test, bins=price_bins, labels=False)
    price_bin_centers = pd.cut(y_test, bins=price_bins, retbins=True)[1]
    bin_centers = (price_bin_centers[:-1] + price_bin_centers[1:]) / 2

    # Calculate mean percentage error for each bin
    bin_errors = []
    bin_centers_actual = []

    for i in range(price_bins):
        mask = price_ranges == i
        if np.any(mask):
            mean_error = np.mean(percentage_errors[mask])
            bin_errors.append(mean_error)
            bin_centers_actual.append(bin_centers[i])

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot 1: Percentage error vs price bins
    plt.subplot(1, 2, 1)
    plt.plot(bin_centers_actual, bin_errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Price (USD)')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.title('Percentage Error by Price Range')
    plt.grid(True, alpha=0.3)

    # Plot 2: Scatter plot of individual percentage errors vs price
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, percentage_errors, alpha=0.5, s=20)
    plt.xlabel('Actual Price (USD)')
    plt.ylabel('Absolute Percentage Error (%)')
    plt.title('Individual Percentage Errors vs Price')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nPercentage Error Summary:")
    print(f"Overall MAPE: {np.mean(percentage_errors):.2f}%")
    print(f"MAPE for diamonds < $2,500: {np.mean(percentage_errors[y_test < 2500]):.2f}%")
    print(f"MAPE for diamonds $2,500-$6,000: {np.mean(percentage_errors[(y_test >= 2500) & (y_test < 6000)]):.2f}%")
    print(f"MAPE for diamonds > $6,000: {np.mean(percentage_errors[y_test >= 6000]):.2f}%")

    return percentage_errors, bin_centers_actual, bin_errors

def split_data(data, low_threshold=5000, high_threshold=10000):
    """Split the data into low, medium and high price tiers.

    Args:
        data (pandas.DataFrame): The diamonds dataset
        low_threshold (int): Threshold for low price tier, defaults to 5000
        high_threshold (int): Threshold for high price tier, defaults to 10000

    Returns:
        tuple: (low_data, medium_data, high_data)
    """
    low_data = data[data['price'] < low_threshold]
    medium_data = data[(data['price'] >= low_threshold) & (data['price'] <= high_threshold)]
    high_data = data[data['price'] > high_threshold]
    return low_data, medium_data, high_data


def create_price_tiers(diamonds, low_threshold=5000, high_threshold=10000):
    """Create price tier categories for classification tasks.

    Args:
        diamonds (pandas.DataFrame): The diamonds dataset
        low_threshold (int): Threshold for low price tier, defaults to 5000
        high_threshold (int): Threshold for high price tier, defaults to 10000

    Returns:
        pandas.DataFrame: Dataset with price_tier column added
    """
    diamonds_copy = diamonds.copy()
    diamonds_copy['price_tier'] = np.where(
        diamonds_copy['price'] < low_threshold, 'low',
        np.where(
            (diamonds_copy['price'] >= low_threshold) & (diamonds_copy['price'] <= high_threshold),
            'medium', 'high'
        )
    )
    return diamonds_copy

def remove_specific_outliers(diamonds, verbose=True):
    """Remove specific outliers based on domain knowledge and extreme values.

    This function removes diamonds with:
    - Zero dimensions (x=0, y=0, z=0)
    - Extreme table values (table=95)
    - Extreme y values (31.8, 58.9)
    - Extreme z values (31.8)

    Args:
        diamonds (pandas.DataFrame): The diamonds dataset
        verbose (bool): Whether to print removal statistics, defaults to True

    Returns:
        pandas.DataFrame: Dataset with specific outliers removed
    """
    if verbose:
        print(f"Before removing specific outliers: {diamonds.shape}")

    # Store original shape for reporting
    original_shape = diamonds.shape

    # Remove rows with zero dimensions
    diamonds_clean = diamonds[diamonds['x'] != 0]
    diamonds_clean = diamonds_clean[diamonds_clean['y'] != 0]
    diamonds_clean = diamonds_clean[diamonds_clean['z'] != 0]

    # Remove rows with extreme table values
    diamonds_clean = diamonds_clean[diamonds_clean['table'] != 95]

    # Remove rows with extreme y values
    diamonds_clean = diamonds_clean[diamonds_clean['y'] != 31.8]
    diamonds_clean = diamonds_clean[diamonds_clean['y'] != 58.9]

    # Remove rows with extreme z values
    diamonds_clean = diamonds_clean[diamonds_clean['z'] != 31.8]

    if verbose:
        print(f"After removing specific outliers: {diamonds_clean.shape}")
        removed_count = original_shape[0] - diamonds_clean.shape[0]
        print(f"Removed {removed_count} rows ({removed_count/original_shape[0]*100:.2f}% of data)")

        # Show what was removed
        print("\nRemoval Summary:")
        print(f"- Zero dimensions (x=0, y=0, z=0): {len(diamonds[diamonds['x'] == 0]) + len(diamonds[diamonds['y'] == 0]) + len(diamonds[diamonds['z'] == 0])} rows")
        print(f"- Table = 95: {len(diamonds[diamonds['table'] == 95])} rows")
        print(f"- Y = 31.8: {len(diamonds[diamonds['y'] == 31.8])} rows")
        print(f"- Y = 58.9: {len(diamonds[diamonds['y'] == 58.9])} rows")
        print(f"- Z = 31.8: {len(diamonds[diamonds['z'] == 31.8])} rows")

    return diamonds_clean
