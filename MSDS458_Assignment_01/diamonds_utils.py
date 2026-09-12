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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def get_device():
    """Select the best available torch device (CUDA → MPS → CPU).

    Set DIAMONDS_DISABLE_GPU to 1/true/yes/on, or DIAMONDS_USE_GPU to
    0/false/no/off, to force CPU.
    """
    if os.getenv('DIAMONDS_DISABLE_GPU', '').strip().lower() in {"1", "true", "yes", "on"}:
        return torch.device('cpu')
    if os.getenv('DIAMONDS_USE_GPU', '').strip().lower() in {"0", "false", "no", "off"}:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()
                        else 'cpu')

_default_device = get_device()

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
    """Create the neural network model for price prediction.

    Args:
        input_dim (int): Number of input features

    Returns:
        torch.nn.Module: Neural network model (not yet moved to a device)
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1)
    )
    return model

def create_classifier_model(input_dim, num_classes):
    """Create a DNN classifier for price tier prediction.

    The output layer emits raw logits (no softmax) because
    nn.CrossEntropyLoss applies softmax internally.

    Args:
        input_dim (int): Number of input features
        num_classes (int): Number of output classes

    Returns:
        torch.nn.Module: Classifier model (not yet moved to a device)
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, num_classes)
    )
    return model


class HistoryObject:
    """Per-epoch metrics exposed as a `.history` dict.

    This is the shape `plot_training_history()` expects, so the training
    functions below return one of these rather than a bare dict.
    """
    def __init__(self, history):
        self.history = history


def predict(model, X, device=None):
    """Run a forward pass on numpy features and return a numpy array.

    Works for both torch nn.Module models and sklearn models.

    Args:
        model: Trained model (torch nn.Module or sklearn)
        X (numpy.ndarray): Input features
        device: torch device (defaults to the model's own device)

    Returns:
        numpy.ndarray: Model outputs
    """
    if isinstance(model, nn.Module):
        device = device or next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(device)
            return model(X_tensor).cpu().numpy()
    return model.predict(X)


def fit_regression(model, X_train, y_train, X_val, y_val,
                   lr=0.001, epochs=50, batch_size=32, patience=10,
                   device=None, verbose=True):
    """Train a regression model with the standard PyTorch training loop.

    Uses Adam + MSE loss with early stopping and best-weight restore, so the
    returned model is the best one seen, not the one from the final epoch.

    Args:
        model (torch.nn.Module): Model to train (modified in place)
        X_train, y_train: Training features and targets (numpy)
        X_val, y_val: Validation features and targets (numpy)
        lr (float): Learning rate for Adam
        epochs (int): Maximum number of epochs
        batch_size (int): Mini-batch size
        patience (int): Early stopping patience (epochs without val_loss improvement)
        device: torch device, defaults to best available
        verbose (bool): Print per-epoch metrics

    Returns:
        HistoryObject: per-epoch loss/mae/mape (+ val_ versions)
    """
    device = device or _default_device
    model = model.to(device)

    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(np.asarray(X_train, dtype=np.float32)),
                      torch.as_tensor(y_train)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(np.asarray(X_val, dtype=np.float32)),
                      torch.as_tensor(y_val)),
        batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'loss': [], 'mae': [], 'mape': [],
               'val_loss': [], 'val_mae': [], 'val_mape': []}
    best_val_loss, patience_counter, best_state = float('inf'), 0, None

    def _epoch_metrics(loader, training):
        total_loss = total_mae = total_mape = 0.0
        n_samples = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if training:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            # Weight by batch size so a short final batch does not skew the epoch
            # average. The epoch metric is a sample-weighted mean, not a mean of
            # per-batch means, which differ whenever the last batch is short.
            batch_size_actual = y_batch.size(0)
            total_loss += loss.item() * batch_size_actual
            total_mae += torch.sum(torch.abs(outputs - y_batch)).item()
            total_mape += torch.sum(torch.abs((y_batch - outputs) / y_batch)).item() * 100
            n_samples += batch_size_actual
        return total_loss / n_samples, total_mae / n_samples, total_mape / n_samples

    for epoch in range(epochs):
        model.train()
        train_loss, train_mae, train_mape = _epoch_metrics(train_loader, training=True)

        model.eval()
        with torch.no_grad():
            val_loss, val_mae, val_mape = _epoch_metrics(val_loader, training=False)

        history['loss'].append(train_loss); history['mae'].append(train_mae)
        history['mape'].append(train_mape)
        history['val_loss'].append(val_loss); history['val_mae'].append(val_mae)
        history['val_mape'].append(val_mape)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - mae: {train_mae:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return HistoryObject(history)


def fit_classifier(model, X_train, y_train, X_val, y_val,
                   lr=0.001, epochs=50, batch_size=32, patience=10,
                   device=None, verbose=True):
    """Train a classifier with the standard PyTorch training loop.

    Uses Adam + CrossEntropyLoss (expects integer class labels and a model
    that outputs logits) with early stopping and best-weight restore.

    Args:
        model (torch.nn.Module): Model to train (modified in place)
        X_train, y_train: Training features and integer labels (numpy)
        X_val, y_val: Validation features and integer labels (numpy)
        lr (float): Learning rate for Adam
        epochs (int): Maximum number of epochs
        batch_size (int): Mini-batch size
        patience (int): Early stopping patience (epochs without val_loss improvement)
        device: torch device, defaults to best available
        verbose (bool): Print per-epoch metrics

    Returns:
        HistoryObject: per-epoch loss/accuracy (+ val_ versions)
    """
    device = device or _default_device
    model = model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(np.asarray(X_train, dtype=np.float32)),
                      torch.as_tensor(np.asarray(y_train), dtype=torch.long)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(np.asarray(X_val, dtype=np.float32)),
                      torch.as_tensor(np.asarray(y_val), dtype=torch.long)),
        batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss, patience_counter, best_state = float('inf'), 0, None

    def _epoch_metrics(loader, training):
        total_loss = 0.0
        correct = total = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if training:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
            else:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)
        return total_loss / total, correct / total

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = _epoch_metrics(train_loader, training=True)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = _epoch_metrics(val_loader, training=False)

        history['loss'].append(train_loss); history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_accuracy'].append(val_acc)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return HistoryObject(history)

def _select_feature_columns(diamonds, numerical_features, categorical_features, target):
    """Validate the feature lists and return an explicit feature-only DataFrame (allowlist).

    Only the columns named in `numerical_features` / `categorical_features` become model
    inputs. This prevents any other column in the dataframe (for example a transformed
    copy of the target) from silently being used as a feature.

    Rules:
      - Hard error if `target` appears in either feature list (this would be data leakage).
      - Warn (and skip) if a named feature column is missing from the dataframe.

    Returns:
        tuple: (X, numerical_present, categorical_present)
    """
    leaked = [c for c in (list(numerical_features) + list(categorical_features)) if c == target]
    if leaked:
        raise ValueError(
            f"prepare_data(): target '{target}' is listed as a feature "
            f"{numerical_features=} {categorical_features=}. Remove it from the feature "
            f"lists \u2014 using the target as an input is data leakage."
        )

    numerical_present = [c for c in numerical_features if c in diamonds.columns]
    categorical_present = [c for c in categorical_features if c in diamonds.columns]
    missing = [c for c in (list(numerical_features) + list(categorical_features))
               if c not in diamonds.columns]
    if missing:
        print(f"\u26a0\ufe0f  prepare_data(): named feature column(s) not in dataframe, skipping: {missing}")

    # Allowlist: only the named feature columns become inputs; any other column in the
    # dataframe is never passed through.
    X = diamonds[numerical_present + categorical_present].copy()
    return X, numerical_present, categorical_present


def _print_prepare_summary(target, numerical_present, categorical_present, n_output_cols):
    """Print what actually went into the model, so students can verify the feature set."""
    print(f"prepare_data(): target = '{target}'")
    print(f"  numerical   ({len(numerical_present)}): {numerical_present}")
    print(f"  categorical ({len(categorical_present)}): {categorical_present}")
    print(f"  output feature columns: {n_output_cols}")


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
    # Separate features (allowlist) and target
    X, numerical_present, categorical_present = _select_feature_columns(
        diamonds, numerical_features, categorical_features, target
    )
    y = diamonds[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_present),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_present)
        ],
        remainder='drop'
    )

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    _print_prepare_summary(target, numerical_present, categorical_present, X_train_processed.shape[1])

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
    # Separate features (allowlist) and target
    X, numerical_present, categorical_present = _select_feature_columns(
        diamonds, numerical_features, categorical_features, target
    )
    y = diamonds[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_present),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_present)
        ],
        remainder='drop'
    )

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    _print_prepare_summary(target, numerical_present, categorical_present, X_processed.shape[1])

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

    Holds out 20% of the training data for validation.

    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        patience (int): Early stopping patience, defaults to 10
        epochs (int): Maximum number of training epochs, defaults to 50

    Returns:
        tuple: (trained_model, training_history)
    """
    model = create_model(X_train.shape[1])

    y_np = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_np, test_size=0.2, random_state=42
    )

    history = fit_regression(
        model, X_tr, y_tr, X_val, y_val,
        epochs=epochs, batch_size=32, patience=patience
    )

    return model, history

def plot_training_history(history):
    """Plot the training history of the model, showing loss and MAE over epochs.

    Args:
        history: HistoryObject (or plain dict) of per-epoch metrics
    """
    h = history.history if hasattr(history, 'history') else history

    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(h['loss'], label='Training Loss')
    plt.plot(h['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(h['mae'], label='Training MAE')
    plt.plot(h['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance using various metrics and visualizations.

    Args:
        model: Trained model (PyTorch or scikit-learn)
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets

    Returns:
        numpy.ndarray: Predicted values
    """
    # Evaluate the model
    y_pred = predict(model, X_test).ravel()
    y_true = np.asarray(y_test)
    test_mae = mean_absolute_error(y_true, y_pred)
    print(f"\nTest MAE: ${test_mae:.2f}")
    if isinstance(model, nn.Module):
        test_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"Test MAPE: {test_mape:.2f}%")

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
