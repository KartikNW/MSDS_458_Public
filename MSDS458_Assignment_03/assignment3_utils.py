"""
Assignment 3 Utilities
======================

Shared utilities for MiniTransformer (Part A) and CodeT5 (Part B) notebooks.
This module provides common functions for data loading, processing, and evaluation
to eliminate code duplication across notebooks.

Usage:
    from assignment3_utils import *

    # Or import specific functions:
    from assignment3_utils import prepare_training_data, evaluate_model_with_bleu

Author: Assignment 3 - Encoder-Decoder Transformers
Date: December 2025
"""

# ============================================
# IMPORTS
# ============================================

# Dataset loading
from datasets import load_dataset

# Numerical operations
import numpy as np

# Syntax validation
import ast

# Evaluation metrics
import evaluate

# Progress bars
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt


# ============================================
# SECTION 1: CONFIGURATION
# ============================================
# Dataset selection and configuration settings

# Dataset to use: "MBPP" or "CodeContests"
DATASET = "MBPP"

# Dataset-specific configuration
DATASET_CONFIG = {
    "MBPP": {
        "max_train": 374,
        "max_val": 90,
        "max_test": 50,
        "name": "mbpp",
        "config": "full",  # Use 'full' configuration (374/90/500 examples)
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test"
    },
    "CodeContests": {
        "max_train": 2000,
        "max_val": 200,
        "max_test": 100,
        "name": "deepmind/code_contests",
        "config": None,  # CodeContests doesn't have multiple configs
        "train_split": "train",
        "val_split": "valid",
        "test_split": "test"
    }
}


def get_dataset_config(dataset_name=None):
    """
    Get configuration for specified dataset.

    Args:
        dataset_name (str, optional): Dataset name ("MBPP" or "CodeContests").
                                      If None, uses global DATASET variable.

    Returns:
        dict: Configuration dictionary with keys:
              - max_train: Maximum training examples
              - max_val: Maximum validation examples
              - max_test: Maximum test examples
              - name: HuggingFace dataset name
              - train_split: Name of training split
              - val_split: Name of validation split
              - test_split: Name of test split

    Example:
        >>> config = get_dataset_config("MBPP")
        >>> print(config['max_train'])
        374
    """
    if dataset_name is None:
        dataset_name = DATASET

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be 'MBPP' or 'CodeContests'")

    return DATASET_CONFIG[dataset_name]


# ============================================
# SECTION 2: DATA LOADING
# ============================================
# Functions to load and extract examples from MBPP and CodeContests datasets


def load_dataset_splits(dataset_name=None):
    """
    Load dataset from HuggingFace and return train/val/test splits.

    Args:
        dataset_name (str, optional): Dataset name ("MBPP" or "CodeContests").
                                      If None, uses global DATASET variable.

    Returns:
        dict: Dictionary with keys 'train', 'val', 'test' containing dataset splits.

    Example:
        >>> dataset = load_dataset_splits("MBPP")
        >>> print(f"Train size: {len(dataset['train'])}")
        Train size: 374
    """
    config = get_dataset_config(dataset_name)

    # Load raw dataset from HuggingFace
    # Use config parameter if specified (e.g., MBPP has 'full' and 'sanitized' versions)
    if config.get('config'):
        raw_dataset = load_dataset(config['name'], config['config'])
    else:
        raw_dataset = load_dataset(config['name'])

    # Create standardized split dictionary
    dataset = {
        'train': raw_dataset[config['train_split']],
        'val': raw_dataset[config['val_split']],
        'test': raw_dataset[config['test_split']]
    }

    return dataset


def extract_mbpp_example(example):
    """
    Extract prompt and code from MBPP dataset example.

    Args:
        example (dict): MBPP example with keys 'text' and 'code'.

    Returns:
        tuple: (prompt, code) or (None, None) if extraction fails.

    Example:
        >>> example = {'text': 'Write a function...', 'code': 'def func(): pass'}
        >>> prompt, code = extract_mbpp_example(example)
    """
    try:
        prompt = example['text']
        code = example['code']
        return prompt, code
    except (KeyError, TypeError):
        return None, None


def extract_codecontests_example(example):
    """
    Extract description and Python code from CodeContests dataset example.

    Args:
        example (dict): CodeContests example with keys 'description' and 'solutions'.

    Returns:
        tuple: (description, code) or (None, None) if no Python solution found.

    Note:
        - Truncates description to 500 characters
        - Only returns Python solutions (language code 3)
        - Returns first valid Python solution

    Example:
        >>> example = {
        ...     'description': 'Problem description...',
        ...     'solutions': {'language': [3, 1], 'solution': ['python_code', 'cpp_code']}
        ... }
        >>> prompt, code = extract_codecontests_example(example)
    """
    try:
        # Extract and truncate description
        description = example['description']
        if len(description) > 500:
            description = description[:500] + "..."

        # Extract solutions
        solutions = example['solutions']
        languages = solutions['language']
        codes = solutions['solution']

        # Find first Python solution (language code 3)
        for lang, code in zip(languages, codes):
            if lang == 3:  # Python
                return description, code

        # No Python solution found
        return None, None
    except (KeyError, TypeError, IndexError):
        return None, None


def extract_example(example, dataset_name=None):
    """
    Extract prompt and code from example using appropriate extractor.

    Args:
        example (dict): Dataset example.
        dataset_name (str, optional): Dataset name ("MBPP" or "CodeContests").
                                      If None, uses global DATASET variable.

    Returns:
        tuple: (prompt, code) or (None, None) if extraction fails.

    Example:
        >>> # For MBPP
        >>> prompt, code = extract_example(mbpp_example, "MBPP")

        >>> # For CodeContests
        >>> prompt, code = extract_example(cc_example, "CodeContests")
    """
    if dataset_name is None:
        dataset_name = DATASET

    if dataset_name == "MBPP":
        return extract_mbpp_example(example)
    elif dataset_name == "CodeContests":
        return extract_codecontests_example(example)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================
# SECTION 3: DATA PROCESSING
# ============================================
# Functions to process raw datasets into training-ready format


def process_dataset_split(split_data, dataset_name, max_examples):
    """
    Process a dataset split by extracting prompts and codes.

    Args:
        split_data: Dataset split (e.g., dataset['train'])
        dataset_name (str): Dataset name ("MBPP" or "CodeContests")
        max_examples (int): Maximum number of examples to extract

    Returns:
        dict: Dictionary with keys 'prompts' and 'code' containing lists

    Example:
        >>> dataset = load_dataset_splits("MBPP")
        >>> train_data = process_dataset_split(dataset['train'], "MBPP", 374)
        >>> print(f"Extracted {len(train_data['prompts'])} examples")
        Extracted 374 examples
    """
    prompts = []
    codes = []
    count = 0

    for example in split_data:
        if count >= max_examples:
            break

        prompt, code = extract_example(example, dataset_name)

        if prompt is not None and code is not None:
            prompts.append(prompt)
            codes.append(code)
            count += 1

    return {
        'prompts': prompts,
        'code': codes
    }


def prepare_training_data(dataset_name=None):
    """
    Load and process entire dataset (train/val/test splits).

    This is a convenience function that:
    1. Gets dataset configuration
    2. Loads dataset from HuggingFace
    3. Processes all three splits (train, val, test)
    4. Returns data ready for training

    Args:
        dataset_name (str, optional): Dataset name ("MBPP" or "CodeContests").
                                      If None, uses global DATASET variable.

    Returns:
        dict: Dictionary with keys 'train', 'val', 'test', each containing:
              - 'prompts': List of problem descriptions/prompts
              - 'code': List of corresponding code solutions

    Example:
        >>> # Load MBPP dataset
        >>> data = prepare_training_data("MBPP")
        >>> print(f"Train: {len(data['train']['prompts'])} examples")
        >>> print(f"Val: {len(data['val']['prompts'])} examples")
        >>> print(f"Test: {len(data['test']['prompts'])} examples")
        Train: 374 examples
        Val: 90 examples
        Test: 50 examples

        >>> # Access data
        >>> first_prompt = data['train']['prompts'][0]
        >>> first_code = data['train']['code'][0]
    """
    # Get configuration
    config = get_dataset_config(dataset_name)
    if dataset_name is None:
        dataset_name = DATASET

    # Load dataset
    dataset = load_dataset_splits(dataset_name)

    # Process each split
    print(f"Processing {dataset_name} dataset...")

    train_data = process_dataset_split(
        dataset['train'],
        dataset_name,
        config['max_train']
    )
    print(f"  Train: {len(train_data['prompts'])} examples")

    val_data = process_dataset_split(
        dataset['val'],
        dataset_name,
        config['max_val']
    )
    print(f"  Val: {len(val_data['prompts'])} examples")

    test_data = process_dataset_split(
        dataset['test'],
        dataset_name,
        config['max_test']
    )
    print(f"  Test: {len(test_data['prompts'])} examples")

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }


# ============================================
# SECTION 4: BLEU EVALUATION
# ============================================
# Functions for evaluating models using BLEU score and syntax validity


def add_python_hint(prompt):
    """
    Add explicit Python language hint to prompts.

    Problem: Models like CodeT5 are trained on multi-language code and may generate
    JavaScript, Java, etc. when given ambiguous prompts like "Write a function to..."

    Solution: Add "Python:\n" prefix to guide the model to generate Python code.

    Args:
        prompt (str): Original prompt (e.g., "Write a function to find...")

    Returns:
        str: Modified prompt with Python hint (e.g., "Python:\nWrite a function to...")

    Example:
        >>> prompt = "Write a function to reverse a string"
        >>> add_python_hint(prompt)
        'Python:\\nWrite a function to reverse a string'

        >>> prompt = "Write a python function to reverse a string"  # already has 'python'
        >>> add_python_hint(prompt)
        'Write a python function to reverse a string'  # unchanged
    """
    # Check if prompt already mentions Python explicitly in first 20 characters
    if 'python' in prompt.lower()[:20]:
        return prompt

    # Add Python prefix to guide model
    return f"Python:\n{prompt}"


def compute_bleu_for_code(prompt, reference_code, generate_fn, bleu_metric, model_label="model"):
    """
    Generate code for a single prompt and compute BLEU score + syntax validity.

    This function is generic and works with any code generation model via the
    generate_fn callback.

    Args:
        prompt (str): Natural language problem description
        reference_code (str): Expected/ground-truth Python code
        generate_fn (callable): Function that takes (prompt) and returns generated code
        bleu_metric: Loaded BLEU metric from evaluate library (e.g., evaluate.load("bleu"))
        model_label (str): Label for tracking which model generated this (for debugging)

    Returns:
        tuple: (bleu_score, is_valid_syntax, generated_code)
            - bleu_score (float): BLEU score on 0-100 scale
            - is_valid_syntax (bool): True if generated code is syntactically valid Python
            - generated_code (str): The generated code string

    Example:
        >>> def my_generator(prompt):
        ...     return "def solution():\\n    pass"
        >>>
        >>> bleu_metric = evaluate.load("bleu")
        >>> score, valid, code = compute_bleu_for_code(
        ...     "Write a function",
        ...     "def solution():\\n    pass",
        ...     my_generator,
        ...     bleu_metric,
        ...     "my-model"
        ... )
        >>> print(f"BLEU: {score:.2f}, Valid: {valid}")
        BLEU: 100.00, Valid: True
    """
    # Generate code using provided generation function
    generated_code = generate_fn(prompt)

    # Normalize: strip whitespace
    generated_clean = generated_code.strip()
    reference_clean = reference_code.strip()

    # Handle empty generation (BLEU metric can't handle empty strings)
    if not generated_clean or len(generated_clean) == 0:
        # Empty generation gets BLEU score of 0 and invalid syntax
        bleu_score = 0.0
        is_valid_syntax = False
    else:
        # Compute BLEU score
        # BLEU expects: predictions as list of strings, references as list of list of strings
        try:
            bleu_result = bleu_metric.compute(
                predictions=[generated_clean],
                references=[[reference_clean]]
            )
            # Extract BLEU score and multiply by 100 for readability (0-100 scale)
            bleu_score = bleu_result['bleu'] * 100
        except (ZeroDivisionError, ValueError):
            # If BLEU computation fails (e.g., no matching n-grams), assign score of 0
            bleu_score = 0.0

        # Validate Python syntax using AST parser (only if non-empty)
        is_valid_syntax = False
        try:
            ast.parse(generated_clean)
            is_valid_syntax = True
        except (SyntaxError, ValueError):
            is_valid_syntax = False

    return bleu_score, is_valid_syntax, generated_code


def evaluate_model_with_bleu(test_prompts, test_codes, generate_fn, bleu_metric, model_label="model", generated_codes=None):
    """
    Evaluate a model on entire test set using BLEU metric.

    This function is generic and can be reused for any model (zero-shot, fine-tuned,
    from-scratch, etc.) by providing a different generate_fn.

    Args:
        test_prompts (list): List of problem descriptions (strings)
        test_codes (list): List of reference code solutions (strings)
        generate_fn (callable): Function that takes (prompt) and returns generated code.
                                Ignored if generated_codes is provided.
        bleu_metric: Loaded BLEU metric from evaluate library
        model_label (str): String label for this evaluation (e.g., "zero-shot", "fine-tuned")
        generated_codes (list, optional): Pre-generated codes (e.g., from batch generation).
                                          If provided, skips calling generate_fn and scores
                                          these directly. Use this for faster evaluation.

    Returns:
        tuple: (bleu_scores, syntax_valid, generated_codes, stats_dict)
            - bleu_scores (list): BLEU score for each example
            - syntax_valid (list): Boolean for each example indicating syntax validity
            - generated_codes (list): Generated code for each example
            - stats_dict (dict): Statistics with keys:
                - 'mean', 'median', 'std', 'min', 'max': BLEU statistics
                - 'syntax_valid_count': Number of syntactically valid examples
                - 'syntax_valid_pct': Percentage of syntactically valid examples

    Example:
        >>> prompts = ["Write function A", "Write function B"]
        >>> codes = ["def a(): pass", "def b(): pass"]
        >>>
        >>> def my_gen(p):
        ...     return "def solution(): pass"
        >>>
        >>> bleu_metric = evaluate.load("bleu")
        >>> scores, valid, gen, stats = evaluate_model_with_bleu(
        ...     prompts, codes, my_gen, bleu_metric, "test-model"
        ... )
        >>> print(f"Mean BLEU: {stats['mean']:.2f}")
        >>>
        >>> # Or with pre-generated codes (skips generation, much faster):
        >>> pre_gen = ["def a(): pass", "def b(): pass"]
        >>> scores, valid, gen, stats = evaluate_model_with_bleu(
        ...     prompts, codes, None, bleu_metric, "test-model",
        ...     generated_codes=pre_gen
        ... )
    """
    using_pregenerated = generated_codes is not None
    if using_pregenerated:
        print(f"🔬 Scoring {model_label.upper()} on {len(test_prompts)} test examples (pre-generated)...")
    else:
        print(f"🔬 Evaluating {model_label.upper()} model on {len(test_prompts)} test examples...")
        print(f"This will take ~2-3 minutes (generating code for each example)")
    print("=" * 80)

    # Storage for results
    bleu_scores = []
    syntax_valid = []
    scored_codes = []

    # Evaluate each test example
    for i in tqdm(range(len(test_prompts)), desc=f"{model_label} BLEU Evaluation"):
        prompt = test_prompts[i]
        reference = test_codes[i]

        if using_pregenerated:
            # Use pre-generated code — just compute BLEU + syntax validity
            generated = generated_codes[i]
            generated_clean = generated.strip()
            reference_clean = reference.strip()

            if not generated_clean:
                bleu_score = 0.0
                is_valid = False
            else:
                try:
                    bleu_result = bleu_metric.compute(
                        predictions=[generated_clean],
                        references=[[reference_clean]]
                    )
                    bleu_score = bleu_result['bleu'] * 100
                except (ZeroDivisionError, ValueError):
                    bleu_score = 0.0

                try:
                    ast.parse(generated_clean)
                    is_valid = True
                except (SyntaxError, ValueError):
                    is_valid = False
        else:
            # Generate on-the-fly using callback
            bleu_score, is_valid, generated = compute_bleu_for_code(
                prompt, reference, generate_fn, bleu_metric, model_label=model_label
            )

        # Store results
        bleu_scores.append(bleu_score)
        syntax_valid.append(is_valid)
        scored_codes.append(generated)

    # Compute statistics
    syntax_valid_count = sum(syntax_valid)
    syntax_valid_pct = (syntax_valid_count / len(syntax_valid)) * 100

    stats = {
        'mean': np.mean(bleu_scores),
        'median': np.median(bleu_scores),
        'std': np.std(bleu_scores),
        'min': np.min(bleu_scores),
        'max': np.max(bleu_scores),
        'syntax_valid_count': syntax_valid_count,
        'syntax_valid_pct': syntax_valid_pct
    }

    # Print summary
    print("\n" + "=" * 80)
    print(f"📊 {model_label.upper()} EVALUATION RESULTS")
    print("=" * 80)
    print(f"BLEU Score (mean):        {stats['mean']:.2f}")
    print(f"BLEU Score (median):      {stats['median']:.2f}")
    print(f"BLEU Score (std dev):     {stats['std']:.2f}")
    print(f"BLEU Score (min):         {stats['min']:.2f}")
    print(f"BLEU Score (max):         {stats['max']:.2f}")
    print(f"Syntax Validity:          {syntax_valid_count}/{len(syntax_valid)} ({syntax_valid_pct:.1f}%)")
    print("=" * 80)
    print(f"\n✅ {model_label} evaluation complete!")

    return bleu_scores, syntax_valid, scored_codes, stats


def analyze_bleu_results(bleu_scores, syntax_valid, generated_codes, test_prompts, test_codes, stats, model_label="model"):
    """
    Visualize and analyze BLEU evaluation results.

    Creates visualizations and displays best/worst examples to help understand
    model performance.

    Args:
        bleu_scores (list): BLEU score for each example
        syntax_valid (list): Boolean for each example indicating syntax validity
        generated_codes (list): Generated code for each example
        test_prompts (list): Problem descriptions
        test_codes (list): Reference solutions
        stats (dict): Statistics dictionary from evaluate_model_with_bleu()
        model_label (str): Model name for display

    Returns:
        None (displays visualizations and prints analysis)

    Example:
        >>> # After running evaluate_model_with_bleu:
        >>> analyze_bleu_results(
        ...     bleu_scores, syntax_valid, generated_codes,
        ...     test_prompts, test_codes, stats, "my-model"
        ... )
        # Displays histogram, pie chart, and example outputs
    """
    print("\n" + "=" * 80)
    print(f"📊 DETAILED ANALYSIS: {model_label.upper()}")
    print("=" * 80)

    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: BLEU Score Distribution (Histogram)
    axes[0].hist(bleu_scores, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.2f}')
    axes[0].axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.2f}')
    axes[0].set_xlabel('BLEU Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{model_label.upper()} - BLEU Score Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Syntax Validity (Pie Chart)
    syntax_count = sum(syntax_valid)
    syntax_invalid_count = len(syntax_valid) - syntax_count

    axes[1].pie(
        [syntax_count, syntax_invalid_count],
        labels=['Valid Syntax', 'Invalid Syntax'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'],
        startangle=90
    )
    axes[1].set_title(f'{model_label.upper()} - Syntax Validity\n({syntax_count}/{len(syntax_valid)} valid)')

    plt.tight_layout()
    plt.show()

    # Show best and worst examples
    print("\n" + "=" * 80)
    print(f"🏆 BEST {model_label.upper()} EXAMPLES (Highest BLEU)")
    print("=" * 80)

    # Get indices sorted by BLEU score (descending)
    sorted_indices = np.argsort(bleu_scores)[::-1]

    # Show top 3
    for rank, idx in enumerate(sorted_indices[:3], 1):
        print(f"\n{'=' * 80}")
        print(f"Rank #{rank} - Example {idx + 1}")
        print("=" * 80)
        print(f"BLEU Score: {bleu_scores[idx]:.2f}")
        print(f"Syntax Valid: {syntax_valid[idx]}")
        print(f"\n📝 Prompt:")
        print(test_prompts[idx][:150] + ("..." if len(test_prompts[idx]) > 150 else ""))
        print(f"\n🎯 Reference:")
        print(test_codes[idx][:150] + ("..." if len(test_codes[idx]) > 150 else ""))
        print(f"\n🤖 Generated ({model_label}):")
        print(generated_codes[idx][:150] + ("..." if len(generated_codes[idx]) > 150 else ""))

    print("\n" + "=" * 80)
    print(f"💔 WORST {model_label.upper()} EXAMPLES (Lowest BLEU)")
    print("=" * 80)

    # Show bottom 3
    for rank, idx in enumerate(sorted_indices[-3:][::-1], 1):
        print(f"\n{'=' * 80}")
        print(f"Rank #{rank} from bottom - Example {idx + 1}")
        print("=" * 80)
        print(f"BLEU Score: {bleu_scores[idx]:.2f}")
        print(f"Syntax Valid: {syntax_valid[idx]}")
        print(f"\n📝 Prompt:")
        print(test_prompts[idx][:150] + ("..." if len(test_prompts[idx]) > 150 else ""))
        print(f"\n🎯 Reference:")
        print(test_codes[idx][:150] + ("..." if len(test_codes[idx]) > 150 else ""))
        print(f"\n🤖 Generated ({model_label}):")
        print(generated_codes[idx][:150] + ("..." if len(generated_codes[idx]) > 150 else ""))

    print("\n" + "=" * 80)
    print(f"✅ {model_label} analysis complete!")
    print("=" * 80)


# ============================================
# SECTION 5: TRAINING UTILITIES
# ============================================
# Helper functions for model training (callbacks, plotting, etc.)


def plot_training_history(history_obj):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history_obj: Keras History object returned by model.fit()
    """
    history_dict = history_obj.history

    loss = history_dict["loss"]
    val_loss = history_dict.get("val_loss")
    acc = history_dict.get("accuracy")
    val_acc = history_dict.get("val_accuracy")

    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label="Train Loss")
    if val_loss is not None:
        plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()

    # Accuracy
    if acc is not None:
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, acc, label="Train Accuracy")
        if val_acc is not None:
            plt.plot(epochs_range, val_acc, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training vs. Validation Accuracy")
        plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================
# SECTION 6: TRANSFORMER BUILDING BLOCKS
# ============================================
# Core transformer components used in Part A.
# Reference: Vaswani et al., "Attention Is All You Need" (2017)
#            https://arxiv.org/abs/1706.03762

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding from the original Transformer paper.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_length: Maximum sequence length
        d_model: Embedding / model dimension
    """

    def __init__(self, max_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model

        position = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = np.zeros((max_length, d_model), dtype="float32")
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.position_encoding = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.position_encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_length": self.max_length, "d_model": self.d_model})
        return config


class ScaledDotProductAttention(layers.Layer):
    """
    Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q · K^T / sqrt(d_k)) · V
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_scores = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_scores += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_scores, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights


class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention — runs scaled dot-product attention in parallel across heads.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of parallel attention heads
    """

    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = self.attention(q, k, v, mask=mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config


def create_padding_mask(seq):
    """Create padding mask: 1 where token == 0. Shape: (batch, 1, 1, seq_len)."""
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(seq_len):
    """Create causal look-ahead mask: 1 in upper triangle. Shape: (seq_len, seq_len)."""
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)


def create_decoder_mask(target_seq):
    """Combine padding and look-ahead masks for the decoder. Shape: (batch, 1, seq_len, seq_len)."""
    seq_len = tf.shape(target_seq)[1]
    look_ahead = create_look_ahead_mask(seq_len)
    padding_mask = create_padding_mask(target_seq)
    look_ahead = tf.reshape(look_ahead, (1, 1, seq_len, seq_len))
    return tf.maximum(look_ahead, padding_mask)


def point_wise_feed_forward_network(d_model, d_ff):
    """Position-wise feed-forward network: Dense(d_ff, relu) → Dense(d_model)."""
    return keras.Sequential([
        layers.Dense(d_ff, activation="relu"),
        layers.Dense(d_model),
    ])


class EncoderBlock(layers.Layer):
    """
    Single encoder block: Self-Attention → Add & Norm → FFN → Add & Norm.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        dropout_rate: Dropout probability
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, padding_mask, training=False):
        attn_output, _ = self.mha(v=x, k=x, q=x, mask=padding_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.mha.d_model, "num_heads": self.mha.num_heads})
        return config


class DecoderBlock(layers.Layer):
    """
    Single decoder block with three sub-layers:
    1. Masked Self-Attention  2. Cross-Attention  3. Feed-Forward Network

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        dropout_rate: Dropout probability
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=False):
        attn1, attn_weights_block1 = self.mha1(v=x, k=x, q=x, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, attn_weights_block2 = self.mha2(v=enc_output, k=enc_output, q=out1, mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output), attn_weights_block1, attn_weights_block2


def build_transformer_model(
    input_vocab_size,
    target_vocab_size,
    num_layers=2,
    d_model=256,
    num_heads=4,
    d_ff=512,
    dropout_rate=0.1,
    max_input_len=100,
    max_target_len=99,
):
    """
    Build a full encoder-decoder transformer model.

    Args:
        input_vocab_size: Size of input (prompt) vocabulary
        target_vocab_size: Size of target (code) vocabulary
        num_layers: Number of encoder/decoder blocks to stack
        d_model: Model / embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        dropout_rate: Dropout probability
        max_input_len: Maximum input sequence length
        max_target_len: Maximum target sequence length

    Returns:
        Keras Model with inputs {"encoder_inputs", "decoder_inputs"}
    """
    encoder_inputs = keras.Input(shape=(max_input_len,), name="encoder_inputs")
    decoder_inputs = keras.Input(shape=(max_target_len,), name="decoder_inputs")

    enc_padding_mask = layers.Lambda(create_padding_mask)(encoder_inputs)
    dec_combined_mask = layers.Lambda(create_decoder_mask)(decoder_inputs)
    dec_padding_mask = enc_padding_mask

    enc_emb = layers.Embedding(input_vocab_size, d_model, mask_zero=False)(encoder_inputs)
    enc_emb = PositionalEncoding(max_length=max_input_len, d_model=d_model)(enc_emb)
    x = layers.Dropout(dropout_rate)(enc_emb)

    for _ in range(num_layers):
        x = EncoderBlock(d_model, num_heads, d_ff, dropout_rate)(x, enc_padding_mask)
    enc_output = x

    dec_emb = layers.Embedding(target_vocab_size, d_model, mask_zero=False)(decoder_inputs)
    dec_emb = PositionalEncoding(max_length=max_target_len, d_model=d_model)(dec_emb)
    y = layers.Dropout(dropout_rate)(dec_emb)

    for _ in range(num_layers):
        y, _, _ = DecoderBlock(d_model, num_heads, d_ff, dropout_rate)(
            y, enc_output, dec_combined_mask, dec_padding_mask
        )

    final_output = layers.Dense(target_vocab_size, activation="softmax")(y)

    return keras.Model(
        inputs={"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
        outputs=final_output,
    )


# ============================================
# SECTION 7: DECODING & GENERATION UTILITIES
# ============================================
# Helper functions for inference / code generation from a trained transformer.


def ids_to_code_text(token_ids, id_to_token, start_token="[START]", end_token="[END]"):
    """
    Convert a 1D array of token ids back into a code string.
    Skips padding (id=0), [START], and [END] tokens.

    Args:
        token_ids: Array of integer token IDs
        id_to_token: Dict mapping token ID → token string
        start_token: Start-of-sequence marker
        end_token: End-of-sequence marker

    Returns:
        Decoded code string
    """
    tokens = []
    start_lower = start_token.lower()
    end_lower = end_token.lower()
    for tid in token_ids:
        if tid == 0:
            continue
        tok = id_to_token.get(int(tid), "")
        if tok.lower() in ("", start_lower, end_lower):
            continue
        tokens.append(tok)
    return " ".join(tokens)


def strip_special_tokens_from_target(raw_target_str, start_token="[START]", end_token="[END]"):
    """
    Remove [START] and [END] markers from a raw target string.

    Args:
        raw_target_str: e.g. "[START] def foo(x): ... [END]"

    Returns:
        Cleaned code string
    """
    return raw_target_str.replace(start_token, "").replace(end_token, "").strip()


def generate_code_for_prompt(
    prompt_text,
    model,
    input_vectorizer,
    start_id,
    end_id,
    id_to_token,
    max_len=99,
    start_token="[START]",
    end_token="[END]",
):
    """
    Generate code using greedy decoding (argmax at each step).

    Args:
        prompt_text: Natural language prompt string
        model: Trained transformer model
        input_vectorizer: TextVectorization layer for prompts
        start_id: Token ID for [START]
        end_id: Token ID for [END]
        id_to_token: Dict mapping token ID → token string
        max_len: Maximum generation length

    Returns:
        Generated code string
    """
    encoder_inputs = input_vectorizer([prompt_text]).numpy()
    generated_tokens = [start_id]

    for t in range(1, max_len):
        current_length = len(generated_tokens)
        decoder_inputs = np.zeros((1, max_len), dtype="int32")
        decoder_inputs[0, :current_length] = generated_tokens

        preds = model(
            {"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
            training=False,
        )

        next_token_logits = preds[0, current_length - 1]
        next_token_id = int(tf.argmax(next_token_logits).numpy())
        generated_tokens.append(next_token_id)

        if next_token_id == end_id:
            break

    return ids_to_code_text(generated_tokens[1:], id_to_token, start_token, end_token)


def generate_best(
    prompt_text,
    model,
    input_vectorizer,
    start_id,
    end_id,
    id_to_token,
    max_len=99,
    temperature=0.7,
    repetition_penalty=1.3,
    top_p=0.9,
    start_token="[START]",
    end_token="[END]",
):
    """
    Generate code with repetition penalty + nucleus (top-p) sampling.

    Args:
        prompt_text: Natural language prompt string
        model: Trained transformer model
        input_vectorizer: TextVectorization layer for prompts
        start_id: Token ID for [START]
        end_id: Token ID for [END]
        id_to_token: Dict mapping token ID → token string
        max_len: Maximum generation length
        temperature: Sampling temperature (lower = more deterministic)
        repetition_penalty: Penalty factor for repeated tokens (>1.0)
        top_p: Nucleus sampling threshold (0-1)

    Returns:
        Generated code string
    """
    encoder_inputs = input_vectorizer([prompt_text]).numpy()
    decoder_inputs = np.zeros((1, max_len), dtype="int32")
    decoder_inputs[0, 0] = start_id
    token_counts = {}

    for t in range(1, max_len):
        preds = model(
            {"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
            training=False,
        )
        next_token_logits = preds[0, t - 1].numpy()

        for token_id, count in token_counts.items():
            next_token_logits[token_id] = next_token_logits[token_id] / (repetition_penalty ** count)

        next_token_logits[0] = -1e9  # suppress padding
        next_token_logits = next_token_logits / temperature
        next_token_probs = tf.nn.softmax(next_token_logits).numpy()

        sorted_indices = np.argsort(next_token_probs)[::-1]
        sorted_probs = next_token_probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)
        nucleus_size = np.searchsorted(cumsum_probs, top_p) + 1
        nucleus_indices = sorted_indices[:nucleus_size]
        nucleus_probs = sorted_probs[:nucleus_size]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        next_token_id = np.random.choice(nucleus_indices, p=nucleus_probs)
        token_counts[next_token_id] = token_counts.get(next_token_id, 0) + 1
        decoder_inputs[0, t] = next_token_id

        if next_token_id == end_id:
            break

    return ids_to_code_text(decoder_inputs[0, 1:], id_to_token, start_token, end_token)


# ============================================
# SECTION 8: BATCH GENERATION UTILITIES
# ============================================
# Pre-generate code for all test prompts before BLEU scoring.
# Two variants:
#   - batch_generate_codes()      — HuggingFace model/tokenizer (PartB, PartC)
#   - sequential_generate_codes() — custom TF transformer (PartA)


def batch_generate_codes(
    prompts,
    model,
    tokenizer,
    batch_size=8,
    max_length=128,
    num_beams=1,
):
    """
    Generate code for multiple prompts using batched HuggingFace inference.

    Use this for HuggingFace seq2seq models (e.g., CodeT5+).

    Args:
        prompts (list): List of natural language prompt strings
        model: HuggingFace TF seq2seq model
        tokenizer: HuggingFace tokenizer matching the model
        batch_size (int): Number of prompts per batch
        max_length (int): Maximum number of tokens to generate
        num_beams (int): Beam search width (1 = greedy decoding)

    Returns:
        list: Generated code strings, one per prompt
    """
    all_generated = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch generation"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="tf",
            truncation=True,
            max_length=128,
            padding=True,
        )
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        for output in outputs:
            all_generated.append(tokenizer.decode(output, skip_special_tokens=True))
    return all_generated


def sequential_generate_codes(prompts, generate_fn, desc="Generating code"):
    """
    Generate code for multiple prompts sequentially using a provided function.

    Use this for custom TF transformers (PartA) where true batching is not
    straightforward. Shows a tqdm progress bar during generation.

    Args:
        prompts (list): List of natural language prompt strings
        generate_fn (callable): Function that takes a single prompt string
                                and returns a generated code string
        desc (str): Label shown in the tqdm progress bar

    Returns:
        list: Generated code strings, one per prompt
    """
    return [generate_fn(p) for p in tqdm(prompts, desc=desc)]


# ============================================
# MODULE INFO
# ============================================

__version__ = "1.0.0"
__all__ = [
    # Configuration
    "DATASET",
    "DATASET_CONFIG",
    "get_dataset_config",
    # Data loading
    "load_dataset_splits",
    "extract_mbpp_example",
    "extract_codecontests_example",
    "extract_example",
    # Data processing
    "process_dataset_split",
    "prepare_training_data",
    # BLEU evaluation
    "add_python_hint",
    "compute_bleu_for_code",
    "evaluate_model_with_bleu",
    "analyze_bleu_results",
    # Training utilities (Section 5)
    "plot_training_history",
    # Transformer building blocks (Section 6)
    "PositionalEncoding",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "create_padding_mask",
    "create_look_ahead_mask",
    "create_decoder_mask",
    "point_wise_feed_forward_network",
    "EncoderBlock",
    "DecoderBlock",
    "build_transformer_model",
    # Decoding & generation (Section 7)
    "ids_to_code_text",
    "strip_special_tokens_from_target",
    "generate_code_for_prompt",
    "generate_best",
    # Batch generation utilities (Section 8)
    "batch_generate_codes",
    "sequential_generate_codes",
]

print(f"✅ assignment3_utils v{__version__} loaded successfully")
