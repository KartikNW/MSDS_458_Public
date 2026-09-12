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
import math
import numpy as np

# PyTorch (transformer building blocks, decoding, batch generation)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary building (TextVectorizer)
from collections import Counter

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
        "name": "google-research-datasets/mbpp",
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


def extract_example(example, dataset_name=None):
    """
    Extract prompt and code from MBPP dataset example.

    Args:
        example (dict): Dataset example.
        dataset_name (str, optional): Dataset name. If None, uses global DATASET variable.

    Returns:
        tuple: (prompt, code) or (None, None) if extraction fails.

    Example:
        >>> prompt, code = extract_example(mbpp_example, "MBPP")
    """
    if dataset_name is None:
        dataset_name = DATASET

    if dataset_name == "MBPP":
        return extract_mbpp_example(example)
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
    code_list = []
    count = 0

    for example in split_data:
        if count >= max_examples:
            break

        prompt, code = extract_example(example, dataset_name)

        if prompt is not None and code is not None:
            prompts.append(prompt)
            code_list.append(code)
            count += 1

    return {
        'prompts': prompts,
        'code': code_list
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


def evaluate_model_with_bleu(test_prompts, test_code, generate_fn, bleu_metric, model_label="model", generated_code=None):
    """
    Evaluate a model on entire test set using BLEU metric.

    This function is generic and can be reused for any model (zero-shot, fine-tuned,
    from-scratch, etc.) by providing a different generate_fn.

    Args:
        test_prompts (list): List of problem descriptions (strings)
        test_code (list): List of reference code solutions (strings)
        generate_fn (callable): Function that takes (prompt) and returns generated code.
                                Ignored if generated_code is provided.
        bleu_metric: Loaded BLEU metric from evaluate library
        model_label (str): String label for this evaluation (e.g., "zero-shot", "fine-tuned")
        generated_code (list, optional): Pre-generated code (e.g., from batch generation).
                                         If provided, skips calling generate_fn and scores
                                         these directly. Use this for faster evaluation.

    Returns:
        tuple: (bleu_scores, syntax_valid, generated_code, stats_dict)
            - bleu_scores (list): BLEU score for each example
            - syntax_valid (list): Boolean for each example indicating syntax validity
            - generated_code (list): Generated code for each example
            - stats_dict (dict): Statistics with keys:
                - 'mean', 'median', 'std', 'min', 'max': BLEU statistics
                - 'syntax_valid_count': Number of syntactically valid examples
                - 'syntax_valid_pct': Percentage of syntactically valid examples

    Example:
        >>> prompts = ["Write function A", "Write function B"]
        >>> reference_code = ["def a(): pass", "def b(): pass"]
        >>>
        >>> def my_gen(p):
        ...     return "def solution(): pass"
        >>>
        >>> bleu_metric = evaluate.load("bleu")
        >>> scores, valid, gen, stats = evaluate_model_with_bleu(
        ...     prompts, reference_code, my_gen, bleu_metric, "test-model"
        ... )
        >>> print(f"Mean BLEU: {stats['mean']:.2f}")
        >>>
        >>> # Or with pre-generated code (skips generation, much faster):
        >>> pre_gen = ["def a(): pass", "def b(): pass"]
        >>> scores, valid, gen, stats = evaluate_model_with_bleu(
        ...     prompts, reference_code, None, bleu_metric, "test-model",
        ...     generated_code=pre_gen
        ... )
    """
    using_pregenerated = generated_code is not None
    if using_pregenerated:
        print(f"🔬 Scoring {model_label.upper()} on {len(test_prompts)} test examples (pre-generated)...")
    else:
        print(f"🔬 Evaluating {model_label.upper()} model on {len(test_prompts)} test examples...")
        print(f"This will take ~2-3 minutes (generating code for each example)")
    print("=" * 80)

    # Storage for results
    bleu_scores = []
    syntax_valid = []
    scored_code = []

    # Evaluate each test example
    for i in tqdm(range(len(test_prompts)), desc=f"{model_label} BLEU Evaluation"):
        prompt = test_prompts[i]
        reference = test_code[i]

        if using_pregenerated:
            # Use pre-generated code — just compute BLEU + syntax validity
            generated = generated_code[i]
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
        scored_code.append(generated)

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

    return bleu_scores, syntax_valid, scored_code, stats


def analyze_bleu_results(bleu_scores, syntax_valid, generated_code, test_prompts, test_code, stats, model_label="model"):
    """
    Visualize and analyze BLEU evaluation results.

    Creates visualizations and displays best/worst examples to help understand
    model performance.

    Args:
        bleu_scores (list): BLEU score for each example
        syntax_valid (list): Boolean for each example indicating syntax validity
        generated_code (list): Generated code for each example
        test_prompts (list): Problem descriptions
        test_code (list): Reference solutions
        stats (dict): Statistics dictionary from evaluate_model_with_bleu()
        model_label (str): Model name for display

    Returns:
        None (displays visualizations and prints analysis)

    Example:
        >>> # After running evaluate_model_with_bleu:
        >>> analyze_bleu_results(
        ...     bleu_scores, syntax_valid, generated_code,
        ...     test_prompts, test_code, stats, "my-model"
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
        print(test_code[idx][:150] + ("..." if len(test_code[idx]) > 150 else ""))
        print(f"\n🤖 Generated ({model_label}):")
        print(generated_code[idx][:150] + ("..." if len(generated_code[idx]) > 150 else ""))

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
        print(test_code[idx][:150] + ("..." if len(test_code[idx]) > 150 else ""))
        print(f"\n🤖 Generated ({model_label}):")
        print(generated_code[idx][:150] + ("..." if len(generated_code[idx]) > 150 else ""))

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
        history_obj: Either a plain dict with keys "loss", "val_loss",
                     "accuracy", "val_accuracy" (as built by the PyTorch
                     training loop) or an object with a `.history` dict.
    """
    history_dict = history_obj.history if hasattr(history_obj, "history") else history_obj

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
# SECTION 5B: TEXT VECTORIZATION (TOKENIZER)
# ============================================
# A small, dependency-free text vectorizer producing integer token ids.
# Behavior: standardize → whitespace split → frequency-ranked vocabulary
# with '' (padding, id 0) and '[UNK]' (OOV, id 1) reserved, fixed-length output.


class TextVectorizer:
    """
    Simple whitespace tokenizer + vocabulary lookup.

    - Index 0 is reserved for padding ('').
    - Index 1 is reserved for out-of-vocabulary tokens ('[UNK]').
    - Vocabulary is built by token frequency (most frequent first),
      capped at max_tokens (including the two reserved slots).
    - Output sequences are padded with 0 / truncated to output_sequence_length.

    Args:
        max_tokens: Maximum vocabulary size (including '' and '[UNK]')
        output_sequence_length: Fixed output length per sequence
        standardize: Callable applied to each string before splitting
                     (default: lowercase)

    Example:
        >>> vectorizer = TextVectorizer(max_tokens=100, output_sequence_length=8)
        >>> vectorizer.adapt(["write a function", "write a class"])
        >>> vectorizer(["write a function"])
        array([[2, 3, 4, 0, 0, 0, 0, 0]])
    """

    def __init__(self, max_tokens, output_sequence_length, standardize=None):
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.standardize = standardize if standardize is not None else (lambda s: s.lower())
        self.vocab = ["", "[UNK]"]
        self.token_to_id = {"": 0, "[UNK]": 1}

    def adapt(self, texts):
        """Build the vocabulary from an iterable of strings."""
        counter = Counter()
        for text in texts:
            counter.update(self.standardize(str(text)).split())
        most_common = counter.most_common(self.max_tokens - 2)
        self.vocab = ["", "[UNK]"] + [tok for tok, _ in most_common]
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}

    def __call__(self, texts):
        """Convert string(s) to an int array of shape (n, output_sequence_length)."""
        if isinstance(texts, str):
            texts = [texts]
        seq_len = self.output_sequence_length
        output = np.zeros((len(texts), seq_len), dtype="int64")
        for row, text in enumerate(texts):
            tokens = self.standardize(str(text)).split()[:seq_len]
            for col, tok in enumerate(tokens):
                output[row, col] = self.token_to_id.get(tok, 1)  # 1 = [UNK]
        return output

    def get_vocabulary(self):
        """Return the vocabulary as a list of tokens (list index = token id)."""
        return list(self.vocab)


# ============================================
# SECTION 6: TRANSFORMER BUILDING BLOCKS
# ============================================
# Core transformer components used in Part A (PyTorch nn.Modules).
# Reference: Vaswani et al., "Attention Is All You Need" (2017)
#            https://arxiv.org/abs/1706.03762


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from the original Transformer paper.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_length: Maximum sequence length
        d_model: Embedding / model dimension
    """

    def __init__(self, max_length, d_model):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model

        position = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = np.zeros((max_length, d_model), dtype="float32")
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Buffer (not a parameter): moves with the model to GPU/MPS, never trained
        self.register_buffer("position_encoding", torch.from_numpy(pe).unsqueeze(0))

    def forward(self, inputs):
        seq_len = inputs.size(1)
        return inputs + self.position_encoding[:, :seq_len, :]


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q · K^T / sqrt(d_k)) · V
    """

    def forward(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = k.size(-1)
        scaled_scores = matmul_qk / math.sqrt(dk)
        if mask is not None:
            scaled_scores = scaled_scores + (mask * -1e9)
        attention_weights = F.softmax(scaled_scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention — runs scaled dot-product attention in parallel across heads.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of parallel attention heads
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = self.attention(q, k, v, mask=mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)
        return output, attention_weights


def create_padding_mask(seq):
    """Create padding mask: 1.0 where token == 0. Shape: (batch, 1, 1, seq_len)."""
    mask = (seq == 0).float()
    return mask[:, None, None, :]


def create_look_ahead_mask(seq_len):
    """Create causal look-ahead mask: 1 in upper triangle. Shape: (seq_len, seq_len)."""
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1)


def create_decoder_mask(target_seq):
    """Combine padding and look-ahead masks for the decoder. Shape: (batch, 1, seq_len, seq_len)."""
    seq_len = target_seq.size(1)
    look_ahead = create_look_ahead_mask(seq_len).to(target_seq.device)
    padding_mask = create_padding_mask(target_seq)
    look_ahead = look_ahead.view(1, 1, seq_len, seq_len)
    return torch.maximum(look_ahead, padding_mask)


def point_wise_feed_forward_network(d_model, d_ff):
    """Position-wise feed-forward network: Linear(d_ff) + ReLU → Linear(d_model)."""
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model),
    )


class EncoderBlock(nn.Module):
    """
    Single encoder block: Self-Attention → Add & Norm → FFN → Add & Norm.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        dropout_rate: Dropout probability
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask):
        attn_output, _ = self.mha(v=x, k=x, q=x, mask=padding_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class DecoderBlock(nn.Module):
    """
    Single decoder block with three sub-layers:
    1. Masked Self-Attention  2. Cross-Attention  3. Feed-Forward Network

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        dropout_rate: Dropout probability
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(v=x, k=x, q=x, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        attn2, attn_weights_block2 = self.mha2(v=enc_output, k=enc_output, q=out1, mask=padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        return self.layernorm3(out2 + ffn_output), attn_weights_block1, attn_weights_block2


class Transformer(nn.Module):
    """
    Full encoder-decoder transformer.

    forward(encoder_inputs, decoder_inputs) → logits of shape
    (batch, target_seq_len, target_vocab_size).

    Note: this model returns raw **logits**, not probabilities.
    nn.CrossEntropyLoss applies the softmax internally, which is the standard
    PyTorch pattern.
    """

    def __init__(
        self,
        input_vocab_size,
        target_vocab_size,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        dropout_rate,
        max_input_len,
        max_target_len,
    ):
        super().__init__()
        self.target_vocab_size = target_vocab_size

        # Encoder: Embedding → Positional Encoding → Dropout → N × EncoderBlock
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.encoder_pos_encoding = PositionalEncoding(max_length=max_input_len, d_model=d_model)
        self.encoder_dropout = nn.Dropout(dropout_rate)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        )

        # Decoder: Embedding → Positional Encoding → Dropout → N × DecoderBlock
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.decoder_pos_encoding = PositionalEncoding(max_length=max_target_len, d_model=d_model)
        self.decoder_dropout = nn.Dropout(dropout_rate)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        )

        # Output projection to target vocabulary (logits — no softmax here)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, encoder_inputs, decoder_inputs):
        # Build masks from the raw token ids
        enc_padding_mask = create_padding_mask(encoder_inputs)
        dec_combined_mask = create_decoder_mask(decoder_inputs)
        dec_padding_mask = enc_padding_mask

        # Encoder pass
        x = self.encoder_embedding(encoder_inputs)
        x = self.encoder_pos_encoding(x)
        x = self.encoder_dropout(x)
        for block in self.encoder_blocks:
            x = block(x, enc_padding_mask)
        enc_output = x

        # Decoder pass
        y = self.decoder_embedding(decoder_inputs)
        y = self.decoder_pos_encoding(y)
        y = self.decoder_dropout(y)
        for block in self.decoder_blocks:
            y, _, _ = block(y, enc_output, dec_combined_mask, dec_padding_mask)

        return self.final_layer(y)


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
        PyTorch nn.Module — call as model(encoder_inputs, decoder_inputs),
        returns logits of shape (batch, target_seq_len, target_vocab_size)
    """
    return Transformer(
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
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
        model: Trained transformer model (PyTorch nn.Module)
        input_vectorizer: TextVectorizer for prompts
        start_id: Token ID for [START]
        end_id: Token ID for [END]
        id_to_token: Dict mapping token ID → token string
        max_len: Maximum generation length

    Returns:
        Generated code string
    """
    device = next(model.parameters()).device
    model.eval()

    encoder_inputs = torch.as_tensor(input_vectorizer([prompt_text]), dtype=torch.long, device=device)
    generated_tokens = [start_id]

    with torch.no_grad():
        for t in range(1, max_len):
            current_length = len(generated_tokens)
            decoder_inputs = torch.zeros((1, max_len), dtype=torch.long, device=device)
            decoder_inputs[0, :current_length] = torch.as_tensor(generated_tokens, dtype=torch.long, device=device)

            logits = model(encoder_inputs, decoder_inputs)

            next_token_logits = logits[0, current_length - 1]
            next_token_id = int(torch.argmax(next_token_logits).item())
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
        model: Trained transformer model (PyTorch nn.Module)
        input_vectorizer: TextVectorizer for prompts
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
    device = next(model.parameters()).device
    model.eval()

    encoder_inputs = torch.as_tensor(input_vectorizer([prompt_text]), dtype=torch.long, device=device)
    decoder_inputs = torch.zeros((1, max_len), dtype=torch.long, device=device)
    decoder_inputs[0, 0] = start_id
    token_counts = {}

    with torch.no_grad():
        for t in range(1, max_len):
            logits = model(encoder_inputs, decoder_inputs)
            # The model emits logits. Convert to probabilities first, because the
            # repetition-penalty and temperature math below assumes probabilities.
            next_token_logits = F.softmax(logits[0, t - 1], dim=-1).cpu().numpy()

            for token_id, count in token_counts.items():
                next_token_logits[token_id] = next_token_logits[token_id] / (repetition_penalty ** count)

            next_token_logits[0] = -1e9  # suppress padding
            next_token_logits = next_token_logits / temperature
            shifted = next_token_logits - next_token_logits.max()  # numerically stable softmax
            next_token_probs = np.exp(shifted) / np.exp(shifted).sum()

            sorted_indices = np.argsort(next_token_probs)[::-1]
            sorted_probs = next_token_probs[sorted_indices]
            cumsum_probs = np.cumsum(sorted_probs)
            nucleus_size = np.searchsorted(cumsum_probs, top_p) + 1
            nucleus_indices = sorted_indices[:nucleus_size]
            nucleus_probs = sorted_probs[:nucleus_size]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()

            next_token_id = int(np.random.choice(nucleus_indices, p=nucleus_probs))
            token_counts[next_token_id] = token_counts.get(next_token_id, 0) + 1
            decoder_inputs[0, t] = next_token_id

            if next_token_id == end_id:
                break

    return ids_to_code_text(decoder_inputs[0, 1:].cpu().numpy(), id_to_token, start_token, end_token)


# ============================================
# SECTION 8: BATCH GENERATION UTILITIES
# ============================================
# Pre-generate code for all test prompts before BLEU scoring.
# Two variants:
#   - batch_generate_code()      — HuggingFace model/tokenizer (PartB, PartC)
#   - sequential_generate_code() — custom PyTorch transformer (PartA)


def batch_generate_code(
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
        model: HuggingFace PyTorch seq2seq model
        tokenizer: HuggingFace tokenizer matching the model
        batch_size (int): Number of prompts per batch
        max_length (int): Maximum number of tokens to generate
        num_beams (int): Beam search width (1 = greedy decoding)

    Returns:
        list: Generated code strings, one per prompt
    """
    device = next(model.parameters()).device
    model.eval()

    all_generated = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch generation"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        all_generated.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return all_generated


def sequential_generate_code(prompts, generate_fn, desc="Generating code"):
    """
    Generate code for multiple prompts sequentially using a provided function.

    Use this for the custom PyTorch transformer (PartA) where true batching is
    not straightforward. Shows a tqdm progress bar during generation.

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
# SECTION 9: CROSS-PART RESULTS HANDOFF
# ============================================
# Part C compares three models but trains only one. Parts A and B run in
# separate notebooks (and, on Colab, separate sessions), so their results have
# to be carried across. Two routes are supported:
#
#   1. a results file written by Parts A/B and read by Part C (works when the
#      notebooks share a directory, which is the usual case running locally)
#   2. a block of assignments printed by Parts A/B for the student to paste
#      into Part C (works everywhere, including Colab)
#
# If neither is present, Part C falls back to the reference values written into
# the notebook, and labels every output accordingly, so a comparison is never
# presented as the student's own work when it is not.

_RESULT_KEYS = ("bleu_mean", "bleu_min", "bleu_max", "syntax_pct")


def _results_filename(part):
    """Filename this module reads and writes for a given part ('A' or 'B')."""
    return f"part{part.upper()}_results.json"


def save_part_results(part, stats, out_dir=".", source=None):
    """Save Part A/B results for Part C, and print a paste-able copy.

    Writes ``partX_results.json`` next to the notebook and prints the same
    numbers as assignment statements, so the results can travel either as a
    file (local) or via copy and paste (Colab).

    Args:
        part (str): "A" or "B".
        stats (dict): The stats dict from ``evaluate_model_with_bleu``; needs
            'mean', 'min', 'max' and 'syntax_valid_pct'.
        out_dir (str): Directory to write the file into. Defaults to the
            working directory.
        source (str): Provenance label recorded in the file. Defaults to
            "my own runs", which is what lets Part C collapse the Part A and
            Part B labels into one when both files are present.

    Returns:
        dict: The payload that was written.
    """
    import json
    import os
    from datetime import datetime

    part = part.upper()
    if part not in ("A", "B"):
        raise ValueError(f"part must be 'A' or 'B', got {part!r}")

    missing = [k for k in ("mean", "min", "max", "syntax_valid_pct") if k not in stats]
    if missing:
        raise KeyError(f"stats is missing {missing}; expected the dict from evaluate_model_with_bleu()")

    payload = {
        "part": part,
        "source": source or "my own runs",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "bleu_mean": float(stats["mean"]),
            "bleu_min": float(stats["min"]),
            "bleu_max": float(stats["max"]),
            "syntax_pct": float(stats["syntax_valid_pct"]),
        },
    }

    path = os.path.join(out_dir, _results_filename(part))
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)

    m = payload["metrics"]
    print(f"✅ Saved {path}")
    print(f"   Part C picks this up automatically if it runs in the same folder.")
    print()
    where = "Part B (Step 6) and Part C (Step 8)" if part == "A" else "Part C (Step 8)"
    print(f"   If you are on Colab, or running the later notebooks elsewhere,")
    print(f"   copy these lines into {where}, over the values already there:")
    print()
    print(f'PART_{part}_RESULTS = {{"bleu_mean": {m["bleu_mean"]:.2f}, '
          f'"bleu_min": {m["bleu_min"]:.2f}, "bleu_max": {m["bleu_max"]:.2f}, '
          f'"syntax_pct": {m["syntax_pct"]:.1f}}}')
    print('RESULTS_SOURCE = "my own runs"')
    print()
    return payload


def _load_one(part, search_dir):
    """Return (metrics, label) from a results file, or (None, None)."""
    import json
    import os

    path = os.path.join(search_dir, _results_filename(part))
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as fh:
            payload = json.load(fh)
        metrics = payload["metrics"]
        missing = [k for k in _RESULT_KEYS if k not in metrics]
        if missing:
            print(f"⚠️  {path} is missing {missing}; ignoring it.")
            return None, None
        # Keep the label short: it is printed beside every result and appears in
        # a chart legend, so the full path and timestamp would swamp both.
        run_day = str(payload.get("run_at", "unknown"))[:10]
        label = f"{payload.get('source', 'unknown run')} (file, {run_day})"
        return {k: float(metrics[k]) for k in _RESULT_KEYS}, label
    except Exception as exc:
        print(f"⚠️  Could not read {path} ({type(exc).__name__}: {exc}); ignoring it.")
        return None, None


def load_part_results(part, reference, reference_source, search_dir="."):
    """Load one earlier part's results, falling back to the values passed in.

    Part B needs Part A's numbers; Part C needs both. This is the single-part
    version, used by Part B directly and by ``resolve_part_results`` for each of
    the two parts it resolves.

    Args:
        part (str): "A" or "B", the part whose results file to look for.
        reference (dict): Values to use when no file is found. Needs the keys
            'bleu_mean', 'bleu_min', 'bleu_max', 'syntax_pct'.
        reference_source (str): Label describing where `reference` came from.
        search_dir (str): Directory to look for the results file in.

    Returns:
        tuple: (values, source_label).
    """
    missing = [k for k in _RESULT_KEYS if k not in reference]
    if missing:
        raise KeyError(f"reference is missing {missing}; expected keys {list(_RESULT_KEYS)}")
    from_file, label = _load_one(part, search_dir)
    if from_file:
        return from_file, label
    return dict(reference), reference_source


def warn_if_reference(source, reference_source):
    """Print a warning when results are still the instructor's, not the student's."""
    if source == reference_source and reference_source.lower().startswith("instructor"):
        print()
        print("⚠️  These are the INSTRUCTOR'S numbers, not yours. Before submitting,")
        print("    run the earlier parts and either keep their results files next to")
        print("    this notebook, or paste the block they print over the values above.")


def resolve_part_results(part_a, part_b, reference_source, search_dir="."):
    """Resolve the Part A and Part B results Part C compares against.

    Priority: a results file written by Parts A/B, else the values passed in
    (either the reference values in the notebook, or ones the student pasted).

    Args:
        part_a (dict): Part A values, keys 'bleu_mean', 'bleu_min', 'bleu_max',
            'syntax_pct'.
        part_b (dict): Part B values, same keys.
        reference_source (str): Label describing where the passed-in values
            came from, e.g. "instructor reference run".
        search_dir (str): Directory to look for results files in.

    Returns:
        tuple: (part_a, part_b, source_label). The label names whichever source
        actually supplied the numbers and is printed alongside every result.
    """
    resolved_a, label_a = load_part_results("A", part_a, reference_source, search_dir)
    resolved_b, label_b = load_part_results("B", part_b, reference_source, search_dir)

    if label_a == label_b:
        # Both from the same place, which is the normal case: one label reads
        # better than "A: x | B: x".
        source = label_a
    else:
        source = f"A: {label_a} | B: {label_b}"

    print(f"Parts A & B source: {source}")
    warn_if_reference(source, reference_source)
    return resolved_a, resolved_b, source


# ============================================
# MODULE INFO
# ============================================

__version__ = "2.2.0"
__all__ = [
    # Configuration
    "DATASET",
    "DATASET_CONFIG",
    "get_dataset_config",
    # Data loading
    "load_dataset_splits",
    "extract_mbpp_example",
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
    # Text vectorization (Section 5B)
    "TextVectorizer",
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
    "Transformer",
    "build_transformer_model",
    # Decoding & generation (Section 7)
    "ids_to_code_text",
    "strip_special_tokens_from_target",
    "generate_code_for_prompt",
    "generate_best",
    # Batch generation utilities (Section 8)
    "batch_generate_code",
    "sequential_generate_code",
    # Cross-part results handoff (Section 9)
    "save_part_results",
    "load_part_results",
    "resolve_part_results",
    "warn_if_reference",
]

print(f"✅ assignment3_utils v{__version__} loaded successfully")
