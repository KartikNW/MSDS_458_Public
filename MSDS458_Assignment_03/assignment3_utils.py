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


def evaluate_model_with_bleu(test_prompts, test_codes, generate_fn, bleu_metric, model_label="model"):
    """
    Evaluate a model on entire test set using BLEU metric.

    This function is generic and can be reused for any model (zero-shot, fine-tuned,
    from-scratch, etc.) by providing a different generate_fn.

    Args:
        test_prompts (list): List of problem descriptions (strings)
        test_codes (list): List of reference code solutions (strings)
        generate_fn (callable): Function that takes (prompt) and returns generated code
        bleu_metric: Loaded BLEU metric from evaluate library
        model_label (str): String label for this evaluation (e.g., "zero-shot", "fine-tuned")

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
    """
    print(f"🔬 Evaluating {model_label.upper()} model on {len(test_prompts)} test examples...")
    print(f"This will take ~2-3 minutes (generating code for each example)")
    print("=" * 80)

    # Storage for results
    bleu_scores = []
    syntax_valid = []
    generated_codes = []

    # Evaluate each test example
    for i in tqdm(range(len(test_prompts)), desc=f"{model_label} BLEU Evaluation"):
        prompt = test_prompts[i]
        reference = test_codes[i]

        # Use helper function to compute metrics
        bleu_score, is_valid, generated = compute_bleu_for_code(
            prompt, reference, generate_fn, bleu_metric, model_label=model_label
        )

        # Store results
        bleu_scores.append(bleu_score)
        syntax_valid.append(is_valid)
        generated_codes.append(generated)

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

    return bleu_scores, syntax_valid, generated_codes, stats


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
# Will be populated in future steps if needed


# ============================================
# MODULE INFO
# ============================================

__version__ = "1.0.0"
__all__ = [
    # Configuration (Step 2)
    "DATASET",
    "DATASET_CONFIG",
    "get_dataset_config",
    # Data loading (Step 3)
    "load_dataset_splits",
    "extract_mbpp_example",
    "extract_codecontests_example",
    "extract_example",
    # Data processing (Step 4)
    "process_dataset_split",
    "prepare_training_data",
    # BLEU evaluation (Step 5)
    "add_python_hint",
    "compute_bleu_for_code",
    "evaluate_model_with_bleu",
    "analyze_bleu_results",
]

print(f"✅ assignment3_utils v{__version__} loaded successfully")
