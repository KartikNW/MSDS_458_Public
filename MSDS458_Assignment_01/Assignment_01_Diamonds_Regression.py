# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: diamonds
#     language: python
#     name: python3
# ---

# %% [markdown]
# <img src="https://github.com/KartikNW/MSDS_458_Public/blob/main/images2/NorthwesternHeader.png?raw=1" />

# %% [markdown]
# # Assignment 01 – Diamonds Regression
#
# In this assignment, we use the **Seaborn Diamonds dataset**, which contains detailed information on **53,940 diamonds**, including attributes such as **carat**, **cut**, **color**, **clarity**, **depth**, **table**, and **price**. The goal is to build and evaluate deep learning models that can **predict a diamond’s price** based on its physical and categorical characteristics.
#
# The notebook walks through the **end-to-end regression workflow**, including:
#
# - Loading and exploring the dataset to understand key features and distributions  
# - Performing **data preprocessing**, including normalization, encoding of categorical variables, and train/validation/test splits  
# - Building a **neural network regression model** using TensorFlow/Keras  
# - **Training, evaluating, and tuning** the model to minimize prediction error  
# - Visualizing results through **loss curves and predicted vs. actual price plots** to assess model performance  
#
# By the end of this assignment, you will gain hands-on experience applying deep learning techniques to a real-world structured dataset and interpreting model outputs for a regression problem.
#

# %% [markdown]
# ## Import Required Libraries
#
# In this section, we import the essential Python libraries needed for data manipulation, visualization, and building deep learning models.  
#
# - **Pandas** and **NumPy** are used for data handling and numerical computations.  
# - **Matplotlib** and **Seaborn** help visualize feature distributions and model performance.  
# - **TensorFlow** and **Keras** are used to construct, train, and evaluate the neural network regression model.  
# - A private helper module, **`diamonds_utils.py`**, is also imported. This module contains utility functions that simplify repetitive tasks such as data preprocessing, visualization, and evaluation—making the notebook cleaner and easier to follow.  
#
# > ⚠️ **Note:** The `diamonds_utils.py` file is not part of the standard library and must be downloaded from the course GitHub repository before running this notebook.
#
#


# %%
import os, sys, importlib.util, urllib.request

IN_COLAB = 'google.colab' in sys.modules
UTILS_PATH = 'diamonds_utils.py'

def fetch_github_raw(user, repo, branch, file_path, local_path):
    url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"
    urllib.request.urlretrieve(url, local_path)
    print(f"Fetched {file_path} from branch '{branch}'.")

if IN_COLAB and not os.path.exists(UTILS_PATH):
    fetch_github_raw("kartikNW", "MSDS_458_Public", "future", 
                     "MSDS458_Assignment_01/diamonds_utils.py", UTILS_PATH)
else:
    print("Using local diamonds_utils.py")

# %%
# Set the environment variable to disable GPU
os.environ['DIAMONDS_DISABLE_GPU'] = '1'

# %%
import numpy as np
import tensorflow as tf
from packaging import version

from diamonds_utils import (
    load_data,
    display_data_info,
    analyze_categorical_variables,
    analyze_price_distribution,
    analyze_carat_price_relationship,
    analyze_cut_impact,
    analyze_correlations,
    prepare_data,
    train_model,
    plot_training_history,
    plot_percentage_error_by_price,
    evaluate_model,
    residual_plot,
    remove_specific_outliers
)

# %% [markdown]
# ### Version Requirements
#
# This assignment requires recent versions of **Python** and **TensorFlow/Keras** to ensure compatibility with the neural network code.  
#
# | Library | Minimum Version | Purpose |
# |----------|------------------|----------|
# | Python | 3.10 | Core language |
# | TensorFlow / Keras | 2.15 | Deep learning framework |
#
# If you're running this notebook locally, you can verify your setup using the code cell below.
#

# %%
print("Python version:", sys.version.split()[0])
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Minimum version requirements
min_versions = {
    "Python": "3.10",
    "TensorFlow": "2.15",
    "Keras": "3.0",
}

def check_version(name, current, minimum):
    if version.parse(current) < version.parse(minimum):
        print(f"⚠️ {name} version {current} < required {minimum}. Please upgrade.")

check_version("Python", sys.version.split()[0], min_versions["Python"])
check_version("TensorFlow", tf.__version__, min_versions["TensorFlow"])
check_version("Keras", tf.keras.__version__, min_versions["Keras"])


# %% [markdown]
# ## Load and Display Data
#
# First, let's load the dataset and display its basic information.

# %%
# Load the dataset
diamonds = load_data()

# Display basic information
display_data_info(diamonds)



# %% [markdown]
# ## Exploratory Data Analysis (EDA)
#
# In this section, we explore the **Seaborn Diamonds dataset** to better understand the features that influence diamond pricing.  
# The goal of EDA is to identify key relationships, detect patterns, and spot potential data quality issues before modeling.
#
# The notebook guides you through:
# - Viewing the first few rows of the dataset to understand its structure and column types  
# - Checking for **missing values** and **basic statistics** using `info()` and `describe()`  
# - Visualizing key numerical features such as **carat**, **depth**, **table**, and **price**  
# - Examining categorical features like **cut**, **color**, and **clarity** using **count plots** and **box plots** to see how they relate to price  
# - Using **pairplots** or **correlation heatmaps** to highlight relationships between numerical variables  
#
# ---
#
# ### 💡 Suggestions for Further Exploration
# If you want to go a step further, you could also:
# - Investigate **outliers** in `price` or `carat` (e.g., via boxplots or z-scores)  
# - Explore **interactions between features** such as `carat × clarity` or `color × cut` and their combined effect on price  
# - Analyze whether **feature distributions differ** significantly across categories (e.g., does “Ideal” cut always correspond to higher carat weights?)  
# - Try **log-transforming the price variable** to see if it reduces skewness  
# - Examine **correlation between numerical variables** using a heatmap to decide which features may carry redundant information  
#
# By performing these additional analyses, you can gain deeper insight into the dataset and make more informed choices during feature preprocessing and model design.
#

# %%
# Analyze categorical variables
analyze_categorical_variables(diamonds)

# %%
# Analyze price distribution
analyze_price_distribution(diamonds)

# %%
# Analyze carat-price relationship
analyze_carat_price_relationship(diamonds)

# %%
# Analyze cut impact on price
analyze_cut_impact(diamonds)

# %%
# Analyze correlations
analyze_correlations(diamonds)

# %% [markdown]
# ## Remove Outliers
#
# In this step, we remove a small number of clearly invalid or extreme records (e.g., zero or unrealistic dimensions) to ensure the dataset is clean before modeling.
#
# ### 💡 Suggestions for Further Exploration
# - Explore additional outlier detection methods such as **IQR filtering** or **z-scores**  
# - Compare model performance **before and after** removing outliers  
# - Visualize potential outliers using **boxplots** or **scatter plots** (e.g., carat vs. price)  
# - Consider **transforming** rather than removing outliers (e.g., log transformation) for highly skewed features  
#

# %%
# Remove outliers
diamonds = remove_specific_outliers(diamonds)


# %% [markdown]
# ## Model Training and Evaluation
#
# In this section, you’ll train a **neural network regression model** using TensorFlow/Keras to predict diamond prices.  
# The model will learn how features like **carat**, **cut**, **color**, and **clarity** relate to price.
#
# You’ll split the data into training, validation, and test sets, define a simple feedforward network, and track progress using **loss curves** and **validation metrics**.  
# Finally, you’ll evaluate your model on the test set and visualize **predicted vs. actual prices** to see how well it performs.
#
# ### 💡 Suggestions for Further Exploration
# - Experiment with different network sizes or learning rates  
# - Compare evaluation metrics such as **MAE**, **MSE**, and **R²**  
# - Investigate how **feature scaling** affects model performance  
#

# %%

numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
categorical_features = ['cut', 'color', 'clarity']

# Prepare data for model training
X_train, X_test, y_train, y_test, preprocessor = prepare_data(diamonds,numerical_features,categorical_features)

# %%
# Train the model
model, history = train_model(X_train, y_train,patience=3,epochs=200)

# %%
# Plot training history
plot_training_history(history)

# %%
# Evaluate the model
y_pred_orig = evaluate_model(model, X_test, y_test)
residuals_orig = y_test - y_pred_orig


# %%
# Plot percentage error by price
percentage_errors, bin_centers_actual, bin_errors = plot_percentage_error_by_price(y_test, y_pred_orig)

# %% [markdown]
# ## Residual Plot
# We can see that The residuals are mostly centered around zero, indicating a reasonable fit.  
# However, the wider spread at higher prices suggests the model struggles with more expensive diamonds.
#
#

# %%
residual_plot(y_test, y_pred_orig)
