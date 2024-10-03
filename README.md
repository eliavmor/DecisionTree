# Decision Tree Classifier

This repository contains a custom implementation of a Decision Tree Classifier in Python. The classifier can be used for classification tasks and supports different splitting criteria, such as entropy and Gini index. The implementation includes features for tree pruning to enhance the model's generalization performance.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Functions](#functions)
- [License](#license)

## Features

- Customizable splitting criteria (`entropy` or `gini index`).
- Supports pruning of the decision tree to prevent overfitting.
- Predictive performance evaluation on training and test datasets.
- Visualization of training and test errors based on tree depth.

## Installation

To run this implementation, ensure you have the following Python packages installed:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

1. **Import the necessary modules:**

   ```python
   import pandas as pd
   from decision_tree import DecisionTree  # Adjust the import path as necessary
   ```

2. **Prepare your dataset:**

   Ensure your data is in the form of a Pandas DataFrame or NumPy array. The feature matrix `X` should be 2D, and the target vector `Y` should be 1D.

3. **Instantiate and fit the Decision Tree:**

   ```python
   dt = DecisionTree(criterion="entropy")  # or "gini_index"
   dt.fit(X, Y, max_depth=5, min_samples_split=2, min_samples_leaf=1)
   ```

4. **Make predictions:**

   ```python
   predictions = dt.predict(X_test)
   ```

## Example

The following example demonstrates how to use the Decision Tree classifier with the Iris dataset:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X, Y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the Decision Tree classifier
dt = DecisionTree(criterion="entropy")
dt.fit(X_train, Y_train, max_depth=5)

# Make predictions and evaluate the model
predictions = dt.predict(X_test)
```

## Functions

### `fit(X, Y, max_depth=2, min_samples_split=2, min_samples_leaf=1, n_features_select=None)`

Fits the decision tree model to the training data.

- `X`: Feature matrix.
- `Y`: Target vector.
- `max_depth`: Maximum depth of the tree.
- `min_samples_split`: Minimum number of samples required to split an internal node.
- `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
- `n_features_select`: Number of features to consider when looking for the best split.

### `predict(X)`

Predicts the class labels for the input data `X`.

- `X`: Feature matrix for which predictions are made.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
