import pandas as pd
from utils import entropy, gini_index
from collections import Counter
import numpy as np
from copy import deepcopy


class Node:

    def __init__(self, feature_idx, split_value, left=None, right=None, value=None, count=None):
        self.feature_idx = feature_idx  # Index of the feature to split on
        self.split_value = split_value  # Value of the feature to split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value at a leaf node
        self.count = count  # Counter values to determine this node prediction.

    def is_leaf(self):
        return self.left is None and self.right is None

    def _predict_helper(self, X):
        if self.is_leaf():
            return self.value

        if X[self.feature_idx] <= self.split_value:
            return self.left._predict_helper(X)
        else:
            return self.right._predict_helper(X)

    def predict(self, X):
        return np.apply_along_axis(func1d=self._predict_helper, axis=1, arr=X)

    def print_tree(self, level=0, prefix="Root"):
        indent = "    " * level  # Indentation based on the level of the tree
        if self.is_leaf():
            print(f"{indent}{prefix} -> Leaf: predicted label={self.value} based on {self.count} samples.")
        else:
            print(f"{indent}{prefix} -> Feature {self.feature_idx + 1} <= {self.split_value}")
            if self.left:
                self.left.print_tree(level + 1, prefix="L")
            if self.right:
                self.right.print_tree(level + 1, prefix="R")

    def __eq__(self, other):
        current_node = self.feature_idx == other.feature_idx and self.split_value == other.split_value and self.is_leaf() == other.is_leaf()
        if self.is_leaf():
            current_node &= self.value == other.value and self.count == other.count
        return current_node and self.right == other.right and self.left == other.left


class DecisionTree:

    def __init__(self, criterion="entropy"):
        self.max_depth = None
        self.min_samples_split = None
        self.min_samples_leaf = None
        self.criterion = None
        self._set_critertion(criterion=criterion)
        self.tree = None
        self._fit = False

    def _set_critertion(self, criterion: str):
        self._valid_criterion = {"entropy": entropy, "gini_index": gini_index}
        try:
            self.criterion = self._valid_criterion[criterion]
        except KeyError:
            raise ValueError(f"DecisionTree initialized with invalid criterion={criterion}.")

    def _information_gain(self, Y, Y_l, Y_r):
        N = Y.shape[0]
        return self.criterion(Y) - (Y_l.shape[0] / N) * self.criterion(Y_l) - (Y_r.shape[0] / N) * self.criterion(Y_r)

    def _split_data(self, X, Y, feature_idx, split_value):
        left_mask = X[:, feature_idx] <= split_value
        return X[left_mask, :], Y[left_mask], X[~left_mask, :], Y[~left_mask]

    def _rank_splits(self, X, Y, n_features_select, min_samples_split):
        ranks = []
        for feature_idx in range(X.shape[1]):
            feature_values = np.unique(X[:, feature_idx])
            feature_values = np.random.choice(feature_values,
                                              size=len(feature_values) if n_features_select is None else min(
                                                  len(feature_values), n_features_select), replace=False)

            for split_value in feature_values:
                X_l, Y_l, X_r, Y_r = self._split_data(X=X, Y=Y, feature_idx=feature_idx, split_value=split_value)
                if len(Y_l) < min_samples_split or len(Y_r) < min_samples_split:
                    continue
                IG = self._information_gain(Y=Y, Y_l=Y_l, Y_r=Y_r)
                ranks.append((feature_idx, split_value, IG))

            if not len(ranks):
                raise Exception(
                    f"DecisionTree.fit failed. Consider to decrease min_samples_split={min_samples_split} "
                    f"or setting n_features_select=None")

        ranks = sorted(ranks, key=lambda x: -x[-1])
        return ranks

    def _build_tree(self, X, Y, max_depth, min_samples_split, min_samples_leaf, n_features_select):
        if max_depth == -1:
            return None

        split_ranks = self._rank_splits(X=X, Y=Y, n_features_select=n_features_select,
                                        min_samples_split=min_samples_split)
        if len(np.unique(Y)) == 1:
            return Node(feature_idx=0, split_value=0, left=None, right=None, value=np.unique(Y)[0], count=len(Y))

        for split_idx, split in enumerate(split_ranks):
            feature_idx, split_value, split_IG = split
            most_common = Counter(Y).most_common(1).pop()
            X_l, Y_l, X_r, Y_r = self._split_data(X=X, Y=Y, feature_idx=feature_idx, split_value=split_value)

            if max_depth > 0 and (len(Y_l) < min_samples_split or len(Y_r) < min_samples_split):
                print("max_depth > 0 and (len(Y_l) < min_samples_split or len(Y_r) < min_samples_split)")
                continue

            if max_depth == 0 and (len(Y_l) < min_samples_leaf or len(Y_r) < min_samples_leaf):
                print("max_depth == 0 and (len(Y_l) < min_samples_leaf or len(Y_r) < min_samples_leaf)")
                continue
            try:
                left = self._build_tree(X=X_l, Y=Y_l, max_depth=max_depth - 1, min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf, n_features_select=n_features_select)
                right = self._build_tree(X=X_r, Y=Y_r, max_depth=max_depth - 1, min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf, n_features_select=n_features_select)
            except Exception as e:
                continue
            tree = Node(feature_idx=feature_idx, split_value=split_value, left=left, right=right,
                        value=most_common[0], count=most_common[1])
            return tree

        raise Exception(
            "Failed to build tree. Please try to change max_depth, min_samples_split, min_samples_leaf,"
            " n_features_select parameters.")

    def _prune_tree_helper(self, tree):
        if tree.is_leaf():
            return tree

        if tree.left.is_leaf() and tree.right.is_leaf() and tree.left.value == tree.right.value:
            return Node(tree.left.feature_idx, tree.left.split_value,
                        left=None, right=None, value=tree.left.value,
                        count=tree.left.count + tree.right.count)

        tree.left = self._prune_tree_helper(tree=tree.left)
        tree.right = self._prune_tree_helper(tree=tree.right)
        return tree

    def _prune_tree(self, tree):
        prune_tree = self._prune_tree_helper(tree=deepcopy(tree))

        while tree != prune_tree:
            new_prune_tree = self._prune_tree_helper(tree=deepcopy(prune_tree))
            tree = deepcopy(prune_tree)
            prune_tree = deepcopy(new_prune_tree)
        return prune_tree

    def fit(self, X, Y, max_depth=2, min_samples_split=2, min_samples_leaf=1, n_features_select=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.shape[0] != Y.shape[0]:
            raise ValueError("DecisionTree.fit: X.shape[0] must be equal to Y.shape[0].")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        tree = self._build_tree(X=X, Y=Y, max_depth=max_depth, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, n_features_select=n_features_select)
        # Prune Tree
        self.tree = self._prune_tree(tree=tree)
        self._fit = True

    def print_tree(self):
        self.tree.print_tree()

    def predict(self, X):
        try:
            return self.tree.predict(X=X)
        except Exception:
            raise Exception("DecisionTree.predict can be called only after calling to fit().")


if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import plot_tree
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np

    feature_names = ['feature1', 'feature2', 'feature3', 'feature4']  # Replace with actual names
    class_names = ['class1', 'class2', 'class3']

    # Load dataset
    iris = load_iris()
    X, Y = iris.data, iris.target

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    dt = DecisionTree(criterion="gini_index")
    train_errors, test_errors = [], []
    sklearn_train_errors, sklearn_test_errors = [], []
    tree_depths = list(np.arange(1, 9).astype(int))
    for i in tree_depths:
        np.random.seed(100)
        dt_sklearn = DecisionTreeClassifier(criterion="gini", max_depth=i, min_samples_split=2, min_samples_leaf=1,
                                            random_state=100)
        dt.fit(X=X_train, Y=Y_train, max_depth=i, min_samples_split=2)
        dt_sklearn.fit(X=X_train, y=Y_train)

        Y_prime_test = dt.predict(X_test)
        Y_prime_train = dt.predict(X_train)

        sklearn_Y_prime_test = dt_sklearn.predict(X_test)
        sklearn_Y_prime_train = dt_sklearn.predict(X_train)

        # Calculate test error
        test_error = np.sum(Y_prime_test != Y_test) / len(Y_test)
        train_error = np.sum(Y_prime_train != Y_train) / len(Y_train)

        sklearn_test_error = np.sum(sklearn_Y_prime_test != Y_test) / len(Y_test)
        sklearn_train_error = np.sum(sklearn_Y_prime_train != Y_train) / len(Y_train)

        train_errors.append(train_error)
        test_errors.append(test_error)
        sklearn_train_errors.append(sklearn_train_error)
        sklearn_test_errors.append(sklearn_test_error)
        print(f"Numpy Max Depth: {i}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")
        print(f"Sklearn Max Depth: {i}, Train Error: {sklearn_train_error:.4f}, Test Error: {sklearn_test_error:.4f}")
        print("=" * 100)

    plt.plot(tree_depths, train_errors, label="train error", marker='o')
    plt.plot(tree_depths, test_errors, label="test error", marker='x')
    plt.legend()
    plt.title("NumPy Decision Tree Error - Iris Dataset")
    plt.ylabel("error")
    plt.xlabel("tree-depth")
    plt.grid()
    plt.show()

    plt.plot(tree_depths, sklearn_train_errors, label="train error", marker='o')
    plt.plot(tree_depths, sklearn_test_errors, label="test error", marker='x')
    plt.legend()
    plt.title("Sklearn Decision Tree Error - Iris Dataset")
    plt.ylabel("error")
    plt.xlabel("tree-depth")
    plt.grid()
    plt.show()
