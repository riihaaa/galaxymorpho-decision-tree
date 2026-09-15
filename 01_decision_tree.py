"""
01_decision_tree.py — Decision Tree baseline, run independently.

Gauthier et al. (Paper 1) included this as one of their 5 classifiers
(training accuracy 0.91, CV accuracy 0.53 on their 5-class problem --
the big gap between those two numbers is a classic decision tree
overfitting signature, worth watching for in your own results too).

Usage: python 01_decision_tree.py
Requires: gz2_labeled.csv (from 00_data_prep.py) and common.py in the same folder.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from common import get_train_test_pca_features, report_and_save, save_confusion_matrix, RANDOM_STATE, CLASSES

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_pca_features()

    # Small grid search over depth/split params -- an untuned decision tree
    # will happily overfit to ~100% training accuracy and generalize badly,
    # which is exactly what Gauthier et al. saw (0.91 train vs 0.53 CV).
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 5, 10],
        },
        cv=3, n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print("Best Decision Tree params:", grid.best_params_)

    preds = grid.best_estimator_.predict(X_test)
    report_and_save("DecisionTree", y_test, preds)
    save_confusion_matrix("DecisionTree", y_test, preds, classes=CLASSES)