from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from common import get_train_test_pca_features, report_and_save, save_confusion_matrix, RANDOM_STATE, CLASSES

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_pca_features()

    
    X_sub = X_train[:20000]
    y_sub = y_train[:20000]

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 5, 10],
        },
        cv=3, 
        n_jobs=4,  
    )
    print("Searching best hyper-parameters...")
    grid.fit(X_sub, y_sub)
    print("Best Decision Tree params:", grid.best_params_)

    # Retrain best model on FULL training set
    best_dt = DecisionTreeClassifier(
        **grid.best_params_, 
        random_state=RANDOM_STATE, 
        class_weight="balanced"
    )
    best_dt.fit(X_train, y_train)

    preds = best_dt.predict(X_test)
    report_and_save("DecisionTree", y_test, preds)
    save_confusion_matrix("DecisionTree", y_test, preds, classes=CLASSES)
