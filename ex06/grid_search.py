"""Exercise 6: search the best hyperparameters with a GridSearchCV.

The hyperparameters are the settings of the model, chosen before it learns.
The grid search trains one model per combination and keeps the best one. The
`cv` given here is a single split: the first 18576 rows are the train set,
the last 2064 the test set.
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

housing = fetch_california_housing()
X, y = housing['data'], housing['target']

new_point = np.array([[3.2031, 52., 5.47761194, 1.07960199, 910., 2.26368159,
                       37.85, -122.26]])

parameters = {
    'n_estimators': [10, 50, 75],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [10, 20, 30],
}


def select_model_verbose(gs):
    """The best model, the hyperparameters that gave it and its score."""
    return gs.best_estimator_, gs.best_params_, gs.best_score_


if __name__ == "__main__":
    # 1. one model per combination, trained in parallel on all the CPUs
    rf = RandomForestRegressor(random_state=43)
    gridsearch = GridSearchCV(rf,
                              parameters,
                              cv=[(np.arange(18576), np.arange(18576, 20640))],
                              n_jobs=-1)
    gridsearch.fit(X, y)

    # 2. the best of the 27 combinations
    model, best_params, best_score = select_model_verbose(gridsearch)
    print("best parameters:", best_params)
    print("best score (r2):", best_score, end="\n\n")

    # 3. a prediction on a new house
    print("prediction on the new point:", model.predict(new_point), end="\n\n")

    print("The score stays below the Gradient Boosting of ex05: the grid "
          "limits the depth of the trees, which reduces the overfitting but "
          "also what the model can learn. A wider grid would do better.")
