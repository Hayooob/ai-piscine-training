"""Exercise 5: compare five models on the same data.

The same pipeline is used for every model, only the final estimator changes.
Comparing the train and the test scores shows which models overfit.
"""
# imports
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# data
housing = fetch_california_housing()
X, y = housing['data'], housing['target']

# split data train test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    shuffle=True,
                                                    random_state=43)

MODELS = {
    "Linear Regression": LinearRegression(),
    "SVM": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=43),
    "Random Forest": RandomForestRegressor(random_state=43),
    "Gradient Boosting": GradientBoostingRegressor(random_state=43),
}


def make_pipeline(model):
    """The imputer and the scaler stay the same, only the model changes."""
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler()),
                     ('model', model)])


def evaluate(pipe, X, y):
    predictions = pipe.predict(X)
    return {
        "r2 score": r2_score(y, predictions),
        "MAE": mean_absolute_error(y, predictions),
        "MSE": mean_squared_error(y, predictions),
    }


if __name__ == "__main__":
    for name, model in MODELS.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)

        print("~~~")
        print(name, end="\n\n")
        for set_name, features, target in (("TRAIN", X_train, y_train),
                                           ("TEST", X_test, y_test)):
            print(set_name)
            for metric, value in evaluate(pipe, features, target).items():
                print(f"{metric}:\t {value}")
            print()
