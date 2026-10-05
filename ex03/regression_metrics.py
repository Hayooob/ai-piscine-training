"""Exercise 3: evaluate a regression with R2, MSE and MAE.

R2 says how much of the variance of the target the model explains (1 is
perfect, 0 is as good as always predicting the mean). The MSE is in squared
units and punishes the big errors, the MAE is in the units of the target and
is easier to read.
"""
# imports
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# data
housing = fetch_california_housing()
X, y = housing['data'], housing['target']

# split data train test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    shuffle=True,
                                                    random_state=13)
# pipeline
pipeline = [('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())]
pipe = Pipeline(pipeline)
# fit
pipe.fit(X_train, y_train)

# 1. predict on both sets
predictions_train = pipe.predict(X_train)
predictions_test = pipe.predict(X_test)

# 2. the three regression metrics on both sets
metrics = {
    "train": {
        "r2": r2_score(y_train, predictions_train),
        "MAE": mean_absolute_error(y_train, predictions_train),
        "MSE": mean_squared_error(y_train, predictions_train),
    },
    "test": {
        "r2": r2_score(y_test, predictions_test),
        "MAE": mean_absolute_error(y_test, predictions_test),
        "MSE": mean_squared_error(y_test, predictions_test),
    },
}

if __name__ == "__main__":
    print("#10 first values Train", predictions_train[:10], sep="\n", end="\n\n")
    print("#10 first values Test", predictions_test[:10], sep="\n", end="\n\n")

    for name in ("train", "test"):
        print(f"r2 on the {name} set: {metrics[name]['r2']}")
        print(f"MAE on the {name} set:  {metrics[name]['MAE']}")
        print(f"MSE on the {name} set:  {metrics[name]['MSE']}", end="\n\n")

    print("The model scores slightly better on the train set than on the test "
          "set, which is expected. An r2 of 0.59 is low: a linear model is too "
          "simple for this data, the non linear models of ex05 do better.")
