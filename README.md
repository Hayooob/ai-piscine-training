# Training

Solutions for the Training day of the AI piscine: the metrics used to
evaluate a regression and a classification, an overview of five machine
learning models, and the search of the best hyperparameters with a grid
search.

## Structure

```
.
├── ex00/
│   ├── requirements.txt
│   └── Notebook_ex00.ipynb
├── ex01/mse.py
├── ex02/accuracy.py
├── ex03/regression_metrics.py
├── ex04/classification_metrics.py   -> roc_auc.png
├── ex05/models.py
└── ex06/grid_search.py
```

## Virtual environment (ex00)

```bash
python -m venv env/ex00
source env/ex00/Scripts/activate      # Windows (Git Bash); Linux/macOS: source env/ex00/bin/activate
pip install -r ex00/requirements.txt
python --version
python -c "import jupyter, numpy, pandas, matplotlib, sklearn"
```

Tested with Python 3.13, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.3 and
matplotlib 3.11.2.

## Data

No file to download: the breast cancer data set ships with Scikit-learn, and
`fetch_california_housing()` downloads the housing data set once and caches
it in `~/scikit_learn_data`.

## Running an exercise

```bash
python ex04/classification_metrics.py
```

ex05 takes a few minutes (the SVM is slow on 18576 rows), ex06 around 20
seconds.

## Which metric for which problem

- **Regression** — `r2` says how much of the variance of the target the
  model explains (1 is perfect, 0 is as good as always predicting the mean).
  The `MSE` is in squared units and punishes the big errors; the `MAE` is in
  the units of the target and is easier to read.
- **Classification** — the `accuracy` alone lies on an imbalanced data set:
  predicting always the majority class already scores well. The `precision`
  answers "among the positives I predicted, how many were right?", the
  `recall` "among the real positives, how many did I find?", and the `F1`
  combines both. The `ROC AUC` is computed on the probabilities and measures
  how well the model separates the classes whatever the threshold.

## Results

- **ex01 / ex02** — MSE `2.25`, accuracy `0.5714285714285714`.
- **ex03** — a linear regression on the housing data: r2 `0.608` on the train
  set and `0.590` on the test set. The gap is small (no overfitting), but the
  score is low: the model is too simple for this data.
- **ex04** — F1 `0.980`, accuracy `0.974`, recall `0.987`, precision `0.974`
  and ROC AUC `0.986` on the test set, with a confusion matrix of
  `[[37, 2], [1, 74]]`.
- **ex05** — on the test set: Linear Regression `0.613`, SVM `0.730`,
  Decision Tree `0.635`, **Random Forest `0.812`**, Gradient Boosting
  `0.790`.
- **ex06** — best parameters `max_depth=7, min_samples_leaf=10,
  n_estimators=75`, score `0.627`, and `2.656` predicted for the new point.

## What the numbers say

- The **Decision Tree** scores a perfect r2 of 1.0 on the train set and only
  0.63 on the test set: it learns the training data by heart and cannot
  generalize. This is overfitting in its purest form, and the reason why the
  algorithm is rarely used alone.
- The **Random Forest** and the **Gradient Boosting** correct that by
  combining many trees, and reach the best scores here. The Random Forest
  still overfits a bit (0.974 on train against 0.812 on test) because its
  `max_depth` is `None` by default.
- The **grid search** of ex06 returns a lower score (0.627) than the
  Gradient Boosting of ex05. Limiting the depth of the trees reduces the
  overfitting but also what the model can learn; a wider grid would do
  better. Tomorrow's cross validation will tell whether such a result is
  stable.
- The **99% ROC AUC** of ex04 is not usual. This data set is easy to
  classify; on real data such a score should make you look for a data leak.

## Note

The values match the audit, except the Decision Tree of ex05 on the test set
(r2 `0.635` here against `0.623` in the audit). A tree splits on ties in an
order that depends on the version of Scikit-learn, which changes a few
leaves. The train score (1.0) and the conclusion are the same.
