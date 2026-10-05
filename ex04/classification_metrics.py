"""Exercise 4: evaluate a classification with F1, accuracy, precision, recall
and ROC AUC.

Accuracy alone is misleading on an imbalanced data set: always predicting the
majority class already scores well. Precision answers "among the positives I
predicted, how many were right?", recall "among the real positives, how many
did I find?", and the F1 combines both. The ROC AUC is computed on the
probabilities, not on the classes, and says how well the model separates the
two classes whatever the threshold.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTPUT = Path(__file__).resolve().parent / "roc_auc.png"

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=43)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
classifier = LogisticRegression()
classifier.fit(X_train_scaled, y_train)

# the test set is scaled with the scaler fitted on the train set
X_test_scaled = scaler.transform(X_test)

# 1. predict on both sets
predictions_train = classifier.predict(X_train_scaled)
predictions_test = classifier.predict(X_test_scaled)

# the AUC needs the probabilities of the positive class, not the classes
probabilities_train = classifier.predict_proba(X_train_scaled)[:, 1]
probabilities_test = classifier.predict_proba(X_test_scaled)[:, 1]


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "F1": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
    }


# 2. the metrics on both sets
metrics_train = compute_metrics(y_train, predictions_train, probabilities_train)
metrics_test = compute_metrics(y_test, predictions_test, probabilities_test)
confusion_test = confusion_matrix(y_test, predictions_test)

if __name__ == "__main__":
    print("# 10 first values Train", predictions_train[:10], sep="\n", end="\n\n")
    print("# 10 first values Test", predictions_test[:10], sep="\n", end="\n\n")

    for name, metrics in (("train", metrics_train), ("test", metrics_test)):
        for metric, value in metrics.items():
            print(f"{metric} on the {name} set:  {value}")
        print()

    print("confusion matrix on the test set:", confusion_test, sep="\n", end="\n\n")

    # 3. the ROC curve: the true positive rate against the false positive rate
    fpr, tpr, thresholds = roc_curve(y_test, probabilities_test)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="darkorange",
            label=f"ROC curve (area = {metrics_test['ROC_AUC']:.2f})")
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--", label="random model")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right")
    fig.savefig(OUTPUT)

    print("A ROC AUC of 99% is not usual: this data set is easy to classify. "
          "On real data, such a score should make you look for a data leak.")
    plt.show()
