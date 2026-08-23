import pandas as pd
import numpy as np
import json
import os
import app.preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import joblib


def find_best_threshold(y_true, probabilities, thresholds=None):
    """
    Sweeps thresholds and returns the one that maximizes F1 for the 'spam' class.
    y_true: array of string labels ('spam' / 'ham')
    probabilities: array of P(spam) from predict_proba
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    best_f1 = -1
    best_threshold = 0.5

    for t in thresholds:
        preds = np.where(probabilities >= t, "spam", "ham")
        f1 = f1_score(y_true, preds, pos_label="spam")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def train_model():
    print("Loading dataset...")
    df = pd.read_csv("../data/SMSSpamCollection.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,  # reproducible
        stratify=df["label"],  # keep class ratio consistent across split
    )

    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print("Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vectorized, y_train)

    print("Evaluating model at default 0.5 threshold...")
    default_preds = model.predict(X_test_vectorized)
    default_f1 = f1_score(y_test, default_preds, pos_label="spam")
    print(f"Default threshold (0.5) F1: {default_f1:.4f}")

    print("Searching for best decision threshold...")
    # probability of the 'spam' class - sklearn orders classes alphabetically,
    # so index 1 corresponds to 'spam' (ham=0, spam=1)
    spam_class_index = list(model.classes_).index("spam")
    probabilities = model.predict_proba(X_test_vectorized)[:, spam_class_index]

    best_threshold, best_f1 = find_best_threshold(y_test, probabilities)
    print(f"Best threshold: {best_threshold:.2f} | F1 at best threshold: {best_f1:.4f}")

    final_preds = np.where(probabilities >= best_threshold, "spam", "ham")

    accuracy = accuracy_score(y_test, final_preds)
    precision = precision_score(y_test, final_preds, pos_label="spam")
    recall = recall_score(y_test, final_preds, pos_label="spam")
    f1 = f1_score(y_test, final_preds, pos_label="spam")

    print("-" * 30)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("-" * 30)
    print("Confusion Matrix (rows=actual, cols=predicted, order=[ham, spam]):")
    print(confusion_matrix(y_test, final_preds, labels=["ham", "spam"]))
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_test, final_preds))

    print("Exporting model artifacts...")
    os.makedirs("../artifacts/models", exist_ok=True)
    joblib.dump(vectorizer, "../artifacts/models/tfidf_vectorizer.joblib")
    joblib.dump(model, "../artifacts/models/logistic_regression_model.joblib")

    with open("../artifacts/models/threshold.json", "w") as f:
        json.dump(
            {
                "threshold": float(best_threshold),
                "spam_class_index": spam_class_index,
                "model_classes": list(model.classes_),
            },
            f,
            indent=2,
        )

    print("Models and threshold saved to the artifacts directory.")


if __name__ == "__main__":
    train_model()