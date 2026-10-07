"""
Customer Support Ticket Classification
======================================
Train a Logistic Regression model (with TF-IDF features) to predict
Ticket Type from Ticket Description using the Kaggle Customer Support
Ticket Dataset.

This script:
  1. Loads and inspects the CSV
  2. Cleans / preprocesses the text
  3. Splits data with a stratified train/test split
  4. Vectorizes text with TF-IDF
  5. Trains Logistic Regression
  6. Evaluates with accuracy, precision, recall, F1, confusion matrix,
     and a full classification report
  7. Tests a few new sample messages
  8. Saves the vectorizer and model with joblib into the model/ folder
"""

from __future__ import annotations

import os
import re
import string

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_support_tickets.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
)
MLFLOW_ARTIFACT_LOCATION = os.getenv(
    "MLFLOW_ARTIFACT_LOCATION", f"file://{os.path.join(BASE_DIR, 'mlartifacts')}"
)
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "customer-support-ticket-classifier"
)
MLFLOW_REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "customer-support-ticket-classifier"
)

# Column names we will use (confirmed from the real Kaggle CSV)
TEXT_COLUMN = "Ticket Description"
LABEL_COLUMN = "Ticket Type"

# Columns that must NEVER be used as input features
FORBIDDEN_INPUT_COLUMNS = {"Ticket Type", "Resolution"}


# ---------------------------------------------------------------------------
# 1. Load and inspect the dataset
# ---------------------------------------------------------------------------
def load_and_inspect(csv_path: str) -> pd.DataFrame:
    """Load the CSV and print a clear summary of its structure."""
    print("=" * 70)
    print("STEP 1: Load and inspect the dataset")
    print("=" * 70)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Download the Kaggle Customer Support Ticket Dataset and place "
            "customer_support_tickets.csv inside the data/ folder."
        )

    df = pd.read_csv(csv_path)

    print(f"\nFile loaded from: {csv_path}")
    print(f"Number of rows   : {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nExact column names:")
    for i, col in enumerate(df.columns, start=1):
        print(f"  {i:2d}. {col}")

    print("\nFirst 3 rows (selected columns):")
    preview_cols = [c for c in [TEXT_COLUMN, LABEL_COLUMN, "Ticket Subject"] if c in df.columns]
    print(df[preview_cols].head(3).to_string(index=False))

    print("\nTicket Type class distribution:")
    print(df[LABEL_COLUMN].value_counts(dropna=False).to_string())

    print("\nMissing values in key columns:")
    print(df[[TEXT_COLUMN, LABEL_COLUMN]].isna().sum().to_string())

    print("\nInput feature column :", TEXT_COLUMN)
    print("Target label column  :", LABEL_COLUMN)
    print("Forbidden as inputs  :", sorted(FORBIDDEN_INPUT_COLUMNS))

    return df


# ---------------------------------------------------------------------------
# 2. Clean and preprocess text
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Basic, beginner-friendly text cleaning for ticket descriptions.

    Steps:
      - Convert to lowercase
      - Replace the dataset placeholder {product_purchased} with a plain word
      - Remove URLs and email addresses
      - Keep letters, numbers, and spaces only
      - Collapse repeated whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("{product_purchased}", "product")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    allowed = string.ascii_lowercase + string.digits + " "
    text = "".join(ch if ch in allowed else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_messages(messages) -> list[str]:
    """Clean a batch of raw messages for use inside the registered pipeline."""
    if isinstance(messages, pd.DataFrame):
        if TEXT_COLUMN in messages.columns:
            messages = messages[TEXT_COLUMN]
        else:
            messages = messages.iloc[:, 0]
    elif isinstance(messages, pd.Series):
        messages = messages.tolist()

    return [clean_text(message) for message in messages]


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Select the text/label columns, drop empty rows, and clean the text."""
    print("\n" + "=" * 70)
    print("STEP 2: Clean and preprocess the text data")
    print("=" * 70)

    data = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

    before = len(data)
    data = data.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    after_na = len(data)

    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str).map(clean_text)
    data = data[data[TEXT_COLUMN].str.len() > 0]
    after_clean = len(data)

    print(f"Rows before cleaning     : {before}")
    print(f"Rows after dropping NA   : {after_na}")
    print(f"Rows after text cleaning : {after_clean}")
    print(f"Rows removed             : {before - after_clean}")

    print("\nSample cleaned description:")
    print(" ", data[TEXT_COLUMN].iloc[0][:200], "...")

    X = data[TEXT_COLUMN]
    y = data[LABEL_COLUMN]
    return X, y


# ---------------------------------------------------------------------------
# 3. Stratified train / test split
# ---------------------------------------------------------------------------
def split_data(X: pd.Series, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split with stratification so each Ticket Type stays balanced."""
    print("\n" + "=" * 70)
    print("STEP 3: Stratified train / test split")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Test size        : {test_size:.0%}")
    print(f"Stratified on    : {LABEL_COLUMN}")

    print("\nTrain class distribution:")
    print(y_train.value_counts().to_string())
    print("\nTest class distribution:")
    print(y_test.value_counts().to_string())

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. TF-IDF vectorization
# ---------------------------------------------------------------------------
def build_tfidf_features(X_train: pd.Series, X_test: pd.Series):
    """Convert ticket descriptions into numerical TF-IDF features."""
    print("\n" + "=" * 70)
    print("STEP 4: TF-IDF vectorization")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.95,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Vocabulary size       : {len(vectorizer.vocabulary_)}")
    print(f"Train feature matrix  : {X_train_tfidf.shape}")
    print(f"Test feature matrix   : {X_test_tfidf.shape}")

    return vectorizer, X_train_tfidf, X_test_tfidf


# ---------------------------------------------------------------------------
# 5. Train Logistic Regression
# ---------------------------------------------------------------------------
def train_model(X_train_tfidf, y_train) -> LogisticRegression:
    """Train a multiclass Logistic Regression classifier."""
    print("\n" + "=" * 70)
    print("STEP 5: Train Logistic Regression")
    print("=" * 70)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)
    print("Training complete.")
    print(f"Classes learned: {list(model.classes_)}")
    return model


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model: LogisticRegression, X_test_tfidf, y_test) -> dict[str, float]:
    """Print accuracy, precision, recall, F1, confusion matrix, and report."""
    print("\n" + "=" * 70)
    print("STEP 6: Evaluate the model")
    print("=" * 70)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    n_classes = len(model.classes_)
    chance_baseline = 1.0 / n_classes

    print("\n----- Summary Metrics -----")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}  (weighted)")
    print(f"Recall    : {recall:.4f}  (weighted)")
    print(f"F1-score  : {f1:.4f}  (weighted)")
    print(f"\nChance baseline for {n_classes} classes: {chance_baseline:.4f}")
    print(
        "Note: On this Kaggle dataset, Ticket Description and Ticket Type are only"
        " weakly related, so scores near chance are common when using description alone."
    )

    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("\n----- Confusion Matrix -----")
    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {l}" for l in labels],
        columns=[f"Pred: {l}" for l in labels],
    )
    print(cm_df.to_string())

    print("\n----- Classification Report -----")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    return {
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
    }


# ---------------------------------------------------------------------------
# 7. Predict on new sample messages
# ---------------------------------------------------------------------------
def predict_new_messages(model: LogisticRegression, vectorizer: TfidfVectorizer) -> None:
    """Run a few realistic support messages through the trained pipeline."""
    print("\n" + "=" * 70)
    print("STEP 7: Test the trained model on new messages")
    print("=" * 70)

    sample_messages = [
        "My laptop keeps restarting unexpectedly and the screen flickers during use.",
        "I was charged twice for my monthly subscription. Please check my invoice.",
        "I would like to cancel my account and stop future renewals.",
        "Can you tell me the difference between the standard and premium plans?",
        "I want a refund for the headphones I returned last week. The package was damaged.",
    ]

    cleaned = [clean_text(msg) for msg in sample_messages]
    features = vectorizer.transform(cleaned)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    class_names = list(model.classes_)

    for i, (message, pred) in enumerate(zip(sample_messages, predictions), start=1):
        print(f"\nMessage {i}:")
        print(f"  Text      : {message}")
        print(f"  Predicted : {pred}")
        top_idx = probabilities[i - 1].argmax()
        conf = probabilities[i - 1][top_idx]
        print(f"  Confidence: {conf:.2%} ({class_names[top_idx]})")


# ---------------------------------------------------------------------------
# 8. Save artifacts for a future web application
# ---------------------------------------------------------------------------
def save_artifacts(vectorizer: TfidfVectorizer, model: LogisticRegression) -> None:
    """Save the TF-IDF vectorizer and Logistic Regression model with joblib."""
    print("\n" + "=" * 70)
    print("STEP 8: Save model artifacts with joblib")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved TF-IDF vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved Logistic Regression model -> {MODEL_PATH}")
    print("These files can be loaded later by a web application.")


# ---------------------------------------------------------------------------
# 9. Log and register the complete inference pipeline with MLflow
# ---------------------------------------------------------------------------
def register_with_mlflow(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    metrics: dict[str, float],
    training_samples: int,
) -> None:
    """Log the full raw-text inference pipeline and register its model version."""
    print("\n" + "=" * 70)
    print("STEP 9: Register the model with MLflow")
    print("=" * 70)

    inference_pipeline = Pipeline(
        [
            ("clean_text", FunctionTransformer(clean_messages, validate=False)),
            ("tfidf", vectorizer),
            ("classifier", model),
        ]
    )
    input_example = pd.DataFrame(
        {
            TEXT_COLUMN: [
                "I was charged twice for my subscription.",
                "My laptop screen keeps flickering.",
            ]
        }
    )
    signature = infer_signature(
        input_example,
        inference_pipeline.predict(input_example),
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="logistic-regression-tfidf") as run:
        mlflow.log_params(
            {
                "classifier": "LogisticRegression",
                "vectorizer": "TfidfVectorizer",
                "max_features": vectorizer.max_features,
                "ngram_range": str(vectorizer.ngram_range),
                "training_samples": training_samples,
                "classes": len(model.classes_),
            }
        )
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model=inference_pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=input_example,
        )

        print(f"Tracking URI          : {MLFLOW_TRACKING_URI}")
        print(f"Experiment            : {MLFLOW_EXPERIMENT_NAME}")
        print(f"Run ID                : {run.info.run_id}")
        print(f"Registered model      : {MLFLOW_REGISTERED_MODEL_NAME}")
        print(f"Logged model URI      : {model_info.model_uri}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_and_inspect(DATA_PATH)
    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    vectorizer, X_train_tfidf, X_test_tfidf = build_tfidf_features(X_train, X_test)
    model = train_model(X_train_tfidf, y_train)
    metrics = evaluate_model(model, X_test_tfidf, y_test)
    predict_new_messages(model, vectorizer)
    save_artifacts(vectorizer, model)
    register_with_mlflow(vectorizer, model, metrics, len(X_train))

    print("\n" + "=" * 70)
    print("Done! Training pipeline finished successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
