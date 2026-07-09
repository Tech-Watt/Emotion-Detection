import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.common import DATASET_PATH, MODEL_PATH, ensure_directories
except ImportError:  # pragma: no cover - fallback for direct execution
    from common import DATASET_PATH, MODEL_PATH, ensure_directories


def train_model(dataset_path: str | None = None, model_path: str | None = None) -> None:
    ensure_directories()
    dataset_path = Path(dataset_path or DATASET_PATH)
    model_path = Path(model_path or MODEL_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = df.dropna()

    features = df.drop("Class", axis=1)
    labels = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print(classification_report(y_test, predictions))

    with model_path.open("wb") as handle:
        pickle.dump(pipeline, handle)

    print(f"Model saved to {model_path}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
