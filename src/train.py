
from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)


def train_models():
    X, y = make_classification(
        n_samples=5000,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        weights=[0.90, 0.10],
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "v1": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "v2": ExtraTreesClassifier(
            n_estimators=100, random_state=43
        ),
        "v3": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

        print(
            f"{name.upper()} | "
            f"Accuracy: {accuracy_score(y_test, predictions):.4f} | "
            f"F1: {f1_score(y_test, predictions):.4f}"
        )

    print(f"\nModels saved to: {MODEL_DIR}")


if __name__ == "__main__":
    train_models()
