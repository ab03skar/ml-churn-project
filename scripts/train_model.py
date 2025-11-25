import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def train_and_eval():
    root = Path(__file__).parents[1]

    df = pd.read_csv(root / "data" / "training_data.csv")

    X = df.drop(columns=["userId", "churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    report_dict = classification_report(y_test, preds, output_dict=True)

    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    dump(model, models_dir / "churn_model.joblib")

    with open(models_dir / "last_metrics.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    df_out = X_test.copy()
    df_out["actual"] = y_test.values
    df_out["predicted"] = preds
    df_out.to_csv(root / "data" / "model_predictions.csv", index=False)

    errors = X_test.copy()
    errors["true"] = y_test.values
    errors["pred"] = preds

    false_negative = errors[(errors["true"] == 1) & (errors["pred"] == 0)]
    false_positive = errors[(errors["true"] == 0) & (errors["pred"] == 1)]

    false_negative.to_csv(root / "data" / "false_negative.csv", index=False)
    false_positive.to_csv(root / "data" / "false_positive.csv", index=False)

    return report_dict


def main():
    train_and_eval()


if __name__ == "__main__":
    main()
