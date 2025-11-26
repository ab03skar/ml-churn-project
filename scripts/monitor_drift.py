import glob
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_baseline_stats(root: Path):
    path = root / "data" / "baseline_stats.json"
    with open(path, "r") as f:
        return json.load(f)


def compute_current_stats(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("userId", "churn")]
    stats = {}
    for col in feature_cols:
        s = df[col].describe()
        stats[col] = {
            "mean": float(s["mean"]),
            "std": float(s["std"]),
            "min": float(s["min"]),
            "max": float(s["max"]),
        }
    return stats


def detect_data_drift(baseline, current, threshold=0.2):
    drifted_features = []

    for col, base_stats in baseline.items():
        if col not in current:
            continue

        base_mean = base_stats["mean"]
        curr_mean = current[col]["mean"]

        denom = abs(base_mean) + 1e-8
        rel_change = abs(curr_mean - base_mean) / denom

        if rel_change > threshold:
            drifted_features.append(
                {
                    "feature": col,
                    "baseline_mean": base_mean,
                    "current_mean": curr_mean,
                    "relative_change": rel_change,
                }
            )

    return drifted_features


def load_retrain_metrics(root: Path):
    history_dir = root / "retrain_history"
    pattern = str(history_dir / "metrics_*.json")
    files = sorted(glob.glob(pattern))

    runs = []
    for path in files:
        with open(path, "r") as f:
            metrics = json.load(f)
        runs.append({"path": path, "metrics": metrics})
    return runs


def detect_concept_drift(runs, drop_threshold=0.1):
    if len(runs) < 2:
        return None

    first = runs[0]["metrics"]
    last = runs[-1]["metrics"]

    base_acc = first["accuracy"]
    last_acc = last["accuracy"]

    base_f1_1 = first["1"]["f1-score"]
    last_f1_1 = last["1"]["f1-score"]

    acc_drop = base_acc - last_acc
    f1_drop = base_f1_1 - last_f1_1

    alert = {
        "accuracy_drop": acc_drop,
        "f1_class1_drop": f1_drop,
        "has_concept_drift": (acc_drop > drop_threshold) or (f1_drop > drop_threshold),
        "baseline_accuracy": base_acc,
        "latest_accuracy": last_acc,
        "baseline_f1_class1": base_f1_1,
        "latest_f1_class1": last_f1_1,
    }

    return alert


def main():
    root = Path(__file__).parents[1]

    df = pd.read_csv(root / "data" / "training_data.csv")

    baseline_stats = load_baseline_stats(root)
    current_stats = compute_current_stats(df)

    data_drift = detect_data_drift(baseline_stats, current_stats)

    runs = load_retrain_metrics(root)
    concept_drift = detect_concept_drift(runs)

    report = {
        "timestamp": datetime.now().isoformat(),
        "num_rows": int(df.shape[0]),
        "num_features": len(current_stats),
        "data_drift_features": data_drift,
        "concept_drift": concept_drift,
    }

    output_dir = root / "monitoring_reports"
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"drift_report_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Drift report saved to {out_path}")
    print("\n=== Data Drift ===")
    if data_drift:
        for item in data_drift:
            print(
                f"- {item['feature']}: baseline_mean={item['baseline_mean']:.3f}, "
                f"current_mean={item['current_mean']:.3f}, "
                f"relative_change={item['relative_change']:.2%}"
            )
    else:
        print("No significant data drift detected.")

    print("\n=== Concept Drift ===")
    if concept_drift is None:
        print("Not enough runs in retrain_history to evaluate concept drift.")
    else:
        print(
            f"Baseline accuracy: {concept_drift['baseline_accuracy']:.3f}, "
            f"Latest accuracy: {concept_drift['latest_accuracy']:.3f}, "
            f"Accuracy drop: {concept_drift['accuracy_drop']:.3f}"
        )
        print(
            f"Baseline F1 (class 1): {concept_drift['baseline_f1_class1']:.3f}, "
            f"Latest F1 (class 1): {concept_drift['latest_f1_class1']:.3f}, "
            f"F1 drop: {concept_drift['f1_class1_drop']:.3f}"
        )
        if concept_drift["has_concept_drift"]:
            print("Concept drift suspected (performance drop above threshold).")
        else:
            print("No significant concept drift detected.")


if __name__ == "__main__":
    main()
