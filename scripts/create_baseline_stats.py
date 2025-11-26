import json
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).parents[1]

    df = pd.read_csv(root / "data" / "training_data.csv")

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

    output_path = root / "data" / "baseline_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Baseline stats saved to {output_path}")


if __name__ == "__main__":
    main()
