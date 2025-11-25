import pandas as pd
from pathlib import Path


def main():
    root = Path(__file__).parents[1]

    features = pd.read_csv(root / "data" / "user_features.csv")
    labels = pd.read_csv(root / "data" / "user_labels.csv")

    df = features.merge(labels, on="userId", how="left")

    df.to_csv(root / "data" / "training_data.csv", index=False)


if __name__ == "__main__":
    main()
