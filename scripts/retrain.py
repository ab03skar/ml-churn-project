import json
from datetime import datetime
from pathlib import Path

from create_labels import main as create_labels_main
from create_features import main as create_features_main
from training_data import main as create_training_data_main
from train_model import train_and_eval


def main():
    root = Path(__file__).parents[1]

    print("Step 1/4: updating labels...")
    create_labels_main()

    print("Step 2/4: updating user features...")
    create_features_main()

    print("Step 3/4: building training_data.csv...")
    create_training_data_main()

    print("Step 4/4: training model...")
    metrics = train_and_eval()

    history_dir = root / "retrain_history"
    history_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = history_dir / f"metrics_{timestamp}.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Retraining finished. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
