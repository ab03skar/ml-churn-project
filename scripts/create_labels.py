import pandas as pd
from pathlib import Path

def main():
    
    DATA_PATH = Path(__file__).parents[1] / "data" / "customer_churn_mini.json"

    df = pd.read_json(DATA_PATH, lines=True)

    churn_users = df[df["page"] == "Cancellation Confirmation"]["userId"].unique()

    labels = pd.DataFrame({
        "userId": df["userId"].unique()
    })

    labels["churn"] = labels["userId"].isin(churn_users).astype(int)

    OUTPUT_PATH = Path(__file__).parents[1] / "data" / "user_labels.csv"
    labels.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
