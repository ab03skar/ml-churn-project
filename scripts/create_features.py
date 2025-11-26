import pandas as pd
from pathlib import Path


def main():
    root = Path(__file__).parents[1]
    df = pd.read_json(root / "data" / "customer_churn_mini.json", lines=True)

    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["registration"] = pd.to_datetime(df["registration"], unit="ms")

    grouped = df.groupby("userId")

    features = pd.DataFrame(
        {
            "userId": grouped.size().index,
            "num_events": grouped.size().values,
            "num_sessions": grouped["sessionId"].nunique().values,
            "num_songs": grouped.apply(
                lambda x: (x["page"] == "NextSong").sum()
            ).values,
            "num_ads": grouped.apply(
                lambda x: (x["page"] == "Roll Advert").sum()
            ).values,
            "thumbs_up": grouped.apply(
                lambda x: (x["page"] == "Thumbs Up").sum()
            ).values,
            "thumbs_down": grouped.apply(
                lambda x: (x["page"] == "Thumbs Down").sum()
            ).values,
            "help_events": grouped.apply(lambda x: (x["page"] == "Help").sum()).values,
            "error_events": grouped.apply(
                lambda x: (x["page"] == "Error").sum()
            ).values,
            "downgrade_events": grouped.apply(
                lambda x: (x["page"] == "Downgrade").sum()
            ).values,
            "distinct_artists": grouped["artist"].nunique().values,
            "distinct_songs": grouped["song"].nunique().values,
            "is_paid": grouped["level"].apply(lambda x: int("paid" in x.values)).values,
            "days_on_platform": grouped.apply(
                lambda x: (x["ts"].max() - x["registration"].min()).days
            ).values,
            "activity_span_days": grouped.apply(
                lambda x: (x["ts"].max() - x["ts"].min()).days
            ).values,
        }
    )

    features["songs_per_session"] = features["num_songs"] / features[
        "num_sessions"
    ].replace(0, 1)
    features["ads_per_session"] = features["num_ads"] / features[
        "num_sessions"
    ].replace(0, 1)
    features["thumbs_up_ratio"] = features["thumbs_up"] / (
        features["thumbs_up"] + features["thumbs_down"] + 1
    )

    features.to_csv(root / "data" / "user_features.csv", index=False)


if __name__ == "__main__":
    main()
