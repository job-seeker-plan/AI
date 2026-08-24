from os import getenv
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression


load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(getenv("USER_MONTH_FEATURES_CSV", ROOT / "Data" / "processed" / "user_month_features.csv"))


def load_history(user_id: str, records: list[dict] | None = None) -> pd.DataFrame:
    if records:
        frame = pd.DataFrame(records)
        user_frame = frame[frame["user_id"] == user_id].copy()
        if user_frame.empty:
            raise ValueError(f"No financial history for user_id={user_id}")
        user_frame["month"] = pd.to_datetime(user_frame["month"])
        return user_frame.sort_values("month")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Real feature file is required: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    user_frame = frame[frame["user_id"] == user_id].copy()
    if user_frame.empty:
        raise ValueError(f"No financial history for user_id={user_id}")
    user_frame["month"] = pd.to_datetime(user_frame["month"])
    return user_frame.sort_values("month")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "credit_score" not in result:
        result["credit_score"] = 0
    result["credit_score"] = result["credit_score"].fillna(0)
    result["spend_lag_1"] = result["spend"].shift(1)
    result["spend_2m_avg"] = result["spend"].rolling(2).mean()
    result["spend_3m_avg"] = result["spend"].rolling(3).mean()
    result["spend_growth_rate"] = result["spend"].pct_change().fillna(0)
    result["bill_to_spend_ratio"] = result["bill"] / result["spend"].clip(lower=1)
    return result.bfill().fillna(0)


def predict_next_spend(user_id: str, records: list[dict] | None = None) -> dict:
    history = load_history(user_id, records)
    featured = build_features(history)
    feature_columns = ["spend", "bill", "balance", "credit_score", "spend_lag_1", "spend_2m_avg", "spend_3m_avg", "spend_growth_rate", "bill_to_spend_ratio"]

    training = featured.copy()
    training["target"] = training["spend"].shift(-1)
    training = training.dropna(subset=["target"])

    if len(training) >= 5:
        model = LinearRegression()
        model.fit(training[feature_columns], training["target"])
        prediction = int(max(0, model.predict(featured.tail(1)[feature_columns])[0]))
        impacts = sorted(
            [
                {"feature": feature, "impact": abs(float(coef))}
                for feature, coef in zip(feature_columns, model.coef_, strict=True)
            ],
            key=lambda item: item["impact"],
            reverse=True,
        )[:3]
    elif len(featured) >= 2:
        recent_average_window = min(3, len(featured))
        recent_average_value = featured["spend"].tail(recent_average_window).mean()
        last_month_spend_value = featured["spend"].iloc[-1]
        recent_growth = featured["spend_growth_rate"].tail(recent_average_window).mean()
        prediction = int(max(0, (recent_average_value * 0.7) + (last_month_spend_value * 0.3)) * (1 + max(-0.1, min(0.1, recent_growth))))
        impacts = []
    else:
        prediction = int(featured["spend"].iloc[-1])
        impacts = [{"feature": "last_month_spend", "impact": 1.0}]

    recent_average = int(featured["spend"].tail(3).mean())
    last_month_spend = int(featured["spend"].iloc[-1])
    if abs(prediction - recent_average) > recent_average * 0.25:
        prediction = int((recent_average * 0.7) + (last_month_spend * 0.3))

    return {
        "user_id": user_id,
        "predicted_next_spend": prediction,
        "recent_average_spend": recent_average,
        "baseline_last_month_spend": last_month_spend,
        "feature_importance": impacts,
    }
