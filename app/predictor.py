from os import getenv
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(getenv("USER_MONTH_FEATURES_CSV", ROOT / "Data" / "processed" / "user_month_features.csv"))
MODEL_PATH = Path(getenv("SPEND_MODEL_PATH", Path(__file__).resolve().parent / "models" / "spend_predictor.pkl"))

_model_bundle: dict | None = None
_model_load_attempted = False


def _get_model_bundle() -> dict | None:
    """train.py로 미리 학습해둔 LightGBM 모델을 최초 1회만 로드해 재사용한다.
    모델 파일이 없으면(아직 학습 전이면) None을 반환해 휴리스틱 경로로 폴백한다."""
    global _model_bundle, _model_load_attempted
    if not _model_load_attempted:
        _model_load_attempted = True
        if MODEL_PATH.exists():
            _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


# BE 실시간 요청 스키마(FinancialRecordInput)는 balance/credit_score라는 이름을 그대로 씀.
# 학습 데이터(user_month_features.csv)는 오해를 막기 위해 이름을 정정했으므로,
# records로 들어온 값만 여기서 내부 명칭으로 맞춰준다. BE 쪽 계약은 건드리지 않는다.
_BE_FIELD_RENAME = {"balance": "card_outstanding_balance", "credit_score": "credit_limit"}


def load_history(user_id: str, records: list[dict] | None = None) -> pd.DataFrame:
    if records:
        frame = pd.DataFrame(records).rename(columns=_BE_FIELD_RENAME)
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
    """미래 데이터로 결측을 채우지 않는다(bfill 금지, train.py와 동일 원칙).
    실시간 추론은 항상 마지막 행(featured.tail(1))만 사용하므로, 이력이 짧아 앞쪽
    행의 2/3개월 평균이 비어도 그 행이 실제로 예측에 쓰이는 일은 없다."""
    result = frame.copy()
    for optional_column in ("credit_limit", "card_outstanding_balance", "edu_spend", "fin_stress"):
        if optional_column not in result:
            result[optional_column] = 0
        result[optional_column] = result[optional_column].fillna(0)
    result["spend_lag_1"] = result["spend"].shift(1)
    result["spend_2m_avg"] = result["spend"].rolling(2).mean()
    result["spend_3m_avg"] = result["spend"].rolling(3).mean()
    result["spend_growth_rate"] = result["spend"].pct_change().replace([float("inf"), float("-inf")], 0).fillna(0)
    result["bill_to_spend_ratio"] = result["bill"] / result["spend"].clip(lower=1)
    return result.fillna(0)


def predict_next_spend(user_id: str, records: list[dict] | None = None) -> dict:
    history = load_history(user_id, records)
    featured = build_features(history)

    bundle = _get_model_bundle()

    if bundle is not None and bundle.get("mode") == "two_stage":
        feature_columns = bundle["feature_columns"]
        latest_row = featured.tail(1)[feature_columns]
        will_spend = int(bundle["classifier"].predict(latest_row)[0])
        if will_spend == 0:
            prediction = 0
            importances = bundle["classifier"].feature_importances_
        else:
            prediction = int(max(0, bundle["regressor"].predict(latest_row)[0]))
            importances = bundle["regressor"].feature_importances_
        impacts = sorted(
            [
                {"feature": feature, "impact": float(value)}
                for feature, value in zip(feature_columns, importances, strict=True)
            ],
            key=lambda item: item["impact"],
            reverse=True,
        )[:3]
    elif bundle is not None:
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
        latest_row = featured.tail(1)[feature_columns]
        prediction = int(max(0, model.predict(latest_row)[0]))
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            impacts = sorted(
                [
                    {"feature": feature, "impact": float(value)}
                    for feature, value in zip(feature_columns, importances, strict=True)
                ],
                key=lambda item: item["impact"],
                reverse=True,
            )[:3]
        else:
            impacts = []
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
