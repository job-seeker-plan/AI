"""
전체 후보군 데이터를 모아서 다음달 지출 예측 모델을 학습한다.

v3 (2026-08-26) 변경점 — 리뷰에서 지적된 시간 누수 버그 수정:
- v2까지는 각 유저의 초반 달(7,8월)에서 spend_2m_avg/spend_3m_avg가 계산 안 되는 결측을
  bfill()로 채웠는데, 이게 미래 달(예: 9월) 값을 과거(7월) 자리에 끌어오는 구조였다.
  7월 행의 target은 8월 지출인데, bfill로 채워진 7월의 spend_3m_avg 안에 8월 실제값이
  섞여 들어가 있어 정답을 일부 미리 알고 맞추는 꼴이었다. 즉 여태 나온 MAE/RMSE는
  실제보다 좋게 나온 값이었다.
- 수정: bfill 제거. spend_lag_1/spend_2m_avg/spend_3m_avg가 전부 과거 데이터만으로
  안전하게 계산되는 시점(유저별 9월 이후, 즉 9월->10월/10월->11월/11월->12월 3개 시점)만
  학습·평가에 사용한다. 유저 15.7만명 x 안전한 시점 3개 = 약 47만 행.
- 학습 결과를 train_metrics.json으로 저장해 재현 가능하게 함.
- 피처명도 credit_limit / card_outstanding_balance로 정정 반영.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
import lightgbm as lgb
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Data" / "processed" / "user_month_features.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "spend_predictor.pkl"
METRICS_PATH = Path(__file__).resolve().parent / "models" / "train_metrics.json"

FEATURE_COLUMNS = [
    "spend", "bill", "card_outstanding_balance", "credit_limit",
    "spend_lag_1", "spend_2m_avg", "spend_3m_avg",
    "spend_growth_rate", "bill_to_spend_ratio",
    "edu_spend", "fin_stress",
]

LAG_FEATURE_COLS = ["spend_lag_1", "spend_2m_avg", "spend_3m_avg", "spend_growth_rate", "bill_to_spend_ratio"]


def build_features(group: pd.DataFrame) -> pd.DataFrame:
    """미래 값으로 결측을 채우지 않는다(bfill 금지). 초반 달의 2/3개월 평균이
    비어있으면 그대로 NaN으로 남겨서, 이후 dropna로 안전한 시점만 걸러낸다."""
    group = group.sort_values("month").copy()
    for col in ["credit_limit", "edu_spend", "fin_stress"]:
        group[col] = group[col].fillna(0)
    group["spend_lag_1"] = group["spend"].shift(1)
    group["spend_2m_avg"] = group["spend"].rolling(2).mean()
    group["spend_3m_avg"] = group["spend"].rolling(3).mean()
    # pct_change의 첫 행 NaN은 "이전 달 자체가 없음"을 0으로 정의하는 것이라 미래를 안 끌어옴(누수 아님)
    group["spend_growth_rate"] = group["spend"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    group["bill_to_spend_ratio"] = group["bill"] / group["spend"].clip(lower=1)
    group["target"] = group["spend"].shift(-1)
    return group


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(y_true, 1, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_unreliable_due_to_zeros": mape(y_true, y_pred),
    }


def report(name: str, metrics: dict) -> None:
    print(f"[{name}] MAE={metrics['mae']:,.0f}  RMSE={metrics['rmse']:,.0f}  MAPE={metrics['mape_unreliable_due_to_zeros']:.1f}%")


PARAM_GRID = {
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 5, 8],
    "min_child_samples": [10, 20, 50],
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1],
}


def tune(estimator, X, y, scoring, n_iter=12):
    search = RandomizedSearchCV(
        estimator, PARAM_GRID, n_iter=n_iter, cv=3, scoring=scoring,
        random_state=42, n_jobs=-1,
    )
    search.fit(X, y)
    print(f"  best_params={search.best_params_}  best_score={search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    raw["month"] = pd.to_datetime(raw["month"])

    featured = raw.groupby("user_id", group_keys=False).apply(build_features, include_groups=False)
    featured["user_id"] = raw.loc[featured.index, "user_id"].values

    # 안전한 행만 남김: target과 모든 lag 피처가 실제 과거 데이터만으로 계산된 경우만
    # (유저별 9월/10월/11월 -> 각각 10월/11월/12월을 예측하는 3개 시점)
    required = ["target"] + LAG_FEATURE_COLS
    before = len(featured)
    featured = featured.dropna(subset=required)
    print(f"누수 제거 전 {before:,}행 -> 안전한 행만 {len(featured):,}행")

    users = featured["user_id"].unique()
    train_users, test_users = train_test_split(users, test_size=0.2, random_state=42)
    train_df = featured[featured["user_id"].isin(train_users)]
    test_df = featured[featured["user_id"].isin(test_users)]

    print(f"학습 유저: {len(train_users):,}명 / 테스트 유저: {len(test_users):,}명")
    print(f"학습 행: {len(train_df):,} / 테스트 행: {len(test_df):,}")
    print(f"학습 행 중 target>0 비율: {(train_df['target'] > 0).mean() * 100:.1f}%")
    print("")

    # ---- 비교군: 단일 회귀모델 ----
    single_model = lgb.LGBMRegressor(random_state=42, n_estimators=200, learning_rate=0.05, verbose=-1)
    single_model.fit(train_df[FEATURE_COLUMNS], train_df["target"])
    single_pred = np.clip(single_model.predict(test_df[FEATURE_COLUMNS]), 0, None)

    # ---- 2단계 모델 ----
    y_train_binary = (train_df["target"] > 0).astype(int)

    print("[튜닝] 분류기(지출>0 여부)")
    classifier, clf_params = tune(
        lgb.LGBMClassifier(random_state=42, verbose=-1),
        train_df[FEATURE_COLUMNS], y_train_binary,
        scoring="f1", n_iter=12,
    )

    reg_train_df = train_df[train_df["target"] > 0]
    print("[튜닝] 회귀기(지출액, target>0인 행만)")
    regressor, reg_params = tune(
        lgb.LGBMRegressor(random_state=42, verbose=-1),
        reg_train_df[FEATURE_COLUMNS], reg_train_df["target"],
        scoring="neg_mean_absolute_error", n_iter=12,
    )
    print("")

    test_binary_pred = classifier.predict(test_df[FEATURE_COLUMNS])
    test_binary_true = (test_df["target"] > 0).astype(int)
    clf_f1 = f1_score(test_binary_true, test_binary_pred)
    print(f"분류기 성능: F1={clf_f1:.3f}")

    regressor_pred_all = np.clip(regressor.predict(test_df[FEATURE_COLUMNS]), 0, None)
    two_stage_pred = np.where(test_binary_pred == 1, regressor_pred_all, 0)

    y_true = test_df["target"].to_numpy()

    metrics = {
        "single_model": compute_metrics(y_true, single_pred),
        "two_stage_model": compute_metrics(y_true, two_stage_pred),
        "baseline_last_month": compute_metrics(y_true, test_df["spend"].to_numpy()),
        "baseline_3m_avg": compute_metrics(y_true, test_df["spend_3m_avg"].to_numpy()),
        "classifier_f1": float(clf_f1),
        "nonzero_only": {
            **compute_metrics(y_true[y_true > 0], two_stage_pred[y_true > 0]),
            "n_rows": int((y_true > 0).sum()),
            "median_actual": float(np.median(y_true[y_true > 0])),
        },
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_users": int(len(train_users)),
        "n_test_users": int(len(test_users)),
        "classifier_best_params": clf_params,
        "regressor_best_params": reg_params,
        "leakage_fix_applied": True,
    }

    print("")
    report("기존 단일모델(비교군)", metrics["single_model"])
    report("2단계 모델(분류+회귀, 튜닝됨)", metrics["two_stage_model"])
    report("Baseline-직전월값", metrics["baseline_last_month"])
    report("Baseline-3개월평균", metrics["baseline_3m_avg"])
    print(f"[2단계, 실제지출>0 구간만] MAE={metrics['nonzero_only']['mae']:,.0f}  "
          f"(실제값 중간값={metrics['nonzero_only']['median_actual']:,.0f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "mode": "two_stage",
            "classifier": classifier,
            "regressor": regressor,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n모델 저장 완료: {MODEL_PATH}")
    print(f"지표 저장 완료: {METRICS_PATH}")


if __name__ == "__main__":
    main()
