"""
전체 후보군 데이터를 모아서 다음달 지출 예측 모델을 학습한다.

v2 (2026-08-26) 변경점:
- 피처에 edu_spend(교육지출액), fin_stress(재정불안정 여부) 추가
  (후보 선정에만 쓰던 조건의 실제값을 모델 입력으로도 재활용)
- 단일 회귀모델 대신 2단계(Zero-Inflated) 모델로 변경:
    1) 분류기: 다음달 지출이 0보다 큰지(할지 안 할지) 예측
    2) 회귀기: 0보다 크다고 예측된 경우에만 실제 금액 예측
  후보군 상당수가 특정 달에 지출이 0원이라(중간값 자체가 0), 이 분포 특성상
  "쓸지 안 쓸지"와 "쓴다면 얼마나 쓸지"를 분리하는 게 이론적으로 더 맞다.
- RandomizedSearchCV로 분류기/회귀기 각각 가벼운 하이퍼파라미터 탐색 추가.

기존 단일 LightGBM 회귀모델도 비교군으로 같이 평가해 실제 개선 여부를 확인한다.
"""

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

FEATURE_COLUMNS = [
    "spend", "bill", "balance", "credit_score",
    "spend_lag_1", "spend_2m_avg", "spend_3m_avg",
    "spend_growth_rate", "bill_to_spend_ratio",
    "edu_spend", "fin_stress",
]

LAG_FEATURE_COLS = ["spend_lag_1", "spend_2m_avg", "spend_3m_avg", "spend_growth_rate", "bill_to_spend_ratio"]


def build_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("month").copy()
    for col in ["credit_score", "edu_spend", "fin_stress"]:
        group[col] = group[col].fillna(0)
    group["spend_lag_1"] = group["spend"].shift(1)
    group["spend_2m_avg"] = group["spend"].rolling(2).mean()
    group["spend_3m_avg"] = group["spend"].rolling(3).mean()
    group["spend_growth_rate"] = group["spend"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    group["bill_to_spend_ratio"] = group["bill"] / group["spend"].clip(lower=1)
    group["target"] = group["spend"].shift(-1)
    group[LAG_FEATURE_COLS] = group[LAG_FEATURE_COLS].bfill().fillna(0)
    return group


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(y_true, 1, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"[{name}] MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape(y_true, y_pred):.1f}%")


CLASSIFIER_PARAM_GRID = {
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 5, 8],
    "min_child_samples": [10, 20, 50],
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1],
}

REGRESSOR_PARAM_GRID = {
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 5, 8],
    "min_child_samples": [10, 20, 50],
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1],
}


def tune(estimator, param_grid, X, y, scoring, n_iter=12):
    search = RandomizedSearchCV(
        estimator, param_grid, n_iter=n_iter, cv=3, scoring=scoring,
        random_state=42, n_jobs=-1,
    )
    search.fit(X, y)
    print(f"  best_params={search.best_params_}  best_score={search.best_score_:.4f}")
    return search.best_estimator_


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    raw["month"] = pd.to_datetime(raw["month"])

    featured = raw.groupby("user_id", group_keys=False).apply(build_features, include_groups=False)
    featured["user_id"] = raw.loc[featured.index, "user_id"].values
    featured = featured.dropna(subset=["target"])

    users = featured["user_id"].unique()
    train_users, test_users = train_test_split(users, test_size=0.2, random_state=42)
    train_df = featured[featured["user_id"].isin(train_users)]
    test_df = featured[featured["user_id"].isin(test_users)]

    print(f"학습 유저: {len(train_users):,}명 / 테스트 유저: {len(test_users):,}명")
    print(f"학습 행: {len(train_df):,} / 테스트 행: {len(test_df):,}")
    print(f"학습 행 중 target>0 비율: {(train_df['target'] > 0).mean() * 100:.1f}%")
    print("")

    # ---- 비교군: 기존 방식 그대로인 단일 회귀모델 ----
    single_model = lgb.LGBMRegressor(random_state=42, n_estimators=200, learning_rate=0.05, verbose=-1)
    single_model.fit(train_df[FEATURE_COLUMNS], train_df["target"])
    single_pred = np.clip(single_model.predict(test_df[FEATURE_COLUMNS]), 0, None)

    # ---- 2단계 모델 ----
    y_train_binary = (train_df["target"] > 0).astype(int)

    print("[튜닝] 분류기(지출>0 여부)")
    classifier = tune(
        lgb.LGBMClassifier(random_state=42, verbose=-1),
        CLASSIFIER_PARAM_GRID, train_df[FEATURE_COLUMNS], y_train_binary,
        scoring="f1", n_iter=12,
    )

    reg_train_df = train_df[train_df["target"] > 0]
    print("[튜닝] 회귀기(지출액, target>0인 행만)")
    regressor = tune(
        lgb.LGBMRegressor(random_state=42, verbose=-1),
        REGRESSOR_PARAM_GRID, reg_train_df[FEATURE_COLUMNS], reg_train_df["target"],
        scoring="neg_mean_absolute_error", n_iter=12,
    )
    print("")

    test_binary_pred = classifier.predict(test_df[FEATURE_COLUMNS])
    test_binary_true = (test_df["target"] > 0).astype(int)
    print(f"분류기 성능: F1={f1_score(test_binary_true, test_binary_pred):.3f}")

    regressor_pred_all = np.clip(regressor.predict(test_df[FEATURE_COLUMNS]), 0, None)
    two_stage_pred = np.where(test_binary_pred == 1, regressor_pred_all, 0)

    y_true = test_df["target"].to_numpy()

    print("")
    report("기존 단일모델(비교군)", y_true, single_pred)
    report("2단계 모델(분류+회귀, 튜닝됨)", y_true, two_stage_pred)
    report("Baseline-직전월값", y_true, test_df["spend"].to_numpy())
    report("Baseline-3개월평균", y_true, test_df["spend_3m_avg"].to_numpy())

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
    print(f"\n모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()
