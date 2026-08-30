"""
전체 후보군 데이터를 모아서 다음달 지출 예측 모델을 학습한다.

v3 (2026-08-26) — 시간 누수 버그 수정:
- v2까지는 각 유저의 초반 달(7,8월)에서 spend_2m_avg/spend_3m_avg가 계산 안 되는 결측을
  bfill()로 채웠는데, 이게 미래 달(예: 9월) 값을 과거(7월) 자리에 끌어오는 구조였다.
  7월 행의 target은 8월 지출인데, bfill로 채워진 7월의 spend_3m_avg 안에 8월 실제값이
  섞여 들어가 있어 정답을 일부 미리 알고 맞추는 꼴이었다. 즉 여태 나온 MAE/RMSE는
  실제보다 좋게 나온 값이었다.
- 수정: bfill 제거. spend_lag_1/spend_2m_avg/spend_3m_avg가 전부 과거 데이터만으로
  안전하게 계산되는 시점(유저별 9월 이후, 즉 9월->10월/10월->11월/11월->12월 3개 시점)만
  학습·평가에 사용한다.

v4 (2026-08-26) — 피처 확장, tier별 성능 비교:
피처를 추가하되, "실서비스 반영 가능성"이 서로 다르다는 걸 숨기지 않기 위해
누적 tier로 나눠서 각각 따로 평가한다.
- tier0(baseline): 기존 피처 그대로.
- tier1(즉시 반영 가능): 한도대비 부담률(spend/balance/bill/installment_balance ÷ credit_limit).
  BE가 이미 보내는 값들로만 계산되므로 지금 바로 배포 가능 -> 이 tier로 학습한 모델을
  spend_predictor.pkl로 저장해 실서비스에 쓴다.
- tier2(BE 스키마 확장 필요): 01번 정적 인구통계(성별/거주지역/카드보유개수/가입경과개월/
  신용·체크카드 보유여부/Life_Stage). BE의 FinancialRecordInput에 아직 없는 필드라
  실서비스 반영은 BE에 필드 추가를 요청해야 풀림. 오프라인 검증에는 쓰지만 배포는 안 함.
- tier3(오프라인 검증 전용): 03번 업종별 소비(식비/교통/납부/쇼핑/여가 비중·증감률),
  이용건수, 할부, RP(정기결제), 온라인비중, 현금서비스 이용여부. 실제 서비스는 사용자가
  총액만 입력하는 구조라(카드사 자동연동 없음) 업종별 세분화는 반영 불가 -
  income과 같은 카테고리의 구조적 한계. "피처가 풍부해지면 얼마나 좋아지는지"를
  보여주는 잠재력 지표로만 리포트에 남긴다.

하이퍼파라미터는 tier3(전체 피처)로 한 번만 튜닝하고, 그 값을 tier0/1/2에도 동일하게
적용해서 MAE/RMSE만 비교한다. tier마다 따로 튜닝하면 "성능 차이가 피처 때문인지
튜닝 운 때문인지" 헷갈리기 때문 - 피처 자체의 순수 효과를 보기 위한 선택.

v5 (2026-08-30) — /tmp/work 오프라인 실험(tier3 기준)에서 검증된 개선 2건을 실제
배포 파이프라인에 반영:
- spend_cv_3m(변동계수 = 3개월 롤링 표준편차 / 3개월 평균): spend_3m_avg와 완전히
  같은 규칙(rolling(3), shift 없음 - 현재+과거만)이라 leak-safe. spend만 있으면
  계산되므로 tier0(baseline)부터 추가 - 배포 tier(tier1_immediate)에도 자동 포함됨.
- 회귀기 타깃을 log1p로 변환해 학습하고 예측 시 expm1로 역변환: 오른쪽 꼬리가 긴
  고액 지출 이상치의 영향을 줄여 MAE가 개선됨(RMSE는 소폭 악화, MAE 개선 폭이 더 큼
  - 오프라인 실험에서 tier3 기준 raw MAE 39,776 -> 39,563). 회귀기 하이퍼파라미터도
  log1p 스케일에 맞춰 "역변환 후 원본 스케일 MAE"를 스코어로 재탐색.
  predictor.py가 이 변환 여부를 알 수 있도록 deploy_bundle/metrics에
  target_transform="log1p"를 기록한다(하위호환: 이 키가 없는 과거 모델은 raw 그대로 사용).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, make_scorer
import lightgbm as lgb
import joblib

TARGET_TRANSFORM = "log1p"

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Data" / "processed" / "user_month_features.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "spend_predictor.pkl"
METRICS_PATH = Path(__file__).resolve().parent / "models" / "train_metrics.json"

CATEGORICAL_COLUMNS = ["gender", "residence_region", "life_stage"]
CATEGORY_SPEND_COLS = ["dining_spend", "transport_spend", "payment_spend", "shopping_spend", "leisure_spend"]

TIER0_BASELINE_COLUMNS = [
    "spend", "bill", "card_outstanding_balance", "credit_limit",
    "spend_lag_1", "spend_2m_avg", "spend_3m_avg", "spend_cv_3m",
    "spend_growth_rate", "bill_to_spend_ratio",
    "edu_spend", "fin_stress",
]

TIER1_IMMEDIATE_COLUMNS = TIER0_BASELINE_COLUMNS + [
    "spend_to_limit_ratio", "balance_to_limit_ratio", "bill_to_limit_ratio",
    "installment_balance_to_limit_ratio",
]

TIER2_BE_SCHEMA_COLUMNS = TIER1_IMMEDIATE_COLUMNS + [
    "gender", "residence_region", "total_card_count", "membership_months",
    "has_credit_card", "has_check_card", "life_stage",
]

TIER3_OFFLINE_ONLY_COLUMNS = TIER2_BE_SCHEMA_COLUMNS + [
    "txn_count", "installment_spend", "installment_count", "rp_amount", "rp_count",
    "online_ratio", "cash_service_used",
] + CATEGORY_SPEND_COLS + [f"{c}_ratio" for c in CATEGORY_SPEND_COLS] + [f"{c}_growth_rate" for c in CATEGORY_SPEND_COLS]

TIERS = {
    "tier0_baseline": TIER0_BASELINE_COLUMNS,
    "tier1_immediate": TIER1_IMMEDIATE_COLUMNS,
    "tier2_be_schema_change_needed": TIER2_BE_SCHEMA_COLUMNS,
    "tier3_offline_only": TIER3_OFFLINE_ONLY_COLUMNS,
}
DEPLOY_TIER = "tier1_immediate"

LAG_FEATURE_COLS = ["spend_lag_1", "spend_2m_avg", "spend_3m_avg", "spend_growth_rate", "bill_to_spend_ratio"]

ZERO_FILL_COLUMNS = [
    "credit_limit", "edu_spend", "fin_stress", "installment_balance",
    "total_card_count", "membership_months", "has_credit_card", "has_check_card",
    "dining_spend", "transport_spend", "payment_spend", "shopping_spend", "leisure_spend",
    "txn_count", "installment_spend", "installment_count", "rp_amount", "rp_count",
    "online_spend", "offline_spend", "cash_service_spend",
]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """미래 값으로 결측을 채우지 않는다(bfill 금지). 초반 달의 2/3개월 평균이
    비어있으면 그대로 NaN으로 남겨서, 이후 dropna로 안전한 시점만 걸러낸다.

    groupby().apply(파이썬 함수) 대신 groupby().shift()/rolling()/pct_change()만 쓴다
    (2026-08-28: 157,353개 그룹에 매번 파이썬 함수를 호출하는 apply 방식이 30분 넘게
    걸려서 벡터 연산으로 재작성 - 결과는 동일, 속도만 개선)."""
    df = raw.sort_values(["user_id", "month"]).copy()
    for col in ZERO_FILL_COLUMNS:
        df[col] = df[col].fillna(0)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("UNKNOWN")

    by_user = df.groupby("user_id")
    df["spend_lag_1"] = by_user["spend"].shift(1)
    df["spend_2m_avg"] = by_user["spend"].rolling(2).mean().reset_index(level=0, drop=True)
    df["spend_3m_avg"] = by_user["spend"].rolling(3).mean().reset_index(level=0, drop=True)
    # spend_3m_avg와 동일한 rolling(3)/shift 없음 규칙이라 NaN 패턴도 동일하게 맞물림(leak-safe).
    spend_std_3m = by_user["spend"].rolling(3).std().reset_index(level=0, drop=True)
    df["spend_cv_3m"] = spend_std_3m / df["spend_3m_avg"].clip(lower=1)
    # pct_change의 첫 행 NaN은 "이전 달 자체가 없음"을 0으로 정의하는 것이라 미래를 안 끌어옴(누수 아님)
    df["spend_growth_rate"] = by_user["spend"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df["bill_to_spend_ratio"] = df["bill"] / df["spend"].clip(lower=1)

    limit_denom = df["credit_limit"].clip(lower=1)
    df["spend_to_limit_ratio"] = df["spend"] / limit_denom
    df["balance_to_limit_ratio"] = df["card_outstanding_balance"] / limit_denom
    df["bill_to_limit_ratio"] = df["bill"] / limit_denom
    df["installment_balance_to_limit_ratio"] = df["installment_balance"] / limit_denom

    spend_denom = df["spend"].clip(lower=1)
    for col in CATEGORY_SPEND_COLS:
        df[f"{col}_ratio"] = df[col] / spend_denom
        df[f"{col}_growth_rate"] = by_user[col].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    online_offline_total = (df["online_spend"] + df["offline_spend"]).clip(lower=1)
    df["online_ratio"] = df["online_spend"] / online_offline_total
    df["cash_service_used"] = (df["cash_service_spend"] > 0).astype(int)

    df["target"] = by_user["spend"].shift(-1)
    return df


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
    # estimator는 n_jobs=1(단일스레드)로 넘기고 RandomizedSearchCV가 n_jobs=-1로 바깥에서
    # 병렬화한다. 둘 다 병렬이면 LightGBM 내부 OpenMP 스레드와 joblib 워커가 중첩되면서
    # Windows에서 행업(deadlock)이 남 - 실제로 2026-08-26 첫 시도에서 이틀간 멈춰있었음.
    search = RandomizedSearchCV(
        estimator, PARAM_GRID, n_iter=n_iter, cv=3, scoring=scoring,
        random_state=42, n_jobs=-1,
    )
    search.fit(X, y)
    print(f"  best_params={search.best_params_}  best_score={search.best_score_:.4f}", flush=True)
    return search.best_estimator_, search.best_params_


def fit_tier(tier_name, columns, train_df, test_df, clf_params, reg_params):
    X_train = train_df[columns]
    y_train_binary = (train_df["target"] > 0).astype(int)

    classifier = lgb.LGBMClassifier(random_state=42, verbose=-1, **clf_params)
    classifier.fit(X_train, y_train_binary)

    reg_train_df = train_df[train_df["target"] > 0]
    regressor = lgb.LGBMRegressor(random_state=42, verbose=-1, **reg_params)
    # log1p로 학습 -> 오른쪽 꼬리가 긴 고액 지출 이상치의 영향을 줄여 raw-scale MAE 개선
    # (RMSE는 소폭 악화되지만 MAE 개선폭이 더 큼, /tmp/work 오프라인 실험으로 검증됨).
    regressor.fit(reg_train_df[columns], np.log1p(reg_train_df["target"]))

    X_test = test_df[columns]
    test_binary_pred = classifier.predict(X_test)
    test_binary_true = (test_df["target"] > 0).astype(int)
    clf_f1 = f1_score(test_binary_true, test_binary_pred)

    regressor_pred_all = np.clip(np.expm1(regressor.predict(X_test)), 0, None)
    two_stage_pred = np.where(test_binary_pred == 1, regressor_pred_all, 0)
    y_true = test_df["target"].to_numpy()

    metrics = {
        "n_features": len(columns),
        "two_stage_model": compute_metrics(y_true, two_stage_pred),
        "classifier_f1": float(clf_f1),
        "nonzero_only": {
            **compute_metrics(y_true[y_true > 0], two_stage_pred[y_true > 0]),
            "n_rows": int((y_true > 0).sum()),
        },
    }
    report(tier_name, metrics["two_stage_model"])
    return classifier, regressor, metrics


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    raw["month"] = pd.to_datetime(raw["month"])

    featured = build_features(raw)
    for col in CATEGORICAL_COLUMNS:
        featured[col] = featured[col].astype("category")

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

    y_train_binary_full = (train_df["target"] > 0).astype(int)
    reg_train_df_full = train_df[train_df["target"] > 0]

    print(f"[튜닝, tier3 전체 피처 기준] 분류기(지출>0 여부)", flush=True)
    _, clf_params = tune(
        lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),
        train_df[TIER3_OFFLINE_ONLY_COLUMNS], y_train_binary_full,
        scoring="f1", n_iter=12,
    )
    print("[튜닝, tier3 전체 피처 기준] 회귀기(지출액 log1p, target>0인 행만, 스코어는 expm1 역변환 후 원본 스케일 MAE)", flush=True)

    def _raw_scale_mae(y_true_log, y_pred_log):
        return -np.mean(np.abs(np.expm1(y_true_log) - np.clip(np.expm1(y_pred_log), 0, None)))

    log1p_raw_mae_scorer = make_scorer(_raw_scale_mae, greater_is_better=True)
    _, reg_params = tune(
        lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=1),
        reg_train_df_full[TIER3_OFFLINE_ONLY_COLUMNS], np.log1p(reg_train_df_full["target"]),
        scoring=log1p_raw_mae_scorer, n_iter=12,
    )
    print("")
    print("=== tier별 성능 비교 (동일 하이퍼파라미터, 피처만 누적 추가) ===")

    baseline_metrics = {
        "baseline_last_month": compute_metrics(test_df["target"].to_numpy(), test_df["spend"].to_numpy()),
        "baseline_3m_avg": compute_metrics(test_df["target"].to_numpy(), test_df["spend_3m_avg"].to_numpy()),
    }
    report("Baseline-직전월값", baseline_metrics["baseline_last_month"])
    report("Baseline-3개월평균", baseline_metrics["baseline_3m_avg"])
    print("")

    tier_results = {}
    deploy_bundle = None
    for tier_name, columns in TIERS.items():
        classifier, regressor, metrics = fit_tier(tier_name, columns, train_df, test_df, clf_params, reg_params)
        tier_results[tier_name] = metrics
        if tier_name == DEPLOY_TIER:
            deploy_bundle = {
                "mode": "two_stage",
                "classifier": classifier,
                "regressor": regressor,
                "feature_columns": columns,
                "target_transform": TARGET_TRANSFORM,
            }

    metrics_out = {
        "baseline": baseline_metrics,
        "tiers": tier_results,
        "deploy_tier": DEPLOY_TIER,
        "tier_definitions": {
            "tier0_baseline": "기존 배포 피처 그대로",
            "tier1_immediate": "한도대비 부담률 추가 - BE 스키마 변경 없이 즉시 배포 (spend_predictor.pkl에 이 tier 저장)",
            "tier2_be_schema_change_needed": "01번 정적 인구통계 추가 - BE의 FinancialRecordInput 확장 필요, 오프라인 검증만",
            "tier3_offline_only": "03번 업종별 소비/이용행태 추가 - 실서비스는 사용자가 총액만 입력하는 구조라 반영 불가, 오프라인 잠재력 지표",
        },
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_users": int(len(train_users)),
        "n_test_users": int(len(test_users)),
        "classifier_best_params": clf_params,
        "regressor_best_params": reg_params,
        "hyperparameter_tuning_note": "tier3(전체 피처)로 1회만 튜닝, 동일 파라미터를 전 tier에 적용(피처 자체의 순수 효과 비교 목적)",
        "leakage_fix_applied": True,
        "target_transform": TARGET_TRANSFORM,
        "target_transform_note": "회귀기는 log1p(target)로 학습, 예측 시 expm1로 역변환 - 고액 지출 이상치 영향을 줄여 MAE 개선(2026-08-30 오프라인 실험으로 검증)",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(deploy_bundle, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    print(f"\n모델 저장 완료 (배포 tier={DEPLOY_TIER}): {MODEL_PATH}")
    print(f"지표 저장 완료: {METRICS_PATH}")


if __name__ == "__main__":
    main()
