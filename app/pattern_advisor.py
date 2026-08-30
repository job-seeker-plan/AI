"""
pattern_advisor.py

"다음달 지출액을 숫자로" 대신 "다음주에는 어떻게 쓰면 좋을지"를 준다.

왜 숫자가 아니라 분류인가
--------------------------
AI Hub 카드데이터는 이용금액_신용_B0M처럼 전부 "당월 합계"만 있고 실제 결제
날짜가 원본에 없다(요일별 요청 때와 같은 제약). 그래서 "이번 주 지출액"을
회귀로 정확히 맞히는 건 이 데이터로는 애초에 불가능하다. 대신 이번 달까지
누적된 월별 신호(지출 추세, 변동성, 한도 대비 부담, 교육비 비중)로 "이 사람이
지금 어떤 소비 흐름 위에 있는지"를 분류하고, 그 흐름에 맞는 행동 조언을 텍스트로
낸다 - 숫자의 정확도보다 조언이 실제로 말이 되는지가 중요한 방식이라 이 데이터의
한계와 맞는 접근이다.

분류 우선순위 (위에서부터 먼저 만족하는 것으로 확정)
------------------------------------------------
1. 무지출형    : 이번 달 카드 지출 자체가 없음
2. 한도부담형  : 잔액/청구액이 한도 대비 부담스러운 수준
3. 급증/변동형 : 최근 지출이 크게 늘거나(추세) 달마다 들쭉날쭉함(변동계수)
4. 교육집중형  : 교육비 비중/증가율이 두드러짐
5. 절약/감소형 : 최근 지출이 뚜렷하게 줄어드는 추세
6. 안정형      : 위 어디에도 해당 안 됨(기본값)

임계값 출처
-----------
임의로 정하지 않고, 실제 후보군 데이터(157,353명, leak-safe 유효 472,059행 -
train.py의 dropna(subset=["target"]+LAG_FEATURE_COLS) 기준과 동일)의 분포에서
75~90분위수를 근거로 잡았다. 이 임계값으로 전체를 나눠보면:
  무지출형 55.4% / 한도부담형 27.3% / 급증·변동형 6.1% /
  교육집중형 4.4% / 절약·감소형 4.4% / 안정형 2.4%
(2026-08-30 검증, Data/processed/user_month_features.csv 기준. 재현하려면
predict_next_spend와 같은 방식으로 build_features를 돌리고 classify_pattern_row를
apply하면 된다.) 안정형이 제일 작은 건 버그가 아니라, 이 후보군 자체가 재정적으로
빠듯한 취준생 위주라 "특별히 문제 없는 달"이 원래 드물기 때문이다.

predictor.py와의 관계
----------------------
load_history와 tier1 build_features(spend_lag_1/2m_avg/3m_avg/growth_rate,
*_to_limit_ratio)는 predictor.py 걸 그대로 재사용한다. 여기서는 분류에만 필요한
spend_cv_3m(변동계수), edu_spend_ratio/growth_rate만 얹는다. 배포 모델
(spend_predictor.pkl)의 피처 구성은 건드리지 않는다.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import load_history, build_features as _tier1_build_features  # noqa: E402

THRESHOLDS = {
    "balance_to_limit_ratio": 0.58,   # 유효 데이터 75분위
    "bill_to_limit_ratio": 0.33,      # 유효 데이터 75분위
    "spend_growth_rate_high": 0.13,   # 유효 데이터 90분위
    "spend_cv_3m_high": 0.41,         # 유효 데이터 90분위
    "edu_spend_ratio": 0.026,         # 유효 데이터 90분위
    "edu_spend_growth_rate": 0.5,     # 전월 대비 50% 이상 증가
    "spend_growth_rate_low": -0.017,  # 유효 데이터 25분위
}

ADVICE = {
    "무지출형": {
        "summary": "이번 달 카드 지출이 거의 없었어요.",
        "advice": "다음 주도 필수 지출만 있다면 지금 흐름 그대로 괜찮아요. 다만 갑자기 큰 지출(면접비, 자격증 접수 등)이 예상되면 미리 얼마나 필요한지 적어두고 시작하세요.",
    },
    "한도부담형": {
        "summary": "카드 잔액·청구액이 한도 대비 부담스러운 수준이에요.",
        "advice": "다음 주는 새 지출을 최대한 줄이고, 다음 결제일 전에 얼마를 갚을 수 있을지부터 확인해보세요. 리볼빙이나 카드론이 있다면 이자 부담도 같이 점검하는 게 좋아요.",
    },
    "급증/변동형": {
        "summary": "최근 지출이 평소보다 크게 늘었거나 달마다 들쭉날쭉해요.",
        "advice": "다음 주는 꼭 필요한 지출만 하면서 한 주 정도 흐름을 지켜보세요. 어느 항목에서 갑자기 늘었는지 한 번 확인해보면 다음 계획을 세우기 쉬워져요.",
    },
    "교육집중형": {
        "summary": "최근 교육·자격증 관련 지출 비중이 눈에 띄게 높아요.",
        "advice": "취업 준비를 위한 투자로 보이니 이 부분은 유지해도 좋아요. 대신 다음 주는 쇼핑·여가 같은 다른 항목에서 씀씀이를 조금 줄여 균형을 맞춰보세요.",
    },
    "절약/감소형": {
        "summary": "최근 지출이 뚜렷하게 줄어드는 추세예요.",
        "advice": "좋은 흐름이니 다음 주도 이어가면 됩니다. 다만 이력서 인쇄비, 면접 교통비처럼 취업 준비에 꼭 필요한 지출까지 과하게 아끼지는 마세요.",
    },
    "안정형": {
        "summary": "최근 소비 패턴이 비교적 안정적이에요.",
        "advice": "다음 주도 평소 예산 범위 안에서 편하게 쓰셔도 괜찮은 상태예요.",
    },
}


def _build_classification_features(frame: pd.DataFrame) -> pd.DataFrame:
    """predictor.py의 tier1 build_features에 분류 전용 파생값만 추가.
    spend_cv_3m은 2026-08-30부터 배포 모델(tier0 이상)에도 쓰여 predictor.py의
    build_features가 직접 계산하므로 여기서는 재사용만 하고 중복 계산하지 않는다."""
    result = _tier1_build_features(frame)
    if "edu_spend" not in result:
        result["edu_spend"] = 0
    result["spend_cv_3m"] = result["spend_cv_3m"].replace([np.inf, -np.inf], 0).fillna(0)
    # 원본 데이터에서 카테고리 합이 총액을 넘는 달이 드물게 있어(예: 총액은 환불 상계 후,
    # 카테고리는 상계 전) 비율을 [0, 1]로 clip한다 - 분류 임계값 판정에는 영향 없음.
    result["edu_spend_ratio"] = (
        (result["edu_spend"] / result["spend"].clip(lower=1)).fillna(0).clip(upper=1.0)
    )
    result["edu_spend_growth_rate"] = (
        result["edu_spend"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    )
    return result


def classify_pattern_row(row: pd.Series) -> str:
    if row["spend"] <= 0:
        return "무지출형"
    if (
        row["balance_to_limit_ratio"] >= THRESHOLDS["balance_to_limit_ratio"]
        or row["bill_to_limit_ratio"] >= THRESHOLDS["bill_to_limit_ratio"]
    ):
        return "한도부담형"
    if (
        row["spend_growth_rate"] >= THRESHOLDS["spend_growth_rate_high"]
        or row["spend_cv_3m"] >= THRESHOLDS["spend_cv_3m_high"]
    ):
        return "급증/변동형"
    if (
        row["edu_spend_ratio"] >= THRESHOLDS["edu_spend_ratio"]
        or row["edu_spend_growth_rate"] >= THRESHOLDS["edu_spend_growth_rate"]
    ):
        return "교육집중형"
    if row["spend_growth_rate"] <= THRESHOLDS["spend_growth_rate_low"]:
        return "절약/감소형"
    return "안정형"


def advise_next_week(user_id: str, records: list[dict] | None = None) -> dict:
    """최신 1개월치 신호로 패턴을 분류하고 다음 주 행동 조언을 낸다.
    records를 주면 BE 실시간 스키마(FinancialRecordInput)로 취급하고,
    안 주면 predictor.py와 동일하게 오프라인 CSV에서 그 user_id의 이력을 읽는다."""
    history = load_history(user_id, records)
    featured = _build_classification_features(history)
    latest = featured.iloc[-1]
    pattern = classify_pattern_row(latest)
    info = ADVICE[pattern]
    return {
        "user_id": user_id,
        "pattern": pattern,
        "summary": info["summary"],
        "advice": info["advice"],
        "signals": {
            "spend_this_month": int(latest["spend"]),
            "spend_growth_rate": float(latest["spend_growth_rate"]),
            "spend_cv_3m": float(latest["spend_cv_3m"]),
            "balance_to_limit_ratio": float(latest["balance_to_limit_ratio"]),
            "bill_to_limit_ratio": float(latest["bill_to_limit_ratio"]),
            "edu_spend_ratio": float(latest["edu_spend_ratio"]),
        },
    }
