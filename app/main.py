from os import getenv

from fastapi import FastAPI, Header, HTTPException, status

from .pattern_advisor import advise_next_week
from .predictor import predict_next_spend
from .schemas import (
    GuideRequest,
    GuideResponse,
    PatternAdviceResponse,
    SpendingPredictionRequest,
    SpendingPredictionResponse,
)


app = FastAPI(title="Job Seeker Financial Planner AI Service")


def verify_internal_key(x_internal_api_key: str | None) -> None:
    expected = getenv("AI_SERVICE_TOKEN", "")
    if not expected or x_internal_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal service credential")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/spending", response_model=SpendingPredictionResponse)
def predict_spending(payload: SpendingPredictionRequest, x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key")) -> SpendingPredictionResponse:
    try:
        verify_internal_key(x_internal_api_key)
        records = [record.model_dump() for record in payload.records]
        return SpendingPredictionResponse(**predict_next_spend(payload.user_id, records))
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/predict/pattern", response_model=PatternAdviceResponse)
def predict_pattern(payload: SpendingPredictionRequest, x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key")) -> PatternAdviceResponse:
    """다음달 지출액 대신, 이번 달까지의 흐름으로 소비 패턴을 분류하고 다음 주
    행동 조언을 준다 - AI Hub 원본에 결제 날짜가 없어 주 단위 금액 예측 자체가
    불가능해서(pattern_advisor.py 모듈 docstring 참고) 택한 방식."""
    try:
        verify_internal_key(x_internal_api_key)
        records = [record.model_dump() for record in payload.records]
        return PatternAdviceResponse(**advise_next_week(payload.user_id, records))
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/guide", response_model=GuideResponse)
def build_guide(payload: GuideRequest, x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key")) -> GuideResponse:
    verify_internal_key(x_internal_api_key)
    if payload.status == "risk":
        guide = f"{payload.shortage_month}에 자금 부족이 예상됩니다. 월 지출 한도를 {payload.recommended_monthly_spend_limit:,}원으로 낮추는 시나리오를 먼저 검토하세요."
    elif payload.status == "caution":
        guide = "목표 취업월까지 현금흐름이 빠듯합니다. 확정 일정 비용을 이번 달 예산에 먼저 반영하고 정책 후보를 적용해 보세요."
    else:
        guide = "목표 취업월까지 현금흐름은 안정적입니다. 시험·면접 비용이 늘어나는 달만 별도로 관리하면 됩니다."
    return GuideResponse(guide=guide)
