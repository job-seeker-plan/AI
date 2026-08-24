from os import getenv

from fastapi import FastAPI, Header, HTTPException, status

from .predictor import predict_next_spend
from .schemas import GuideRequest, GuideResponse, SpendingPredictionRequest, SpendingPredictionResponse


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
