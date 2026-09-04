from os import getenv

from fastapi import FastAPI, Header, HTTPException, status

from .hiring_agent.hiring_pattern_pipeline import get_hiring_season, search_companies
from .pattern_advisor import advise_next_week
from .predictor import predict_next_spend
from .linkareer_macro import collect_recruitments
from .financial_rag import build_personalized_guide, save_contexts
from .schemas import (
    CompanySuggestion,
    FinancialContextRequest,
    FinancialContextResponse,
    GuideRequest,
    GuideResponse,
    HiringSeasonResponse,
    LinkareerRecruitSearchRequest,
    LinkareerRecruitSearchResponse,
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


@app.post("/crawl/linkareer/recruitments", response_model=LinkareerRecruitSearchResponse)
async def crawl_linkareer_recruitments(payload: LinkareerRecruitSearchRequest, x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key")) -> LinkareerRecruitSearchResponse:
    verify_internal_key(x_internal_api_key)
    try:
        jobs, source_url, total_count, cached = await collect_recruitments(payload.keyword, payload.category_id, payload.region_id, payload.job_type, payload.page, payload.limit)
        return LinkareerRecruitSearchResponse(jobs=jobs, source_url=source_url, total_count=total_count, page=payload.page, page_size=payload.limit, cached=cached)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


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
    if not payload.user_id:
        return GuideResponse(guide="개인화 가이드를 만들려면 사용자 정보가 필요합니다.")
    return GuideResponse(**build_personalized_guide(payload.model_dump()))


@app.post("/financial-contexts", response_model=FinancialContextResponse)
def save_financial_contexts(payload: FinancialContextRequest, x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key")) -> FinancialContextResponse:
    verify_internal_key(x_internal_api_key)
    try:
        return FinancialContextResponse(saved_count=save_contexts(payload.user_id, [item.model_dump() for item in payload.contexts]))
    except Exception as error:
        raise HTTPException(status_code=503, detail="개인화 정보를 저장하지 못했습니다.") from error


@app.get("/hiring/season", response_model=HiringSeasonResponse)
def hiring_season(company: str, job_family: str, x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key")) -> HiringSeasonResponse:
    """회사명+직무로 과거 채용시즌 패턴을 반환한다. SARAMIN_ACCESS_KEY 승인 전에는
    자동으로 mock 데이터를 쓴다(hiring_pattern_pipeline.get_hiring_season 참고)."""
    verify_internal_key(x_internal_api_key)
    return HiringSeasonResponse(**get_hiring_season(company, job_family))


@app.get("/hiring/companies", response_model=list[CompanySuggestion])
def hiring_companies(q: str = "", x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key")) -> list[CompanySuggestion]:
    """회사명 입력창 자동완성용. 등록된 관심기업(COMPANY_INDUSTRY) 중 q가 포함된 것만 반환.
    q가 비어있으면 등록된 회사 전체를 반환한다(FE는 지금 빈 값으로는 호출 안 하지만,
    엔드포인트 자체는 전체 목록 조회도 지원해둔다)."""
    verify_internal_key(x_internal_api_key)
    return [CompanySuggestion(**item) for item in search_companies(q)]
