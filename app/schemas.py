from pydantic import BaseModel, Field


class SpendingPredictionRequest(BaseModel):
    user_id: str
    records: list["FinancialRecordInput"] = []


class FinancialRecordInput(BaseModel):
    user_id: str
    month: str
    spend: int
    bill: int
    balance: int
    credit_score: int | None = None
    income: int = 0


class FeatureImpact(BaseModel):
    feature: str
    impact: float


class SpendingPredictionResponse(BaseModel):
    user_id: str
    predicted_next_spend: int
    recent_average_spend: int
    baseline_last_month_spend: int
    feature_importance: list[FeatureImpact]


class PatternSignals(BaseModel):
    spend_this_month: int
    spend_growth_rate: float
    spend_cv_3m: float
    balance_to_limit_ratio: float
    bill_to_limit_ratio: float
    edu_spend_ratio: float


class PatternAdviceResponse(BaseModel):
    user_id: str
    pattern: str
    summary: str
    advice: str
    signals: PatternSignals


class GuideRequest(BaseModel):
    user_id: str | None = None
    status: str
    target_month_balance: int
    shortage_month: str | None = None
    recommended_monthly_spend_limit: int
    related_category: str = "cashflow"


class GuideResponse(BaseModel):
    guide: str
    personalized: bool = False
    context_count: int = 0


class FinancialContextInput(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    data_type: str = Field(default="onboarding", max_length=40)
    related_category: str = Field(default="cashflow", max_length=40)
    emotion_tag: str | None = Field(default=None, max_length=40)
    urgency_level: str = Field(default="normal", max_length=20)


class FinancialContextRequest(BaseModel):
    user_id: str
    contexts: list[FinancialContextInput] = Field(min_length=1, max_length=8)


class FinancialContextResponse(BaseModel):
    saved_count: int


class HiringSeasonMonthly(BaseModel):
    month: int
    posting_count: int
    seasonality_share: float


class SkillTrendItem(BaseModel):
    skill: str
    count: int
    n_postings: int


class JdEvidenceItem(BaseModel):
    chunk_text: str
    posted_date: str
    distance: float | None = None


class HiringSeasonResponse(BaseModel):
    company: str
    job_family: str
    n_postings_analyzed: int
    observed_windows: list[str]
    observed_dates: list[str] = []
    monthly_breakdown: list[HiringSeasonMonthly] = []
    skill_trend: list[SkillTrendItem] = []
    jd_evidence: list[JdEvidenceItem] = []
    basis: str
    caveat: str


class CompanySuggestion(BaseModel):
    company: str
    industry: str


class LinkareerRecruitSearchRequest(BaseModel):
    keyword: str = Field(default="", max_length=80)
    category_id: str | None = Field(default=None, max_length=20)
    region_id: str | None = Field(default=None, max_length=20)
    job_type: str | None = Field(default=None, max_length=20)
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=20)


class LinkareerRecruitment(BaseModel):
    id: str
    title: str
    company: str
    categories: list[str]
    locations: list[str]
    employment_type: str
    deadline: str
    url: str


class LinkareerRecruitSearchResponse(BaseModel):
    jobs: list[LinkareerRecruitment]
    source_url: str
    total_count: int
    page: int
    page_size: int
    cached: bool
