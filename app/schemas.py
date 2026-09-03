from pydantic import BaseModel


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
    status: str
    target_month_balance: int
    shortage_month: str | None = None
    recommended_monthly_spend_limit: int


class GuideResponse(BaseModel):
    guide: str


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
