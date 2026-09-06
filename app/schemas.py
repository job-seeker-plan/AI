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


class NextEventInfo(BaseModel):
    title: str
    event_type: str
    event_date: str


class GuideRequest(BaseModel):
    user_id: str | None = None
    status: str
    target_month_balance: int
    shortage_month: str | None = None
    recommended_monthly_spend_limit: int
    related_category: str = "cashflow"
    # 취업 일정 에이전트(캘린더)와 재무 가이드 에이전트를 한 번의 생성 호출로 묶기
    # 위한 값. 있으면 "왜 지금 지출을 줄여야 하는지"와 "왜 이 일정부터 챙겨야
    # 하는지"를 하나의 근거로 같이 설명한다.
    next_event: NextEventInfo | None = None


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


class FinancialContextItem(BaseModel):
    text: str
    data_type: str
    related_category: str
    emotion_tag: str | None = None
    urgency_level: str


class FinancialContextListResponse(BaseModel):
    contexts: list[FinancialContextItem]


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
    region_name: str | None = Field(default=None, max_length=20)
    experience: str | None = Field(default=None, max_length=20)
    deadline_within_days: int | None = Field(default=None, ge=0, le=3650)


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


class EmailMessageInput(BaseModel):
    message_id: str
    subject: str = ""
    sender: str = ""
    date: str = ""
    body: str = ""


class EmailParseRequest(BaseModel):
    messages: list[EmailMessageInput] = Field(max_length=50)


class EmailEventCandidate(BaseModel):
    message_id: str
    title: str
    event_type: str
    event_date: str
    memo: str = ""


class EmailParseResponse(BaseModel):
    events: list[EmailEventCandidate]
