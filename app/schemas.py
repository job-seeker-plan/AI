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


class GuideRequest(BaseModel):
    status: str
    target_month_balance: int
    shortage_month: str | None = None
    recommended_monthly_spend_limit: int


class GuideResponse(BaseModel):
    guide: str
