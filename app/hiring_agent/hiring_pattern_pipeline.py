"""
채용시즌 분석 파이프라인 프로토타입 (사람인 API 승인 전, mock 데이터로 검증용)

사람인 API 실제 응답 스키마(oapi.saramin.co.kr/guide/job-search 확인함):
{
  "jobs": {
    "count": 10,
    "start": 0,
    "total": "42",                    # 문자열로 옴, int 변환 필요
    "job": [
      {
        "url": "...",
        "active": 1,                  # 1=진행중, 0=마감
        "company": {"detail": {"href": "...", "name": "(주)삼성전자"}},
        "position": {
          "title": "...",
          "industry": {"code": "...", "name": "..."},
          "job-mid-code": {"code": "...", "name": "데이터"},
          "job-code": {"code": "...", "name": "..."},
          "location": {"code": "...", "name": "..."},
          "job-type": {"code": "...", "name": "..."},
          "experience-level": {"code": "...", "min": 0, "max": 0, "name": "신입"},
        },
        "id": "12345",
        "posting-date": "2025-03-05 09:00:00",     # fields=posting-date 파라미터 필요
        "expiration-date": "2025-03-20 24:00:00",  # fields=expiration-date 파라미터 필요
      },
      ...
    ]
  }
}

이 스크립트는 이 구조를 그대로 가정하고 파싱한다. 실제 키 발급되면
fetch_postings()의 requests.get() 부분만 주석 풀면 됨 - 나머지(파싱/표준화/
시즌계산/출력)는 이미 이 스키마 기준으로 완성돼있어서 안 건드려도 됨.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from datetime import date
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)
_llm_client: OpenAI | None = None


def _get_llm_client() -> OpenAI | None:
    """키 없으면(아직 설정 전) None 반환 - 호출부에서 템플릿으로 폴백한다."""
    global _llm_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if _llm_client is None:
        _llm_client = OpenAI(api_key=api_key)
    return _llm_client

# =====================================================================
# ① 사람인 API 호출 (승인 후 access_key 채우면 바로 실동작)
# =====================================================================

SARAMIN_BASE_URL = "https://oapi.saramin.co.kr/job-search"


def fetch_postings(keywords: str, published_min: str, published_max: str,
                    access_key: Optional[str] = None, count: int = 110) -> dict:
    """실제 API 호출. access_key 없으면(아직 승인 전) mock 응답을 반환한다.

    published_min/max 형식: "YYYYMMDD" (API 가이드 기준, 정확한 포맷은
    승인 후 guide/job-search 페이지에서 재확인 필요 - datetime/timestamp라고만
    적혀있어서 실제 요구 포맷은 첫 호출 때 검증할 것)."""
    if access_key is None:
        return _mock_saramin_response(keywords)

    import requests  # 실제 호출 시에만 필요
    params = {
        "access-key": access_key,
        "keywords": keywords,
        "published_min": published_min,
        "published_max": published_max,
        "count": count,
        "fields": "posting-date,expiration-date",  # 이 필드들은 명시 안 하면 응답에 안 옴
    }
    resp = requests.get(SARAMIN_BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _mock_saramin_response(keywords: str) -> dict:
    """실제 API 스키마를 그대로 흉내낸 가짜 응답. 회사명 표기가 갈라지는 경우
    (주)삼성전자 vs 삼성전자)까지 일부러 섞어서 ③ 표준화 로직도 테스트되게 함."""
    fixtures = {
        "삼성전자": [
            ("2022-03-07", "(주)삼성전자", "데이터 분석 신입", "데이터", "SQL,Python,데이터분석,통계"),
            ("2022-09-02", "삼성전자", "데이터 엔지니어 경력", "데이터", "빅데이터,SQL,머신러닝,클라우드"),
            ("2023-03-10", "(주)삼성전자", "데이터 분석 신입", "데이터", "SQL,Python,통계,데이터시각화"),
            ("2023-09-04", "삼성전자㈜", "데이터 엔지니어 경력", "데이터", "머신러닝,Python,빅데이터,SQL"),
            ("2024-03-05", "(주)삼성전자", "데이터 분석 신입", "데이터", "SQL,데이터분석,통계,Tableau"),
            ("2025-03-11", "삼성전자", "데이터 분석 신입", "데이터", "Python,SQL,데이터분석,머신러닝"),
        ],
        "삼성SDS": [
            ("2022-03-14", "삼성에스디에스", "IT컨설팅 신입", "IT컨설팅", "SQL,클라우드,IT컨설팅"),
            ("2023-03-13", "(주)삼성에스디에스", "IT컨설팅 신입", "IT컨설팅", "클라우드,SAP,IT컨설팅"),
            ("2024-03-06", "삼성SDS", "IT컨설팅 경력", "IT컨설팅", "클라우드,SQL,IT컨설팅"),
        ],
        "카카오": [
            ("2022-08-01", "카카오", "서비스 기획 신입", "서비스기획", "서비스기획,데이터분석,UX리서치"),
            ("2023-07-31", "(주)카카오", "서비스 기획 신입", "서비스기획", "서비스기획,UX리서치,SQL"),
            ("2024-08-05", "카카오", "서비스 기획 경력", "서비스기획", "서비스기획,데이터분석,SQL"),
        ],
        "카카오뱅크": [
            ("2022-10-10", "카카오뱅크", "금융IT 신입", "금융IT", "핀테크,백엔드,금융시스템"),
            ("2023-10-16", "(주)카카오뱅크", "금융IT 신입", "금융IT", "핀테크,Java,금융시스템"),
            ("2024-10-14", "카카오뱅크", "금융IT 경력", "금융IT", "핀테크,백엔드,Java"),
        ],
        "CJ ENM": [
            ("2022-09-05", "(주)씨제이이엔엠", "콘텐츠 기획 신입", "콘텐츠", "콘텐츠기획,트렌드분석,커뮤니케이션"),
            ("2023-08-28", "CJ ENM", "콘텐츠 기획 신입", "콘텐츠", "콘텐츠기획,SNS운영,트렌드분석"),
            ("2024-09-02", "(주)씨제이이엔엠", "콘텐츠 기획 신입", "콘텐츠", "콘텐츠기획,커뮤니케이션,엔터테인먼트"),
        ],
        "CJ대한통운": [
            ("2022-06-08", "씨제이대한통운", "물류 운영 신입", "물류", "물류관리,SCM,재고관리"),
            ("2023-06-12", "(주)씨제이대한통운", "물류 운영 신입", "물류", "SCM,ERP,물류관리"),
            ("2023-12-04", "CJ대한통운", "SCM 기획 경력", "물류", "SCM,데이터분석,ERP"),
            ("2024-06-10", "씨제이대한통운", "물류 운영 신입", "물류", "물류관리,재고관리,SCM"),
        ],
        "CJ제일제당": [
            ("2022-02-07", "씨제이제일제당", "생산관리 신입", "생산관리", "품질관리,생산관리,식품위생"),
            ("2023-02-13", "(주)씨제이제일제당", "생산관리 신입", "생산관리", "생산관리,공정개선,품질관리"),
            ("2024-02-05", "CJ제일제당", "생산관리 신입", "생산관리", "품질관리,생산관리,ERP"),
        ],
        "CJ올리브영": [
            ("2022-11-07", "씨제이올리브영", "MD 신입", "MD", "상품기획,트렌드분석,엑셀"),
            ("2023-11-13", "(주)씨제이올리브영", "MD 신입", "MD", "MD,상품기획,데이터분석"),
            ("2024-11-04", "CJ올리브영", "MD 경력", "MD", "상품기획,트렌드분석,데이터분석"),
        ],
    }
    rows = fixtures.get(keywords, [])
    jobs = []
    for i, (posted, company_raw, title, job_family, keyword_raw) in enumerate(rows):
        jobs.append({
            "url": f"https://example.saramin.co.kr/mock/{i}",
            "active": 0,
            "company": {"detail": {"href": "", "name": company_raw}},
            "keyword": keyword_raw,
            "position": {
                "title": title,
                "job-mid-code": {"code": "999", "name": job_family},
                "job-code": {"code": "999-1", "name": job_family},
                "experience-level": {"code": "1", "min": 0, "max": 0, "name": "신입"},
                "job-type": {"code": "1", "name": "정규직"},
                "location": {"code": "101000", "name": "서울"},
            },
            "id": f"mock-{keywords}-{i}",
            "posting-date": f"{posted} 09:00:00",
            "expiration-date": "",
        })
    return {"jobs": {"count": len(jobs), "start": 0, "total": str(len(jobs)), "job": jobs}}


# =====================================================================
# ② Raw 저장 - API 응답 -> raw_job_postings 행으로 변환 (가공 없이 그대로)
# =====================================================================

def parse_response_to_rows(response: dict) -> list[dict]:
    """사람인 응답(jobs.job 배열)을 raw_job_postings 스키마 행 리스트로 변환.
    이 단계에서는 값을 고치거나 표준화하지 않는다 - 원본 그대로 저장."""
    rows = []
    for job in response.get("jobs", {}).get("job", []):
        position = job.get("position", {})
        rows.append({
            "posting_id": job.get("id"),
            "company_name_raw": job.get("company", {}).get("detail", {}).get("name", ""),
            "title": position.get("title", ""),
            "job_family_raw": position.get("job-mid-code", {}).get("name", ""),
            "experience_level": position.get("experience-level", {}).get("name", ""),
            "employment_type": position.get("job-type", {}).get("name", ""),
            "location": position.get("location", {}).get("name", ""),
            "posted_date": job.get("posting-date", "")[:10] or None,  # "YYYY-MM-DD HH:MM:SS" -> 앞 10자리
            "expiration_date": job.get("expiration-date", "")[:10] or None,
            "source_url": job.get("url", ""),
            "keyword_raw": job.get("keyword", ""),  # 쉼표구분 태그, 스킬트렌드 집계용(JD 본문 아님)
        })
    return rows


# =====================================================================
# ③ 기업명 표준화 (Entity Resolution) - 관심기업 소수 스케일이라 수동 매핑 우선
# =====================================================================

_STRIP_PATTERN = re.compile(r"(주식회사|\(주\)|㈜)")


def normalize_company_name(name_raw: str) -> str:
    """"(주)삼성전자", "삼성전자㈜", "삼성전자 " -> "삼성전자" 로 정규화.
    이 정도 문자열 제거만으로 안 잡히는 케이스는 COMPANY_ALIASES에 수동으로 추가.

    앞뒤 공백만 strip()으로 제거하고 내부 공백은 건드리지 않는다 - "\\s+" 전체를 지우면
    "CJ ENM" 같은 다단어 회사명이 "CJENM"으로 뭉개져서, "(주)씨제이이엔엠"에서 정규화된
    "CJ ENM"과 서로 다른 company_id로 갈라진다."""
    return _STRIP_PATTERN.sub("", name_raw).strip()


# 정규화로도 안 잡히는 표기(완전히 다른 이름을 쓰는 경우)는 여기 수동 등록.
# 예: "씨제이이엔엠" <-> "CJ ENM"
COMPANY_ALIASES: dict[str, str] = {
    "씨제이이엔엠": "CJ ENM",
    "씨제이대한통운": "CJ대한통운",
    "씨제이제일제당": "CJ제일제당",
    "씨제이올리브영": "CJ올리브영",
    "삼성에스디에스": "삼성SDS",
}

# 관심기업 등록 시 같이 정하는 업종 태그 (피어그룹 묶을 때 씀, 자동판단 안 함).
# 부분일치 자동완성 데모를 위해 그룹별로 여러 계열사를 등록해뒀다(삼성 2개/CJ 4개/
# 카카오 2개) - 실제 관심기업은 나중에 채울 것.
COMPANY_INDUSTRY: dict[str, str] = {
    "삼성전자": "제조/전자",
    "삼성SDS": "IT서비스",
    "CJ ENM": "엔터테인먼트",
    "CJ대한통운": "물류",
    "CJ제일제당": "식품/제조",
    "CJ올리브영": "헬스&뷰티 리테일",
    "카카오": "IT/플랫폼",
    "카카오뱅크": "핀테크",
}


def search_companies(query: str) -> list[dict]:
    """자동완성용 - COMPANY_INDUSTRY(관심기업 목록)에서 query가 포함된 회사명만 찾는다.
    빈 문자열이면 등록된 회사 전체를 반환한다(FE는 사용자가 뭔가 입력하기 전까진
    이 함수를 안 부르지만, 함수 자체는 빈 값도 지원해둔다).
    이 목록에 없는 회사는 애초에 후보로 안 뜨니, get_hiring_season이 모르는 회사명으로
    호출되는 경우를 UI 단에서 막아주는 효과도 있다."""
    q = query.strip().lower()
    return [
        {"company": name, "industry": industry}
        for name, industry in COMPANY_INDUSTRY.items()
        if q in name.lower()
    ]


def resolve_company(name_raw: str) -> str:
    """canonical_name(표준 회사명)을 반환. 새 회사면 정규화된 이름을 그대로 씀."""
    normalized = normalize_company_name(name_raw)
    return COMPANY_ALIASES.get(normalized, normalized)


# =====================================================================
# ④ 시즌 통계 계산 - 월별 집계 + (방법A) 관측된 날짜 구간
# =====================================================================

_RAW_POSTING_COLUMNS = [
    "posting_id", "company_name_raw", "title", "job_family_raw", "experience_level",
    "employment_type", "location", "posted_date", "expiration_date", "source_url", "keyword_raw",
]


def build_raw_postings_df(all_rows: list[dict]) -> pd.DataFrame:
    """all_rows가 빈 리스트(검색 결과 0건)여도 컬럼이 살아있는 빈 표를 만든다 - 컬럼을
    명시 안 하면 빈 리스트로 만든 DataFrame엔 컬럼 자체가 없어서 다음 줄 df["posted_date"]
    에서 KeyError가 난다. 자동완성으로 회사명을 등록된 것만 고르게 해도, API를 직접
    호출하는 경우까지 대비해 방어적으로 남겨둔다."""
    df = pd.DataFrame(all_rows, columns=_RAW_POSTING_COLUMNS)
    df["posted_date"] = pd.to_datetime(df["posted_date"])
    df["company_id"] = df["company_name_raw"].map(resolve_company)
    df["job_family"] = df["job_family_raw"]  # 필요시 여기도 표준화 매핑 추가 가능
    return df


def compute_hiring_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """company_id x job_family x month 별 공고 건수 + 시즌 비중.
    주의: 이 비중은 "그 회사가 관측된 달들 중 몇 %가 이 달이었나"이지,
    확률적 예측치가 아니다 - 표본이 작을 때(공고 몇 건뿐인 회사) 과신하지 않도록 주의."""
    work = df.copy()
    work["month"] = work["posted_date"].dt.month
    monthly = (
        work.groupby(["company_id", "job_family", "month"])
        .size()
        .reset_index(name="posting_count")
    )
    total_by_group = monthly.groupby(["company_id", "job_family"])["posting_count"].transform("sum")
    monthly["seasonality_share"] = monthly["posting_count"] / total_by_group
    monthly["n_observations"] = total_by_group
    return monthly.sort_values(["company_id", "job_family", "posting_count"], ascending=[True, True, False])


def compute_skill_trend(df: pd.DataFrame, company_id: str, job_family: str, top_n: int = 5) -> list[dict]:
    """"경로A"(keyword 태그 카운팅) - JD 본문 크롤링(RAG) 없이도 사람인 API가 주는
    keyword 필드만으로 만들 수 있는 스킬트렌드. RAG랑 달리 확률 계산이 아니라
    단순 등장 횟수 집계라 지어낸 숫자가 없다."""
    subset = df[(df["company_id"] == company_id) & (df["job_family"] == job_family)]
    total = len(subset)
    if total == 0:
        return []
    tags = subset["keyword_raw"].fillna("").str.split(",").explode().str.strip()
    tags = tags[tags != ""]
    counts = tags.value_counts().head(top_n)
    return [
        {"skill": skill, "count": int(count), "n_postings": total}
        for skill, count in counts.items()
    ]


def compute_observed_window(df: pd.DataFrame, company_id: str, job_family: str,
                              tolerance_days: int = 3) -> dict:
    """방법A: 확률(%)을 지어내지 않고, 과거 실제 게시일들을 모아서
    '이 구간 안에 다 들어온다'는 식으로 반환. 연도 정보를 떼고 월만 비교해서
    묶는다(달마다 반복되는 계절성을 보는 것이므로 연도는 무시).

    가장 많이 나온 달이 여러 개(동률)면 전부 별도 구간으로 반환한다 - 대기업
    상반기/하반기 공채처럼 1년에 두 번 몰리는 경우를 하나로 뭉개지 않기 위함
    (최빈값 하나만 고르면 이 케이스를 놓친다)."""
    subset = df[(df["company_id"] == company_id) & (df["job_family"] == job_family)]
    if subset.empty:
        return {"company_id": company_id, "job_family": job_family, "n_observations": 0,
                "peak_months": [], "windows": [], "observed_dates": []}

    # 표시용 observed_dates는 연도 포함 전체 날짜(YYYY-MM-DD) - 사용자가 실제 근거를
    # 검증할 수 있어야 하므로. 클러스터링(peak_months, window)만 연도 무시하고 월-일로 봄.
    md = subset["posted_date"].dt.strftime("%Y-%m-%d").tolist()
    month_counts = subset["posted_date"].dt.month.value_counts()
    peak_months = sorted(month_counts[month_counts == month_counts.max()].index.tolist())

    windows = []
    for month in peak_months:
        cluster = subset[subset["posted_date"].dt.month == month]["posted_date"]
        window_start = (cluster.min() - pd.Timedelta(days=tolerance_days)).strftime("%m-%d")
        window_end = (cluster.max() + pd.Timedelta(days=tolerance_days)).strftime("%m-%d")
        windows.append(f"{window_start} ~ {window_end}")

    return {
        "company_id": company_id,
        "job_family": job_family,
        "n_observations": len(subset),
        "peak_months": [int(m) for m in peak_months],
        "windows": windows,
        "observed_dates": sorted(md),
    }


# =====================================================================
# ⑤ 출력 - 다른 에이전트(오케스트레이터)가 호출할 API 계약
# =====================================================================

def _week_label(day: int) -> str:
    if day <= 7:
        return "첫째주"
    if day <= 14:
        return "둘째주"
    if day <= 21:
        return "셋째주"
    return "넷째주"


def _describe_window(window: str) -> str:
    """"03-04 ~ 03-14" -> "3월 첫째주~둘째주" 같은 자연스러운 문구로 변환.
    LLM 아님 - 그냥 문자열 조합 템플릿(가계부AI의 /guide도 지금 같은 방식)."""
    start, end = window.split(" ~ ")
    start_month, start_day = (int(x) for x in start.split("-"))
    end_month, end_day = (int(x) for x in end.split("-"))
    start_week, end_week = _week_label(start_day), _week_label(end_day)
    if start_month == end_month:
        return f"{start_month}월 {start_week}" if start_week == end_week else f"{start_month}월 {start_week}~{end_week}"
    return f"{start_month}월 {start_week}~{end_month}월 {end_week}"


def _skill_phrase(skill_trend: list[dict]) -> str:
    if not skill_trend:
        return ""
    top3 = skill_trend[:3]
    n = top3[0]["n_postings"] if top3 else 0
    return ", ".join(f"{s['skill']}({s['count']}/{n}건)" for s in top3)


def _template_basis(company_id: str, job_family: str, window_info: dict, peak_count: int, skill_trend: list[dict]) -> str:
    """LLM 호출 실패/키 없음 시 폴백용. LLM 버전이랑 같은 숫자를 쓰되 문장만 고정형."""
    season_phrase = ", ".join(_describe_window(w) for w in window_info["windows"])
    sentence = (
        f"{company_id} {job_family} 직무는 최근 채용 이력을 보면 {season_phrase}에 "
        f"채용공고가 몰리는 경향이 있어요. 최근 {window_info['n_observations']}건의 공고 중 "
        f"{peak_count}건이 이 시기에 나왔어요."
    )
    skills = _skill_phrase(skill_trend)
    if skills:
        sentence += f" 요구 스킬은 {skills} 순으로 자주 나왔어요."
    return sentence


def generate_basis_sentence(company_id: str, job_family: str, window_info: dict, peak_count: int,
                              skill_trend: list[dict], jd_evidence: list[dict] | None = None) -> str:
    """구조화된 통계(코드가 계산한 숫자)와, 벡터검색으로 찾은 실제 공고 발췌(jd_evidence)를
    LLM한테 넘겨서 자연스러운 문장으로 만든다. 숫자를 LLM이 새로 지어내지 않도록, 쓸 수
    있는 숫자/날짜/발췌를 프롬프트에 전부 명시하고 "이것만 써라"고 못박는다 - 근거(RAG
    grounding) 원칙: 검색/계산은 코드, 서술만 LLM.

    jd_evidence를 실제로 프롬프트에 넣는 게 핵심이다 - 이게 빠지면 화면에 "관련 공고
    내용"이 따로 보이더라도, 이 문장 자체는 그 근거를 전혀 반영 안 한 채 skill_trend
    숫자만 보고 쓴 게 된다(검색 따로, 생성 따로 - 진짜 RAG가 아님)."""
    client = _get_llm_client()
    if client is None:
        return _template_basis(company_id, job_family, window_info, peak_count, skill_trend)

    season_phrase = ", ".join(_describe_window(w) for w in window_info["windows"])
    skills = _skill_phrase(skill_trend)
    skill_line = f"자주 등장한 요구 스킬(태그): {skills}\n" if skills else ""
    evidence_line = ""
    if jd_evidence:
        excerpts = "\n".join(f"- {e['chunk_text']}" for e in jd_evidence)
        evidence_line = f"실제 과거 공고에서 발췌한 내용({len(jd_evidence)}건):\n{excerpts}\n"
    prompt = f"""아래 채용 시즌 분석 결과를 바탕으로, 사용자에게 보여줄 자연스러운 한국어 한두 문장을 만들어주세요.

회사: {company_id}
직무: {job_family}
관측된 공고 수: {window_info['n_observations']}건
채용이 몰리는 시기: {season_phrase}
이 시기에 발생한 공고 수: {peak_count}건
과거 실제 게시일 목록: {', '.join(window_info['observed_dates'])}
{skill_line}{evidence_line}
규칙:
- 위에 주어진 숫자·날짜·시기·스킬·발췌 외에 새로운 정보를 만들어내지 마세요.
- "예측", "확실히", "~% 확률" 같은 단정적 표현은 쓰지 말고 "경향이 있어요", "~인 편이에요" 처럼 표현하세요.
- 발췌가 있으면 "과거 공고 n건에서 이런 내용이 확인됐어요"처럼 근거로 자연스럽게 녹여서, 시즌 정보 문장까지 포함해 최대 세 문장으로 작성하세요. 발췌가 없으면 시즌+스킬 문장만 최대 두 문장으로 작성하세요.
- 친근하고 서비스스러운 톤으로 작성하세요. 따옴표나 설명 없이 문장만 출력하세요."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or _template_basis(company_id, job_family, window_info, peak_count, skill_trend)
    except Exception:
        # 네트워크 오류/쿼터 초과 등 - 서비스가 죽지 않게 템플릿으로 조용히 폴백
        return _template_basis(company_id, job_family, window_info, peak_count, skill_trend)


def _fetch_jd_evidence(company_id: str, job_family: str, skill_trend: list[dict]) -> list[dict]:
    """skill_trend(이미 계산된 실제 요구 스킬)를 벡터 쿼리로 재사용해 JD 근거를 가져온다.
    쿼리 없이 부르면 "최신 공고 하나의 섹션들"만 몰려나오는데(같은 posting_id의 여러
    섹션이 같은 날짜를 공유해서), skill_trend로 검색하면 여러 공고에 걸쳐 스킬 관련성
    높은 섹션들이 뽑힌다. pgvector DB 연결 안 되어있거나 진짜 JD 본문이 아직 없는
    경우 - 시즌분석/스킬트렌드는 이거 없이도 완성된 결과라 조용히 빈 리스트로
    폴백하되, 원인은 로그에 남긴다(그냥 삼켜버리면 DB/임베딩 장애를 못 알아챈다)."""
    try:
        from .jd_embeddings import get_jd_evidence
        skill_query = ", ".join(item["skill"] for item in skill_trend) or None
        return get_jd_evidence(company_id, job_family, query=skill_query)
    except Exception:
        logger.exception("jd_evidence 조회 실패 (company=%s, job_family=%s)", company_id, job_family)
        return []


def build_agent_output(df: pd.DataFrame, company_name: str, job_family: str) -> dict:
    company_id = resolve_company(company_name)
    window_info = compute_observed_window(df, company_id, job_family)

    if window_info["n_observations"] == 0:
        # 관측 데이터가 없는 회사/직무 조합 - 하위 로직이 기대하는 키를 다 채운
        # 빈 결과를 명시적으로 반환한다(키가 빠지면 호출부에서 KeyError가 난다).
        return {
            "company": company_id,
            "job_family": job_family,
            "n_postings_analyzed": 0,
            "observed_windows": [],
            "monthly_breakdown": [],
            "skill_trend": [],
            "jd_evidence": [],
            "basis": "관측된 공고 데이터가 없습니다.",
            "caveat": "이 회사/직무 조합은 아직 데이터가 수집되지 않았습니다.",
        }

    pattern = compute_hiring_pattern(df)
    pattern_subset = pattern[(pattern["company_id"] == company_id) & (pattern["job_family"] == job_family)]
    peak_count = int(pattern_subset["posting_count"].max()) if not pattern_subset.empty else 0
    skill_trend = compute_skill_trend(df, company_id, job_family)
    # jd_evidence를 basis 문장 생성보다 먼저 가져와야, generate_basis_sentence가
    # 실제 검색 결과를 프롬프트에 넣어서 쓸 수 있다(검색 따로 생성 따로면 RAG가 아님).
    jd_evidence = _fetch_jd_evidence(company_id, job_family, skill_trend)

    return {
        "company": company_id,
        "job_family": job_family,
        "n_postings_analyzed": window_info["n_observations"],
        "observed_windows": window_info["windows"],
        "observed_dates": window_info["observed_dates"],
        "monthly_breakdown": pattern_subset[["month", "posting_count", "seasonality_share"]].to_dict("records"),
        "skill_trend": skill_trend,
        "jd_evidence": jd_evidence,
        "basis": generate_basis_sentence(company_id, job_family, window_info, peak_count, skill_trend, jd_evidence),
        "caveat": "⚠️ 예측이 아닌 참고용 경향이에요",
    }


# =====================================================================
# FastAPI 엔드포인트에서 호출하는 진입점 - ①~⑤를 한 번에 실행
# =====================================================================

def get_hiring_season(company: str, job_family: str,
                        published_min: str = "20220101", published_max: str = "20261231") -> dict:
    """SARAMIN_ACCESS_KEY가 .env에 있으면 실제 API, 없으면(승인 전) mock으로 자동 전환된다
    (fetch_postings 자체 로직) - main.py는 이 함수 하나만 부르면 됨."""
    access_key = os.getenv("SARAMIN_ACCESS_KEY") or None
    response = fetch_postings(company, published_min, published_max, access_key=access_key)
    rows = parse_response_to_rows(response)
    df = build_raw_postings_df(rows)
    return build_agent_output(df, company, job_family)


# =====================================================================
# 데모 실행 (mock 데이터로 전체 파이프라인 검증)
# =====================================================================

if __name__ == "__main__":
    all_rows = []
    for keyword in ["삼성전자", "CJ ENM"]:
        response = fetch_postings(keyword, published_min="20220101", published_max="20251231")
        all_rows.extend(parse_response_to_rows(response))

    df = build_raw_postings_df(all_rows)

    print("=== ② raw_job_postings ===")
    print(df[["posting_id", "company_name_raw", "company_id", "job_family", "posted_date"]].to_string(index=False))

    print("\n=== ③ 표준화 확인 (원본 표기 -> company_id) ===")
    print(df[["company_name_raw", "company_id"]].drop_duplicates().to_string(index=False))

    print("\n=== ④ hiring_pattern (월별 집계) ===")
    print(compute_hiring_pattern(df).to_string(index=False))

    print("\n=== ⑤ 최종 출력 (오케스트레이터가 받는 형태) ===")
    import json
    output = build_agent_output(df, "삼성전자", "데이터")
    print(json.dumps(output, ensure_ascii=False, indent=2))
