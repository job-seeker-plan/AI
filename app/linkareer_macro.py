"""Small, bounded browser collector for Linkareer public recruitment listings.

It does not sign in, solve CAPTCHAs, retain cookies, or access non-public
pages. It reads one requested public-results page at a time (up to 20 jobs) and
keeps each page in memory for five minutes to avoid repeated page loads.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright


SOURCE_URL = "https://linkareer.com/list/recruit"
MAX_LIMIT = 20
# A broad/empty search can match thousands of postings on Linkareer. Each page
# costs a full headless-browser launch (see _load_public_listing), and only one
# of these crawls can run at a time (_browser_lock), so an unbounded page count
# here previously let one search monopolize the lock for minutes while every
# other user's request queued behind it and timed out upstream (BE's AiClient
# read timeout is far shorter than that). Keep this small enough that a worst
# case crawl finishes within that timeout.
MAX_PAGES = 5
CACHE_TTL = timedelta(minutes=5)
_cache: dict[str, tuple[datetime, list[dict[str, Any]], int]] = {}
_browser_lock = asyncio.Lock()


async def collect_recruitments(
    keyword: str,
    category_id: str | None,
    region_id: str | None,
    job_type: str | None,
    page: int,
    limit: int,
    region_name: str | None = None,
    experience: str | None = None,
    deadline_within_days: int | None = None,
) -> tuple[list[dict[str, Any]], str, int, bool]:
    normalized_keyword = keyword.strip().casefold()
    effective_limit = min(max(limit, 1), MAX_LIMIT)
    effective_page = max(page, 1)
    source_url = _source_url(keyword, category_id, region_id, job_type, effective_page)
    cache_key = "|".join([
        keyword.strip(), category_id or "", region_id or "", job_type or "",
        region_name or "", experience or "", str(deadline_within_days or ""),
    ])
    cached = _cache.get(cache_key)
    now = datetime.now(UTC)
    if cached and now - cached[0] < CACHE_TTL:
        start = (effective_page - 1) * effective_limit
        return cached[1][start:start + effective_limit], source_url, cached[2], True

    # A single browser page at a time prevents an accidental burst of requests.
    async with _browser_lock:
        cached = _cache.get(cache_key)
        now = datetime.now(UTC)
        if cached and now - cached[0] < CACHE_TTL:
            return cached[1][:effective_limit], source_url, cached[2], True
        query_urls = [
            _source_url(search_keyword, category_id, region_id, job_type, 1)
            for search_keyword in _search_variants(normalized_keyword)
        ] if normalized_keyword else [_source_url("", category_id, region_id, job_type, 1)]
        all_jobs: dict[str, dict[str, Any]] = {}
        for query_url in query_urls:
            for job in await _load_all_pages(query_url):
                all_jobs[job["id"]] = job
        jobs = [
            job for job in all_jobs.values()
            if (not normalized_keyword or _job_contains_keyword(job, normalized_keyword))
            and _region_matches(job, region_name)
            and _experience_matches(job, experience)
            and _deadline_matches(job, deadline_within_days)
        ]
        jobs.sort(key=lambda job: (job.get("deadline", ""), job.get("id", "")))
        total_count = len(jobs)
        _cache[cache_key] = (now, jobs, total_count)
        start = (effective_page - 1) * effective_limit
        return jobs[start:start + effective_limit], source_url, total_count, False


def _source_url(keyword: str, category_id: str | None, region_id: str | None, job_type: str | None, page: int) -> str:
    query: dict[str, str] = {}
    if keyword.strip():
        # Linkareer treats a space-separated query as one exact phrase. Use the
        # first term for the upstream request, then apply the broader contains
        # matching in _jobs_from_apollo so related titles are not lost.
        query["filterBy_q"] = keyword.strip().split()[0]
    if category_id:
        query["filterBy_categoryIDs"] = category_id
    if region_id:
        query["filterBy_regionIDs"] = region_id
    if job_type:
        query["filterBy_jobTypes"] = job_type
    if page > 1:
        query["page"] = str(page)
    return SOURCE_URL if not query else f"{SOURCE_URL}?{urlencode(query)}"


async def _load_public_listing(source_url: str, keyword: str) -> tuple[list[dict[str, Any]], int]:
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                # Linkareer keeps some third-party resources open for a long
                # time. The server-rendered Next data is available as soon as
                # the document response starts, so do not wait for the whole
                # page's DOMContentLoaded event.
                await page.goto(source_url, wait_until="commit", timeout=15_000)
                payload = await page.locator("script#__NEXT_DATA__").text_content(timeout=10_000)
            finally:
                await browser.close()
    except PlaywrightTimeoutError as error:
        raise RuntimeError("링커리어 목록을 시간 내에 불러오지 못했습니다.") from error
    except Exception as error:
        raise RuntimeError("링커리어 공개 채용 목록을 불러오지 못했습니다.") from error

    if not payload:
        raise RuntimeError("링커리어 목록 데이터가 비어 있습니다.")
    try:
        page_props = json.loads(payload)["props"]["pageProps"]
        apollo_state = page_props.get("__APOLLO_STATE__") or page_props.get("apolloState")
        if not isinstance(apollo_state, dict):
            raise KeyError("Apollo state")
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("링커리어 목록 데이터 형식이 변경되었습니다.") from error

    return _jobs_from_apollo(apollo_state, keyword)


async def _load_all_pages(source_url: str) -> list[dict[str, Any]]:
    first_page_jobs, total_count = await _load_public_listing(source_url, "")
    jobs = list(first_page_jobs)
    page_size = max(len(first_page_jobs), MAX_LIMIT)
    total_pages = min(MAX_PAGES, max(1, (total_count + page_size - 1) // page_size))
    for page_number in range(2, total_pages + 1):
        page_url = f"{source_url}&page={page_number}" if "?" in source_url else f"{source_url}?page={page_number}"
        page_jobs, _ = await _load_public_listing(page_url, "")
        jobs.extend(page_jobs)
        if not page_jobs:
            break
    return jobs


def _region_matches(job: dict[str, Any], region_name: str | None) -> bool:
    if not region_name:
        return True
    locations = {str(value).strip() for value in job.get("locations", [])}
    if region_name in locations:
        return True
    grouped = {
        "충북": "충청", "충남": "충청",
        "전북": "전라", "전남": "전라",
        "경북": "경상", "경남": "경상",
    }
    return grouped.get(region_name, region_name) in locations


def _experience_matches(job: dict[str, Any], experience: str | None) -> bool:
    if not experience or experience == "any":
        return True
    terms = {
        "entry": "신입", "experienced": "경력",
        "intern": "인턴", "contract": "계약",
    }
    return terms.get(experience, experience) in str(job.get("employment_type", ""))


def _deadline_matches(job: dict[str, Any], deadline_within_days: int | None) -> bool:
    if not deadline_within_days:
        return True
    deadline = job.get("deadline", "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(deadline)):
        return False
    try:
        days_until_deadline = (datetime.fromisoformat(deadline).date() - datetime.now().date()).days
    except ValueError:
        return False
    return 0 <= days_until_deadline <= deadline_within_days


def _search_variants(keyword: str) -> list[str]:
    variants = [keyword]
    # Linkareer treats these as separate exact search phrases. Include the
    # common compound form so "프론트" also finds "프론트엔드" listings.
    if keyword == "프론트":
        variants.append("프론트엔드")
    return variants


async def _load_matching_listings(base_url: str, filtered_urls: list[str], keyword: str, requested_page: int, limit: int) -> tuple[list[dict[str, Any]], int]:
    matches: list[dict[str, Any]] = []
    # Keep Linkareer's own result set as a fallback. It is phrase-oriented, so
    # it is not sufficient by itself for contains matching, but it prevents a
    # valid search from becoming empty when the broad scan has not reached the
    # relevant page yet.
    for filtered_url in filtered_urls:
        filtered_jobs, _ = await _load_public_listing(filtered_url, "")
        existing_ids = {job["id"] for job in matches}
        matches.extend(job for job in filtered_jobs if job["id"] not in existing_ids and _job_contains_keyword(job, keyword))
    # Fetch a bounded number of unfiltered pages, then apply contains matching
    # locally. This prevents Linkareer's exact phrase search from dropping
    # "프론트엔드" when the user searches for "프론트".
    for source_page in range(1, max(5, requested_page) + 1):
        page_url = base_url if source_page == 1 else f"{base_url}&page={source_page}" if "?" in base_url else f"{base_url}?page={source_page}"
        jobs, _ = await _load_public_listing(page_url, "")
        existing_ids = {job["id"] for job in matches}
        matches.extend(job for job in jobs if job["id"] not in existing_ids and _job_contains_keyword(job, keyword))
        if len(matches) >= requested_page * limit:
            break
    start = (requested_page - 1) * limit
    return matches[start:start + limit], len(matches)


def _job_contains_keyword(job: dict[str, Any], keyword: str) -> bool:
    searchable = " ".join([
        str(job.get("title", "")), str(job.get("company", "")),
        *(str(value) for value in job.get("categories", [])),
        *(str(value) for value in job.get("locations", [])),
    ])
    return _contains_keyword(searchable, keyword)


def _jobs_from_apollo(state: dict[str, Any], keyword: str) -> tuple[list[dict[str, Any]], int]:
    root = state.get("ROOT_QUERY", {})
    activity_refs: list[str] = []
    total_count = 0
    for key, value in root.items():
        if key.startswith("activities(") and isinstance(value, dict):
            activity_refs = [item["__ref"] for item in value.get("nodes", []) if isinstance(item, dict) and "__ref" in item]
            total_count = int(value.get("totalCount", 0))
            break

    jobs: list[dict[str, Any]] = []
    for reference in activity_refs:
        activity = state.get(reference, {})
        if not isinstance(activity, dict):
            continue
        categories = _reference_names(state, activity.get("categories", []))
        locations = _reference_names(state, activity.get("regions", []))
        title = str(activity.get("title", "")).strip()
        company = str(activity.get("organizationName", "")).strip()
        searchable = " ".join([title, company, *categories, *locations]).casefold()
        if keyword and not _contains_keyword(searchable, keyword):
            continue
        activity_id = str(activity.get("id", "")).strip()
        if not activity_id or not title:
            continue
        jobs.append({
            "id": activity_id,
            "title": title,
            "company": company or "회사명 미표기",
            "categories": categories,
            "locations": locations,
            "employment_type": _employment_type(activity),
            "deadline": _deadline(activity.get("recruitCloseAt")),
            "url": f"https://linkareer.com/activity/{activity_id}",
        })
    return jobs, total_count


def _contains_keyword(searchable: str, keyword: str) -> bool:
    """Match any meaningful search term anywhere in the parsed job fields.

    The upstream listing search is phrase-oriented, while the UI search is
    intended to find jobs containing a keyword. Splitting the query prevents
    inputs such as '백엔드 개발' from requiring that exact phrase to appear
    contiguously in a title.
    """
    normalized = re.sub(r"\s+", "", searchable.casefold())
    terms = [re.sub(r"\s+", "", term.casefold()) for term in re.split(r"\s+", keyword.strip()) if term.strip()]
    return bool(terms) and any(term in normalized for term in terms)


def _reference_names(state: dict[str, Any], references: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for reference in references:
        value = state.get(reference.get("__ref", ""), reference)
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        # Linkareer assigns a child category called "전체" to many jobs.  The
        # useful classification is its parent (for example IT/개발), so include
        # that path and omit the non-informative child label.
        if isinstance(name, str) and name and name != "전체" and name not in names:
            names.append(name)
        parent = value.get("parent")
        if isinstance(parent, dict) and parent.get("__ref"):
            for parent_name in _reference_names(state, [parent]):
                if parent_name not in names:
                    names.append(parent_name)
    return names


def _employment_type(activity: dict[str, Any]) -> str:
    labels = {"NEW": "신입", "EXPERIENCED": "경력직", "CONTRACT": "계약직", "INTERN": "인턴"}
    values = [labels.get(value, value) for value in activity.get("jobTypes", [])]
    return " · ".join(values) or "채용 형태 원문 확인"


def _deadline(timestamp: Any) -> str:
    if not isinstance(timestamp, (int, float)):
        return "채용 시 마감 또는 원문 확인"
    return datetime.fromtimestamp(timestamp / 1000, tz=UTC).astimezone().date().isoformat()
