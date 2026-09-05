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
CACHE_TTL = timedelta(minutes=5)
_cache: dict[str, tuple[datetime, list[dict[str, Any]], int]] = {}
_browser_lock = asyncio.Lock()


async def collect_recruitments(keyword: str, category_id: str | None, region_id: str | None, job_type: str | None, page: int, limit: int) -> tuple[list[dict[str, Any]], str, int, bool]:
    normalized_keyword = keyword.strip().casefold()
    effective_limit = min(max(limit, 1), MAX_LIMIT)
    effective_page = max(page, 1)
    source_url = _source_url(keyword, category_id, region_id, job_type, effective_page)
    cache_key = f"{source_url}|contains-search"
    cached = _cache.get(cache_key)
    now = datetime.now(UTC)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1][:effective_limit], source_url, cached[2], True

    # A single browser page at a time prevents an accidental burst of requests.
    async with _browser_lock:
        cached = _cache.get(cache_key)
        now = datetime.now(UTC)
        if cached and now - cached[0] < CACHE_TTL:
            return cached[1][:effective_limit], source_url, cached[2], True
        if normalized_keyword:
            jobs, total_count = await _load_matching_listings(
                _source_url("", category_id, region_id, job_type, 1),
                [
                    _source_url(search_keyword, category_id, region_id, job_type, 1)
                    for search_keyword in _search_variants(normalized_keyword)
                ],
                normalized_keyword,
                effective_page,
                effective_limit,
            )
        else:
            jobs, total_count = await _load_public_listing(source_url, "")
        _cache[cache_key] = (now, jobs, total_count)
        return jobs[:effective_limit], source_url, total_count, False


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
