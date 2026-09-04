"""Gmail 채용 메일 -> 캘린더 일정 추출.

BE(Gmail API 연동, OAuth 토큰 관리)와 AI(LLM 파싱)를 분리한다 - 이 앱의 다른 기능들과
동일하게 "LLM 호출은 전부 AI 서비스가 담당" 원칙을 따른다. BE는 정규화된 이메일
{message_id, subject, sender, date, body}만 넘기고, Gmail 고유의 인증/조회 방식은
전혀 몰라도 된다.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta

from openai import OpenAI

# BE의 JobEventType과 반드시 동기화 - 여기 없는 값을 내려주면 BE에서 역직렬화 실패로
# 이어진다(실제로 겪은 버그: 잘못된 enum 값이 400 대신 401로 오분류됐던 것과 별개 이슈).
VALID_EVENT_TYPES = {
    "document_deadline", "coding_test", "aptitude_test", "interview",
    "language_test", "certificate", "education", "lecture", "other",
}

SYSTEM_PROMPT = """당신은 한국 취업준비생을 위한 채용 이메일 파서입니다.
주어진 이메일 목록에서 채용 관련 일정을 추출하여 JSON으로만 응답하세요.

event_type 값은 반드시 아래 중 하나만 사용하세요:
document_deadline(서류 마감/지원), coding_test(코딩테스트),
aptitude_test(인적성/GSAT/NCS/AI역량), interview(면접),
language_test(어학시험), certificate(자격증), education(교육), lecture(강의), other(기타)

응답 형식(반드시 이 JSON만 반환):
{"events":[{"message_id":"입력에 주어진 그대로","title":"회사명 + 일정종류(예: 카카오 코딩테스트)","event_type":"coding_test","event_date":"YYYY-MM-DD 또는 null","memo":"이메일 한줄 요약"}]}

규칙:
- 채용과 무관한 이메일은 events에 포함하지 마세요
- 날짜를 특정할 수 없으면 event_date를 null로 두세요
- title은 반드시 한국어로 작성하세요
- message_id는 절대 새로 만들거나 바꾸지 말고, 입력에 준 값을 그대로 반환하세요
"""

_client: OpenAI | None = None


def _get_client() -> OpenAI | None:
    global _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if _client is None:
        _client = OpenAI(api_key=api_key)
    return _client


def extract_email_events(messages: list[dict]) -> list[dict]:
    """messages: [{message_id, subject, sender, date, body}]
    returns: [{message_id, title, event_type, event_date, memo}]

    GPT가 우선, 키 없거나 호출 실패 시 키워드+정규식 폴백 - 이 서비스의 다른 기능들과
    동일한 안전망 원칙(financial_rag/hiring_pattern_pipeline과 동일 패턴)."""
    if not messages:
        return []
    client = _get_client()
    if client is None:
        events = _fallback_parse(messages)
    else:
        try:
            events = _parse_with_gpt(client, messages)
        except Exception:
            events = _fallback_parse(messages)
    return _merge_duplicate_events(events)


def _merge_duplicate_events(events: list[dict]) -> list[dict]:
    """같은 회사가 같은 시험/면접을 두고 안내·초대·상세일정 메일을 따로따로 보내는
    경우가 흔해서(예: SK텔레콤이 코딩테스트 하나로 메일 4통), (제목, 유형, 날짜)가
    같은 이벤트는 하나로 합친다. message_id는 쉼표로 이어붙여서 - 합쳐진 항목을
    사용자가 실제로 추가하면 원본 메일 전부가 "이미 가져옴" 처리되게 한다(이거 없이
    하나만 저장하면 나머지 메일들이 다음 새로고침 때 다시 후보로 뜬다)."""
    groups: dict[tuple[str, str, str], dict] = {}
    order: list[tuple[str, str, str]] = []
    for event in events:
        key = (event["title"], event["event_type"], event["event_date"])
        if key not in groups:
            groups[key] = {**event, "message_ids": [event["message_id"]]}
            order.append(key)
        else:
            groups[key]["message_ids"].append(event["message_id"])

    merged = []
    for key in order:
        group = groups[key]
        message_ids = group["message_ids"]
        memo = group["memo"]
        if len(message_ids) > 1:
            memo = f"관련 메일 {len(message_ids)}건" + (f" · {memo}" if memo else "")
        merged.append({
            "message_id": ",".join(message_ids),
            "title": group["title"],
            "event_type": group["event_type"],
            "event_date": group["event_date"],
            "memo": memo,
        })
    return merged


def _parse_with_gpt(client: OpenAI, messages: list[dict]) -> list[dict]:
    emails_text = "\n\n".join(
        f"[이메일 message_id={m['message_id']}]\n"
        f"제목: {m.get('subject', '')}\n"
        f"발신: {m.get('sender', '')}\n"
        f"날짜: {m.get('date', '')}\n"
        f"본문: {(m.get('body') or '')[:2000]}"
        for m in messages
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"다음 이메일들에서 채용 일정을 추출해주세요:\n\n{emails_text}"},
        ],
    )
    content = completion.choices[0].message.content or "{}"
    parsed = json.loads(content)
    valid_ids = {m["message_id"] for m in messages}
    events = []
    cutoff = date.today() - timedelta(days=7)
    for item in parsed.get("events", []):
        event = _normalize_event(item, valid_ids, cutoff)
        if event:
            events.append(event)
    return events


def _normalize_event(item: dict, valid_ids: set[str], cutoff: date) -> dict | None:
    message_id = item.get("message_id")
    title = (item.get("title") or "").strip()
    event_type = item.get("event_type") if item.get("event_type") in VALID_EVENT_TYPES else "other"
    event_date = item.get("event_date")
    if message_id not in valid_ids or not title or not event_date or event_date == "null":
        return None
    try:
        parsed_date = datetime.strptime(event_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    if parsed_date < cutoff:
        return None
    return {
        "message_id": message_id,
        "title": title,
        "event_type": event_type,
        "event_date": event_date,
        "memo": (item.get("memo") or "").strip(),
    }


# =====================================================================
# 키워드+정규식 폴백 (OPENAI_API_KEY 없거나 GPT 호출 실패 시) - 본문까지 포함해서 검색
# =====================================================================

_TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("coding_test", ("코딩테스트", "코딩 테스트", "coding test", "온라인테스트", "온라인 테스트")),
    ("aptitude_test", ("인적성", "적성검사", "gsat", "ncs", "ai역량", "역량검사")),
    ("language_test", ("어학", "toeic", "토익", "opic", "toefl")),
    ("interview", ("면접", "interview")),
    ("document_deadline", ("서류", "서류전형", "서류 전형", "지원 마감", "서류 마감", "지원서")),
    ("education", ("교육", "부트캠프")),
    ("lecture", ("특강", "강의")),
    ("certificate", ("자격증", "자격 시험")),
]

_KO_DATE_FULL = re.compile(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일")
_ISO_DATE = re.compile(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})")
_KO_DATE_SHORT = re.compile(r"(\d{1,2})월\s*(\d{1,2})일")


def _detect_type(text: str) -> str:
    lower = text.lower()
    for event_type, keywords in _TYPE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return event_type
    return "other"


def _extract_date(text: str) -> date | None:
    match = _KO_DATE_FULL.search(text)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _ISO_DATE.search(text)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _KO_DATE_SHORT.search(text)
    if match:
        month, day = int(match.group(1)), int(match.group(2))
        today = date.today()
        try:
            candidate = today.replace(month=month, day=day)
        except ValueError:
            return None
        return candidate if candidate >= today else candidate.replace(year=today.year + 1)
    return None


def _extract_sender_name(sender: str) -> str:
    if not sender:
        return ""
    match = re.match(r"^\s*\"?([^\"<]+)\"?\s*<", sender)
    if match:
        return match.group(1).strip()
    at_index = sender.find("@")
    if at_index > 0:
        domain = sender[at_index + 1:]
        return domain.split(".")[0]
    return ""


_TYPE_LABELS = {
    "document_deadline": "서류 마감", "coding_test": "코딩테스트", "aptitude_test": "인적성검사",
    "interview": "면접", "language_test": "어학시험", "certificate": "자격증 시험",
    "education": "교육", "lecture": "강의", "other": "채용 일정",
}


def _fallback_parse(messages: list[dict]) -> list[dict]:
    cutoff = date.today() - timedelta(days=7)
    results = []
    for message in messages:
        subject = message.get("subject", "")
        body = message.get("body", "")
        if not subject:
            continue
        searchable = f"{subject}\n{body}"
        event_type = _detect_type(searchable)
        event_date = _extract_date(searchable)
        if event_date is None or event_date < cutoff:
            continue
        company = _extract_sender_name(message.get("sender", ""))
        title = f"{company} {_TYPE_LABELS[event_type]}" if company else subject.strip()
        results.append({
            "message_id": message["message_id"],
            "title": title,
            "event_type": event_type,
            "event_date": event_date.isoformat(),
            "memo": f"이메일: {subject.strip()}",
        })
    return results
