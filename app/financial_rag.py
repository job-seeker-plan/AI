"""User-owned financial-context RAG for the cash-flow guide.

Only text explicitly supplied by the user is persisted.  If an embedding or
LLM credential is unavailable, the module uses the same stored user context in
recency order and returns a deterministic safety-first template instead.
"""

from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

from .hiring_agent.jd_embeddings import EMBEDDING_DIM, embed_text, embed_texts


# Local development starts AI separately from BE.  Docker supplies the
# PGVECTOR_* variables, while this loads BE/.env only when those are absent.
load_dotenv(Path(__file__).resolve().parents[2] / "BE" / ".env")


def _connection():
    database_url = os.getenv("DATABASE_URL", "").removeprefix("jdbc:")
    parsed = urlparse(database_url) if database_url else None
    return psycopg2.connect(
        host=os.getenv("PGVECTOR_HOST") or (parsed.hostname if parsed else "localhost"),
        port=int(os.getenv("PGVECTOR_PORT") or (parsed.port if parsed and parsed.port else 5432)),
        dbname=os.getenv("PGVECTOR_DB") or (parsed.path.lstrip("/") if parsed else "job_planner"),
        user=os.getenv("PGVECTOR_USER") or os.getenv("DATABASE_USERNAME", "job_planner"),
        password=os.getenv("PGVECTOR_PASSWORD") or os.getenv("DATABASE_PASSWORD", ""),
    )


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def save_contexts(user_id: str, contexts: list[dict]) -> int:
    """Upsert active user-entered context. Embeddings are optional at write time.

    Leaving the vector null is intentional when no OpenAI key exists: the text
    remains available for a privacy-preserving recency fallback, and can be
    re-saved after a key is configured to enable semantic retrieval.
    """
    valid = [item for item in contexts if item.get("text", "").strip()]
    if not valid:
        return 0
    embeddings: list[list[float] | None] = [None] * len(valid)
    if os.getenv("OPENAI_API_KEY"):
        try:
            embeddings = embed_texts([item["text"].strip() for item in valid])
        except Exception:
            # Context capture must not fail just because personalization is
            # temporarily unavailable.
            embeddings = [None] * len(valid)

    with closing(_connection()) as connection:
        with connection, connection.cursor() as cursor:
            for item, embedding in zip(valid, embeddings):
                cursor.execute(
                    """UPDATE user_context_embeddings SET is_active = false
                       WHERE user_id = %s AND data_type = %s AND related_category = %s AND is_active""",
                    (user_id, item.get("data_type", "onboarding"), item.get("related_category", "cashflow")),
                )
                cursor.execute(
                    """INSERT INTO user_context_embeddings
                       (user_id, text, embedding, data_type, related_category, emotion_tag, urgency_level, is_active)
                       VALUES (%s, %s, %s::vector, %s, %s, %s, %s, true)""",
                    (
                        user_id,
                        item["text"].strip(),
                        _vector_literal(embedding) if embedding else None,
                        item.get("data_type", "onboarding"),
                        item.get("related_category", "cashflow"),
                        item.get("emotion_tag"),
                        item.get("urgency_level", "normal"),
                    ),
                )
    return len(valid)


def list_contexts(user_id: str) -> list[dict]:
    """Active context rows for a user, for prefilling an edit UI. No embedding call needed."""
    with closing(_connection()) as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT text, data_type, related_category, emotion_tag, urgency_level FROM user_context_embeddings
               WHERE user_id = %s AND is_active ORDER BY created_at""",
            (user_id,),
        )
        return [
            {
                "text": row[0],
                "data_type": row[1],
                "related_category": row[2],
                "emotion_tag": row[3],
                "urgency_level": row[4],
            }
            for row in cursor.fetchall()
        ]


def retrieve_context(user_id: str, related_category: str, query: str, top_k: int = 3) -> list[str]:
    with closing(_connection()) as connection, connection.cursor() as cursor:
        embedding: list[float] | None = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                embedding = embed_text(query)
            except Exception:
                embedding = None
        if embedding:
            cursor.execute(
                """SELECT text FROM user_context_embeddings
                   WHERE user_id = %s AND is_active AND embedding IS NOT NULL
                     AND (related_category = %s OR related_category = 'cashflow')
                   ORDER BY embedding <=> %s::vector ASC LIMIT %s""",
                (user_id, related_category, _vector_literal(embedding), top_k),
            )
        else:
            cursor.execute(
                """SELECT text FROM user_context_embeddings
                   WHERE user_id = %s AND is_active
                     AND (related_category = %s OR related_category = 'cashflow')
                   ORDER BY created_at DESC LIMIT %s""",
                (user_id, related_category, top_k),
            )
        return [str(row[0]) for row in cursor.fetchall()]


EVENT_TYPE_LABELS = {
    "document_deadline": "서류 마감",
    "coding_test": "코딩테스트",
    "aptitude_test": "인적성",
    "interview": "면접",
    "language_test": "어학시험",
    "certificate": "자격증",
    "education": "교육",
    "lecture": "강의",
    "other": "기타",
}


def _describe_event(next_event: dict | None) -> str | None:
    if not next_event:
        return None
    label = EVENT_TYPE_LABELS.get(next_event.get("event_type", ""), next_event.get("event_type", ""))
    return f"{next_event.get('event_date')} {next_event.get('title')}({label})"


def _template(status: str, shortage_month: str | None, recommended_limit: int, contexts: list[str], next_event: dict | None = None) -> str:
    prefix = ""
    if contexts:
        prefix = f"기록해 둔 목표를 고려하면, {contexts[0]} "
    if status == "risk":
        body = f"{prefix}{shortage_month or '예상 기간'}에 자금 부족 가능성이 있어요. 월 지출을 {recommended_limit:,}원 이하로 먼저 조정하고, 확정된 취업 준비 비용과 지원금을 다시 확인해 보세요."
    elif status == "caution":
        body = f"{prefix}목표 취업월까지 여유가 크지 않아요. 이번 달에는 고정비와 면접·시험 비용을 먼저 남겨두고, 선택 지출은 한도를 정해 관리해 보세요."
    else:
        body = f"{prefix}현금흐름은 현재 안정권이에요. 취업 준비에 필요한 지출은 유지하되, 캘린더에 잡힌 비용이 늘어날 때만 예산을 다시 점검해 보세요."
    event_description = _describe_event(next_event)
    if event_description:
        body += f" 가장 가까운 일정은 {event_description}이니, 이 일정부터 먼저 준비하고 남는 예산으로 지출 계획을 세우세요."
    return body


def build_personalized_guide(payload: dict) -> dict:
    next_event = payload.get("next_event")
    contexts = retrieve_context(
        payload["user_id"], payload.get("related_category", "cashflow"),
        f"{payload['status']} 현금흐름, 예상 부족월 {payload.get('shortage_month') or '없음'}, 권장 한도 {payload['recommended_monthly_spend_limit']}",
    )
    fallback = _template(payload["status"], payload.get("shortage_month"), payload["recommended_monthly_spend_limit"], contexts, next_event)
    if not contexts or not os.getenv("OPENAI_API_KEY"):
        return {"guide": fallback, "personalized": bool(contexts), "context_count": len(contexts)}
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        event_description = _describe_event(next_event)
        event_line = f"\n가장 가까운 취업 일정: {event_description}" if event_description else ""
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=220,
            messages=[
                {"role": "system", "content": "당신은 취업준비생의 금융비서입니다. 의료·법률·투자 조언은 하지 말고, 제공된 숫자와 사용자 메모, 일정에만 근거해 2~4문장의 실행 가능한 한국어 가이드를 작성하세요. 가장 가까운 취업 일정이 주어지면 '왜 이 일정부터 챙겨야 하는지'와 '왜 지금 지출을 줄여야 하는지'를 하나의 흐름으로 같이 설명하세요. 단정하거나 수치·사실을 지어내지 마세요."},
                {"role": "user", "content": f"상태: {payload['status']}\n부족 예상월: {payload.get('shortage_month') or '없음'}\n권장 월 지출 한도: {payload['recommended_monthly_spend_limit']:,}원\n사용자 메모: {' | '.join(contexts)}{event_line}"},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
        return {"guide": text or fallback, "personalized": True, "context_count": len(contexts)}
    except Exception:
        return {"guide": fallback, "personalized": True, "context_count": len(contexts)}
