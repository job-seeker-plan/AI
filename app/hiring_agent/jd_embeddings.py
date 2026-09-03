"""
JD(채용공고 본문) RAG - OpenAI 임베딩(text-embedding-3-small) 사용.

기본 1536차원을 dimensions=384로 잘라서 요청 - job_jd_embeddings 테이블의
VECTOR(384)와 맞추고, 가계부AI RAG의 user_context_embeddings 테이블도 같은
384차원이라 두 테이블이 같은 pgvector 확장을 공유해도 임베딩 계열이 다르지 않게 통일.

DB 연결 정보는 지금은 로컬 테스트용 pgvector 컨테이너(포트 5433)를 가리킨다.
나중에 BE 공용 DB로 옮기면 DATABASE_* 환경변수만 바꾸면 됨(코드 수정 불필요).
"""
from __future__ import annotations

import os
import re

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =====================================================================
# 원문 -> 조각(chunk) 분할 - 채용공고 원문은 [담당업무]/[자격요건]/[우대사항] 같은
# 섹션으로 나뉘어 있는 게 보통이라, 그 섹션 단위로 쪼개서 각각 따로 임베딩한다.
# job_jd_embeddings 테이블이 posting_id에 유니크 제약을 안 걸어둔 것도 원래 이렇게
# 한 공고당 여러 행(조각)이 생기는 걸 전제로 설계된 것.
# =====================================================================

_SECTION_HEADER_PATTERN = re.compile(r"\[(.+?)\]")


def chunk_jd_text(raw_text: str) -> list[str]:
    """"[자격요건]\\nSQL...\\n\\n[우대사항]\\nTableau..." 형태의 원문을
    ["자격요건: SQL...", "우대사항: Tableau..."] 처럼 섹션별 조각으로 쪼갠다.
    헤더가 하나도 없는 원문이면 통째로 하나의 조각으로 취급한다(크롤링 결과가
    항상 섹션 구조를 갖는 건 아닐 수 있으므로)."""
    text = raw_text.strip()
    if not text:
        return []
    parts = _SECTION_HEADER_PATTERN.split(text)
    if len(parts) < 3:
        return [text]
    chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            chunks.append(f"{header}: {body}")
    return chunks

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 384

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 .env에 없습니다 - AI/.env 확인할 것")
        _client = OpenAI(api_key=api_key)
    return _client


def embed_text(text: str) -> list[float]:
    """텍스트 하나를 384차원 벡터로 변환."""
    response = _get_client().embeddings.create(
        model=EMBEDDING_MODEL, input=text, dimensions=EMBEDDING_DIM
    )
    return response.data[0].embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """여러 개를 한 번의 API 호출로 변환(건별 호출보다 빠르고 저렴)."""
    response = _get_client().embeddings.create(
        model=EMBEDDING_MODEL, input=texts, dimensions=EMBEDDING_DIM
    )
    return [item.embedding for item in response.data]


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PGVECTOR_HOST", "localhost"),
        port=int(os.getenv("PGVECTOR_PORT", "5433")),
        dbname=os.getenv("PGVECTOR_DB", "jobtest"),
        user=os.getenv("PGVECTOR_USER", "postgres"),
        password=os.getenv("PGVECTOR_PASSWORD", "test1234"),
    )


def store_jd_chunks(chunks: list[dict]) -> None:
    """chunks: [{posting_id, company_id, job_family, posted_date, chunk_text}, ...]"""
    texts = [c["chunk_text"] for c in chunks]
    embeddings = embed_texts(texts)

    conn = get_connection()
    cur = conn.cursor()
    for chunk, embedding in zip(chunks, embeddings):
        cur.execute(
            """INSERT INTO job_jd_embeddings (posting_id, company_id, job_family, posted_date, chunk_text, embedding)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (chunk["posting_id"], chunk["company_id"], chunk["job_family"],
             chunk["posted_date"], chunk["chunk_text"], embedding),
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"{len(chunks)}건 저장 완료")


def search_jd_chunks(company_id: str, job_family: str, query: str | None = None, top_k: int = 3) -> list[dict]:
    """company_id/job_family로 먼저 필터링 후, query가 있으면 유사도 검색으로 top_k만 추림.
    query가 없으면(예: 회사 클릭만 한 경우) 최신순으로 top_k 반환 - 굳이 임베딩 검색 안 함."""
    conn = get_connection()
    cur = conn.cursor()

    if query:
        query_embedding = embed_text(query)
        cur.execute(
            """SELECT chunk_text, posted_date, embedding <=> %s::vector AS distance
               FROM job_jd_embeddings
               WHERE company_id = %s AND job_family = %s
               ORDER BY distance ASC LIMIT %s""",
            (query_embedding, company_id, job_family, top_k),
        )
        rows = [{"chunk_text": r[0], "posted_date": str(r[1]), "distance": float(r[2])} for r in cur.fetchall()]
    else:
        cur.execute(
            """SELECT chunk_text, posted_date FROM job_jd_embeddings
               WHERE company_id = %s AND job_family = %s
               ORDER BY posted_date DESC LIMIT %s""",
            (company_id, job_family, top_k),
        )
        rows = [{"chunk_text": r[0], "posted_date": str(r[1]), "distance": None} for r in cur.fetchall()]

    cur.close()
    conn.close()
    return rows


# =====================================================================
# 목업용 시딩 - 진짜 JD 본문 크롤링은 robots.txt 미해결이라, 사람인 승인 전까지는
# hiring_pattern_pipeline.py의 mock 회사에 맞춰 가짜 본문을 씀.
# =====================================================================

# 공고 한 건의 원문(raw_text) - [담당업무]/[자격요건]/[우대사항] 섹션으로 구성해서
# 실제 채용공고 본문 구조를 흉내냈다. chunk_jd_text()가 이 섹션 단위로 쪼갠다.
MOCK_JD_POSTINGS: dict[tuple[str, str], list[dict]] = {
    ("삼성전자", "데이터"): [
        {"posting_id": "mock-jd-1", "posted_date": "2025-03-11", "raw_text": """
[담당업무]
데이터 기반 사업 인사이트 도출 및 대시보드 구축, 유관부서 분석 요청 대응 업무를 수행합니다.
[자격요건]
SQL 및 Python을 활용한 데이터 분석 경험자 우대. 통계 기반 문제해결 능력 필요.
[우대사항]
Tableau 등 데이터 시각화 도구 활용 경험자 우대.
"""},
        {"posting_id": "mock-jd-2", "posted_date": "2024-03-05", "raw_text": """
[담당업무]
사내 데이터 파이프라인 구축 및 운영, 데이터 품질 관리 업무를 담당합니다.
[자격요건]
데이터베이스(RDB) 다뤄본 경험, Python 데이터 처리 라이브러리(pandas 등) 활용 가능자.
[우대사항]
클라우드 환경에서의 데이터 처리 경험자 우대.
"""},
        {"posting_id": "mock-jd-3", "posted_date": "2023-09-04", "raw_text": """
[담당업무]
대용량 데이터 파이프라인 설계 및 머신러닝 기반 예측 모델 운영을 지원합니다.
[자격요건]
머신러닝 기초 이해, 대용량 데이터 파이프라인 구축 경험 우대.
[우대사항]
통계학·산업공학 등 관련 전공자 우대.
"""},
    ],
    ("CJ ENM", "콘텐츠"): [
        {"posting_id": "mock-jd-4", "posted_date": "2024-09-02", "raw_text": """
[담당업무]
콘텐츠 기획 및 트렌드 분석 기반 신규 콘텐츠 발굴 업무를 수행합니다.
[자격요건]
콘텐츠 기획 및 트렌드 분석 역량, 엔터테인먼트 산업 이해도 필요.
[우대사항]
커뮤니케이션 역량 우수자 우대.
"""},
        {"posting_id": "mock-jd-5", "posted_date": "2023-08-28", "raw_text": """
[담당업무]
SNS 채널 운영 및 콘텐츠 트렌드 모니터링 업무를 담당합니다.
[자격요건]
SNS 채널 운영 경험, 대중과의 커뮤니케이션 역량 우대.
[우대사항]
콘텐츠 기획 유관 경험자 우대.
"""},
    ],
    ("CJ대한통운", "물류"): [
        {"posting_id": "mock-jd-6", "posted_date": "2023-12-04", "raw_text": """
[담당업무]
SCM 데이터 분석 및 ERP 시스템 기반 물류 프로세스 개선 업무를 수행합니다.
[자격요건]
SCM 데이터 분석 및 ERP 시스템 운영 경험자 우대. 물류 프로세스 개선 역량 필요.
[우대사항]
데이터 기반 의사결정 경험자 우대.
"""},
        {"posting_id": "mock-jd-7", "posted_date": "2024-06-10", "raw_text": """
[담당업무]
재고관리 및 물류 운영 실무, ERP 시스템을 활용한 데이터 관리를 담당합니다.
[자격요건]
재고관리 및 물류 운영 실무 경험, 엑셀·ERP 활용 가능자.
[우대사항]
SCM 관련 자격증 소지자 우대.
"""},
    ],
    ("CJ제일제당", "생산관리"): [
        {"posting_id": "mock-jd-8", "posted_date": "2024-02-05", "raw_text": """
[담당업무]
식품 생산 공정 관리 및 ERP 기반 생산관리 시스템 운영 업무를 수행합니다.
[자격요건]
식품 생산 공정 이해 및 품질관리 경험자 우대. ERP 기반 생산관리 시스템 활용 가능자.
[우대사항]
식품위생 관련 자격증 소지자 우대.
"""},
        {"posting_id": "mock-jd-9", "posted_date": "2023-02-13", "raw_text": """
[담당업무]
생산 공정개선 프로젝트 기획 및 실행 업무를 담당합니다.
[자격요건]
공정개선 프로젝트 참여 경험, 식품위생 관련 지식 보유자 우대.
[우대사항]
품질관리 관련 자격증 소지자 우대.
"""},
    ],
    ("CJ올리브영", "MD"): [
        {"posting_id": "mock-jd-10", "posted_date": "2024-11-04", "raw_text": """
[담당업무]
상품기획 및 트렌드 분석 기반 신규 상품 소싱 업무를 수행합니다.
[자격요건]
상품기획 및 트렌드 분석 역량, 데이터 기반 의사결정 경험자 우대.
[우대사항]
헬스&뷰티 산업 이해도 있는 자 우대.
"""},
        {"posting_id": "mock-jd-11", "posted_date": "2023-11-13", "raw_text": """
[담당업무]
리테일 MD 실무 지원 및 매출 데이터 정리·분석 업무를 담당합니다.
[자격요건]
리테일 MD 실무 또는 인턴 경험, 엑셀 데이터 정리·분석 가능자.
[우대사항]
상품기획 유관 경험자 우대.
"""},
    ],
    ("삼성SDS", "IT컨설팅"): [
        {"posting_id": "mock-jd-12", "posted_date": "2024-03-06", "raw_text": """
[담당업무]
고객사 IT 시스템 구축 컨설팅 및 클라우드 전환 프로젝트를 수행합니다.
[자격요건]
클라우드(AWS/Azure) 환경 이해 및 SAP 등 ERP 컨설팅 경험자 우대.
[우대사항]
SQL 활용 가능자 우대.
"""},
        {"posting_id": "mock-jd-13", "posted_date": "2023-03-13", "raw_text": """
[담당업무]
IT 시스템 구축 프로젝트 참여 및 요구사항 분석 업무를 담당합니다.
[자격요건]
SQL 활용 가능자, IT 시스템 구축 프로젝트 참여 경험 우대.
[우대사항]
SAP 등 ERP 시스템 이해도 있는 자 우대.
"""},
    ],
    ("카카오", "서비스기획"): [
        {"posting_id": "mock-jd-14", "posted_date": "2024-08-05", "raw_text": """
[담당업무]
데이터 기반 서비스 기획 및 지표 모니터링 업무를 수행합니다.
[자격요건]
데이터 기반 서비스 기획 경험, SQL로 지표 직접 조회 가능한 수준 우대.
[우대사항]
UX 리서치 경험자 우대.
"""},
        {"posting_id": "mock-jd-15", "posted_date": "2023-07-31", "raw_text": """
[담당업무]
사용자 리서치 기반 서비스 개선 기획 업무를 담당합니다.
[자격요건]
UX 리서치 및 사용자 인터뷰 진행 경험자 우대.
[우대사항]
SQL 활용 가능자 우대.
"""},
    ],
    ("카카오뱅크", "금융IT"): [
        {"posting_id": "mock-jd-16", "posted_date": "2024-10-14", "raw_text": """
[담당업무]
금융 서비스 백엔드 개발 및 시스템 안정성 관리 업무를 수행합니다.
[자격요건]
Java 기반 백엔드 개발 경험, 금융 시스템 안정성에 대한 이해 필요.
[우대사항]
대용량 트래픽 처리 경험자 우대.
"""},
        {"posting_id": "mock-jd-17", "posted_date": "2023-10-16", "raw_text": """
[담당업무]
핀테크 서비스 개발 및 금융 도메인 시스템 운영을 담당합니다.
[자격요건]
핀테크 도메인 이해, 대용량 트래픽 처리 경험자 우대.
[우대사항]
Java 기반 백엔드 개발 경험자 우대.
"""},
    ],
}


def _has_stored_chunks(company_id: str, job_family: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM job_jd_embeddings WHERE company_id = %s AND job_family = %s LIMIT 1",
        (company_id, job_family),
    )
    found = cur.fetchone() is not None
    cur.close()
    conn.close()
    return found


def get_jd_evidence(company_id: str, job_family: str, query: str | None = None, top_k: int = 3) -> list[dict]:
    """hiring_pattern_pipeline.get_hiring_season()에서 부르는 진입점.
    저장된 게 없으면(진짜 크롤링 전이라 항상 없음) mock 공고 원문을 섹션 단위로
    쪼개서(chunk_jd_text) 먼저 채워넣고 검색한다. 사람인 API로 진짜 JD 본문을
    확보하게 되면, MOCK_JD_POSTINGS 대신 그 원문을 같은 방식(chunk_jd_text ->
    store_jd_chunks)으로 넣는 별도 배치 스크립트로 교체하면 됨(이 함수 시그니처는
    안 바뀜)."""
    if not _has_stored_chunks(company_id, job_family):
        postings = MOCK_JD_POSTINGS.get((company_id, job_family), [])
        if not postings:
            return []
        chunks = [
            {"posting_id": posting["posting_id"], "posted_date": posting["posted_date"],
             "chunk_text": chunk_text, "company_id": company_id, "job_family": job_family}
            for posting in postings
            for chunk_text in chunk_jd_text(posting["raw_text"])
        ]
        store_jd_chunks(chunks)
    return search_jd_chunks(company_id, job_family, query=query, top_k=top_k)


if __name__ == "__main__":
    conn = get_connection()
    conn.cursor().execute("DELETE FROM job_jd_embeddings")
    conn.commit()
    conn.close()

    print("=== 회사 클릭만 한 경우 (쿼리 없음, 최신순) ===")
    for r in get_jd_evidence("삼성전자", "데이터"):
        print(f"({r['posted_date']}) {r['chunk_text']}")

    print("\n=== 특정 질문이 있는 경우 (유사도 검색) ===")
    for r in get_jd_evidence("삼성전자", "데이터", query="프로그래밍 언어나 툴 요구사항"):
        print(f"[거리 {r['distance']:.4f}] ({r['posted_date']}) {r['chunk_text']}")
