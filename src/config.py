"""전역 설정 — 경로·하이퍼파라미터·접속정보를 한곳에 모은다.

경로는 `__file__` 기준으로 프로젝트 루트를 계산하므로 **현재 작업 디렉토리(cwd)와 무관**하다.
노트북(`notebooks/`)에서 호출하든 스크립트(`scripts/`)에서 호출하든 동일한 경로를 가리킨다.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- 경로 (cwd 독립) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
GOLDEN_PATH = DATA_DIR / "golden_set.json"
RESULTS_DIR = PROJECT_ROOT / "results"
ENV_PATH = PROJECT_ROOT / ".env"

# .env는 프로젝트 루트에서 로드한다 (cwd 무관)
load_dotenv(ENV_PATH)

# --- 골든셋 ---
GOLDEN_SIZE = 50
SEED = 42

# --- 청킹 ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- 임베딩 / 컬렉션 ---
EMBED_MODEL = "dragonkue/BGE-m3-ko"
COLLECTION_NAME = "rag_docs_bge_m3_ko"
EMBED_BATCH_SIZE = 32

# --- 검색 ---
TOP_K = 4

# --- 평가자(judge) LLM ---
JUDGE_MODEL = "claude-haiku-4-5"

# --- 결과 파일 ---
EVAL_INPUTS_PATH = RESULTS_DIR / "eval_inputs.json"
RAGAS_RESULT_PATH = RESULTS_DIR / "ragas_baseline.json"
RAGAS_DETAIL_PATH = RESULTS_DIR / "ragas_baseline_detail.csv"

# --- Phase 5 벤치마크 ---
# Round 1: 임베딩 고정, DB 3종 비교 / Round 2: DB 고정, 임베딩 3종 비교
BENCH_BACKENDS = ("pgvector", "qdrant", "neo4j")
BENCH_EMBED_MODELS = (
    "dragonkue/BGE-m3-ko",  # baseline (Phase 3·4와 동일)
    "Qwen/Qwen3-Embedding-0.6B",
    "google/embeddinggemma-300m",
)
BENCH_COLLECTION = "bench_chunks"  # 벤치 전용 컬렉션/테이블/인덱스 기본 이름
BENCH_DB_PATH = RESULTS_DIR / "bench_db.csv"
BENCH_EMBED_PATH = RESULTS_DIR / "bench_embedding.csv"


def pg_conn() -> str:
    """pgvector 접속 문자열 (langchain-postgres v2 PGEngine → psycopg3)."""
    return (
        f"postgresql+psycopg://{os.getenv('PGVECTOR_USER', 'user')}:"
        f"{os.getenv('PGVECTOR_PASSWORD', 'test123!')}@"
        f"{os.getenv('PGVECTOR_HOST', 'localhost')}:"
        f"{os.getenv('PGVECTOR_PORT', '5432')}/"
        f"{os.getenv('PGVECTOR_DB', 'ragdb')}"
    )


def ollama_config() -> dict[str, str]:
    """Ollama 모델·엔드포인트 설정."""
    return {
        "model": os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    }


def pg_raw_conn() -> str:
    """pgvector raw 접속 문자열 (psycopg3 libpq conninfo).

    `pg_conn()`는 langchain-postgres(SQLAlchemy URL)용이고, 이쪽은 벤치마크에서
    psycopg로 직접 SQL을 던질 때 쓴다. 기본값은 `docker-compose.yml`/`.env.example`와 일치(rag/rag).
    """
    return (
        f"host={os.getenv('PGVECTOR_HOST', 'localhost')} "
        f"port={os.getenv('PGVECTOR_PORT', '5432')} "
        f"dbname={os.getenv('PGVECTOR_DB', 'ragdb')} "
        f"user={os.getenv('PGVECTOR_USER', 'rag')} "
        f"password={os.getenv('PGVECTOR_PASSWORD', 'rag')}"
    )


def qdrant_config() -> dict[str, object]:
    """Qdrant 접속 설정 (REST 포트)."""
    return {
        "host": os.getenv("QDRANT_HOST", "localhost"),
        "port": int(os.getenv("QDRANT_PORT", "6333")),
    }


def neo4j_config() -> dict[str, str]:
    """Neo4j Bolt 접속 설정."""
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "ragpassword"),
    }
