"""pgvector 적재 & 리트리버 — langchain-postgres v2 (PGEngine + PGVectorStore).

Phase 3(`01_pipeline.ipynb`) 3단계 로직.

구버전 `PGVector`(고정 2테이블)와 달리 v2는 **엔진 생성 → 전용 테이블 초기화 → 팩토리 생성**의
구성이며 `table_name`마다 테이블 1개를 만든다.
- `init_table(overwrite_existing=True)` → 재실행 시 테이블 재생성(멱등)
- `vector_size`는 임베딩 차원과 반드시 일치해야 한다 (BGE-m3-ko = 1024)
"""

from __future__ import annotations

from . import config


def get_engine(pg_conn: str | None = None):
    """PGEngine 생성."""
    from langchain_postgres import PGEngine

    return PGEngine.from_connection_string(url=pg_conn or config.pg_conn())


def init_table(engine, table_name: str, vector_size: int, overwrite_existing: bool = True):
    """전용 벡터 테이블을 생성한다. ⚠️ overwrite_existing=True는 기존 데이터를 지운다."""
    engine.init_vectorstore_table(
        table_name=table_name,
        vector_size=vector_size,
        overwrite_existing=overwrite_existing,
    )


def get_store(engine, embeddings, table_name: str = config.COLLECTION_NAME):
    """기존 테이블에 연결만 한다 (재생성 X). 적재·검색 공용."""
    from langchain_postgres import PGVectorStore

    return PGVectorStore.create_sync(
        engine=engine,
        table_name=table_name,
        embedding_service=embeddings,
    )


def index_chunks(store, chunks: list, batch_size: int = 256, show_progress: bool = True) -> int:
    """청크를 배치로 나눠 적재한다. tqdm으로 진행도를 표시한다. 적재한 청크 수 반환.

    `add_documents`는 내부에서 모든 청크를 임베딩하므로 시간이 걸린다. 배치로 쪼개
    진행 상황을 보이게 하고, 대량 적재 도중 끊겨도 어디까지 됐는지 알 수 있게 한다.
    """
    total = len(chunks)
    batches = range(0, total, batch_size)
    if show_progress:
        from tqdm.auto import tqdm

        batches = tqdm(batches, desc="pgvector 적재", unit="batch")
    for start in batches:
        store.add_documents(chunks[start : start + batch_size])
    return total


def get_retriever(store, k: int = config.TOP_K):
    """유사도 검색 리트리버 반환."""
    return store.as_retriever(search_kwargs={"k": k})
