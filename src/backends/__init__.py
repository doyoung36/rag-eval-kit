"""벡터DB 백엔드 — 공통 인터페이스 + 팩토리.

사용:
    from src.backends import get_backend
    be = get_backend("qdrant", collection="bench_chunks")
    be.recreate(dim); be.add(vectors, texts, titles); be.finalize()
    hits = be.search(query_vector, k=4)
    be.close()
"""

from __future__ import annotations

from .. import config
from .base import Hit, VectorBackend


def get_backend(name: str, collection: str = config.BENCH_COLLECTION) -> VectorBackend:
    """이름으로 백엔드를 생성한다. 접속 정보는 `config`에서 읽는다.

    클라이언트 import는 함수 내부에서 지연 로드 → 선택한 DB의 의존성만 필요.
    """
    name = name.lower()
    if name == "pgvector":
        from .pgvector_backend import PgVectorBackend

        return PgVectorBackend(table=collection, conninfo=config.pg_raw_conn())
    if name == "qdrant":
        from .qdrant_backend import QdrantBackend

        cfg = config.qdrant_config()
        return QdrantBackend(collection=collection, host=cfg["host"], port=cfg["port"])
    if name == "neo4j":
        from .neo4j_backend import Neo4jBackend

        cfg = config.neo4j_config()
        return Neo4jBackend(
            collection=collection,
            uri=cfg["uri"],
            user=cfg["user"],
            password=cfg["password"],
        )
    raise ValueError(f"알 수 없는 백엔드: {name!r} (지원: {config.BENCH_BACKENDS})")


__all__ = ["Hit", "VectorBackend", "get_backend"]
