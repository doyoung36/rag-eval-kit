"""pgvector 백엔드 — raw psycopg3 + pgvector.

langchain-postgres(`retriever.py`)는 RAG 파이프라인용 추상화이고, 벤치마크에서는
인덱싱 시간·검색 latency를 정밀하게 재기 위해 psycopg로 직접 SQL을 던진다.

인덱스 전략: 대량 적재 후 HNSW(cosine) 인덱스를 `finalize()`에서 생성한다
(빈 테이블에 인덱스를 먼저 만들고 넣는 것보다 bulk-load → index가 일반적으로 빠르다).
"""

from __future__ import annotations

import re

from .base import Hit, VectorBackend


def _safe_ident(name: str) -> str:
    """SQL 식별자 안전화 — 영숫자/언더스코어만 허용 (인젝션 방지)."""
    ident = re.sub(r"\W", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"t_{ident}"
    return ident


class PgVectorBackend(VectorBackend):
    name = "pgvector"

    def __init__(self, table: str, conninfo: str):
        import psycopg
        from pgvector.psycopg import register_vector

        self.table = _safe_ident(table)
        self.conn = psycopg.connect(conninfo, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def recreate(self, dim: int) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
        self.conn.execute(
            f"CREATE TABLE {self.table} ("
            "  id bigserial PRIMARY KEY,"
            "  text text,"
            "  title text,"
            f"  embedding vector({dim})"
            ")"
        )

    def add(
        self,
        vectors: list[list[float]],
        texts: list[str],
        titles: list[str | None],
        batch_size: int = 256,
    ) -> int:
        import numpy as np

        sql = f"INSERT INTO {self.table} (text, title, embedding) VALUES (%s, %s, %s)"
        with self.conn.cursor() as cur:
            for start in range(0, len(vectors), batch_size):
                rows = [
                    (t, ti, np.asarray(v, dtype=np.float32))
                    for v, t, ti in zip(
                        vectors[start : start + batch_size],
                        texts[start : start + batch_size],
                        titles[start : start + batch_size],
                        strict=True,
                    )
                ]
                cur.executemany(sql, rows)
        return len(vectors)

    def finalize(self) -> None:
        # HNSW + cosine. 적재가 끝난 뒤 한 번에 구축한다.
        self.conn.execute(
            f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_cosine_ops)"
        )

    def search(self, vector: list[float], k: int) -> list[Hit]:
        import numpy as np

        v = np.asarray(vector, dtype=np.float32)
        # `<=>` = 코사인 거리 → 유사도 = 1 - 거리
        rows = self.conn.execute(
            f"SELECT text, title, 1 - (embedding <=> %s) AS score "
            f"FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
            (v, v, k),
        ).fetchall()
        return [Hit(text=t, title=ti, score=float(s)) for t, ti, s in rows]

    def count(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
