"""Neo4j 백엔드 — Neo4j 5의 네이티브 벡터 인덱스.

각 청크를 `:Chunk` 노드로 저장하고 `n.embedding` 속성에 벡터를 둔다.
`CREATE VECTOR INDEX ... cosine`으로 인덱스를 만들고 `db.index.vector.queryNodes`로 검색한다.
인덱스는 `recreate()`에서 생성하고, `finalize()`에서 온라인 상태가 될 때까지 대기한다.
"""

from __future__ import annotations

import re

from .base import Hit, VectorBackend

_LABEL = "Chunk"


def _safe_index_name(name: str) -> str:
    ident = re.sub(r"\W", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"idx_{ident}"
    return ident


class Neo4jBackend(VectorBackend):
    name = "neo4j"

    def __init__(self, collection: str, uri: str, user: str, password: str):
        from neo4j import GraphDatabase

        self.index = _safe_index_name(collection)
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def recreate(self, dim: int) -> None:
        with self.driver.session() as s:
            s.run(f"MATCH (n:{_LABEL}) DETACH DELETE n")
            s.run(f"DROP INDEX {self.index} IF EXISTS")
            s.run(
                f"CREATE VECTOR INDEX {self.index} IF NOT EXISTS "
                f"FOR (n:{_LABEL}) ON (n.embedding) "
                "OPTIONS {indexConfig: {"
                "  `vector.dimensions`: $dim,"
                "  `vector.similarity_function`: 'cosine'"
                "}}",
                dim=dim,
            )

    def add(
        self,
        vectors: list[list[float]],
        texts: list[str],
        titles: list[str | None],
        batch_size: int = 256,
    ) -> int:
        with self.driver.session() as s:
            for start in range(0, len(vectors), batch_size):
                rows = [
                    {"text": t, "title": ti, "embedding": [float(x) for x in v]}
                    for v, t, ti in zip(
                        vectors[start : start + batch_size],
                        texts[start : start + batch_size],
                        titles[start : start + batch_size],
                    )
                ]
                s.run(
                    f"UNWIND $rows AS row CREATE (n:{_LABEL}) "
                    "SET n.text = row.text, n.title = row.title, n.embedding = row.embedding",
                    rows=rows,
                )
        return len(vectors)

    def finalize(self) -> None:
        # 벡터 인덱스가 질의 가능(ONLINE) 상태가 될 때까지 블로킹 대기.
        with self.driver.session() as s:
            s.run("CALL db.awaitIndexes(300)")

    def search(self, vector: list[float], k: int) -> list[Hit]:
        with self.driver.session() as s:
            res = s.run(
                "CALL db.index.vector.queryNodes($index, $k, $v) "
                "YIELD node, score RETURN node.text AS text, node.title AS title, score",
                index=self.index,
                k=k,
                v=[float(x) for x in vector],
            )
            return [
                Hit(text=r["text"], title=r["title"], score=float(r["score"]))
                for r in res
            ]

    def count(self) -> int:
        with self.driver.session() as s:
            return s.run(f"MATCH (n:{_LABEL}) RETURN count(n) AS c").single()["c"]

    def close(self) -> None:
        self.driver.close()
