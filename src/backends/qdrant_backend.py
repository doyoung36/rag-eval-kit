"""Qdrant 백엔드 — qdrant_client (REST).

Qdrant는 upsert 시점에 HNSW 인덱스를 점진적으로 구축하므로 `finalize()`는 no-op이다.
검색은 최신 API인 `query_points`를 쓴다 (`search`는 deprecated).
"""

from __future__ import annotations

from .base import Hit, VectorBackend


class QdrantBackend(VectorBackend):
    name = "qdrant"

    def __init__(self, collection: str, host: str, port: int):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.client = QdrantClient(host=host, port=port)

    def recreate(self, dim: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
        )

    def add(
        self,
        vectors: list[list[float]],
        texts: list[str],
        titles: list[str | None],
        batch_size: int = 256,
    ) -> int:
        from qdrant_client import models

        for start in range(0, len(vectors), batch_size):
            points = [
                models.PointStruct(
                    id=start + i,
                    vector=list(v),
                    payload={"text": t, "title": ti},
                )
                for i, (v, t, ti) in enumerate(
                    zip(
                        vectors[start : start + batch_size],
                        texts[start : start + batch_size],
                        titles[start : start + batch_size],
                    )
                )
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(vectors)

    def search(self, vector: list[float], k: int) -> list[Hit]:
        res = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=k,
            with_payload=True,
        ).points
        return [
            Hit(
                text=p.payload.get("text", ""),
                title=p.payload.get("title"),
                score=float(p.score),
            )
            for p in res
        ]

    def count(self) -> int:
        return self.client.count(self.collection).count

    def close(self) -> None:
        self.client.close()
