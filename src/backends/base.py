"""벡터DB 공통 인터페이스 — Phase 5 벤치마크의 DB 추상화.

세 DB(pgvector / Qdrant / Neo4j)를 **동일한 raw 벡터 인터페이스**로 다룬다.
백엔드는 임베딩을 직접 하지 않고 **이미 계산된 벡터만** 받는다 — 이렇게 해야
검색 latency에 임베딩 비용이 섞이지 않아 DB 간 비교가 공정하다.

수명주기:
    backend.recreate(dim)   # 스키마/컬렉션을 비우고 재생성 (멱등)
    backend.add(...)        # 벡터 적재
    backend.finalize()      # ANN 인덱스를 질의 가능 상태로 만든다
    backend.search(v, k)    # 코사인 유사도 top-k
    backend.close()

모든 백엔드는 **코사인 유사도**를 쓴다 — 임베딩이 정규화(normalize=True)되어 있다는 전제.
`score`는 코사인 유사도(클수록 가까움)로 통일한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Hit:
    """검색 결과 1건. 백엔드와 무관하게 동일한 형태로 반환한다."""

    text: str
    score: float  # 코사인 유사도 (클수록 유사)
    title: str | None = None


class VectorBackend(ABC):
    """벡터DB 백엔드 공통 인터페이스."""

    name: str  # "pgvector" | "qdrant" | "neo4j"

    @abstractmethod
    def recreate(self, dim: int) -> None:
        """기존 데이터를 지우고 빈 컬렉션/테이블을 재생성한다 (스키마만, 멱등)."""

    @abstractmethod
    def add(
        self,
        vectors: list[list[float]],
        texts: list[str],
        titles: list[str | None],
        batch_size: int = 256,
    ) -> int:
        """벡터·본문·제목을 적재한다. 적재한 건수를 반환."""

    def finalize(self) -> None:  # noqa: B027  선택적 훅 — 기본 no-op, DB별로만 override
        """적재 후 ANN 인덱스를 질의 가능 상태로 만든다 (기본: 별도 작업 없음).

        DB마다 인덱스 구축 시점이 달라(아래 참고) 인덱싱 시간 측정엔
        `add() + finalize()` 전체 wall-clock을 쓴다.
        - pgvector: 적재 후 HNSW 인덱스를 여기서 생성
        - Qdrant: upsert 시 점진적으로 구축 → no-op
        - Neo4j: recreate에서 인덱스 생성 → 여기서 온라인 대기
        """

    @abstractmethod
    def search(self, vector: list[float], k: int) -> list[Hit]:
        """질의 벡터의 코사인 유사도 top-k를 반환한다."""

    @abstractmethod
    def count(self) -> int:
        """적재된 벡터 수."""

    def close(self) -> None:  # noqa: B027  선택적 훅 — 기본 no-op, DB별로만 override
        """커넥션 정리 (기본: 별도 작업 없음)."""
