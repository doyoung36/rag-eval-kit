"""벤치마크 타이밍 하니스 — 임베딩 / 인덱싱 / 검색 latency 측정.

설계 원칙: **임베딩과 DB를 분리**한다.
- 코퍼스·질의 임베딩은 한 번 계산해 재사용한다 (DB 비교 시 동일 벡터 투입).
- 백엔드는 raw 벡터만 받으므로 검색 latency에 임베딩 비용이 섞이지 않는다.

따라서 한 (DB, 임베딩) 조합의 측정은:
    vectors, t_embed = embed_corpus(emb, texts)     # 임베딩 1회
    t_index = index_corpus(backend, dim, vectors, ...)  # add + finalize
    results, latencies = run_queries(backend, qvectors, k)
    stats = latency_stats(latencies)                # p50 / p95 / mean
"""

from __future__ import annotations

import time

from .backends.base import Hit, VectorBackend


def embed_corpus(embeddings, texts: list[str]) -> tuple[list[list[float]], float]:
    """코퍼스 전체를 배치 임베딩한다. (벡터 리스트, 소요 초)를 반환."""
    t0 = time.perf_counter()
    vectors = embeddings.embed_documents(texts)
    return vectors, time.perf_counter() - t0


def embed_queries(embeddings, questions: list[str]) -> tuple[list[list[float]], float]:
    """질의를 임베딩한다. 실제 검색과 동일하게 `embed_query`를 건별로 호출한다
    (instruct 계열은 질의 프리픽스가 문서용과 다를 수 있어 문서 임베딩과 분리한다)."""
    t0 = time.perf_counter()
    vectors = [embeddings.embed_query(q) for q in questions]
    return vectors, time.perf_counter() - t0


def embedding_dim(vectors: list[list[float]]) -> int:
    """임베딩 차원."""
    return len(vectors[0])


def index_corpus(
    backend: VectorBackend,
    dim: int,
    vectors: list[list[float]],
    texts: list[str],
    titles: list[str | None],
    batch_size: int = 256,
) -> float:
    """백엔드를 비우고 적재 + 인덱스 구축까지의 wall-clock(초)을 잰다.

    `recreate`(스키마 생성)는 측정에서 제외하고, **데이터 적재 + 인덱스 구축**
    (`add` + `finalize`)만 인덱싱 시간으로 본다.
    """
    backend.recreate(dim) # 측정 밖: 기존 데이터 삭제 + 빈 스키마 생성
    t0 = time.perf_counter() # 측정 시작
    backend.add(vectors, texts, titles, batch_size=batch_size) # 적재
    backend.finalize() # 인덱스 구축 완료 대기
    return time.perf_counter() - t0 # 최종 소요시간 return


def run_queries(
    backend: VectorBackend, query_vectors: list[list[float]], k: int
) -> tuple[list[list[Hit]], list[float]]:
    """질의 벡터들을 순차 검색한다. (질의별 Hit 리스트, 질의별 latency[ms])를 반환.

    순차로 도는 이유: 동시 부하가 아니라 **단건 검색 latency**를 재기 위함.
    """
    results: list[list[Hit]] = []
    latencies_ms: list[float] = []
    for v in query_vectors:
        t0 = time.perf_counter()
        hits = backend.search(v, k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        results.append(hits)
    return results, latencies_ms


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """latency 분포 요약: p50 / p95 / mean (ms)."""
    import numpy as np

    a = np.asarray(latencies_ms, dtype=float)
    return {
        "latency_p50_ms": float(np.percentile(a, 50)),
        "latency_p95_ms": float(np.percentile(a, 95)),
        "latency_mean_ms": float(a.mean()),
    }
