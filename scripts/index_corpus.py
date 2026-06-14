"""전체 코퍼스 인덱싱 — 청킹 → 임베딩 → pgvector 적재 (대량/무거운 작업).

1,000건이 넘는 임베딩은 노트북이 아니라 이 스크립트로 실행한다. tqdm으로 진행도를 본다.

사용 예:
    # 전체 1,417개 문서 인덱싱 (진짜 baseline용)
    python scripts/index_corpus.py

    # 빠른 동작 확인용으로 200개만
    python scripts/index_corpus.py --max-docs 200

    # 다른 임베딩 모델/컬렉션으로 (Phase 5 벤치마크)
    python scripts/index_corpus.py --embed-model BAAI/bge-m3 --collection rag_docs_bge_m3
"""

from __future__ import annotations

import argparse
import time

from src import chunker, config, embedder, retriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="코퍼스를 청킹·임베딩하여 pgvector에 적재한다.")
    p.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="인덱싱할 문서 수 상한 (기본: 전체). 빠른 확인 시 작게 지정.",
    )
    p.add_argument("--embed-model", default=config.EMBED_MODEL, help="임베딩 모델 이름")
    p.add_argument("--collection", default=config.COLLECTION_NAME, help="pgvector 테이블 이름")
    p.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    p.add_argument("--index-batch-size", type=int, default=256, help="pgvector 적재 배치 크기")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    # 1) 로드 & 청킹
    documents = chunker.load_documents(config.DOCS_DIR, max_docs=args.max_docs)
    chunks = chunker.chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    print(
        f"문서 {len(documents)}개 → 청크 {len(chunks)}개 "
        f"(max_docs={args.max_docs}, chunk_size={args.chunk_size}/{args.chunk_overlap})"
    )

    # 2) 임베딩 모델
    device = embedder.detect_device()
    embeddings = embedder.build_embeddings(args.embed_model, device=device)
    dim = len(embeddings.embed_query("임베딩 차원 확인"))
    print(f"임베딩: {args.embed_model} (device={device}, dim={dim})")

    # 3) 테이블 (재)생성 후 적재
    engine = retriever.get_engine()
    retriever.init_table(engine, args.collection, vector_size=dim, overwrite_existing=True)
    store = retriever.get_store(engine, embeddings, table_name=args.collection)

    n = retriever.index_chunks(
        store, chunks, batch_size=args.index_batch_size, show_progress=True
    )

    elapsed = time.perf_counter() - t0
    print(f"\n✅ 적재 완료: {n}개 청크 → 테이블 {args.collection!r}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
