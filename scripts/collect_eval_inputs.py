"""골든셋 RAG 추론 수집 — 50개 질문을 RAG 체인에 통과시켜 평가입력을 모은다 (Ollama 추론).

결과는 `results/eval_inputs.json`에 저장되어 RAGAS 평가 시 재사용된다.
사전조건: `scripts/index_corpus.py`로 pgvector 테이블이 적재된 상태여야 한다.

사용 예:
    python scripts/collect_eval_inputs.py
    python scripts/collect_eval_inputs.py --collection rag_docs_bge_m3_ko --top-k 4
"""

from __future__ import annotations

import argparse

from src import config, dataset, embedder, evaluator, rag, retriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="골든셋을 RAG 체인에 통과시켜 평가입력을 수집한다.")
    p.add_argument("--embed-model", default=config.EMBED_MODEL)
    p.add_argument("--collection", default=config.COLLECTION_NAME)
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    golden_set = dataset.load_golden_set()
    print(f"골든셋: {len(golden_set)}개")

    embeddings = embedder.build_embeddings(args.embed_model)
    engine = retriever.get_engine()
    store = retriever.get_store(engine, embeddings, table_name=args.collection)

    # 적재 여부 확인 — 비어 있으면 먼저 index_corpus.py를 실행해야 한다
    probe = store.similarity_search(golden_set[0]["question"], k=1)
    if not probe:
        raise SystemExit(
            f"pgvector 테이블 {args.collection!r}이 비어 있습니다. "
            "먼저 scripts/index_corpus.py로 적재하세요."
        )

    rag_retriever = retriever.get_retriever(store, k=args.top_k)
    qa_chain = rag.build_qa_chain(rag_retriever, rag.build_llm())

    records = evaluator.collect_eval_inputs(qa_chain, golden_set, show_progress=True)
    path = evaluator.save_eval_inputs(records)
    print(f"\n✅ 수집 완료: {len(records)}개 → {path}")


if __name__ == "__main__":
    main()
