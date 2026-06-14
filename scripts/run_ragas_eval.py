"""RAGAS 평가 실행 — 수집된 평가입력으로 4개 지표를 측정한다 (Claude API 호출, 비용/시간 소요).

사전조건: `scripts/collect_eval_inputs.py`로 `results/eval_inputs.json`이 생성된 상태여야 한다.
(`--collect`를 주면 평가입력 수집부터 한 번에 수행한다.)

사용 예:
    python scripts/run_ragas_eval.py
    python scripts/run_ragas_eval.py --collect          # 평가입력 수집 + 평가
    python scripts/run_ragas_eval.py --max-workers 2     # rate limit 더 완화
"""

from __future__ import annotations

import argparse

from src import config, dataset, embedder, evaluator, rag, retriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGAS 4개 지표를 측정하고 결과를 저장한다.")
    p.add_argument("--embed-model", default=config.EMBED_MODEL)
    p.add_argument("--collection", default=config.COLLECTION_NAME)
    p.add_argument("--judge-model", default=config.JUDGE_MODEL)
    p.add_argument(
        "--max-workers", type=int, default=4, help="동시 API 호출 수 (낮출수록 rate limit 완화)"
    )
    p.add_argument(
        "--collect",
        action="store_true",
        help="eval_inputs.json이 있어도 골든셋 RAG 추론부터 다시 수집",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = embedder.build_embeddings(args.embed_model)

    # 1) 평가입력 — 캐시가 있으면 재사용, 없거나 --collect면 RAG 추론으로 수집
    if config.EVAL_INPUTS_PATH.exists() and not args.collect:
        records = evaluator.load_eval_inputs()
        print(f"평가입력 캐시 로드: {len(records)}개 ({config.EVAL_INPUTS_PATH})")
    else:
        golden_set = dataset.load_golden_set()
        engine = retriever.get_engine()
        store = retriever.get_store(engine, embeddings, table_name=args.collection)
        rag_retriever = retriever.get_retriever(store, k=config.TOP_K)
        qa_chain = rag.build_qa_chain(rag_retriever, rag.build_llm())
        records = evaluator.collect_eval_inputs(qa_chain, golden_set, show_progress=True)
        evaluator.save_eval_inputs(records)
        print(f"평가입력 수집 완료: {len(records)}개")

    # 2) RAGAS 평가
    judge = evaluator.build_judge(args.judge_model)
    evaluator_llm, evaluator_emb = evaluator.build_evaluators(judge, embeddings)
    eval_dataset = evaluator.build_dataset(records)
    print(f"평가 시작 — {len(eval_dataset)}개 샘플 / judge={args.judge_model}")

    result = evaluator.run_ragas(
        eval_dataset, evaluator_llm, evaluator_emb, max_workers=args.max_workers
    )
    aggregate = evaluator.save_results(result)

    print("\n=== 지표별 평균 점수 ===")
    for k, v in aggregate.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"\n저장: {config.RAGAS_RESULT_PATH}\n      {config.RAGAS_DETAIL_PATH}")


if __name__ == "__main__":
    main()
