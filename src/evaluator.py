"""RAGAS 평가 — 호환성 셰임 / judge 설정 / 평가입력 수집 / 4개 지표 측정.

Phase 4(`02_ragas_eval.ipynb`) 로직.

⚠️ **이 모듈을 import하는 것만으로 호환성 셰임이 적용된다** — 아래 `_apply_compat_shim()`이
모듈 로드 시 가장 먼저 실행되어, 이후 ragas import가 안전해진다. ragas 관련 import는 전부
함수 내부에서 지연 로드한다(셰임이 먼저 걸리도록).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from . import config


def _apply_compat_shim() -> None:
    """RAGAS 0.4.3 × langchain 1.x 호환 셰임.

    RAGAS 0.4.3은 `llms/base.py`에서 langchain 1.x가 삭제한
    `langchain_community.chat_models.vertexai.ChatVertexAI`를 모듈 로드 시 무조건 import한다.
    Anthropic 래퍼만 쓰므로 Vertex 경로는 불필요 → 빈 자리표시 모듈을 주입해 import만 통과시킨다.
    (RAGAS가 langchain 1.x를 지원하면 이 셰임은 삭제 가능.)
    """
    import langchain_community.chat_models as _cm

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        _vmod = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # 자리표시 — Vertex를 선택하지 않는 한 사용되지 않음
            ...

        _vmod.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = _vmod
        _cm.vertexai = _vmod

    import langchain_community.llms as _llms

    if not hasattr(_llms, "VertexAI"):

        class VertexAI:  # 자리표시
            ...

        _llms.VertexAI = VertexAI


# 모듈 로드 시점에 셰임을 먼저 적용한다 (이후 어떤 ragas import보다 앞섬).
_apply_compat_shim()


def build_judge(model: str = config.JUDGE_MODEL):
    """평가자 LLM(Claude) — ChatAnthropic. 일관된 채점을 위해 temperature=0."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, temperature=0, max_tokens=1024, max_retries=3)


def build_evaluators(judge, embeddings):
    """ragas용 (LLM 래퍼, 임베딩 래퍼) 튜플 반환.

    `answer_relevancy`가 임베딩 유사도를 쓰므로 RAG와 동일한 임베딩을 감싸 넘긴다.
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(judge), LangchainEmbeddingsWrapper(embeddings)


def collect_eval_inputs(
    qa_chain, golden_set: list[dict], show_progress: bool = True
) -> list[dict]:
    """골든셋을 RAG 체인에 통과시켜 RAGAS 입력 레코드를 수집한다.

    SingleTurnSample 필드명에 맞춰 키를 구성한다:
    `user_input`(질문) / `retrieved_contexts`(검색 청크) / `response`(답변) / `reference`(정답).
    실패한 샘플은 건너뛴다.
    """
    items = golden_set
    if show_progress:
        from tqdm.auto import tqdm

        items = tqdm(golden_set, desc="RAG 추론", unit="q")

    records = []
    for item in items:
        q = item["question"]
        try:
            out = qa_chain.invoke(q)
        except Exception as e:  # noqa: BLE001 — 한 샘플 실패가 전체를 막지 않게 한다
            print(f"  실패: {q[:40]!r} — {e!r} — 건너뜀")
            continue
        records.append(
            {
                "user_input": q,
                "retrieved_contexts": [d.page_content for d in out["source_documents"]],
                "response": out["answer"].strip(),
                "reference": item["ground_truth"],
            }
        )
    return records


def save_eval_inputs(records: list[dict], path: Path = config.EVAL_INPUTS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_eval_inputs(path: Path = config.EVAL_INPUTS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset(records: list[dict]):
    """RAGAS 입력 레코드를 EvaluationDataset으로 구성."""
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            retrieved_contexts=r["retrieved_contexts"],
            response=r["response"],
            reference=r["reference"],
        )
        for r in records
    ]
    return EvaluationDataset(samples=samples)


def run_ragas(eval_dataset, evaluator_llm, evaluator_emb, max_workers: int = 4):
    """4개 지표(faithfulness, answer_relevancy, context_precision, context_recall) 측정.

    `max_workers`를 낮춰 Claude API rate limit을 완화한다.
    """
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    run_config = RunConfig(timeout=180, max_retries=5, max_workers=max_workers)
    return evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        run_config=run_config,
        show_progress=True,
    )


def build_retrieval_records(golden_set: list[dict], results: list[list]) -> list[dict]:
    """Phase 5용 — 검색 결과(Hit 리스트)와 골든셋으로 RAGAS 입력 레코드를 만든다.

    검색 전용 지표(context_precision/recall)는 `response`(생성 답변)가 필요 없으므로
    LLM 생성 단계 없이 검색된 청크만으로 구성한다. `results[i]`는 i번째 질의의 Hit 리스트.
    """
    return [
        {
            "user_input": g["question"],
            "retrieved_contexts": [h.text for h in hits],
            "reference": g["ground_truth"],
        }
        for g, hits in zip(golden_set, results)
    ]


def run_ragas_retrieval(records: list[dict], evaluator_llm, evaluator_emb, max_workers: int = 4):
    """검색 전용 RAGAS — context_precision / context_recall만 측정 (Phase 5).

    생성 지표(faithfulness/answer_relevancy)는 Phase 5에서 제외한다 — DB·임베딩만
    바뀌고 생성 LLM은 고정이라 비교 변별력이 없고 judge 비용만 든다.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import context_precision, context_recall
    from ragas.run_config import RunConfig

    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in records
    ]
    run_config = RunConfig(timeout=180, max_retries=5, max_workers=max_workers)
    return evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        run_config=run_config,
        show_progress=True,
    )


_METRIC_COLS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def save_results(
    result,
    detail_path: Path = config.RAGAS_DETAIL_PATH,
    result_path: Path = config.RAGAS_RESULT_PATH,
) -> dict[str, float]:
    """샘플별 점수 → CSV, 지표별 평균 → JSON 저장. 집계 딕셔너리 반환."""
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    df = result.to_pandas()
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    aggregate = {c: float(df[c].mean(skipna=True)) for c in _METRIC_COLS if c in df.columns}
    result_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate
