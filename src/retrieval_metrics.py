"""결정론적 검색 지표 — LLM 판정 없이 검색 순위 품질만 잰다.

Phase 5는 DB·임베딩 비교가 목적이라 검색 품질이 핵심이다. RAGAS의
`context_precision`/`context_recall`은 judge LLM에 의존해 변동·비용이 있으므로,
판정 LLM과 무관한 **결정론적 지표**를 함께 둔다.

관련성 판정: 단답형 골든셋(`ground_truth` = 짧은 정답 스팬)에 맞춰,
**검색된 청크 본문에 정답 문자열이 포함되면 관련(relevant)** 으로 본다
(공백 무시·소문자화 후 부분일치). 직관적이고 재현 가능하다.

지표:
- hit_rate@k : top-k 안에 정답을 담은 청크가 하나라도 있는 질의의 비율
- mrr@k      : 정답을 담은 첫 청크의 역순위(1/rank) 평균
"""

from __future__ import annotations

import re

from .backends.base import Hit


def _norm(s: str) -> str:
    """공백 제거 + 소문자화 — 청크/정답 표면형 차이를 흡수한다."""
    return re.sub(r"\s+", "", s).lower()


def is_relevant(hit_text: str, ground_truth: str) -> bool:
    """검색된 청크가 정답 스팬을 포함하면 관련."""
    gt = _norm(ground_truth)
    return bool(gt) and gt in _norm(hit_text)


def _first_relevant_rank(hits: list[Hit], ground_truth: str, k: int) -> int | None:
    """top-k 중 정답을 담은 첫 청크의 1-based 순위. 없으면 None."""
    for rank, h in enumerate(hits[:k], start=1):
        if is_relevant(h.text, ground_truth):
            return rank
    return None


def evaluate_retrieval(
    results: list[list[Hit]], golden: list[dict], k: int
) -> dict[str, float]:
    """질의별 검색 결과(`results`)와 골든셋(`golden`, 같은 순서)으로 집계 지표를 낸다.

    `golden[i]["ground_truth"]`가 정답 스팬. `results[i]`는 i번째 질의의 top-k Hit.
    """
    assert len(results) == len(golden), "results와 golden의 길이/순서가 일치해야 한다"

    ranks = [
        _first_relevant_rank(hits, g["ground_truth"], k)
        for hits, g in zip(results, golden, strict=True)
    ]
    n = len(ranks)
    hit_rate = sum(r is not None for r in ranks) / n if n else 0.0
    mrr = sum((1.0 / r) if r else 0.0 for r in ranks) / n if n else 0.0
    return {f"hit_rate@{k}": hit_rate, f"mrr@{k}": mrr}
