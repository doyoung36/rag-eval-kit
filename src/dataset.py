"""데이터셋 준비 — KorQuAD 로드 → title 그룹핑/dedup → docs, 골든셋 샘플링.

Phase 2(`00_dataset_prep.ipynb`)의 로직을 함수로 분리한 모듈.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from . import config


def load_korquad():
    """`KorQuAD/squad_kor_v1` 원본 로드. (train, validation) splits를 가진 DatasetDict 반환."""
    from datasets import load_dataset

    return load_dataset("KorQuAD/squad_kor_v1")


def build_docs(train) -> dict[str, list[str]]:
    """title 기준 그룹핑 + context dedup(첫 등장 순서 보존 → 재현성)."""
    docs: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in train:
        title, context = row["title"], row["context"]
        if context not in seen[title]:
            seen[title].add(context)
            docs[title].append(context)
    return dict(docs)


def slugify(title: str) -> str:
    """파일명 안전화: 파일시스템 금지 문자 치환, 앞뒤 공백/점 제거."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" .")
    return s or "untitled"


def save_docs(docs: dict[str, list[str]], docs_dir: Path = config.DOCS_DIR) -> list[Path]:
    """title별 context들을 빈 줄로 이어 하나의 .txt 문서로 저장. 저장된 경로 리스트 반환."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for title, contexts in docs.items():
        path = docs_dir / f"{slugify(title)}.txt"
        path.write_text("\n\n".join(contexts), encoding="utf-8")
        saved.append(path)
    return saved


def _is_valid(row) -> bool:
    """골든셋 필터: context 100자 이상 + answer 비어있지 않음."""
    answers = row["answers"]["text"]
    return len(row["context"]) >= 100 and len(answers) > 0 and answers[0].strip() != ""


def build_golden_set(
    train, size: int = config.GOLDEN_SIZE, seed: int = config.SEED
) -> list[dict]:
    """`(question, context, answers)` 삼중쌍에서 RAGAS 입력 형식 골든셋을 seed 고정 샘플링."""
    candidates = [i for i in range(len(train)) if _is_valid(train[i])]
    random.seed(seed)
    picked = random.sample(candidates, size)
    return [
        {
            "question": train[i]["question"],
            "ground_truth": train[i]["answers"]["text"][0],
            "reference_contexts": [train[i]["context"]],
        }
        for i in picked
    ]


def save_golden_set(golden_set: list[dict], path: Path = config.GOLDEN_PATH) -> Path:
    """골든셋을 JSON으로 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden_set, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_golden_set(path: Path = config.GOLDEN_PATH) -> list[dict]:
    """골든셋 JSON 로드."""
    return json.loads(path.read_text(encoding="utf-8"))
