"""임베딩 모델 — HuggingFaceEmbeddings 빌드 (CUDA/MPS/CPU 자동 선택).

Phase 3(`01_pipeline.ipynb`) 2단계 로직. Phase 4·5에서 동일 모델을 재사용한다.
"""

from __future__ import annotations

from . import config


def detect_device() -> str:
    """사용 가능한 가속기 자동 선택: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_embeddings(
    model_name: str = config.EMBED_MODEL,
    device: str | None = None,
    batch_size: int = config.EMBED_BATCH_SIZE,
):
    """코사인 유사도 검색을 위해 정규화(`normalize_embeddings=True`)된 임베딩 모델을 로드.

    `device=None`이면 자동 선택. 인덱싱과 검색에 **반드시 동일한 모델**을 써야 한다.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    if device is None:
        device = detect_device()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )
