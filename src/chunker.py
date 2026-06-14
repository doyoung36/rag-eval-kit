"""문서 로드 & 청킹 — `data/docs/`의 .txt를 Document로 감싸 RecursiveCharacterTextSplitter로 분할.

Phase 3(`01_pipeline.ipynb`) 1단계 로직.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def load_documents(docs_dir: Path = config.DOCS_DIR, max_docs: int | None = None) -> list:
    """`docs_dir`의 title별 .txt를 Document로 로드. `max_docs`로 앞 N개만(이름순) 제한 가능.

    .txt 로드는 파일을 읽어 Document로 감싸는 것뿐이라 TextLoader(langchain-community, sunset 예정)
    없이 pathlib로 직접 처리한다. 출처 추적용으로 title(=파일명) 메타데이터를 부여한다.
    """
    from langchain_core.documents import Document

    doc_paths = sorted(docs_dir.glob("*.txt"))
    if max_docs:
        doc_paths = doc_paths[:max_docs]
    return [
        Document(
            page_content=p.read_text(encoding="utf-8"),
            metadata={"title": p.stem, "source": str(p)},
        )
        for p in doc_paths
    ]


def chunk_documents(
    documents: list,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list:
    """Document 리스트를 청크로 분할. 문단 → 문장 → 단어 순으로 분할 시도."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
