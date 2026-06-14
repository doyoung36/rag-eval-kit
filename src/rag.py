"""RAG 체인 — Ollama LLM + LCEL(stuff) 구성. 검색 근거(source_documents)도 함께 반환.

Phase 3(`01_pipeline.ipynb`) 4·5단계 로직.
과거의 `RetrievalQA`는 LangChain 1.x에서 deprecated(langchain-classic)이므로 LCEL로 대체한다.
"""

from __future__ import annotations

from . import config

PROMPT_TEMPLATE = (
    "당신은 주어진 문맥(context)에만 근거해 한국어로 간결히 답하는 어시스턴트입니다.\n"
    '문맥에서 답을 찾을 수 없으면 "문맥에서 답을 찾을 수 없습니다."라고 답하세요.\n\n'
    "문맥:\n{context}\n\n"
    "질문: {question}\n"
    "답변:"
)


def build_llm(temperature: float = 0):
    """로컬 Ollama 서버의 모델을 ChatOllama로 연결 (재현성을 위해 temperature=0)."""
    from langchain_ollama import ChatOllama

    cfg = config.ollama_config()
    return ChatOllama(model=cfg["model"], base_url=cfg["base_url"], temperature=temperature)


def format_docs(docs) -> str:
    """검색 문서들을 빈 줄로 이어 하나의 context 문자열로 결합."""
    return "\n\n".join(d.page_content for d in docs)


def build_qa_chain(retriever, llm):
    """retriever + llm을 LCEL로 묶어 `{question, source_documents, answer}`를 반환하는 체인 구성.

    질문(문자열) → 검색(source_documents) + 생성(answer)을 함께 반환한다.
    RetrievalQA의 return_source_documents=True와 동일하게 근거 문서를 보존한다.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    answer_chain = prompt | llm | StrOutputParser()

    return RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retriever,
    ).assign(
        answer=lambda x: answer_chain.invoke(
            {"context": format_docs(x["source_documents"]), "question": x["question"]}
        )
    )
