# rag-eval-kit 구현 계획

## 목표

RAGAS로 RAG 파이프라인 품질 측정 + pgvector / Qdrant / Weaviate 벡터DB 비교 벤치마크

## 기술 스택

| 역할           | 선택                                                        |
| -------------- | ----------------------------------------------------------- |
| 데이터셋       | `KorQuAD/squad_kor_v1` (HuggingFace)                        |
| RAG 추론 LLM   | Ollama — `gemma4-e2b` (로컬)                                |
| RAGAS 평가 LLM | `langchain-anthropic` — Claude (API)                        |
| 임베딩         | Qwen3-Embedding-0.6B / embeddingGemma / dragonkue/BGE-m3-ko |
| 벡터DB         | pgvector / Qdrant / Neo4j                                   |

---

## Phase 1 — 환경 세팅

- [x] `docker-compose.yml` 작성 (pgvector, Qdrant, Neo4j 컨테이너)
- [x] Python 의존성 정의 (`pyproject.toml`)
  - `ragas`, `langchain`, `langchain-community`, `langchain-anthropic`
  - `psycopg2-binary`, `pgvector`, `qdrant-client`, `neo4j`
  - `sentence-transformers`, `datasets`
- [x] `.env.example` 작성 (ANTHROPIC_API_KEY, DB 접속 정보)
- [x] Ollama 설치 확인 및 `gemma4-e2b` 모델 pull
- [x] `README.md` 뼈대 작성 (프로젝트 목적, 실행 방법 placeholder)

---

## Phase 2 — 데이터셋 준비 및 가공

`notebooks/00_dataset_prep.ipynb`

### 원본 로드

- [ ] `datasets` 라이브러리로 `KorQuAD/squad_kor_v1` 로드

### 구조 파악 및 가공

> `squad_kor_v1`은 `title` 단위로 여러 paragraph가 묶여 있고,
> 같은 paragraph(context)가 다른 QA 쌍에서 중복 등장함

- [ ] `title` 기준으로 context 그룹핑 — 문서(Document) 단위 구성
- [ ] 같은 `title` 내 중복 context 제거 (dedup)
- [ ] 문서 원본을 `data/docs/` 에 title별 텍스트 파일로 저장

### 골든셋 추출

- [ ] (question, context, answers) 삼중쌍에서 골든셋 50개 샘플링
  - 조건: context 길이 100자 이상, answer 비어있지 않은 것만
- [ ] RAGAS 입력 형식으로 변환
  ```json
  { "question": "...", "ground_truth": "...", "reference_contexts": ["..."] }
  ```
- [ ] `data/golden_set.json` 저장

---

## Phase 3 — RAG 파이프라인 기본 구성

`notebooks/01_pipeline.ipynb`

- [ ] `data/docs/` 문서 로드 & 청킹 (`RecursiveCharacterTextSplitter`, chunk_size=500, overlap=50)
- [ ] 기본 임베딩(`dragonkue/BGE-m3-ko`)으로 벡터화
- [ ] pgvector에 임베딩 저장 & 유사도 검색 확인
- [ ] Ollama `gemma4-e2b` LLM 연결 (`ChatOllama`)
- [ ] LangChain `RetrievalQA` 체인으로 end-to-end 응답 생성 확인

---

## Phase 4 — RAGAS 평가

`notebooks/02_ragas_eval.ipynb`

- [ ] Claude 평가자 설정
  ```python
  from ragas.llms import LangchainLLMWrapper
  from langchain_anthropic import ChatAnthropic
  evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001"))
  ```
- [ ] 골든셋으로 RAG 체인 실행 → (question, answer, contexts) 수집
- [ ] 4가지 지표 측정
  - [ ] `faithfulness`
  - [ ] `answer_relevancy`
  - [ ] `context_precision`
  - [ ] `context_recall`
- [ ] 결과를 `results/ragas_baseline.json`으로 저장
- [ ] 지표별 점수 해석 메모 노트북에 기록

---

## Phase 5 — 벡터DB 벤치마크

`notebooks/03_vectordb_bench.ipynb`

- [ ] `src/retriever.py` — pgvector / Qdrant / Neo4j 공통 인터페이스 추상화

### Round 1 — 임베딩 고정(`dragonkue/BGE-m3-ko`), DB 3종 비교
- [ ] 동일 문서·동일 쿼리로 pgvector / Qdrant / Neo4j 각각 인덱싱 + 검색
- [ ] 측정 항목
  - [ ] 인덱싱 소요 시간
  - [ ] 검색 latency (골든셋 50개 쿼리, p50 / p95)
  - [ ] RAGAS `context_precision` / `context_recall`
- [ ] 결과를 `results/bench_db.csv`로 저장

### Round 2 — DB 고정(Round 1 우승 DB), 임베딩 3종 비교
- [ ] 동일 문서·동일 쿼리로 `Qwen3-Embedding-0.6B` / `embeddingGemma` / `dragonkue/BGE-m3-ko` 각각 벡터화
- [ ] 측정 항목
  - [ ] 임베딩 생성 속도
  - [ ] 검색 latency (p50 / p95)
  - [ ] RAGAS `context_precision` / `context_recall`
- [ ] 결과를 `results/bench_embedding.csv`로 저장

### 종합
- [ ] README에 DB 비교표 + 임베딩 비교표 삽입

---

## Phase 6 — 코드 정리 & 마무리

- [ ] `src/` 모듈 정리 (`chunker.py`, `embedder.py`, `retriever.py`, `evaluator.py`)
- [ ] GitHub Actions CI 추가 (lint + 노트북 실행 smoke test)
- [ ] `Dockerfile` 추가 (주피터 서버 실행용)
- [ ] README 완성
  - 아키텍처 다이어그램
  - 벡터DB 비교 결과 표
  - 실행 방법 (`docker-compose up` → notebook 열기)

---

## 디렉토리 구조 (목표)

```
rag-eval-kit/
├── data/
│   ├── docs/               # title별 텍스트 파일 (squad_kor_v1 가공)
│   └── golden_set.json     # QA 골든셋 50개
├── notebooks/
│   ├── 00_dataset_prep.ipynb   # 데이터 가공 및 골든셋 추출
│   ├── 01_pipeline.ipynb       # RAG 파이프라인 구성
│   ├── 02_ragas_eval.ipynb     # RAGAS 평가
│   └── 03_vectordb_bench.ipynb # 벡터DB 비교
├── src/
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   └── evaluator.py
├── results/
│   ├── ragas_baseline.json
│   ├── bench_db.csv
│   └── bench_embedding.csv
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 진행 순서 요약

```
Phase 1 (환경)  →  Phase 2 (데이터 가공)  →  Phase 3 (파이프라인)
     ↓
Phase 4 (RAGAS 평가)  →  Phase 5 (DB 벤치마크)  →  Phase 6 (정리)
```
