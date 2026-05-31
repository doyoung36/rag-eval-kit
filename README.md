# rag-eval-kit

RAGAS로 RAG 파이프라인 품질을 측정하고, **pgvector / Qdrant / Neo4j** 벡터DB와 임베딩 모델을 비교 벤치마크하는 키트.

## 목적

- KorQuAD(`squad_kor_v1`) 기반 한국어 골든셋으로 RAG 파이프라인 구성
- RAGAS 4대 지표(`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`)로 품질 평가
- 동일 조건에서 벡터DB 3종 / 임베딩 3종을 latency·정확도 기준으로 비교

## 기술 스택

| 역할           | 선택                                                        |
| -------------- | ----------------------------------------------------------- |
| 데이터셋       | `KorQuAD/squad_kor_v1` (HuggingFace)                        |
| RAG 추론 LLM   | Ollama — `gemma4-e2b` (로컬)                                |
| RAGAS 평가 LLM | `langchain-anthropic` — Claude (API)                        |
| 임베딩         | Qwen3-Embedding-0.6B / embeddingGemma / dragonkue/BGE-m3-ko |
| 벡터DB         | pgvector / Qdrant / Neo4j                                   |

## 빠른 시작

```bash
# 1. 환경 변수 준비
cp .env.example .env
# .env 에 ANTHROPIC_API_KEY 입력

# 2. 벡터DB 컨테이너 실행 (pgvector / Qdrant / Neo4j)
docker compose up -d
docker compose ps        # 모두 healthy 확인

# 3. Python 의존성 설치
uv sync                  # 또는: pip install -e ".[dev]"

# 4. Ollama 모델 준비 (로컬 추론 LLM)
ollama pull gemma4-e2b

# 5. 노트북 실행
jupyter lab
```

### 접속 정보

| 서비스   | 주소                                   | 계정                   |
| -------- | -------------------------------------- | ---------------------- |
| pgvector | `localhost:5432`                       | `rag` / `rag`          |
| Qdrant   | http://localhost:6333/dashboard        | -                      |
| Neo4j    | http://localhost:7474                  | `neo4j` / `ragpassword`|

## 디렉토리 구조

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
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## 진행 단계

> 상세 계획은 [plan.md](plan.md) 참고.

- [x] **Phase 1** — 환경 세팅 (docker-compose, 의존성, Ollama)
- [ ] **Phase 2** — 데이터셋 준비 및 골든셋 추출
- [ ] **Phase 3** — RAG 파이프라인 기본 구성
- [ ] **Phase 4** — RAGAS 평가
- [ ] **Phase 5** — 벡터DB / 임베딩 벤치마크
- [ ] **Phase 6** — 코드 정리 & 마무리

## 벤치마크 결과

> Phase 5 완료 후 DB 비교표 / 임베딩 비교표 삽입 예정.
