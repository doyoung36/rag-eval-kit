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
ollama pull gemma4:e2b

# 5. 단위 디버깅은 노트북에서
jupyter lab
```

### 무거운 작업은 스크립트로

주요 로직은 `src/` 모듈로 분리되어 있어 노트북에서 `from src.xxx import ...`로 단위 확인하고,
1,000건이 넘는 임베딩 적재 같은 대량 작업은 스크립트로 실행한다 (tqdm 진행도 표시).

```bash
# 전체 코퍼스(1,417개) 청킹·임베딩·pgvector 적재 — 진짜 baseline용
python scripts/index_corpus.py
python scripts/index_corpus.py --max-docs 200   # 빠른 동작 확인용

# 골든셋 RAG 추론 수집 → results/eval_inputs.json
python scripts/collect_eval_inputs.py

# RAGAS 4개 지표 평가 (수집 + 평가 한 번에: --collect)
python scripts/run_ragas_eval.py --collect
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
├── notebooks/              # 단위 디버깅 — src 함수를 단계별로 호출
│   ├── 00_dataset_prep.ipynb   # 데이터 가공 및 골든셋 추출
│   ├── 01_pipeline.ipynb       # RAG 파이프라인 구성 (소량 확인)
│   ├── 02_ragas_eval.ipynb     # RAGAS 평가
│   └── 03_vectordb_bench.ipynb # 벡터DB 비교 (예정)
├── src/                    # 재사용 모듈 (editable 설치 → from src.x import)
│   ├── config.py               # 경로·하이퍼파라미터·접속정보 (cwd 독립)
│   ├── dataset.py              # KorQuAD 로드 → dedup → docs / 골든셋
│   ├── chunker.py              # 문서 로드 & 청킹
│   ├── embedder.py             # 임베딩 모델 (device 자동)
│   ├── retriever.py            # pgvector 엔진/적재/리트리버
│   ├── rag.py                  # LCEL RAG 체인
│   └── evaluator.py            # RAGAS 셰임/judge/수집/평가
├── scripts/                # 대량/무거운 작업 CLI (tqdm 진행도)
│   ├── index_corpus.py         # 전체 코퍼스 임베딩·적재
│   ├── collect_eval_inputs.py  # 골든셋 RAG 추론 수집
│   └── run_ragas_eval.py       # RAGAS 평가 실행
├── results/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## 진행 단계

> 상세 계획은 [plan.md](plan.md) 참고.

- [x] **Phase 1** — 환경 세팅 (docker-compose, 의존성, Ollama)
- [x] **Phase 2** — 데이터셋 준비 및 골든셋 추출
- [x] **Phase 3** — RAG 파이프라인 기본 구성
- [ ] **Phase 4** — RAGAS 평가
- [ ] **Phase 5** — 벡터DB / 임베딩 벤치마크
- [ ] **Phase 6** — 코드 정리 & 마무리

## 벤치마크 결과

> Phase 5 완료 후 DB 비교표 / 임베딩 비교표 삽입 예정.
