# AI/LLM 사이드 프로젝트 커리큘럼

> 목표: 회사 업무 외 개인 깃허브를 채울 포트폴리오용 프로젝트 로드맵.
> 대형 프로젝트 지양, 프로토타입 위주. 짬날 때마다 레이어별로 쌓아나가는 구조.
>
> 공통 원칙
> - 모든 레포에 Dockerfile + docker-compose 기본 적용, GitHub Actions CI 연결.
> - 프론트엔드 스택은 **Next.js** (AI 생태계 표준, Vercel 배포로 데모 링크 바로 공유 가능) Angular는 회사 이력에서 충분히 증명됨.
> - 백엔드는 기존 강점인 **Nest.js** 유지.

---

## 레이어 구조

| 레이어 | 분야 | 핵심 목적 |
|---|---|---|
| L1 | 추론/서빙 | LLM이 어떻게 서빙되는지 이해 및 실습 |
| L2 | 평가/실험 | 모델과 RAG 품질을 측정하는 도구 |
| L3 | 파인튜닝 | 모델 학습 경험 정리 및 재현 가능한 레시피 |
| L4 | LangChain/LangGraph/MCP 활용 | 에이전트·워크플로우 패턴을 실제 툴로 구현 |
| L5 | Observability | LLM 호출 트레이싱, 비용, 품질 모니터링 |
| L6 | 애플리케이션 | L1~L5를 엮어 실제 쓸 수 있는 서비스로 완성 |

---

## 프로젝트 목록 (레이어 × 소요 시간)

### L1 · 추론/서빙

| 소요 시간 | 레포명 | 내용 |
|---|---|---|
| 1시간 | `llm-streaming-demo` | SSE 기반 스트리밍 응답 최소 구현 (Nest.js + Next.js). 서빙 레이어의 데이터 흐름 이해 |
| 하루 | `local-llm-serving` | Ollama 또는 vLLM으로 로컬/인스턴스 모델 서빙 + FastAPI 래핑. 직접 서버 구성 경험 정리 |

### L2 · 평가/실험

| 소요 시간 | 레포명 | 내용 |
|---|---|---|
| 하루 | `rag-eval-kit` | RAGAS 활용 RAG 파이프라인 품질 측정. pgvector / Qdrant / Weaviate 벡터DB 비교 벤치마크 포함 |
| 하루 | `korean-llm-benchmark` | KLUE, KorNLI 등 한국어 태스크로 모델 성능 비교하는 평가 스크립트. 실험 노트 형태로 정리 |

### L3 · 파인튜닝

| 소요 시간 | 레포명 | 내용 |
|---|---|---|
| 하루 | `finetune-recipes` | Llama 3.1 / Gemma4 / Qwen3 대상 PEFT(LoRA, QLoRA), CPT 설정 모음 + 실험 결과 비교 노트북 |
| 일주일 | `finetune-pipeline` | 데이터 전처리 → 학습 → 평가까지 이어지는 재현 가능한 파이프라인 (DVC or MLflow 연동) |

### L4 · LangChain/LangGraph/MCP 활용

| 소요 시간 | 레포명 | 내용 |
|---|---|---|
| 1시간 | `langgraph-patterns` | conditional edge, loop, branch, subgraph 패턴 예제 모음 |
| 1시간 | `lcel-chain-examples` | LangChain LCEL 기반 다양한 체인 구성 예제 (sequential, parallel, branching) |
| 1시간 | `rag-minimal` | 파일 하나 올리면 Q&A 되는 가장 단순한 RAG. pgvector + LangChain 최소 구성 |
| 하루 | `tool-calling-agent` | 웹검색 + 계산기 + 날씨 API 붙인 도구 사용 에이전트. LangGraph tool node 활용 |
| 하루 | `reflection-agent` | 에이전트가 자기 답변을 스스로 검토 후 재시도하는 self-correction 루프 |
| 하루 | `mcp-tool-server` | MCP(Model Context Protocol) 서버 직접 구현 + LangGraph 에이전트 연결. 2026년 에이전틱 AI 표준 인프라 |
| 하루 | `korean-doc-classifier` | 한국어 문서 분류/요약 API. FastAPI + Docker. R&D가 아닌 서비스 개발 역량 타겟 |
| 일주일 | `multi-agent-researcher` | Supervisor + Worker 패턴. 주제 입력 → 검색/요약/합성 에이전트가 분업해 리포트 생성 |
| 일주일 | `ltm-chatbot` | 대화 내용을 장기 기억하는 챗봇. Mem0 or pgvector 기반 직접 설계한 memory store 적용 |
| 일주일 | `hitl-workflow-demo` | LangGraph `interrupt` 활용. 에이전트가 민감 액션 전에 사람 승인 요청하는 워크플로우 |

### L5 · Observability

| 소요 시간 | 레포명 | 내용 |
|---|---|---|
| 하루 | `langsmith-tracing-demo` | LangSmith 연동해 LLM 호출 트레이싱, 토큰 비용, latency 추적하는 예제 프로젝트 |

### L6 · 애플리케이션

> L6 공통: 에러 핸들링, 로깅, 환경변수 분리, rate limiting 포함. README에 아키텍처 다이어그램 + 설계 트레이드오프 기술.

| 소요 시간 | 레포명 | 내용 |
|---|---|---|
| 일주일 | `doc-intelligence-api` | 문서 업로드 → 자동 태그/요약/Q&A 통합 API. L4 RAG + L2 eval 패턴 결합. FastAPI + Docker |
| 일주일 | `smart-work-assistant` | 할일 자동 생성 + LTM + Multi-agent + MCP 연동을 엮은 미니 업무 어시스턴트. Next.js + Nest.js + LangGraph |

---

## 권장 진행 순서

```
1단계 (각 1시간, 깃허브 활동 빠르게 채우기)
  └─ langgraph-patterns
  └─ lcel-chain-examples
  └─ rag-minimal
  └─ llm-streaming-demo

2단계 (각 하루, 기능 하나씩 깊게)
  └─ tool-calling-agent
  └─ reflection-agent
  └─ mcp-tool-server          ← MCP 표준 대응
  └─ finetune-recipes
  └─ rag-eval-kit             ← 벡터DB 비교 포함
  └─ korean-llm-benchmark
  └─ korean-doc-classifier    ← 서비스 개발 역량 타겟

3단계 (각 일주일, 완성도 있는 결과물)
  └─ multi-agent-researcher
  └─ ltm-chatbot              ← 석박사 사업 LTM 연구와 연결
  └─ hitl-workflow-demo

4단계 (통합 애플리케이션, 프로덕션 수준)
  └─ doc-intelligence-api     ← Docker + 프로덕션 시그널 포함
  └─ smart-work-assistant     ← MCP 연동 포함
```

---

## 비고

- `finetune-recipes`는 이미 수행한 실험들을 정리하는 것이므로 착수 시점을 앞당길 수 있음
- `ltm-chatbot`은 현재 석박사 양성 사업에서 진행 중인 LTM 연구와 직접 연결되는 주제
- `hitl-workflow-demo`는 창업중심대학 계획서 아이디어를 LangGraph로 구현한 것
- L5 Observability는 독립 레포보다 다른 프로젝트에 LangSmith 연동 형태로 녹이는 것도 가능
- Docker는 모든 레포 공통 적용, GitHub Actions CI까지 연결하면 "배포를 아는 사람" 시그널
