# `bench_one()` 완전 해부 — A to Z

> 노트북 [`notebooks/03_vectordb_bench.ipynb`](../notebooks/03_vectordb_bench.ipynb) 셀 `61dcbb29`에 정의된 핵심 측정 헬퍼.
> "함수 안에서 부른 함수가 또 함수를 부르는" 깊이를 한 번에 펼쳐서 정리한다.

---

## 0. 한 줄 요약

`bench_one`은 **`(벡터DB 1개, 이미 계산된 임베딩 벡터)` 한 조합**을 받아서
→ 적재(인덱싱) 시간, 검색 latency(p50/p95/mean), 검색 품질(hit_rate/mrr)을
→ **표의 한 행(dict)** 으로 돌려주는 함수다.

Round 1은 이걸 **DB만 바꿔 3번**, Round 2는 **임베딩만 바꿔 3번** 호출해 비교표를 채운다.

핵심 설계: **임베딩과 DB를 분리**한다. 임베딩(수백 초·수 GB)은 `bench_one` _바깥에서_ 한 번만 계산하고,
`bench_one`에는 다 계산된 raw 벡터만 넘긴다 → 그래서 검색 latency에 임베딩 비용이 절대 안 섞인다.

---

## 1. 시그니처와 입력

```python
def bench_one(backend_name, vectors, qvectors, *,
              embed_corpus_s=None, embed_model=None,
              collection=config.BENCH_COLLECTION):
```

| 인자             | 뜻                                              | 누가 만들어 주나           |
| ---------------- | ----------------------------------------------- | -------------------------- |
| `backend_name`   | `"pgvector"` / `"qdrant"` / `"neo4j"` 중 하나   | 루프가 넘김                |
| `vectors`        | **청크 임베딩** (13808개 × 1024차원 리스트)     | `bench.embed_corpus(...)`  |
| `qvectors`       | **골든 질문 임베딩** (50개 × 1024차원)          | `bench.embed_queries(...)` |
| `embed_corpus_s` | 코퍼스 임베딩에 걸린 초 (기록만, 측정 안 함)    | 호출부에서 전달            |
| `embed_model`    | 결과 행에 적을 임베딩 모델명                    | 호출부에서 전달            |
| `collection`     | 테이블/컬렉션/인덱스 이름 (기본 `bench_chunks`) | config                     |

`*` 뒤는 **키워드 전용 인자** — 위치로 못 넘기고 `embed_corpus_s=...`처럼 이름을 붙여야 한다 (호출부 가독성·실수 방지).

**왜 `vectors`와 `qvectors`를 따로 받나?** instruct 계열 임베딩은 "문서용 프리픽스"와 "질의용 프리픽스"가 달라서,
문서 임베딩(`embed_documents`)과 질의 임베딩(`embed_query`)을 분리해 계산하기 때문이다 ([bench.py:28-33](../src/bench.py#L28-L33)).

---

## 2. 호출 트리 (깊이 펼치기)

이게 네가 헷갈린 "뎁스"의 전체 그림이다. 들여쓰기 = 호출 깊이.

```
bench_one(backend_name, vectors, qvectors, ...)
│
├─ bench.embedding_dim(vectors)                      → 차원 1024 (vectors[0] 길이)
│
├─ get_backend(backend_name, collection)             → 백엔드 객체 생성 (DB 커넥션 오픈)
│     └─ (지연 import) PgVectorBackend/QdrantBackend/Neo4jBackend(...)
│           └─ config.pg_raw_conn() / qdrant_config() / neo4j_config()   ← 접속정보
│
├─ bench.index_corpus(be, dim, vectors, texts, titles)        ← 인덱싱 시간 측정 구간
│     ├─ be.recreate(dim)        # 스키마/컬렉션 비우고 재생성 (측정 제외)
│     ├─ t0 = perf_counter()     # ⏱ 여기서부터 잰다
│     ├─ be.add(vectors, ...)    # 배치 적재 (256개씩)
│     ├─ be.finalize()           # ANN 인덱스 질의가능 상태로
│     └─ return perf_counter()-t0   → index_s
│
├─ bench.run_queries(be, qvectors, K)                        ← 검색 latency 측정 구간
│     └─ for v in qvectors:      # 질의 50개 순차 (동시부하 아님)
│           ├─ t0 = perf_counter()
│           ├─ be.search(v, K)   # top-K 코사인 검색 → [Hit, ...]
│           └─ latency_ms 기록
│        return results, latencies
│
├─ be.count()                    # 실제 적재된 벡터 수 (검증용)
├─ be.close()                    # finally — 성공/실패 무관 커넥션 정리
│
├─ bench.latency_stats(latencies)               → {p50, p95, mean} ms (numpy percentile)
│
└─ retrieval_metrics.evaluate_retrieval(results, golden, K)  → {hit_rate@K, mrr@K}
      └─ _first_relevant_rank(hits, ground_truth, K)
            └─ is_relevant(hit.text, ground_truth)
                  └─ _norm(s)    # 공백제거+소문자 후 부분일치
```

`bench_one`이 직접 부르는 건 **6개**(`embedding_dim`, `get_backend`, `index_corpus`, `run_queries`,
`latency_stats`, `evaluate_retrieval`)뿐이고, 나머지는 그 안에서 한 단계씩 더 내려가는 것이다.

---

## 3. 단계별 상세 (위→아래 실행 순서)

### ① 차원 구하기 — `bench.embedding_dim(vectors)`

```python
return len(vectors[0])
```

첫 벡터의 길이 = 임베딩 차원(BGE-m3-ko면 1024). DB 테이블/컬렉션을 만들 때 `vector(1024)`처럼
차원을 박아야 해서 가장 먼저 구한다. ([bench.py:36-38](../src/bench.py#L36-L38))

### ② 백엔드 생성 — `get_backend(backend_name, collection)`

이름 문자열 → 실제 DB 클라이언트 객체로 바꾸는 **팩토리**다. ([backends/\_\_init\_\_.py:17-42](../src/backends/__init__.py#L17-L42))

- `"pgvector"` → `PgVectorBackend(table, conninfo=config.pg_raw_conn())`
- `"qdrant"` → `QdrantBackend(collection, host, port)` (`config.qdrant_config()`)
- `"neo4j"` → `Neo4jBackend(collection, uri, user, password)` (`config.neo4j_config()`)

**왜 import를 함수 안에서 하나?** (지연 로드) 셋 다 클라이언트 라이브러리가 다른데,
qdrant만 쓸 때 psycopg/neo4j까지 설치돼 있을 필요가 없게 하려는 것이다.

생성자(`__init__`)가 **이미 DB 커넥션을 연다** — 그래서 아래 `try/finally`로 반드시 닫아야 한다.

세 백엔드는 전부 [`base.py`](../src/backends/base.py)의 `VectorBackend` 추상 인터페이스를 구현한다.
그래서 `bench_one`은 어느 DB인지 몰라도 똑같이 `recreate/add/finalize/search/count/close`만 부르면 된다
(= 다형성. 이게 DB 3종을 한 함수로 비교할 수 있는 비결).

### ③ 인덱싱 — `bench.index_corpus(be, dim, vectors, texts, titles)`

([bench.py:41-58](../src/bench.py#L41-L58)) 측정의 첫 번째 핵심.

```python
backend.recreate(dim)            # 측정 밖: 기존 데이터 삭제 + 빈 스키마 생성
t0 = time.perf_counter()         # ⏱ 여기서부터
backend.add(vectors, texts, titles, batch_size=256)   # 적재
backend.finalize()               # 인덱스 구축 완료 대기
return time.perf_counter() - t0  # = index_s
```

**중요 — 무엇을 "인덱싱 시간"으로 보나:** `recreate`(스키마 만들기)는 **제외**하고
**`add`(적재) + `finalize`(인덱스 구축)** 만 잰다. DB마다 인덱스를 만드는 시점이 달라서
이 둘을 묶어 wall-clock으로 재야 공정하다:

| DB       | `add`                    | `finalize`                                                  |
| -------- | ------------------------ | ----------------------------------------------------------- |
| pgvector | INSERT만                 | **여기서** HNSW 인덱스 생성 (bulk-load 후 인덱싱이 더 빠름) |
| Qdrant   | upsert하며 점진적 인덱싱 | no-op (이미 다 됨)                                          |
| Neo4j    | 노드 CREATE              | 인덱스가 ONLINE 될 때까지 `db.awaitIndexes` 블로킹          |

→ 어느 DB든 "넣고 → 검색 가능해질 때까지" 전체를 잡으므로 비교가 공정해진다.

### ④ 검색 — `bench.run_queries(be, qvectors, K)`

([bench.py:61-75](../src/bench.py#L61-L75)) 측정의 두 번째 핵심.

```python
for v in qvectors:                       # 질의 50개를 하나씩
    t0 = time.perf_counter()
    hits = backend.search(v, k)          # top-K 코사인 검색
    latencies_ms.append((perf_counter()-t0)*1000)
    results.append(hits)
return results, latencies_ms
```

**왜 순차(병렬 아님)?** 처리량(throughput)이 아니라 **단건 검색 latency**를 재는 게 목적이라,
한 번에 하나씩 돌려 질의별 소요(ms)를 깨끗하게 잡는다.

`backend.search`는 DB마다 SQL/API가 다르지만 (`<=>` 코사인 거리 / `query_points` / `db.index.vector.queryNodes`)
**반환은 전부 `Hit(text, score, title)` 리스트로 통일**된다 ([base.py:24-30](../src/backends/base.py#L24-L30)).
이 통일 덕에 다음 단계(품질 평가)가 DB를 신경 안 써도 된다.

### ⑤ 검증 + 정리 — `be.count()` / `be.close()`

```python
    n = be.count()      # 실제 적재된 벡터 수 (13808이 맞는지 확인용)
finally:
    be.close()          # 성공/예외 무관하게 커넥션 닫기 (누수 방지)
```

`count`/`close`가 `try/finally`에 들어가 있어서, 중간에 터져도 커넥션은 반드시 닫힌다.

### ⑥ latency 요약 — `bench.latency_stats(latencies)`

([bench.py:78-87](../src/bench.py#L78-L87)) 50개 latency 리스트 → numpy로 분포 요약.

```python
{"latency_p50_ms": p50, "latency_p95_ms": p95, "latency_mean_ms": mean}
```

p95(상위 5% 꼬리)가 평균보다 실제 체감 성능을 잘 보여줘서 같이 본다.

### ⑦ 검색 품질 — `retrieval_metrics.evaluate_retrieval(results, golden, K)`

([retrieval_metrics.py:42-58](../src/retrieval_metrics.py#L42-L58)) **LLM 없이** 결정론적으로 품질 측정.

판정 로직을 안쪽부터 보면:

1. `_norm(s)` — 공백 제거 + 소문자화 (표면형 차이 흡수). [retrieval_metrics.py:23-25](../src/retrieval_metrics.py#L23-L25)
2. `is_relevant(hit_text, ground_truth)` — **정답 스팬이 청크 본문에 부분일치하면 관련**. [:28-31](../src/retrieval_metrics.py#L28-L31)
3. `_first_relevant_rank(hits, gt, k)` — top-k 중 정답을 담은 **첫 청크의 순위**(1-based), 없으면 `None`. [:34-39](../src/retrieval_metrics.py#L34-L39)
4. 집계:
   - **hit_rate@k** = (정답을 하나라도 찾은 질의 수) / (전체 질의 수)
   - **mrr@k** = Σ(1/첫정답순위) / 질의 수 — 정답을 **얼마나 위에** 띄웠는지까지 본다

> `results[i]` ↔ `golden[i]` 가 **같은 순서**여야 한다 (함수 첫 줄 `assert`로 강제). 둘 다 `questions` 순서를 따른다.

---

## 4. 반환값 — 표의 한 행이 만들어지는 과정

```python
row = {'backend': backend_name, 'embed_model': embed_model,
       'n_chunks': n, 'dim': dim, 'index_s': round(t_index, 2)}
if embed_corpus_s is not None:
    row['embed_corpus_s'] = round(embed_corpus_s, 2)       # Round 2 핵심 지표
row.update(latency_stats(...))        # p50/p95/mean 합치기
row.update(evaluate_retrieval(...))   # hit_rate/mrr 합치기
return row, results
```

최종 `row` 하나는 이렇게 생긴다 (Round 1 pgvector 실제값):

| backend  | embed_model | n_chunks | dim  | index_s | embed_corpus_s | p50   | p95   | mean  | hit_rate@4 | mrr@4  |
| -------- | ----------- | -------- | ---- | ------- | -------------- | ----- | ----- | ----- | ---------- | ------ |
| pgvector | BGE-m3-ko   | 13808    | 1024 | 20.24   | 1067.37        | 30.40 | 34.25 | 31.33 | 0.98       | 0.9367 |

- **`row`** → 루프가 리스트에 쌓아 `pd.DataFrame`으로 만든다 (비교표).
- **`results`** → 셀 4·5의 RAGAS 평가(`context_precision/recall`)에서 재사용한다.

---

## 5. 두 번 호출되는 방식 (Round 1 vs Round 2)

```python
# Round 1 — 임베딩 1회 계산 → DB만 바꿔 3번 (vectors/qvectors 재사용)
vectors, t_embed = bench.embed_corpus(embeddings, texts)
qvectors, _      = bench.embed_queries(embeddings, questions)
for name in config.BENCH_BACKENDS:          # pgvector, qdrant, neo4j
    row, res = bench_one(name, vectors, qvectors, embed_corpus_s=t_embed, ...)
# → 같은 벡터라 hit_rate/mrr은 비슷, 차이는 index_s·latency에서 난다.

# Round 2 — DB 고정 → 임베딩만 바꿔 3번 (모델마다 새로 임베딩)
for model in config.BENCH_EMBED_MODELS:
    emb = embedder.build_embeddings(model)
    vecs, te  = bench.embed_corpus(emb, texts)
    qvecs, _  = bench.embed_queries(emb, questions)
    row, res = bench_one(WINNER_DB, vecs, qvecs, embed_corpus_s=te, embed_model=model, ...)
# → 벡터가 모델마다 달라 hit_rate/mrr·RAGAS가 실제로 갈린다.
```

같은 `bench_one`이지만 **무엇을 고정하고 무엇을 바꾸느냐**가 다르다.
이게 "조합 측정 헬퍼"라고 부르는 이유 — `(DB, 임베딩)` 격자의 칸 하나를 채우는 함수다.

---

## 6. 핵심 3가지

1. **임베딩은 바깥, 측정은 안.** `bench_one`은 임베딩을 절대 직접 안 한다. 다 된 벡터만 받아서 적재·검색만 잰다 → latency가 깨끗하다.
2. **DB가 뭐든 인터페이스는 하나.** `recreate/add/finalize/search/count/close` 6개 메서드 + `Hit` 반환형이 모든 DB에 동일 → 한 함수로 3종 비교 가능.
3. **두 측정 구간만 기억.** `index_corpus`(=인덱싱 시간) 와 `run_queries`(=검색 latency). 나머지(`embedding_dim`, `count`, `latency_stats`, `evaluate_retrieval`)는 보조·집계다.
