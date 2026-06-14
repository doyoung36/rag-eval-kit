# 주피터 서버 실행용 이미지 — 노트북에서 src 모듈을 그대로 import 한다.
# 벡터DB(pgvector/Qdrant/Neo4j)는 docker-compose의 다른 서비스로 뜨고,
# 이 컨테이너는 서비스명(예: PGVECTOR_HOST=pgvector)으로 접속한다(compose에서 주입).
FROM python:3.11-slim

# uv (의존성 설치기) 복사
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1) 의존성 레이어 — 패키지 메타 + src만 먼저 복사해 캐시를 살린다.
#    (hatchling이 packages=["src"]를 빌드하므로 editable 설치에 src가 필요)
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache -e ".[dev]"

# 2) 나머지 소스(노트북·scripts·data 등)
COPY . .

EXPOSE 8888

# 로컬 개발용 — 토큰 없이 바로 접속. 외부 노출 환경에서는 토큰을 설정할 것.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
