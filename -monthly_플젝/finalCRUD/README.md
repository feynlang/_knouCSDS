# todolist
- [ ] 요구사항 문서작성
- [ ] DB테이블 설계
- [ ] API 설계
- [ ] ORM 모델링
- [ ] 캐싱 적용(redis)
- [ ] 로그/모니터링 구조
- [ ] 에러 처리 미들웨어
- [ ] 유닛 테스트+Integration 테스트
- [ ] Docker+Nginx 리버스 프록시
- [ ] 배포(EC2 or Render)

# detail
## 1. 요구사항 문서작성
- 서비스 기능
- 비기능(성능/보안/운영)
요구사항 명확->개발범위 고정
### 작업항목

### 샘플항목 진척도
### 산출물
### 체크리스트


<!-- 프로젝트 설계 단계별 구체화 -->

# 백엔드 프로젝트 #4 실서비스 수준 API — 단계별 구체화
아래는 “초기 스타트업 최소기능” 수준의 실서비스급 API를 완성하기 위한 **각 단계별 작업 항목, 산출물, 구현 예시(코드/설정), 체크리스트, 권장 기술 스택**이다. 가능한 한 바로 적용할 수 있게 구체적인 샘플(스키마·엔드포인트·설정)을 포함했다.

## ✔ To-do 리스트
1. **요구사항 문서 작성**
2. **DB 테이블 설계(ERD)**
3. **API 설계**
4. **ORM 모델링**
5. **캐싱 적용(Redis)**
6. **로그/모니터링 구조**
7. **에러 처리 미들웨어**
8. **Unit test + Integration test**
9. **Docker + Nginx 리버스 프록시**
10. **배포(EC2 or Render)**

## 1. 요구사항 문서 작성
- 목적: 서비스 기능, 비기능(성능/보안/운영) 요구사항을 명확히 해 개발 범위를 고정.

### 작업 항목
- 기능 목록(유저/인증/리소스/관리자 등)
- 각 기능의 API 시나리오(입출력, 기대 응답)
- 비기능 요구사항: SLA, RPS(요청량), 응답시간 목표, 보안(HTTPS, OWASP), 데이터 보존 정책, 스케일링 요구
- 우선순위(MVP/Phase2)
- 수락 기준(acceptance criteria)

### 샘플 항목 (간단)
- 사용자: 가입/로그인(JWT), 프로필 조회/수정
- 리소스: CRUD (게시물, 댓글)
- 관리: 유저 차단, 로그 액세스
- 비기능: 99.9% 가용성 목표, 500 RPS 예상, 평균 응답 < 200ms

### 산출물
- `requirements.md` (기능/비기능/용어정의)
- 사용자 스토리/프로토타입 링크(있으면 포함)

### 체크리스트
- [ ]  모든 엔티티와 행위 정의됨
- [ ]  수락 기준 명시됨
- [ ]  보안·백업·복구 요구 포함됨

### 권장 기술
- 문서: Markdown + Git (repo 루트에 두기)
- 협업: Notion / Confluence (선택)


## 2. DB 테이블 설계(ERD)
- 목적: 데이터 모델 확정 — 정규화/인덱싱/관계 설계.

### 작업 항목
- 엔티티 리스트 + 속성(타입, null/unique)
- PK/FK, 인덱스, 제약조건
- 조회 성능 고려한 역정규화 여부
- 마이그레이션 전략 (Alembic / Flyway)

### 샘플 ERD (간단한 SNS 스타일)
```sql
-- users
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  hashed_password VARCHAR(255) NOT NULL,
  username VARCHAR(50) UNIQUE NOT NULL,
  bio TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- posts
CREATE TABLE posts (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  body TEXT,
  is_published BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
CREATE INDEX idx_posts_userid_createdat ON posts(user_id, created_at DESC);

-- comments
CREATE TABLE comments (
  id UUID PRIMARY KEY,
  post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  body TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```
### 인덱스 전략
- 자주 조회하는 컬럼(예: `posts(user_id, created_at)`, `users(email)`)에 인덱스
- 텍스트 검색이 필요하면 PostgreSQL `GIN` + `tsvector`

### 마이그레이션
- Python: Alembic
- Node: TypeORM / Knex migrations

### 산출물
- `erd.png` 또는 dbml 파일
- 마이그레이션 스크립트

### 체크리스트
- [ ]  FK/제약조건 정의됨
- [ ]  인덱스 전략 기술됨
- [ ]  샘플 쿼리 성능 고려됨


## 3. API 설계
- 목적: RESTful(혹은 GraphQL) API 계약 정의 — 엔드포인트, 메서드, 페이로드, 상태 코드, 에러 포맷.

### 작업 항목
- 엔드포인트 목록 + HTTP 메서드
- 요청/응답 JSON 스키마 (예: OpenAPI/Swagger)
- 인증 방식(JWT/Bearer, OAuth2)
- 페이징/정렬/필터 규칙
- 에러 포맷 표준화

### 샘플 엔드포인트 (REST + JWT)
```
POST /api/v1/auth/signup
POST /api/v1/auth/login
GET  /api/v1/users/{id}
PATCH /api/v1/users/{id}
GET  /api/v1/posts?page=1&size=20
POST /api/v1/posts
GET  /api/v1/posts/{id}
PATCH /api/v1/posts/{id}
DELETE /api/v1/posts/{id}
POST /api/v1/posts/{id}/comments
```
### 샘플 응답 표준 (JSON)
```json
// 성공
{ "status": "ok", "data": { ... } }

// 에러
{
  "status": "error",
  "error": {
    "code": "INVALID_INPUT",
    "message": "email is required",
    "details": { "email": "must be a valid email" }
  }
}
```
### 문서화
- OpenAPI 3.0 스펙 파일(`openapi.yaml`)
- 자동 생성: Swagger UI / Redoc

### 체크리스트
- [ ]  OpenAPI 문서 완성
- [ ]  인증/권한 흐름 명시
- [ ]  응답/에러 표준화

### 권장 도구
- Node: Express + swagger-jsdoc
- Python: FastAPI (자동 문서화) 또는 Flask + Connexion


## 4. ORM 모델링
- 목적: DB 스키마와 서비스 계층 매핑, 유효성/관계 처리.

### 작업 항목
- ORM 엔티티 정의 (모델, relation)
- 도메인 로직(저장/조회/트랜잭션) 캡슐화
- DTO/Schema (pydantic / Joi / class-transformer)로 입력 검증

### 샘플 (Python + SQLAlchemy + Pydantic)
```python
# models.py
class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    username = Column(String, unique=True, nullable=False)
    posts = relationship("Post", back_populates="author")

# schemas.py (pydantic)
class UserCreate(BaseModel):
    email: EmailStr
    password: constr(min_length=8)
    username: constr(min_length=3)
```
### 트랜잭션/동시성
- 중요한 multi-step 작업은 DB 트랜잭션으로 묶기
- 예: 포스팅 + 알림 발행은 트랜잭션 + outbox 패턴 고려

### 테스트용 DB
- 테스트는 독립 DB(또는 in-memory)에서 실행
- pytest fixtures / docker-compose 테스트 DB

### 체크리스트
- [ ]  모델과 마이그레이션 일치
- [ ]  입력 검증 분리
- [ ]  트랜잭션 경계 정의


## 5. 캐싱 적용(Redis)
- 목적: 읽기 성능 개선, 세션/토큰 블랙리스트, rate-limiting, pub/sub.

### 사용 사례 & 전략
- Response cache: 인기 게시물 `GET /posts` 캐시 (TTL 30s–5m)
- Object cache: user profile 캐시(변경 시 무효화)
- Session/token blacklist: 로그아웃 시 토큰을 Redis에 블랙리스트
- Rate limiting: IP 또는 API key 별 request 카운트 (sliding window)
- Pub/Sub: 비동기 알림/이벤트 전달(옵션)

### 샘플 키 설계
```
posts:list:page:{page}:size:{size} -> JSON (TTL 60)
user:profile:{user_id} -> JSON (TTL 300)
rate:ip:{ip}:{endpoint} -> int (expire window)
```
### 코드(예시, Node + ioredis)
```jsx
const cacheKey = `posts:list:page:${page}:size:${size}`;
const cached = await redis.get(cacheKey);
if (cached) return JSON.parse(cached);
// ... fetch from db
await redis.set(cacheKey, JSON.stringify(result), 'EX', 60);

```
### 무효화 전략
- 쓰기(Post create/update/delete) 시 관련 캐시 키 삭제 또는 pub/sub으로 캐시 갱신

### 체크리스트
- [ ]  Redis 인스턴스 설정(ACL, 비밀번호)
- [ ]  캐시 키 규칙 문서화
- [ ]  캐시 무효화 지점 정의
- [ ]  모니터링: Redis 메모리/eviction

### 권장
- Managed Redis (AWS Elasticache / Upstash) for production


## 6. 로그/모니터링 구조
- 목적: 문제 탐지/트러블슈팅/성능 분석.

### 로깅 (앱)
- 구조화 로그(JSON: timestamp, level, service, trace_id, user_id, msg)
- 레벨: ERROR/WARN/INFO/DEBUG
- 요청/응답 로깅: sensitive 필드 마스킹(비밀번호, 토큰)
- correlation id: `X-Request-ID` 또는 OpenTelemetry trace id

### 로그 수집/검색
- ELK Stack (Filebeat → Elasticsearch → Kibana) 또는 Grafana Loki + Promtail
- 로그 TTL / 인덱스 롤오버 전략

### 모니터링(메트릭)
- Prometheus (app-exporter) + Grafana 대시보드
- 모니터 항목: request rate, error rate, p95/p99 latency, DB connections, Redis memory, CPU/RAM
- 알림: Grafana Alerting / PagerDuty/Slack 연동

### 분산 추적
- OpenTelemetry (traces) → Jaeger / Tempo (Grafana Tempo)
- DB 쿼리/외부 API 호출 추적

### 샘플 모니터링 항목
- `http_requests_total{method, endpoint, status}`
- `http_request_duration_seconds_bucket`
- `db_query_duration_seconds`

### 체크리스트
- [ ]  구조화 로그 적용
- [ ]  Prometheus metrics 노출
- [ ]  대시보드(요청량/latency/error) 구성
- [ ]  알림 규칙 설정 (예: error rate > 5% for 5m)


## 7. 에러 처리 미들웨어
- 목적: 일관된 에러 응답, 예외 로깅, 보안상 내부 정보 노출 금지.

### 구조
- 공통 에러 타입 정의 (ValidationError, NotFound, Unauthorized, Forbidden, InternalError)
- 미들웨어: 예외 포착 → 로깅(trace_id 포함) → 클라이언트 응답 변환
- 개발 모드에서는 스택트레이스 포함(단, 절대로 production에서 노출 금지)

### 샘플 (Express)
```jsx
app.use((err, req, res, next) => {
  const traceId = req.headers['x-request-id'] || generateId();
  logger.error({ err, traceId, path: req.path });
  if (err instanceof ValidationError) {
    return res.status(400).json({ status: 'error', error: { code: 'INVALID_INPUT', message: err.message } });
  }
  // default
  res.status(500).json({ status: 'error', error: { code: 'INTERNAL', message: 'Internal server error', trace_id: traceId } });
});
```
### 보안 주의
- 민감한 내부 메시지/쿼리/스택 노출 금지
- 에러 코드/trace_id만 제공하여 내부 로그에서 추적 가능하게 함

### 체크리스트
- [ ]  에러 포맷 표준화
- [ ]  모든 비동기 경로에서 예외 처리 적용
- [ ]  에러 로깅에 trace_id 포함

## 8. Unit test + Integration test
- 목적: 기능 신뢰성 확보, 회귀 방지.

### 전략
- Unit tests: 비즈니스 로직, 유틸리티, 서비스 레이어 (mock DB)
- Integration tests: 실제 DB(테스트 전용), Redis, 외부 서비스는 모킹 또는 테스트 스텁
- E2E(선택): API 전체 플로우를 검증 (테스트 DB + docker-compose)

### 도구 예시
- Python: pytest, pytest-asyncio, requests, factory-boy
- Node: Jest + supertest, nock (외부 API 모킹)

### 샘플 구성 (Node)
```
/tests
  /unit
    user.service.test.js
  /integration
    auth.int.test.js  # runs against test DB, uses docker-compose to spin Postgres
```
### CI 연동
- GitHub Actions: PR 시 테스트 자동 실행, Coverage 보고
- Test matrix: node versions 등

### 체크리스트
- [ ]  핵심 경로에 대해 unit coverage >= 70% (프로젝트 기준)
- [ ]  CI에서 test 통과 없이 merge 금지
- [ ]  DB 마이그레이션/시드 테스트 흐름 자동화


## 9. Docker + Nginx 리버스 프록시
목적: 컨테이너화 + 정적 파일/SSL termination/로드 분배.

### Dockerfile (예: FastAPI)
```docker
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip && pip install poetry && poetry install --no-dev
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
```
### docker-compose (dev)
```yaml
version: '3.8'
services:
  app:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [db, redis]
  db:
    image: postgres:15
    environment: POSTGRES_PASSWORD: example
  redis:
    image: redis:7
```
### Nginx (reverse proxy)
```
server {
  listen 80;
  server_name example.com;
  location / {
    proxy_pass http://app:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Request-ID $request_id;
  }
}
```
### SSL
- Production: Nginx에서 Let's Encrypt certbot or use cloud LB (AWS ALB) for TLS termination

### 체크리스트
- [ ]  Multi-stage build로 이미지 최적화
- [ ]  최소 권한으로 컨테이너 실행 (non-root)
- [ ]  Healthcheck 정의 (Dockerfile `HEALTHCHECK` or docker-compose)
- [ ]  Nginx에서 요청 헤더 전달 및 client_max_body_size 설정


## 10. 배포(EC2 or Render)
- 목적: 프로덕션에 서비스 배포, 운영 준비.

### 옵션 A — EC2 (관리형 인프라)
- 구성: EC2 (ASG), RDS(Postgres), ElastiCache(Redis), ALB, EFS(선택)
- 배포 방식: Docker image → ECR → ECS Fargate 또는 EC2에서 docker-compose / systemd
- CI/CD: GitHub Actions 빌드→테스트→이미지 Push→배포 트리거
- 운영: CloudWatch 로그/metrics, IAM roles, Security Groups

### 옵션 B — Render / Heroku / Fly (PaaS, 간단)
- 장점: 단순한 설정, 배포 빠름
- 단점: 고급 네트워크 제어 제한, 비용/성능 고려
- 구성: Docker deploy or native service, managed Postgres/Redis addons

### 배포 체크리스트 (공통)
- [ ]  환경변수 관리(Secrets) — GitHub Secrets / Parameter Store / Vault
- [ ]  Health check endpoint (`/healthz`) 구현
- [ ]  DB 백업/복구 정책 수립
- [ ]  롤백 전략 (이미지 태깅)
- [ ]  모니터링/알림 연결
- [ ]  SSL 적용 및 HTTP → HTTPS 리디렉션
- [ ]  Autoscaling 룰 정의(예: CPU > 70% 또는 RPS 기준)

### 샘플 GH Actions (빌드 → push → deploy via SSH)
```yaml
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t ghcr.io/${{ github.repository }}:${{ github.sha }} .
      - name: Push image
        run: docker push ...
      - name: Deploy via SSH
        uses: appleboy/ssh-action@v0.1.5
        with:
          host: ${{ secrets.HOST }}
          script: |
            docker pull ...
            docker-compose down && docker-compose up -d
```

<!-- -->
# 보충자료1: 운영,보안 세부항목

## 운영·보안 세부 항목 (cross-cutting)
- 인증: JWT(만료/리프레시 토큰), refresh 토큰 보관(secure cookie) 권장
- 권한: RBAC 또는 ABAC(관리자/유저 권한 분리)
- 데이터 보안: 암호화(전송 TLS, 저장 필요 시 at-rest 암호화)
- 시크릿 관리: Vault / AWS Parameter Store / GitHub Secrets
- 취약점 검사: Snyk/Dependabot, 정적 분석(ESLint, bandit)
- 백업: DB 스냅샷 주기, 복원 절차 문서화

## 샘플 프로젝트 구조
```
/project
  /app
    main.py
    /api
    /services
    /models
    /schemas
    /tests
  /migrations
  Dockerfile
  docker-compose.yml
  openapi.yaml
  requirements.md
  README.md
```
## 우선순위(빠른 MVP 제작 권장 순서)
1. 요구사항 문서(1)
2. DB 설계 + 마이그레이션(2)
3. 기본 API + 인증(3,4)
4. Docker화 + 로컬 docker-compose(9)
5. 테스트(8)
6. Redis 캐시/Rate limit(5)
7. 에러 미들웨어 + 로깅(7,6)
8. 모니터링 + 배포(6,10)

## Acceptance Criteria (MVP)
- 인증된 사용자 흐름(회원가입→로그인→JWT로 보호된 엔드포인트 접근) 동작
- CRUD 주요 리소스 동작(POST/GET/PATCH/DELETE)
- Docker로 로컬에서 서비스 기동 가능
- Unit/Integration test가 CI에서 통과
- Redis 캐시(간단한 GET 캐시) 적용
- 구조화 로그 + `/healthz` 존재
- 배포 파이프라인(이미지 빌드 및 배포 스크립트) 동작

## 빠른 시작 체크리스트 (즉시 실행 가능한 작업)
1. Repo 생성 + `requirements.md`
2. `openapi.yaml` 최소 엔드포인트 5개 작성
3. DB 마이그레이션 스크립트 1개 작성(사용자 테이블)
4. 기본 auth(회원가입/로그인) 구현
5. Dockerfile, docker-compose 기본본 추가
6. GitHub Actions에 CI 기본 파이프라인 추가(테스트 + lint + build)

<!-- -->
# 보충자료2: 샘플 엔드포인트 API설계
### 1. “샘플 엔드포인트”는 뭐냐?
샘플 엔드포인트는 **API를 어떻게 설계하면 좋을지 보여주는 “예시”**일 뿐이다.
하드 규약이 정해진 것은 아니다.
다만 업계에서 **사람들이 일반적으로 쓰는 관례(best practice)**는 존재한다.

예:
```
GET /users/123
POST /posts
GET /posts?page=1
DELETE /posts/33
POST /posts/33/comments
```
이런 패턴을 다 REST 관례(리소스 중심, 명사 중심)라고 부른다.

즉:
- 정해진 절대 규약은 없음
- 그러나 **“관례”는 있음**
- 실무자·면접관은 “관례를 올바르게 알고 있나”를 본다

### 2. API 설계는 어떻게 하는가?
API 설계는 사실 아래 3개를 만들면 끝이다.
1. **엔드포인트 목록(URI)**
2. **요청(Request) JSON 정의**
3. **응답(Response) JSON 정의**
예:
```
POST /auth/signup
{
  "email": "a@a.com",
  "password": "1234"
}

→
201 Created
{
  "user_id": "abcdefg",
  "email": "a@a.com"
}
```
이걸 **OpenAPI(Swagger) 문서**로 표현하면
면접에서 “API 설계 잘하네”라고 평가받는다.

### 3. Node.js를 잘 몰라도 되나?
당장 깊게 배워 들어갈 필요 없다.
**선택 흐름**
- 백엔드 언어 하나를 골라서 → API만 만들 수 있으면 됨
- 그 위의 Docker/Nginx/Redis는 언어와 무관한 “운영 기술”임

Node.js를 쓸지 Python(FastAPI)·Java(Spring)·Go를 쓸지는 **네가 과제용으로 편한 것**을 선택하면 된다.

실제 난이도 비교:
- **가장 배우기 쉬움**: Python(FastAPI)
- **가장 포트폴리오용으로 인정 많이 받음**: Node.js or Java(Spring)
- **가장 성능/구조 깔끔**: Go

만약 **node.js 문법 때문에 시작이 안 된다 → FastAPI**가 훨씬 빠르고 단순하다.
잘 모르는 Node.js 때문에 막히면 **언어 전체를 배우는 건 비효율**이다.
포트폴리오 핵심은 *언어*가 아니라 *API 설계와 시스템 구성 능력*이기 때문이다.