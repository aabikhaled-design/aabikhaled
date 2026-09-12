# LLM 프로덕션을 위한 카오스 엔지니어링

> LLM을 위한 카오스 엔지니어링(chaos engineering)은 2026년에 그 자체로 하나의 분야다. 프로덕션(production)에서 실험을 돌리기 전 전제 조건: 정의된 SLI/SLO, 트레이스+메트릭+로그 관측성(observability), 자동 롤백(rollback), 런북(runbook), 온콜(on-call). 아키텍처는 네 개의 평면(plane)을 가진다: 제어(control)(실험 스케줄러), 대상(target)(서비스, 인프라, 데이터 스토어), 안전(safety)(가드(guard) + 중단(abort) + 트래픽 필터), 관측성(observability)(메트릭 + 트레이스 + 로그), 피드백(feedback)(SLO 조정으로). 가드레일(guardrail)은 필수다: 소진율(burn-rate) 경보는 일일 오류 예산(error-budget) 소진이 예상의 2배를 넘으면 실험을 일시정지한다. 억제 윈도(suppression window) + 트레이스 ID 상관(trace-ID correlation)이 경보 잡음을 중복 제거한다. 주기: 매주 소규모 카나리(canary) + SLO 검토; 매월 게임 데이(game day) + 포스트모템(postmortem); 분기마다 교차 팀 회복력(resilience) 감사 + 의존성 매핑. LLM 특화 실험: 메모리 과부하, 네트워크 실패, 프로바이더(provider) 정전(outage), 비정상 프롬프트, KV 캐시 축출(eviction) 폭풍. 도구: Harness Chaos Engineering(LLM 유도 추천, 블래스트 반경(blast-radius) 축소, MCP 도구 통합); LitmusChaos(CNCF); Chaos Mesh(CNCF Kubernetes 네이티브).

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 다섯 가지 카오스 엔지니어링 전제 조건(SLI/SLO, 관측성, 롤백, 런북, 온콜)을 명명하고, 어느 하나라도 건너뛰면 왜 그 관행이 무너지는지 설명하기.
- 네 평면(제어, 대상, 안전, 관측성)과 SLO로의 피드백 루프를 다이어그램으로 그리기.
- 다섯 가지 LLM 특화 실험(메모리 과부하, 네트워크 실패, 프로바이더 정전, 비정상 프롬프트, KV 축출 폭풍)을 열거하기.
- 스택이 주어졌을 때 도구를 고르기: Harness, LitmusChaos, Chaos Mesh.

## 문제 (The Problem)

전통적 스택의 카오스 테스트는 정립되어 있다. LLM 스택은 새로운 실패 양상(failure mode)을 더한다. 독성 문자(poison character)가 든 4K 토큰 프롬프트가 토크나이저(tokenizer)를 12초 동안 멈춰 세운다. 업스트림 프로바이더가 429를 내뱉는다. 게이트웨이(gateway)가 재시도한다. 재시도로 증폭된 동시성에 서비스가 OOM에 빠진다. 버스트 부하 하의 KV 캐시 축출 폭풍이 재-프리필(re-prefill) 연쇄를 일으켜 연산을 포화(saturate)시킨다.

이 중 어느 것도 단위 테스트에 나타나지 않는다. 카오스 엔지니어링은 사용자가 발견하기 전에 이것들을 발견하는 방법이다.

## 개념 (The Concept)

### 전제 조건

다음 없이는 프로덕션에서 카오스를 돌리지 마라.

1. **SLI/SLO**: 정의된 서비스 수준 지표(indicator)와 목표(objective).
2. **관측성**: 트레이스, 메트릭, 로그를 대시보드에 연결.
3. **자동 롤백**: Phase 17 · 20 정책 플래그(policy-flag) 롤백.
4. **런북**: 구조화됨, Phase 17 · 23.
5. **온콜**: 대응할 누군가.

어느 하나라도 빠지면 카오스가 실제 인시던트가 된다.

### 네 평면 + 피드백

**제어 평면(Control plane)**: 실험 스케줄러(Litmus 워크플로, Chaos Mesh 스케줄, Harness UI).

**대상 평면(Target plane)**: 서비스, 파드(pod), 노드, 로드 밸런서, 데이터 스토어.

**안전 평면(Safety plane)**: 킬 스위치(kill switch), 억제 윈도, 블래스트 반경 제한, 오류 예산 게이트.

**관측성 평면(Observability plane)**: 정상 메트릭 + 카오스 유발 실패와 자연 실패를 구별하는 트레이스 ID 상관.

**피드백 루프(Feedback loop)**: 발견 사항이 SLO 조정, 런북 갱신, 코드 수정으로 되먹임된다.

### 가드레일은 필수다

- **소진율 경보**: 일일 오류 예산 소진이 예상의 2배를 넘으면 실험을 일시정지.
- **억제 윈도**: 실험 중 블래스트 반경 안의 비-실험 경보를 침묵시킨다.
- **트레이스 ID 상관**: 모든 실험 유발 오류에 태그가 붙어 있어 온콜이 중복을 걸러낸다.

### 다섯 가지 LLM 특화 실험

1. **메모리 과부하**: 높은 동시성으로 긴 컨텍스트 요청을 보내 KV 캐시 선점(preemption) 폭풍을 강제한다. 관찰: 서비스가 우아하게 부하를 떨어내는가(shed), 아니면 충돌하는가?

2. **네트워크 실패**: 추론 게이트웨이와 프로바이더 사이 연결을 끊는다. 관찰: 폴백(fallback)이 SLA 안에 작동하는가? (Phase 17 · 19)

3. **프로바이더 정전 시뮬레이션**: OpenAI에서 100% 429. 관찰: 라우팅이 Anthropic으로 페일오버(failover)하는가? (Phase 17 · 16, 19)

4. **비정상 프롬프트**: 토크나이저를 멈추게 하는 페이로드(예: 깊게 중첩된 유니코드, 거대한 UTF-8 코드포인트)를 주입한다. 관찰: 단일 요청이 워커(worker)를 잠그는가?

5. **KV 축출 폭풍**: vLLM 블록 예산을 포화시켜 축출을 강제한다. 관찰: LMCache가 복구하는가, 아니면 서비스가 저하되는가?

### 주기 (Cadence)

- **매주**: 스테이징(staging)에서 소규모 카나리 실험, 어쩌면 프로덕션 5%.
- **매월**: 특정 시나리오에 대한 예정된 게임 데이; 교차 팀 참석; 포스트모템.
- **분기마다**. 교차 팀 회복력 감사; 의존성 맵 갱신.

### 도구 (Tooling)

- **Harness Chaos Engineering**: 상용; AI 유도 실험 추천; 블래스트 반경 축소; MCP 도구 통합.
- **LitmusChaos**: CNCF 졸업(graduated); Kubernetes 워크플로 기반.
- **Chaos Mesh**: CNCF 샌드박스; Kubernetes 네이티브 CRD 스타일.
- **Gremlin**: 상용; 폭넓은 지원.
- **AWS FIS** / **Azure Chaos Studio**: 매니지드 클라우드 제공.

### 작게 시작하기

첫 실험: 정상 트래픽 하에서 디코드(decode) 복제본 하나를 파드 킬(pod-kill). 재라우팅과 복구를 관찰한다. 이것이 동작하고 안전해 보이면, 네트워크 카오스로 진급한다.

첫 LLM 특화 실험: 프로바이더 429 하나를 5분간 주입한다. 폴백을 관찰한다. 대부분의 팀은 자신의 폴백이 완전히 테스트되지 않았음을 발견한다.

### 기억해 둘 숫자들

- 네 평면: 제어, 대상, 안전, 관측성.
- 소진율 일시정지: 예상 일일 예산 소진의 2배.
- 주기: 매주 카나리, 매월 게임 데이, 분기마다 감사.
- 다섯 LLM 실험: 메모리, 네트워크, 프로바이더, 비정상 프롬프트, KV 폭풍.

```figure
i4-chaos-guard
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 안전 평면 게이트를 둔 세 가지 카오스 실험을 시뮬레이션한다. 어느 실험이 소진율 중단을 발동시킬지 보고한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-chaos-plan.md`를 만든다. 스택과 성숙도가 주어지면 첫 세 실험과 도구를 고른다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 어느 실험이 소진율 게이트를 발동시키며 왜 그런가?
2. vLLM 기반 RAG 서비스를 위한 첫 다섯 카오스 실험을 설계하라. 성공 기준을 포함하라.
3. 소진율 경보가 실험을 일시정지했다. 근본 원인이 카오스인지 자연인지 어떻게 판별하는가?
4. 카오스가 프로덕션에서 돌아야 하는지, 아니면 스테이징에서만 돌아야 하는지 논증하라. 프로덕션이 옳은 답일 때는 언제인가?
5. 범용 네트워크 카오스가 재현할 수 없는 세 가지 LLM 특화 실패 양상을 명명하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| SLI / SLO | "서비스 목표" | 지표 + 목표; 필수 전제 조건 |
| 블래스트 반경 (Blast radius) | "범위" | 실험에 영향받는 서비스 / 사용자 집합 |
| 소진율 경보 (Burn-rate alert) | "예산 게이트" | 오류 예산 소진율 > 예상의 2배일 때 발동 |
| 게임 데이 (Game day) | "월간 훈련" | 예정된 교차 팀 카오스 연습 |
| LitmusChaos | "CNCF 워크플로" | 졸업한 CNCF Kubernetes 카오스 도구 |
| Chaos Mesh | "CNCF CRD" | CNCF 샌드박스 Kubernetes 네이티브 카오스 |
| Harness CE | "상용 AI 지원" | AI 추천을 가진 Harness 카오스 |
| 비정상 프롬프트 (Malformed prompt) | "토크나이저 폭탄" | 토큰화를 멈추게 하는 입력 |
| KV 축출 폭풍 (KV eviction storm) | "선점 연쇄" | 재-프리필을 유발하는 대량 축출 |

## 더 읽을거리 (Further Reading)

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
