# AI를 위한 SRE: 멀티 에이전트 인시던트 대응, 런북, 예측적 탐지

> AI SRE는 RAG로 인프라 데이터(로그, 런북(runbook), 서비스 토폴로지(topology))에 근거(ground)한 LLM을 써서 조사, 문서화, 조율 단계를 자동화한다. 2026년 아키텍처 패턴은 멀티 에이전트 오케스트레이션(multi-agent orchestration)이다. 슈퍼바이저(supervisor)가 조율하는 특화 에이전트(agent)(로그, 메트릭, 런북)들. AI는 가설과 쿼리를 제안하고, 판단이 필요한 결정은 사람이 승인한다. Datadog Bits AI와 Azure SRE Agent가 이것을 매니지드 제품으로 출시한다. 런북은 진화하고 있다: NeuBird Hawkeye는 적대적 평가(adversarial evaluation)를 사용한다(두 모델이 같은 인시던트를 분석한다. 일치 = 확신, 불일치 = 불확실성). 운영 메모리(operational memory)는 팀 변동을 넘어 지속된다. 자동 교정(auto-remediation)은 신중함을 유지한다: AI가 제안하고, 사람이 승인한다. 완전 자율 행동은 좁다(파드(pod) 재시작, 특정 배포(deploy) 롤백(rollback)): 빡빡한 가드레일(guardrail)과 함께. "설정하고 잊으세요(set it and forget it)"를 파는 사람은 누구든 과장하고 있다. 떠오르는 프런티어: 인시던트 전 예측. MIT 연구는 과거 로그 + GPU 온도 + API 오류 패턴으로 학습한 LLM이 정전(outage)의 89%를 10~15분 일찍 예측했다고 보고한다. 전망: 2026년 말까지 엔터프라이즈 LLM의 95%가 자동 페일오버(failover)를 갖춘다.

**Type:** Learn
**Languages:** Python (stdlib, toy multi-agent incident triage simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 멀티 에이전트 AI SRE 아키텍처를 다이어그램으로 그리기: 슈퍼바이저 + 특화 에이전트(로그, 메트릭, 런북) + 사람 승인 게이트(gate).
- 자동 교정이 (서비스 재설계 같은) 넓은 것이 아니라 (파드 재시작, 배포 되돌리기 같은) 좁은 이유를 설명하기.
- 적대적 평가 패턴(NeuBird Hawkeye)을 명명하기: 두 모델 일치 = 확신; 불일치 = 에스컬레이션.
- MIT의 89% 조기 탐지 결과와 운영 제약을 인용하기: 작동(actuation) 없는 예측은 그저 대시보드일 뿐이다.

## 문제 (The Problem)

온콜(on-call) 엔지니어가 새벽 3시에 호출(page)을 받는다. "체크아웃에서 높은 오류율." 그는 Datadog, Loki, 런북 셋, 배포 로그를 확인한다. 30분 후 근본 원인이 KV 캐시 급증으로 인한 vLLM OOM임을 깨닫는다. 파드를 재시작하니 오류가 사라진다.

2026년에는 그 조사의 첫 20분이 자동화 가능하다. 로그를 서비스별로 그룹화하고, 최근 배포와 상관시키고, 런북에 대조하는 것: 모두 RAG + 도구 사용(tool-use)이다. 감독되는(supervised) 에이전트가 1차 트리아지(triage)를 수행해, 사람이 Datadog을 열기 전에 가설을 제시한다.

완전 자율 교정은 다른 문제다. 파드 재시작: 안전. GPU 풀 확장: 정책이 허용하면 안전. 서비스 재설계: 절대 안 됨. 규율(discipline)이란 그 좁은 선을 긋는 것이다.

## 개념 (The Concept)

### 멀티 에이전트 아키텍처

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

슈퍼바이저는 인시던트를 하위 쿼리로 나눈다. 특화 에이전트들은 도구 접근(로그 검색, PromQL, 문서 검색) 권한을 갖는다. 슈퍼바이저는 이를 종합해 가설 + 근거를 사람에게 제시한다. 사람은 승인하거나 방향을 바꾼다.

### 자동 교정 범위

**안전(좁음)**: 파드 재시작, 특정 배포 되돌리기, 사전 승인된 경계 내에서 풀 확장, 사전 승인된 기능 플래그(feature flag) 활성화.

**안전하지 않음(넓음)**: 서비스 토폴로지 변경, 리소스 제한 수정, 새 코드 배포, IAM 변경, 데이터베이스 변경.

"설정하고 잊으세요"를 파는 사람은 누구든 과장하고 있다. 안전한 집합은 AI SRE가 성숙하면서 커지지만, 경계는 실재한다.

### 적대적 평가 (NeuBird Hawkeye)

두 모델이 같은 인시던트를 독립적으로 분석한다. 근본 원인에 일치하면 확신이 높다. 불일치하면 두 가설을 모두 드러내 사람에게 에스컬레이션한다. 단순한 패턴이지만, 환각된(hallucinated) 근본 원인에 대한 효과적인 필터다.

### 운영 메모리 (Operational memory)

팀 이직은 전통적 SRE의 조용한 살인자다. 부족 지식(tribal knowledge)이 떠난다. AI SRE는 런북 + 포스트모템(post-mortem)을 벡터 DB에 저장한다. 에이전트는 모든 새 인시던트마다 검색한다. 새 엔지니어가 합류해도 AI는 전체 이력을 가지고 있다.

### 인시던트 전 예측

MIT 2025 연구: 과거 로그, GPU 온도, API 오류 패턴으로 학습한 LLM이 테스트 셋에서 정전의 89%를 발생 10~15분 전에 예측했다.

현실 점검: 작동 없는 예측은 대시보드다. 운영상의 질문은 "예측할 때, 우리는 무엇을 하는가?"이다. 선제적 드레인(pre-emptive drain)? 호출? 오토스케일? 답은 정책에 따라 다르다.

### 2026년의 제품들

- **Datadog Bits AI**: Datadog 안의 매니지드 SRE 코파일럿(copilot).
- **Azure SRE Agent**: Azure 네이티브.
- **NeuBird Hawkeye**: 적대적 평가 + 운영 메모리.
- **PagerDuty AIOps**: 트리아지 + 중복 제거(deduplication).
- **Incident.io Autopilot**: 인시던트 커맨더(incident commander) + 조율.

### 코드로서의 런북 (Runbooks as code)

런북은 Confluence 페이지에서 구조화된 섹션(증상, 가설, 검증, 조치)을 가진 버전 관리되는 마크다운으로 진화한다. 구조화된 런북은 더 나은 RAG 검색을 가능하게 한다. 어떤 AI-SRE 롤아웃이든 비구조화된 런북을 구조화하는 것에서 시작하라.

### 기억해 둘 숫자들

- MIT 조기 탐지: 정전의 89%, 10~15분 선행 시간.
- 멀티 에이전트 트리아지: 슈퍼바이저 + (로그, 메트릭, 런북) + 사람.
- 안전한 자동 교정 집합: 파드 재시작, 배포 되돌리기, 경계 내 확장.
- 적대적 평가: 두 모델 독립; 일치 = 확신.

```figure
i4-incident-agents
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 멀티 에이전트 트리아지를 시뮬레이션한다: 로그 에이전트가 오류를 찾고, 메트릭 에이전트가 CPU 급증을 찾고, 런북 에이전트가 알려진 이슈에 대조한다. 슈퍼바이저가 가설들의 순위를 매긴다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-ai-sre-plan.md`를 만든다. 현재 온콜, 인시던트 양, 팀 성숙도가 주어지면 AI SRE 롤아웃을 설계한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 로그 에이전트와 메트릭 에이전트가 불일치하면 어떻게 되는가? 슈퍼바이저는 어떻게 해결하는가?
2. 자기 서비스에 맞는 세 가지 "안전한" 자동 교정 조치를 정의하라. 각각을 정당화하라.
3. 구조화된 런북 템플릿을 작성하라: 섹션, 필수 필드, 검증 명령어.
4. 예측적 탐지가 12분 선행으로 발동한다. 정책을 어떻게 세울 것인가: 호출, 선제적 드레인, 아니면 둘 다?
5. 3인 팀이 2026년에 AI SRE를 도입해야 하는지 기다려야 하는지 논증하라. 성숙도, 양, 위험을 고려하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| AI SRE | "온콜용 에이전트" | LLM 기반 인시던트 조사 + 조율 |
| 슈퍼바이저 에이전트 (Supervisor agent) | "오케스트레이터" | 인시던트를 하위 쿼리로 나누는 최상위 에이전트 |
| 특화 에이전트 (Specialized agent) | "도메인 에이전트" | 도구 접근(로그, 메트릭, 런북)을 가진 하위 에이전트 |
| 자동 교정 (Auto-remediation) | "AI가 고친다" | 좁은 사전 승인 조치; 넓은 재설계가 아님 |
| 운영 메모리 (Operational memory) | "벡터 런북" | RAG를 위한 벡터 DB 내 포스트모템 + 런북 |
| 적대적 평가 (Adversarial eval) | "두 모델 점검" | 독립적 분석; 일치 = 확신 |
| NeuBird Hawkeye | "그 적대적 평가 제품" | 적대적 평가 + 메모리 패턴을 가진 제품 |
| Bits AI | "Datadog의 SRE 에이전트" | Datadog 매니지드 AI SRE |
| 인시던트 전 예측 (Pre-incident prediction) | "조기 탐지" | 정전 예측에서 10~15분 선행 시간 |

## 더 읽을거리 (Further Reading)

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
