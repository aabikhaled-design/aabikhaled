# LLM API 부하 테스트: k6와 Locust가 거짓말하는 이유

> 전통적인 부하 테스터(load tester)는 스트리밍 응답, 가변 출력 길이, 토큰(token) 수준 지표, GPU 포화(saturation)를 위해 설계되지 않았다. 함정 두 가지가 대부분의 팀을 무너뜨린다. GIL 함정: Locust의 토큰 수준 측정은 Python GIL 아래에서 토큰화(tokenization)를 실행하는데, 이는 높은 동시성에서 요청 생성과 경쟁한다. 그러면 토큰화 백로그(backlog)가 보고되는 토큰 간 지연 시간(inter-token latency)을 부풀린다. 병목은 서버가 아니라 클라이언트 쪽이다. 프롬프트 균일성 함정: 루프 안의 동일한 프롬프트는 토큰 분포의 한 점만 테스트한다. 실제 트래픽은 가변 길이와 다양한 접두사 일치(prefix match)를 가진다. LLMPerf는 `--mean-input-tokens` + `--stddev-input-tokens`로 이를 해결한다. 2026년 도구 매핑: LLM 특화(GenAI-Perf, LLMPerf, LLM-Locust, guidellm)는 토큰 수준 정확도용. **k6 v2026.1.0** + **k6 Operator 1.0 GA(2025년 9월)**: 스트리밍 인지(streaming-aware), TestRun/PrivateLoadZone CRD를 통한 Kubernetes 네이티브 분산, CI/CD 게이트(gate)에 최적. Vegeta는 Go 기반 고정 속도(constant-rate) 포화용. Locust 2.43.3은 스트리밍을 위해서는 LLM-Locust 확장과 함께만 쓴다. 부하 패턴: 정상 상태(steady-state), 램프(ramp), 스파이크(spike)(오토스케일링 테스트), 소크(soak)(메모리 누수).

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 범용 부하 테스터가 LLM API에 대해 거짓말하게 만드는 두 안티패턴(GIL 함정, 프롬프트 균일성 함정)을 설명하기.
- 주어진 목적에 맞는 도구를 고르기: LLMPerf(벤치마크 실행), k6 + 스트리밍 확장(CI 게이트), guidellm(대규모 합성), GenAI-Perf(NVIDIA 레퍼런스).
- 네 가지 부하 패턴(정상, 램프, 스파이크, 소크)을 설계하고 각각이 잡아내는 실패 양상을 명명하기.
- 고정 길이가 아니라 입력 토큰의 평균 + 표준편차를 사용해 현실적인 프롬프트 분포를 구축하기.

## 문제 (The Problem)

LLM 엔드포인트를 동시 사용자 500명으로 k6 테스트했다. 버텼다. 배포했다. 프로덕션에서 실제 사용자 200명에 서비스가 무너졌다. P99 TTFT가 폭발하고, GPU가 고정되었다.

두 가지 일이 일어났다. 첫째, k6는 동일한 프롬프트 500개를 보냈다. 요청 병합(request-coalescing)과 접두사 캐싱(prefix caching) 덕분에 실제로는 하나를 처리하면서도 500개의 동시 디코드(decode)를 처리하는 것처럼 보였다. 둘째, k6는 사용자 눈에 보이는 방식으로 스트리밍 응답의 토큰 간 지연 시간을 추적하지 않는다. 다양한 간격으로 도착하는 토큰 500개가 아니라 하나의 HTTP 연결을 볼 뿐이다.

LLM의 부하 테스트는 그 자체로 하나의 분야다.

## 개념 (The Concept)

### GIL 함정 (Locust)

Locust는 Python을 사용하며 GIL 아래에서 클라이언트 측 토큰화를 실행한다. 높은 동시성에서 토크나이저(tokenizer)는 요청 생성 뒤에 줄을 선다. 보고되는 토큰 간 지연 시간에는 클라이언트 측 토큰화 백로그가 섞여 들어간다. 서버가 느리다고 생각하지만 사실 느린 것은 테스트 하네스(harness)다.

해결: LLM-Locust 확장은 토큰화를 별도 프로세스로 옮긴다. 또는 컴파일 언어 하네스(k6, tokenizers.rs를 쓰는 LLMPerf)를 쓴다.

### 프롬프트 균일성 함정

알려진 모든 부하 테스터는 프롬프트 하나를 설정하게 한다. 1만 회 반복 루프 테스트에서는 정확히 같은 프롬프트가 매번 전송된다. 서버는 매번 같은 접두사를 본다. 접두사 캐시 적중(prefix cache hit)이 100%에 가까워지고, 처리량(throughput)이 훌륭해 보인다.

해결: 프롬프트 분포에서 샘플링한다. LLMPerf는 `--mean-input-tokens 500 --stddev-input-tokens 150`을 쓴다. 다양한 길이, 다양한 내용.

### 네 가지 부하 패턴

1. **정상 상태(Steady-state)**: 30~60분 동안 일정한 RPS. 잡아내는 것: 베이스라인(baseline) 성능 회귀(regression).
2. **램프(Ramp)**: 15분에 걸쳐 RPS를 0에서 목표까지 선형 증가. 잡아내는 것: 용량 분기점(capacity breakpoint), 워밍업 이상.
3. **스파이크(Spike)**: 갑작스러운 3~10배 RPS를 2분간, 그다음 복귀. 잡아내는 것: 오토스케일링 지연, 큐 포화(queue saturation), 콜드 스타트(cold-start) 영향.
4. **소크(Soak)**: 4~8시간 정상 상태. 잡아내는 것: 메모리 누수, 커넥션 풀(connection-pool) 드리프트, 관측성(observability) 오버플로.

### 2026 도구 매핑

**LLMPerf**(Anyscale): Python이지만 Rust 기반 토큰화. 평균/표준편차 프롬프트. 스트리밍 인지. 성능 실행의 최선 기본값.

**NVIDIA GenAI-Perf**: NVIDIA의 레퍼런스. Triton 클라이언트를 사용하며 지표 커버리지가 포괄적이다. 단, 그것의 ITL은 TTFT를 제외하고 LLMPerf의 것은 포함한다. 두 도구는 같은 서버에 대해 다른 TPOT를 낸다.

**LLM-Locust**(TrueFoundry): GIL 함정을 고치는 Locust 확장. 익숙한 Locust DSL + 스트리밍 지표.

**guidellm**: 대규모 합성(synthetic) 벤치마킹.

**k6 v2026.1.0** + **k6 Operator 1.0 GA(2025년 9월)**:
- k6 자체(Go, 컴파일, GIL 없음)가 스트리밍 인지 지표를 추가했다.
- k6 Operator는 Kubernetes 네이티브 분산 테스트를 위해 TestRun / PrivateLoadZone CRD를 사용한다.
- CI/CD 게이트와 SLA 테스트에 최적.

**Vegeta**: Go, k6보다 단순. 고정 속도 HTTP 포화. LLM 인지는 아니지만 게이트웨이(gateway) / 속도 제한(rate-limit) 테스트에 좋다.

**Locust 2.43.3 기본(stock)**: LLM에 대해 GIL 함정이 있다. LLM-Locust 확장과 함께만 쓴다.

### CI에서의 SLA 게이트

PR에서 k6를 다음과 같이 실행한다.

- 베이스라인 RPS에서 각각 30~50회 반복.
- 게이트: P50/P95 TTFT, 5xx < 5%, TPOT 임계값 이하.
- 위반 시 빌드를 깬다.

### 현실적인 프롬프트 분포

실제 트래픽 샘플(있다면)에서, 또는 공개된 분포(예: 채팅용 ShareGPT 프롬프트, 코드용 HumanEval)에서 구축한다. 평균 + 표준편차를 LLMPerf에 넣는다. 하나의 프롬프트로 루프 도는 것을 무슨 수를 써서라도 피하라.

### 기억해 둘 숫자들

- k6 Operator 1.0 GA: 2025년 9월.
- k6 v2026.1.0: 스트리밍 인지 지표.
- 전형적 LLMPerf 실행: 동시성 X에서 100~1000 요청.
- 전형적 CI 게이트: PR당 30~50회 반복.
- 네 가지 패턴: 정상, 램프, 스파이크, 소크.

```figure
load-pattern-waves
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 현실적인 프롬프트 분포로 부하 테스트를 시뮬레이션하고, 유효 TPOT를 측정하며, 균일 프롬프트 함정을 시연한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-load-test-plan.md`를 만든다. 워크로드(workload)와 SLA가 주어지면 도구를 고르고 네 가지 부하 패턴을 설계한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 균일 분포 vs 현실적 분포를 비교하라. 격차는 어디에 있는가?
2. CI 게이트를 위한 k6 스크립트를 작성하라: 동시 100에서 TTFT P95 < 800ms, 실행 시간 5분.
3. 소크 테스트가 시간당 50MB씩 메모리가 증가함을 보인다. 세 가지 원인과 그중에서 가려내기 위한 계측(instrumentation)을 명명하라.
4. 10 RPS에서 100 RPS로 스파이크 테스트. Karpenter + vLLM production-stack이 갖춰져 있다면(Phase 17 · 03 + 18) 예상 복구 시간은 얼마인가?
5. GenAI-Perf는 TPOT=6ms를, LLMPerf는 같은 서버에서 TPOT=11ms를 보고한다. 설명하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| LLMPerf | "그 LLM 하네스" | Anyscale 벤치마크 도구, 스트리밍 인지 |
| GenAI-Perf | "NVIDIA 도구" | NVIDIA 레퍼런스 하네스 |
| LLM-Locust | "LLM용 Locust" | GIL 함정을 고치는 Locust 확장 |
| guidellm | "합성 벤치마크" | 대규모 합성 도구 |
| k6 Operator | "K8s k6" | CRD 기반 분산 k6 |
| GIL 함정 (GIL trap) | "Python 클라이언트 오버헤드" | 토큰화 백로그가 보고 지연 시간을 부풀림 |
| 프롬프트 균일성 함정 (Prompt-uniformity trap) | "단일 프롬프트 거짓말" | 같은 프롬프트로 루프 돌면 캐시에 적중해 처리량이 부풀려짐 |
| 정상 상태 (Steady-state) | "일정한 부하" | N분 동안 평탄한 RPS |
| 램프 (Ramp) | "선형 증가" | 일정 시간에 걸쳐 0에서 목표까지 |
| 스파이크 (Spike) | "버스트 테스트" | 갑작스러운 배수 후 복귀 |
| 소크 (Soak) | "긴 테스트" | 누수 탐지를 위한 수 시간 |

## 더 읽을거리 (Further Reading)

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
