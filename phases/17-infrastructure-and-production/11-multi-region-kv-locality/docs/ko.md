# 멀티 리전 LLM 서빙과 KV 캐시 지역성(Multi-Region LLM Serving and KV Cache Locality)

> 라운드 로빈(round-robin) 부하 분산은 캐시된 LLM 추론(inference)에 적극적으로 해롭다. 자신의 프리픽스(prefix)를 보유한 노드에 도착하지 못한 요청은 전체 프리필(prefill) 비용을 치른다. 긴 프롬프트에서 P50 기준 대략 800ms인데, 캐시 적중 시에는 약 80ms다. 2026년의 프로덕션(production) 패턴은 KV 캐시 이벤트를 소비하고 프리픽스 해시(prefix-hash) 일치 기준으로 라우팅하는 캐시 인식 라우터(cache-aware router)다(Rust로 작성된 vLLM Router, llm-d router). 최근 연구(GORGO)는 리전 간(cross-region) 네트워크 지연 시간(latency)을 라우팅 목적 함수의 명시적 항으로 만든다. 상용 "리전 간 추론(cross-region inference)" 제품(Bedrock cross-region inference, GKE 멀티 클러스터 게이트웨이)은 추론을 불투명하게 취급한다. 이들은 가용성을 다루지 TTFT를 다루지 않는다. JPMorgan과 Mayo Clinic은 2024년 11월 us-east-1 페일오버(failover)를 약 22분에 수행했다. DR(재해 복구) 현실: LLM DR 실패의 32%는 팀이 가중치(weight)는 백업했지만 토크나이저(tokenizer) 파일이나 양자화(quantization) 설정을 빠뜨렸기 때문이다.

**Type:** Learn
**Languages:** Python (stdlib, toy prefix-cache-aware router simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 라운드 로빈 부하 분산이 캐시된 추론을 어떻게 망가뜨리는지 설명하고 TTFT 페널티를 정량화하기.
- 캐시 인식 라우터를 다이어그램으로 그리기: 입력(KV 캐시 이벤트), 알고리즘(프리픽스 해시 일치), 동점 처리(tie-breaker, GPU 사용률).
- LLM의 32% DR 실패 원인(누락된 토크나이저 파일 / 양자화 설정)을 짚고 3개 파일 DR 체크리스트를 진술하기.
- 상용 리전 간 제품(Bedrock CRI, GKE Multi-Cluster Gateway)과 KV 인식 라우팅을 구별하기.

## 문제 (The Problem)

서비스가 us-east-1, us-west-2, eu-west-1에서 돌아간다. 앞단에 라운드 로빈으로 ALB를 두었다. 프로덕션의 프리픽스 캐시 적중률이 8%로 떨어진다. TTFT P50이 세 배가 된다. vLLM 로그를 보면 모든 요청이 전체 프리필 비용을 치르고 있다.

라운드 로빈은 무상태(stateless) 서비스에 최적이다. LLM 추론은 설계상 상태가 있다(stateful). KV 캐시는 모델이 본 모든 것을 인코딩한다. 보지 않고 라우팅하면 잘못된 캐시로 라우팅하게 된다.

별개로, 팀에는 DR 계획이 있다. 모델 가중치를 리전 간 S3에 백업한다. 리전 장애가 닥친다. 페일오버를 시도한다. 레플리카(replica)가 시작을 거부한다. tokenizer.json, 양자화 설정, RoPE 스케일링 설정이 동기화해 두지 않은 별도 버킷에 있었음을 깜빡한 것이다.

멀티 리전 LLM 서빙은 캐시 문제이자 라우팅 문제이며 DR 위생 문제다. 부하 분산기 문제가 아니다.

## 개념 (The Concept)

### 캐시 인식 라우팅 (Cache-aware routing)

요청이 프롬프트와 함께 도착한다. 라우터는 프리픽스(가령 처음 512 토큰)를 해시한다. 그리고 각 레플리카에게 "이 프리픽스를 캐시에 갖고 있는가?"를 묻는다. 레플리카들은 블록을 할당하고 축출(evict)할 때 KV 캐시 이벤트를 pub/sub 채널에 게시한다. 라우터는 일치하는 레플리카를 고르고, 아무도 일치하지 않으면 GPU 사용률 기반 동점 처리로 넘어간다.

**vLLM Router** (Rust, 2026 production-stack): `kv.cache.block_added` 이벤트를 구독하고, 프리픽스 해시 → 레플리카 인덱스를 유지하며, O(1) 조회로 라우팅한다. 일치가 없으면 최소 큐 깊이(least-queue-depth)로 넘어간다.

**llm-d router**: 동일한 패턴, 쿠버네티스 네이티브(Kubernetes-native). ControlPlane API를 통해 이벤트를 게시한다.

**SGLang RadixAttention** (Phase 17 · 06)은 레플리카 내부(intra-replica) 등가물이다. 레플리카 간(cross-replica) 라우팅은 엄밀히 그 상류(upstream)에 있다.

### 숫자 (Numbers)

2K 토큰 프롬프트, Llama 3.3 70B FP8, H100에서의 TTFT P50:
- 캐시 적중(같은 레플리카, 프리픽스 상주): ~80ms.
- 캐시 미스(콜드 프리필): ~800ms.

10배 차이. 라우터가 레플리카 전반에서 프리픽스 캐시를 60-80% 적중하면, N개 레플리카 용량에서 단일 레플리카 성능에 근접한다. 10%를 적중하면 순진한(naive) 스케일링에 근접한다.

### 리전 간에는 새로운 제약이 있다. 네트워크 지연 시간

리전 간 RTT:
- us-east-1 ↔ us-west-2: ~65ms.
- us-east-1 ↔ eu-west-1: ~75ms.
- us-east-1 ↔ ap-southeast-1: ~220ms.

라우팅이 us-east-1의 요청을 ap-southeast-1의 핫 프리픽스로 보내면, 절약된 프리필(800 → 80ms)은 440ms 왕복(round-trip)에 압도당한다. GORGO(2026 연구)는 이를 명시적으로 만든다. 프리필만이 아니라 `prefill_time + network_latency`를 공동으로 최소화한다. 종종 답은, 프리필이 지배하는 거대한 수 MB 프리픽스를 제외하고는 라우팅을 리전 내로 유지하는 것이다.

### 상용 "리전 간 추론"은 여기서 도움이 되지 않는다

AWS Bedrock 리전 간 추론은 용량 압박 시 요청을 다른 리전으로 자동 라우팅한다. 이는 TTFT가 아니라 가용성을 최적화하며, 추론을 불투명하게 취급한다. GKE Multi-Cluster Gateway도 마찬가지다. 서비스 수준 페일오버이며, KV 캐시에 대한 인식이 없다.

이것들을 쓸 때조차 앱 계층의 캐시 인식 라우터가 여전히 필요하다. 이들은 "us-east-1이 불타고 있다" 경우를 다룬다. 캐시 인식 라우팅은 TTFT 경우를 다룬다.

### DR 위생: 32% 파일 누락 문제

널리 인용되는 2026년 통계: LLM DR 실패의 32%는 팀이 가중치는 백업했지만 다음을 빠뜨려서 발생한다:

- `tokenizer.json` 또는 `tokenizer.model`
- 양자화 설정(`quantize_config.json`, AWQ scales, GPTQ zero-points)
- 모델 고유 설정(RoPE 스케일링, 어텐션 마스크, 채팅 템플릿)
- 엔진 설정(`vllm_config.yaml`, 샘플링 기본값, LoRA 어댑터 매니페스트)

해결책은 3개 파일 최소 DR 매니페스트다:

1. HF 모델 리포지토리 하위의 모든 파일(가중치 + 설정 + 토크나이저).
2. 엔진 고유 서빙 설정.
3. 배포 매니페스트(K8s YAML, Dockerfile, 의존성 락).

추가로: 분기마다 DR 훈련을 실시한다. JPMorgan의 us-east-1 훈련이 2024년 11월에 복구 22분을 기록한 것은 오직 플레이북이 리허설되어 있었기 때문이다.

### 데이터 거주성은 직교한다 (Data residency is orthogonal)

EU 고객의 PHI는 EU를 떠날 수 없다. 캐시 인식 라우터가 파리에서 발신된 요청을 프리픽스 일치를 위해 us-east-1로 보내면, TTFT 이득과 무관하게 GDPR을 위반한 것이다. 캐시를 최적화하기 전에 라우터를 거주성 경계로 분할하라.

### 기억해야 할 숫자들

- 캐시 적중 대 미스 TTFT 차이: ~10배(2K 프롬프트에서 80ms 대 800ms).
- 미국-유럽 리전 간 RTT: ~75ms.
- DR 실패: 32%가 토크나이저/양자화 설정을 빠뜨린다.
- JPMorgan us-east-1 페일오버 2024년 11월: 22분(30분 SLA).

```figure
cache-aware-router
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 멀티 리전 워크로드에서 세 가지 라우팅 전략(라운드 로빈, 캐시 인식 리전 내, 캐시 인식 글로벌)을 시뮬레이션한다. 캐시 적중률, TTFT P50/P99, 리전 간 청구액을 보고한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-multi-region-router.md`를 생성한다. 리전, 거주성 제약, SLA가 주어지면 라우팅 계획을 설계한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. RTT 75ms가 주어졌을 때, 어느 프롬프트 길이에서 리전 간 라우팅이 로컬 전용 라우팅을 이기는가?
2. 캐시 적중률이 70%에서 12%로 떨어진다. 가능한 원인 세 가지와 각각을 확인해 줄 관측치(observable)를 진단하라.
3. vLLM에서 5개의 LoRA 어댑터로 서빙되는 70B AWQ 양자화 모델을 위한 DR 매니페스트를 설계하라. 모든 파일과 설정을 나열하라.
4. 엄격한 TTFT SLO를 가진 핀테크에게 Bedrock 리전 간 추론이 "충분한지" 논하라. 구체적인 동작을 인용하라.
5. 파리 발신 요청이 us-east-1의 프리픽스와 일치한다. 그 요청을 라우팅할 것인가? 정책을 작성하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 캐시 인식 라우팅 (Cache-aware routing) | "스마트 LB" | 프리픽스 해시 일치 기준으로 KV 캐시 보유 레플리카로 라우팅 |
| KV 캐시 이벤트 (KV-cache events) | "캐시 pub-sub" | 레플리카가 블록 추가/축출을 게시하고 라우터가 인덱싱 |
| 프리픽스 해시 (Prefix hash) | "캐시 키" | 라우터 조회에 쓰이는 처음 N 토큰의 해시 |
| GORGO | "리전 간 라우팅 연구" | arXiv 2602.11688; 네트워크 지연 시간을 명시적 항으로 |
| 리전 간 추론 (Cross-region inference) | "Bedrock CRI" | AWS 제품; 가용성 페일오버이며 TTFT 인식 아님 |
| DR 매니페스트 (DR manifest) | "백업 목록" | 복원에 필요한 모든 파일: 가중치만이 아니다 |
| 데이터 거주성 (Data residency) | "GDPR 경계" | 어느 리전이 사용자 데이터를 보는지에 대한 법적 제약 |
| RTT | "왕복 시간" | 네트워크 지연 시간; 미국-유럽 75ms, 미국-APAC 220ms |
| LLM 인식 LB (LLM-aware LB) | "캐시 적중 LB" | 제품 범주로서의 캐시 인식 라우터 |

## 더 읽을거리 (Further Reading)

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv: GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) 네트워크 지연 시간 항을 포함한 리전 간 KV 캐시 재사용.
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html): 가용성 페일오버 문서.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack): 캐시 인식 라우터 소스.
