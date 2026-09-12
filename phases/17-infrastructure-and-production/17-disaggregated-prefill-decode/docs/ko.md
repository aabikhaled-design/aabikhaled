# 분리형 프리필/디코드(Disaggregated Prefill/Decode): NVIDIA Dynamo와 llm-d

> 프리필(prefill)은 연산 바운드(compute-bound)이고, 디코드(decode)는 메모리 바운드(memory-bound)다. 둘 다 같은 GPU에서 돌리면 한 자원을 낭비한다. 분리(disaggregation)는 이들을 별도 풀로 쪼개고 그 사이에서 KV 캐시를 NIXL(RDMA/InfiniBand 또는 TCP 폴백)로 전송한다. NVIDIA Dynamo(GTC 2025 발표, 1.0 GA)는 vLLM/SGLang/TRT-LLM 위에 위치한다. 그 Planner Profiler + SLA Planner가 SLO를 충족하도록 prefill:decode 비율을 자동으로 율 매칭(rate-match)한다. NVIDIA는 이 정도 범위의 처리량 이득을 공개한다. developer.nvidia.com(2025-06)은 중간 지연 영역(medium-latency regime)에서 GB200 NVL72 + Dynamo의 DeepSeek-R1 MoE에 대해 ~6배 개선을 보여주고, Dynamo 제품 페이지(developer.nvidia.com, 날짜 미상)는 GB300 NVL72 + Dynamo에서 Hopper 대비 최대 50배 MoE 처리량을 광고한다. "30배" 수치는 풀스택 Blackwell + Dynamo + DeepSeek-R1 보고들 전반의 커뮤니티 집계다. 정확히 30배라고 명시한 단일 1차 출처를 찾지 못했으므로, 방향성 주장으로 취급하라. llm-d(Red Hat + AWS)는 쿠버네티스 네이티브다: prefill / decode / router를 역할별 HPA를 가진 독립 Service로 둔다. llm-d 0.5는 계층적 KV 오프로딩, 캐시 인식 LoRA 라우팅, UCCL 네트워킹, 스케일 투 제로(scale-to-zero)를 추가한다. 경제성: 여러 고객 공개의 내부 합산은 일정 SLA에서 코로케이션(colocated) 서빙에서 Dynamo 분리형으로 전환할 때 $2M급 추론 지출에서 30–40% 절감(즉 연간 $600-800K)을 시사한다. 구체적인 $2M→$600-800K 수치는 내부 합성치이지 단일 발표 사례 연구가 아니다. 참조 인용이 아니라 자릿수 앵커로 사용하라. 짧은 프롬프트(<512 토큰, 짧은 출력)는 전송 비용을 정당화하지 못한다.

**Type:** Learn
**Languages:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 프리필과 디코드가 왜 다른 최적 GPU 할당을 갖는지 설명하고 코로케이션 하의 낭비를 정량화하기.
- 분리형 아키텍처를 다이어그램으로 그리기: 프리필 풀, 디코드 풀, NIXL을 통한 KV 전송, 라우터.
- 분리가 이득이 되지 않는 조건(짧은 프롬프트, 짧은 출력)을 짚기.
- NVIDIA Dynamo(스택 위)와 llm-d(쿠버네티스 네이티브)를 구별하고 각각을 운영 맥락에 매칭하기.

## 문제 (The Problem)

Llama 3.3 70B를 8개의 H100에서 돌린다고 하자. 혼합 워크로드(긴 프롬프트 + 짧은 출력)에서는 연산 대부분이 프리필에 쓰이므로 디코드 동안 GPU가 유휴 상태가 된다. 다른 워크로드(짧은 프롬프트 + 긴 출력)에서는 반대가 일어난다. 프리필과 디코드를 한데 묶어 코로케이션하면 둘 다 과다 프로비저닝하게 된다.

예산 영향: GPU 시간의 20-40%가 잘못된 자원에 낭비된다. 메모리 바운드 디코드를 돌리려고 H100 연산을 사거나, 연산 바운드 프리필을 돌리려고 H100 HBM 대역폭을 사고 있다. 둘 다 비싼 낭비다.

분리는 프리필과 디코드를 각각의 병목에 맞게 크기 조정된 별도 풀로 쪼갠다. KV 캐시는 프리필 풀에서 디코드 풀로 고대역폭 인터커넥트를 통해 전송된다.

## 개념 (The Concept)

### 왜 병목이 다른가

**프리필(Prefill)**(전체 입력 프롬프트에 대해 트랜스포머를 한 번의 순방향으로 실행한다. 행렬 곱셈이 지배한다. 연산 바운드. H100 FP8은 유용 처리량 ~2000 TFLOPS를 준다. 배치 효율이 좋다) 한 번의 순방향이 많은 토큰을 처리한다.

**디코드(Decode)**(한 번에 한 토큰을 생성하며, 매 반복마다 전체 가중치를 읽는다. 메모리 대역폭 바운드. HBM3는 ~3 TB/s를 준다. 배치 효율은 높은 동시성에서만 좋다) 가중치 읽기가 배치 전반에 분할 상환된다.

이들을 코로케이션하면: 둘 다에 최적화된 GPU를 산다. H100은 둘 다 잘하지만 어느 쪽이든 같은 비용이 든다. 규모가 커지면 프리필 풀은 H100 / 연산 중심에, 디코드 풀은 H200 / 메모리 중심에, 또는 공격적 양자화와 함께 두는 편이 낫다.

### 아키텍처

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL은 NVIDIA의 노드 간 전송이다. 가능할 때 RDMA/InfiniBand를 쓰고, 아니면 TCP 폴백을 쓴다. 전송 지연 시간은 실재한다. 보통 70B FP8에서 4K 토큰 프롬프트의 KV 캐시에 대해 20-80ms. 짧은 프롬프트가 분리를 정당화하지 못하는 이유가 여기에 있다. 전송 세금이 절감을 초과하기 때문이다.

### Dynamo 대 llm-d

**NVIDIA Dynamo** (GTC 2025 발표, 1.0 GA):
- vLLM, SGLang, TRT-LLM 위에 오케스트레이터로 위치한다.
- Planner Profiler가 워크로드를 측정하고, SLA Planner가 prefill:decode 비율을 자동 설정한다.
- Rust 코어, Python 확장성.
- 처리량 이득: NVIDIA는 중간 지연 영역에서 GB200 NVL72 + Dynamo의 DeepSeek-R1 MoE에 대해 6배를 보고한다(developer.nvidia.com, 2025-06). 풀 Blackwell + Dynamo + DeepSeek-R1 스택의 "최대 30배" 커뮤니티 보고는 단일 1차 출처가 없으며 방향성으로 취급해야 한다.
- GB300 NVL72 + Dynamo: Dynamo 제품 페이지에 따르면 Hopper 대비 최대 50배 MoE 처리량(developer.nvidia.com, 날짜 미상).

**llm-d** (Red Hat + AWS, 쿠버네티스 네이티브):
- prefill / decode / router를 독립 쿠버네티스 Service로 둔다.
- 큐 깊이(prefill) / KV 사용률(decode) 신호로 역할별 HPA.
- `topologyConstraint packDomain: rack`이 고대역폭 KV 전송을 위해 prefill+decode 클리크(clique)를 같은 랙에 묶는다.
- llm-d 0.5(2026): 계층적 KV 오프로딩, 캐시 인식 LoRA 라우팅, UCCL 네트워킹, 스케일 투 제로.

관리형 스택 위 오케스트레이터를 원하면 Dynamo를 쓰라. 쿠버네티스 네이티브 프리미티브를 원하고 CNCF 생태계에 전념한다면 llm-d를 쓰라.

### 경제성

내부 합성치(단일 발표 사례 연구가 아님: 자릿수 앵커):

- 코로케이션 서빙에 연간 $2M 추론 지출.
- Dynamo 분리형으로 전환.
- 같은 요청 볼륨, 같은 P99 지연 시간 SLA.
- 보고된 절감: 연간 $600K–$800K(30–40% 감소).
- 신규 하드웨어 없음.

우리는 이 수치를 단일 인용 가능한 사례 연구가 아니라 여러 고객 공개로부터 종합한다. 가장 가까운 발표 데이터 포인트는 Dynamo KV 라우팅으로 Baseten의 2배 빠른 TTFT / 61% 높은 처리량(baseten.co, 2025-10)과, 40–60% KV 적중률에서 VAST + CoreWeave의 토큰/$ 60–130% 증가 예측(vastdata.com, 2025-12)이다. 절감은 각 풀의 적정 크기 조정에서 온다. 프리필 중심 워크로드(8K+ 프리픽스가 있는 RAG)가 균형 잡힌 것보다 더 많은 이득을 본다.

### 분리하면 안 되는 경우

- 프롬프트 < 512 토큰이고 출력 < 200 토큰: 전송 세금이 이득을 지배한다.
- 작은 클러스터(< 4 GPU): 풀 다양성이 충분하지 않다.
- 팀이 역할별 스케일링으로 두 GPU 풀을 운영할 수 없음: Dynamo가 돕지만 간단하지는 않다.
- RDMA 패브릭 없음: TCP 전송 세금이 더 무겁다.

### 라우터는 Phase 17 · 11과 통합된다

분리형 라우터는 KV 캐시 인식(Phase 17 · 11)이다. 요청이 자신의 프리픽스를 보유한 디코드 풀에 도착한다. 일치가 없으면 프리필 → 디코드로 흐른다. 적중률과 분리는 복합적이다. 캐시 인식 라우터가 새 프리필이 애초에 필요한지를 결정한다.

### Blackwell의 MoE가 진짜 숫자가 있는 곳이다

GB300 NVL72 + Dynamo는 Hopper 베이스라인 대비 50배 MoE 처리량을 보여준다. MoE 전문가 라우팅(expert routing)은 프리필에서 연산 중심이지만 디코드에서 메모리 중심(전문가 캐시)이므로, 분리는 이중 이득이다. 2026년 프런티어 모델 서빙은 MoE 지배적이다(DeepSeek-V3, 미래의 GPT-5 변형).

### 기억해야 할 숫자들

벤치마크 숫자는 바뀐다. NVIDIA와 추론 스택은 매 분기 갱신된 결과를 게시한다. 인용 전에 다시 확인하라.

- GB200 NVL72 + Dynamo의 DeepSeek-R1: 중간 지연 영역에서 베이스라인 대비 ~6배 처리량(developer.nvidia.com, 2025-06). 풀 Blackwell + Dynamo 스택의 "최대 30배" 커뮤니티 주장은 단일 1차 출처가 없는 방향성 집계다.
- GB300 NVL72 + Dynamo: Hopper 대비 최대 50배 MoE 처리량(developer.nvidia.com, 날짜 미상).
- 절감 앵커(내부 합성치, 단일 사례 연구 아님): 일정 SLA에서 연간 $2M 지출 중 연간 $600-800K 절감.
- 분리 임계값: 프롬프트 >512 토큰 + 출력 >200 토큰.
- NIXL을 통한 KV 전송: 70B FP8에서 4K 프롬프트 KV에 대해 20-80ms.

```figure
prefill-decode-split
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 코로케이션 대 분리형 서빙을 시뮬레이션한다. 처리량, 요청당 비용, 프롬프트 길이 교차점을 보고한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-disaggregation-decider.md`를 생성한다. 워크로드와 클러스터가 주어지면 분리할지를 결정한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 어느 프롬프트 길이에서 분리가 코로케이션을 이기는가?
2. P99 프리픽스 길이 8K, 출력 300인 RAG 서비스를 위한 프리필 풀과 디코드 풀을 설계하라.
3. Dynamo 대 llm-d: Python 런타임 선호가 없는 순수 쿠버네티스 샵을 위해 하나를 고르라.
4. KV 전송 비용을 계산하라: 70B FP8에서 4K 프리필 = ~500 MB KV. RDMA 100 GB/s에서 전송 = 5ms. TCP 10 GB/s에서 = 50ms. 어느 쪽이 SLA에 중요한가?
5. MoE 전문가 라우팅은 KV 접근 패턴을 바꾼다. 토큰마다 다른 전문가를 활성화하는 MoE에서 분리는 어떻게 동작하는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 분리형 서빙 (Disaggregated serving) | "프리필/디코드 분할" | 각 단계를 위한 별도 GPU 풀 |
| NIXL | "NVIDIA 전송" | Dynamo의 노드 간 KV 전송(RDMA/TCP) |
| NVIDIA Dynamo | "오케스트레이터" | vLLM/SGLang/TRT-LLM의 스택 위 코디네이터 |
| llm-d | "쿠버네티스 네이티브" | Red Hat + AWS의 K8s 분리형 스택 |
| Planner Profiler | "Dynamo 자동 설정" | 워크로드를 측정하고 풀 비율을 설정 |
| SLA Planner | "Dynamo 정책" | SLO 충족을 위해 prefill:decode를 자동 율 매칭 |
| `packDomain: rack` | "llm-d 토폴로지" | 빠른 KV를 위해 prefill+decode를 같은 랙에 묶음 |
| UCCL | "통합 컬렉티브" | 스케일 투 제로를 위한 llm-d 0.5 네트워킹 계층 |
| MoE 전문가 라우팅 (MoE expert routing) | "토큰당 전문가" | DeepSeek-V3 패턴; 분리가 도움 |

## 더 읽을거리 (Further Reading)

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
