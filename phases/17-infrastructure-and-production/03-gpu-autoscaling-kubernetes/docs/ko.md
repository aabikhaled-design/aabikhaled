# Kubernetes에서의 GPU 오토스케일링(GPU Autoscaling): Karpenter, KAI Scheduler, 갱 스케줄링(Gang Scheduling)

> 하나가 아니라 세 계층. Karpenter는 노드를 동적으로 프로비저닝(provisioning)한다(1분 미만, Cluster Autoscaler보다 40% 빠름). KAI Scheduler는 갱 스케줄링(gang scheduling), 토폴로지 인식(topology awareness), 계층적 큐를 처리한다. 일곱 개 노드가 빠진 GPU 하나를 기다리며 비용을 태우는 8개 중 7개 부분 할당(partial allocation) 함정을 방지한다. 애플리케이션 수준 오토스케일러(NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler)는 CPU/DCGM 듀티 사이클(duty cycle)이 아니라 추론에 특화된 신호, 즉 큐 깊이(queue depth)와 KV 캐시 사용률로 스케일링한다. 전형적인 HPA 함정은 `DCGM_FI_DEV_GPU_UTIL`이 듀티 사이클 측정이라는 점이다: 100%는 10개 요청일 수도 100개일 수도 있다. vLLM은 KV 캐시 메모리를 미리 할당하므로, 메모리는 결코 스케일 다운을 트리거하지 않는다. 이 레슨은 세 계층을 조합하고, 추론 도중 실행 중인 GPU 작업을 종료시키는 기본 Karpenter `WhenEmptyOrUnderutilized` 정책을 피하는 법을 가르친다.

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 세 가지 오토스케일링 계층(노드 프로비저닝, 갱 스케줄링, 애플리케이션 수준)을 도식화하고 각 계층에서 사용되는 도구를 명명하기.
- vLLM에 대해 `DCGM_FI_DEV_GPU_UTIL`이 왜 잘못된 HPA 신호인지 설명하고 두 가지 대체물(큐 깊이, KV 캐시 사용률)을 명명하기.
- 갱 스케줄링과 KAI Scheduler가 방지하는 부분 할당 실패 모드(8개 GPU 중 7개 유휴)를 설명하기.
- 실행 중인 GPU 작업을 종료시키는 Karpenter 통합(consolidation) 정책(`WhenEmptyOrUnderutilized`)을 명명하고 2026년 안전한 대안을 진술하기.

## 문제 (The Problem)

팀이 Kubernetes에서 LLM 서빙 서비스를 출시한다. `DCGM_FI_DEV_GPU_UTIL`을 신호로 하여 HPA를 설정했다. 서비스는 업무 시간 동안 100% 사용률에 고정된다. HPA는 스케일 업하지 않는다. 이미 가득 찼다고 보기 때문이다. 레플리카(replica)를 수동으로 추가하니 TTFT가 떨어진다. 그래도 HPA는 스케일링하지 않는다. 신호가 거짓말을 하고 있다.

별개로, 노드에는 Cluster Autoscaler를 쓴다. 새벽 2시에 100만 토큰 프롬프트가 도착하고, 클러스터는 노드를 프로비저닝하는 데 3분을 쓰며, 요청은 타임아웃된다.

또 별개로, 2개 노드에 걸쳐 8개 GPU가 필요한 70B 모델을 배포한다. 클러스터에는 7개 GPU가 비어 있고 1개는 3개 노드에 흩어져 있다. Cluster Autoscaler는 빠진 GPU 1개를 위해 노드를 프로비저닝한다. Kubernetes가 마지막 GPU를 띄우는 동안 일곱 개 노드가 4분간 돈을 태우며 기다린다.

세 계층, 세 가지 다른 실패 모드. 2026년의 GPU 인식 오토스케일링은 "HPA를 켜는 것"이 아니다. 노드 프로비저닝, 갱 스케줄링, 애플리케이션 신호 오토스케일링을 조합하는 일이다.

## 개념 (The Concept)

### 계층 1: 노드 프로비저닝 (Karpenter)

Karpenter는 대기 중인 파드(pod)를 감시하고 약 45~60초 내에 노드를 프로비저닝한다(Cluster Autoscaler는 GPU 노드에 대해 일반적으로 90~120초 소요). `NodePool` 제약에 따라 인스턴스 타입을 동적으로 고른다. 파드가 8개 H100을 필요로 하는데 클러스터에 일치하는 노드가 없으면, Karpenter는 기존 그룹을 스케일링하는 대신 하나를 직접 프로비저닝한다.

**통합 함정**: Karpenter의 기본 `consolidationPolicy: WhenEmptyOrUnderutilized`는 GPU 풀에 위험하다. 더 저렴한 적정 크기 인스턴스로 파드를 이주시키려고 실행 중인 GPU 노드를 종료하기 때문이다. 추론 워크로드에서 이는 실행 중인 요청을 축출하고 새 노드에서 70B 모델을 다시 로드한다는 뜻이다. 손실은 수 분의 용량에 더해 요청 실패다.

GPU 풀을 위한 안전한 설정:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

Karpenter가 한 시간 후 진짜로 비어 있는 노드를 통합하게 하되 실행 중인 작업은 결코 축출하지 않게 한다.

### 계층 2: 갱 스케줄링 (KAI Scheduler)

KAI Scheduler(프로젝트명 "Karp"였다가 이후 개명)는 기본 kube-scheduler가 하지 않는 것을 처리한다:

**갱 스케줄링**: 전부 아니면 전무로 스케줄링한다. 8개 GPU가 필요한 분산 추론 파드는 8개가 모두 함께 시작되거나 하나도 시작되지 않는다. 이것이 없으면 부분 할당 함정에 빠진다: 8개 중 7개 파드가 시작되어 무기한 기다리며 돈을 태운다.

**토폴로지 인식**: 어느 GPU가 NVLink를 공유하는지, 어느 것이 같은 랙에 있는지, 어느 것 사이에 InfiniBand가 있는지 안다. 그에 따라 파드를 배치한다. DeepSeek-V3 67B 텐서 병렬(tensor-parallel) 워크로드는 하나의 NVLink 도메인에 머물러야 한다. KAI Scheduler는 그것을 존중한다.

**계층적 큐**: 여러 팀이 우선순위와 쿼터를 가지고 같은 GPU 풀을 두고 경쟁한다. 팀 A의 프로덕션 압박은 우선순위 규칙이 허용할 때에만 팀 B의 학습 작업에 의해 선점된다.

KAI는 kube-scheduler와 나란히 보조 스케줄러로 배포된다. 워크로드에 이를 사용하도록 어노테이션(annotation)을 단다. Ray와 vLLM 프로덕션 스택 둘 다 통합한다.

### 계층 3: 애플리케이션 수준 신호

**HPA 함정**: `DCGM_FI_DEV_GPU_UTIL`은 듀티 사이클 지표다. 각 샘플링 간격에 GPU가 작업을 하고 있었는지를 측정한다. 100% 사용률은 동시 요청 10개를 의미할 수도, 100개를 의미할 수도 있다. 어느 쪽이든 GPU는 바빴다. 듀티 사이클로 스케일링하는 것은 눈을 감고 스케일링하는 것이다.

더 나쁜 것은, vLLM과 유사 엔진들이 KV 캐시 메모리를 미리 할당한다는 점이다(`--gpu-memory-utilization`까지). 메모리 사용량은 요청 하나에서도 90% 가까이 유지된다. 메모리 기반 HPA는 결코 스케일 다운하지 않는다.

**2026년 대체 신호**:

- 큐 깊이(프리필(prefill)을 기다리는 요청 수).
- KV 캐시 사용률(활성 시퀀스에 할당된 블록의 비율).
- 레플리카별 P99 TTFT(SLA 신호).
- 굿풋(goodput)(초당 모든 SLO를 충족하는 요청).

NVIDIA Dynamo Planner와 llm-d Workload Variant Autoscaler는 이 신호들을 소비하고 레플리카를 스케일링한다. LLM 서빙에서는 HPA를 완전히 대체한다.

### 무엇을 언제 쓸지

| 스케일 결정 | 도구 |
|----------------|------|
| 노드 추가/제거 | Karpenter |
| 다중 GPU 작업 스케줄링 | KAI Scheduler |
| 레플리카 추가/제거 | Dynamo Planner / llm-d WVA (또는 큐 깊이 기반 커스텀 HPA) |
| GPU 타입 선택 | Karpenter NodePool |
| 낮은 우선순위 선점 | KAI Scheduler 큐 |

### 분리형(disaggregated) 프리필/디코드는 모든 것을 복잡하게 만든다

분리형 프리필/디코드(Phase 17 · 17)를 돌린다면, 서로 다른 스케일링 트리거를 가진 두 파드 클래스가 있다: 프리필 파드는 큐 깊이로 스케일링하고, 디코드 파드는 KV 캐시 압력으로 스케일링한다. llm-d는 이것들을 역할별 HPA를 가진 별도 `Services`로 노출한다. 둘 앞에 단일 HPA를 두려 하지 마라.

### 콜드 스타트도 여기서 중요하다

콜드 스타트 완화(Phase 17 · 10)는 노드 프로비저닝 시간이 사용자에게 보이게 되는 지점이다. Karpenter의 45~60초 예열에 더해 20GB 모델 로드, 엔진 초기화는 제로에서 시작하는 요청이 2~5분 걸린다는 의미다. SLO에 중요한 경로에는 예열된 풀(`min_workers=1`)을 유지하거나, 애플리케이션 계층에서 Modal 스타일 체크포인팅(checkpointing)을 사용하라.

### 기억해야 할 숫자

- Karpenter 노드 프로비저닝: 약 45~60초 vs Cluster Autoscaler 약 90~120초(GPU 노드).
- KAI Scheduler는 부분 할당 낭비를 방지한다. 8개 중 7개 함정.
- HPA 신호로서 `DCGM_FI_DEV_GPU_UTIL`: 망가짐. 큐 깊이나 KV 사용률을 쓰라.
- Karpenter `WhenEmptyOrUnderutilized`: 실행 중인 GPU 작업을 종료한다. 추론에는 `WhenEmpty + consolidateAfter: 1h`를 쓰라.

```figure
autoscaling
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 버스트성 GPU 워크로드에서 3계층 오토스케일러를 시뮬레이션한다. 순진한 HPA(듀티 사이클), 큐 깊이 HPA, KAI 갱 스케줄링 스케일링을 비교한다. 미충족 요청, 유휴 GPU 분(minute), 그리고 종합 점수를 보고한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-gpu-autoscaler-plan.md`를 만들어낸다. 클러스터 토폴로지, 워크로드 형태, SLO가 주어지면, 3계층 오토스케일링 계획을 설계한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 버스트성 워크로드에서, 순진한 듀티 사이클 HPA가 떨어뜨리는데 큐 깊이 HPA가 잡아내는 요청은 몇 개인가? 그 차이는 어디서 오는가?
2. H100 SXM5에서 Llama 3.3 70B FP8을 서빙하는 클러스터를 위한 Karpenter NodePool을 설계하라. `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter`, 그리고 이 노드들에서 비-GPU 워크로드를 막는 테인트(taint)를 명시하라.
3. 팀이 "GPU는 가용한데 파드가 스케줄링되지 않는다"는 이유로 배포가 Pending에 멈췄다고 보고한다. 진단하라. 이것은 Karpenter인가, kube-scheduler인가, KAI Scheduler인가? 어떤 지표가 확인해 주는가?
4. 분리형 프리필 파드를 오토스케일링할 신호 하나와 디코드 파드를 위한 다른 신호를 고르라. 둘 다 정당화하라.
5. P99 TTFT > 10초에서 하루 평균 60건의 요청 드롭 이벤트가 발생하는 24x7 프로덕션 서비스에서 `WhenEmptyOrUnderutilized` 통합 함정의 비용을 계산하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| Karpenter | "노드 프로비저너" | Kubernetes 노드 오토스케일러. 1분 미만 프로비저닝 |
| Cluster Autoscaler | "옛 스케일러" | Kubernetes 노드 오토스케일러 전신. 더 느리고 그룹 기반 |
| KAI Scheduler | "GPU 스케줄러" | 갱 + 토폴로지 + 큐를 위한 보조 스케줄러 |
| Gang scheduling | "전부 아니면 전무" | N개 파드를 원자적으로 스케줄링하거나 전부 연기 |
| Topology awareness | "랙 인식" | NVLink/IB/랙 배치에 기반해 파드 배치 |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 사용률" | 듀티 사이클 지표. LLM의 스케일링 신호가 아님 |
| Queue depth | "대기 요청" | 프리필 바운드 스케일링을 위한 올바른 HPA 신호 |
| KV cache utilization | "메모리 압력" | 디코드 바운드 스케일링을 위한 올바른 HPA 신호 |
| Consolidation | "Karpenter 통합" | 더 저렴한 인스턴스 타입으로의 노드 종료 |
| `WhenEmpty + 1h` | "안전한 통합" | 실행 중인 GPU 작업을 축출하지 않는 정책 |

## 더 읽을거리 (Further Reading)

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler): 설계 문서와 설정 예제.
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/): 통합 정책 의미론과 GPU 안전 기본값.
- [NVIDIA: Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) Dynamo Planner 스케일링 신호.
- [Ray docs: KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) Ray 통합 패턴.
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html): 매니지드 Kubernetes 특화 가이드.
- [llm-d GitHub](https://github.com/llm-d/llm-d): Workload Variant Autoscaler 설계.
