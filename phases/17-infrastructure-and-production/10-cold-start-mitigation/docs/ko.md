# 서버리스 LLM을 위한 콜드 스타트 완화(Cold Start Mitigation)

> 20GB 모델 이미지는 콜드(cold) 상태에서 서빙까지 가는 데 5~10분(7B)에서 20분 이상(70B)이 걸린다. 진정한 서버리스(serverless) 세계에서 그것은 예열이 아니라 장애다. 완화는 다섯 계층에서 작동한다: 사전 시딩된 노드 이미지(AWS의 Bottlerocket, 이중 볼륨 아키텍처), 모델 스트리밍(NVIDIA Run:ai Model Streamer, vLLM에 네이티브), GPU 메모리 스냅샷(Modal 체크포인트, 최대 10배 빠른 재시작), 웜 풀(warm pool)(`min_workers=1`), 계층형 로딩(ServerlessLLM의 NVMe→DRAM→HBM 파이프라인, 10~200배 지연 시간 감소), 그리고 KV 캐시(GB)가 아니라 입력 토큰(token)(KB)을 옮기는 라이브 마이그레이션(live migration)이다. Modal은 바닥값으로 2~4초 콜드 스타트를 공표한다. Baseten은 기본 5~10초, 사전 예열(pre-warming)로 1초 미만. 이 레슨은 다섯 계층을 측정하고, 예산하고, 쌓는 법을 가르친다.

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 콜드 스타트 완화의 다섯 계층을 열거하고 각 계층에서 하나의 도구나 패턴을 명명하기.
- 70B 모델에 대해 총 콜드 스타트 시간을 (노드 프로비저닝) + (가중치 다운로드) + (HBM으로 가중치 로드) + (엔진 초기화)의 합으로 계산하기.
- 라이브 마이그레이션이 왜 KV 캐시(GB)가 아니라 입력 토큰(KB)을 전송하는지, 그리고 페널티가 무엇인지(재계산) 설명하기.
- 웜 풀 트레이드오프(유휴 GPU에 대해 지불 또는 콜드 스타트 꼬리 수용)와 `min_workers > 0`이 필수가 되는 SLA 임계값을 명명하기.

## 문제 (The Problem)

서버리스 LLM 엔드포인트가 밤새 0으로 스케일된다. 오전 8시에 트래픽이 치솟는다. 첫 요청은 다음 동안 기다린다:

1. Karpenter가 GPU 노드를 프로비저닝: 45~60초.
2. 컨테이너가 가중치를 가진 30GB 이미지를 풀(pull): 120~300초.
3. 엔진이 가중치를 HBM에 로드: 모델 크기와 스토리지 속도에 따라 45~120초.
4. vLLM 또는 TRT-LLM이 CUDA 그래프, KV 캐시 풀, 토크나이저를 초기화: 10~30초.

합계: 한 토큰이 돌아오기 전에 220~510초(대략 3~8분)다. SLA는 2초다. 웜 풀(`min_workers=1`)을 출하하면 문제가 사라지는 것 같지만, 이제 24x7로 유휴 GPU 하나에 대해 지불한다. 서비스에 웜 레플리카(replica)를 하나씩 둔 5개 제품이 있다면, 단 한 명의 사용자가 호출했든 안 했든 월 5 × 24 × 30 = 3,600 GPU 시간이다.

콜드 스타트 완화는 항상 켜진 것의 지연 시간을 근사하면서 서버리스 경제성을 유지하는 방법이다.

## 개념 (The Concept)

### 계층 1: 사전 시딩된 노드 이미지 (Bottlerocket)

AWS에서, Bottlerocket의 이중 볼륨 아키텍처는 OS를 데이터에서 분리한다. 컨테이너 이미지가 미리 풀된 데이터 볼륨을 스냅샷하고, `EC2NodeClass`에서 스냅샷 ID를 참조하라. 새 노드는 가중치가 이미 로컬 NVMe에 있는 채로 부팅한다. 2단계와 3단계의 일부가 사라진다. Karpenter와 네이티브로 작동한다. 일반적 절감: 큰 모델에 대해 콜드 스타트당 2~4분.

GCP에서의 동등물: 사전 구워진 컨테이너 레이어를 가진 커스텀 VM 이미지. Azure에서: 같은 패턴의 매니지드 디스크 스냅샷.

### 계층 2: 모델 스트리밍 (Run:ai Model Streamer)

첫 요청에 답하기 전에 전체 파일을 로드하는 대신, 가중치를 GPU 메모리에 층별로 스트리밍하고 첫 트랜스포머 블록이 상주하자마자 처리를 시작한다. NVIDIA Run:ai Model Streamer는 vLLM 2026에 네이티브로 출하된다. S3, GCS, 로컬 NVMe와 작동한다. I/O를 연산 셋업과 겹치게 하여 큰 모델의 가중치 로드 시간을 대략 절반으로 줄인다.

### 계층 3: GPU 메모리 스냅샷 (Modal)

Modal은 첫 로드 후 GPU 상태(가중치, CUDA 그래프, KV 캐시 영역)의 체크포인트를 찍는다. 이후 재시작은 HBM으로 직접 역직렬화한다. 재초기화보다 10배 빠르다. 이것은 "2초 안에 웜 GPU를 부팅"에 가장 가까운 것이다. 트레이드오프: 스냅샷은 GPU 토폴로지별이라, Karpenter가 다른 SKU로 마이그레이션하면 재체크포인트한다.

### 계층 4: 웜 풀 (min_workers=1)

가장 단순한 완화: 레플리카 하나를 항상 준비 상태로 유지한다. 비용은 24x7로 GPU 하나의 시간당 요율이다. 산술은 작은 모델에는 가혹하고(30초 콜드 스타트를 피하려고 시간당 0.85~1.50달러를 낸다) 큰 모델에는 친절하다(5분 콜드 스타트를 피하려고 시간당 4달러를 낸다). 웜 풀이 필수가 되는 SLA 임계값: 일반적으로 70B 이상 모델에서 TTFT P99 < 60초.

### 계층 5: 계층형 로딩 (ServerlessLLM)

ServerlessLLM은 스토리지를 계층 구조로 취급한다: NVMe(빠르지만 큼), DRAM(중간이지만 계층화됨), HBM(작지만 즉각적). 가중치는 DRAM에 사전 로드되고, HBM으로는 온디맨드 로드된다. 논문은 순진한 디스크-HBM 대비 콜드 로드에서 10~200배 지연 시간 감소를 보고한다. 프로덕션 채택은 초기지만 vLLM과의 통합이 존재한다.

### 계층 6: 라이브 마이그레이션 (보너스 패턴)

노드가 사용 불가능해질 때(스팟 축출, 노드 드레인), 전통적 패턴은 다른 레플리카를 콜드 스타트하고 요청 큐를 비우는 것이다. 라이브 마이그레이션은 입력 토큰(킬로바이트)을 모델이 로드된 목적지로 옮기고 목적지에서 KV 캐시를 재계산한다. 재계산은 GB의 KV 캐시를 네트워크로 전송하는 것보다 저렴하다. 분리형(disaggregated) 배포에 적용 가능하다.

### 웜 풀 수학

P99 TTFT SLA가 2초인 서비스의 경우, 질문은 "웜 풀 예/아니오"가 아니라 "웜 레플리카 몇 개, 그리고 어느 경로가 그것을 받는가"이다.

- 고가치 대화형 경로(라이브 채팅, 음성 에이전트): `min_workers=1-2`.
- 백그라운드 배치 경로(야간 분류): 0으로 스케일 수용, 5~10분 콜드 스타트 허용 가능.
- 프리미엄 등급: 전용 용량을 가진 테넌트별 `min_workers`.

### 최적화 전에 측정하라

새 노드에서 70B 모델의 콜드 스타트 해부(예시):

| 단계 | 시간 | 완화 |
|-------|------|-----------|
| 노드 프로비저닝 | 50s | Bottlerocket + 사전 시딩 이미지, 웜 풀 |
| 이미지 풀 | 180s | 사전 시딩 데이터 볼륨(제거) |
| HBM으로 가중치 | 75s | 모델 스트리머(절반); GPU 스냅샷(제거) |
| 엔진 초기화 | 20s | 영속적 CUDA 그래프 캐시 |
| 첫 순방향 | 3s | 최소 본질적 지연 시간 |
| **총 콜드** | **328s** | |
| **완화 적용 총합** | **~15s** | 22배 감소 |

### 기억해야 할 숫자

- Modal 콜드 스타트: 2~4초(GPU 스냅샷 사용).
- Baseten 기본 콜드 스타트: 5~10초. 사전 예열로 1초 미만.
- 순수 70B 콜드 스타트: 3~8분.
- Run:ai Model Streamer: 약 2배 가중치 로드 속도 향상.
- ServerlessLLM 계층형 로딩: 10~200배 지연 시간 감소(논문 숫자).

```figure
cold-start-pipeline
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 각 완화가 있을 때와 없을 때의 콜드 스타트 경로를 모델링한다. 총 콜드 스타트 시간, 웜 풀 비용, 그리고 웜 풀이 본전을 뽑는 손익분기 요청 비율을 보고한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-cold-start-planner.md`를 만들어낸다. SLA, 모델 크기, 트래픽 형태가 주어지면, 어느 완화를 쌓을지 고른다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. SLO에서 추가 요청 드롭을 통해 콜드 스타트 세금을 내는 것보다 웜 레플리카가 더 저렴해지는 손익분기 요청 비율을 계산하라.
2. P99 TTFT SLA가 3초인 13B 모델을 배포한다. 그것을 달성하는 최소 완화 스택(가장 적은 계층)을 고르라.
3. Bottlerocket 사전 시딩은 이미지 풀을 제거하지만 가중치는 여전히 스냅샷에서 HBM으로 로드된다. 스냅샷 기반 NVMe가 7GB/s로 읽는다면 70B 모델의 실제 경과 시간을 계산하라.
4. 서버리스 제공자가 GPU 스냅샷(Modal)을 제공하는데 팀이 "스냅샷이 PII를 누출한다"며 거부한다. 양측을 논하라. 현실적 위험은 무엇이고, 완화는 무엇인가(임시 스냅샷, 암호화, 네임스페이스 격리)?
5. 계층형 웜 풀 정책을 설계하라: 유료 사용자, 체험 사용자, 배치 워크로드에 각각 웜 레플리카 몇 개? 수학을 보여라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| Cold start | "큰 멈춤" | 새 레플리카에서 요청부터 첫 토큰까지의 시간 |
| Warm pool | "항상 켜진 최소값" | 최소 하나의 레플리카를 준비 상태로 유지하는 `min_workers >= 1` |
| Pre-seeded image | "구워진 AMI" | 컨테이너 가중치가 사전 상주된 노드 이미지 |
| Bottlerocket | "AWS 노드 OS" | 이중 볼륨 스냅샷 지원을 가진 AWS 컨테이너 최적화 OS |
| Model streamer | "스트리밍 로드" | 가중치 I/O를 연산 셋업과 겹침 |
| GPU snapshot | "HBM으로 체크포인트" | 로드 후 GPU 상태를 직렬화. 재시작 시 역직렬화 |
| Tiered loading | "NVMe + DRAM + HBM" | 스토리지 계층 구조. 온디맨드 로드 |
| Live migration | "토큰 이동" | 입력(KB) 전송, 목적지에서 KV 재계산 |
| `min_workers` | "웜 레플리카" | 서버리스 최소 유지 개수 |
| Scale-to-zero | "완전 서버리스" | 유휴 시 비용 없음. 전체 콜드 스타트 세금 수용 |

## 더 읽을거리 (Further Reading)

- [Modal: Cold start performance](https://modal.com/docs/guide/cold-start) Modal의 공개된 벤치마크와 체크포인트 아키텍처.
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket): 사전 시딩 데이터 볼륨 스냅샷 패턴.
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer): 가중치 로드를 연산 셋업과 겹침.
- [Baseten: Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) 사전 예열 플레이북.
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu): 계층형 로딩 설계.
- [NVIDIA: Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) 분리형 배포를 위한 라이브 마이그레이션.
