# FP8과 NVFP4를 활용한 Blackwell에서의 TensorRT-LLM

> TensorRT-LLM은 NVIDIA 전용이지만 Blackwell에서 이긴다. Dynamo 오케스트레이션을 갖춘 GB200 NVL72에서, SemiAnalysis InferenceX는 2026년 1~2분기에 120B 모델에서 100만 토큰(token)당 0.012달러를 측정했다. 이는 H100 + vLLM의 100만당 0.09달러 대비 7배의 경제적 격차다. 스택은 세 가지 부동소수점(floating-point) 체제가 복리로 쌓인 것이다: FP8은 KV 캐시와 어텐션 커널에 필요한 동적 범위(dynamic range)를 가지므로 여전히 결정적이다. NVFP4(4비트 마이크로스케일링)는 가중치(weight)와 활성값(activation)을 다룬다. 다중 토큰 예측(multi-token prediction, MTP)과 분리형(disaggregated) 프리필/디코드가 그 위에 2~3배를 더 얹는다. Day-0 모델 지원은 학습 후 변환 없이 FP4 가중치를 직접 로드한다. 2026년 엔지니어링 팀이 빠지기 쉬운 함정: TRT-LLM은 폐쇄된 NVIDIA 스택이므로, 채택은 이식성과 처리량을 맞바꾸는 일이다. 결정하기 전에 자신의 모델과 하드웨어 조합으로 직접 계산을 돌려 보라.

**Type:** Learn
**Languages:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 13 (Quantization)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 가중치가 NVFP4일 때조차 FP8이 KV 캐시와 어텐션에 왜 여전히 결정적인지 설명하기.
- BF16, FP8, NVFP4 하에서 프런티어 모델의 HBM 풋프린트를 계산하고 절감이 어디서 오는지 추론하기.
- TRT-LLM이 활용하는 Blackwell 특화 기능(day-0 FP4, MTP, 분리형 서빙, all-to-all 프리미티브)을 명명하기.
- TRT-LLM의 NVIDIA 종속이 Hopper에서의 vLLM 대비 7배 비용 격차의 가치가 있는 때를 판단하기.

## 문제 (The Problem)

2026년 추론(inference) 경제학의 프런티어는 "달러당 토큰 몇 개"이다. 답은 네 가지 쌓인 선택에 달려 있다: 하드웨어 세대(Hopper H100/H200 vs Blackwell B200/GB200), 정밀도(BF16 → FP8 → NVFP4), 서빙 엔진(vLLM vs SGLang vs TRT-LLM), 그리고 오케스트레이션(평범 vs 분리형 vs Dynamo).

vLLM을 갖춘 Hopper에서, 120B MoE는 100만 토큰당 약 0.09달러로 돈다. TRT-LLM + Dynamo를 갖춘 Blackwell에서, 같은 모델은 약 0.012달러로 돈다. 7배 저렴하다. 그 격차의 일부는 하드웨어다(Blackwell은 Hopper 대비 GPU당 LLM 처리량이 11~15배). 일부는 스택이다: FP4 가중치, MTP 드래프트, 분리형 프리필/디코드, 그리고 MoE 전문가 통신을 위한 NVLink 5 all-to-all.

이것을 NVIDIA 스택 밖에서 복제할 수는 없다. 그것이 트레이드오프다. 경제성을 얻는 대신 이식성을 내준다. 어느 스택 선택이 격차의 어느 몫을 만드는지 이해하는 것이 이 레슨의 핵심이다.

## 개념 (The Concept)

### KV 캐시에 FP8이 여전히 바닥인 이유

2026년의 흔한 실수: NVFP4가 모든 곳에 적용된다고 가정하는 것. 그렇지 않다. KV 캐시는 넓은 동적 범위에 걸친 어텐션 키와 값을 저장하므로 FP8(8비트 부동소수점)이 필요하다. KV를 FP4로 양자화하면 치명적인 정확도 손실이 생긴다. 분포의 꼬리가 떨어져 나가고 어텐션 점수가 붕괴한다. FP8의 지수 비트가 KV 캐시에 필요한 범위를 준다.

NVFP4(2025-2026)는 가중치와 활성값에 적용된다. 마이크로스케일링(microscaling): 각 가중치 블록이 자신의 스케일 팩터를 가져, 작은 블록들이 텐서별 스케일 손실 없이 서로 다른 동적 범위에 걸칠 수 있다. 활성값의 경우, 활성값이 한 층 안에서 작은 범위이므로 FP4가 견딘다.

전형적인 Blackwell 구성:

- 가중치: NVFP4(4비트 마이크로스케일링).
- 활성값: NVFP4.
- KV 캐시: FP8.
- 어텐션 누산기: FP32(소프트맥스 안정성).

### TRT-LLM이 사용하는 Blackwell 특화 프리미티브

- **Day-0 FP4 가중치**: 모델 제공자가 FP4 가중치를 직접 출하한다. TRT-LLM은 학습 후 변환 없이 로드한다. FP4를 위한 AWQ / GPTQ 단계 없음.
- **다중 토큰 예측(MTP)**: EAGLE(Phase 17 · 05)과 같은 아이디어지만 TRT-LLM 빌드에 통합됨.
- **분리형 서빙**: 별도 GPU 풀의 프리필과 디코드, KV 캐시는 NVLink나 InfiniBand로 전송. Dynamo(Phase 17 · 20)와 같은 아이디어.
- **All-to-all 통신 프리미티브**: NVLink 5는 Hopper 대비 MoE 전문가 통신 지연 시간을 3배 줄였다. TRT-LLM의 MoE 커널은 이것에 맞춰 튜닝되어 있다.
- **NVFP4 + MXFP8 마이크로스케일링**: Blackwell 텐서 코어에서의 하드웨어 가속 스케일 팩터 처리.

### 외워야 할 숫자

- TRT-LLM을 통해 GPT-OSS-120B에서 HGX B200 100만 토큰당 0.02달러.
- Dynamo(TRT-LLM 오케스트레이션)를 통해 GB200 NVL72 100만 토큰당 0.012달러.
- 비교 가능한 워크로드에서 H100 + vLLM ≈ 100만 토큰당 0.09달러.
- 3개월간의 TRT-LLM 업데이트에서 2.8배 처리량 이득(2026).
- GPU당 LLM 처리량 11~15배, Blackwell vs Hopper.
- MLPerf Inference v6.0(2026년 4월): Blackwell이 제출된 모든 작업을 지배.

### FP4가 실제로 품질에서 치르는 비용

NVFP4는 공격적이다. 추론 중심 워크로드(연쇄 추론, 수학, 긴 컨텍스트의 코드 생성)에서, FP4 가중치는 눈에 띄게 저하된다. 블록별 보정(calibration)이 완화하지만 제거하지는 않는다. 추론 모델을 출하하는 팀은 종종 타협으로 FP8 가중치 + FP4 활성값을 쓰거나, FP8을 전반에 쓰는 H200을 고수한다.

규칙: NVFP4 가중치를 결정하기 전에 항상 자신의 평가 셋에서 작업 품질을 검증하라.

### 이것이 왜 NVIDIA 종속 결정인가

TRT-LLM은 C++ + CUDA + 폐쇄 소스 커널이다. 모델은 특정 GPU SKU에 맞춰 컴파일되어야 한다. AMD 없음, Intel 없음, ARM 없음. 인프라 전략이 멀티 벤더라면, TRT-LLM 서빙 계층은 아예 출발선에도 서지 못한다. 혼합 하드웨어에서는 vLLM으로 서빙하면 된다. NVIDIA 전용이라면, 7배 격차가 종속의 값을 충분히 치러 준다.

### 2026년 실용 레시피

연 1억 달러 이상의 추론 청구액이라면, Hopper + vLLM에서 돌리는 것은 7~10배의 절감 기회를 그냥 흘려보내는 셈이다. 비용 지배적 워크로드를 Blackwell + TRT-LLM + Dynamo로 마이그레이션하라. 모델 반복 속도를 위해 실험 계층은 H100 + vLLM에 유지하라. 프로덕션 전에 각 NVFP4 변환 모델에서 품질을 검증하라.

### 분리화 보너스

TRT-LLM의 분리형 서빙(별도 프리필과 디코드 풀)은 Phase 17 · 20에서 깊이 다룬다. Blackwell에서, 승수가 쌓인다: FP4 가중치 × MTP 속도 향상 × 분리형 배치 × 캐시 인식 라우팅. 7배 숫자는 이 전체 스택을 가정한다.

```figure
pipeline-parallel
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 세 스택에 걸쳐 모델의 HBM 풋프린트, 디코드 처리량(메모리 바운드 체제), 100만 토큰당 달러를 계산한다: H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM. 실행하여 복리 효과와 각 변경이 기여하는 격차의 몫을 보라.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-trtllm-blackwell-advisor.md`를 만들어낸다. 워크로드, 모델 크기, 연간 토큰 볼륨이 주어지면, Blackwell + TRT-LLM 스택이 NVIDIA 종속의 가치가 있는지 결정한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 활성 파라미터 30%인 120B MoE에서, H100 BF16, H100 FP8, B200 NVFP4/FP8의 메모리 대역폭 제한 디코드 처리량을 계산하라. 가장 큰 도약은 어디서 오는가?
2. 한 고객이 H100 + vLLM에 연 200만 달러를 쓴다. 7배 경제적 격차가 주어졌을 때, 12개월 안에 TRT-LLM 마이그레이션을 분할상환하기 위해 사야 할 Blackwell GPU의 손익분기 개수는 얼마인가?
3. NVFP4 가중치 변환 후 MATH에서 정확도가 3점 떨어지는 것을 본다. 두 회복 경로를 명명하라: 하나는 품질 우선(FP8 가중치 유지), 하나는 비용 우선(도메인 내 데이터로 보정).
4. MLPerf v6.0 추론 결과를 읽어라. 어느 작업이 Blackwell-대-Hopper 격차가 가장 작으며, 왜인가?
5. 128k 컨텍스트에서 NVFP4 가중치 + FP8 KV 캐시의 405B 모델에 필요한 HBM을 계산하라. 단일 GB200 NVL72 노드에 들어가는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| FP8 | "8비트 부동소수점" | 8비트 부동소수점. 동적 범위 때문에 KV 캐시와 어텐션에 사용 |
| NVFP4 | "4비트 마이크로" | NVIDIA의 4비트 마이크로스케일링 FP 포맷. Blackwell에서 가중치와 활성값 |
| MXFP8 | "MX 8" | 마이크로스케일링 FP8 변형. Blackwell 텐서 코어에서 하드웨어 가속 |
| Day-0 FP4 | "FP4 가중치 출하" | 모델 제공자가 이미 FP4인 가중치를 출시. 학습 후 변환 단계 없음 |
| MTP | "다중 토큰 예측" | TRT-LLM의 통합된 추측 디코딩 드래프트(Phase 17 · 05) |
| Disaggregated serving | "프리필/디코드 분리" | 별도 GPU 풀의 프리필과 디코드. KV는 NVLink/IB로 전송 |
| All-to-all | "MoE 전문가 통신" | 토큰을 전문가 GPU로 라우팅하는 통신 패턴. NVLink 5가 3배 단축 |
| InferenceX | "SemiAnalysis 추론 벤치" | 2026년 업계에서 인정받는 토큰당 비용 벤치마크 |

## 더 읽을거리 (Further Reading)

- [NVIDIA: Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) 2026년 4월 MLPerf 결과.
- [NVIDIA: MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) NVLink 5 all-to-all과 MoE 커널.
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html): 공식 엔진 문서.
- [NVIDIA: Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) TRT-LLM 위의 분리형 오케스트레이션.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/): Blackwell 숫자를 공표하는 벤치마크 스위트.
