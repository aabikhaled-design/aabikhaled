# 프로덕션 양자화(Production Quantization): AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> 양자화(quantization) 포맷은 보편적 선택이 아니라 하드웨어, 서빙 엔진(serving engine), 워크로드에 따라 정해진다. GGUF Q4_K_M 또는 Q5_K_M은 llama.cpp와 Ollama로 서빙되며 CPU와 엣지(edge)를 차지한다. 같은 베이스(base)에서 멀티 LoRA가 필요할 때는 vLLM 안에서 GPTQ가 앞선다. Marlin-AWQ 커널을 갖춘 AWQ는 7B급 모델의 INT4에서 최고의 Pass@1과 함께 약 741 토큰(token)/초를 내며, 2026년 데이터센터 프로덕션 기본값이다. FP8은 Hopper, Ada, Blackwell에서 중간 지점으로, 거의 무손실이며 널리 지원된다. NVFP4와 MXFP4(Blackwell 마이크로스케일링)는 공격적이라 블록별 검증이 필요하다. 함정은 두 가지다. 보정(calibration) 데이터셋은 배포 도메인과 일치해야 하고, KV 캐시는 가중치(weight) 양자화와 별개다. "내 모델은 이제 4GB"라는 AWQ의 교훈은 프로덕션 배치(batch) 크기에서의 10~30GB KV 캐시를 잊는다.

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 여섯 가지 프로덕션 양자화 포맷과 2026년 그것들의 스위트 스폿을 명명하기.
- 하드웨어(CPU vs GPU, Hopper vs Blackwell), 엔진(vLLM, TRT-LLM, llama.cpp), 워크로드(일상 채팅, 추론, 멀티 LoRA)가 주어졌을 때 포맷을 고르기.
- 선택한 포맷에 대해 절약된 가중치 메모리와 손대지 않은 채 남은 KV 캐시를 계산하기.
- 도메인 트래픽에서 양자화된 모델을 저하시키는 보정 데이터셋 함정을 명명하기.

## 문제 (The Problem)

양자화는 메모리와 HBM 대역폭을 줄이는데, 그것이 바로 디코드(decode)가 필요로 하는 것이다. FP16 70B 모델은 140GB의 가중치다. 가중치를 INT4(AWQ 또는 GPTQ)로 양자화하면 모델은 35GB가 된다. KV 캐시를 위한 여유와 함께 하나의 H100에 들어가는데, 2k 컨텍스트의 128 동시 시퀀스에서 KV 캐시만 20~30GB이므로 이것이 중요하다.

하지만 양자화는 공짜가 아니다. 공격적 양자화는 품질을 저하시키며, 특히 추론 중심 작업에서 그렇다. 포맷마다 작동하는 엔진이 다르고, 하드웨어마다 네이티브로 지원하는 정밀도가 다르다. 2026년의 난립한 포맷 지형은 실재하므로 남의 선택을 베낄 수 없다. 자기 스택에 맞춰 골라야 한다.

## 개념 (The Concept)

### 여섯 가지 포맷

| 포맷 | 비트 | 스위트 스폿 | 엔진 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, 엣지, 노트북 | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM에서의 멀티 LoRA | vLLM, TGI |
| AWQ | 4 | 데이터센터 GPU 프로덕션 | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 데이터센터 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 멀티 유저 | TRT-LLM |
| NVFP4 | 4 | Blackwell 멀티 유저 | TRT-LLM |

### GGUF: CPU/엣지 기본값

GGUF는 그 자체가 양자화 방식이 아니라 파일 포맷으로, K-quant 변형(Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0)을 하나의 컨테이너에 묶는다. 프로덕션 기본값은 Q4_K_M과 Q5_K_M으로, 4~5비트에서 거의 BF16 품질을 낸다. llama.cpp가 가장 빠른 CPU 추론(inference) 엔진이므로 CPU나 엣지 서빙에는 이 포맷이 최선이다.

vLLM에서의 처리량 페널티: 7B에서 약 93 토큰/초: 그 포맷은 GPU 커널에 최적화되어 있지 않다. 배포 타깃이 CPU/엣지일 때 GGUF를 쓰라. 그 외에는 아니다.

### GPTQ: vLLM에서의 멀티 LoRA

GPTQ는 보정 패스를 가진 학습 후 양자화 알고리즘이다. Marlin 커널이 GPU에서 그것을 빠르게 만든다(비-Marlin GPTQ 대비 2.6배 속도 향상). 7B에서 약 712 토큰/초.

고유한 이점은 GPTQ-Int4가 vLLM에서 LoRA 어댑터를 지원한다는 것이다. 베이스 모델에 더해 10~50개의 파인튜닝(fine-tuning)된 변형(각각 LoRA로)을 서빙한다면 GPTQ가 답이다. NVFP4는 2026년 초 기준 아직 LoRA를 지원하지 않는다.

### AWQ: 데이터센터 GPU 기본값

활성화 인식 가중치 양자화(Activation-aware Weight Quantization). 양자화 동안 가장 두드러진 약 1%의 가중치를 보호한다. Marlin-AWQ 커널은 단순 구현 대비 10.9배 빠르다. 7B에서 약 741 토큰/초로, INT4 포맷 중 Pass@1이 가장 높다.

멀티 LoRA(GPTQ)나 공격적 Blackwell FP4(NVFP4)가 필요하지 않은 한 새 GPU 서빙에는 AWQ를 고르라.

### FP8: 신뢰할 수 있는 중간

8비트 부동소수점(floating point). 거의 무손실. 널리 지원됨. Hopper 텐서 코어가 FP8을 네이티브로 가속한다. Blackwell이 상속한다. FP8은 품질이 타협 불가능할 때(추론, 의료, 코드 생성) 안전한 2026년 기본값이다. 메모리 절감은 INT4의 절반이지만 품질 위험은 훨씬 낮다.

### MXFP4 / NVFP4: Blackwell 공격적

마이크로스케일링 FP4. 각 가중치 블록이 자신의 스케일 팩터를 갖는다. 공격적이지만 Blackwell 텐서 코어에서 하드웨어 가속된다. FP8 대비 토큰당 바이트를 절반으로: Phase 17 · 07의 경제적 이점.

주의사항:
- 아직 LoRA 지원 없음(2026년 초).
- 추론 중심 워크로드에서 품질 저하가 눈에 띔.
- 모델별로 자체 평가 셋에서 검증하라.

### 보정 함정

AWQ와 GPTQ는 보정 데이터셋을 필요로 한다. 일반적으로 C4나 WikiText를 쓴다. 도메인 모델(코드, 의료, 법률)의 경우 일반 웹 텍스트로 보정하면 알고리즘이 어느 가중치를 보호할지 잘못 판단한다. HumanEval에서의 Pass@1이 몇 점 떨어질 수 있다.

해결: 도메인 내 데이터로 보정하라. 수백 개의 도메인 샘플이면 보통 충분하다. 출하 전에 평가 셋에서 테스트하라.

### KV 캐시 함정

AWQ는 가중치를 4비트로 줄인다. KV 캐시는 별개이며 FP16/FP8에 머문다. AWQ를 가진 70B 모델의 경우:

- 가중치: 약 35GB(140GB에서 INT4).
- 128 동시 × 2k 컨텍스트에서 KV 캐시: 약 20GB.
- 활성값(activation): 약 5GB.
- 합계: 약 60GB: H100 80GB에 들어감.

"내 모델을 4GB로 양자화했다"는 단순한 계산은 나머지 30~50GB를 빠뜨린다. HBM 전체를 두고 예산을 잡아야 한다.

별개로, KV 캐시 양자화(FP8 KV 또는 INT8 KV)는 자체적인 트레이드오프를 가진 다른 선택이다. 어텐션 정확도에 직접 영향을 미치므로 공짜 이점이 아니다.

### AWQ INT4는 추론에 위험하다

연쇄 추론, 수학, 긴 컨텍스트의 코드 생성은 공격적 양자화에서 눈에 띄게 품질이 떨어진다. AWQ INT4는 MATH에서 약 3~5점을 잃는다. 추론 중심 워크로드라면 FP8이나 BF16을 출하하고 메모리 비용을 감수하라.

### 2026년 선택 가이드

- CPU/엣지 서빙: GGUF Q4_K_M. 끝.
- GPU 서빙, 일상 채팅, LoRA 없음: AWQ.
- GPU 서빙, 멀티 LoRA: Marlin을 가진 GPTQ.
- 추론 워크로드: FP8.
- Blackwell 데이터센터, 검증된 품질: NVFP4 + FP8 KV.
- 애매한 경우: 각 후보 포맷에서 1,000 샘플 평가를 돌려라.

```figure
gpu-memory-breakdown
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 다양한 모델 크기에 대해 여섯 포맷 전반의 메모리 풋프린트(가중치 + KV + 활성값)와 상대 처리량을 계산한다. KV 캐시가 지배하는 곳, 가중치 압축이 이득이 되는 곳, FP8이 안전한 선택인 곳을 보여준다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-quantization-picker.md`를 만들어낸다. 하드웨어, 모델 크기, 워크로드 유형, 품질 허용치가 주어지면, 포맷을 고르고 보정/검증 계획을 만들어낸다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 2k 컨텍스트의 128 동시에서 70B 모델에 대해, 각 포맷의 총 HBM을 계산하라. 어느 포맷이 하나의 H100 80GB에 들어가게 하는가?
2. 7B 코딩 모델이 있다. 포맷을 고르고 정당화하라. 품질 허용치에 대해 틀렸다면, 회복 경로는 무엇인가?
3. 의료 도메인 모델을 위해 AWQ를 보정하는 데 필요한 보정 데이터셋 크기를 계산하라. 왜 데이터가 많은 것이 항상 더 좋지는 않은가?
4. Marlin-AWQ 커널 논문이나 릴리스 노트를 읽어라. AWQ가 7B에서 741 토큰/초를 치는 반면 순수 GPTQ는 약 712를 치는 이유를 세 문장으로 설명하라.
5. AWQ 가중치를 FP8 KV 캐시와 결합하는 것이 KV를 BF16에 유지하는 것 대비 언제 의미가 있는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| GGUF | "llama.cpp 포맷" | K-quant 변형을 묶는 파일 포맷. CPU/엣지 기본값 |
| Q4_K_M | "Q4 K M" | 4비트 K-quant 미디엄. 프로덕션 GGUF 기본값 |
| GPTQ | "지피티큐" | 보정을 가진 학습 후 INT4. vLLM에서 LoRA 지원 |
| AWQ | "에이더블유큐" | 활성화 인식 INT4. Marlin 커널. INT4에서 최고의 Pass@1 |
| Marlin kernels | "빠른 INT4 커널" | Hopper에서 INT4를 위한 커스텀 CUDA 커널. 10배 속도 향상 |
| FP8 | "8비트 부동소수점" | Hopper/Ada/Blackwell에서 안전한 정밀도 기본값 |
| MXFP4 / NVFP4 | "마이크로스케일링 4" | 블록별 스케일 팩터를 가진 Blackwell 4비트 FP |
| Calibration dataset | "보정 데이터" | 양자화 파라미터를 고르는 데 쓰는 입력 텍스트. 도메인과 일치해야 함 |
| KV cache quantization | "KV INT8" | 가중치와 별개인 선택. 어텐션 정확도에 영향 |

## 더 읽을거리 (Further Reading)

- [VRLA Tech: LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) 비교 벤치마크.
- [Jarvis Labs: vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) 포맷별 처리량 숫자.
- [PremAI: GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) 포맷별 선택.
- [vLLM docs: Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) 지원 포맷과 플래그.
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978): 원본 AWQ 정식화.
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323): 원본 GPTQ 정식화.
