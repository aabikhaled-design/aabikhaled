# 엣지 추론(Edge Inference): Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> 핵심 엣지 제약은 연산이 아니라 메모리 대역폭(memory bandwidth)이다. 모바일 DRAM은 50-90 GB/s에 머무는 반면, 데이터센터 HBM3는 2-3 TB/s를 넘긴다. 30-50배 차이다. 디코드(decode)는 메모리 바운드(memory-bound)이므로 이 차이가 결정적이다. 2026년 지형은 네 갈래로 나뉜다. Apple M4/A18 Neural Engine은 통합 메모리(unified memory, CPU↔NPU 복사 없음)와 함께 38 TOPS로 정점을 찍는다. Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon은 45 TOPS에 도달한다. WebGPU + WebLLM은 M3 Max에서 Llama 3.1 8B(Q4)를 ~41 tok/s로 실행한다(대략 네이티브의 70-80%). GitHub 스타 17.6k, OpenAI 호환 API, 모바일 커버리지 ~70-75%. NVIDIA Jetson Orin Nano Super(8GB)는 Llama 3.2 3B / Phi-3에 들어맞고, AGX Orin은 vLLM을 통해 gpt-oss-20b를 ~40 tok/s로 실행하며, Jetson T4000(JetPack 7.1)은 AGX Orin의 2배다. TensorRT Edge-LLM은 EAGLE-3, NVFP4, 청크드 프리필(chunked prefill)을 지원한다. CES 2026에서 Bosch, ThunderSoft, MediaTek가 시연했다.

**Type:** Learn
**Languages:** Python (stdlib, toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 모바일 LLM 추론이 왜 메모리 대역폭 바운드이고 연산은 부차적인지 설명하기.
- 네 가지 엣지 타깃(Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson)을 열거하고 각각을 사용 사례에 매칭하기.
- 2026년 WebGPU 커버리지 격차(따라잡는 중인 Firefox Android)와 Safari iOS 26 안착을 짚기.
- 타깃별 양자화 형식 고르기(ANE는 Core ML INT4 + FP16, Hexagon은 QNN INT8/INT4, 브라우저는 WebGPU Q4, Jetson Thor는 NVFP4).

## 문제 (The Problem)

한 고객이 온디바이스 챗봇을 원한다: 음성 우선, 기본적으로 프라이빗, 오프라인 동작. MacBook Pro M3 Max에서 Llama 3.1 8B Q4는 ~55 tok/s로 돌아간다. 괜찮다. iPhone 16 Pro에서 같은 모델은 3 tok/s로 돌아간다. 괜찮지 않다. Snapdragon 8 Gen 3을 쓰는 중급 안드로이드에서는 7 tok/s. Chrome Android v121+에서 WebGPU를 통해 브라우저에서는 기기에 따라 4-8 tok/s.

처리량 편차는 포팅(porting) 문제가 아니다. 대역폭 격차 × 양자화 형식 × NPU가 유저 스페이스(user-space)에서 접근 가능한지의 곱이다. 2026년 엣지 추론은 네 가지 다른 해법을 가진 네 가지 다른 문제다.

## 개념 (The Concept)

### 대역폭이 진짜 천장이다

디코드는 매 토큰마다 가중치 전체 집합을 읽는다. Q4의 7B 모델 하나는 3.5 GB다. 50 GB/s로 3.5 GB를 읽으면 70ms가 걸린다. 이론적 천장은 ~14 tok/s. 90 GB/s(고급 모바일 DRAM)에서는 천장이 ~25 tok/s로 이동한다. 이 수치 아래에서는 어떤 연산도 도움이 되지 않는다.

3 TB/s의 데이터센터 HBM3는 같은 3.5 GB를 1.2ms에 처리한다. 천장은 830 tok/s. 같은 모델, 같은 가중치. 다른 메모리 서브시스템.

### Apple Neural Engine (M4 / A18)

- 최대 38 TOPS. 통합 메모리(CPU와 ANE가 같은 풀을 공유): 복사 오버헤드 없음.
- Core ML + `.mlmodel` 컴파일 모델로 접근하거나, PyTorch를 통한 Metal Performance Shaders(MPS)로 접근.
- Llama.cpp Metal 백엔드는 ANE를 직접 쓰지 않고 MPS를 사용한다. 네이티브 ANE는 Core ML 변환이 필요하다.
- 2026년 iOS 앱의 최선의 실용 경로: INT4 가중치 + FP16 활성값을 쓰는 Core ML.

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 최대 45 TOPS. SoC 내에서 CPU 및 GPU와 통합되지만 별도 메모리 도메인.
- QNN(Qualcomm Neural Network) SDK와 AI Hub가 PyTorch/ONNX를 변환해 준다.
- 채팅 템플릿, Llama 3.2, Phi-3 모두 AI Hub에서 일급(first-class) 아티팩트로 출시된다.

### Intel / AMD NPU (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS. 소프트웨어는 Apple/Qualcomm에 뒤처진다. OpenVINO가 개선 중이지만 틈새다.
- Windows ARM 코파일럿 앱에 최적. 로컬 우선을 위한 AMD/Intel 데스크톱에서 네이티브.

### WebGPU + WebLLM

- WebGPU 컴퓨트 셰이더(compute shader)를 통해 브라우저에서 모델 실행. 설치 불필요.
- M3 Max에서 Llama 3.1 8B Q4를 ~41 tok/s: 같은 백엔드를 통해 대략 네이티브의 70-80%.
- WebLLM GitHub 스타 17.6k. OpenAI 호환 JS API. Apache 2.0.
- 2026년 커버리지: Chrome Android v121+, Safari iOS 26 GA, Firefox Android는 여전히 따라잡는 중. 전체 모바일 커버리지 ~70-75%.

### NVIDIA Jetson 패밀리

- Orin Nano Super(8GB): Llama 3.2 3B, Phi-3가 좋은 tok/s로 들어맞음.
- AGX Orin: vLLM을 통해 gpt-oss-20b를 ~40 tok/s로 실행.
- Thor / T4000(JetPack 7.1): AGX Orin의 2배 성능, EAGLE-3과 NVFP4 지원.
- TensorRT Edge-LLM(2026)은 EAGLE-3 추측 디코딩(speculative decoding), NVFP4 가중치, 청크드 프리필을 지원한다. 데이터센터 최적화를 엣지로 포팅한 것.

### 타깃별 양자화 선택

| 타깃 | 형식 | 비고 |
|--------|--------|-------|
| Apple ANE | INT4 가중치 + FP16 활성값 | Core ML 변환 경로 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 변환기 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | `mlc_llm convert_weight` + 컴파일된 `.wasm` 사용; GGUF는 미지원 |
| Jetson Orin Nano | Q4 GGUF 또는 TRT-LLM INT4 | 메모리 바운드 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 경로 |

### 엣지에서의 롱 컨텍스트 함정

Llama 3.1의 128K 컨텍스트는 데이터센터 기능이다. RAM 8 GB 폰에서 32K 토큰에 대해 모델 4 GB + KV 캐시 2 GB + OS 오버헤드 = OOM. 엣지 배포는 공격적인 KV 양자화(Q4 KV)를 수용하지 않는 한 컨텍스트를 4K-8K로 유지한다.

### 음성이 킬러 앱이다

음성 에이전트는 지연 시간에 민감하다(첫 토큰 < 500ms). 로컬 추론은 네트워크 지연 시간을 전적으로 제거한다. 음성-텍스트 변환(Whisper Turbo 변형이 엣지에서 실행됨)과 결합하면 엣지 추론은 프로덕션 품질의 음성 루프가 된다.

### 기억해야 할 숫자들

- Apple M4 / A18 ANE: 38 TOPS.
- Qualcomm Hexagon SD X Elite: 45 TOPS.
- WebLLM M3 Max: Llama 3.1 8B Q4에서 ~41 tok/s.
- AGX Orin: vLLM을 통해 gpt-oss-20b에서 ~40 tok/s.
- 데이터센터-엣지 대역폭 격차: 30-50배.
- WebGPU 모바일 커버리지: ~70-75%(Firefox Android 지연).

```figure
edge-bandwidth-pipe
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 엣지 타깃 전반에서 대역폭 바운드 수식으로부터 이론적 디코드 처리량 천장을 계산한다. 관측된 벤치마크와 비교하고 연산이 아니라 대역폭이 병목인 지점을 강조한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-edge-target-picker.md`를 생성한다. 플랫폼(iOS/Android/브라우저/Jetson), 모델, 지연 시간/메모리 예산이 주어지면 양자화 형식과 변환 파이프라인을 고른다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. Snapdragon 8 Gen 3(~77 GB/s 대역폭)에서 Q4의 7B 모델에 대해 디코드 천장을 계산하라. 관측된 6-8 tok/s와 비교하라. 런타임이 효율적인가?
2. 안드로이드의 WebGPU는 Chrome v121+를 요구한다. 구형 브라우저를 위한 폴백(fallback)을 설계하라. 같은 OpenAI 호환 API를 통한 서버 사이드.
3. iOS 앱에서 4K 컨텍스트 스트리밍이 필요하다. iPhone 16에서 활성 메모리 4 GB 미만으로 유지하게 해주는 모델/형식 조합은 무엇인가?
4. Jetson AGX Orin은 gpt-oss-20b를 40 tok/s로 실행한다. Jetson Nano는 3B만 들어맞는다. 제품이 둘 다 타깃으로 한다면, 추론 스택을 어떻게 통일할 것인가?
5. "WebLLM이 2026년에 프로덕션 준비가 되었는지"를 논하라. 커버리지, 성능, 그리고 Firefox Android 격차를 인용하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| ANE | "Apple 뉴럴 엔진" | M 시리즈와 A 시리즈의 온디바이스 NPU; 통합 메모리 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU; 접근에 QNN SDK |
| WebGPU | "브라우저 GPU" | W3C 표준화된 브라우저 GPU API; 2026년 Chrome/Safari |
| WebLLM | "브라우저 LLM 런타임" | MLC-LLM 프로젝트; Apache 2.0; OpenAI 호환 JS |
| Jetson | "NVIDIA 엣지" | Orin Nano / AGX / Thor / T4000 패밀리 |
| TRT Edge-LLM | "엣지 TensorRT" | 2026년 TensorRT-LLM의 엣지 포팅; EAGLE-3 + NVFP4 |
| 통합 메모리 (Unified memory) | "공유 풀" | CPU와 NPU가 같은 RAM을 봄; 복사 오버헤드 없음 |
| 대역폭 바운드 (Bandwidth-bound) | "메모리 제한" | 가중치를 읽는 바이트/초로 게이팅되는 디코드 |
| Core ML | "Apple 변환" | ANE 네이티브 모델을 위한 Apple 프레임워크 |
| QNN | "Qualcomm 스택" | Qualcomm Neural Network SDK |

## 더 읽을거리 (Further Reading)

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/): 지형과 벤치마크.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/): 2026년 엣지 포팅 발표.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2): 설계와 벤치마크.
- [Apple Core ML](https://developer.apple.com/documentation/coreml): ANE 네이티브 변환.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/): Hexagon용 사전 변환 모델.
