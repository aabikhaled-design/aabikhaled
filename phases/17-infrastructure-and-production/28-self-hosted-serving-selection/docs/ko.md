# 셀프 호스팅 서빙 선택: llama.cpp, Ollama, TGI, vLLM, SGLang

> 네 엔진이 2026년 셀프 호스팅(self-hosted) 추론(inference)을 지배한다. 하드웨어, 규모, 생태계에 따라 고른다. **llama.cpp**는 CPU에서 가장 빠르다. 가장 넓은 모델 지원, 양자화(quantization)와 스레딩(threading)에 대한 완전한 제어. **Ollama**는 개발 노트북용 한 줄(one-command) 설치로, llama.cpp보다 약 15~30% 느리고(Go + CGo + HTTP 직렬화), 프로덕션(production) 유사 부하에서는 처리량(throughput) 격차가 3배다. **TGI는 2025년 12월 11일 유지보수 모드(maintenance mode)에 들어갔다**. 버그 수정만, vLLM보다 원시 처리량이 약 10% 느리지만 역사적으로 최상위 관측성(observability)과 HF 생태계 통합을 가졌다. 그 유지보수 상태는 장기적으로 위험한 베팅이 되게 한다. 새 프로젝트에는 SGLang이나 vLLM이 더 안전한 기본값이다. **vLLM**은 범용 프로덕션 기본값이다. v0.15.1(2026년 2월)이 PyTorch 2.10, RTX Blackwell SM120, H200 최적화를 추가한다. **SGLang**은 에이전트형(agentic) 멀티턴(multi-turn) / 접두사 집약(prefix-heavy) 전문가다. 프로덕션에서 400,000개 이상의 GPU(xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). 하드웨어 제약: CPU 전용 → llama.cpp만. AMD / 비-NVIDIA → vLLM만(TRT-LLM은 NVIDIA에 묶임). 2026 파이프라인(pipeline) 패턴: 개발 = Ollama, 스테이징 = llama.cpp, 프로덕션 = vLLM 또는 SGLang. 전 구간에 동일한 GGUF/HF 가중치(weight).

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Time:** ~45분

## 학습 목표 (Learning Objectives)

- 하드웨어(CPU / AMD / NVIDIA Hopper / Blackwell), 규모(사용자 1 / 100 / 10,000), 워크로드(workload)(일반 채팅 / 에이전트 / 긴 컨텍스트)가 주어졌을 때 엔진 고르기.
- 2026년 TGI 유지보수 모드 상태(2025년 12월 11일)와 그것이 왜 새 프로젝트를 vLLM 또는 SGLang으로 기울게 하는지 명명하기.
- 전 구간에 동일한 GGUF 또는 HF 가중치를 사용하는 개발/스테이징/프로덕션 파이프라인을 설명하기.
- "CPU 전용"이 llama.cpp를 강제하고 "AMD"가 TRT-LLM을 배제하는 이유를 설명하기.

## 문제 (The Problem)

팀이 새 셀프 호스팅 LLM 프로젝트를 시작한다. 한 엔지니어는 Ollama를 말하고, 다른 이는 vLLM을, 세 번째는 "TGI가 그냥 박스에서 꺼내자마자(out of the box) 되지 않나?"라고 말한다. 셋 다 서로 다른 맥락에서 옳지만, 어느 하나도 모든 경우에 옳지는 않다.

2026년에는 선택 트리(choice tree)가 중요하다. 하드웨어 첫째, 규모 둘째, 워크로드 셋째다. 그리고 2025년의 한 사건, 곧 TGI가 12월 11일 유지보수 모드에 들어간 일이 새 프로젝트의 기본값을 바꿔 놓았다.

## 개념 (The Concept)

### 다섯 엔진

| 엔진 | 최적 용도 | 비고 |
|--------|----------|-------|
| **llama.cpp** | CPU / 엣지 / 최소 의존성 / 가장 넓은 모델 지원 | CPU에서 가장 빠름, 완전한 제어 |
| **Ollama** | 개발 노트북, 단일 사용자, 한 줄 설치 | llama.cpp보다 15~30% 느림; 프로덕션 처리량 격차 3배 |
| **TGI** | HF 생태계, 규제 산업 | **유지보수 모드 2025년 12월 11일** |
| **vLLM** | 범용 프로덕션, 100명 이상 사용자 | 폭넓은 프로덕션 기본값; v0.15.1 2026년 2월 |
| **SGLang** | 에이전트형 멀티턴, 접두사 집약 워크로드 | 프로덕션에서 400,000개 이상 GPU |

### 하드웨어 우선 결정

**CPU 전용** → llama.cpp. Ollama도 되지만 더 느리다. 다른 어떤 엔진도 CPU에서 경쟁력이 없다.

**AMD GPU** → vLLM(AMD ROCm 지원). SGLang도 된다. TRT-LLM은 NVIDIA에 묶여 있어 제외된다.

**NVIDIA Hopper (H100 / H200)** → vLLM 또는 SGLang 또는 TRT-LLM. 셋 다 최상위.

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM이 처리량 선두(Phase 17 · 07). vLLM과 SGLang이 근소하게 뒤따른다.

**Apple Silicon (M 시리즈)** → llama.cpp(Metal). Ollama가 이것을 감싼다.

### 규모 둘째 결정

**사용자 1 / 로컬 개발** → Ollama. 한 줄, 첫 토큰(first-token)이 몇 초.

**사용자 10~100 / 소규모 팀** → vLLM 단일 GPU.

**사용자 100~1만 / 프로덕션** → vLLM production-stack(Phase 17 · 18) 또는 SGLang.

**사용자 1만 이상 / 엔터프라이즈** → vLLM production-stack + 분리(disaggregated)(Phase 17 · 17) + LMCache(Phase 17 · 18).

### 워크로드 셋째 결정

**일반 채팅 / Q&A** → 폭넓은 기본값에서 vLLM이 이긴다.

**에이전트형 멀티턴(도구, 계획, 메모리)** → SGLang의 RadixAttention(Phase 17 · 06)이 지배한다.

**접두사 재사용이 많은 RAG** → SGLang.

**코드 생성** → vLLM도 괜찮음; 캐시에서 SGLang이 약간 낫다.

**긴 컨텍스트(128K 이상)** → vLLM + 청크 프리필(chunked prefill); SGLang + 계층 KV(tiered KV).

### TGI 유지보수 함정

Hugging Face TGI는 2025년 12월 11일 유지보수 모드에 들어갔다. 앞으로 버그 수정만. 역사적으로: 최상위 관측성, 동급 최고의 HF 생태계 통합(모델 카드, 안전 도구), 원시 처리량에서 vLLM보다 약간 뒤.

2026년 새 프로젝트라면 TGI에서 벗어나는 쪽을 기본으로 잡아라. 기존 TGI 배포는 그대로 굴러가겠지만 결국 마이그레이션해야 한다. SGLang과 vLLM이 더 안전한 기본값이다.

### 파이프라인 패턴

개발(Ollama) → 스테이징(llama.cpp) → 프로덕션(vLLM). 전 구간에 동일한 GGUF 또는 HF 가중치를 쓴다. 엔지니어는 노트북에서 빠르게 반복하고, 스테이징은 프로덕션 양자화를 그대로 본떠 두며, 프로덕션이 실제 서빙을 맡는다.

### Ollama 주의점

Ollama는 개발에는 훌륭하지만 공유 프로덕션에는 그렇지 않다. Go HTTP 직렬화가 오버헤드를 더하고, 동시성 관리가 vLLM보다 단순하며, OpenTelemetry 지원이 뒤처진다. 한 사용자, 한 줄로 빛나는 자리에서만 쓰고, 공유 환경에는 vLLM으로 전환하라.

### 셀프 호스팅 vs 매니지드는 별개의 결정이다

Phase 17 · 01(매니지드 하이퍼스케일러)과 · 02(추론 플랫폼)가 매니지드를 다룬다. 이 레슨은 이미 셀프 호스팅하기로 결정했다고 가정한다. 셀프 호스팅하는 이유는 데이터 거주지(data residency), 커스텀 파인튜닝(fine-tune), 규모를 키웠을 때의 총소유비용(total cost ownership), 호스팅에 없는 도메인 모델이다.

### 기억해 둘 숫자들

- TGI 유지보수 모드: 2025년 12월 11일.
- vLLM v0.15.1: 2026년 2월; PyTorch 2.10; Blackwell SM120 지원.
- SGLang 프로덕션 규모: 400,000개 이상 GPU.
- Ollama 처리량 격차 vs llama.cpp: 15~30% 느림; 프로덕션 부하에서 3배.

```figure
data-parallel
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 결정 트리 워커(decision-tree walker)다: 하드웨어 + 규모 + 워크로드가 주어지면 엔진을 고르고 이유를 설명한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-engine-picker.md`를 만든다. 제약이 주어지면 엔진을 고르고 마이그레이션 계획을 작성한다.

## 연습 문제 (Exercises)

1. 자신의 하드웨어 / 규모 / 워크로드로 `code/main.py`를 실행하라. 출력이 직관과 일치하는가?
2. 인프라가 H100 12장과 MI300X AMD 8장이다. 어느 엔진인가? TRT-LLM은 왜 제외되는가?
3. 한 팀이 "우리가 아는 것이라서" 2026년에도 TGI를 쓰고 싶어 한다. 마이그레이션을 설득할 논거를 세워라.
4. Ollama 개발에서 vLLM 프로덕션으로: 양자화, 설정, 관측성에서 무엇이 바뀌는가?
5. 테넌트 간 P99 접두사 길이가 8K이고 재사용이 높은 RAG 제품. 엔진을 고르고 Phase 17 · 11 + 18과 함께 스택하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| llama.cpp | "그 CPU 엔진" | 가장 넓은 모델 지원, CPU에서 가장 빠름 |
| Ollama | "그 노트북 엔진" | 한 줄 설치, 개발 등급 처리량 |
| TGI | "HF의 서빙" | 2025년 12월부터 유지보수 모드 |
| vLLM | "그 기본값" | 폭넓은 프로덕션 기준선 2026 |
| SGLang | "그 에이전트형 엔진" | 접두사 집약, RadixAttention |
| TRT-LLM | "NVIDIA에 묶임" | Blackwell 처리량 선두, NVIDIA 전용 |
| GGUF | "llama.cpp 형식" | 번들된 K-양자화(K-quant) 변형 |
| Production-stack | "vLLM K8s" | Phase 17 · 18 참조 배포 |
| 파이프라인 패턴 (Pipeline pattern) | "개발→스테이지→프로덕션" | 동일 가중치에서 Ollama → llama.cpp → vLLM |

## 더 읽을거리 (Further Reading)

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
