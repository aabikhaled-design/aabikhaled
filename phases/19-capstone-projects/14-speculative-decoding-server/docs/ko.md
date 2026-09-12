# Capstone 14: 추측 디코딩 추론 서버 (Speculative-Decoding Inference Server)

> vLLM 0.7의 EAGLE-3은 실제 트래픽에서 2.5-3배 처리량(throughput)을 낸다. P-EAGLE(AWS 2026)은 병렬 추측(speculation)을 한층 더 밀어붙였다. SGLang의 SpecForge는 드래프트 헤드(draft head)를 대규모로 학습시켰다. Red Hat의 Speculators 허브는 흔한 오픈 모델용 정렬된(aligned) 드래프트를 공개했다. TensorRT-LLM은 NVIDIA에서 추측 디코딩(speculative decoding)을 일급(first-class)으로 만들었다. 2026 프로덕션(production) 서빙 스택은 EAGLE 계열 드래프트, FP8 또는 INT4 양자화(quantization), 그리고 큐 대기(queue-wait)에 대한 HPA를 갖춘 vLLM 또는 SGLang이다. 이 캡스톤(capstone)은 두 오픈 모델을 베이스라인(baseline) 대비 2.5배 이상의 처리량으로 서빙하고 완전한 꼬리 지연 시간(tail-latency) 보고서를 내는 것이다.

**Type:** Capstone
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Prerequisites:** Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 17 (infrastructure)
**Phases exercised:** P3 · P7 · P10 · P17
**Time:** 30시간

## 문제 (Problem)

추측 디코딩은 2026년에 일상재(commodity)가 되었다. EAGLE-3 드래프트 헤드는 타깃(target) 모델의 은닉 상태(hidden state)로 학습하고 N개 토큰을 앞서 예측한다. 타깃 모델은 단일 패스로 검증한다. 60-80%의 수용률(acceptance rate)은 종단 간(end-to-end) 처리량 2-3배로 환산된다. vLLM 0.7은 이것을 네이티브로 통합한다. SGLang + SpecForge는 학습 파이프라인을 제공한다. Red Hat의 Speculators는 Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B용 정렬된 드래프트를 공개한다.

기술은 모델이 아니라 서빙 운영에 있다. 수용률은 트래픽 분포(ShareGPT 대 코드 대 도메인 데이터)에 따라 드리프트(drift)한다. 거부 시 꼬리 지연 시간은 추측이 없을 때보다 더 나쁘다. 정상 상태(steady-state) 초당 토큰만이 아니라 여러 배치(batch) 크기에서의 p99를 보고해야 한다. Anthropic / OpenAI API 대비 100만 토큰당 비용이 신뢰성의 지렛대다.

## 개념 (Concept)

추측 디코딩에는 두 계층이 있다. **드래프트** 모델(EAGLE-3 헤드, ngram, 또는 더 작은 타깃 정렬 모델)이 스텝마다 k개의 후보 토큰을 제안한다. **타깃** 모델은 k개 모두를 한 번의 패스로 검증한다. 수용된 접두(prefix)는 탐욕(greedy) 경로를 대체한다. 수용률은 드래프트-타깃 정렬과 입력 분포에 달려 있다.

EAGLE-3은 대부분의 트래픽에서 ngram 드래프트를 능가한다. P-EAGLE은 더 깊은 드래프트 트리(draft tree)를 위해 병렬 추측을 실행한다. 트레이드오프(trade-off): 검증 패스가 더 크기 때문에 거부 시 P99 지연 시간이 더 높다. 서빙 설정은 이것을 드러내기 위해 배치 크기별로 버킷화된(batch-size-bucketed) 지연 시간을 보고해야 한다.

배포는 쿠버네티스(Kubernetes)다. vLLM 0.7은 GPU당 또는 텐서 병렬(tensor-parallel) 샤드당 레플리카 하나를 실행한다. HPA는 CPU가 아니라 큐 대기로 오토스케일한다. FP8(Marlin)과 INT4(AWQ) 양자화는 GPU 메모리를 H100 / H200 한도 안에 유지한다. 종단 간 보고서는 처리량, 수용률, 배치 1/8/32에서의 p50/p99, 그리고 100만 토큰당 비용($/1M tokens)이다.

## 아키텍처 (Architecture)

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 스택 (Stack)

- 서빙: vLLM 0.7 또는 SGLang 0.4
- 추측 방법: EAGLE-3 드래프트 헤드, P-EAGLE 병렬 추측, ngram 폴백(fallback)
- 드래프트 학습: SpecForge (SGLang) 또는 Red Hat Speculators
- 타깃 모델: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B
- 양자화: FP8 (Marlin), INT4 AWQ
- 배포: Kubernetes + NVIDIA device plugin; 큐 대기 메트릭에 대한 HPA
- 평가: 도메인 분포별 수용률 측정을 위한 ShareGPT, MT-Bench-v2, GSM8K, HumanEval
- 레퍼런스: 벤더 베이스라인을 위한 TensorRT-LLM 추측 디코딩

```figure
cf-spec-decode
```

## 직접 만들기 (Build It)

1. **타깃 모델 준비.** Llama 3.3 70B를 고른다. Marlin으로 FP8 양자화한다. 1xH100(또는 2x 텐서 병렬)에서 vLLM 0.7로 배포한다.

2. **드래프트 소스.** Red Hat Speculators에서 정렬된 EAGLE-3 드래프트 헤드를 가져온다(또는 SpecForge로 직접 학습한다). vLLM의 추측 디코딩 설정에 로드한다.

3. **베이스라인 수치.** 추측 전: 배치 1/8/32에서 초당 토큰, p50/p99 지연 시간, GPU 사용률. 공개한다.

4. **EAGLE-3 활성화.** 설정을 전환한다. 동일한 벤치마크를 다시 실행한다. 속도 향상, 수용률, p99 꼬리 지연 시간 차이를 보고한다.

5. **P-EAGLE.** 병렬 추측을 활성화한다. 직렬 EAGLE-3 대비 더 깊은 드래프트 트리를 측정한다. P-EAGLE이 도움이 되는 지점과 해가 되는 지점의 변곡점을 보고한다.

6. **도메인 트래픽.** 동일 서버에 ShareGPT 대 HumanEval 대 도메인 특화 트래픽을 흘려보낸다. 분포별 수용률을 측정한다. 드래프트가 언제 드리프트하는지 식별한다.

7. **두 번째 타깃 모델.** 동일한 파이프라인을 Qwen3-Coder-30B MoE에 실행한다. 드래프트가 더 까다롭다(MoE 라우팅 노이즈). 보고한다.

8. **K8s HPA.** `queue_wait_ms`를 추적하는 HPA와 함께 K8s로 배포한다. 부하가 3배가 될 때 스케일 아웃을 시연한다.

9. **비용 비교.** 동일 평가에서 Anthropic Claude Sonnet 4.7 및 OpenAI GPT-5.4 대비 100만 토큰당 비용을 계산한다. 공개한다.

## 라이브러리로 써보기 (Use It)

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 산출물 (Ship It)

`outputs/skill-inference-server.md`가 결과물을 설명한다. 추측 디코딩, 완전한 벤치마크 보고서, K8s 배포를 갖춘 측정된 서빙 스택.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | 베이스라인 대비 측정된 속도 향상 | 두 모델에서 동등 품질에 2.5배 이상 처리량 |
| 20 | 현실적 트래픽에서의 수용률 | 분포별 수용률 보고서 |
| 20 | P99 꼬리 지연 시간 규율 | 추측 유무에 따른 배치 1/8/32에서의 p99 |
| 20 | 운영 | K8s 배포, 큐 대기에 대한 HPA, 매끄러운 롤아웃 |
| 15 | 작성 및 방법론 | 무엇이 왜 바뀌었는지에 대한 명확한 설명 |
| **100** | | |

## 연습 문제 (Exercises)

1. 드래프트가 타깃보다 한 버전 뒤처졌을 때(예: Llama 3.3 -> 3.4 드리프트) 수용률 저하를 측정한다. 모니터링 경보를 만든다.

2. ngram 폴백을 구현한다: EAGLE-3 수용률이 임계값 아래로 떨어지면 ngram 드래프트로 전환한다. 신뢰성 개선을 보고한다.

3. 통제된 MoE 실험을 실행한다: 라우팅 노이즈가 주입된 동일 Qwen3-Coder-30B 대 주입되지 않은 것. 드래프트 수용 민감도를 측정한다.

4. H200(141 GB)으로 확장한다. 확보된 레플리카당 모델 크기 여유를 보고하고, 양자화되지 않은 Llama 3.3 70B를 서빙할 수 있는지 보고한다.

5. 동일한 H100 하드웨어에서 TensorRT-LLM 추측 디코딩을 벤치마크한다. vLLM 대비 어디서 이기는지 보고한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| Draft model | "추측기(speculator)" | 타깃이 검증하도록 N개 토큰을 제안하는 작은 모델 |
| EAGLE-3 | "2026 드래프트 아키텍처" | 타깃 은닉 상태로 학습한 드래프트 헤드; ~75% 수용 |
| P-EAGLE | "병렬 추측" | 한 번의 타깃 패스로 검증되는 드래프트 분기 트리 |
| Acceptance rate | "적중률" | 재샘플링 없이 수용된 드래프트 토큰의 비율 |
| Quantization | "FP8 / INT4" | GPU 메모리에 더 많은 모델을 담기 위한 저정밀 가중치 |
| Queue wait | "HPA 메트릭" | 추론이 시작되기 전 요청이 대기 큐에서 기다리는 시간 |
| Speculators hub | "정렬된 드래프트" | 흔한 오픈 모델용 EAGLE 드래프트의 Red Hat Neural Magic 허브 |

## 더 읽을거리 (Further Reading)

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai): 레퍼런스 서빙 스택
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/): 병렬 추측 디코딩 논문 + 통합
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge): 드래프트 헤드 학습 파이프라인
- [Red Hat Speculators](https://github.com/neuralmagic/speculators): 정렬된 드래프트 허브
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/): 벤더 대안
- [Fireworks.ai serving architecture](https://fireworks.ai/blog): 상용 레퍼런스
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840): 방법 논문
- [vLLM repository](https://github.com/vllm-project/vllm): 코드와 벤치마크
