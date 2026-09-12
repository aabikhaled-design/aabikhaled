# 추론 지표(Inference Metrics): TTFT, TPOT, ITL, 굿풋(Goodput), P99

> 추론(inference) 배포가 제대로 작동하는지는 네 가지 지표로 갈린다. TTFT는 프리필(prefill) + 큐 + 네트워크다. TPOT(ITL과 같다)는 토큰(token)당 메모리 바운드(memory-bound) 디코드(decode) 비용이다. 종단 간(end-to-end) 지연 시간은 TTFT에 TPOT 곱하기 출력 길이를 더한 값이다. 처리량(throughput)은 플릿(fleet) 전반에 걸쳐 집계한 초당 토큰이다. 그러나 제품에서 중요한 지표는 하나, 굿풋(goodput), 곧 모든 SLO를 동시에 충족한 요청의 비율이다. 굿풋이 낮은데 처리량이 높다면, 제때 사용자에게 닿지 못하는 토큰을 처리하고 있다는 뜻이다. 2026년 TRT-LLM에서 Llama-3.1-8B-Instruct의 참조 숫자는 평균 TTFT 162ms, 평균 TPOT 7.33ms, 평균 E2E 1,093ms다. 평균만 보고하지 말고 항상 P50, P90, P99를 보고하라. 그리고 측정 함정을 보라. GenAI-Perf는 ITL 계산에서 TTFT를 제외하고, LLMPerf는 포함한다. 같은 실행을 두고도 두 도구의 TPOT가 갈린다.

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- TTFT, TPOT, ITL, E2E, 처리량, 굿풋을 정확히 정의하고 각각이 측정하는 구성 요소를 명명하기.
- 평균이 왜 LLM 서빙에 잘못된 통계량인지, 그리고 P50/P90/P99를 어떻게 읽는지 설명하기.
- SLO 다중 제약(예: TTFT<500ms AND TPOT<15ms AND E2E<2s)을 구성하고 그에 대해 굿풋을 계산하기.
- 같은 실행에 대해 TPOT를 두고 의견이 갈리는 두 벤치마크 도구를 명명하고 그 이유를 설명하기.

## 문제 (The Problem)

"우리 처리량은 초당 15,000 토큰입니다." 그래서 뭐? 요청의 40%가 종단 간 2초를 넘겼다면, 사용자는 세션을 버렸다. 처리량만으로는 제품이 작동하는지 알려주지 않는다.

추론에는 여러 지연 시간 축이 있고 각각이 다르게 실패한다. 프리필은 연산 바운드(compute-bound)이며 프롬프트 길이에 따라 늘어난다. 디코드는 메모리 바운드이며 배치 크기에 따라 늘어난다. 큐잉 지연은 운영 문제고, 네트워크는 물리적 거리 문제다. 축마다 구별되는 지표가 필요하고, 백분위수가 필요하며, "사용자가 기대한 것을 받았는가"를 말해 주는 단일 종합 지표가 필요하다. 그것이 굿풋이다.

## 개념 (The Concept)

### TTFT: 첫 토큰까지의 시간

`TTFT = queue_time + network_request + prefill_time`

프롬프트가 길면 프리필이 지배한다. H100의 Llama-3.3-70B FP8에서 32k 프롬프트는 순수 프리필만 약 800ms 걸린다. 큐 시간은 부하 상태에서의 스케줄러 동작이고, 네트워크 요청은 TLS를 포함한 와이어 시간이다. TTFT는 무엇이든 스트리밍되어 돌아오기 전에 사용자가 겪는 지연 시간이다.

### TPOT / ITL: 토큰 간 지연 시간

하나의 양을 여러 이름으로 부른다. `TPOT`(time per output token, 출력 토큰당 시간), `ITL`(inter-token latency, 토큰 간 지연 시간), `토큰당 디코드 지연 시간`은 모두 같다. 첫 번째 이후 연속 스트리밍 토큰 사이의 시간이다.

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

청크 프리필(chunked prefill)을 갖춘 같은 Llama-3.3-70B H100 스택에서 TPOT는 평균 약 7ms다. 청크 프리필이 없으면 이웃 시퀀스의 긴 프리필 동안 TPOT가 50ms까지 치솟기도 한다. 평균이 아니라 P99를 보라.

### E2E 지연 시간

`E2E = TTFT + TPOT * output_tokens + network_response`

출력이 길면(>500 토큰) E2E는 TPOT가 지배하고, 프롬프트가 길고 출력이 짧으면 TTFT가 지배한다. 출력 길이로 조건화한 E2E를 보고하라.

### 처리량

`throughput = total_output_tokens / elapsed_time`

집계 지표다. 플릿 효율은 알려주지만 개별 요청의 건강은 알려주지 않는다.

### 굿풋: 실제로 신경 쓰는 지표

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO는 다중 제약이다. 요청은 모든 제약이 유지되었을 때에만 "좋은" 것이고, 굿풋은 그 비율이다. 굿풋 60%에서 높은 처리량은 실패이며, 굿풋 99%에서 더 낮은 처리량이 목표다.

2026년에, 굿풋은 MLPerf Inference v6.0 제출과 AI 플랫폼 제공자의 내부 SLA 추적에 사용되는 지표다.

### 평균이 잘못된 통계량인 이유

LLM 지연 시간 분포는 오른쪽으로 치우쳐 있다. 긴 프리필 이웃이 하나 낀 디코드 배치는 TPOT 약 7ms로 500 토큰을 출하하다가 TPOT 약 60ms로 20 토큰을 출하하기도 한다. 평균 TPOT는 9ms지만 P99 TPOT는 65ms다. 사용자는 P99에 정기적으로 부딪히고, 그래서 떠난다.

항상 세 쌍(P50, P90, P99)을 보고하라. 사용자 경험을 위해 최적화할 대상은 P99다.

### 참조 숫자. TRT-LLM의 Llama-3.1-8B-Instruct, 2026

- 평균 TTFT: 162ms
- 평균 TPOT: 7.33ms
- 평균 E2E: 1,093ms
- P99 TPOT: 청크 프리필 구성에 따라 10~25ms 변동.

이것들은 공개된 NVIDIA 참조 지점이다. 모델 크기(70B는 3~5배로 나타난다), 하드웨어(H100 대 B200 약 3배), 부하에 따라 바뀐다.

### 측정 함정

2026년에 가장 많이 쓰이는 벤치마크 도구 둘은 같은 실행을 두고도 TPOT가 갈린다.

- **NVIDIA GenAI-Perf**: ITL 계산에서 TTFT를 제외한다. ITL은 토큰 2부터 시작한다.
- **LLMPerf**: TTFT를 포함한다. ITL은 토큰 1부터 시작한다.

TTFT가 500ms이고 100개 출력 토큰을 총 700ms에 디코드한 요청이라면, GenAI-Perf는 `ITL = 700/99 = 7.07 ms`로, LLMPerf는 `ITL = 1200/100 = 12.00 ms`로 보고한다. 어느 도구를 쓰느냐가 숫자를 바꾼다.

항상 어느 도구인지 명시하라. 항상 정의를 공표하라.

### SLO 구성하기

2026년 70B 채팅 모델을 위한 합리적인 소비자 대면 SLO:

- TTFT P99 <= 800ms.
- TPOT P99 <= 25ms.
- <300 토큰 출력에 대해 E2E P99 <= 3s.
- 굿풋 목표 >= 99%.

엔터프라이즈 SLO는 TTFT를 조이고(200~400ms) E2E를 느슨하게 한다. 핵심은 이를 적어 두고, 셋 다 측정하고, 굿풋을 단일 종합 지표로 추적하는 것이다.

### 측정하는 법

- 실제 트래픽이나 현실적 합성(`--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`을 가진 LLMPerf)을 돌려라.
- 벤치마크 실행에 대해 피크 동시성의 2배를 목표로 하라.
- 30~50회 반복하고, 결합된 샘플의 백분위수를 취하라.
- 도구 이름, 도구 버전, 모델, 하드웨어, 동시성, 프롬프트 분포와 함께 공표하라.

```figure
throughput-latency
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 굿풋 계산기다. 합성 지연 시간 분포를 생성하고, SLO를 적용하고, 굿풋을 계산하라. 같은 트레이스에서 GenAI-Perf와 LLMPerf의 TPOT 차이도 보여준다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-slo-goodput-gate.md`를 만들어낸다. 워크로드와 SLO를 주면, 처리량이 아니라 굿풋으로 배포를 게이트하는 CI/CD 준비된 벤치마크 레시피를 만들어낸다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 1% 꼬리 스파이크를 가진 분포를 생성하라. P99 TPOT를 30ms에서 15ms로 조일 때 굿풋이 어떻게 바뀌는가?
2. 한 벤더가 "Llama 3.3 70B H100에서 초당 15,000 토큰"을 인용한다. 이를 믿기 전에 물어볼 세 질문을 명명하라.
3. 청크 프리필이 왜 P99 TPOT는 보호하지만 평균 TPOT는 보호하지 않는가?
4. 음성 어시스턴트(첫 토큰을 읽는 게 아니라 듣는다)를 위한 소비자 SLO를 구성하라. 어느 지표가 가장 사용자에게 보이는가?
5. LLMPerf README와 GenAI-Perf 문서를 읽어라. 도구들이 의견이 갈리는 다른 세 지표를 식별하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| TTFT | "첫 토큰까지의 시간" | 큐 + 네트워크 + 프리필. 긴 프롬프트에서는 프리필이 지배 |
| TPOT | "출력 토큰당 시간" | 첫 번째 이후 토큰당 메모리 바운드 디코드 비용 |
| ITL | "토큰 간 지연 시간" | 대부분 도구에서 TPOT와 같음(전부는 아님: GenAI-Perf 참조) |
| E2E | "종단 간" | TTFT + TPOT * output_len. 위에 응답 측 네트워크 |
| Throughput | "토큰/초" | 플릿 효율. 지연 시간 백분위수 없이는 쓸모없음 |
| Goodput | "SLO 충족률" | 모든 SLO 제약을 동시에 충족하는 요청의 비율 |
| P99 | "꼬리" | 100분의 1 최악 지연 시간. 사용자 경험 지표 |
| SLO multi-constraint | "결합" | 세 지연 시간 경계 모두의 AND. 하나라도 위반되면 요청 실패 |
| GenAI-Perf vs LLMPerf | "도구 함정" | ITL이 TTFT를 포함하는지를 두고 도구들이 의견이 갈림 |

## 더 읽을거리 (Further Reading)

- [NVIDIA NIM: LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) TTFT, ITL, TPOT의 정전(canonical) 정의.
- [Anyscale: LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) 대안적 정의와 측정 레시피.
- [BentoML: LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) 실제 배포에서의 응용 측정.
- [LLMPerf](https://github.com/ray-project/llmperf): Ray 기반 오픈소스 벤치마크.
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html): NVIDIA의 벤치마크 도구.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/): 업계에서 인정받는 굿풋 기반 벤치마크.
