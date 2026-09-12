# 프로덕션에서의 EAGLE-3 추측 디코딩(Speculative Decoding)

> 추측 디코딩(speculative decoding)은 빠른 드래프트 모델(draft model)을 타깃 모델(target model)과 짝짓는다. 드래프트가 K개 토큰(token)을 제안하면, 타깃은 단일 순방향에서 검증한다. 수용된 토큰은 공짜다. 2026년 EAGLE-3은 프로덕션급(production-grade) 변형이다. 원시 토큰이 아니라 타깃 모델의 은닉 상태(hidden state) 위에서 드래프트 헤드를 학습시켜, 일반 채팅에서 수용률 alpha를 0.6~0.8 구간으로 밀어올린다. 올바른 질문은 "드래프트가 얼마나 빠른가"가 아니라 "내 트래픽에서 alpha가 얼마인가"이다. alpha가 약 0.55 아래로 떨어지면, 거부된 모든 드래프트가 두 번째 타깃 순방향 패스(forward pass) 비용을 치르기 때문에 추측 디코딩은 높은 동시성에서 순(net) 음수가 된다. 이 레슨은 먼저 alpha를 측정하고 나중에 플래그를 켜는 법을 가르친다.

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 추측 디코딩의 세 세대를 명명하고 EAGLE-3이 EAGLE-2와 고전적 드래프트 모델로부터 무엇을 바꾸는지 설명하기.
- 수용률 alpha를 정의하고, alpha와 K(드래프트 길이)로부터 기대 속도 향상을 계산하며, 타깃 동시성에 대한 손익분기 alpha를 식별하기.
- vLLM 2026에서 추측 디코딩이 왜 기본값이 아니라 옵트인(opt-in)인지, 그리고 alpha를 측정하지 않고 켜는 것이 왜 프로덕션 안티패턴인지 설명하기.
- 측정 계획을 작성하기: 어느 벤치마크, 어느 프롬프트 분포, 어느 동시성 지점, 어느 지표로 게이트할지.

## 문제 (The Problem)

디코드(decode)는 메모리 바운드(memory-bound)다. Llama 3.3 70B FP8을 돌리는 H100에서, 디코드된 각 토큰은 약 140GB/s의 가중치를 읽고 토큰 하나를 방출한다. 디코드 동안 GPU 연산은 거의 유휴 상태다. 병목은 행렬곱(matmul) 처리량이 아니라 HBM 대역폭이다.

추측 디코딩은 그 간극을 이용한다. 값싼 드래프트 모델로 K개 후보 토큰을 생성한 뒤, 타깃 모델에게 단일 순방향 패스에서 K개 전부를 검증하도록 요청한다. 검증된 각 토큰은 사실상 공짜다(타깃이 어차피 해야 했을 K-배치 순방향에 분할상환된다).

고전적 드래프트 모델 접근법은 같은 패밀리의 더 작은 모델(Llama 3.3 70B를 위해 Llama 3.2 1B가 드래프트)을 쓴다. 작동하지만 수용률은 평범하다. 더 작은 모델의 분포가 타깃에서 발산한다. EAGLE, 다음 EAGLE-2, 다음 EAGLE-3은 가벼운 드래프트 헤드를 타깃 모델의 내부 상태 위에서 직접 학습시켜, 드래프트의 분포가 타깃을 훨씬 더 가깝게 추적한다. 이것이 alpha가 드래프트 모델에서 0.4였던 것이 EAGLE-3에서 0.6~0.8로 가는 이유다.

함정: EAGLE-3은 vLLM 2026에서 옵트인이다. `speculative_config`를 명시적으로 설정해야 한다. 플래그 없으면 가속 없다. 실제 트래픽에서 alpha를 측정하지 않고 켜는 팀은 꼬리 지연 시간(tail latency)이 더 좋아지기는커녕 오히려 나빠지는 경우를 흔히 겪는다.

## 개념 (The Concept)

### 추측 디코딩이 실제로 사주는 것

추측 디코딩 없이는 토큰당 비용이 타깃 순방향 하나다. 드래프트 길이 K와 수용률 alpha의 추측 디코딩에서는, 타깃 순방향당 기대 토큰이 `1 + K * alpha`다. 속도 향상은 `(1 + K * alpha) / (1 + epsilon)`이며 여기서 epsilon은 드래프트 + 검증 오버헤드다. K=5, alpha=0.7의 경우: `(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`. 실제 수치는 약 2~3x 부근에 모이는데, alpha가 프로덕션 트래픽에서 그렇게 높은 경우는 드물고 높은 배치 크기에서 epsilon이 커지기 때문이다.

### alpha가 유일하게 중요한 지표인 이유

거부된 토큰은 사라지지 않는다. 첫 거부된 토큰에 대해 두 번째 타깃 순방향을 강제한다. alpha가 0.4로 떨어지는 워크로드에서는, 드래프트 오버헤드에 더해 검증과 리롤(re-roll)을 치른다. 높은 동시성(가령 동시 256)에서는 디코드 배치(batch)가 이미 충분히 커서 "타깃 단독"과 "검증 포함 타깃" 사이의 메모리 대역폭 간극이 줄어든다. 대부분의 2026년 하드웨어에서 alpha 0.55 아래에서는 추측 디코딩이 순 음수다.

alpha는 워크로드에 따라 다르다. ShareGPT 스타일 일반 채팅에서, ShareGPT로 학습된 EAGLE-3은 0.6~0.8을 친다. 도메인 특화 트래픽(코드, 의료, 법률)에서는 일반 데이터로 학습된 드래프트 헤드가 0.4~0.6으로 떨어진다. 도메인 특화 드래프트 헤드를 학습시키면 alpha가 회복된다. 그것은 타깃 파인튜닝(finetuning)에 비해 가볍고 빠른 학습 작업이다.

### EAGLE 세대 한눈에 보기

- **고전적 드래프트 모델**: 같은 패밀리의 작은 모델. Alpha 0.3~0.5. 인프라가 단순함: 두 모델 로드, 드래프트가 타깃 순방향당 K개 순방향을 돌림.
- **EAGLE-1 (2024)**: 타깃 은닉 상태(마지막 층) 위에서 학습된 단일 드래프트 헤드. Alpha 약 0.5~0.6. 타깃 위에 작은 파라미터 오버헤드.
- **EAGLE-2 (2025)**: 적응적 드래프트 길이와 트리 기반 드래프트(하나의 타깃 패스에서 여러 분기 검증). Alpha 약 0.6~0.7. 더 복잡한 드래프트 스케줄러.
- **EAGLE-3 (2025-2026)**: 여러 타깃 층(마지막만이 아니라) 위에서 학습된 드래프트 헤드, 더 나은 정렬. 일반 채팅에서 Alpha 약 0.6~0.8.

### 2026년 프로덕션 레시피

1. 타깃 모델을 그대로 출하한다. 타깃 동시성에서 베이스라인(baseline) TTFT, ITL, 처리량을 측정한다.
2. vLLM `speculative_config`를 통해 EAGLE-3 드래프트를 활성화한다. 벤치마크를 다시 돌린다.
3. 수용률 alpha를 로깅한다. vLLM V1은 이를 `spec_decode_metrics.accepted_tokens_per_request`로 보고한다. 요청한 드래프트 길이로 나누어 alpha를 얻는다.
4. 프로덕션 트래픽 분포에서 alpha < 0.55라면, 추측 디코딩을 끄거나 도메인 특화 EAGLE-3 드래프트를 학습시킨다.
5. 프로덕션 동시성에서 다시 돌린다. P99 ITL이 나빠지지 않았는지 확인한다.

### 프로덕션 함정: P99 꼬리

평균 ITL은 추측 디코딩으로 떨어진다. 튜닝하지 않으면 P99는 나빠질 수 있다. 거부된 드래프트는 두 패스 시퀀스(드래프트 + 검증 실패 + 리롤)를 트리거한다. 배치가 가득 찬 상태에서는 그 두 패스가 직렬화된다. P50이 아니라 P99 ITL을 보라.

### EAGLE-3이 이미 배포된 곳

Google은 2025년 AI Overviews에 추측 디코딩을 배포했다(같은 품질, 더 빠른 응답). vLLM V1은 `speculative_config`를 문서화된 인터페이스로 출하한다. V1의 N-gram GPU 추측 디코딩은 청크 프리필(chunked prefill)과 호환되는 변형이다. SGLang은 접두사 중심(prefix-heavy) 워크로드를 위한 권장 드래프트 경로로 EAGLE-3을 지원한다.

### 한 줄로 된 손익분기 수학

기대 속도 향상: `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`. `S = 1`로 설정하면 alpha에 대해 풀린다: `alpha_breakeven = verify_overhead / K`. 일반적인 verify_overhead 약 0.15와 K=5의 경우: `alpha_breakeven = 0.03`. 다만 이는 순수 디코드 수학이다. 높은 동시성에서는 검증 오버헤드가 오르고 디코드 배치가 이미 시퀀스 전반에 메모리 읽기를 분할상환하므로, 실효 alpha_breakeven은 실제로 약 0.45~0.55로 오른다.

### 추측 디코딩을 쓰지 말아야 할 때

- 지연 시간이 중요하지 않은 배치-1 오프라인 생성. 그냥 타깃을 쓰라.
- 매우 짧은 출력(50 토큰 미만). 드래프트 오버헤드와 검증 비용이 지배한다.
- 도메인 학습된 드래프트 헤드가 없는 특수 도메인. Alpha가 너무 낮다.
- vLLM v0.18.0 + 드래프트 모델 추측 디코딩 + `--enable-chunked-prefill`. 이 조합은 컴파일되지 않는다. 문서화된 예외는 V1의 N-gram GPU 추측 디코딩이다.

```figure
mx-speculative-tree
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 다양한 alpha 값과 드래프트 길이 K에 걸쳐 추측 디코딩이 있을 때와 없을 때의 디코드 루프를 시뮬레이션한다. 손익분기 alpha, 측정된 속도 향상, 꼬리 동작을 출력한다. 여러 (alpha, K) 조합에서 실행하여 추측 디코딩이 정확히 어디서 더 이상 이득이 되지 않는지 보라.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-eagle3-rollout.md`를 만들어낸다. 타깃 모델, 트래픽 분포 설명, 동시성 목표가 주어지면, 단계적 EAGLE-3 롤아웃(rollout) 계획을 만들어낸다. 베이스라인 벤치마크, 설정 활성화, alpha 측정, alpha >= 0.55로 게이트, P99 ITL 감시.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. K=5에서, 2배 속도 향상에 어떤 alpha가 필요한가? 3배 속도 향상에는? 그것이 verify_overhead에 얼마나 민감한가?
2. 프로덕션 트래픽이 일반 채팅 70%, 코드 30%로 나뉜다고 상상하라. 일반 채팅은 ShareGPT로 학습된 EAGLE-3으로 alpha 0.7을 치고, 코드는 alpha 0.4를 친다. 혼합 alpha는 얼마이며 추측 디코딩은 순 양수인가?
3. vLLM `speculative_config` 문서를 읽어라. 세 모드(드래프트 모델, EAGLE, N-gram)를 명명하고 어느 것이 청크 프리필과 호환되는지 말하라.
4. EAGLE-3을 활성화한 후 평균 ITL이 25% 떨어지는데 P99 ITL이 15% 올라가는 것을 본다. 진단하고 완화책을 제안하라.
5. Llama 3.3 70B에 대한 EAGLE-3 드래프트 헤드의 메모리 비용을 계산하라. 고전적 드래프트로 Llama 3.2 1B를 돌리는 것과 어떻게 비교되는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| Speculative decoding | "드래프트 + 검증" | 값싼 모델로 K개 토큰을 제안하고, 하나의 타깃 순방향에서 K개 전부 검증 |
| Acceptance rate alpha | "추측 수용률" | 타깃이 수용한 드래프트 토큰의 비율. 유일하게 중요한 지표 |
| Draft length K | "추측 k" | 드래프트가 타깃 순방향당 제안하는 토큰 수. 일반적으로 4~8 |
| Verify overhead epsilon | "추측 오버헤드" | 그냥 타깃 순방향 대비 검증 및 리롤의 추가 비용. 배치와 함께 커짐 |
| EAGLE-3 | "최신 EAGLE" | 2025-2026 변형. 여러 타깃 층 위에서 드래프트 헤드 학습. 일반 채팅에서 alpha 0.6~0.8 |
| `speculative_config` | "vLLM 추측 설정" | vLLM V1의 명시적 옵트인. 기본값 없음은 가속 없음을 의미 |
| N-gram spec decode | "N-gram 드래프트" | 프롬프트 내 N-gram 조회를 쓰는 GPU 측 드래프트. 청크 프리필 호환 |
| Break-even alpha | "무효 alpha" | 추측 디코딩이 영(zero) 속도 향상을 주는 alpha. 프로덕션 동시성에서 이것을 보라 |
| Rejected-draft two-pass | "리롤 비용" | 드래프트가 거부될 때의 두 타깃 순방향. P99 꼬리를 견인 |

## 더 읽을거리 (Further Reading)

- [vLLM: Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) V1의 `speculative_config`와 청크 프리필 호환성에 관한 권위 있는 출처.
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/): 정확한 필드 집합.
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077): 원본 EAGLE 드래프트 헤드 정식화.
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858): 적응적 드래프트와 트리.
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html): 추측 디코딩을 활용한 효율적 LLM 시스템.
- [BentoML: Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) 프로덕션 롤아웃 체크리스트.
