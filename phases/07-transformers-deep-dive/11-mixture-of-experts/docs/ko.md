# 전문가 혼합(Mixture of Experts, MoE)

> 밀집(dense) 70B 트랜스포머(transformer)는 모든 토큰(token)에 대해 모든 파라미터(parameter)를 활성화한다. 671B MoE는 토큰당 37B만 활성화하면서 모든 벤치마크(benchmark)에서 그것을 이긴다. 희소성(sparsity)은 이 시대의 가장 중요한 스케일링 아이디어다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45분

## 문제 (The Problem)

밀집 트랜스포머의 추론(inference) 시 FLOPs는 파라미터 수와 같다(순방향 패스(forward pass)의 경우 2배). 밀집 모델(model)을 키우면 모든 토큰이 전체 비용을 치른다. 2024년에 이르러 프런티어(frontier)는 연산의 벽에 부딪혔다. 의미 있게 더 똑똑해지려면 토큰당 기하급수적으로 더 많은 FLOPs가 필요했다.

전문가 혼합은 이 연결을 끊는다. 각 FFN을 `E`개의 독립적인 전문가(expert) + 토큰마다 `k`개의 전문가를 고르는 라우터(router)로 교체한다. 전체 파라미터 = `E × FFN_size`. 토큰당 활성 파라미터 = `k × FFN_size`. 2026년의 일반적인 구성: `E=256`, `k=8`. 저장(storage)은 `E`에 비례해 늘고, 연산은 `k`에 비례해 늘어난다.

2026년 프런티어는 거의 전부 MoE다: DeepSeek-V3(전체 671B / 활성 37B), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. Artificial Analysis의 독립 리더보드에서 상위 10개 오픈소스 모델이 모두 MoE다.

## 개념 (The Concept)

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### FFN 교체

밀집 트랜스포머 블록:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 블록:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

모든 전문가는 독립적인 FFN(보통 SwiGLU)이다. 라우터는 단일 선형 층(linear layer)이다. 각 토큰은 자신의 `k`개 전문가를 고르고, 그 출력의 게이팅된(gated) 혼합을 얻는다.

### 부하 분산(load-balancing) 문제

라우터가 토큰의 90%를 전문가 3번으로 보내면, 나머지 전문가들은 굶주린다. 세 가지 해법이 시도되었다:

1. **보조 부하 분산 손실(auxiliary load-balancing loss)** (Switch Transformer, Mixtral). 전문가 사용량의 분산에 비례하는 페널티를 더한다. 작동하지만, 하이퍼파라미터(hyperparameter)와 두 번째 그래디언트(gradient) 신호를 추가한다.
2. **전문가 용량(expert capacity) + 토큰 드롭(token dropping)** (초기 Switch). 각 전문가는 최대 `C × N/E`개의 토큰을 처리하고, 넘치는 토큰은 그 층을 건너뛴다. 품질을 해친다.
3. **보조 손실 없는 분산(auxiliary-loss-free balancing)** (DeepSeek-V3). 라우터의 top-k 선택을 이동시키는, 전문가별 학습된 편향(bias)을 더한다. 편향은 학습 손실(loss) 바깥에서 갱신된다. 주 목적함수에 페널티가 없다. 2024년의 큰 돌파구다.

DeepSeek-V3의 접근: 매 학습 스텝 이후, 모든 전문가에 대해 사용량이 목표보다 높은지 낮은지 확인한다. 편향을 `±γ`만큼 미세 조정한다. 선택은 `scores + bias`를 사용한다. 게이팅에 쓰이는 전문가 확률은 원래의 `scores` 그대로다. 라우팅과 표현을 분리한다.

### 공유 전문가(shared experts)

DeepSeek-V2/V3는 또한 전문가를 *공유(shared)*와 *라우티드(routed)*로 나눈다. 모든 토큰은 모든 공유 전문가를 통과한다. 라우티드 전문가는 top-k로 선택된다. 공유 전문가는 공통 지식을 포착하고, 라우티드 전문가는 특화된다. V3는 공유 전문가 1개에 256개 라우티드 중 top-8을 실행한다.

### 세분화된 전문가(fine-grained experts)

고전적 MoE(GShard, Switch): 각 전문가는 완전한 FFN만큼 넓다. `E`가 작고(8~64), `k`가 작다(1~2).

현대적 세분화 MoE(DeepSeek-V3, Qwen-MoE): 각 전문가가 더 좁다(FFN 크기의 1/8). `E`가 크고(256+), `k`가 더 크다(8+). 같은 전체 파라미터지만, 조합은 훨씬 빠르게 늘어난다. 토큰당 가능한 "전문가" 조합은 `C(256, 8) = 400조`다. 품질은 올라가고, 지연 시간(latency)은 그대로 유지된다.

### 비용 프로파일

토큰당, 층당:

| 구성 | 토큰당 활성 파라미터 | 전체 파라미터 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (밀집) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3는 **토큰당 더 적은 활성 FLOPs**를 쓰면서도 거의 모든 벤치마크에서 Llama 3 70B(밀집)를 이긴다. 더 많은 파라미터 = 더 많은 지식. 더 많은 활성 FLOPs = 토큰당 더 많은 연산. MoE는 이 둘을 분리한다.

### 함정: 메모리

어떤 전문가가 작동하든 모든 전문가는 GPU에 상주한다. 671B 모델은 fp16 가중치(weight)에 ~1.3 TB의 VRAM이 필요하다. 프런티어 MoE 배포(deployment)에는 전문가 병렬화(expert parallelism)가 필요하다. 전문가를 GPU에 걸쳐 샤딩(shard)하고 토큰을 네트워크에 걸쳐 라우팅한다. 지연 시간은 행렬곱(matmul)이 아니라 all-to-all 통신이 지배한다.

```figure
expert-routing
```

## 직접 만들기 (Build It)

`code/main.py`를 참고하라. 다음을 갖춘, 순수 표준 라이브러리(stdlib)로 만든 간결한 MoE 층:

- `n_experts=8`개의 SwiGLU 스타일 전문가(설명을 위해 각각 선형 하나)
- top-k=2 라우팅
- softmax로 정규화된(normalized) 게이팅 가중치
- 전문가별 편향을 통한 보조 손실 없는 분산

### 1단계: 라우터

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

편향은 선택에는 영향을 주지만 게이트 가중치에는 영향을 주지 않는다. 이것이 DeepSeek-V3의 트릭이다. 편향은 모델의 예측을 조종하지 않으면서 부하 불균형을 교정한다.

### 2단계: 라우터에 100개 토큰 통과시키기

어떤 전문가가 얼마나 자주 작동하는지 추적한다. 편향이 없으면 사용량이 치우친다. 편향 갱신 루프(과사용 전문가에 `-γ`, 저사용 전문가에 `+γ`)를 두면 사용량은 몇 번의 반복 만에 균등 분포로 수렴(convergence)한다.

### 3단계: 파라미터 수 비교

MoE 구성의 "밀집 등가(dense equivalent)"를 출력한다. DeepSeek-V3 형태: 라우티드 256 + 공유 1, 활성 8, d_model=7168. 전체 파라미터 수는 눈이 휘둥그레질 정도다. 활성 수는 밀집 Llama 3 70B의 7분의 1이다.

## 라이브러리로 써보기 (Use It)

HuggingFace 로딩:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026년 프로덕션(production) 추론: vLLM은 MoE 라우팅을 네이티브로 지원한다. SGLang은 가장 빠른 전문가 병렬 경로를 가지고 있다. 둘 다 top-k 선택과 전문가 병렬화를 자동으로 처리한다.

**MoE를 선택할 때:**
- 토큰당 더 낮은 추론 비용으로 프런티어 품질을 원한다.
- VRAM / 전문가 병렬 인프라를 갖추고 있다.
- 워크로드가 컨텍스트가 무거운(긴 문서) 것이 아니라 토큰이 무거운(채팅, 코드) 것이다.

**MoE를 선택하지 말아야 할 때:**
- 엣지 배포: 어떤 활성 FLOP에 대해서든 전체 저장 비용을 치른다.
- 지연 시간이 중요한 단일 사용자 서빙(serving): 전문가 라우팅이 오버헤드를 더한다.
- 작은 모델(<7B): MoE의 품질 이점은 연산 임계값(활성 파라미터 ~6B) 위에서만 나타난다.

## 산출물 (Ship It)

`outputs/skill-moe-configurator.md`를 참고하라. 이 스킬은 파라미터 예산, 학습 토큰, 배포 대상이 주어진 새 MoE에 대해 E, k, 그리고 공유 전문가 레이아웃을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 보조 손실 없는 편향 갱신이 50번의 반복에 걸쳐 전문가 사용량을 어떻게 고르게 만드는지 관찰하라.
2. **중간.** 학습된 라우터를 해시 기반 라우터(결정론적, 학습 없음)로 교체하라. 품질과 균형을 비교하라. 학습된 라우터가 왜 더 나은가?
3. **어려움.** GRPO 스타일의 "롤아웃 일치 라우팅(rollout-matched routing)"(DeepSeek-V3.2 트릭)을 구현하라: 추론 중 어떤 전문가가 작동하는지 기록하고, 그래디언트 계산 중 같은 라우팅을 강제하라. 장난감 정책 경사(policy-gradient) 설정에서 그 효과를 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 전문가(Expert) | "여럿 중 하나의 FFN" | 독립적인 순방향 신경망(feed-forward network); FFN 연산의 희소한 한 조각에 전용된 파라미터. |
| 라우터(Router) | "게이트" | 각 토큰을 각 전문가에 대해 점수 매기는 작은 선형 층; top-k 선택. |
| Top-k 라우팅 | "토큰당 k개의 활성 전문가" | 각 토큰의 FFN 연산이 정확히 k개의 전문가를 거치며, 게이트로 가중된다. |
| 보조 손실(Auxiliary loss) | "부하 분산 페널티" | 치우친 전문가 사용량에 페널티를 주는 추가 손실 항. |
| 보조 손실 없음(Auxiliary-loss-free) | "DeepSeek-V3의 트릭" | 라우터의 선택에만 작용하는 전문가별 편향을 통한 분산; 추가 그래디언트가 없다. |
| 공유 전문가(Shared expert) | "항상 켜짐" | 모든 토큰이 통과하는 추가 전문가; 공통 지식을 포착한다. |
| 전문가 병렬화(Expert parallelism) | "전문가별로 샤딩" | 서로 다른 전문가를 서로 다른 GPU에 분산; 토큰을 네트워크에 걸쳐 라우팅한다. |
| 희소성(Sparsity) | "활성 파라미터 < 전체 파라미터" | `k × expert_size / (E × expert_size)` 비율; DeepSeek-V3의 경우 37/671 ≈ 5.5%. |

## 더 읽을거리 (Further Reading)

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538): 그 아이디어.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961): Switch, 고전적 MoE.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): MLA + 보조 손실 없는 MoE + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664): 편향 기반 분산 논문.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066): 이 레슨의 라우터가 사용하는 세분화 + 공유 전문가 분할.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596): 원조 공유 전문가 논문.
