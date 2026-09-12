# T5, BART: 인코더-디코더 모델(Encoder-Decoder Models)

> 인코더(encoder)는 이해한다. 디코더(decoder)는 생성한다. 둘을 다시 합치면 입력 → 출력 과제를 위해 만들어진 모델을 얻는다. 번역, 요약, 재작성, 전사(transcribe).

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45분

## 문제 (The Problem)

디코더 전용 GPT와 인코더 전용 BERT는 각자 다른 목표에 맞춰 2017년 아키텍처를 덜어낸다. 하지만 많은 과제는 본래 입력-출력이다.

- 번역: 영어 → 프랑스어.
- 요약: 5,000 토큰(token) 기사 → 200 토큰 요약.
- 음성 인식: 오디오 토큰 → 텍스트 토큰.
- 구조화된 추출: 산문 → JSON.

이런 과제에는 인코더-디코더가 가장 깔끔하게 맞는다. 인코더는 소스의 조밀한 표현을 만든다. 디코더는 매 스텝 그 표현에 크로스 어텐션(cross-attention)하며 출력을 생성한다. 학습(training)은 출력 쪽에서 한 칸 이동이다. GPT와 같은 손실(loss), 단지 인코더 출력을 조건으로 삼을 뿐이다.

두 논문이 현대의 플레이북을 정의했다.

1. **T5** (Raffel et al. 2019). "Text-to-Text Transfer Transformer." 모든 NLP 과제를 텍스트 입력, 텍스트 출력으로 재구성했다. 단일 아키텍처, 단일 어휘(vocabulary), 단일 손실. 마스킹된 스팬(span) 예측으로 사전 학습(pretraining)했다(입력에서 스팬을 손상시키고, 출력에서 그것들을 디코딩한다).
2. **BART** (Lewis et al. 2019). "Bidirectional and Auto-Regressive Transformer." 잡음 제거 오토인코더(denoising autoencoder)다. 입력을 여러 방식으로 손상시키고(섞기, 마스킹, 삭제, 회전), 디코더에게 원본을 복원하도록 요청한다.

2026년에 인코더-디코더 형식은 입력 구조가 중요한 곳에서 살아남는다.

- Whisper(음성 → 텍스트).
- Google의 번역 스택.
- 뚜렷한 컨텍스트-및-편집 구조를 가진 일부 코드 완성 / 수리 모델.
- 구조화된 추론 과제를 위한 Flan-T5와 변형들.

디코더 전용이 각광을 차지했지만, 인코더-디코더는 결코 사라지지 않았다.

## 개념 (The Concept)

![크로스 어텐션을 갖춘 인코더-디코더](../assets/encoder-decoder.svg)

### 순방향 루프

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

결정적으로, 인코더는 입력당 한 번 실행된다. 디코더는 자기회귀(autoregressive)적으로 실행되지만 매 스텝 *같은* 인코더 출력에 크로스 어텐션한다. 인코더 출력을 캐싱하면 긴 입력에서 공짜로 속도가 빨라진다.

### T5 사전 학습: 스팬 손상

입력의 무작위 스팬(평균 길이 3 토큰, 총 15%)을 고른다. 각 스팬을 고유한 센티넬(sentinel)로 교체한다: `<extra_id_0>`, `<extra_id_1>` 등. 디코더는 센티넬 접두사와 함께 손상된 스팬만 출력한다.

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

전체 시퀀스를 예측하는 것보다 저렴한 신호다. T5 논문의 절제 실험에서 MLM(BERT) 및 prefix-LM(UniLM)과 경쟁력 있었다.

### BART 사전 학습: 다중 잡음 제거

BART는 다섯 가지 잡음 함수를 시도한다.

1. 토큰 마스킹.
2. 토큰 삭제.
3. 텍스트 채우기(text infilling)(스팬을 마스킹하고, 디코더가 올바른 길이를 삽입).
4. 문장 순열.
5. 문서 회전.

텍스트 채우기와 문장 순열을 결합하면 가장 좋은 하류 수치가 나왔다. 디코더는 항상 원본을 복원한다. BART의 출력은 손상된 스팬만이 아니라 전체 시퀀스다. 그래서 사전 학습 계산이 T5보다 많다.

### 추론

GPT와 같은 자기회귀 생성이다. greedy / 빔(beam) / top-p 샘플링(sampling)이 적용된다. 출력 분포가 챗보다 좁기 때문에 번역과 요약에는 빔 서치(폭 4-5)가 표준이다.

### 2026년에 각 변형을 언제 고를까

| 과제 | 인코더-디코더? | 이유 |
|------|------------------|-----|
| 번역 | 보통 그렇다 | 명확한 소스 시퀀스. 고정된 출력 분포. 빔 서치가 동작 |
| 음성-투-텍스트 | 그렇다 (Whisper) | 입력 모달리티가 출력과 다름. 인코더가 오디오 특성을 형성 |
| 챗 / 추론 | 아니다, 디코더 전용 | 지속적 "입력" 없음: 대화가 곧 시퀀스 |
| 코드 완성 | 보통 아니다 | 긴 컨텍스트의 디코더 전용이 이김. Qwen 2.5 Coder 같은 코드 모델은 디코더 전용 |
| 요약 | 둘 다 동작 | BART, PEGASUS가 초기 디코더 전용 베이스라인을 이김. 현대 디코더 전용 LLM이 이를 맞춤 |
| 구조화된 추출 | 둘 다 | "텍스트 → 텍스트"가 어떤 출력 형식이든 흡수하므로 T5가 깔끔함 |

2022년 무렵부터의 추세: 디코더 전용이 인코더-디코더가 도맡던 과제를 넘겨받는다. 이유는 (a) 명령 튜닝된 디코더 전용 LLM이 프롬프트(prompt)로 무엇에든 일반화하고, (b) 둘보다 하나의 아키텍처가 더 쉽게 확장되며, (c) RLHF가 디코더를 가정하기 때문이다. 인코더-디코더는 입력 모달리티가 다르거나(음성, 이미지) 빔 서치 품질이 중요한 곳에서 버틴다.

```figure
encoder-decoder
```

## 직접 만들기 (Build It)

`code/main.py`를 참조하라. 장난감 코퍼스에 대해 T5 스타일 스팬 손상을 구현한다. 이 레슨에서 가장 유용한 단일 조각이다. 이후의 모든 인코더-디코더 사전 학습 레시피에 등장하기 때문이다.

### 1단계: 스팬 손상

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

타깃 형식은 T5 관례다: `<sent0> span0 <sent1> span1 ...`. 손상된 입력은 변경되지 않은 토큰과 스팬 위치의 센티넬 토큰을 교차로 배치한다.

### 2단계: 왕복 검증

손상된 입력과 타깃이 주어지면 원래 문장을 복원한다. 손상이 가역적이면 순방향 패스(forward pass)가 잘 정의된다. 이것은 온전성 검사다. 실제 학습은 결코 이렇게 하지 않지만, 테스트가 저렴하고 스팬 부기(bookkeeping)의 off-by-one 버그를 잡아낸다.

### 3단계: BART 잡음 추가

다섯 함수: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. 그중 두 개를 합성하고 결과를 보여준다.

## 라이브러리로 써보기 (Use It)

HuggingFace 레퍼런스:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5의 트릭: 과제 이름이 입력 텍스트에 들어간다. 각 과제가 텍스트 입력, 텍스트 출력이기 때문에 같은 모델이 수십 가지 과제를 다룬다. 2026년에 이 패턴은 명령 튜닝된 디코더 전용 모델로 일반화되었지만, T5가 먼저 이를 성문화했다.

## 산출물 (Ship It)

`outputs/skill-seq2seq-picker.md`를 참조하라. 이 스킬은 입력-출력 구조, 지연 시간(latency), 품질 목표가 주어지면 새 과제에 대해 인코더-디코더와 디코더 전용 사이에서 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하고, 30 토큰 문장에 스팬 손상을 적용하고, 센티넬이 아닌 소스 토큰과 디코딩된 타깃 스팬을 이어 붙이면 원본이 재현되는지 검증한다.
2. **보통.** BART의 `text_infill` 잡음을 구현한다: 무작위 스팬을 단일 `<mask>` 토큰으로 교체하고, 디코더가 올바른 스팬 길이와 내용을 추론해야 한다. 예시 하나를 보여준다.
3. **어려움.** 작은 영어 → 피그 라틴(pig-Latin) 코퍼스(200쌍)에 `flan-t5-small`을 파인튜닝(fine-tuning)한다. 홀드아웃(held-out) 50쌍 셋에서 BLEU를 측정한다. 같은 데이터에 같은 계산으로 `Llama-3.2-1B`를 파인튜닝한 것과 비교한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Encoder-decoder | "seq2seq 트랜스포머" | 두 스택: 입력용 양방향 인코더, 크로스 어텐션이 있는 출력용 인과 디코더. |
| Cross-attention | "소스가 타깃에게 말하는 곳" | 디코더의 Q × 인코더의 K/V. 인코더 정보가 디코더에 들어가는 유일한 곳. |
| Span corruption | "T5의 사전 학습 트릭" | 무작위 스팬을 센티넬 토큰으로 교체. 디코더가 스팬을 출력. |
| Denoising objective | "BART의 게임" | 입력에 잡음 함수를 적용하고, 디코더가 깨끗한 시퀀스를 복원하도록 학습. |
| Sentinel token | "`<extra_id_N>` 자리표시자" | 소스에서 손상된 스팬을 태그하고 타깃에서 다시 태그하는 특수 토큰. |
| Flan | "명령 튜닝된 T5" | 1,800개 이상의 과제에 파인튜닝된 T5. 명령 따르기에서 인코더-디코더를 경쟁력 있게 만듦. |
| Beam search | "디코딩 전략" | 각 스텝에서 상위 k개 부분 시퀀스 유지. 번역/요약의 표준. |
| Teacher forcing | "학습 시 입력" | 학습 중 샘플링된 토큰이 아니라 참 이전 출력 토큰을 디코더에 넣음. |

## 더 읽을거리 (Further Reading)

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356): Whisper, 정전적 2026년 인코더-디코더.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py): 레퍼런스 구현.
