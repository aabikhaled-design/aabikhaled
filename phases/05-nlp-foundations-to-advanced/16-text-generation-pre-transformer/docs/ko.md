# Text Generation Before Transformers — N-gram Language Models

> 어떤 단어가 놀랍다면, 그 모델은 나쁜 것이다. 펄플렉시티(perplexity)는 놀라움을 하나의 숫자로 만든다. 스무딩(smoothing)은 그것을 유한하게 유지한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45분

## 문제 (The Problem)

트랜스포머(transformer) 이전, RNN 이전, 단어 임베딩(word embedding) 이전에, 언어 모델(language model)은 어떤 단어가 앞선 `n-1`개 단어를 얼마나 자주 뒤따르는지 세어서 다음 단어를 예측했다. "the cat" → "sat"을 47번, "the cat" → "jumped"를 12번, "the cat" → "refrigerator"를 0번 센다. 정규화(normalization)하여 확률 분포(probability distribution)를 얻는다.

그것이 n-gram 언어 모델이다. 1980년부터 2015년까지 모든 음성 인식기, 모든 맞춤법 검사기, 모든 구문 기반 기계 번역(machine translation) 시스템을 구동했다. 저렴한 온디바이스(on-device) 언어 모델링이 필요할 때는 여전히 돌아간다.

흥미로운 문제는 보지 못한 n-gram을 어떻게 할 것인가다. 원시 카운트 기반 모델은 본 적 없는 모든 것에 0의 확률을 할당하는데, 이는 치명적이다. 문장은 길고 거의 모든 긴 문장은 적어도 하나의 보지 못한 수열을 포함하기 때문이다. 50년간의 스무딩 연구가 그것을 고쳤다. Kneser-Ney 스무딩이 그 결과이며, 현대 딥러닝(deep learning)은 그 경험적 전통을 물려받았다.

## 개념 (The Concept)

![N-gram model: count, smooth, generate](../assets/ngram.svg)

### 맞히기 놀이

이 기계 장치가 생기기 전에, 언어 모델이 무엇인지를 정의한 실험이 하나 있었다. 영어 문장의 다음 글자를 가린다. 누군가에게 한 번에 하나씩, 맞힐 때까지 추측하게 한다. 몇 번 만에 맞혔는지 적는다. 이것을 몇백 글자에 대해 되풀이한다.

그 추측 횟수는 잡지식이 아니다. 그것은 원문을 손실 없이 다시 부호화한 것이다. 그 횟수 수열을 똑같은 방식으로 추측하는 두 번째 사람에게 건네면 모든 글자를 복원할 수 있다. 각 자리에서 어떤 추측이 먼저 오는지 정확히 알기 때문이다. 더 적은 기호로 다시 부호화할 수 있는 메시지는 기호 하나에 담는 정보가 더 적으므로, 이 추측 횟수 통계가 영어의 엔트로피에 천장을 씌운다.

섀넌은 1951년에 이 실험을 돌려 이 분야를 여전히 지배하는 숫자를 얻었다. 기호 27개(글자 26개에 공백)로 이루어진 알파벳은 글자당 `log2(27) ≈ 4.75` 비트를 실어 나를 수 있다. 100글자의 맥락을 가진 사람은 글자당 0.6에서 1.3비트 사이에 떨어졌다. 영어는 대략 4분의 3이 정해진 수순이다. 모델이 배워야 할 구조가, 어떤 모델도 배울 수 없던 시절에 이미 측정된 것이다.

그 뒤의 모든 언어 모델은 이 놀이를 기계로 하는 선수이며, 이 레슨의 모든 평가 수치는 그 놀이의 점수다.

- **교차 엔트로피 손실**은 모델이 기호 하나에 필요로 하는 평균 비트 수다. 언어 모델을 학습시키는 일은 말 그대로 이 맞히기 놀이의 점수를 낮추는 일이다.
- **당혹도(perplexity)**는 `2^bits`(또는 `e^nats`)다. 추측을 마친 뒤에도 모델 앞에 남아 있는 갈래 수를 뜻한다. 기호 27개를 고르게 추측하면 당혹도가 27이고, 글자당 1비트인 선수는 당혹도가 2다.
- **맥락 길이는 선수의 기억력이다.** 3-gram 모델은 토큰 두 개의 기억으로 놀이를 한다. 트랜스포머는 같은 놀이를 토큰 10만 개로 한다. 규칙은 한 번도 바뀌지 않았고, 선수가 좋아졌을 뿐이다.

단위 하나만 따라가면 된다. 이 놀이는 글자당 비트(`log2`)로 점수를 매기고, 아래의 n-gram 공식은 단어 토큰당 내트(자연로그)로 점수를 매긴다. 내트로 잰 당혹도 `e^H`는 비트로 잰 `2^H`와 같으므로, 두 관점은 단위만 다른 같은 측정이다.

```figure
prediction-game
```

**N-gram 확률:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. `n`을 고정한다(보통 트라이그램(trigram)은 3, 4-gram은 4). 카운트로부터 계산한다:

```text
P(w | context) = count(context, w) / count(context)
```

**제로 카운트 문제.** 학습에서 보지 못한 모든 n-gram은 확률 0을 받는다. Brown 코퍼스(corpus)에 대한 2007년 연구는 4-gram 모델조차 홀드아웃(held-out) 4-gram의 30%가 학습에서 보지 못한 것임을 발견했다. 스무딩 없이는 어떤 실제 텍스트에서도 평가할 수 없다.

**스무딩 접근법, 정교함 순서:**

1. **라플라스(Laplace, add-one).** 모든 카운트에 1을 더한다. 단순하지만 희귀 사건에 형편없다.
2. **Good-Turing.** 빈도의 빈도(frequency-of-frequencies)에 기반해, 더 높은 빈도 사건에서 보지 못한 사건으로 확률 질량을 재할당한다.
3. **보간(Interpolation).** n-gram, (n-1)-gram 등의 추정치를 튜닝 가능한 가중치로 결합한다.
4. **백오프(Backoff).** n-gram의 카운트가 0이면 (n-1)-gram으로 후퇴한다. Katz 백오프가 이를 정규화한다.
5. **절대 할인(Absolute discounting).** 모든 카운트에서 고정 할인 `D`를 빼고, 보지 못한 것으로 재분배한다.
6. **Kneser-Ney.** 절대 할인에 더해, 저차(lower-order) 모델에 대한 영리한 선택: 원시 빈도 대신 *연속 확률(continuation probability)*(어떤 단어가 몇 개의 맥락에 나타나는지)을 사용한다.

Kneser-Ney 통찰은 심오하다. "San Francisco"는 흔한 바이그램(bigram)이다. 유니그램(unigram) "Francisco"는 대부분 "San" 뒤에 나타난다. 순진한 절대 할인은 "Francisco"에 높은 유니그램 확률을 준다(카운트가 높기 때문). Kneser-Ney는 "Francisco"가 단 하나의 맥락에만 나타남을 알아채고 그에 맞춰 연속 확률을 낮춘다. 결과적으로 "Francisco"로 끝나는 새로운 바이그램은 적절한 낮은 확률을 받는다.

**평가: 펄플렉시티.** 홀드아웃 테스트 세트에서 단어당 평균 음의 로그 가능도(negative log-likelihood)의 지수다. 낮을수록 좋다. 펄플렉시티 100은 모델이 100개 단어 중에서 균등하게 고를 때만큼 혼란스럽다는 뜻이다.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 직접 만들기 (Build It)

### Step 1: 트라이그램 카운트

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

입력은 토큰화된 문장의 리스트다. 출력은 n-gram 카운트와 맥락 카운트다. `<s>`와 `</s>`는 문장 경계다.

### Step 2: 라플라스 스무딩

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

모든 카운트에 1을 더한다. 스무딩은 되지만 보지 못한 사건에 질량을 과다 할당하여, 희귀하지만 알려진 사건까지 해친다.

### Step 3: Kneser-Ney (바이그램, 보간)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

세 개의 움직이는 부분. `continuation_prob`는 "이 단어가 몇 개의 서로 다른 맥락에 나타나는가?"를 포착한다(Kneser-Ney 혁신). `lambda_prev`는 할인으로 풀려난 질량이며, 백오프에 가중치를 주는 데 쓰인다. 최종 확률은 할인된 주 항에 가중된 연속 항을 더한 값이다.

### Step 4: 샘플링으로 텍스트 생성

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

확률에 비례하는 샘플링(sampling)이다. 시드(seed)마다 출력이 달라진다. 빔 서치(beam-search) 같은 출력을 위해서는 각 스텝에서 argmax를 고르고(그리디), 작은 무작위성 노브(온도, temperature)를 추가한다.

### Step 5: 펄플렉시티

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

낮을수록 좋다. Brown 코퍼스의 경우, 잘 튜닝된 4-gram KN 모델은 펄플렉시티 약 140에 이른다. 트랜스포머 LM은 같은 테스트 세트에서 15-30에 이른다. 격차는 약 10배다. 그 격차가 이 분야가 나아간 이유다.

## 라이브러리로 써보기 (Use It)

- **고전 NLP 교육.** 얻을 수 있는 스무딩, MLE, 펄플렉시티에 대한 가장 명료한 노출.
- **KenLM.** 프로덕션(production) n-gram 라이브러리. 낮은 지연 시간(latency)이 중요한 음성 및 MT 시스템에서 리스코어러(rescorer)로 사용된다.
- **온디바이스 자동완성.** 키보드의 트라이그램 모델. 여전히.
- **베이스라인(Baseline).** 신경망 LM이 좋다고 선언하기 전에 항상 n-gram LM 펄플렉시티를 계산하라. 트랜스포머가 KN을 큰 차이로 이기지 못한다면, 무언가 잘못된 것이다.

## 산출물 (Ship It)

`outputs/prompt-lm-baseline.md`로 저장하라:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 연습 문제 (Exercises)

1. **Easy.** 1,000문장짜리 셰익스피어 코퍼스에 트라이그램 LM을 학습시켜라. 20개 문장을 생성하라. 국소적으로는 그럴듯하지만 전역적으로는 일관성이 없을 것이다. 이것이 표준 데모다.
2. **Medium.** 홀드아웃 셰익스피어 분할에서 KN 모델의 펄플렉시티를 구현하라. 라플라스와 비교하라. KN이 펄플렉시티를 30-50% 낮추는 것을 볼 것이다.
3. **Hard.** 트라이그램 맞춤법 교정기를 만들어라. 잘못 쓰인 단어와 그 맥락이 주어지면, 교정을 생성하고 LM 하의 맥락 확률로 순위를 매겨라. Birkbeck 맞춤법 코퍼스(공개)에서 평가하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| N-gram | 단어 수열 | `n`개의 연속된 토큰의 수열. |
| Smoothing | 0 피하기 | 보지 못한 사건이 0이 아닌 확률을 받도록 확률 질량을 재할당하기. |
| Perplexity | LM 품질 지표 | 홀드아웃 데이터에서의 `exp(-average log-prob)`. 낮을수록 좋다. |
| Backoff | 더 짧은 맥락으로 후퇴 | 트라이그램 카운트가 0이면 바이그램을 사용한다. Katz 백오프가 이를 공식화한다. |
| Kneser-Ney | n-gram을 위한 최고의 스무딩 | 절대 할인 + 저차 모델을 위한 연속 확률. |
| Continuation probability | KN 특화 | 원시 카운트가 아니라 `w`가 나타나는 맥락의 수로 가중된 `P(w)`. |
| Entropy of text | 기호당 정보량 | 맥락이 주어졌을 때 다음 기호를 부호화하는 데 필요한 평균 비트 수. 최대 100글자의 맥락을 준 인쇄 영어에 대한 섀넌의 1951년 추정치는 글자당 0.6-1.3비트로, 어떤 모델도 존재하기 전에 측정됐다. |

## 더 읽을거리 (Further Reading)

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) 모든 언어 모델이 여전히 최적화하는 목표를 정의한 맞히기 놀이 실험.
- [Jurafsky and Martin: Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) n-gram LM과 스무딩에 대한 표준 해설.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739): Kneser-Ney를 최고의 n-gram 스무더로 정착시킨 논문.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394): 원래의 KN 논문.
- [KenLM](https://kheafield.com/code/kenlm/): 빠른 프로덕션 n-gram LM, 2026년에도 지연 시간에 민감한 응용에 여전히 사용됨.
