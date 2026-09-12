# BERT: 마스킹된 언어 모델링(Masked Language Modeling)

> GPT는 다음 단어를 예측한다. BERT는 빠진 단어를 예측한다. 한 문장의 차이: 그리고 임베딩(embedding) 모양의 모든 것의 반세기.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45분

## 문제 (The Problem)

2018년에는 모든 NLP 과제(감성, NER, QA, 함의(entailment))가 각자의 레이블된 데이터로 자체 모델을 밑바닥부터 학습(training)했다. 파인튜닝(fine-tuning)할 수 있는 사전 학습된 "영어 이해" 체크포인트가 없었다. ELMo(2018)는 양방향 LSTM으로 맥락적 임베딩을 사전 학습할 수 있음을 보였다. 도움은 됐지만 일반화하지는 못했다.

BERT(Devlin et al. 2018)는 이렇게 물었다. 트랜스포머(transformer) 인코더(encoder)를 가져와 인터넷의 모든 문장으로 학습하고, 양쪽 맥락에서 빠진 단어를 예측하도록 강제하면 어떨까? 그다음 하류 과제에 헤드 하나만 파인튜닝한다. 파라미터(parameter) 효율성이 그야말로 새로운 발견이었다.

그 결과: 18개월 만에 BERT와 그 변형들(RoBERTa, ALBERT, ELECTRA)이 존재하는 모든 NLP 리더보드를 지배했다. 2020년에는 지구상의 모든 검색 엔진, 콘텐츠 모더레이션 파이프라인, 의미 검색 시스템 안에 BERT가 들어 있었다.

2026년에 인코더 전용 모델은 여전히 분류(classification), 검색, 구조화된 추출에 올바른 도구다. 토큰(token)당 디코더(decoder)보다 5-10배 빠르게 동작하고, 그 임베딩은 모든 현대 검색 스택의 중추다. ModernBERT(2024년 12월)는 Flash Attention + RoPE + GeGLU로 아키텍처를 8K 컨텍스트까지 밀어붙였다.

## 개념 (The Concept)

![마스킹된 언어 모델링: 토큰을 고르고, 마스킹하고, 원본을 예측한다](../assets/bert-mlm.svg)

### 학습 신호

문장을 가져온다: `the quick brown fox jumps over the lazy dog`.

토큰의 15%를 무작위로 마스킹한다.

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

마스킹된 위치에서 원본 토큰을 예측하도록 모델을 학습한다. 인코더가 양방향이라 위치 1의 `[MASK]`를 예측할 때 위치 2 이후의 `brown fox jumps`를 끌어다 쓸 수 있다. GPT는 이걸 하지 못한다.

### BERT 마스크 규칙

예측을 위해 선택된 15%의 토큰 중:

- 80%는 `[MASK]`로 교체된다.
- 10%는 무작위 토큰으로 교체된다.
- 10%는 그대로 둔다.

왜 항상 `[MASK]`가 아닌가? `[MASK]`는 추론(inference) 시점에는 결코 나타나지 않기 때문이다. 마스킹된 위치의 100%에서 `[MASK]`를 기대하도록 모델을 학습하면 사전 학습과 파인튜닝 사이에 분포 이동(distribution shift)이 생긴다. 10% 무작위 + 10% 변경 없음을 섞어 모델이 한쪽으로 치우치지 않게 한다.

### 다음 문장 예측(Next Sentence Prediction, NSP): 그리고 왜 버려졌는가

원래 BERT는 NSP로도 학습했다. 두 문장 A와 B가 주어지면 B가 A 뒤에 오는지 예측한다. RoBERTa(2019)는 이를 절제(ablate)해 NSP가 도움은커녕 해가 된다는 사실을 보였다. 현대 인코더는 이 과제를 건너뛴다.

### 2026년에 바뀐 것: ModernBERT

2024년 ModernBERT 논문은 블록을 2026년 기본 요소로 재구축했다.

| 구성요소 | 원래 BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

그리고 2018년 스택과 달리 Flash-Attention 네이티브다. 시퀀스 길이 8K에서 추론이 DeBERTa-v3보다 2-3배 빠르면서 더 나은 GLUE 점수를 낸다.

### 2026년에 여전히 인코더를 고르는 사용 사례

| 과제 | 인코더가 디코더를 이기는 이유 |
|------|---------------------------|
| 검색 / 의미 검색 임베딩 | 양방향 맥락 = 토큰당 더 나은 임베딩 품질 |
| 분류 (감성, 의도, 독성) | 한 번의 순방향 패스. 생성 오버헤드 없음 |
| NER / 토큰 라벨링 | 위치별 출력, 본래 양방향 |
| 제로샷 함의 (NLI) | 인코더 위의 분류기 헤드 |
| RAG용 리랭커 | 크로스 인코더 점수화, LLM 리랭커보다 10배 빠름 |

```figure
transformer-residual
```

## 직접 만들기 (Build It)

### 1단계: 마스킹 로직

`code/main.py`를 참조하라. `create_mlm_batch` 함수는 토큰 ID 목록, 어휘(vocab) 크기, 마스크 확률을 받는다. 입력 ID(마스크 적용됨)와 레이블(마스킹된 위치에만, 그 외에는 -100: PyTorch의 무시 인덱스 관례)을 반환한다.

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 2단계: 작은 코퍼스에 MLM 예측 실행하기

20개 단어 어휘, 200개 문장에 2층 인코더 + MLM 헤드를 학습한다. 그래디언트(gradient) 없음: 순방향 패스 온전성 검사를 한다. 전체 학습에는 PyTorch가 필요하다.

### 3단계: 마스크 유형 비교하기

세 갈래 규칙 덕분에 `[MASK]` 없이도 모델을 쓸 수 있다는 점을 보여준다. 마스킹하지 않은 문장과 마스킹한 문장 양쪽에 예측을 돌린다. 모델이 학습에서 두 패턴을 모두 봤으므로 둘 다 합리적인 토큰 분포를 내야 한다.

### 4단계: 헤드 파인튜닝

MLM 헤드를 장난감 감성 데이터셋(dataset)의 분류 헤드로 교체한다. 헤드만 학습한다. 인코더는 동결(freeze)된다. 이것이 모든 BERT 응용이 따르는 패턴이다.

## 라이브러리로 써보기 (Use It)

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**임베딩 모델은 파인튜닝된 BERT다.** `all-MiniLM-L6-v2` 같은 `sentence-transformers` 모델은 대조 손실(contrastive loss)로 학습된 BERT다. 인코더는 같다. 손실(loss)이 바뀌었다.

**크로스 인코더 리랭커도 파인튜닝된 BERT다.** `[CLS] query [SEP] doc [SEP]`에 대한 쌍 분류. 쿼리와 문서 사이의 양방향 어텐션(attention)이 바로 크로스 인코더가 바이인코더 대비 품질 우위를 갖게 하는 것이다.

**2026년에 BERT를 고르지 말아야 할 때.** 생성적인 모든 것. 인코더는 토큰을 자기회귀(autoregressive)적으로 생성할 합리적 방법이 없다. 또한: 작은 디코더가 더 큰 유연성으로 품질을 맞출 수 있는 1B 파라미터 미만의 모든 것(Phi-3-Mini, Qwen2-1.5B).

## 산출물 (Ship It)

`outputs/skill-bert-finetuner.md`를 참조하라. 이 스킬은 새 분류 또는 추출 과제에 대해 BERT 파인튜닝(백본 선택, 헤드 사양, 데이터, 평가, 중단)을 범위 짓는다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하고 10,000개 토큰에 걸친 마스크 분포를 출력한다. 약 15%가 선택되고, 그중 약 80%가 `[MASK]`가 되는지 확인한다.
2. **보통.** 전체 단어 마스킹(whole-word masking)을 구현한다: 한 단어가 하위 단어(subword)로 토큰화되면, 모든 하위 단어를 함께 마스킹하거나 아무것도 마스킹하지 않는다. 이것이 500문장 코퍼스에서 MLM 정확도를 개선하는지 측정한다.
3. **어려움.** 공개 데이터셋의 문장 10,000개로 작은(2층, d=64) BERT를 학습한다. SST-2 감성을 위해 `[CLS]` 토큰을 파인튜닝한다. 동일 파라미터에서 디코더 전용 베이스라인과 비교한다. 어느 쪽이 이기는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| MLM | "마스킹된 언어 모델링" | 학습 신호: 토큰의 15%를 무작위로 `[MASK]`로 교체하고 원본을 예측. |
| Bidirectional | "양쪽을 봄" | 인코더 어텐션에 인과 마스크가 없음: 모든 위치가 다른 모든 위치를 봄. |
| `[CLS]` | "풀러(pooler) 토큰" | 모든 시퀀스 앞에 붙는 특수 토큰. 그 최종 임베딩이 문장 수준 표현으로 쓰임. |
| `[SEP]` | "세그먼트 구분자" | 쌍 시퀀스(예: query/doc, 문장 A/B)를 구분. |
| NSP | "다음 문장 예측" | BERT의 두 번째 사전 학습 과제. RoBERTa에서 쓸모없음이 드러나 2019년 이후 버려짐. |
| Fine-tuning | "과제에 적응" | 인코더를 대부분 동결하고 하류 과제를 위해 위에 작은 헤드를 학습. |
| Cross-encoder | "리랭커" | 쿼리와 문서를 둘 다 입력받아 관련성 점수를 출력하는 BERT. |
| ModernBERT | "2024년 리프레시" | RoPE, RMSNorm, GeGLU, 교대 로컬/글로벌 어텐션, 8K 컨텍스트로 재구축된 인코더. |

## 더 읽을거리 (Further Reading)

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805): 원조 논문.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692): BERT를 제대로 학습하는 법. NSP를 죽임.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555): 교체된 토큰 탐지가 동일 계산에서 MLM을 이김.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663): ModernBERT 논문.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py): 정전적 인코더 레퍼런스.
