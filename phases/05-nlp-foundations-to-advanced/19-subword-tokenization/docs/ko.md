# Subword Tokenization — BPE, WordPiece, Unigram, SentencePiece

> 단어 토크나이저(tokenizer)는 보지 못한 단어 앞에서 막힌다. 문자 토크나이저는 시퀀스 길이를 폭발시킨다. 서브워드(subword) 토크나이저는 그 사이를 가른다. 모든 현대 LLM은 그중 하나 위에 올라가 배포된다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60분

## 문제 (The Problem)

어휘에 단어가 50,000개 있다. 사용자가 "untokenizable"을 입력한다. 토크나이저는 `[UNK]`를 반환한다. 이제 모델은 그 단어에 대한 어떤 신호도 갖지 못한다. 더 나쁜 경우도 있다. 코퍼스(corpus)에서 90 백분위수 문서에 희귀 단어가 40개 들어 있으면, 문서당 40비트의 정보가 버려진다는 뜻이다.

서브워드 토큰화(subword tokenization)는 이를 해결한다. 흔한 단어는 단일 토큰으로 남는다. 희귀 단어는 의미 있는 조각으로 분해된다: `untokenizable` → `un`, `token`, `izable`. 어떤 문자열도 결국 바이트(byte)의 수열이므로, 학습 데이터가 모든 것을 커버한다.

2026년의 모든 프런티어 LLM은 세 가지 알고리즘(BPE, Unigram, WordPiece) 중 하나 위에 배포되며, 세 가지 라이브러리(tiktoken, SentencePiece, HF Tokenizers) 중 하나로 감싼다. 하나를 고르지 않고는 언어 모델(language model)을 배포할 수 없다.

## 개념 (The Concept)

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** 문자 수준 어휘에서 시작한다. 모든 인접 쌍을 센다. 가장 빈번한 쌍을 새 토큰으로 병합(merge)한다. 목표 어휘 크기에 도달할 때까지 반복한다. 지배적 알고리즘: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**바이트 수준 BPE(Byte-level BPE).** 같은 알고리즘이지만 유니코드 문자 대신 원시 바이트(256개 기본 토큰)에 대해 작동한다. 어떤 바이트 수열도 인코딩되므로 `[UNK]` 토큰이 0개임을 보장한다. GPT-2는 50,257개 토큰을 사용한다(256개 바이트 + 50,000개 병합 + 1개 특수).

**Unigram.** 거대한 어휘에서 시작한다. 각 토큰에 유니그램(unigram) 확률을 할당한다. 제거했을 때 코퍼스 로그 가능도(log-likelihood)를 가장 적게 떨어뜨리는 토큰을 반복적으로 가지친다(prune). 추론(inference) 시에는 확률적이라, 토큰화를 샘플링(sampling)할 수 있다(서브워드 정규화(subword regularization)를 통한 데이터 증강에 유용). T5, mBART, ALBERT, XLNet, Gemma가 사용한다.

**WordPiece.** 원시 빈도가 아니라 학습 코퍼스의 가능도를 최대화하는 쌍을 병합한다. BERT, DistilBERT, ELECTRA가 사용한다.

**SentencePiece 대 tiktoken.** SentencePiece는 원시 유니코드 텍스트에서 직접 어휘(BPE 또는 Unigram)를 *학습하는* 라이브러리이며, 공백을 `▁`로 인코딩한다. tiktoken은 미리 만들어진 어휘를 쓰는 OpenAI의 빠른 *인코더*다. 학습은 하지 않는다.

경험 법칙:

- **새 어휘 학습:** SentencePiece(다국어, 사전 토큰화 없음) 또는 HF Tokenizers.
- **GPT 어휘에 대한 빠른 추론:** tiktoken(cl100k_base, o200k_base).
- **둘 다:** HF Tokenizers: 하나의 라이브러리, 학습 + 서빙.

```figure
bpe-merge
```

## 직접 만들기 (Build It)

### Step 1: 밑바닥부터 만드는 BPE

`code/main.py`를 보라. 루프:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

알고리즘이 인코딩하는 세 가지 사실. `</w>`는 단어 끝을 표시해 "low"(접미사)와 "lower"(접두사)를 구별되게 한다. 빈도 가중치는 고빈도 쌍이 일찍 이기게 만든다. 병합 목록은 순서가 있다. 추론은 학습 순서대로 병합을 적용한다.

### Step 2: 학습된 병합으로 인코딩

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

순진한 O(n·|merges|)다. 프로덕션 구현(tiktoken, HF Tokenizers)은 우선순위 큐를 가진 병합 순위 조회를 써서 거의 선형 시간에 실행된다.

### Step 3: 실전에서의 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

주목하라: 사전 토큰화가 필요 없고, 공백이 `▁`로 인코딩되며, `character_coverage`는 희귀 문자를 보존할지 아니면 `<unk>`로 매핑할지를 얼마나 공격적으로 정할지 제어한다.

### Step 4: OpenAI 호환 어휘를 위한 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

인코딩 전용이다. 빠르다(Rust 백엔드). 바이트 카운팅, 비용 추정, 컨텍스트 윈도우(context-window) 예산 산정을 위해 GPT-4/5 토큰화와 정확히 일치한다.

## 2026년에도 여전히 배포되는 함정

- **토크나이저 표류(Tokenizer drift).** 어휘 A로 학습하고, 어휘 B로 배포한다. 토큰 ID가 어긋난다. 모델 출력이 쓰레기가 된다. CI에서 `tokenizer.json` 해시를 확인하라.
- **공백 모호성(Whitespace ambiguity).** BPE에서 "hello"와 " hello"는 서로 다른 토큰을 만든다. 항상 `add_special_tokens`와 `add_prefix_space`를 명시적으로 지정하라.
- **다국어 과소 학습(Multilingual undertraining).** 영어 위주 코퍼스는 비라틴 문자 체계를 5-10배 더 많은 토큰으로 쪼개는 어휘를 만든다. 같은 프롬프트(prompt)가 GPT-3.5에서 일본어/아랍어로는 5-10배 더 비싸다. o200k_base가 이를 부분적으로 고쳤다.
- **이모지 분할(Emoji splits).** 단일 이모지가 5개 토큰을 차지할 수 있다. 컨텍스트 예산을 잡을 때 이모지 처리를 점검하라.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 밑바닥부터 단일 언어 모델 학습 | HF Tokenizers (BPE) |
| 다국어 모델 학습 | SentencePiece (Unigram, `character_coverage=0.9995`) |
| OpenAI 호환 API 서빙 | tiktoken (GPT-4+에는 `o200k_base`) |
| 도메인별 어휘(코드, 수학, 단백질) | 도메인 코퍼스에 커스텀 BPE를 학습하고 기본 어휘와 병합 |
| 엣지 추론, 작은 모델 | Unigram (더 작은 어휘가 더 잘 작동) |

어휘 크기는 상수가 아니라 스케일링 결정이다. 대략적인 휴리스틱: <1B 파라미터(parameter)에는 32k, 1-10B에는 50-100k, 다국어/프런티어에는 200k+.

## 산출물 (Ship It)

`outputs/skill-bpe-vs-wordpiece.md`로 저장하라:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 연습 문제 (Exercises)

1. **Easy.** `code/main.py`의 작은 코퍼스에 500-병합 BPE를 학습시켜라. 홀드아웃(held-out) 단어 세 개를 인코딩하라. 정확히 토큰 1개를 만든 것과 토큰을 1개보다 많이 만든 것은 각각 몇 개인가?
2. **Medium.** 100개의 영어 위키피디아 문장에서 `cl100k_base`, `o200k_base`, 그리고 vocab=32k로 직접 학습한 SentencePiece BPE 사이의 토큰 수를 비교하라. 각각의 압축률을 보고하라.
3. **Hard.** 같은 코퍼스를 BPE, Unigram, WordPiece로 학습시켜라. 작은 감성 분류기(classifier)에서 각각을 사용했을 때의 다운스트림 정확도를 측정하라. 선택에 따라 F1이 1점 넘게 움직이는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | 목표 어휘 크기에 도달할 때까지 가장 빈번한 문자 쌍을 탐욕적으로 병합. |
| Byte-level BPE | 절대 미지 토큰 없음 | 원시 256바이트에 대한 BPE. GPT-2 / Llama가 사용. |
| Unigram | 확률적 토크나이저 | 로그 가능도를 사용해 큰 후보 집합에서 가지치기. T5, Gemma가 사용. |
| SentencePiece | 공백을 다루는 것 | 원시 텍스트에 BPE/Unigram을 학습하는 라이브러리. 공백을 `▁`로 인코딩. |
| tiktoken | 빠른 것 | 미리 만들어진 어휘를 위한 OpenAI의 Rust 백엔드 BPE 인코더. 학습 없음. |
| Merge list | 마법의 숫자들 | 순서가 있는 `(a, b) → ab` 병합 목록. 추론은 순서대로 적용. |
| Character coverage | 얼마나 희귀해야 너무 희귀한가? | 토크나이저가 커버해야 하는 학습 코퍼스 내 문자의 비율. ~0.9995가 일반적. |

## 더 읽을거리 (Further Reading)

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909): BPE 논문.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959): Unigram 논문.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226): 라이브러리.
- [Hugging Face: Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) 간결한 레퍼런스.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken): 쿡북 + 인코딩 목록.
