# 밑바닥부터 토크나이저 만들기

> Lesson 01이 장난감을 줬다면, 이 레슨은 무기를 준다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90분

## 학습 목표 (Learning Objectives)

- 유니코드(Unicode), 공백 정규화(whitespace normalization), 특수 토큰을 처리하는 프로덕션(production) 등급 BPE 토크나이저(tokenizer) 만들기
- 바이트 수준 폴백(byte-level fallback)을 구현해 토크나이저가 (이모지, CJK, 코드를 포함한) 어떤 입력이든 알 수 없는 토큰 없이 인코딩할 수 있게 하기
- BPE 병합(merge)을 적용하기 전에 단어 경계에서 텍스트를 나누는 사전 토큰화(pre-tokenization) 정규식 패턴 추가하기
- 말뭉치(corpus)에서 커스텀 토크나이저를 학습시키고 다국어 텍스트에서 tiktoken과 압축비(compression ratio)를 비교 평가하기

## 문제 (The Problem)

Lesson 01에서 만든 BPE 토크나이저는 영어 텍스트에서 작동한다. 이제 거기에 일본어를 던져 보라. 또는 이모지를. 또는 탭과 공백이 섞인 Python 코드를.

깨진다.

BPE가 틀렸기 때문이 아니라 구현이 불완전하기 때문이다. 프로덕션 토크나이저는 어떤 인코딩의 원시 바이트든 처리하고, 나누기 전에 유니코드를 정규화하며, 절대 병합되지 않는 특수 토큰을 관리하고, 사전 토큰화를 서브워드(subword) 분할과 연결한다. 그리고 이 모든 것을 15조 토큰을 처리하는 학습 파이프라인(pipeline)에 병목이 되지 않을 만큼 빠르게 해낸다.

GPT-2의 토크나이저는 50,257개의 토큰을 가진다. Llama 3는 128,256개다. GPT-4는 대략 100,000개다. 이것들은 장난감 숫자가 아니다. 그 어휘 뒤에 있는 병합 테이블(merge table)은 수백 기가바이트의 텍스트에서 학습되었고, 그 주변의 기계 장치, 곧 정규화와 사전 토큰화, 특수 토큰 주입, 채팅 템플릿(chat template) 포맷팅이 "hello world"를 처리하는 토크나이저와 인터넷 전체를 처리하는 토크나이저를 가른다.

이제 그 기계 장치를 직접 만든다.

## 개념 (The Concept)

### 전체 파이프라인

프로덕션 토크나이저는 알고리즘 하나가 아니다. 각각 다른 문제를 해결하는 다섯 단계의 파이프라인이다.

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

각 단계에는 구체적인 역할이 있다:

| 단계 | 하는 일 | 왜 중요한가 |
|-------|-------------|----------------|
| 정규화(Normalize) | NFKC 유니코드, 선택적 소문자화, 선택적 악센트 제거 | "fi" 합자(U+FB01)가 "fi"(두 글자)가 된다. 이것이 없으면 같은 단어가 다른 토큰을 얻는다. |
| 사전 토큰화(Pre-Tokenize) | BPE 전에 텍스트를 덩어리로 나눔 | BPE가 단어 경계를 넘어 병합하는 것을 막는다. "the cat"이 절대 "e c" 토큰을 만들어선 안 된다. |
| BPE 병합(BPE Merge) | 학습된 병합 규칙을 바이트 시퀀스에 적용 | 핵심 압축. 원시 바이트를 서브워드 토큰으로 바꾼다. |
| 특수 토큰(Special Tokens) | [BOS], [EOS], [PAD], 채팅 템플릿 마커 주입 | 이 토큰들은 고정된 ID를 가진다. BPE 병합에 절대 참여하지 않는다. 모델은 구조를 위해 이것들이 필요하다. |
| ID 매핑(ID Mapping) | 토큰 문자열을 정수 ID로 변환 | 모델은 문자열이 아니라 정수를 본다. |

### 바이트 수준 BPE (Byte-Level BPE)

Lesson 01의 토크나이저는 UTF-8 바이트 위에서 동작했다. 그것은 올바른 선택이었다. 하지만 우리는 중요한 것을 건너뛰었다: 그 바이트가 유효한 UTF-8이 아니면 어떻게 되는가?

바이트 수준 BPE(Byte-level BPE)는 가능한 모든 바이트 값(0-255)을 유효한 토큰으로 취급함으로써 이를 해결한다. 기본 어휘는 정확히 256개의 항목이다. 텍스트든 바이너리든 손상된 것이든, 어떤 파일이라도 알 수 없는 토큰을 만들지 않고 토큰화된다.

GPT-2는 한 가지 트릭을 추가했다: 어휘가 사람이 읽을 수 있게 유지되도록 각 바이트를 출력 가능한 유니코드 문자로 매핑한다. 바이트 0x20(공백)은 그들의 매핑에서 문자 "Ġ"가 된다. 이것은 순전히 표면적인 것이다. 알고리즘은 신경 쓰지 않는다.

진짜 힘은 따로 있다. 바이트 수준 BPE는 지구상의 모든 언어를 처리한다. 한자는 각각 3개의 UTF-8 바이트다. 일본어는 3~4 바이트일 수 있다. 아랍어, 데바나가리(Devanagari), 이모지도 전부 그저 바이트 시퀀스다. BPE 알고리즘은 영어 ASCII 바이트에서 패턴을 찾는 것과 정확히 같은 방식으로 이 바이트 시퀀스에서 패턴을 찾는다.

### 사전 토큰화 (Pre-Tokenization)

BPE가 텍스트를 건드리기 전에, 먼저 텍스트를 덩어리로 나눠야 한다. 이것은 병합 알고리즘이 단어 경계를 넘는 토큰을 만드는 것을 막는다.

GPT-2는 정규식 패턴을 사용해 텍스트를 나눈다:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

이 패턴은 축약형("don't"이 "don" + "'t"가 됨), 선택적 선행 공백이 있는 단어, 숫자, 구두점, 공백을 기준으로 나눈다. 선행 공백은 단어에 붙은 채로 유지된다. 그래서 "the cat"은 ["the", " ", "cat"]이 아니라 [" the", " cat"]이 된다.

Llama는 SentencePiece를 사용하는데, 이는 정규식을 완전히 건너뛴다. 원시 바이트 스트림을 하나의 긴 시퀀스로 취급하고 BPE 알고리즘이 경계를 알아내게 둔다. 이것은 더 간단하지만 BPE에게 단어를 넘나드는 토큰을 만들 더 많은 자유를 준다.

선택이 중요하다. GPT-2의 정규식은 토크나이저가 한 단어 끝의 "the"와 다음 단어 시작의 "the"가 병합되어야 한다고 학습하는 것을 막는다. SentencePiece는 그것을 허용하는데, 이는 때때로 더 효율적인 압축을 만들지만 덜 해석 가능한 토큰을 낳는다.

### 특수 토큰 (Special Tokens)

모든 프로덕션 토크나이저는 구조적 마커를 위해 토큰 ID를 예약한다:

| 토큰 | 목적 | 사용처 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 시퀀스 시작 | Llama 3, GPT |
| `[EOS]` / `</s>` | 시퀀스 끝 | 모든 모델 |
| `[PAD]` | 배치 정렬을 위한 패딩 | BERT, T5 |
| `[UNK]` | 알 수 없는 토큰 (바이트 수준 BPE는 이것을 제거함) | BERT, WordPiece |
| `<\|im_start\|>` | 채팅 메시지 경계 시작 | ChatGPT, Qwen |
| `<\|im_end\|>` | 채팅 메시지 경계 끝 | ChatGPT, Qwen |
| `<\|user\|>` | 사용자 차례 마커 | Llama 3 |
| `<\|assistant\|>` | 어시스턴트 차례 마커 | Llama 3 |

특수 토큰은 BPE에 의해 절대 나뉘지 않는다. 병합 알고리즘이 실행되기 전에 정확히 매칭되어 고정된 ID로 대체되고, 주변 텍스트는 정상적으로 토큰화된다.

### 채팅 템플릿 (Chat Templates)

이곳이 대부분의 사람이 혼란스러워하고 대부분의 구현이 깨지는 지점이다.

채팅 모델에 메시지를 보낼 때, API는 메시지 목록을 받는다:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

모델은 JSON을 보지 않는다. 평평한 토큰 시퀀스를 본다. 채팅 템플릿은 특수 토큰을 사용해 메시지를 그 평평한 시퀀스로 변환한다. 모든 모델이 이를 다르게 한다:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

템플릿을 틀리면 모델은 쓰레기를 만들어 낸다. 모델은 정확히 하나의 포맷으로 학습되었다. 빠진 줄바꿈, 바뀐 토큰, 여분의 공백처럼 어떤 일탈이든 입력을 학습 분포 밖으로 밀어낸다.

### 속도 (Speed)

Python은 프로덕션 토큰화에 너무 느리다.

tiktoken(OpenAI)은 Python 바인딩을 가진 Rust로 작성되었다. HuggingFace tokenizers도 Rust다. SentencePiece는 C++다. 이것들은 순수 Python 대비 10~100배의 속도 향상을 달성한다.

관점을 잡기 위해: Llama 3 사전 학습(pre-training)을 위해 15조 토큰을 초당 100만 토큰(빠른 Python)으로 토큰화하면 174일이 걸린다. 초당 1억 토큰(Rust)이면 1.7일이 걸린다.

여기서는 알고리즘을 이해하려고 Python으로 만든다. 프로덕션에서는 컴파일된 구현을 쓰고 Python 래퍼(wrapper)만 건드린다.

```figure
weight-tying
```

## 직접 만들기 (Build It)

### 1단계: 바이트 수준 인코딩

기초다. 임의의 문자열을 바이트 시퀀스로 변환하고, 표시를 위해 각 바이트를 출력 가능한 문자로 매핑하고, 그 과정을 역으로 돌린다.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

바이트 수를 보기 위해 다국어 텍스트에서 테스트해 보라:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello"는 5바이트다. "你好"는 6바이트다(문자당 3). 불 이모지는 4바이트다. 바이트 수준 토크나이저는 그것이 어떤 언어인지 신경 쓰지 않는다. 바이트는 바이트다.

### 2단계: 정규식을 사용한 사전 토크나이저

GPT-2 정규식 패턴을 사용해 텍스트를 덩어리로 나눈다. 각 덩어리는 BPE에 의해 독립적으로 토큰화된다.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 모듈은 유니코드 속성 이스케이프(`\p{L}`은 글자, `\p{N}`은 숫자)를 지원한다. 표준 라이브러리 `re` 모듈은 지원하지 않으므로, ASCII 문자 클래스로 되돌아간다. 프로덕션 다국어 토크나이저를 위해서는 `regex`를 설치하라.

시도해 보라:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

선행 공백은 단어에 붙은 채로 유지된다. 축약형은 아포스트로피에서 나뉜다. 구두점은 자신만의 덩어리가 된다. BPE는 이 경계를 넘어 토큰을 절대 병합하지 않는다.

### 3단계: 바이트 시퀀스 위의 BPE

Lesson 01의 핵심 알고리즘이지만, 이제 사전 토큰화된 덩어리를 독립적으로 처리한다.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 4단계: 특수 토큰 처리

특수 토큰은 정확한 매칭과 고정된 ID가 필요하다. 그것들은 BPE를 완전히 우회한다.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 5단계: 전체 토크나이저 클래스

모든 것을 함께 연결한다: 정규화, 특수 토큰 기준 분할, 사전 토큰화, BPE 병합, ID 매핑.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 6단계: 다국어 테스트

진짜 테스트다. 영어, 중국어, 이모지, 코드를 던져 보라.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

한자는 각각 3바이트를 만든다. 이모지는 4바이트를 만든다. 이 중 어느 것도 토크나이저를 깨뜨리지 않는다. 어느 것도 알 수 없는 토큰을 만들지 않는다. 그것이 바이트 수준 BPE의 힘이다.

## 라이브러리로 써보기 (Use It)

### 실제 토크나이저 비교

Llama 3, GPT-4, Mistral의 실제 토크나이저를 불러온다. 각각이 같은 다국어 문단을 어떻게 처리하는지 보라.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

같은 텍스트에 대해 서로 다른 토큰 수를 보게 된다. 128K 어휘를 가진 Llama 3는 흔한 패턴을 병합하는 데 더 공격적이다. 100K를 가진 GPT-4는 중간에 위치한다. 32K를 가진 Mistral은 더 많은 토큰을 만들지만 더 작은 임베딩(embedding) 층(layer)을 가진다.

트레이드오프(trade-off)는 언제나 동일하다: 더 큰 어휘는 더 짧은 시퀀스를 의미하지만 더 많은 파라미터(parameter)를 의미한다.

## 산출물 (Ship It)

이 레슨은 프로덕션 토크나이저를 만들고 디버깅하기 위한 프롬프트(prompt)를 만든다. `outputs/prompt-tokenizer-builder.md`를 보라.

## 연습 문제 (Exercises)

1. **쉬움:** 임의의 토큰 ID에 대한 원시 바이트를 보여 주는 `get_token_bytes(id)` 메서드를 추가하라. 그것을 사용해 가장 흔한 병합 토큰이 실제로 무엇을 나타내는지 살펴보라.
2. **중간:** 공백과 숫자에서 나누되 선행 공백을 유지하는 Llama 스타일 사전 토크나이저를 구현하라. 같은 말뭉치에서 그 어휘를 GPT-2 정규식 접근법과 비교하라.
3. **어려움:** `{"role": ..., "content": ...}` 메시지 목록을 받아 Llama 3 채팅 포맷에 맞는 올바른 토큰 시퀀스를 만드는 채팅 템플릿 메서드를 추가하라. HuggingFace 구현과 비교해 테스트하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 바이트 수준 BPE(Byte-level BPE) | "바이트 위에서 동작하는 토크나이저" | 256개의 바이트 값을 기본 어휘로 갖는 BPE -- 어떤 입력이든 알 수 없는 토큰 없이 처리한다 |
| 사전 토큰화(Pre-tokenization) | "BPE 전에 나누기" | BPE가 단어 경계를 넘어 병합하는 것을 막는 정규식 또는 규칙 기반 분할 |
| NFKC 정규화(NFKC normalization) | "유니코드 정리" | 정준 분해 후 호환성 합성 -- "fi" 합자가 "fi"가 되고, 전각 "Ａ"가 "A"가 된다 |
| 채팅 템플릿(Chat template) | "메시지가 토큰이 되는 방식" | role/content 메시지 목록을 평평한 토큰 시퀀스로 변환하는 정확한 포맷 -- 모델별로 다르며 학습 포맷과 일치해야 한다 |
| 특수 토큰(Special tokens) | "제어 토큰" | BPE를 우회하는 예약된 토큰 ID -- [BOS], [EOS], [PAD], 채팅 마커 -- 병합 전에 정확히 매칭된다 |
| 다산성(Fertility) | "단어당 토큰" | 출력 토큰 대 입력 단어의 비율 -- GPT-4의 영어는 1.3, 한국어는 2~3, 높을수록 컨텍스트를 낭비한다 |
| tiktoken | "OpenAI 토크나이저" | Python 바인딩을 가진 Rust BPE 구현 -- 순수 Python보다 10~100배 빠르다 |
| 병합 테이블(Merge table) | "어휘" | 학습 중 학습된 바이트 쌍 병합의 순서 있는 목록 -- 이것이 곧 토크나이저가 학습한 지식이다 |

## 더 읽을거리 (Further Reading)

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- GPT-3.5/4에서 사용되는 Rust BPE 구현
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- BPE, WordPiece, Unigram을 지원하는 Rust 토크나이저 라이브러리
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 128K 어휘와 토크나이저 학습에 대한 세부 사항
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- 언어 독립적 토큰화
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 원본 바이트-유니코드 매핑
