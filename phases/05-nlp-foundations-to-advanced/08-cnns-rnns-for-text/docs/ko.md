# 텍스트를 위한 CNN과 RNN (CNNs and RNNs for Text)

> 합성곱은 n-그램을 학습한다. 순환은 기억한다. 둘 다 어텐션으로 대체되었다. 둘 다 제약된 하드웨어에서는 여전히 중요하다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75분

## 문제 (The Problem)

TF-IDF와 Word2Vec은 단어 순서를 무시하는 평평한 벡터(vector)를 만들었다. 이들 위에 세운 분류기(classifier)는 `dog bites man`과 `man bites dog`를 구별할 수 없었다. 단어 순서가 때로는 신호를 담는다.

트랜스포머(transformer)가 등장하기 전, 두 계열의 아키텍처가 그 빈틈을 메웠다.

**텍스트를 위한 합성곱 신경망(TextCNN).** 단어 임베딩(embedding) 시퀀스에 1차원 합성곱(convolution)을 적용한다. 너비 3의 필터는 학습 가능한 트라이그램 탐지기다. 세 단어에 걸쳐 점수를 출력한다. 서로 다른 너비(2, 3, 4, 5)를 쌓아 다중 스케일 패턴을 탐지한다. 고정 크기 표현으로 맥스 풀링(max-pool)한다. 평평하고, 병렬적이며, 빠르다.

**순환 신경망(RNN, LSTM, GRU).** 토큰(token)을 한 번에 하나씩 처리하며, 정보를 앞으로 운반하는 은닉 상태(hidden state)를 유지한다. 순차적이고, 기억을 지니며, 입력 길이가 유연하다. 2014년부터 2017년까지 시퀀스 모델링을 지배했고, 그다음 어텐션(attention)이 등장했다.

이 레슨은 둘 다 만든 다음, 어텐션을 촉발한 실패를 짚는다.

## 개념 (The Concept)

**TextCNN** (Kim, 2014). 토큰이 임베딩된다. 너비-`k`의 1차원 합성곱이 연속된 `k`-그램 임베딩 위로 필터를 미끄러뜨려 특성 맵(feature map)을 만든다. 그 맵에 대한 전역 맥스 풀링이 가장 강한 활성값(activation)을 고른다. 여러 필터 너비의 맥스 풀링된 출력을 연결한다. 분류기 헤드에 넣는다.

동작 이유. 필터는 학습 가능한 n-그램이다. 맥스 풀링은 위치 불변이라, "not good"은 리뷰의 처음에 있든 중간에 있든 같은 특성을 발화시킨다. 각각 100개의 필터를 가진 세 가지 필터 너비는 300개의 학습된 n-그램 탐지기를 준다. 학습은 병렬적이다. 순차 의존성이 없다.

**RNN.** 각 시간 스텝 `t`에서 은닉 상태 `h_t = f(W * x_t + U * h_{t-1} + b)`. `W`, `U`, `b`를 시간에 걸쳐 공유한다. 시간 `T`의 은닉 상태는 전체 접두부의 요약이다. 분류에는 `h_1 ... h_T`에 걸쳐 풀링한다(맥스, 평균, 또는 마지막).

순수 RNN은 기울기 소실(vanishing gradient)을 겪는다. **LSTM**은 무엇을 잊을지, 무엇을 저장할지, 무엇을 출력할지 결정하는 게이트를 더해, 긴 시퀀스 전반에서 그래디언트(gradient)를 안정화한다. **GRU**는 LSTM을 두 개의 게이트로 단순화한다. 더 적은 파라미터(parameter)로 비슷하게 동작한다.

**양방향 RNN(Bidirectional RNN)**은 하나의 RNN을 순방향으로, 다른 하나를 역방향으로 돌려 은닉 상태를 연결한다. 모든 토큰의 표현이 좌우 맥락을 모두 본다. 태깅 과제에 필수적이다.

```figure
rnn-unroll
```

## 직접 만들기 (Build It)

### 1단계: PyTorch의 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)`는 `[batch, seq_len, embed_dim]`을 `[batch, embed_dim, seq_len]`으로 재배열하는데, `nn.Conv1d`가 가운데 축을 채널로 취급하기 때문이다. 풀링된 출력은 입력 길이와 무관하게 고정 크기다.

### 2단계: LSTM 분류기

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

마지막 상태 풀링이 아니라 시퀀스에 대한 맥스 풀링을 한다. 분류에는 보통 맥스 풀링이 마지막 은닉 상태를 취하는 것보다 낫다. 긴 시퀀스의 끝에 있는 정보가 마지막 상태를 지배하는 경향이 있기 때문이다.

### 3단계: 기울기 소실 데모 (직관)

게이트가 없는 순수 RNN은 장거리 의존성을 학습할 수 없다. 장난감 과제를 보자: 토큰 `A`가 시퀀스 어디든 나타났는지 예측하기. `A`가 위치 1에 있고 시퀀스가 100토큰 길이라면, 손실(loss)에서 나온 그래디언트는 순환 가중치(weight)를 99번 곱하며 거꾸로 거슬러 가야 한다. 가중치가 1보다 작으면 그래디언트가 소실된다. 1보다 크면 폭발한다.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM은 가산적 상호작용만으로 신경망을 관통하는 **셀 상태(cell state)**로 이를 고친다(망각 게이트가 그것을 곱셈적으로 스케일하지만, 그래디언트는 여전히 "고속도로"를 따라 흐른다). GRU는 더 적은 파라미터로 비슷한 일을 한다. 둘 다 100+ 스텝 시퀀스 전반에서 안정적인 학습을 준다.

### 4단계: 이것이 여전히 충분하지 않았던 이유

LSTM이 있어도 세 가지 문제가 남았다.

1. **순차적 병목.** 길이 1000 시퀀스에서 RNN을 학습하려면 1000번의 직렬 순방향/역방향 스텝이 필요하다. 시간에 걸쳐 병렬화할 수 없다.
2. **인코더-디코더 설정에서의 고정 크기 맥락 벡터.** 디코더(decoder)는 전체 입력에 걸쳐 압축된, 인코더(encoder)의 마지막 은닉 상태만 본다. 긴 입력은 세부를 잃는다. 레슨 09가 이것을 직접 다룬다.
3. **원거리 의존성 정확도 상한.** LSTM은 순수 RNN보다 낫지만 200+ 스텝에 걸쳐 특정 정보를 전파하는 데는 여전히 애를 먹는다.

어텐션이 셋 모두를 해결했다. 트랜스포머는 순환을 완전히 버렸다. 레슨 10이 그 전환점이다.

## 라이브러리로 써보기 (Use It)

PyTorch의 `nn.LSTM`, `nn.GRU`, `nn.Conv1d`는 프로덕션(production) 준비가 되어 있다. 학습 코드는 표준적이다.

Hugging Face는 입력층으로 꽂아 넣는 사전 학습 임베딩을 제공한다:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

제약에 맞을 때 쓰기 체크리스트.

- **엣지 / 온디바이스 추론(inference).** GloVe 임베딩을 쓴 TextCNN은 트랜스포머보다 10-100배 작다. 배포 대상이 휴대폰이라면 이것이 그 스택이다.
- **스트리밍 / 온라인 분류.** RNN은 한 번에 토큰 하나를 처리하지만 트랜스포머는 전체 시퀀스가 필요하다. 실시간 들어오는 텍스트에는 LSTM이 여전히 이긴다.
- **베이스라인(baseline)용 작은 모델.** 새 과제에서의 빠른 반복. CPU에서 5분 만에 TextCNN을 학습한다.
- **제한된 데이터의 시퀀스 레이블링.** BiLSTM-CRF(레슨 06)는 1천-1만 개의 레이블된 문장에 대해 여전히 프로덕션 등급 NER 아키텍처다.

그 밖의 모든 것은 트랜스포머로 간다.

## 산출물 (Ship It)

`outputs/prompt-text-encoder-picker.md`로 저장한다:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 연습 문제 (Exercises)

1. **쉬움.** 3-클래스 장난감 데이터셋(dataset, 데이터는 직접 만든다)에서 TextCNN을 학습하라. 필터 너비 (2, 3, 4)가 단일 너비 (3)보다 평균 F1에서 더 나은지 확인하라.
2. **보통.** LSTM 분류기에 대해 맥스 풀, 평균 풀, 마지막 상태 풀링을 구현하라. 작은 데이터셋에서 비교하라. 어떤 풀링이 이기는지 문서화하고 그 이유를 가설로 세워라.
3. **어려움.** BiLSTM-CRF NER 태거를 만들어라(레슨 06과 이번 것을 결합). CoNLL-2003에서 학습하라. 레슨 06의 CRF 단독 베이스라인과 BERT 파인튜닝(fine-tuning)과 비교하라. 학습 시간, 메모리, F1을 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| TextCNN | 텍스트용 CNN | 단어 임베딩 위에 전역 맥스 풀을 둔 1차원 합성곱의 스택. Kim (2014). |
| RNN | 순환 신경망 | 각 시간 스텝에서 갱신되는 은닉 상태: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | 게이트가 있는 RNN | 입력 / 망각 / 출력 게이트 + 셀 상태를 더함. 긴 시퀀스 전반에서 안정적으로 학습. |
| GRU | 더 단순한 LSTM | 세 개 대신 두 개의 게이트. 비슷한 정확도, 더 적은 파라미터. |
| 양방향(Bidirectional) | 양쪽 방향 | 순방향 + 역방향 RNN을 연결. 모든 토큰이 맥락의 양쪽을 봄. |
| 기울기 소실(Vanishing gradient) | 학습 신호가 죽음 | 순수 RNN에서 1보다 작은 가중치의 반복 곱셈이 초기 스텝 그래디언트를 사실상 0으로 만듦. |

## 더 읽을거리 (Further Reading)

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882): TextCNN 논문. 여덟 쪽. 읽기 쉽다.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf): LSTM 논문. 뜻밖에 명료하다.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/): LSTM을 모두가 접근할 수 있게 만든 다이어그램들.
