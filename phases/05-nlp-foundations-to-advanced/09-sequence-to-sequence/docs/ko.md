# 시퀀스 투 시퀀스 모델 (Sequence-to-Sequence Models)

> 번역기인 척하는 두 개의 RNN. 이들이 부딪히는 병목이 바로 어텐션이 존재하는 이유다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75분

## 문제 (The Problem)

분류(classification)는 가변 길이 시퀀스를 단일 레이블(label)로 매핑한다. 번역은 가변 길이 시퀀스를 또 다른 가변 길이 시퀀스로 매핑한다. 입력과 출력은 서로 다른 어휘(vocabulary), 어쩌면 서로 다른 언어에 속하며, 길이가 같다는 보장이 없다.

seq2seq 아키텍처(Sutskever, Vinyals, Le, 2014)는 의도적으로 단순한 레시피로 이를 풀었다. 두 개의 RNN. 하나는 원문 문장을 읽어 고정 크기 맥락 벡터(context vector)를 만든다. 다른 하나는 그 벡터를 읽어 목표 문장을 토큰(token) 단위로 생성한다. 레슨 08에서 작성한 것과 같은 코드를 다르게 이어 붙였을 뿐이다.

이를 공부할 가치는 두 가지다. 첫째, 맥락 벡터 병목은 NLP에서 교육적으로 가장 유용한 실패다. 어텐션(attention)과 트랜스포머(transformer)가 잘하는 모든 것을 촉발한다. 둘째, 학습 레시피(교사 강요(teacher forcing), 스케줄드 샘플링(scheduled sampling), 추론 시점의 빔 서치(beam search))는 LLM을 포함한 모든 현대 생성 시스템에 여전히 적용된다.

## 개념 (The Concept)

**인코더(Encoder).** 원문 문장을 읽는 RNN. 그 마지막 은닉 상태(hidden state)가 **맥락 벡터**(전체 입력의 고정 크기 요약) 다. 이론상으로는 원문에서 아무것도 잃지 않는다.

**디코더(Decoder).** 맥락 벡터로 초기화된 또 다른 RNN. 각 스텝에서 이전에 생성된 토큰을 입력으로 받아 목표 어휘에 대한 분포를 만든다. 샘플링이나 argmax로 다음 토큰을 고른다. 그것을 다시 넣는다. `<EOS>` 토큰이 나오거나 최대 길이에 도달할 때까지 반복한다.

**학습:** 각 디코더 스텝에서의 교차 엔트로피 손실(cross-entropy loss)을, 시퀀스에 걸쳐 합산. 두 신경망을 관통하는 표준 BPTT(backprop through time).

**교사 강요(Teacher forcing).** 학습 중, 스텝 `t`에서 디코더의 입력은 디코더 자신의 이전 예측이 아니라 위치 `t-1`의 *정답* 토큰이다. 이는 학습을 안정화한다. 그것이 없으면 초기 실수가 연쇄되어 모델이 끝내 학습하지 못한다. 추론 시점에는 모델 자신의 예측을 써야 하므로, 항상 학습/추론 분포 격차가 있다. 그 격차를 **노출 편향(exposure bias)**이라 한다.

**병목.** 인코더가 원문에 관해 학습한 모든 것이 그 하나의 맥락 벡터로 짜내어져야 한다. 긴 문장은 세부를 잃는다. 희귀 단어는 흐려진다. 재배열(chat noir 대 black cat)은 계산되는 게 아니라 외워져야 한다.

어텐션(레슨 10)은 디코더가 마지막 것만이 아니라 *모든* 인코더 은닉 상태를 보게 하여 이를 고친다. 그것이 핵심 요지 전부다.

```figure
lstm-gates
```

## 직접 만들기 (Build It)

### 1단계: 인코더

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs`는 `[batch, seq_len, hidden_dim]` 형태다. 입력 위치당 은닉 상태 하나. `hidden`은 `[1, batch, hidden_dim]` 형태다. 마지막 스텝. 레슨 08은 "분류를 위해 outputs에 걸쳐 풀링하라"고 했다. 여기서는 마지막 은닉 상태를 맥락 벡터로 보관하고 스텝별 출력은 무시한다.

### 2단계: 디코더

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

디코더는 한 번에 한 스텝씩 호출된다. 입력: 단일 토큰의 배치(batch)와 현재 은닉 상태. 출력: 다음 토큰에 대한 어휘 로짓(logits)과 갱신된 은닉 상태.

### 3단계: 교사 강요가 있는 학습 루프

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

짚어 둘 만한 손잡이가 두 개 있다. `ignore_index=0`은 패딩 토큰에 대한 손실을 건너뛴다. `teacher_forcing_ratio`는 각 스텝에서 모델의 예측 대신 참 토큰을 쓸 확률이다. 1.0(완전 교사 강요)에서 시작해 학습이 진행되는 동안 약 0.5로 점차 낮춰 노출 편향 격차를 좁힌다.

### 4단계: 추론 루프 (탐욕적)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

탐욕적 디코딩(greedy decoding)은 매 스텝에서 가장 높은 확률의 토큰을 고른다. 길을 잃을 수 있다. 일단 한 토큰에 손을 대면 되돌릴 수 없다. **빔 서치(beam search)**는 상위 `k`개의 부분 시퀀스를 살려 두고 끝에서 가장 높은 점수의 완성 시퀀스를 고른다. 빔 너비 3-5가 표준이다.

### 5단계: 시연으로 보는 병목

장난감 복사 과제로 모델을 학습시킨다: 원문 `[a, b, c, d, e]`, 목표 `[a, b, c, d, e]`. 시퀀스 길이를 늘린다. 정확도(accuracy)를 관찰한다.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

단일 GRU 은닉 상태는 40토큰 입력을 무손실로 외울 수 없다. 정보는 모든 인코더 스텝에 있지만, 디코더는 마지막 상태만 본다. 어텐션이 이를 직접 고친다.

## 라이브러리로 써보기 (Use It)

PyTorch에는 `nn.Transformer`와 `nn.LSTM` 기반 seq2seq 템플릿이 있다. Hugging Face의 `transformers` 라이브러리는 수십억 토큰으로 학습된 완전한 인코더-디코더 모델(BART, T5, mBART, NLLB)을 제공한다.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

현대 인코더-디코더는 RNN을 버리고 트랜스포머를 택했다. 큰 그림의 모양(인코더, 디코더, 토큰 단위 생성)은 2014년 seq2seq 논문과 동일하다. 각 블록 내부의 메커니즘이 다를 뿐이다.

### 여전히 RNN 기반 seq2seq로 손을 뻗을 때

새 프로젝트에는 거의 없다. 구체적 예외:

- 유계 메모리로 입력을 한 번에 토큰 하나씩 소비하는 스트리밍 번역.
- 트랜스포머 메모리 비용이 감당 안 되는 온디바이스 텍스트 생성.
- 교육. 인코더-디코더 병목을 이해하는 것이 왜 트랜스포머가 이겼는지 이해하는 가장 빠른 길이다.

### 노출 편향과 그 완화책

- **스케줄드 샘플링.** 학습 중 교사 강요 비율을 점차 낮춰, 모델이 자신의 실수에서 회복하는 법을 배우게 한다.
- **최소 위험 학습(minimum risk training).** 토큰 단위 교차 엔트로피 대신 문장 단위 BLEU 점수로 학습. 실제로 원하는 것에 더 가깝다.
- **강화 학습 파인튜닝.** 시퀀스 생성기에 지표로 보상을 준다. 현대 LLM RLHF에 쓰인다.

세 가지 모두 트랜스포머 기반 생성에도 여전히 적용된다.

## 산출물 (Ship It)

`outputs/prompt-seq2seq-design.md`로 저장한다:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 연습 문제 (Exercises)

1. **쉬움.** 장난감 복사 과제를 구현하라. 목표가 원문과 같은 입력-출력 쌍에서 GRU seq2seq를 학습하라. 길이 5, 10, 20에서 정확도를 측정하라. 병목을 재현하라.
2. **보통.** 빔 너비 3의 빔 서치 디코딩을 추가하라. 작은 병렬 코퍼스에서 탐욕적 대비 BLEU를 측정하라. 빔 서치가 이기는 곳(보통 마지막 토큰)과 차이가 없는 곳을 문서화하라.
3. **어려움.** 1만 쌍의 패러프레이즈(paraphrase) 데이터셋(dataset)에서 `facebook/bart-base`를 파인튜닝(fine-tuning)하라. 떼어 둔 입력에서 파인튜닝된 모델의 빔-4 출력을 베이스(base) 모델의 것과 비교하라. BLEU를 보고하고 정성적 예시 10개를 골라라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 인코더(Encoder) | 입력 RNN | 원문을 읽음. 스텝별 은닉 상태와 마지막 맥락 벡터를 만듦. |
| 디코더(Decoder) | 출력 RNN | 맥락 벡터로 초기화됨. 목표 토큰을 한 번에 하나씩 생성. |
| 맥락 벡터(Context vector) | 요약 | 마지막 인코더 은닉 상태. 고정 크기. 어텐션이 해결하는 병목. |
| 교사 강요(Teacher forcing) | 참 토큰 사용 | 학습 시점에 정답 이전 토큰을 넣음. 학습을 안정화. |
| 노출 편향(Exposure bias) | 학습/테스트 격차 | 참 토큰으로 학습된 모델은 자신의 실수에서 회복하는 연습을 한 적이 없음. |
| 빔 서치(Beam search) | 더 나은 디코딩 | 탐욕적으로 손대는 대신 각 스텝에서 상위 k개의 부분 시퀀스를 살려 둠. |

## 더 읽을거리 (Further Reading)

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215): 원본 seq2seq 논문. 네 쪽.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078): GRU와 인코더-디코더 구도를 도입.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473): 어텐션 논문. 이 레슨 직후에 읽어라.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html): 직접 만들 수 있는 seq2seq + 어텐션 코드.
