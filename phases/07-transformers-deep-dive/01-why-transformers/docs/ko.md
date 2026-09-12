# 왜 트랜스포머인가(Why Transformers): RNN의 문제들

> RNN은 토큰(token)을 한 번에 하나씩 처리한다. 트랜스포머(transformer)는 모든 토큰을 한꺼번에 처리한다. 이 하나의 아키텍처적 베팅이 2017년 이후 딥러닝의 모든 스케일링 곡선을 바꿨다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45분

## 문제 (The Problem)

2017년 이전에는 지구상의 모든 최신 시퀀스 모델(언어, 번역, 음성)이 순환 신경망(recurrent neural network)이었다. LSTM과 GRU가 반세기 가까이 ImageNet에 견줄 만한 번역 벤치마크(benchmark)에서 우승했다. 누구에게나 그것이 유일한 도구였다.

여기에는 세 가지 치명적 약점이 있었다. 순차적 계산은 시간 축을 따라 병렬화할 수 없었다. 토큰 `t+1`은 토큰 `t`의 은닉 상태(hidden state)가 필요하다. 1,024 토큰 시퀀스는 사이클당 1,000,000번의 부동소수점 연산을 하는 GPU에서도 1,024번의 직렬 스텝을 거쳐야 했다. 병렬성을 위해 설계된 하드웨어에서 학습(training) 실제 시계 시간이 시퀀스 길이에 선형으로 비례한 것이다.

기울기 소실(vanishing gradient) 탓에 50 토큰 뒤의 정보는 이미 50개의 비선형성을 거쳐 압축돼 버린다. 게이트 순환 유닛(LSTM, GRU)이 그 압착을 완화했지만 결코 없애지는 못했다. 그래서 장거리 의존성("내가 지난여름 교토행 비행기에서 읽은 그 책은…")은 일상적으로 실패했다.

고정 폭 은닉 상태에서는 인코더(encoder)가 디코더(decoder)가 무언가를 보기도 전에 전체 소스 시퀀스를 단일 벡터(vector)로 짜 넣어야 했다. 소스가 5 토큰이든 500 토큰이든 상관없다. 병목(bottleneck)은 같은 모양이다.

2017년 논문 "Attention Is All You Need"는 급진적인 것을 제안했다. 순환(recurrence)을 완전히 버려라. 모든 위치가 다른 모든 위치에 병렬로 어텐션(attention)하게 하라. 1,024번의 순차 연산 대신 하나의 큰 행렬 곱(matrix multiplication)으로 학습하라.

그 결과 2026년까지 트랜스포머가 모든 모달리티를 지배한다. 언어(GPT-5, Claude 4, Llama 4), 비전(ViT, DINOv2, SAM 3), 오디오(Whisper), 생물학(AlphaFold 3), 로보틱스(RT-2). 같은 블록, 다른 입력.

## 개념 (The Concept)

![RNN 순차 계산 vs 트랜스포머 병렬 어텐션](../assets/rnn-vs-transformer.svg)

**병목으로서의 순환.** RNN은 `h_t = f(h_{t-1}, x_t)`를 계산한다. 각 스텝은 이전 것에 의존한다. `h_4`보다 먼저 `h_5`를 계산할 수 없다. 10,000개 이상의 병렬 코어를 가진 현대 GPU에서, 이는 긴 시퀀스에 대해 실리콘의 99%를 낭비한다.

**브로드캐스트로서의 어텐션.** 셀프 어텐션(self-attention)은 모든 쌍 `(i, j)`에 대해 `output_i = sum_j(a_ij * v_j)`를 동시에 계산한다. 전체 N×N 어텐션 행렬이 하나의 배치 행렬곱(matmul)으로 채워진다. 어떤 스텝도 다른 것에 의존하지 않는다. GPU가 이를 좋아한다.

**속도 향상은 상수가 아니다.** 그것은 `O(N)` 직렬 깊이와 `O(1)` 직렬 깊이의 차이다. 실제로 트랜스포머는 N=512에서 동일 하드웨어 기준 에폭(epoch)당 5-10배 빠르게 학습하고, 어텐션의 `O(N²)` 메모리 장벽(나중에 Flash Attention이 고쳤다. 레슨 12 참조)에 부딪히기 전까지 시퀀스 길이와 함께 격차가 벌어진다.

**트랜스포머의 비용.** 어텐션 메모리는 `O(N²)`로 비례한다. 2K 컨텍스트에는 괜찮다. 128K 컨텍스트에는 슬라이딩 윈도우, RoPE 외삽, Flash Attention 타일링, 또는 선형 어텐션 변형이 필요하다. 순환은 시간과 메모리 둘 다 `O(N)`이었다. 트랜스포머는 메모리로 시간을 맞바꾼 뒤 병렬성으로 시간을 되찾는다.

**귀납적 편향(inductive bias) 전환.** RNN은 국소성과 최근성을 가정한다. 트랜스포머는 아무것도 가정하지 않는다. 모든 쌍이 어텐션의 후보다. 그래서 트랜스포머는 잘 학습하려면 더 많은 데이터가 필요하지만, 일단 데이터가 있으면 더 멀리 확장된다. Chinchilla(2022)가 이를 공식화했다. 충분한 토큰이 주어지면, 트랜스포머는 항상 동일 파라미터(parameter) 수의 RNN을 이긴다.

```figure
rnn-vs-parallel
```

## 직접 만들기 (Build It)

여기에 신경망(neural network)은 없다. 노트북에서 격차를 느낄 수 있도록 핵심 병목을 수치적으로 시뮬레이션한다.

### 1단계: 직렬 깊이 측정하기

`code/main.py`를 참조하라. 두 함수를 만든다. 하나는 시퀀스를 덧셈의 연쇄로 인코딩한다(직렬, RNN처럼). 하나는 병렬 리덕션(reduction)으로 인코딩한다(브로드캐스트, 어텐션처럼). 같은 수학, 다른 의존성 그래프.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

최대 100,000개 원소의 시퀀스에서 둘 다 시간을 잰다. RNN 버전은 O(N)이고 단일 CPU 파이프라인(pipeline)이다. 순수 파이썬에서조차, 어텐션 스타일 리덕션은 길이 ≥ 1,000에서 이를 이긴다. 파이썬의 `sum()`이 C로 구현되어 스텝마다 인터프리터 오버헤드 없이 반복하기 때문이다.

### 2단계: 이론적 연산 수 세기

두 알고리즘 모두 N번의 덧셈을 한다. 차이는 *의존성 깊이*다. 다음이 시작되기 전에 순차적으로 일어나야 하는 연산이 몇 개인가. RNN 깊이 = N. 어텐션 깊이 = 트리 리덕션이면 log(N), 병렬 스캔이면 1. 연산 수가 아니라 깊이가 GPU 시간을 결정한다.

### 3단계: 긴 시퀀스에서의 실증적 스케일링

O(N) 격차를 가시화하는 타이밍 표를 출력한다. 2026년 Mac 노트북에서, 1,000개 미만 원소의 시퀀스는 측정하기에 너무 빠르다. 100,000개 시퀀스는 깔끔한 선형 스캔을 보여준다. 그것을 12층 LSTM에 상응하는 16,384 토큰 트랜스포머로 확장하면, 2016년에 왜 학습 실제 시계 시간이 걸림돌이었는지 알 수 있다.

## 라이브러리로 써보기 (Use It)

2026년에 여전히 RNN을 고를 때:

| 상황 | 선택 |
|-----------|------|
| 스트리밍 추론, 한 번에 한 토큰, 일정한 메모리 | RNN 또는 상태 공간 모델 (Mamba, RWKV) |
| 어텐션 메모리가 폭발하는 매우 긴 시퀀스 (>1M 토큰) | 선형 어텐션, Mamba 2, Hyena |
| matmul 가속기가 없는 엣지 디바이스 | Depthwise-separable RNN이 여전히 FLOPs/watt에서 이김 |
| 그 외 모든 것 (학습, 배치 추론, 128K까지의 컨텍스트) | 트랜스포머 |

Mamba 같은 상태 공간 모델(state-space model, SSM)은 본질적으로 양쪽의 장점을 주는 구조적 파라미터화를 갖춘 RNN이다. `O(N)` 스캔 메모리, 선택적 스캔을 통한 병렬 학습. 이들은 트랜스포머 품질의 90%를 회복하면서 더 나은 장기 컨텍스트 스케일링을 보인다. 2026년에 대부분의 프런티어 연구소는 하이브리드 SSM+트랜스포머 모델(예: Jamba, Samba)을 학습한다. 순환은 죽지 않았고, 하나의 구성요소다.

## 산출물 (Ship It)

`outputs/skill-architecture-picker.md`를 참조하라. 이 스킬은 길이, 처리량(throughput), 학습 예산 제약이 주어지면 새 시퀀스 문제에 맞는 아키텍처를 고른다. 트레이드오프(trade-off)를 명시하지 않고서는 10억 토큰을 넘는 학습 실행에 순수 RNN을 추천하기를 항상 거부해야 한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`에서 `rnn_style`을 가져와 스칼라 은닉 상태를 길이 64의 은닉 상태 벡터로 교체한다. 다시 측정한다. 직렬 오버헤드가 은닉 상태 차원에 따라 얼마나 커지는가?
2. **보통.** 순수 파이썬으로 병렬 접두사 합(prefix-sum, Hillis-Steele 스캔)을 구현한다. 길이 1024에서 직렬 스캔과 동일한 수치 출력을 내는지 검증한다. 깊이를 센다.
3. **어려움.** 어텐션 스타일 리덕션을 GPU의 PyTorch로 포팅한다. 시퀀스 길이를 64에서 65,536까지 훑으며 둘 다 시간을 잰다. 곡선 모양을 플롯하고 설명한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Recurrence | "RNN은 순차적이다" | 스텝 `t`가 스텝 `t-1`에 의존하여 시간 축을 따라 직렬 실행을 강제하는 계산. |
| Serial depth | "그래프가 얼마나 깊은가" | 의존하는 연산의 가장 긴 연쇄. 무한 하드웨어에서도 실제 시계 시간을 제한함. |
| Attention | "토큰끼리 서로 보게 하기" | 가중합 `sum_j a_ij v_j`. 여기서 `a_ij`는 위치 i와 j 사이의 유사도 점수에서 나옴. |
| Context window | "모델이 얼마나 보는가" | 어텐션 층(layer)이 입력으로 받을 수 있는 위치의 수. 2차 메모리 비용이 여기서 비례함. |
| Inductive bias | "아키텍처에 새겨진 가정" | 데이터가 어떻게 생겼는지에 대한 사전 가정. CNN은 평행이동 불변성을, RNN은 최근성을 가정함. |
| State-space model | "대수를 뒤에 둔 RNN" | 구조적 상태 공간 행렬을 통한 병렬 학습을 위해 파라미터화된 순환. |
| Quadratic bottleneck | "왜 컨텍스트가 그렇게 비싼가" | 어텐션 메모리 = 시퀀스 길이의 `O(N²)`. Flash Attention은 상수를 숨길 뿐 스케일링은 숨기지 못함. |

## 더 읽을거리 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762): 주류 NLP에서 순환을 죽인 논문.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473): 어텐션이 RNN에 볼트로 붙어 탄생한 곳.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf): 기록을 위한 원조 LSTM 논문.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752): 트랜스포머에 대한 현대적 순환 응답.
