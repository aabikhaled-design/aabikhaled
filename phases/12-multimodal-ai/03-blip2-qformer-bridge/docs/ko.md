# CLIP에서 BLIP-2로(모달리티 다리로서의 Q-Former(From CLIP to BLIP-2) Q-Former as Modality Bridge)

> CLIP은 이미지와 텍스트를 정렬하지만, 캡션을 생성하거나, 질문에 답하거나, 대화를 이어갈 수는 없다. BLIP-2(Salesforce, 2023)는 작은 학습 가능한 다리로 이것을 풀었다. 32개의 학습 가능한 쿼리(query) 벡터가 교차 어텐션(cross-attention)을 통해 동결된 ViT의 특성에 주목한 다음, 동결된 LLM의 입력 스트림에 곧장 끼워진다. 1억 8800만 개의 다리 파라미터(parameter)가 110억 개짜리 LLM을 ViT-g/14에 연결했다. 2026년까지의 모든 어댑터(adapter) 기반 VLM(MiniGPT-4, InstructBLIP, LLaVA의 사촌들)은 그 후손이다. 이 레슨은 Q-Former의 아키텍처를 읽고, 그 2단계 학습을 설명하며, 시각 토큰(visual token)을 동결된 텍스트 디코더(decoder)에 먹이는 장난감 버전을 만든다.

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180분

## 학습 목표 (Learning Objectives)

- 동결된 비전 인코더(encoder)와 동결된 LLM 사이의 학습 가능한 병목(bottleneck)이 비용과 안정성 면에서 종단간(end-to-end) 파인튜닝(fine-tuning)을 이기는 이유 설명하기.
- 고정된 학습 가능 쿼리 집합이 외부 이미지 특성에 주목하는 교차 어텐션 블록 구현하기.
- BLIP-2의 2단계 사전 학습 따라가기: 표현 학습(ITC + ITM + ITG) 그다음 생성(동결 디코더로 LM 손실).
- Q-Former를 LLaVA가 쓰는 더 단순한 MLP 투영기(projector)와 비교하고, 각 선택이 언제 이기는지 논증하기.

## 문제 (The Problem)

이미지당 차원 1408의 패치 토큰(patch token) 256개를 만드는 동결된 ViT가 있다고 하자. 차원 4096의 토큰 임베딩(embedding)을 기대하는 동결된 7B LLM도 있다. 명백한 다리(1408에서 4096으로 가는 선형 층)은 동작하지만, 256개 패치 토큰 전부를 LLM의 컨텍스트(context)에 넣으면 이미지당 256개의 추가 토큰이 든다. 32장의 이미지 배치(batch)라면 시각 모달리티(modality)만으로 8192개의 토큰을 쓰는 셈이다.

BLIP-2의 질문: 256토큰 이미지 표현을, LLM이 이미지를 캡션하고, 질문에 답하고, 추론할 만큼의 정보를 보존하면서 훨씬 더 적은 토큰(가령 32개)으로 압축할 수 있는가? 그리고 동결된 백본(backbone)을 건드리지 않고 이 다리를 학습시켜, 학습 비용을 다리의 파라미터만으로 유지할 수 있는가?

답: Q-Former. ViT의 패치 토큰에 교차 어텐션을 하는 32개의 학습 가능한 "쿼리" 벡터가, LLM이 소비하는 32토큰 시각 요약을 만들어 낸다. 총 1억 8800만 파라미터. LLM을 건드리기 전에 대조(contrastive), 매칭, 생성 목적으로 학습된다.

## 개념 (The Concept)

### 학습 가능한 쿼리 (Learnable queries)

Q-Former의 핵심 비결: LLM의 텍스트 토큰이 이미지 패치에 주목하게 하는 대신, 32개의 학습 가능한 쿼리 벡터 `Q`라는 새 집합을 도입하고 *그것들이* 이미지 패치에 주목하게 한다. 쿼리는 모델의 파라미터다. 학습 중에 학습되며, 모든 이미지에 동일한 32개 쿼리가 쓰인다.

교차 어텐션 이후, 각 쿼리는 이미지의 압축된 요약을 담는다. "주요 객체를 묘사하라", "배경을 묘사하라", "객체를 세어라" 등. 쿼리가 문자 그대로 의미 레이블(label)에 특화되는 것은 아니다. 다운스트림 손실(loss)을 떨어뜨리는 인코딩이라면 무엇이든 학습한다.

### 아키텍처 (Architecture)

Q-Former는 두 개의 경로를 가진 작은 트랜스포머(transformer)다(12층, 약 1억 파라미터):

1. 쿼리 경로: 32개 쿼리 벡터가 (자기들끼리의) 셀프 어텐션(self-attention)을 거치고, 그다음 동결된 ViT의 패치 토큰에 대한 교차 어텐션을 거치고, 그다음 FFN을 거친다.
2. 텍스트 경로: BERT 같은 텍스트 인코더가 쿼리 경로와 셀프 어텐션 및 FFN 가중치(weight)를 공유한다. 텍스트 경로에서는 교차 어텐션이 비활성화된다.

학습 시에는 두 경로 모두 실행된다. 쿼리와 텍스트는 공유된 셀프 어텐션을 통해 상호작용하는데, 이는 필요한 작업(ITM, ITG)에서 쿼리가 텍스트에 조건화될 수 있음을 의미한다. VLM 넘겨주기를 위한 추론 시에는 쿼리만 통과하여 32개의 시각 토큰을 산출한다.

### 2단계 학습 (Two-stage training)

BLIP-2는 두 단계로 사전 학습한다:

1단계: 표현 학습(LLM 없음). 세 가지 손실:
- ITC(image-text contrastive): 풀링된 쿼리 토큰과 텍스트 CLS 토큰 사이의 CLIP 스타일 대조.
- ITM(image-text matching): 이진 분류기: 이 이미지-텍스트 쌍은 매칭되는가? 어려운 음성(hard-negative)이 마이닝된다.
- ITG(image-grounded text generation): 쿼리에 조건화된, 텍스트에 대한 인과(causal) LM 헤드. 쿼리가 텍스트로 생성 가능한 내용을 인코딩하도록 강제한다.

Q-Former만 학습한다. ViT는 동결된다. LLM은 관여하지 않는다.

2단계: 생성 학습. 동결된 LLM(OPT-2.7B 또는 Flan-T5-XL 등)을 붙인다. 작은 선형 층을 통해 32개 쿼리 출력을 LLM의 임베딩 차원으로 투영한다. 그것들을 텍스트 프롬프트(prompt) 앞에 붙인다. 연결된 프롬프트 + 이미지 + 캡션 시퀀스에 대한 LM 손실로 선형 투영과 Q-Former만 학습한다.

2단계 이후, Q-Former + 투영이 전체 시각 어댑터다. 추론 시: 이미지 → ViT → Q-Former → 선형 투영 → 텍스트 앞에 붙임 → 동결된 LLM이 출력을 내보낸다.

### 파라미터 경제학 (Parameter economics)

ViT-g/14(11억, 동결) + OPT-6.7B(67억, 동결) + Q-Former(1억 8800만, 학습)를 쓰는 BLIP-2 = 총 80억, 학습 1억 8800만. Q-Former만으로는 전체 스택 파라미터의 약 2.4%다. 학습 비용이 이를 반영한다. 종단간 학습에 몇 주가 걸리는 데 비해, 소수의 A100에서 며칠이다.

품질: BLIP-2는 50배 더 작으면서 제로샷 VQA에서 Flamingo-80B와 동등하거나 그것을 이긴다. 다리가 작동한다.

### InstructBLIP과 명령어 인식 Q-Former (InstructBLIP and the instruction-aware Q-Former)

InstructBLIP(2023)은 추가 입력으로 Q-Former를 확장한다: 명령어 텍스트 그 자체다. 교차 어텐션 시, 이제 쿼리는 이미지 패치와 명령어 양쪽에 접근한다. 쿼리는 하나의 고정된 요약을 학습하는 대신 명령어별로 특화될 수 있다("차를 세어라", "분위기를 묘사하라"). 미공개(held-out) 작업에서 벤치마크(benchmark) 향상.

### MiniGPT-4와 투영기만 쓰는 접근법 (MiniGPT-4 and the projector-only approach)

MiniGPT-4는 Q-Former는 유지하되 나머지 전부를 동결하고 출력 선형 투영만 학습했다. 저렴하지만 그 대가는 품질이다. 쿼리가 내가 학습시킨 것이 아니라 BLIP-2의 것이었기 때문이다. 빠른 반복에는 좋지만, 최선의 아키텍처는 아니다.

### LLaVA가 더 단순해진 이유 (Why LLaVA went simpler)

LLaVA(2023, Lesson 12.05)는 Q-Former를 모든 ViT 패치 토큰을 LLM 공간으로 투영하는 평범한 2층 MLP로 대체했다. 24x24 격자에 대해 이미지당 576개 토큰, 전부 LLM에 먹인다. 압축은 더 나쁘지만 LLM이 원시 패치에 주목할 수 있게 한다. 당시 이것은 논란거리였지만, 2023년 말에는 시각 명령어 데이터(LLaVA-Instruct-150k)가 MLP를 충분한 신호를 보존하도록 학습시킬 수 있음을 증명하면서 지배적이 되었다. 트레이드오프(trade-off): LLaVA의 컨텍스트는 더 빨리 차지만, 멀티 이미지와 비디오로 자연스럽게 스케일링된다.

2026년 무렵 분야는 갈렸다. Q-Former는 토큰 예산(token budget)이 중요한 곳(긴 비디오, 많은 이미지)에서 살아남고, MLP 투영기는 토큰당 원시 품질이 우선인 곳에서 지배한다.

### 게이트 교차 어텐션: 조상 Flamingo (Gated cross-attention: Flamingo, the ancestor)

Flamingo(Lesson 12.04)는 BLIP-2보다 앞섰고 같은 교차 어텐션 아이디어를 썼지만, 단일 다리로서가 아니라 동결된 LLM의 모든 층에서 썼다. BLIP-2는 입력 층에만 압축해도 여전히 작동함을 보였다. Gemini와 Idefics는 둘을 결합한다: 인터리브된(interleaved) 입력 토큰에 더해, 인컨텍스트(in-context) 퓨샷(few-shot)을 위한 선택적 게이트 교차 어텐션.

### 2026년의 후손들 (The 2026 descendants)

- Q-Former: BLIP-2, InstructBLIP, MiniGPT-4, 그리고 토큰 예산 때문에 대부분의 비디오-언어 모델.
- 퍼시버 리샘플러(Perceiver resampler): Flamingo의 변형(Lesson 12.04). Idefics 계열, Eagle, OmniMAE.
- MLP 투영기: LLaVA, LLaVA-NeXT, LLaVA-OneVision, Cambrian-1.
- 어텐션 풀(Attention pool): VILA, PaliGemma.

네 가지 모두 유효하다. 결정을 가르는 질문은 토큰 예산에 제약을 받는지, 토큰당 품질에 제약을 받는지다.

```figure
modality-projection
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 표준 라이브러리로 Q-Former 스타일 교차 어텐션을 만든다:

1. 256개 이미지 패치 토큰(차원 128)을 시뮬레이션한다.
2. 32개 학습 가능한 쿼리(차원 128)를 인스턴스화한다.
3. 스케일드 닷-프로덕트(scaled-dot-product) 교차 어텐션을 실행한다(Q는 쿼리에서, K/V는 패치에서).
4. 선형 층을 통해 LLM 차원(512)으로 투영한다.
5. 32개의 LLM에 바로 쓸 수 있는 시각 토큰을 출력한다.

모든 수학은 순수 Python으로(벡터에 대한 중첩 루프). 장난감이지만 형상은 올바르다. 어텐션 가중치 행렬(matrix)이 출력되므로 각 쿼리가 어떤 패치에서 끌어왔는지 볼 수 있다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-modality-bridge-picker.md`를 만든다. 목표 VLM 설정(비전 인코더 토큰 개수, LLM 컨텍스트 예산, 배포 제약, 품질 목표)이 주어지면, 짧은 근거와 각 다리에 대한 파라미터 개수 추정치와 함께 Q-Former 대 MLP 대 퍼시버 리샘플러를 추천한다.

## 연습 문제 (Exercises)

1. PyTorch로 교차 어텐션 블록을 구현하라. 32개 쿼리와 256개 키/값으로, 어텐션 가중치 행렬이 32 x 256이고 소프트맥스(softmax) 후 각 행의 합이 1인지 검증하라.

2. BLIP-2 1단계에서 Q-Former는 세 가지 손실을 동시에 실행한다: ITC, ITM, ITG. 각각의 순방향(forward) 시그니처를 의사 코드(pseudo-code)로 작성하라. 어느 것이 텍스트 인코더 경로가 활성화되어 있어야 하는가?

3. 파라미터 개수를 비교하라: Q-Former(12층, 은닉 768) 대 2층 MLP 투영기(1408 → 4096, 두 층). 어떤 LLM 규모에서 1억 8800만 Q-Former 비용이 학습 효율로 본전을 뽑는가?

4. BLIP-2 논문(arXiv:2301.12597)의 Section 3.2에서 Q-Former가 어떻게 초기화되는지 읽어라. (무작위가 아니라) BERT-base에서 초기화하는 것이 왜 수렴(convergence)을 가속하는지 설명하라.

5. 1 FPS로 샘플링해 60프레임으로 만든 10분짜리 비디오에 대해, (Q-Former → 프레임당 32토큰) 대 (MLP 투영기 → 프레임당 576토큰)의 프레임당 토큰 비용을 계산하라. 어느 것이 128k 토큰 LLM 컨텍스트 윈도우(context window)에 맞는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Q-Former | "쿼리하는 트랜스포머" | 동결된 ViT 특성에 교차 어텐션을 하는 32개의 학습 가능한 쿼리 벡터를 가진 작은 트랜스포머 |
| 학습 가능한 쿼리(Learnable queries) | "비전용 소프트 프롬프트" | 교차 어텐션의 쿼리 쪽 역할을 하는 고정된 파라미터 집합. 모델별로 학습되고 모든 입력에 공유된다 |
| 교차 어텐션(Cross-attention) | "Q는 여기서, K/V는 저기서" | 쿼리, 키, 값이 서로 다른 출처에서 오는 어텐션. 쿼리가 ViT 패치에서 끌어오는 방식 |
| ITC | "이미지-텍스트 대조" | Q-Former 풀링 쿼리 대 텍스트 CLS에 적용되는 CLIP 스타일 손실 |
| ITM | "이미지-텍스트 매칭" | 어려운 음성이 마이닝된 쌍에 대한 이진 분류기. 쿼리가 세밀한 불일치를 판별하도록 강제한다 |
| ITG | "이미지 기반 텍스트 생성" | 쿼리에 조건화되어 텍스트가 생성되는 인과 LM 손실. 쿼리가 텍스트로 디코딩 가능한 내용을 인코딩하도록 강제한다 |
| 2단계 사전 학습(Two-stage pretraining) | "표현 다음 생성" | 1단계는 Q-Former만 학습(ITC/ITM/ITG), 2단계는 동결된 LLM을 붙이고 투영 + Q-Former만 학습 |
| 동결 백본(Frozen backbone) | "파인튜닝하지 마라" | 비전 인코더와 LLM 가중치는 고정된다. 다리만 학습한다 |
| 투영 헤드(Projection head) | "LLM 차원으로 가는 선형" | Q-Former 출력을 LLM의 임베딩 차원으로 매핑하는 최종 선형 층 |
| 퍼시버 리샘플러(Perceiver resampler) | "Flamingo의 버전" | 비슷한 학습 가능 쿼리 교차 어텐션. Flamingo가 단일 다리가 아니라 모든 층에서 사용한다 |

## 더 읽을거리 (Further Reading)

- [Li et al.: BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) 핵심 논문.
- [Li et al.: BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) ITC/ITM/ITG 삼총사를 가진 선행 연구.
- [Li et al.: ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651): "align before fuse": 1단계 학습의 개념적 조상.
- [Dai et al.: InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) 명령어 인식 Q-Former.
- [Zhu et al.: MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) 투영기만 쓰는 접근법.
- [Jaegle et al.: Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) 학습 가능 쿼리 교차 어텐션을 위한 일반 아키텍처.
