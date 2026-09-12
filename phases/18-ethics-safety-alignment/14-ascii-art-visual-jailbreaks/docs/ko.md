# 아스키 아트와 시각적 탈옥 (ASCII Art and Visual Jailbreaks)

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs"(ACL 2024, arXiv:2402.11753). 유해 요청에서 안전에 관련된 토큰을 가리고, 같은 글자의 아스키 아트(ASCII art) 렌더링으로 바꾼 뒤, 위장된 프롬프트를 보낸다. GPT-3.5, GPT-4, Gemini, Claude, Llama-2 모두 아스키 아트 토큰을 견고하게 인식하지 못한다. 이 공격은 PPL(perplexity 필터), 패러프레이즈(Paraphrase) 방어, 재토큰화(Retokenization)를 우회한다. 관련: ViTC 벤치마크는 비의미적(non-semantic) 시각 프롬프트의 인식을 측정한다; StructuralSleight는 인코딩 공격 계열로서 흔치 않은 텍스트 인코딩 구조(Uncommon Text-Encoded Structures; 트리, 그래프, 중첩 JSON)로 일반화한다.

**Type:** Build
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- ArtPrompt 공격을 기술하기: 단어 식별 단계, 아스키 아트 치환, 최종 위장 프롬프트.
- 표준 방어(PPL, 패러프레이즈, 재토큰화)가 ArtPrompt에서 실패하는 이유를 설명하기.
- ViTC를 정의하고 그것이 측정하는 바를 기술하기.
- 임의의 흔치 않은 텍스트 인코딩 구조로의 일반화로서 StructuralSleight를 기술하기.

## 문제 (The Problem)

패러프레이즈와 역할극으로 하는 공격(Lesson 12), 긴 컨텍스트로 하는 공격(Lesson 13)은 텍스트 수준 패턴에서 작동한다. ArtPrompt는 인식(recognition) 수준에서 작동한다: 모델은 금지된 토큰을 파싱하지 않는다. 모델은 문자로 렌더링된 이미지를 파싱한다. 안전 필터는 무해한 구두점을 본다. 모델은 단어를 본다.

## 개념 (The Concept)

### ArtPrompt, 두 단계

1단계. 단어 식별(Word Identification). 유해 요청이 주어지면, 공격자는 LLM으로 안전에 관련된 단어(예: "how to make a bomb"에서 "bomb")를 식별한다.

2단계. 위장 프롬프트 생성(Cloaked Prompt Generation). 식별된 각 단어를 그 아스키 아트 렌더링(글자 모양을 이루는 7x5 또는 7x7 문자 블록)으로 바꾼다. 모델은 충분히 역량 있는 모델이라면 단어로 인식할 수 있는 구두점과 공백의 격자를 받고, 안전 필터는 격자만 본다.

결과: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 모두 실패한다. 그들의 벤치마크 부분집합에서 공격 성공률 75% 이상.

### 표준 방어가 실패하는 이유

- **PPL(perplexity 필터).** 아스키 아트는 높은 perplexity를 가진다. 하지만 모든 새로운 입력도 그렇다. ArtPrompt를 차단하는 임계값을 고르면 정당한 구조화 입력도 차단한다.
- **패러프레이즈.** 프롬프트를 패러프레이즈하면 아스키 아트가 파괴된다. 실제로는 패러프레이즈 LLM이 종종 아트를 보존하거나 재구성한다.
- **재토큰화.** 토큰을 다르게 쪼개도 모델의 시각이 글자 모양을 인식한다는 사실은 바뀌지 않는다.

근본 문제는 안전 필터가 토큰 또는 의미 수준이라는 것이다; ArtPrompt는 시각적 인식 수준에서 작동한다.

### ViTC 벤치마크

비의미적 시각 프롬프트의 인식. 아스키 아트, 윙딩(wingdings), 기타 비텍스트-의미 시각 콘텐츠를 읽는 모델의 능력을 측정한다. ArtPrompt의 효과는 ViTC 정확도와 상관관계가 있다: 모델이 시각 텍스트를 더 잘 읽을수록 ArtPrompt가 그 모델에 더 잘 통한다. 이것은 역량-안전 트레이드오프(trade-off)다.

### StructuralSleight

ArtPrompt를 일반화한다: 흔치 않은 텍스트 인코딩 구조(Uncommon Text-Encoded Structures, UTES). 트리, 그래프, 중첩 JSON, JSON 안의 CSV, diff 스타일 코드 블록. 어떤 구조가 학습 안전 데이터에서는 드물지만 모델이 파싱할 수 있다면, 유해 콘텐츠를 숨길 수 있다.

방어 함의: 안전은 모델이 파싱할 수 있는 구조화 표현 전반에 걸쳐 일반화되어야 한다. 그 집합은 크고 계속 커지고 있다.

### 이미지 모달리티 유사물

시각 LLM(GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1)은 공격 표면을 확장한다. 실제 이미지를 사용한 ArtPrompt 방식 공격은 아스키 아트 유사물보다 더 강한데, 이미지 인코더가 더 풍부한 신호를 만들기 때문이다.

### Phase 18에서의 위치

Lesson 12-14는 세 가지 직교(orthogonal) 공격 벡터를 기술한다: 반복적 정제(PAIR), 컨텍스트 길이(MSJ), 인코딩(ArtPrompt/StructuralSleight). Lesson 15는 모델 중심 공격에서 시스템 경계 공격(간접 프롬프트 주입, indirect prompt injection)으로 전환한다. Lesson 16은 방어 도구 대응을 기술한다.

```figure
al-ascii-cloak
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 ArtPrompt를 만든다. 유해 쿼리에서 특정 단어를 아스키 아트 글리프(glyph)로 위장하고, 위장된 문자열이 키워드 필터를 통과하는지 검증하고, (선택적으로) 간단한 인식기로 위장된 문자열을 다시 디코딩할 수 있다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-encoding-audit.md`를 만든다. 탈옥 방어 보고서가 주어지면, 다룬 인코딩 공격 계열(아스키 아트, base64, 리트 스피크(leet-speak), UTF-8 동형 문자(homoglyph), UTES)과 각각을 잡아내는 방어 계층을 열거한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 위장된 문자열이 간단한 키워드 필터를 통과하는지 검증하라. 필요한 문자 수준 변경을 보고하라.

2. 두 번째 인코딩을 구현하라: 같은 대상 단어에 대한 base64. ArtPrompt와 비교하여 필터 우회율과 복원 난이도를 비교하라.

3. Jiang 외 2024의 4.3절(다섯 모델 결과)을 읽어라. 같은 벤치마크에서 Claude의 ArtPrompt 저항성이 Gemini보다 높은 이유를 제안하라.

4. 프롬프트에서 아스키 아트 모양의 영역을 탐지하는 생성 전(pre-generation) 방어를 설계하라. 정당한 코드, 표, 수학 표기에 대한 거짓 양성(false-positive) 비율을 측정하라.

5. StructuralSleight는 10개의 인코딩 구조를 나열한다. 10개 모두를 처리하는 일반화된 방어를 스케치하고 방어된 프롬프트당 연산 비용을 추정하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| ArtPrompt | "아스키 아트 공격" | 안전 단어를 아스키 아트 렌더링으로 가리는 2단계 탈옥 |
| 위장 (Cloaking) | "단어 숨기기" | 금지된 토큰을 모델은 읽지만 필터는 읽지 못하는 시각적 표현으로 대체 |
| UTES | "흔치 않은 구조" | Uncommon Text-Encoded Structure: 콘텐츠를 밀반입하는 데 쓰이는 트리, 그래프, 중첩 JSON 등 |
| ViTC | "시각-텍스트 역량" | 비의미적 시각 인코딩을 읽는 모델 능력의 벤치마크 |
| Perplexity 필터 | "PPL 방어" | 높은 perplexity의 프롬프트를 거부; 정당한 구조화 입력도 높게 나와 실패함 |
| 재토큰화 (Retokenization) | "토크나이저 전환 방어" | 다른 토크나이저로 프롬프트를 전처리; 인식이 시각적이라 실패함 |
| 동형 문자 (Homoglyph) | "닮은꼴 문자" | 라틴 문자와 똑같아 보이는 유니코드 문자; 부분 문자열 검사를 우회함 |

## 더 읽을거리 (Further Reading)

- [Jiang et al.: ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) 아스키 아트 탈옥 논문
- [Li et al.: StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) UTES 일반화
- [Chao et al.: PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) 상호 보완적 반복 공격
- [Anil et al.: Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) 상호 보완적 길이 공격
