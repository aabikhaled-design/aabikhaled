# 워터마킹: SynthID, Stable Signature, C2PA

> 세 가지 기술이 2026년 AI 생성 콘텐츠의 출처(provenance)를 구조화한다. SynthID(Google DeepMind): 이미지 워터마킹은 2023년 8월 출시, 텍스트+비디오는 2024년 5월(Gemini + Veo), 텍스트는 2024년 10월 Responsible GenAI Toolkit을 통해 오픈소스화, 통합 멀티미디어 탐지기는 2025년 11월 Gemini 3 Pro와 함께 출시. 텍스트 워터마킹은 다음 토큰(next-token) 샘플링(sampling) 확률을 감지 불가능하게 조정한다. 이미지/비디오 워터마크는 압축, 크롭(cropping), 필터, 프레임 레이트 변경을 견딘다. Stable Signature(Fernandez et al., ICCV 2023, arXiv:2303.15435): 잠재 확산(latent diffusion) 디코더(decoder)를 파인튜닝(fine-tuning)하여 모든 출력이 고정 메시지를 담도록 한다. 크롭된(콘텐츠의 10%) 생성 이미지가 FPR<1e-6에서 90% 초과로 탐지됨. 후속 연구 "Stable Signature is Unstable"(arXiv:2405.07145, 2024년 5월): 파인튜닝이 품질을 보존하면서 워터마크를 제거한다. C2PA: 암호학적으로 서명되고 변조가 드러나는(tamper-evident) 메타데이터 표준(C2PA 2.2 Explainer 2025). 워터마킹과 C2PA는 상호 보완적이다: 메타데이터는 제거될 수 있지만 더 풍부한 출처를 담는다. 워터마크는 트랜스코딩(transcoding)을 통과해 지속되지만 더 적은 정보를 담는다.

**Type:** Build
**Languages:** Python (stdlib, token-watermark embed + detect)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 토큰 수준 워터마킹(SynthID-text 방식)과 그 탐지 메커니즘을 기술하기.
- Stable Signature와, 이를 무력화한 2024년 제거 공격을 기술하기.
- C2PA의 역할과, 왜 워터마킹에 상호 보완적인지 진술하기.
- 핵심 한계들을 기술하기: 모델 특정적(model-specific) 신호, 패러프레이즈(paraphrase) 하에서의 견고성, 의미 보존(meaning-preserving) 공격(arXiv:2508.20228).

## 문제 (The Problem)

2023-2024년에는 딥페이크(deepfake)와 AI 생성 콘텐츠가 정치적·소비자 맥락에 대규모로 진입했다. 워터마킹은 제안된 기술적 출처 신호다. 생성물을 생성 시점에 표시하고 나중에 탐지한다. 2025년의 증거: 어떤 워터마크도 무조건적으로 견고하지는 않지만, C2PA 메타데이터와 계층화하면 그 조합은 쓸 만한 출처 서사를 제공한다.

## 개념 (The Concept)

### 텍스트 워터마킹(SynthID-text 방식)

Google이 프로덕션에 적용한 Kirchenbauer et al. 2023 메커니즘:

1. 각 디코딩 단계에서, 이전 K개 토큰을 해싱하여 어휘(vocabulary)를 "녹색(green)"과 "적색(red)" 집합으로 의사난수(pseudorandom) 분할한다.
2. 녹색 로짓(logit)에 δ를 더해 샘플링을 녹색 집합 쪽으로 편향시킨다.
3. 생성물은 우연이 만들어낼 것보다 더 많은 녹색 토큰을 담는다.

탐지: 각 접두사(prefix)를 다시 해싱하고, 생성물의 녹색 토큰을 세어 z-점수(z-score)를 계산한다. z-점수는 워터마크된 텍스트에서 >0이고, 사람이 쓴 텍스트에서 ~0이다.

성질:
- 독자에게 감지 불가능하다(δ가 충분히 작아 품질 손실이 미미하다).
- 어휘 분할 함수에 접근할 수 있으면 탐지 가능하다.
- 패러프레이즈에 견고하지 않다. 텍스트를 다시 쓰면 신호가 파괴된다.

SynthID-text는 2024년 10월 Google의 Responsible GenAI Toolkit을 통해 오픈소스화되었다.

### Stable Signature(이미지)

Fernandez et al. ICCV 2023. 잠재 확산 디코더를 파인튜닝하여, 모든 생성 이미지가 잠재 표현(latent representation)에 내장된 고정 이진 메시지를 담도록 한다. 탐지는 신경 디코더가 잠재 표현에서 메시지를 디코딩하는 방식이다. (콘텐츠의 10%로) 크롭된 이미지가 FPR<1e-6에서 90% 초과로 탐지됨.

2024년 5월 "Stable Signature is Unstable"(arXiv:2405.07145): 디코더를 파인튜닝하면 이미지 품질을 보존하면서 워터마크가 제거된다. 적대적(adversarial) 생성 후 파인튜닝은 비용이 저렴하다. 워터마크의 적대적 견고성은 제한적이다.

### SynthID 통합 탐지기(2025년 11월)

Gemini 3 Pro와 함께: 텍스트, 이미지, 오디오, 비디오로부터 SynthID 신호를 하나의 API로 읽어내는 멀티미디어 탐지기. Google 출처 스택을 통합한다.

### C2PA

Coalition for Content Provenance and Authenticity. 암호학적으로 서명되고 변조가 드러나는 메타데이터 표준. C2PA 2.2 Explainer(2025). C2PA 매니페스트(manifest)는 출처 주장(누가 만들었고, 언제, 어떤 변환을 거쳤는지)을 기록하며 제작자의 키로 서명된다.

워터마킹에 상호 보완적이다:
- 메타데이터는 제거될 수 있지만, 워터마크는 (쉽게) 제거될 수 없다.
- 메타데이터는 풍부하다(전체 출처 체인). 워터마크는 비트(bit)를 담는다.
- C2PA는 플랫폼 채택에 의존한다. 워터마크는 자동으로 내장된다.

Google은 Search, Ads, "About this image"에 둘 다 통합한다.

### 한계

- **모델 특정적.** SynthID는 SynthID가 활성화된 모델의 생성물에 워터마크를 넣는다. SynthID가 없는 모델의 생성물에는 워터마크가 없으므로, "SynthID 신호 없음"이 진정성(authenticity)의 증거가 되지는 않는다.
- **패러프레이즈.** 텍스트 워터마크는 의미 보존 패러프레이즈를 견디지 못한다.
- **변환 공격.** arXiv:2508.20228(2025)은 텍스트 워터마크와 많은 이미지 워터마크를 모두 파괴하는 의미 보존 공격을 보인다.
- **파인튜닝 제거.** "Stable Signature is Unstable"에 따르면, 생성 후 파인튜닝이 내장된 워터마크를 제거한다.

### EU AI Act 제50조

AI 생성 콘텐츠 라벨링을 위한 투명성 강령(Transparency Code)(첫 초안 2025년 12월, 두 번째 초안 2026년 3월, [유럽위원회 상태 페이지](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)에 따르면 최종본은 2026년 6월 예상). 이 강령은 2026년 4월 기준 여전히 초안 상태이며 일정은 변경될 수 있다. 기술 계층을 요구하는 규제 계층이다. 딥페이크는 라벨이 붙어야 한다.

### Phase 18에서 이 레슨의 위치

레슨 22-23은 모델이 무엇을 방출하는가(사적 데이터, 출처 신호)에 관한 것이다. 레슨 27은 학습 데이터 거버넌스(governance)를 다룬다. 레슨 24는 이러한 기술적 조치를 요구하는 규제 프레임워크다.

```figure
an-watermark-greenlist
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 수준의 텍스트 워터마크를 구성한다. 토큰은 0..N-1의 정수다. 워터마크된 샘플링은 해시로 정의된 녹색 집합 쪽으로 편향된다. 탐지기는 녹색 토큰 z-점수를 계산한다. 1000 토큰 생성에서의 탐지를 관찰하고, 패러프레이즈가 신호를 파괴하는 것을 지켜보며, 사람이 쓴 텍스트에 대한 거짓 양성률(false-positive rate)을 측정할 수 있다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-provenance-audit.md`를 생성한다. 출처 주장이 있는 콘텐츠 배포(deployment)가 주어지면, 다음을 감사한다: (있다면) 워터마크 메커니즘, (있다면) C2PA 서명 체인, 각각의 적대적 견고성, 그리고 모달리티(modality)별 커버리지.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 워터마크된 1000 토큰 생성 대 사람이 작성한 텍스트의 z-점수를 보고하라. 95% 신뢰 임계값에서의 거짓 양성률을 식별하라.

2. 토큰의 30%를 동의어로 교체하는 패러프레이즈 공격을 구현하라. z-점수를 다시 측정하라.

3. 견고성에 관한 Kirchenbauer et al. 2023 6절을 읽어라. 왜 텍스트 워터마크는 패러프레이즈 하에서 실패하지만 이미지 워터마크는 크롭을 견디는가?

4. SynthID-text + C2PA 메타데이터를 사용하는 배포를 설계하라. 소비자가 보는 출처 체인을 기술하라. 각 구성 요소의 실패 모드 하나씩을 식별하라.

5. 2024년 "Stable Signature is Unstable" 결과는 파인튜닝이 이미지 워터마크를 제거함을 보인다. 이 공격을 제한하는 배포 통제를 설계하라. 예를 들어, 파인튜닝된 체크포인트(checkpoint)의 서명된 릴리스를 요구한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| SynthID | "Google의 워터마크" | 교차 모달(cross-modal) 출처 신호. 텍스트, 이미지, 오디오, 비디오 |
| 토큰 워터마크(Token watermark) | "Kirchenbauer 방식" | 녹색 토큰 z-점수로 탐지 가능한 편향 샘플링 텍스트 워터마크 |
| Stable Signature | "이미지 워터마크" | 파인튜닝된 디코더 워터마크. ICCV 2023 |
| C2PA | "그 메타데이터 표준" | 암호학적으로 서명되고 변조가 드러나는 출처 메타데이터 |
| 패러프레이즈 견고성(Paraphrase robustness) | "표현을 바꾸면 깨지는가" | 텍스트 워터마크 성질. 현재로서는 제한적 |
| 파인튜닝 제거(Fine-tune removal) | "적대적 워터마크 제거" | 디코더 파인튜닝으로 이미지 워터마크를 제거하는 공격 |
| 교차 모달 탐지기(Cross-modal detector) | "통합 SynthID" | 모달리티 전반에 걸친 2025년 11월 통합 API |

## 더 읽을거리 (Further Reading)

- [Kirchenbauer et al.: A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) 토큰 워터마크 메커니즘
- [Fernandez et al.: Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) 이미지 워터마크 논문
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145): 제거 공격
- [Google DeepMind: SynthID](https://deepmind.google/models/synthid/) 교차 모달 워터마크
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html): 메타데이터 표준
