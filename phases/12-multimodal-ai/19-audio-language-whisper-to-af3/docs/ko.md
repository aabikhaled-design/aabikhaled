# 오디오-언어 모델: Whisper에서 Audio Flamingo 3까지의 궤적

> Whisper(Radford et al., 2022년 12월)는 음성 인식을 정리했다. 68만 시간의 약지도(weakly-supervised) 다국어 음성, 단순한 인코더-디코더 트랜스포머(transformer), 이후 모든 ASR 출시가 인용할 수밖에 없게 만든 벤치마크(benchmark)다. 하지만 인식은 추론이 아니다. "이 녹음에 어떤 악기가 있는가" "화자가 어떤 감정을 표현하는가" "3분 지점에서 무슨 일이 일어났는가"를 묻는 것은 전사(transcription)가 아니라 오디오 이해를 요구한다. Qwen-Audio, SALMONN, LTU, 그리고 NVIDIA의 Audio Flamingo 3(AF3, 2025년 7월)는 그 스택을 점진적으로 쌓아 올렸다. Whisper급 인코더를 유지하고, Q-former를 덧붙이고, 오디오-텍스트 인스트럭션 데이터로 학습하고, 사고 연쇄(chain-of-thought) 추론을 더했다. 이 레슨은 그 궤적을 따라간다.

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180분

## 학습 목표 (Learning Objectives)

- 파형(waveform)으로부터 로그-멜 스펙트로그램(log-Mel spectrogram)을 계산하기: 윈도잉, FFT, 필터 뱅크, 로그 변환.
- 인코더(encoder) 선택지를 비교하기: Whisper 인코더, BEATs, AF-Whisper 하이브리드. 각각이 우세한 경우.
- 오디오 Q-former를 만들기: 스펙트로그램 패치에 교차 어텐션(cross-attention)하는 N개의 학습 가능한 쿼리.
- 캐스케이드(Whisper 후 LLM) 방식과 종단간(end-to-end) 오디오-LLM 학습을 비교하기: 종단간이 추론에 대해 더 잘 확장되는 이유.

## 문제 (The Problem)

음성 인식은 Whisper로 해결되었다. 오디오의 OCR은 범용 상품(commodity)이 되었다. 하지만 "범용 상품"은 전사에서 멈춘다. 모델이 들은 것, 즉 타이밍, 화자, 감정, 음악 구조, 환경음을 추론하지 못한다면 전사만으로는 제품 기능을 구동할 수 없다.

명백한 세 가지 경로:

1. 캐스케이드(Cascade): Whisper가 전사하고, LLM이 그 전사문에 대해 추론한다. 순수 음성 시나리오에서는 동작한다. 음악, 환경음, 다중 화자 중첩, 감정에서는 실패한다.

2. 종단간(End-to-end) 오디오-LLM: 오디오 인코더가 오디오 토큰(token)을 LLM에 직접 공급하여 전사를 건너뛴다. 음향 정보(감정, 화자, 환경)를 보존한다. 새로운 학습 데이터가 필요하다.

3. 하이브리드(Hybrid): 전사와 추론을 모두 할 수 있는 오디오 인코더 + 텍스트 디코더(decoder). Qwen-Audio와 Audio Flamingo가 이 경로를 택한다.

## 개념 (The Concept)

### 로그-멜 스펙트로그램: 입력 특성

모든 오디오 인코더는 동일한 특성(feature)에서 시작한다: 로그-멜 스펙트로그램.

1. 16 kHz로 리샘플링한다.
2. 25ms 윈도우, 10ms 홉(hop)으로 단시간 푸리에 변환(short-time Fourier transform)을 한다.
3. FFT 결과의 크기(magnitude)를 취한다.
4. 멜 필터 뱅크(Mel filter banks)(보통 0~8000 Hz를 로그 간격으로 나눈 80개 필터)를 적용해 지각적 주파수로 워핑한다.
5. 동적 범위를 위해 로그 압축(log(1 + x))한다.

결과: 형상이 (T, 80)인 2D 배열로, 여기서 T는 시간 프레임의 개수다. 100 Hz 프레임 레이트의 30초 클립의 경우: (3000, 80).

### Whisper의 인코더

Whisper의 인코더는 로그-멜 스펙트로그램을 시간 프레임의 시퀀스로 처리하는 12층 ViT 스타일 트랜스포머다. 출력: 시간 프레임당 하나의 은닉 상태 벡터(vector).

ASR의 경우, Whisper의 디코더는 인코더 출력을 조건으로 텍스트 토큰을 생성하는 교차 어텐션 트랜스포머다. 표준적인 인코더-디코더다.

ALM(오디오-LLM)의 경우, 인코더 출력을 다른 LLM의 입력으로 쓰고자 한다. 패턴은 이렇다: Whisper 인코더는 동결(frozen), Q-former는 학습 가능, LLM은 동결 또는 튜닝.

### BEATs와 오디오 특화 인코더

Whisper는 음성 위주의 데이터로 학습되었다. 음악과 환경음에는 더 약하다.

BEATs(Chen et al., 2022)는 AudioSet으로 학습된 자기지도(self-supervised) 트랜스포머다. 동일한 파라미터(parameter) 수에서 Whisper보다 음악과 환경음을 더 잘 포착한다.

AF-Whisper(Audio Flamingo 3의 하이브리드): Whisper + BEATs 특성을 연결(concat)하여 오디오 입력으로 쓴다. Whisper는 언어적 신호를, BEATs는 음향적 신호를 담는다.

### 오디오 Q-former

BLIP-2의 시각 Q-former와 동일한 패턴이다. 고정된 수의 학습 가능한 쿼리(흔히 32개 또는 64개)가 오디오 인코더의 출력 프레임에 대해 교차 어텐션한다. 이 쿼리들이 LLM이 소비하는 오디오 토큰이 된다.

학습 정렬(alignment) 단계: Q-former 단독, 오디오-텍스트 쌍(AudioCaps, Clotho)에 대한 대조(contrastive) + 캡셔닝 손실(loss). 인스트럭션 단계: 종단간, LLM 동결 해제, 인스트럭션 데이터로 학습.

### 궤적: SALMONN, Qwen-Audio, AF3

SALMONN(Tang et al., 2023): Whisper + BEATs + Q-former + LLaMA. 진지한 추론 능력을 갖춘 최초의 오픈 오디오-LLM. MMAU 벤치마크에서 종합 약 0.55를 기록한다.

Qwen-Audio(Chu et al., 2023): 유사한 아키텍처, 더 풍부한 데이터셋(dataset)으로 학습, 다중 턴 대화에 맞춰 튜닝. MMAU 약 0.60.

LTU: Listen, Think, Understand(Gong et al., 2023): 명시적 추론 데이터, 오디오 클립에 대한 사고 연쇄에 초점. 더 작지만 더 집중되어 있다.

Audio Flamingo 3(Goel et al., 2025년 7월): 현재의 오픈 SOTA. 8B LLM 백본(Qwen2 7B), BEATs를 연결한 Whisper-large 인코더, 64-쿼리 Q-former, 100만 개 이상의 오디오-텍스트 인스트럭션 쌍으로 학습. MMAU 0.72, 일부 하위 과제에서 독점(proprietary) 프런티어와 대등하다.

AF3은 또한 오디오를 위한 온디맨드 사고 연쇄(on-demand chain-of-thought)를 도입한다. 모델은 최종 답 이전에 선택적으로 사고 토큰("먼저 악기를 식별해 보자: ...")을 내보낼 수 있다. 사고가 활성화되면 복잡한 추론 과제에서 정확도가 3~5점 상승한다.

### 캐스케이드 대 종단간

캐스케이드 파이프라인:

1. Whisper가 오디오를 전사 → 텍스트.
2. LLM이 텍스트에 대해 추론.

"이 팟캐스트를 요약하라"에는 완벽하게 동작한다. 다음에서는 실패한다:
- "이 노래의 분위기는 무엇인가?" 분위기는 가사가 아니라 소리에 있다.
- "Alice와 Bob 중 누가 말하고 있는가?" 화자 식별이 필요하다.
- "폭발은 몇 초에 일어나는가?" 시간적 기반(temporal grounding)이 텍스트에서 사라진다.
- "이것은 실제 오디오인가, 생성된 오디오인가?" 딥페이크 탐지는 음향 특성이 필요하다.

종단간은 음향 신호를 보존한다. Qwen-Audio와 AF3은 음악, 환경, 감정을 네이티브로 처리한다.

### 2026년 프로덕션 레시피

새로운 오디오 이해 제품의 경우:

- 캐스케이드: 전사가 목표이고, 음악이 없고, 감정 추론이 없는 경우.
- AF3 / Qwen-Audio 계열: 음악, 감정, 다중 화자, 또는 복잡한 오디오 추론이 있는 경우.

캐스케이드는 더 저렴하고 더 단순하다. 종단간은 더 유능하다.

### MMAU: 오디오 추론 벤치마크

MMAU(Massive Multimodal Audio Understanding)는 2024~2025년 오디오 추론 벤치마크다:

- 음성, 음악, 환경음을 아우르는 10,000개의 오디오-텍스트 QA 쌍.
- 분류(classification), 시간적 추론, 인과적 추론, 개방형 QA를 다룬다.
- 캐스케이드 파이프라인이 체계적으로 놓치는 것을 테스트한다.

오픈 SOTA(AF3)는 0.72, 독점 프런티어는 약 0.78(Gemini 2.5 Pro, Claude Opus 4.7)이다. 이 격차는 VideoMME의 오픈 대 클로즈드 차이보다 작으며, 이는 오디오-LLM이 성숙하고 있음을 시사한다.

```figure
audio-text-ctc
```

## 라이브러리로 써보기 (Use It)

`code/main.py`:

- stdlib로 로그-멜 스펙트로그램 계산을 구현한다: 윈도잉, 단순 DFT, 멜 필터 뱅크.
- 오디오 Q-former 골격: 인코더 출력 프레임이 주어지면 Q, K, V, 어텐션을 계산하고 N개의 토큰을 내보낸다.
- 장난감 과제에서 캐스케이드 대 종단간 비교.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-audio-llm-pipeline-picker.md`를 생성한다. 오디오 과제(전사, 음악 태깅, 감정 추론, 다중 화자 다이어라이제이션, 환경 분류)가 주어지면, 캐스케이드, 종단간 AF3, 또는 하이브리드 중에서 선택한다.

## 연습 문제 (Exercises)

1. 16kHz, 25ms 윈도우, 10ms 홉, 80 멜 빈으로 처리한 30초 클립의 로그-멜 스펙트로그램 차원을 계산하라. 48kHz에서는 어떻게 바뀌는가?

2. Whisper가 음악에서 성능이 낮은 이유는? BEATs는 Whisper가 포착하지 못하는 어떤 오디오 특성을 포착하는가?

3. 쿼리 64개 대 32개의 오디오 Q-former: 어느 과제 복잡도에서 64개가 효과를 발휘하는가? 32개는 무엇을 위해 연산을 절약하는가?

4. AF3의 온디맨드 사고에 관한 4절을 읽으라. 사고 연쇄가 가장 도움이 되는 오디오 과제 세 가지를 제안하라.

5. AF3의 출력을 사용해 최소한의 다이어라이제이션 파이프라인을 구현하라. 화자 전환을 어떻게 신호하는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 로그-멜 스펙트로그램(Log-Mel spectrogram) | "멜 특성" | 멜 필터 뱅크 후의 로그-크기 값으로 이루어진 2D (시간, 주파수) 배열 |
| 오디오 Q-former(Audio Q-former) | "오디오 Perceiver" | 오디오 인코더 출력에서 LLM에 공급되는 고정 길이 쿼리로 가는 교차 어텐션 병목 |
| 캐스케이드(Cascaded) | "ASR 후 LLM" | Whisper가 전사하고 텍스트 LLM이 추론하는 파이프라인; 음향 정보를 잃음 |
| 종단간(End-to-end) | "오디오-LLM" | 오디오 특성이 Q-former를 통해 LLM에 직접 진입; 음향 신호를 보존 |
| BEATs | "오디오 AudioSet 인코더" | AudioSet으로 학습된 SSL 트랜스포머; 음악 + 환경음에 강함 |
| MMAU | "오디오 추론 벤치" | 음성, 음악, 환경에 걸친 10k QA 쌍; 2024년 평가 표준 |
| 온디맨드 사고(On-demand thinking) | "오디오 CoT" | 모델이 최종 답 이전에 선택적으로 추론 토큰을 내보낼 수 있음, 정확도를 3~5점 상승시킴 |

## 더 읽을거리 (Further Reading)

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
