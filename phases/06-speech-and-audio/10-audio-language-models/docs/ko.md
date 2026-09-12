# 오디오-언어 모델(Audio-Language Models): Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> 2026년의 오디오-언어 모델은 음성 + 환경음 + 음악을 두루 추론한다. Qwen2.5-Omni-7B는 MMAU-Pro에서 GPT-4o Audio와 맞먹는다. Audio Flamingo Next는 LongAudioBench에서 Gemini 2.5 Pro를 능가한다. 오픈과 클로즈드 사이의 격차는 사실상 사라졌다. 모두가 무작위에 가까운 성능을 보이는 다중 오디오(multi-audio) 과제만 빼면.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45분

## 문제 (The Problem)

5초짜리 오디오가 있다. 개가 짖고, 누군가 "멈춰!"라고 외치고, 그다음 정적이 흐른다. 유용한 질문은 여러 축에 걸쳐 있다.

- **전사(Transcription).** "무슨 말을 했는가?": ASR의 영역이다.
- **의미 추론(Semantic reasoning).** "그 사람은 위험에 처해 있는가?": 짖는 소리 + 외침 + 정적을 종합적으로 이해해야 한다.
- **음악 추론(Music reasoning).** "어떤 악기가 멜로디를 연주하는가?"
- **긴 오디오 검색(Long-audio retrieval).** "이 90분짜리 강의에서 강사가 경사 하강법(gradient descent)을 설명한 부분은 어디인가?"

이 모든 질문에 단일 프롬프트(prompt)로 답하는 단일 모델이 **오디오-언어 모델(audio-language model)**(LALM / ALM)이다. 순수 ASR과는 구분된다. LALM은 단순한 전사가 아니라 자유 형식의 자연어 답변을 생성한다.

## 개념 (The Concept)

![오디오-언어 모델: 오디오 인코더 + 프로젝터 + LLM 디코더](../assets/alm-architecture.svg)

### 3-구성요소 템플릿

2026년의 모든 LALM은 동일한 골격을 가진다.

1. **오디오 인코더(Audio encoder).** Whisper 인코더 · BEATs · CLAP · WavLM · 또는 모델별 커스텀 인코더.
2. **프로젝터(Projector).** 오디오 인코더의 특성(feature)을 LLM의 토큰 임베딩(token embedding) 공간으로 잇는 선형(Linear) 또는 MLP.
3. **LLM.** Llama / Qwen / Gemma 기반 디코더(decoder). 텍스트 + 오디오 토큰을 교차로 입력받아 텍스트를 생성한다.

학습(training):

- **1단계.** 인코더 + LLM을 동결(freeze)하고, ASR / 캡셔닝 데이터로 프로젝터만 학습한다.
- **2단계.** 명령 따르기(instruction-following) 오디오 과제(QA, 추론, 음악 이해)에 대해 전체 또는 LoRA 파인튜닝(fine-tuning)을 한다.
- **3단계(선택).** 음성 입력 / 음성 출력은 음성 디코더를 추가한다. Qwen2.5-Omni와 AF3-Chat이 이렇게 한다.

### 2026년 모델 지도

| 모델 | 백본 | 오디오 인코더 | 출력 모달리티 | 접근성 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### 벤치마크 현실 점검 (2026)

**MMAU-Pro.** 음성 / 소리 / 음악 / 혼합을 아우르는 QA 쌍 1800개. 다중 오디오 서브셋 포함.

| 모델 | 전체 | 음성 | 소리 | 음악 | 다중 오디오 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

**다중 오디오 열은 모두에게 치명적이다.** 4지선다 객관식에서 무작위 확률은 25%인데, 대부분의 모델이 그 언저리 점수를 받는다. LALM은 두 클립을 비교하는 일을 여전히 어려워한다.

### 2026년 LALM이 유용한 곳

- **콜센터 녹음의 컴플라이언스 감사.** "상담원이 의무 고지 사항을 언급했는가?"
- **접근성(Accessibility).** 청각 장애가 있는 사용자에게 소리 이벤트를 묘사한다(단순 전사가 아니라).
- **콘텐츠 모더레이션.** 폭력적 언어 + 위협적 어조 + 배경 맥락을 탐지한다.
- **팟캐스트 / 회의 챕터링.** 화자 전환만이 아닌 의미 기반 요약.
- **음악 카탈로그 분석.** "B 섹션에서 조성이 바뀌는 모든 트랙을 찾아라."

### 아직 유용하지 않은 곳

- 세밀한 음악 이론(코드 수준 이하).
- 긴 대화에 대한 화자 귀속(speaker-attributed) 추론(10분이 넘어가면 성능이 떨어진다).
- 다중 오디오 비교(22-26%는 무작위보다 간신히 높을 뿐이다).
- 실시간 스트리밍 추론(대부분 오프라인 배치 추론이다).

```figure
v4-alm-tokens
```

## 직접 만들기 (Build It)

### 1단계: Qwen2.5-Omni에 질의하기

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 2단계: 프로젝터 패턴

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

이게 전부다. 프로젝터는 보통 1-3개의 선형 층(layer)이다. ASR 쌍(오디오 → 전사)으로 프로젝터를 학습하는 것이 1단계 사전 학습 과제(pretext task)다.

### 3단계: MMAU / LongAudioBench 벤치마킹

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

카테고리별(음성 / 소리 / 음악 / 다중 오디오)로 따로 보고한다. 집계 수치는 모델이 어디서 실패하는지를 가린다.

## 라이브러리로 써보기 (Use It)

| 과제 | 2026년 선택 |
|------|-----------|
| 자유 형식 오디오 QA (오픈) | Qwen2.5-Omni-7B |
| 긴 오디오 최고 오픈 모델 | Audio Flamingo Next |
| 최고 클로즈드 모델 | Gemini 2.5 Pro |
| 음성 입력 / 음성 출력 에이전트 | Qwen2.5-Omni 또는 GPT-4o Audio |
| 음악 추론 | Audio Flamingo 3 또는 2 (음악 특화 AF-CLAP) |
| 콜센터 감사 | API를 통한 Gemini 2.5 Pro, 정책 문서에 대한 RAG와 함께 |

## 함정 (Pitfalls)

- **다중 오디오에 대한 과신.** 과제가 "어느 클립에 X가 있는가"를 요구한다면, 무작위 수준의 성능은 실제 현실이다.
- **긴 오디오에서의 성능 저하.** 10분을 넘어가면 대부분 모델의 화자 귀속이 무너진다. 먼저 화자 분리(diarize)(Lesson 6)를 한 뒤 요약하라.
- **정적에서의 환각(Hallucinations).** Whisper 인코더를 쓰는 LALM이 물려받는, Whisper 스타일의 동일한 문제다. VAD로 게이팅하라.
- **벤치마크 체리 피킹.** 벤더 블로그 글은 최상의 카테고리를 부각한다. MMAU-Pro 다중 오디오 서브셋을 직접 돌려보라.

## 산출물 (Ship It)

`outputs/skill-alm-picker.md`로 저장한다. 주어진 오디오 이해 과제에 대해 LALM + 벤치마크 서브셋 + 출력 모달리티(텍스트 vs 음성)를 선택한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행해 장난감 프로젝터 패턴 + (오디오 임베딩, 텍스트 토큰) → 출력 토큰의 가짜 LALM 라우팅을 살펴본다.
2. **보통.** MMAU-Pro 음성 항목 100개에 대해 Qwen2.5-Omni-7B를 채점한다. 논문에 보고된 수치와 비교한다.
3. **어려움.** 최소한의 오디오 캡셔닝 베이스라인(baseline)을 만든다. BEATs 인코더 + 2층 프로젝터 + 동결된 Llama-3.2-1B. AudioCaps에서 프로젝터만 파인튜닝한다. Clotho-AQA에서 SALMONN과 비교한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| LALM | 오디오 ChatGPT | 오디오 인코더 + 프로젝터 + LLM 디코더. |
| Projector | 어댑터 | 오디오 특성을 LLM 임베딩 공간으로 매핑하는 작은 MLP. |
| MMAU | 그 벤치마크 | 음성, 소리, 음악에 걸친 1만 개의 오디오-QA 쌍. |
| MMAU-Pro | 더 어려운 MMAU | 1800개의 다중 오디오 / 추론 중심 문제. |
| LongAudioBench | 장형(long-form) 평가 | 의미 기반 질의를 동반한 수 분 길이의 클립. |
| Voice-in / voice-out | 음성 네이티브 | 텍스트를 우회하지 않고 음성을 입력받아 음성을 출력하는 모델. |

## 더 읽을거리 (Further Reading)

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759): 레퍼런스 아키텍처.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B): 음성 입력-음성 출력.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128): 오픈 진영의 긴 오디오 선두주자.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289): 듀얼 인코더의 선구자.
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/): 2026년 실시간 순위.
