# Capstone 12: 영상 이해 파이프라인 (Video Understanding Pipeline: Scene, QA, Search)

> Twelve Labs는 Marengo + Pegasus를 제품화했다. VideoDB는 영상용 CRUD API를 출시했다. AI2의 Molmo 2는 오픈 VLM 체크포인트를 공개했다. Gemini 롱 컨텍스트(long-context)는 수 시간 분량의 영상을 네이티브로 처리한다. TimeLens-100K는 대규모 시간적 그라운딩(temporal grounding)을 정의했다. 2026 파이프라인은 정착되었다: 장면 분할(scene segmentation), 장면별 캡션 + 임베딩(embedding), 트랜스크립트(transcript) 정렬, 다중 벡터(multi-vector) 인덱스, 그리고 (시작, 끝) 타임스탬프와 프레임 미리보기로 답하는 질의. 이 캡스톤(capstone)은 100시간을 수집하고 공개 벤치마크(benchmark)를 달성하며, 세기(counting)와 동작(action) 질문에서의 환각(hallucination)을 측정한다.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (UI)
**Prerequisites:** Phase 4 (CV), Phase 6 (speech), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17
**Time:** 30시간

## 문제 (Problem)

장편 영상 QA는 2026년 규모에서 가장 대역폭을 많이 잡아먹는 멀티모달(multimodal) 문제다. Gemini 2.5 Pro는 2시간짜리 영상을 네이티브로 읽을 수 있지만, 100시간 분량의 영상을 질의 가능한 코퍼스(corpus)로 수집(ingest)하려면 여전히 장면 수준 인덱스가 필요하다. 프로덕션(production)의 형태는 장면 분할(TransNetV2 또는 PySceneDetect), VLM을 사용한 장면별 캡셔닝(Gemini 2.5, Qwen3-VL-Max, 또는 Molmo 2), 트랜스크립트 정렬(단어 타임스탬프를 갖는 Whisper-v3-turbo), 그리고 캡션·프레임 임베딩·트랜스크립트를 나란히 저장하는 다중 벡터 인덱스를 결합한다. 질의 파이프라인은 (시작, 끝) 타임스탬프와 프레임 미리보기로 답한다.

벤치마크는 공개되어 있다(ActivityNet-QA, NeXT-GQA): 여기에 직접 만든 100개 질의 커스텀 세트를 더한다. 세기와 동작 유형 질문에서 나타나는 환각은 알려진 난해한 실패 부류이며, 이 캡스톤은 이를 명시적으로 측정한다.

## 개념 (Concept)

수집 시 세 개의 파이프라인이 병렬로 실행된다. **장면 분할**은 영상을 여러 장면으로 자른다. **VLM 캡셔닝**은 장면마다 캡션 하나와 키프레임(keyframe)에서 프레임 임베딩 하나를 생성한다. **ASR 정렬**은 단어 수준 타임스탬프를 만든다. 세 스트림은 (scene_id, 시간 범위)로 결합된다. 각 장면은 다중 벡터 인덱스(Qdrant)에서 세 가지 벡터 타입을 갖는다: 캡션 임베딩, 키프레임 임베딩, 트랜스크립트 임베딩.

질의 시 자연어 질문이 세 벡터 모두에 발사된다. 결과는 RRF로 병합되고, 시간적 그라운딩 어댑터(TimeLens 스타일)가 상위 장면 내에서 (시작, 끝) 윈도우를 정교화한다. VLM 합성기(Gemini 2.5 Pro 또는 Qwen3-VL-Max)가 질의 + 상위 장면 + 크롭된 프레임을 받아 인용된 타임스탬프와 프레임 미리보기로 답한다.

환각 측정이 중요하다. 세기("방에 몇 명이 들어오는가?")와 동작 유형("셰프가 젓기 전에 붓는가?") 질문은 악명 높을 만큼 신뢰하기 어렵다. 정확도를 서술형 질문과 분리해 보고한다.

## 아키텍처 (Architecture)

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 스택 (Stack)

- 장면 분할: TransNetV2 (2024-26 최첨단) 또는 PySceneDetect
- ASR: 단어 타임스탬프를 갖는 faster-whisper 경유 Whisper-v3-turbo
- VLM 캡셔너 + 답변기: Gemini 2.5 Pro 또는 Qwen3-VL-Max 또는 Molmo 2
- 시간적 그라운딩: TimeLens-100K로 학습된 어댑터 또는 VideoITG
- 인덱스: 다중 벡터를 지원하는 Qdrant (캡션 / 프레임 / 트랜스크립트)
- UI: HTML5 비디오 플레이어와 장면 썸네일을 갖춘 Next.js 15
- 평가: ActivityNet-QA, NeXT-GQA, 직접 레이블링한 100개 질문 커스텀 세트
- 환각 벤치마크: 수작업 레이블이 있는 세기 및 동작 유형 부분집합

```figure
cf-scene-index
```

## 직접 만들기 (Build It)

1. **수집 워커.** YouTube URL 또는 로컬 MP4를 받는다. 필요 시 720p로 다운스케일한다. `{video_id, file_path}`를 영속화한다.

2. **장면 분할.** TransNetV2 또는 PySceneDetect를 실행해 `[{scene_id, start_ms, end_ms, keyframe_path}]`를 생성한다. 100시간 목표: 약 6천~8천 장면.

3. **ASR 패스.** 오디오에 Whisper-v3-turbo를 실행한다. 단어 수준 타임스탬프를 내보낸다. 장면별 트랜스크립트 조각으로 분할한다.

4. **VLM 캡셔닝.** 장면마다 키프레임과 짧은 캡션 템플릿으로 Gemini 2.5 Pro(또는 Qwen3-VL-Max)를 호출한다. 캡션 + 프레임 임베딩을 생성한다.

5. **다중 벡터 인덱스.** 세 개의 명명된 벡터를 갖는 Qdrant 컬렉션. 페이로드: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **질의.** 자연어 질문이 세 개의 밀집(dense) 질의를 발사한다. reciprocal rank fusion으로 병합한다. top-k=5 장면.

7. **시간적 그라운딩.** 상위 장면에 TimeLens 스타일 어댑터를 실행해 장면 내 (시작, 끝) 윈도우를 정교화한다.

8. **VLM 합성.** 질의 + 상위 3개 장면 클립(이미지 또는 짧은 클립으로) + 트랜스크립트로 Gemini 2.5 Pro를 호출한다. `(video_id, start_ms, end_ms)` 인용을 요구한다.

9. **평가.** ActivityNet-QA와 NeXT-GQA를 실행한다. 100개 질의 커스텀 세트를 만든다. 전체 정확도 + 부류별 분석(세기, 동작, 서술형)을 보고한다.

## 라이브러리로 써보기 (Use It)

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 산출물 (Ship It)

`outputs/skill-video-qa.md`가 결과물이다. YouTube URL 또는 업로드된 영상이 주어지면, 파이프라인은 장면을 인덱싱하고 타임스탬프가 달린 인용과 함께 질문에 답한다.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | 시간적 그라운딩 IoU | 홀드아웃 그라운딩 세트에 대한 intersection-over-union |
| 20 | QA 정확도 | NeXT-GQA 및 커스텀 100개 질의 |
| 20 | 수집 처리량 | 지출한 달러당 영상 시간 |
| 20 | UI 및 인용 UX | 타임스탬프 링크, 썸네일 스트립, 프레임으로 이동 |
| 15 | 환각률 | 세기 및 동작 유형 정확도 별도 |
| **100** | | |

## 연습 문제 (Exercises)

1. 캡셔닝 패스에서 Gemini 2.5 Pro를 Qwen3-VL-Max로 교체한다. 사람이 평가한 50개 장면 샘플에서 캡션 품질 차이를 보고한다.

2. 장면별 프레임 임베딩을 다중 벡터 대신 하나의 풀링된 벡터로 줄인다. 검색 회귀(retrieval regression)를 측정한다.

3. "세기 엄격(counting strict)" 모드를 만든다: 합성기가 세어진 각 인스턴스를 타임스탬프와 함께 추출하고 사용자가 클릭해 검증한다. 사용자 검증이 환각을 줄이는지 측정한다.

4. 수집 비용을 벤치마크한다: 세 가지 VLM 선택지에 걸친 달러당 영상 시간. 최적점을 고른다.

5. 화자 분리(speaker diarization) 트랜스크립트를 추가한다: 오디오에 pyannote 화자 분리를 실행하고 화자별 트랜스크립트를 임베딩한다. "Alice가 X에 대해 뭐라고 말했는가?" 질의를 시연한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| Scene segmentation | "샷 탐지(shot detection)" | 샷 경계에서 영상을 여러 장면으로 자르기 |
| Multi-vector index | "캡션 + 프레임 + 트랜스크립트" | 표현마다 명명된 벡터를 갖는 Qdrant 컬렉션 |
| Temporal grounding | "정확히 언제 일어났는가" | 질의 답변에 대한 (시작, 끝) 윈도우를 정교화하기 |
| Frame embedding | "시각적 표현" | 키프레임의 벡터 임베딩; 장면-시각 유사도에 사용됨 |
| RRF fusion | "reciprocal rank fusion" | 여러 순위 목록에 걸친 병합 전략; 고전적 하이브리드 검색 기법 |
| Counting hallucination | "오계수" | "X가 몇 개냐" 질문에 대한 VLM의 알려진 실패 모드 |
| ActivityNet-QA | "영상-QA 벤치마크" | 장편 영상 QA 정확도 벤치마크 |

## 더 읽을거리 (Further Reading)

- [AI2 Molmo 2](https://allenai.org/blog/molmo2): 오픈 VLM 체크포인트
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens): 대규모 시간적 그라운딩
- [Gemini Video long-context](https://deepmind.google/technologies/gemini): 호스팅형 레퍼런스
- [VideoDB](https://videodb.io): 영상용 CRUD API 레퍼런스
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io): 상용 레퍼런스
- [TransNetV2](https://github.com/soCzech/TransNetV2): 장면 분할 모델
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect): 고전적 오픈 대안
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467): 레퍼런스 평가 벤치마크
