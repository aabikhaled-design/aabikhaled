# LLaVA와 시각 명령어 튜닝(LLaVA and Visual Instruction Tuning)

> LLaVA(2023년 4월)는 지구상에서 가장 많이 복제된 멀티모달(multimodal) 아키텍처다. BLIP-2의 Q-Former를 2층 MLP로 대체하고, Flamingo의 게이트 교차 어텐션(gated cross-attention)을 단순한 토큰 연결(concatenation)로 대체했으며, 텍스트 전용 캡션으로부터 GPT-4가 생성한 15만 8천 개의 시각 명령어 턴(turn)으로 학습했다. 2023년에서 2026년 사이에 VLM을 만든 실무자라면 누구나 LLaVA의 어떤 변형을 만들었다. LLaVA-1.5는 AnyRes를 추가했다. LLaVA-NeXT는 해상도를 올렸다. LLaVA-OneVision은 이미지, 멀티 이미지, 비디오를 하나의 레시피로 통합했다. 이 레슨은 그 레시피를 읽고, 투영기(projector)를 구현하고, "더 단순한 것이 이긴" 이유를 설명한다.

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180분

## 학습 목표 (Learning Objectives)

- ViT 패치 임베딩(patch embedding)(차원 1024)을 LLM의 임베딩 차원(차원 4096)으로 매핑하는 2층 MLP 투영기 만들기.
- LLaVA 2단계 레시피 따라가기: (1) 55만 8천 개 캡션 쌍에 대한 투영기 정렬, (2) GPT-4가 생성한 15만 8천 개 턴에 대한 시각 명령어 튜닝.
- 이미지 토큰 플레이스홀더(placeholder), 시스템 프롬프트(prompt), 사용자/어시스턴트 턴으로 LLaVA 포맷 프롬프트 구성하기.
- Q-Former의 토큰 예산(token budget) 승리에도 불구하고 커뮤니티가 Q-Former에서 MLP로 옮겨 간 이유 설명하기.

## 문제 (The Problem)

BLIP-2의 Q-Former(Lesson 12.03)는 이미지를 32토큰으로 압축한다. 깔끔하고, 효율적이고, 벤치마크(benchmark)에 좋다. 하지만 두 가지 문제가 있다.

첫째, Q-Former는 학습 가능하지만 그 손실(loss)이 최종 작업이 아니다. 1단계는 ITC+ITM+ITG를 학습한다. 2단계는 LM 손실을 학습한다. 쿼리(query)는 LLM이 이후 디코딩(decoding)해야 하는 어떤 중간 표현을 학습한다. 정보가 병목(bottleneck)에서 손실된다.

둘째, Q-Former는 1억 8800만 파라미터(parameter)를 차지하며, LLaVA의 2023년 규모에서는 이를 목표 LLM과 함께 설계해야 했다. LLM을 바꾸면 Q-Former를 재학습한다. 비전 인코더(vision encoder)를 바꿔도 재학습한다. 모든 조합이 별도의 연구개발 프로젝트였다.

LLaVA의 답은 그 단순함이 민망할 정도였다. ViT의 576개 패치 토큰을 가져다, 각각을 2층 MLP(`1024 → 4096 → 4096`)에 통과시키고, 576개 전부를 LLM의 입력 시퀀스에 쏟아붓는다. 병목 없음. 이상한 목적에 대한 1단계 사전 학습 없음. 그저 직접적인 LM 손실로 MLP를 학습할 뿐.

데이터는 어디서 오는가? LLaVA의 두 번째 통찰: (텍스트 전용) GPT-4를 써서 명령어 데이터를 생성한다. 한 이미지에 대한 COCO 캡션과 바운딩 박스(bounding-box) 데이터를 GPT-4에 먹이고, 대화, 설명, 복잡한 추론 질문을 만들어 달라고 요청한다. 15만 8천 개의 명령어-응답 턴을 공짜로. 사람의 주석(annotation) 없음.

결과: 8개 A100에서 하루 동안 돌고, MMMU에서 Flamingo를 이기고, 커뮤니티가 확장할 수 있는 오픈 체크포인트를 출시한 VLM. 2023년 말에는 50개 이상의 포크(fork)를 낳았다.

## 개념 (The Concept)

### 아키텍처 (The architecture)

13B의 LLaVA-1.5:
- 비전 인코더: CLIP ViT-L/14 @ 336(1단계에서 동결, 2단계에서 선택적으로 동결 해제).
- 투영기: GELU 활성화(activation)를 쓰는 2층 MLP, `1024 → 4096 → 4096`.
- LLM: Vicuna-13B(이후 Llama-3.1-8B).

이미지 + 텍스트 프롬프트에 대한 순방향 패스(forward pass):

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

이미지는 LLM 컨텍스트(context)의 576토큰을 차지한다. 컨텍스트 2048에서는 텍스트에 1472토큰이 남는다. 컨텍스트 32k에서는 반올림 오차 수준이다.

### 1단계: 투영기 정렬 (Stage 1: projector alignment)

ViT를 동결한다. LLM을 동결한다. 2층 MLP만 학습한다. 데이터셋(dataset): 55만 8천 개 이미지-캡션 쌍(LAION-CC-SBU). 손실: 투영된 이미지 토큰에 조건화된, 캡션에 대한 언어 모델링.

배치(batch) 128에서 단일 에폭(epoch)으로 몇 시간이면 끝난다. 투영기는 ViT 공간을 LLM 공간으로 매핑하는 법을 학습한다. 작업별 지도 신호 없음.

### 2단계: 시각 명령어 튜닝 (Stage 2: visual instruction tuning)

투영기를 동결 해제한다(여전히 학습 가능). LLM을 동결 해제한다(보통 완전히, 때로는 LoRA). 15만 8천 개 시각 명령어 턴으로 학습한다.

명령어 데이터가 비결이다. Liu et al.은 다음과 같이 생성했다:
1. COCO 이미지 하나를 가져온다.
2. 텍스트 설명을 추출한다(사람이 쓴 캡션 5개 + 바운딩 박스 목록).
3. 세 가지 프롬프트 템플릿으로 GPT-4에 보낸다:
   - 대화(Conversation): "이 이미지에 대해 사용자와 어시스턴트 사이의 주고받는 대화를 생성하라."
   - 상세 설명(Detailed description): "이미지에 대한 풍부하고 상세한 설명을 하라."
   - 복잡한 추론(Complex reasoning): "이미지에 대한 추론이 필요한 질문을 하고, 그것에 답하라."
4. GPT-4의 출력을 (명령어, 응답) 쌍으로 파싱한다.

이 중 어느 것도 이미지를 직접 건드리지 않고 텍스트 설명만 쓴다. GPT-4는 그럴듯한 이미지 내용을 환각(hallucinate)한다. 잡음이 약간 있어도 동작했다. 15만 8천 턴이면 대화를 풀어내기에 충분했다.

### 커뮤니티가 이것을 복제한 이유 (Why the community copied this)

- 튜닝할 1단계 전용 손실이 없다. 처음부터 끝까지 LM 손실.
- 투영기가 며칠이 아니라 몇 시간 만에 학습된다.
- 투영기만 재학습하면 LLM을 교체할 수 있다(LLaVA-Llama2, LLaVA-Mistral, LLaVA-Llama3).
- 시각 명령어 데이터 파이프라인(pipeline)은 GPT-4를 쓰며, 새 도메인을 위해 재생성하기 저렴하다.

### LLaVA-1.5와 LLaVA-NeXT

LLaVA-1.5(2023년 10월)는 다음을 추가했다:
- 명령어 튜닝에 섞인 학술 작업 데이터(VQA, OKVQA, RefCOCO).
- 더 나은 시스템 프롬프트.
- 2048 → 32k 컨텍스트.

LLaVA-NeXT(2024년 1월)는 다음을 추가했다:
- AnyRes: 고해상도 이미지를 336x336 크롭(crop)의 2x2 또는 1x3 격자로 쪼개고, 더해서 하나의 전역 저해상도 썸네일. 각 크롭은 576토큰이 되고, 이미지당 총 약 2880개의 시각 토큰. OCR과 차트 작업이 도약했다.
- ShareGPT4V(고품질 GPT-4V 캡션)를 쓴 더 나은 명령어 데이터 혼합.
- 더 강력한 베이스 LLM(Mistral-7B, Yi-34B).

### LLaVA-OneVision

Lesson 12.08이 OneVision을 깊이 다룬다. 짧게 말하면: 같은 투영기이지만, 공유된 시각 토큰 예산으로 단일 이미지, 멀티 이미지, 비디오를 하나의 모델에서 다루는 커리큘럼으로 학습한다.

### Q-Former와의 비교 (The comparison to Q-Former)

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 이미지당 시각 토큰 | 32 | 576(기본) 또는 2880(AnyRes) |
| 학습 가능한 파라미터 | 1억 8800만 + LM | 4000만 + LM |
| 1단계 손실 | ITC+ITM+ITG | LM만 |
| LLM 드롭인(drop-in) | 재학습 필요 | 최소한의 재학습으로 교체 |
| 멀티 이미지 | 어색함 | 자연스러움(연결) |
| 비디오 | 어색함 | 자연스러움(프레임별 연결) |
| 토큰 예산 | 작음 | 큼 |

MLP는 단순함과 토큰 유연성에서 이긴다. Q-Former는 토큰 예산에서 이긴다. 하지만 2023년 말에는 토큰 예산이 더 이상 구속 제약(LLM 컨텍스트가 32k-128k+로 커졌다)이 아니었고, 단순함이 지배했다.

### 프롬프트 포맷 (The prompt format)

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>`는 플레이스홀더 토큰이다. 토큰화 전에, 576개 시각 토큰(또는 AnyRes로 2880개)으로 대체된다. 토크나이저(tokenizer)는 학습된 것보다 약간 더 긴 시퀀스를 보지만, 1단계가 가르쳤기 때문에 LLM은 새로운 입력을 처리한다.

### 파라미터 경제학 (Parameter economy)

LLaVA-1.5-7B 분해:
- CLIP ViT-L/14 @ 336: 3억 300만(1단계 동결, 종종 2단계 동결 해제).
- 투영기(선형 2개): 학습 가능 약 2200만.
- Llama-7B: 70억.
- 총합: 73억 파라미터. 2단계 중 학습 가능: 전체 70억 + 투영기 2200만.

2단계 학습 비용: 8xA100에서 약 20시간. 이것이 핵심 수치다. 하루, 한 노드, 재현 가능. 그래서 LLaVA가 퍼졌다.

```figure
mm-llava-projector
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 다음을 구현한다:

1. 순수 Python으로 구현한 2층 MLP 투영기(장난감 규모로 차원 16 → 32 → 32).
2. 프롬프트 빌딩 파이프라인: 시스템 프롬프트 + N개의 투영된 토큰으로 대체된 `<image>` + 사용자 턴 + 어시스턴트 생성 플레이스홀더.
3. 576토큰 시각 블록이 LLM 컨텍스트에서 어떻게 보이는지 시각화하는 도구(소비된 2k / 32k / 128k 컨텍스트의 백분율).

## 산출물 (Ship It)

이 레슨은 `outputs/skill-llava-vibes-eval.md`를 만든다. LLaVA 계열 체크포인트가 주어지면, 10개 프롬프트로 된 바이브-평가(vibes-eval) 모음(캡션 3개, VQA 3개, 추론 2개, 거부 2개)을 실행하고 사람이 읽을 수 있는 점수표를 보고한다. 벤치마크가 아니라, 투영기와 LLM이 잘 연결되는지 확인하는 스모크 테스트(smoke test)다.

## 연습 문제 (Exercises)

1. `1024 → 4096 → 4096`인 2층 MLP 투영기의 학습 가능 파라미터 개수를 계산하라. GELU와 편향(bias)을 쓰면, LLaVA-13B의 어느 비율을 차지하는가?

2. "거부" 사례에 대한 LLaVA 프롬프트를 구성하라. 이미지에 사적인 개인이 담겨 있다. 예상되는 어시스턴트 응답을 작성하라. 왜 LLaVA가 이것을 제로샷(zero-shot)으로 거부해야 하는가, 그리고 그 거부를 강화하려면 어떤 학습 데이터가 필요한가?

3. LLaVA-NeXT 블로그의 AnyRes 절을 읽어라. AnyRes에서 1344x672 이미지의 시각 토큰 개수를 계산하라. 336x336에서의 기본 576토큰과 비교하라.

4. LLaVA 1단계 투영기는 캡션에 대한 LM 손실로 학습된다. 1단계를 건너뛰고 곧장 2단계(시각 명령어 튜닝)로 가면 어떻게 되는가? 답을 위해 Prismatic VLMs 절제(ablation) 연구(arXiv:2402.07865)를 인용하라.

5. LLaVA-Instruct-150k는 COCO 캡션과 함께 GPT-4를 써서 명령어를 생성한다. 새 도메인(의료 X선, 위성 영상)에 대해, 도메인 명령어를 생성하는 4단계 데이터 파이프라인을 서술하라. 각 단계에서 무엇이 잘못될 수 있는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 투영기(Projector) | "MLP 다리" | ViT 차원을 LLM 차원으로 매핑하는, GELU를 쓰는 2층 MLP |
| 이미지 토큰(Image token) | "<image> 플레이스홀더" | 추론(inference) 전에 N개의 투영된 시각 토큰으로 대체되는 프롬프트 표식 |
| 시각 명령어 튜닝(Visual instruction tuning) | "LLaVA 2단계" | GPT-4가 생성한 (이미지, 명령어, 응답) 삼중쌍에 대한 학습 |
| 1단계 정렬(Stage 1 alignment) | "투영기 사전 학습" | ViT와 LLM을 동결하고, 캡션에 대한 LM 손실로 투영기를 학습 |
| AnyRes | "멀티 크롭 타일링" | 고해상도 이미지를 타일 격자로 쪼개고 각 타일의 시각 토큰을 연결 |
| LLaVA-Instruct | "GPT-4 생성" | COCO 캡션 + GPT-4로 합성한 15만 8천 개 명령어-응답 쌍 |
| 비전 인코더 동결(Vision encoder freeze) | "백본 잠금" | CLIP 가중치가 1단계에서 갱신되지 않으며, 때로는 2단계에서도 그렇다 |
| ShareGPT4V | "더 나은 캡션" | GPT-4V가 생성한 100만 개의 밀집 캡션. 더 높은 품질의 정렬에 쓰임 |
| VQA | "시각 질의응답" | 이미지에 대한 자유형식 질문에 답하는 작업 |
| Prismatic VLMs | "설계 공간 논문" | 투영기와 데이터 선택을 체계적으로 시험한 Karamcheti 2024 절제 연구 |

## 더 읽을거리 (Further Reading)

- [Liu et al.: Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) LLaVA 논문.
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5.
- [Chen et al.: ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) 밀집 캡션 데이터셋.
- [Karamcheti et al.: Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) 설계 공간 절제 연구.
- [Li et al.: LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) 통합된 단일 이미지, 멀티 이미지, 비디오.
