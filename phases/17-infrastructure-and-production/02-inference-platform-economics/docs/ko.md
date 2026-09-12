# 추론 플랫폼 경제학(Inference Platform Economics): Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> 2026년 추론(inference) 시장은 더 이상 GPU 시간 임대가 아니다. 시장은 커스텀 실리콘(custom silicon)(Groq, Cerebras, SambaNova), GPU 플랫폼(Baseten, Together, Fireworks, Modal), API 우선 마켓플레이스(Replicate, DeepInfra)로 갈라진다. Fireworks는 2026년 5월 1일 GPU당 시간당 1달러 가격을 인상했고, 하루 10조 토큰(token) 이상에서 40억 달러 가치 평가가 나온다는 사실은 볼륨 주도 모델이 작동한다는 뜻이다. Baseten은 2026년 1월 50억 달러 가치로 3억 달러 시리즈 E를 마감했다. 경쟁 포지셔닝 규칙은 단순하다: Fireworks는 지연 시간(latency)을 최적화하고, Together는 카탈로그 폭을 최적화하고, Baseten은 엔터프라이즈 완성도를 최적화하고, Modal은 Python 네이티브 DX를 최적화하고, Replicate는 멀티모달(multimodal) 도달 범위를 최적화하고, Anyscale은 분산 Python을 최적화한다. 이 레슨은 창업자에게 건넬 수 있는 매트릭스를 준다.

**Type:** Learn
**Languages:** Python (stdlib, toy per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 세 가지 시장 세그먼트(커스텀 실리콘, GPU 플랫폼, API 우선)를 명명하고 각 벤더를 세그먼트에 대응시키기.
- "토큰당(per-token)" API 가격 모델이 왜 하드웨어가 아니라 서빙 엔진(serving engine)의 비용 곡선 쪽으로 압축되는지 설명하기.
- 최소 세 개 벤더에 걸쳐 요청당 실효 비용을 계산하고, 분당(per-minute)(Baseten, Modal)이 토큰당을 이기는 시점을 설명하기.
- 주어진 워크로드(서버리스 버스트성, 안정적 고처리량, 파인튜닝(fine-tuning)된 변형, 멀티모달)에 어느 플랫폼이 올바른 기본값인지 식별하기.

## 문제 (The Problem)

매니지드 하이퍼스케일러(hyperscaler) 플랫폼들을 평가한 뒤 더 좁고 빠른 공급자가 필요하다고 결정했다고 하자. 지연 시간에는 Fireworks, 폭에는 Together, 파인튜닝된 커스텀 모델에는 Baseten이다. 이제 여섯 개의 실제 선택지가 있는데 가격 페이지가 서로 맞아떨어지지 않는다. Fireworks는 100만 토큰당 달러를 보여주고, Baseten은 분당 달러를, Modal은 초당 달러를, Replicate는 예측(prediction)당 달러를 보여준다. 워크로드를 모델링하지 않고는 이것들을 정면으로 비교할 수 없다.

더 까다로운 점은 각 가격 페이지 뒤의 비즈니스 모델이 다르다는 것이다. Fireworks는 공유 GPU에서 자체 커스텀 엔진(FireAttention)을 돌린다. 토큰당 요금은 그들의 사용률 곡선을 반영한다. Baseten은 Truss + 전용 GPU를 준다. 분당 요금은 배타성을 반영한다. Modal은 진정한 Python 서버리스(serverless)다. 1초 미만 콜드 스타트(cold start)와 함께 초당 청구. 같은 출력(LLM 응답), 세 가지 다른 비용 함수.

이 레슨은 여섯 개를 모델링하고 각각이 언제 이기는지 알려준다.

## 개념 (The Concept)

### 세 가지 세그먼트

**커스텀 실리콘**: Groq(LPU), Cerebras(WSE), SambaNova(RDU). 같은 모델에서 일반적으로 GPU 기반 클러스터보다 디코드(decode)가 5~10배 빠르다. 더 높은 토큰당 가격(Groq는 2025년 말 Llama-70B에서 약 100만당 0.99달러)이지만 지연 시간에 민감한 사용 사례에는 따라올 수 없다. Groq는 음성 에이전트(voice agent)와 실시간 번역의 프로덕션(production) 선택이다.

**GPU 플랫폼**: Baseten, Together, Fireworks, Modal, Anyscale. NVIDIA(2026년 H100, H200, B200) 또는 때때로 AMD에서 돌아간다. "순수 GPU 임대"(RunPod, Lambda)와 "하이퍼스케일러 매니지드 서비스"(Bedrock) 사이의 경제 계층.

**API 우선 마켓플레이스**: Replicate, DeepInfra, OpenRouter, Fal. 넓은 카탈로그, 예측당 또는 초당 과금, 첫 호출까지의 시간을 강조한다.

### Fireworks: 지연 시간 최적화 GPU 플랫폼

- FireAttention 엔진(커스텀). 동등 구성에서 vLLM보다 지연 시간 4배 낮음으로 마케팅된다.
- 비대화형 워크로드를 위한 서버리스 요율의 약 50% 배치(batch) 등급.
- 파인튜닝된 모델을 기본 모델과 같은 요율로 서빙: LoRA에 프리미엄을 부과하는 공급자 대비 진짜 차별화 요소.
- 2026년 중반: 2026년 5월 1일부터 온디맨드(on-demand) GPU 임대를 시간당 1달러 인상. 규모에서 볼륨 가격은 협상 가능.
- 재무 신호: 40억 달러 가치 평가, 하루 10조 토큰 이상 처리.

### Together: 폭 최적화

- 업스트림 공개 후 며칠 내 오픈소스 릴리스를 포함해 200개 이상의 모델.
- 동등 LLM 모델에서 Replicate보다 50~70% 저렴: "AI Native Cloud" 포지셔닝은 볼륨과 카탈로그다.
- 추론 + 파인튜닝 + 학습(training)을 하나의 API에서.

### Baseten: 엔터프라이즈 완성도 최적화

- Truss 프레임워크: 의존성, 시크릿(secrets), 서빙 설정을 하나의 매니페스트(manifest)로 묶는 모델 패키징.
- T4부터 B200까지의 GPU 범위. 합리적인 콜드 스타트 완화를 갖춘 분당 청구.
- SOC 2 Type II, HIPAA 준비. 흔한 핀테크(fintech)·헬스케어 선택.
- 50억 달러 가치 평가, 2026년 1월 시리즈 E(CapitalG, IVP, NVIDIA로부터 3억 달러).

### Modal: Python 네이티브 최적화

- 순수 Python으로 된 코드형 인프라(infrastructure-as-code). 함수에 `@modal.function(gpu="A100")`을 데코레이트하고 한 명령으로 배포한다.
- 초당 청구. 사전 예열(pre-warming)로 콜드 스타트 2~4초. 작은 모델은 1초 미만.
- 8,700만 달러 시리즈 B, 11억 달러 가치 평가(2025년). 독립 설문조사에서 가장 강력한 개발자 경험 점수.

### Replicate: 멀티모달 폭

- 예측당 과금. 이미지, 비디오, 오디오 모델의 기본 플랫폼.
- 통합 생태계(Zapier, Vercel, CMS 플러그인).
- LLM 토큰당 요율에서는 덜 경쟁력 있지만 멀티모달 다양성에서 이긴다.

### Anyscale: Ray 네이티브

- Ray 위에 구축됨. RayTurbo는 Anyscale의 독점 추론 엔진(vLLM과 경쟁)이다.
- 추론 단계가 더 큰 그래프의 한 노드인 분산 Python 워크로드에 가장 적합하다.
- 매니지드 Ray 클러스터. Ray AIR 및 Ray Serve와의 긴밀한 통합.

### 토큰당 vs 분당: 각각이 언제 이기는가

토큰당은 워크로드가 지연 시간에 둔감하고 버스트성일 때 의미가 있다. 사용한 만큼만 낸다. 분당은 사용률이 높고 예측 가능할 때 의미가 있다. GPU를 포화시키면 토큰당을 이긴다.

대략적 규칙: 전용 GPU의 지속 사용률(sustained utilization)이 약 30%를 넘는 워크로드에서는 분당(Baseten, Modal)이 토큰당(Fireworks, Together)을 이기기 시작한다. 그 아래에서는 유휴에 대한 지불을 피하므로 토큰당이 이긴다.

### 커스텀 엔진이 진짜 해자

vLLM과 SGLang 위의 모든 플랫폼은 커스텀 엔진을 주장한다. FireAttention, RayTurbo, Baseten의 추론 스택. 커스텀 엔진 주장은 마케팅의 색을 띤다. 정직한 프레이밍은 vLLM + SGLang이 프로덕션 오픈소스 추론의 약 80%를 차지하며, 플랫폼 계층의 차별화 요소는 DX, 귀속(attribution), SLA라는 것이다.

### 기억해야 할 숫자

- Fireworks GPU 임대: 2026년 5월 1일부터 시간당 1달러 인상.
- Fireworks 주장: 동등 구성에서 vLLM보다 지연 시간 4배 낮음.
- Together: LLM에서 Replicate보다 50~70% 저렴.
- Baseten 가치 평가: 50억 달러(시리즈 E, 2026년 1월, 3억 달러 라운드).
- Modal 가치 평가: 11억 달러(시리즈 B, 2025년).
- 약 30% 지속 사용률 위에서 분당이 토큰당을 이긴다.

```figure
cost-per-token
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 가격 모델 전반에 걸쳐 합성 워크로드에서 여섯 벤더를 비교한다. 하루 달러와 실효 100만 토큰당 달러를 보고한다. 실행하여 토큰당과 분당 사이의 손익분기점(break-even)을 찾아라.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-inference-platform-picker.md`를 만들어낸다. 워크로드 프로필, SLA, 예산이 주어지면, 주(primary) 추론 플랫폼을 고르고 차점자를 명명한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 하나의 H100에서 70B 모델에 대해 어느 지속 사용률에서 Baseten(분당)이 Fireworks(토큰당)를 이기는가? 교차점을 직접 유도하고 경험칙과 비교하라.
2. 어떤 제품이 이미지 생성, 채팅, 음성-텍스트(speech-to-text)를 서빙한다고 하자. 모달리티마다 플랫폼을 고르고 이들을 통합하는 게이트웨이(gateway) 패턴을 명명하라.
3. Fireworks가 주 모델에서 시간당 1달러 가격을 올린다. 트래픽의 40%가 배치 등급(50% 할인)으로 이동할 경우 혼합 비용 영향을 모델링하라.
4. 규제 대상 고객이 SOC 2 Type II + HIPAA + 전용 GPU를 요구한다. 어느 세 플랫폼이 유효하며 어느 것이 FinOps에서 이기는가?
5. Fireworks 서버리스, Together 온디맨드, Baseten 전용, Replicate API에서 Llama 3.1 70B에 대한 1,000 예측당 비용을 비교하라. 하루 10 예측에서 어느 것이 가장 저렴한가? 10,000에서는?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| Custom silicon | "비-GPU 칩" | Groq LPU, Cerebras WSE, SambaNova RDU: 디코드에 최적화됨 |
| FireAttention | "Fireworks 엔진" | 커스텀 어텐션 커널. vLLM보다 지연 시간 4배 낮음으로 마케팅됨 |
| Truss | "Baseten의 포맷" | 모델 패키징 매니페스트. 의존성 + 시크릿 + 서빙 설정 |
| Per-token | "API 가격" | 소비된 토큰으로 과금. 유휴에 대해 지불 안 함 |
| Per-minute | "전용 가격" | 실제 경과 GPU 시간으로 과금. 높은 사용률에서 이김 |
| Per-prediction | "Replicate 가격" | 모델 호출당 과금. 이미지/비디오에 흔함 |
| RayTurbo | "Anyscale 엔진" | Ray 위의 독점 추론. Ray 클러스터에서 vLLM과 경쟁 |
| Batch tier | "50% 할인" | 인하된 요율의 비대화형 큐. Fireworks, OpenAI에 흔함 |
| Fine-tuned at base rate | "Fireworks LoRA" | LoRA로 서빙된 요청을 기본 모델 요율로 과금(차별화 요소) |

## 더 읽을거리 (Further Reading)

- [Fireworks Pricing](https://fireworks.ai/pricing): 토큰당 요율, 배치 등급, GPU 임대.
- [Baseten Pricing](https://www.baseten.co/pricing/): 분당 요율, 약정 용량, 엔터프라이즈 등급.
- [Modal Pricing](https://modal.com/pricing): 초당 GPU 요율과 무료 등급.
- [Together AI Pricing](https://www.together.ai/pricing): 모델 카탈로그와 토큰당 요율.
- [Anyscale Pricing](https://www.anyscale.com/pricing): RayTurbo와 매니지드 Ray 가격.
- [Northflank: Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) 비교 평가.
- [Infrabase: AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) 벤더 지형.
