# 매니지드 LLM 플랫폼(Managed LLM Platforms): Bedrock, Vertex AI, Azure OpenAI

> 세 곳의 하이퍼스케일러(hyperscaler), 세 가지 뚜렷한 전략. AWS Bedrock은 모델 마켓플레이스(marketplace)다. Claude, Llama, Titan, Stability, Cohere가 하나의 API 뒤에 모여 있다. Azure OpenAI는 OpenAI와의 독점 파트너십에 더해 전용 용량을 위한 Provisioned Throughput Units(PTUs)를 제공한다. Vertex AI는 Gemini를 우선하며, 가장 뛰어난 롱컨텍스트(long-context)와 멀티모달(multimodal) 스토리를 갖는다. 2026년 Artificial Analysis는 Llama 3.1 405B 동급 모델에서 Azure OpenAI를 중앙값 약 50ms, Bedrock을 약 75ms로 측정한다. 전용 용량이 공유 온디맨드(on-demand)를 이기기 때문에 PTU가 이 격차를 설명한다. 의사결정 규칙은 "어느 것이 가장 빠른가"가 아니라 "어느 모델 카탈로그와 FinOps 표면이 내 제품에 맞는가"이다. 이 레슨은 감(vibes)이 아니라 트레이드오프(trade-off)를 적어 두고 선택하는 법을 가르친다.

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 세 가지 플랫폼 전략(마켓플레이스 vs 독점 vs Gemini 우선)을 명명하고 각각을 제품 사용 사례에 대응시키기.
- Azure OpenAI에서 Provisioned Throughput Units(PTUs)가 무엇을 사주는지, 그리고 온디맨드 Bedrock이 405B 규모에서 일반적으로 왜 약 25ms 더 느리게 읽히는지 설명하기.
- 각 플랫폼의 FinOps 귀속(attribution) 표면을 도식화하기(Bedrock Application Inference Profiles vs Vertex 팀별 프로젝트 vs Azure 스코프 + PTU 예약).
- "최소 2개 공급자(two-provider minimum)" 정책을 적어 보고, 2026년에 단일 벤더 종속(lock-in)이 왜 비싼 실수인지 설명하기.

## 문제 (The Problem)

제품에 쓸 모델로 Claude 3.7 Sonnet을 골랐다고 하자. 이제 이 모델을 서빙해야 한다. Anthropic API를 직접 호출하거나, AWS Bedrock으로 호출하거나, 게이트웨이(gateway)를 거치면 된다. 직접 API가 가장 단순하다. Bedrock은 BAA, VPC 엔드포인트, IAM, CloudWatch 귀속을 추가한다. 게이트웨이는 페일오버(failover), 통합 청구(billing), 그리고 공급자 전반의 속도 제한(rate limit)을 추가한다.

더 깊은 질문은 카탈로그다. Claude와 Llama와 Gemini가 한 제품에 모두 필요하다면, 한 곳이 Bedrock과 Vertex와 Azure OpenAI를 동시에 끼고 있지 않은 한 이를 한곳에서 다 살 수는 없다. 하이퍼스케일러는 서로 갈아 끼울 수 없다. 모델 계층(model layer)을 누가 소유하느냐를 두고 각자 다른 베팅을 했기 때문이다.

이 레슨은 그 세 가지 베팅, 지연 시간 격차, FinOps 격차, 그리고 종속 위험을 지도화한다.

## 개념 (The Concept)

### 세 가지 전략

**AWS Bedrock**: 마켓플레이스. Claude(Anthropic), Llama(Meta), Titan(AWS 자체 제품), Stability(이미지), Cohere(임베딩), Mistral에 더해 이미지·임베딩 하위 카탈로그까지. 하나의 API, 하나의 IAM 표면, 하나의 CloudWatch 익스포트. Bedrock의 베팅은 고객이 단일 모델보다 선택권(optionality)을 더 원한다는 것이다.

**Azure OpenAI**(독점 파트너십. Azure 데이터센터에서 GPT-4 / 4o / 5 / o 시리즈, DALL·E, Whisper, 그리고 OpenAI 모델의 파인튜닝(fine-tuning)을 얻는다. "Azure OpenAI Service" 카탈로그에는 OpenAI 외 모델이 없다) 그것들은 Azure AI Foundry(별도 제품)로 간다. Azure의 베팅은 OpenAI가 프런티어(frontier)로 남고 고객이 바로 그 특정 관계에 대해 엔터프라이즈 제어를 원한다는 것이다.

**Vertex AI**(Gemini 우선, 그 외 모든 것은 그다음. Gemini 1.5 / 2.0 / 2.5 Flash와 Pro, 그리고 Model Garden(서드파티). Vertex의 베팅은 멀티모달 롱컨텍스트다) 100만 토큰(token) Gemini 컨텍스트가 차별화 요소다.

### 규모에서의 지연 시간 격차

Artificial Analysis는 지속적인 벤치마크(benchmark)를 돌린다. 동급 Llama 3.1 405B 배포(공유 온디맨드)에서 Azure OpenAI의 첫 토큰 지연 시간(first-token latency) 중앙값은 약 50ms다. Bedrock은 약 75ms다. 이 격차는 AWS의 실패가 아니라 용량 모델(capacity model)의 차이다. Azure는 한 테넌트(tenant)를 위해 GPU 용량을 예약해 주는 PTU(Provisioned Throughput Units)를 판매한다. Bedrock의 동등 상품(Provisioned Throughput)도 존재하지만 단위당 시간당 약 21달러부터 시작하며, 대부분의 고객은 공유 온디맨드에 머문다.

온디맨드 공유 용량은 다른 모든 고객의 트래픽과 경쟁하지만, 전용 용량은 그렇지 않다. 제품 SLA가 P99에서 TTFT < 100ms라면, Azure에서 PTU를 사거나, Bedrock Provisioned Throughput을 사거나, 기본 변동성을 받아들이거나 셋 중 하나다.

### Provisioned Throughput 경제학

Azure PTU: 예약된 추론 컴퓨트 블록. 예측 가능한 워크로드(workload)에 대해 온디맨드 대비 최대 약 70% 절감. 트래픽과 무관하게 시간당 비용이 고정된다. 유휴 상태일 때도 예약에 대해 비용을 낸다. 손익분기점(break-even)은 보통 약 40~60%의 지속 사용률(sustained utilization) 부근이다.

Bedrock Provisioned Throughput: 모델과 리전(region)에 따라 시간당 21~50달러. 비슷한 계산: 손익분기점은 피크 사용률의 절반 부근이다. 월 단위 약정이 필요하다.

Vertex 프로비저닝 용량은 Gemini SKU별로 판매된다. 가격은 모델과 리전에 따라 다르며 공개적으로 덜 광고된다.

### FinOps 표면: 진짜 차별화 요소

**Bedrock Application Inference Profiles**는 마켓플레이스에서 가장 깔끔한 귀속이다. 프로필에 `team`, `product`, `feature` 태그를 달고 모든 모델 호출을 이 프로필로 라우팅하면, CloudWatch가 후처리 없이 프로필별 비용을 분해해 준다. 2025년에 추가된 기능으로, 여전히 가장 세분화된 하이퍼스케일러 네이티브 기능이다.

**Vertex** 귀속은 팀별 프로젝트(project-per-team)에 모든 곳의 레이블(label)을 더한 것이다. 각 팀을 GCP 프로젝트로 모델링하고, 모든 리소스에 레이블을 붙이고, 집계를 위해 BigQuery Billing Export + DataStudio를 사용한다. 작업량은 더 많지만, BigQuery는 비용 데이터에 대해 임의의 SQL을 가능하게 한다.

**Azure**는 구독/리소스 그룹 스코프에 태그를 더한 것에 의존하며, PTU 예약을 일급(first-class) 비용 객체로 둔다. 태그는 요청이 아니라 리소스 그룹에서 상속되므로, 요청별 귀속은 Application Insights 사용자 정의 지표나 헤더를 찍어 주는 게이트웨이가 필요하다.

패턴: Bedrock은 네이티브로 가장 깔끔하고, Vertex는 BigQuery를 통해 가장 유연하며, Azure는 계측(instrument)하지 않으면 가장 불투명하다.

### 종속은 2026년의 위험

단일 하이퍼스케일러 약정은 하나의 모델이 지배하던 시절에는 괜찮았다. 2026년에는 프런티어가 매달 움직인다. 한 분기는 Claude 3.7, 다음은 Gemini 2.5, 그다음은 GPT-5다. 한 플랫폼에 묶이면 프런티어의 3분의 2에서 발이 묶인다.

일 잘하는 팀들이 채택하는 패턴은 이렇다. 제품에 중요한 모든 LLM 호출에 최소 2개 공급자를 둔다. Bedrock + Azure OpenAI가 흔한 조합으로, 한쪽에서 Claude, 다른 쪽에서 GPT를 쓰고 둘 사이에 페일오버를 두며 게이트웨이는 하나로 묶는다. 게이트웨이가 최적으로 라우팅하므로 비용 상승은 미미하다. 장애(2025년 1월 Azure OpenAI 사건, AWS us-east-1 중단 같은) 시의 가용성 상승은 결정적이다.

### 데이터 거주성, BAA, 그리고 규제 산업

Bedrock: 대부분 리전에서 BAA. VPC 엔드포인트. 가드레일(guardrails). 흔한 핀테크(fintech) 기본값.
Azure OpenAI: HIPAA, SOC 2, ISO 27001. EU 데이터 거주성. 엔터프라이즈 규제 기본값.
Vertex: HIPAA, GDPR, 리전별 데이터 거주성. Google Cloud의 컴플라이언스(compliance) 스택.

세 곳 모두 기본 체크박스를 충족한다. 차이는 데이터 보존 정책, 로그를 다루는 방식, 그리고 남용 모니터링(abuse-monitoring)이 트래픽을 읽는지 여부(대부분 기본 옵트인이며, 엔터프라이즈에는 옵트아웃 가능)에 있다.

### 기억해야 할 숫자

- Llama 3.1 405B 동급에서 Azure OpenAI 중앙값 TTFT: 약 50ms (PTU 사용 시).
- Bedrock 온디맨드 중앙값 TTFT: 약 75ms.
- Bedrock Provisioned Throughput: 단위당 시간당 21~50달러.
- Azure PTU 손익분기점: 약 40~60% 지속 사용률.
- 높은 사용률에서 온디맨드 대비 PTU 절감: 최대 70%.

```figure
i4-platform-lanes
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 합성 워크로드에서 세 플랫폼을 비교하면서 온디맨드 vs PTU 경제학, TTFT 변동성, 비용 귀속 충실도(fidelity)를 모델링한다. 실행하여 PTU가 어디서 이득을 보는지, 그리고 어디서 마켓플레이스의 모델 폭이 TTFT 격차를 능가하는지 확인하라.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-managed-platform-picker.md`를 만들어낸다. 워크로드 프로필(필요한 모델, TTFT SLA, 일일 볼륨, 컴플라이언스 요건)이 주어지면, 주(primary) 플랫폼, 폴백(fallback), 그리고 FinOps 계측 계획을 추천한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 70B급 모델에서 어느 지속 사용률에서 Azure PTU가 온디맨드를 이기는가? 손익분기점을 계산하고 광고된 40~60% 구간과 비교하라.
2. 제품에 Claude 3.7 Sonnet과 GPT-4o가 모두 필요하다고 하자. 두 공급자 배포를 설계하라. 어느 모델이 어느 하이퍼스케일러로 가는가, 앞단에 어떤 게이트웨이가 놓이는가, 페일오버 정책은 무엇인가?
3. 규제 대상 헬스케어 고객이 BAA, US-East 데이터 거주성, 그리고 100ms 미만 P99 TTFT를 요구한다. 플랫폼을 고르고 세 가지 구체적 기능으로 정당화하라.
4. 트래픽은 그대로인데 Bedrock 청구액이 이번 달 4배로 늘었다고 하자. Application Inference Profiles 없이는 어떻게 범인을 찾겠는가? 프로필이 있으면 얼마나 걸리는가?
5. Azure OpenAI와 Bedrock 가격 페이지를 읽어라. 월 1억 토큰 Claude 워크로드에 대해 어느 것이 더 저렴한가: Anthropic API 직접, Bedrock 온디맨드, 또는 Bedrock Provisioned Throughput?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| Bedrock | "AWS LLM 서비스" | Claude, Llama, Titan, Mistral, Cohere를 아우르는 모델 마켓플레이스 |
| Azure OpenAI | "Azure의 ChatGPT" | 엔터프라이즈 제어를 갖춘, Azure 데이터센터의 독점 OpenAI 모델 |
| Vertex AI | "Google의 LLM" | 서드파티 모델을 위한 Model Garden을 갖춘 Gemini 우선 플랫폼 |
| PTU | "전용 용량" | Provisioned Throughput Unit: 예약된 추론 GPU, 시간당 가격 책정 |
| Application Inference Profile | "Bedrock 태깅" | 태그를 가진 제품별 비용/사용량 프로필, CloudWatch 네이티브 |
| Model Garden | "Vertex 카탈로그" | Gemini와 별개인 Vertex AI의 서드파티 모델 섹션 |
| Two-provider minimum | "LLM 이중화" | 모든 중요한 LLM 경로를 2개 이상의 하이퍼스케일러에 걸쳐 운영하는 정책 |
| BAA | "HIPAA 서류" | Business Associate Agreement. PHI에 필요하며, 세 곳 모두 제공 |
| Abuse monitoring | "로그 감시자" | 프롬프트/출력에 대한 공급자 측 안전 스캔. 엔터프라이즈에서 옵트아웃 가능 |

## 더 읽을거리 (Further Reading)

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/): 권위 있는 요금표와 Provisioned Throughput 가격.
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/): PTU 경제학과 요금표.
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing): Gemini 등급과 Model Garden 추가 요금.
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/): 공급자 전반의 지속적인 지연 시간·처리량 벤치마크.
- [The AI Journal: AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) 엔터프라이즈 의사결정 프레임워크.
- [Finout: Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) 귀속 메커니즘 비교.
