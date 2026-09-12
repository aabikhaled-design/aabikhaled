# Capstone 08: 규제 산업을 위한 프로덕션 RAG 챗봇

> Harvey, Glean, Mendable, LlamaCloud는 모두 2026년에 같은 프로덕션(production) 형태를 운영한다. docling 또는 Unstructured와, 시각 자료를 위한 ColPali로 수집한다. 하이브리드 검색(hybrid search). bge-reranker-v2-gemma로 재순위(re-rank). 60-80% 적중률(hit rate)의 프롬프트 캐싱(prompt caching)을 사용해 Claude Sonnet 4.7로 합성. Llama Guard 4와 NeMo Guardrails로 가드(guard). Langfuse와 Phoenix로 관찰. 200개 질문 골든 셋(golden set)에서 RAGAS로 등급 매기기. 규제 도메인(법률, 임상, 보험)에서 하나 만들면, 캡스톤(capstone)은 골든 셋, 레드팀(red team), 드리프트(drift) 대시보드를 통과해야 한다.

**Type:** Capstone
**Languages:** Python (pipeline + API), TypeScript (chat UI)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18
**Time:** 30 hours

## 문제 (Problem)

규제 도메인 RAG(법률 계약, 임상 시험 프로토콜, 보험 약관)는 ROI가 명백하고 위험이 구체적이어서 2026년에 가장 많이 출시된 프로덕션 형태다. Harvey(Allen & Overy)는 법률용으로 이를 만들었다. Mendable은 개발자 문서 버전을 출시한다. Glean은 엔터프라이즈 검색을 다룬다. 패턴은 이렇다. 고충실도로 수집하고, 재순위와 함께 하이브리드로 검색하고, 인용(citation) 강제와 프롬프트 캐싱으로 합성하고, 여러 안전 계층으로 가드하고, 드리프트를 지속적으로 모니터링한다.

어려운 부분은 모델이 아니라 관할권을 인식하는 컴플라이언스(HIPAA, GDPR, SOC2), 인용 수준의 감사 가능성(auditability), 비용 통제(적중률이 높을 때 프롬프트 캐싱은 60-90% 할인을 가져온다), RAGAS 충실도(faithfulness)로 잡아내는 환각(hallucination) 탐지, 그리고 인덱스가 따라잡지 못한 채 원본 문서가 갱신될 때의 드리프트 탐지다. 이 캡스톤은 레드팀 스위트와 함께 200개 질문 골든 셋에서 이 모든 것을 출시하기를 요구한다.

## 개념 (Concept)

파이프라인에는 두 면이 있다. **수집(Ingestion)**: docling 또는 Unstructured가 구조화된 문서를 파싱한다. ColPali가 시각적으로 풍부한 것을 처리한다. 청크(chunk)는 요약, 태그, 역할 기반 접근 레이블을 얻는다. 벡터는 pgvector + pgvectorscale(5천만 벡터 미만) 또는 Qdrant Cloud로 간다. 희소(sparse) BM25가 함께 실행된다. **대화(Conversation)**: LangGraph가 메모리와 멀티턴(multi-turn)을 처리한다. 각 쿼리는 하이브리드 검색을 실행하고, bge-reranker-v2-gemma-2b로 재순위하고, Claude Sonnet 4.7(프롬프트 캐싱됨)로 합성하고, 출력을 Llama Guard 4와 NeMo Guardrails를 통과시키고, 인용이 앵커된 응답을 방출한다.

평가 스택에는 네 계층이 있다. 정확성을 위한 **골든 셋**(인용이 달린 200개의 레이블링된 Q/A). 안전성을 위한 **레드팀**(탈옥(jailbreak), PII 추출 시도, 도메인 밖 질문). 턴마다 자동으로 충실도 / 답변 관련성 / 컨텍스트 정밀도를 측정하는 **RAGAS**. 검색 품질과 환각 점수를 매주 지켜보는 **드리프트 대시보드**(Arize Phoenix).

프롬프트 캐싱은 비용 레버다. Claude 4.5+와 GPT-5+는 시스템 프롬프트 + 검색된 컨텍스트 캐싱을 지원한다. 60-80% 적중률에서는 쿼리당 비용이 3-5배 떨어진다. 높은 캐시 적중률을 달성하려면 파이프라인이 안정적인 접두사(접두사: 시스템 프롬프트 + 재순위된 컨텍스트 먼저)를 위해 설계되어야 한다.

## 아키텍처 (Architecture)

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 스택 (Stack)

- 수집: 구조화된 문서를 위한 Unstructured.io 또는 docling; 시각적으로 풍부한 PDF를 위한 ColPali
- 벡터 DB: 5천만 벡터 미만에서 pgvector + pgvectorscale; 그 외에는 Qdrant Cloud
- 희소: 필드 가중치를 갖춘 Tantivy BM25
- 오케스트레이션: LlamaIndex Workflows(수집) + LangGraph(대화)
- 재순위 모델: 자체 호스팅된 bge-reranker-v2-gemma-2b 또는 호스팅된 Voyage rerank-2
- LLM: 프롬프트 캐싱을 갖춘 Claude Sonnet 4.7; 폴백으로 자체 호스팅된 Llama 3.3 70B
- 평가: RAGAS 0.2 온라인, 환각 및 탈옥 스위트를 위한 DeepEval
- 관측성(observability): 주석 큐(annotation queue)를 갖춘 자체 호스팅 Langfuse; 드리프트를 위한 Arize Phoenix
- 가드레일: Llama Guard 4 입력/출력 분류기, NeMo Guardrails v0.12 정책, Presidio PII 제거
- 컴플라이언스: 청크에 대한 역할 기반 접근 레이블; GDPR/HIPAA를 위한 관할권 태그

```figure
canary-rollout
```

## 직접 만들기 (Build It)

1. **수집.** Unstructured 또는 docling으로 코퍼스(corpus)(진지한 빌드를 위해 1000-10000 문서)를 파싱한다. 스캔된 / 시각 자료가 많은 페이지는 ColPali로 라우팅한다. 요약, 역할 레이블, 관할권 태그가 달린 청크를 만든다.

2. **인덱스.** 밀집 임베딩(Voyage-3 또는 Nomic-embed-v2)을 pgvector + pgvectorscale로. Tantivy를 통한 BM25 사이드 인덱스. 페이로드로 역할 및 관할권 필터.

3. **하이브리드 검색.** 먼저 역할+관할권으로 필터링; 그다음 병렬 밀집 + BM25; 상호 순위 융합(reciprocal rank fusion)으로 병합; top-20을 재순위 모델로; top-5를 합성으로.

4. **프롬프트 캐싱으로 합성.** 캐시 헤더에 시스템 프롬프트 + 정적 정책; 캐시 확장으로 재순위된 컨텍스트; 캐싱되지 않은 접미사로 사용자 질문. 정상 상태에서 60-80% 캐시 적중률을 목표로 한다.

5. **가드레일.** 입력에 Llama Guard 4; NeMo Guardrails 레일이 도메인 밖 질문이나 정책 금지 주제를 차단; Presidio가 출력의 우발적 PII를 제거; 인용 강제 후처리 필터.

6. **골든 셋.** 도메인 전문가가 (답변, 인용)으로 레이블링한 200개 Q/A 쌍. 정확 인용 일치, 답변 정확성, 충실도(RAGAS)로 에이전트를 채점한다.

7. **레드팀.** 50개의 적대적 프롬프트: 탈옥(PAIR, TAP), PII 유출(exfiltration) 시도, 도메인 밖, 관할권 간 유출. 통과/실패와 심각도로 채점한다.

8. **드리프트 대시보드.** Arize Phoenix가 검색 품질(nDCG, 인용 충실도)을 매주 추적한다. 5% 하락 시 경보.

9. **비용 보고서.** Langfuse: 프롬프트 캐싱 적중률, 쿼리당 토큰(token), 단계별 쿼리당 $ 분해.

## 라이브러리로 써보기 (Use It)

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 산출물 (Ship It)

`outputs/skill-production-rag.md`은 결과물을 기술한다. 컴플라이언스 레이블과 함께 배포되고, 평가 기준을 통과하고, 실시간 드리프트 모니터링으로 관찰되는 규제 도메인 챗봇.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | RAGAS 충실도 + 답변 관련성 | 골든 셋(200 Q/A)에서의 온라인 점수 |
| 20 | 인용 정확성 | 검증 가능한 출처 앵커를 가진 답변의 비율 |
| 20 | 가드레일 커버리지 | Llama Guard 4 통과율 + 탈옥 스위트 결과 |
| 20 | 비용 / 지연 시간 엔지니어링 | 프롬프트 캐시 적중률, p95 지연 시간(latency), 쿼리당 $ |
| 15 | 드리프트 모니터링 대시보드 | 주간 검색 품질 추세를 갖춘 Phoenix 실시간 대시보드 |
| **100** | | |

## 연습 문제 (Exercises)

1. 다른 관할권 하의 두 번째 코퍼스 조각(예: GDPR과 함께 HIPAA)을 만든다. 20개 질문 관할권 간 프로브에서 역할+관할권 필터링이 상호 유출을 방지하는 것을 보여준다.

2. 일주일간의 프로덕션 트래픽에 걸쳐 프롬프트 캐시 적중률을 측정한다. 어떤 쿼리가 캐시 접두사를 깨뜨리는지 식별한다. 재구조화한다.

3. 1만 토큰 요약 버퍼를 가진 멀티턴 메모리를 추가한다. 대화가 길어지면서 충실도가 떨어지는지 측정한다.

4. Claude Sonnet 4.7을 자체 호스팅된 Llama 3.3 70B로 교체한다. 쿼리당 $와 충실도 차이를 측정한다.

5. "불확실(unsure)" 모드를 추가한다. 상위 재순위 점수가 임계값 아래이면, 에이전트는 답하는 대신 "확신할 만한 인용이 없다"고 말한다. 거짓 확신(false-confidence) 감소를 측정한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 프롬프트 캐싱(Prompt caching) | "캐시된 시스템 + 컨텍스트" | Claude/OpenAI 기능: 캐시된 접두 토큰이 적중 시 60-90% 할인 |
| RAGAS | "RAG 평가기" | 충실도, 답변 관련성, 컨텍스트 정밀도의 자동화된 채점 |
| 골든 셋(Golden set) | "레이블링된 평가" | 인용이 달린 200개 이상의 전문가 레이블링 Q/A; 정답 기준(ground truth) |
| 관할권 태그(Jurisdiction tag) | "컴플라이언스 레이블" | 청크에 붙은 GDPR/HIPAA/SOC2 범위; 검색 필터로 강제됨 |
| 인용 충실도(Citation faithfulness) | "근거 있는 답변 비율" | 검색 가능한 출처 범위로 뒷받침되는 주장의 비율 |
| 드리프트(Drift) | "검색 품질 감퇴" | nDCG나 인용 점수의 주간 변화; 경보 임계값 5% |
| 레드팀(Red team) | "적대적 평가" | 출시 전 탈옥, PII 추출, 도메인 밖 프로브 |

## 더 읽을거리 (Further Reading)

- [Harvey AI](https://www.harvey.ai): 레퍼런스 법률 프로덕션 스택
- [Glean enterprise search](https://www.glean.com): 엔터프라이즈 규모의 레퍼런스 RAG
- [Mendable documentation](https://mendable.ai): 개발자 문서 RAG 레퍼런스
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/): 매니지드 수집
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching): 비용 레버 레퍼런스
- [RAGAS 0.2 documentation](https://docs.ragas.io/): 정전(canonical)에 해당하는 RAG 평가 프레임워크
- [Arize Phoenix](https://github.com/Arize-ai/phoenix): 레퍼런스 드리프트 관측성
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/): 2026 안전 분류기
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/): 정책 레일 프레임워크
