# Capstone 02: 코드베이스에 대한 RAG (Cross-Repo Semantic Search)

> 2026년에 진지한 모든 엔지니어링 조직은 단순한 문자열이 아니라 의미를 이해하는 내부 코드 검색을 운영한다. Sourcegraph Amp, Cursor의 코드베이스 답변, Augment의 엔터프라이즈 그래프, Aider의 repomap, Pinterest의 내부 MCP: 모두 같은 형태다. 여러 저장소를 수집하고, tree-sitter로 파싱하고, 함수 및 클래스 수준의 청크(chunk)를 임베딩(embedding)하고, 하이브리드 검색(hybrid-search)하고, 재순위(re-rank)를 매기고, 인용(citation)과 함께 답한다. 이 캡스톤(capstone)에서는 10개 저장소에 걸친 200만 줄의 코드를 처리하고 매 git push마다 증분 재색인(incremental re-indexing)을 견디는 시스템을 만든다.

**Type:** Capstone
**Languages:** Python (ingestion), TypeScript (API + UI)
**Prerequisites:** Phase 5 (NLP foundations), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P17
**Time:** 30 hours

## 문제 (Problem)

2026년이면 모든 프런티어 코딩 에이전트(coding agent)는 코드베이스 검색 계층을 탑재하고 출시된다. 컨텍스트 윈도우(context window)만으로는 저장소를 가로지르는 질문을 풀 수 없기 때문이다. Claude의 100만 토큰(token) 컨텍스트는 도움이 되지만, 순위가 매겨진 검색의 필요성을 없애지는 못한다. 원시 청크에 대한 순진한 코사인(cosine) 검색은 생성된 코드, 모노레포(monorepo) 중복, 그리고 거의 임포트되지 않는 심볼(symbol)들의 롱테일(long tail)에서 결과를 오염시킨다. 프로덕션(production)의 답은 재순위 모델(re-ranker)을 갖춘, AST를 인식하는 청크에 대한 하이브리드(밀집(dense) + BM25) 검색이며, 심볼 참조의 그래프로 뒷받침된다.

이것을 배우려면 튜토리얼용 저장소 하나가 아니라 실제 플릿(fleet)을 색인하고, MRR@10, 인용 충실도(citation faithfulness), 증분 신선도(incremental freshness)를 측정해야 한다. 실패는 대부분 인프라에서 온다. 10만 파일짜리 모노레포, 파일의 절반을 다시 건드리는 push, 정확히 답하려면 네 개 저장소를 가로질러야 하는 쿼리다.

## 개념 (Concept)

AST를 인식하는 수집 파이프라인(pipeline)은 각 파일을 tree-sitter로 파싱하고, 함수 및 클래스 노드를 추출하고, 고정된 토큰 윈도우가 아니라 노드 경계에서 청킹(chunk)한다. 각 청크는 세 가지 표현을 얻는다. 밀집 임베딩(Voyage-code-3 또는 nomic-embed-code), 희소(sparse) BM25 용어, 그리고 짧은 자연어 요약이다. 요약은 검색 가능한 세 번째 양식(modality)을 더한다. 사용자가 "X는 어떻게 인가되는가"라고 물으면, 코드에는 `check_permission`만 있더라도 요약이 "authz"를 언급한다.

검색은 하이브리드다. 쿼리는 밀집 검색과 BM25 검색을 모두 발동시키고, top-k를 병합하고, 그 합집합을 크로스 인코더(cross-encoder) 재순위 모델(Cohere rerank-3 또는 bge-reranker-v2-gemma-2b)에 넘긴다. 재순위가 매겨진 목록은 모든 주장을 파일과 줄 범위로 인용하라는 지시와 함께 장문맥(long-context) 합성기(Claude Sonnet 4.7, 프롬프트 캐싱(prompt caching) 적용, 또는 자체 호스팅된 Llama 3.3 70B)로 간다. 인용이 없는 답변은 후처리 필터(post-filter)에 의해 거부된다.

증분 신선도는 인프라 문제다. Git push가 diff를 트리거해, 어떤 파일과 어떤 심볼이 바뀌었는지 가려낸다. 영향받은 청크만 다시 임베딩한다. 영향받은 파일 간 심볼 간선(imports, method calls)이 재계산된다. 인덱스는 매 커밋마다 200만 줄을 재처리하지 않고도 일관성을 유지한다.

## 아키텍처 (Architecture)

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 스택 (Stack)

- 파싱: 17개 언어 문법(Python, TS, Rust, Go, Java, C++ 등)을 갖춘 tree-sitter
- 밀집 임베딩: Voyage-code-3(호스팅) 또는 nomic-embed-code-v1.5(자체 호스팅), bge-code-v1 폴백
- 희소 인덱스: BM25F를 갖춘 Tantivy(Rust), 심볼 이름 대 본문에 필드 가중치 부여
- 벡터 DB: 하이브리드 검색을 갖춘 Qdrant 1.12, 또는 5천만 벡터 미만 팀을 위한 pgvector + pgvectorscale
- 청크 요약 모델: Claude Haiku 4.5 또는 Gemini 2.5 Flash, 프롬프트 캐싱 적용
- 재순위 모델: Cohere rerank-3 또는 자체 호스팅된 bge-reranker-v2-gemma-2b
- 오케스트레이션: 수집용 LlamaIndex Workflows, 쿼리 에이전트용 LangGraph
- 합성기: 프롬프트 캐싱을 갖춘 Claude Sonnet 4.7 (100만 컨텍스트)
- 심볼 그래프: import 및 call 간선을 위한 Neo4j(매니지드) 또는 kuzu(임베디드)
- 관측성(observability): 검색 + 합성 단계마다 Langfuse 스팬(span)

```figure
ce-hybrid-retrieval
```

## 직접 만들기 (Build It)

1. **수집 워커(ingestion walker).** 매 push 훅마다 git 히스토리를 순회한다. 변경된 파일을 수집한다. 각 파일에 대해 tree-sitter로 파싱하고, 전체 소스 범위와 함께 함수 및 클래스 노드를 추출한다. 청크 레코드 `{repo, path, start_line, end_line, symbol, body}`를 방출한다.

2. **청크 요약기.** 시스템 서두에 프롬프트 캐싱을 적용해 청크들을 Haiku 4.5 호출로 배치(batch) 처리한다. 프롬프트: "이 함수를 한 문장으로 요약하되, 그 공개 계약(public contract)과 부작용(side effects)을 명시하라." 요약을 청크와 함께 저장한다.

3. **임베딩 풀.** 두 개의 병렬 큐: 밀집(Voyage-code-3 배치 128)과 요약(같은 모델이지만 요약 문자열에 대해). 페이로드 `{repo, path, start_line, end_line, symbol, kind}`와 함께 벡터를 Qdrant에 쓴다.

4. **BM25 인덱스.** 필드 가중치를 부여한 Tantivy 인덱스: 심볼 이름 가중치 4, 심볼 본문 가중치 1, 요약 가중치 2. "X라는 이름의 함수를 찾아라" 쿼리를 "X를 하는 함수를 찾아라"와 함께 가능하게 한다.

5. **심볼 그래프.** 각 청크에 대해 간선을 기록한다. imports(이 파일은 저장소 Z의 심볼 Y를 사용한다), calls(이 함수는 클래스 C의 메서드 M을 호출한다), inheritance. kuzu에 저장한다. 쿼리 시점에 저장소 경계를 가로질러 검색을 확장하는 데 사용된다.

6. **쿼리 에이전트.** 세 개의 노드를 가진 LangGraph. `retrieve`는 밀집 + BM25를 병렬로 발동시키고, (repo, path, symbol)로 중복을 제거한다. `rerank`는 top-50에 크로스 인코더를 실행하고 top-10을 유지한다. `synth`는 재순위가 매겨진 청크를 컨텍스트에 담아 Claude Sonnet 4.7을 호출하고, 시스템 프롬프트를 캐싱하고, file:line 인용을 요구한다.

7. **인용 강제.** 모델 출력을 파싱한다. `(repo/path:start-end)` 앵커가 없는 모든 주장은 재질의(re-ask) 대상으로 표시되거나 버려진다. 인용된 부분만 담은 답변을 사용자에게 반환한다.

8. **증분 재색인.** 매 웹훅마다 심볼 수준의 diff를 계산한다. 텍스트가 바뀐 청크만 다시 임베딩한다. imports가 바뀐 청크에 대해 심볼 간선을 재계산한다. 측정: 200만 LOC 플릿에서 50파일 push가 60초 이내에 재색인된다.

9. **평가.** 100개의 저장소 간 질문에 골드(gold) file:line 답을 레이블링(label)한다. MRR@10, nDCG@10, 인용 충실도(검증 가능한 앵커를 가진 주장의 비율), 그리고 p50/p99 지연 시간(latency)을 측정한다.

## 라이브러리로 써보기 (Use It)

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 산출물 (Ship It)

결과물 스킬 `outputs/skill-codebase-rag.md`. 저장소 코퍼스(corpus)가 주어지면, 수집 파이프라인, 하이브리드 인덱스, 쿼리 에이전트를 띄우고, 어떤 저장소 간 질문에 대해서도 인용된 답변을 반환한다. 평가 기준:

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 검색 품질 | 100개 질문 홀드아웃(held-out) 셋에서 MRR@10과 nDCG@10 |
| 20 | 인용 충실도 | 검증 가능한 file:line 앵커를 가진 답변 주장의 비율 |
| 20 | 지연 시간과 규모 | 색인된 코퍼스 크기에서 10k QPS 시 p95 쿼리 지연 시간 |
| 20 | 증분 색인 정확성 | 50파일 커밋에서 git push로부터 검색 가능까지의 시간 |
| 15 | UX와 답변 포매팅 | 인용 클릭 가능성, 스니펫(snippet) 미리보기, 후속 질문 어포던스(affordance) |
| **100** | | |

## 연습 문제 (Exercises)

1. Voyage-code-3을 자체 호스팅된 nomic-embed-code로 교체한다. MRR@10 차이를 측정한다. 재순위를 활성화하면 격차가 좁혀지는지 보고한다.

2. 코퍼스에 생성된 코드(LLM이 만든 보일러플레이트) 20%를 주입하고 재평가한다. 검색 오염을 관찰한다. 페이로드에 "generated" 플래그를 추가하고 그 히트들의 가중치를 낮춘다.

3. 자신의 코퍼스 크기에서 Qdrant 하이브리드 검색과 pgvector + pgvectorscale을 벤치마크(benchmark)한다. 배치 크기 1에서 p99를 보고한다.

4. 샘플링(sampling) 기반 드리프트(drift) 검사를 추가한다. 매주 100개 질문 평가를 다시 실행한다. MRR@10이 5% 넘게 떨어지면 경보를 울린다.

5. 언어 간 심볼 해석으로 확장한다. gRPC로 Go 서비스를 호출하는 Python 함수. 심볼 그래프를 사용해 둘을 연결한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| AST 인식 청킹(AST-aware chunking) | "함수 수준 분할" | 고정된 토큰 윈도우 대신 tree-sitter 노드 경계에서 코드를 자르는 것 |
| 하이브리드 검색(Hybrid search) | "밀집 + 희소" | BM25와 벡터 검색을 병렬로 실행하고, top-k를 병합하고, 재순위 |
| 크로스 인코더 재순위(Cross-encoder rerank) | "2단계 순위" | 각 (쿼리, 후보) 쌍을 함께 채점하는 모델, 코사인보다 정확 |
| 프롬프트 캐싱(Prompt caching) | "캐시된 시스템 프롬프트" | 반복되는 접두 토큰을 최대 90%까지 할인하는 2026 Claude / OpenAI 기능 |
| 심볼 그래프(Symbol graph) | "코드 그래프" | 파일과 저장소를 가로지르는 imports, calls, inheritance 간선 |
| 인용 충실도(Citation faithfulness) | "근거 있는 답변 비율" | 앵커를 클릭하고 참조된 범위를 읽어 사용자가 검증할 수 있는 주장의 비율 |
| 증분 재색인(Incremental re-index) | "push-to-search 시간" | git push로부터 변경된 심볼이 쿼리 가능해질 때까지의 실제 경과 시간 |

## 더 읽을거리 (Further Reading)

- [Sourcegraph Amp](https://ampcode.com): 프로덕션 저장소 간 코드 인텔리전스
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase): 이 캡스톤의 레퍼런스 심층 분석
- [Aider repo-map](https://aider.chat/docs/repomap.html): tree-sitter 순위 저장소 뷰
- [Augment Code enterprise graph](https://www.augmentcode.com): 상용 심볼 그래프 RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/): 레퍼런스 구현
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings): Voyage-code-3 세부사항
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank): 크로스 인코더 레퍼런스
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering): 내부 플랫폼 레퍼런스
