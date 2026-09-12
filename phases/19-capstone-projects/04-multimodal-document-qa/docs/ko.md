# Capstone 04: 멀티모달 문서 QA (Vision-First PDF, Tables, Charts)

> 2026년의 문서 QA 프런티어는 OCR 후 텍스트(OCR-then-text)에서 비전 우선(vision-first) 늦은 상호작용(late interaction)으로 이동했다. ColPali, ColQwen2.5, ColQwen3-omni는 각 PDF 페이지를 이미지로 취급해 다중 벡터(multi-vector) 늦은 상호작용으로 임베딩(embedding)하고, 쿼리가 패치(patch)에 직접 어텐션(attention)하게 한다. 재무 10-K, 과학 논문, 손글씨 노트에서 이 패턴은 OCR 우선 방식을 큰 차이로 앞선다. 1만 페이지에 대해 파이프라인을 처음부터 끝까지 만들고, OCR 후 텍스트 방식과 나란히 비교한 결과를 발행하라.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (viewer UI)
**Prerequisites:** Phase 4 (computer vision), Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17
**Time:** 30 hours

## 문제 (Problem)

기업들은 OCR 파이프라인이 망가뜨리는 PDF 위에 앉아 있다. 회전된 표가 있는 스캔된 10-K, 방정식이 빽빽한 과학 논문, 이미지로만 말이 되는 차트, 손글씨 주석. 이것들을 텍스트 우선으로 취급하면 신호의 절반을 잃는다. 2026년의 답은 원시 페이지 이미지에 대한 늦은 상호작용 다중 벡터 검색이다. ColPali(Illuin Tech)가 이 방식을 도입했고, ColQwen2.5-v0.2와 ColQwen3-omni가 정확도를 끌어올렸다. ViDoRe v3에서 비전 우선 검색은 OCR 후 텍스트보다 의미 있는 차이로 높은 점수를 받으며, 그 격차는 차트, 표, 손글씨에서 더 벌어진다.

트레이드오프(trade-off)는 저장 공간과 지연 시간(latency)이다. ColQwen 임베딩은 단일 1024차원 벡터가 아니라 페이지당 ~2048개의 패치 벡터다. 원시 저장 공간이 급격히 늘어난다. DocPruner(2026)는 측정 가능한 정확도 손실 없이 50% 가지치기(pruning)를 해낸다. 1만 페이지를 색인해 ViDoRe v3 nDCG@5를 측정하고, 답변을 2초 미만으로 서빙하며 OCR 후 텍스트 베이스라인(baseline)과 직접 비교한다.

## 개념 (Concept)

늦은 상호작용은 모든 쿼리 토큰(token)이 모든 패치 토큰에 대해 점수를 매기고, 쿼리 토큰당 최대 점수를 합산한다는 뜻이다. 단일 풀링(pooled) 벡터 없이 세밀한 매칭을 얻는다. 다중 벡터 인덱스(Vespa, Qdrant 다중 벡터, 또는 AstraDB)는 패치별 임베딩을 저장하고 검색 시점에 MaxSim을 실행한다.

답변기(answerer)는 쿼리와 검색된 top-k 페이지를 이미지로 받아, 증거 영역(bounding box 또는 페이지 참조)과 함께 답을 쓰는 비전-언어 모델(vision-language model)이다. Qwen3-VL-30B, Gemini 2.5 Pro, InternVL3이 2026년 프런티어 선택지다. 방정식과 과학 표기법을 위해, OCR 폴백(Nougat, dots.ocr)이 선택적 텍스트 채널로 끼워진다.

평가는 2차원 행렬(matrix)이다. 한 축: 콘텐츠 유형(일반 텍스트 문단, 빽빽한 표, 막대/선 차트, 손글씨 노트, 방정식). 다른 축: 검색 접근법(비전 우선 늦은 상호작용 대 OCR 후 텍스트 대 하이브리드). 각 셀은 nDCG@5와 답변 정확도를 얻는다. 그 보고서가 결과물이다.

## 아키텍처 (Architecture)

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 스택 (Stack)

- 페이지 렌더링: PyMuPDF(fitz), 180 DPI, 세로 방향으로 정규화
- 늦은 상호작용 모델: ColQwen2.5-v0.2 또는 ColQwen3-omni (Hugging Face의 vidore 팀)
- 인덱스: 다중 벡터 필드를 갖춘 Vespa, 또는 Qdrant 다중 벡터, 또는 MaxSim을 갖춘 AstraDB
- 가지치기: DocPruner 2026 정책(고분산 패치를 유지, 정확도 손실 < 0.5%에서 50% 압축)
- OCR 폴백(방정식 / 빽빽한 표): dots.ocr 또는 Nougat
- VLM 답변기: 자체 호스팅된 Qwen3-VL-30B 또는 호스팅된 Gemini 2.5 Pro; 폴백으로 InternVL3
- 평가: ViDoRe v3 벤치마크(benchmark), 다중 페이지 추론을 위한 M3DocVQA
- 뷰어 UI: 증거 영역을 위한 캔버스 오버레이를 갖춘 Next.js 15

```figure
ce-late-interaction
```

## 직접 만들기 (Build It)

1. **수집(Ingest).** 10-K, 과학 논문, 스캔된 문서에 걸친 1만 PDF 페이지 코퍼스(corpus)를 순회한다. 각 페이지를 1536x2048 PNG로 렌더링한다. `{doc_id, page_num, image_path}`를 영속화한다.

2. **임베딩(Embed).** 각 페이지 이미지에 ColQwen2.5-v0.2를 실행한다. 출력 형태는 차원 128의 ~2048개 패치 임베딩이다. DocPruner를 적용해 가장 신호가 강한 절반을 유지한다. Vespa 다중 벡터 필드 또는 Qdrant 다중 벡터에 쓴다.

3. **쿼리(Query).** 들어오는 각 쿼리에 대해, 쿼리 타워(query tower, 토큰 수준 임베딩)로 임베딩한다. 인덱스에 대해 MaxSim을 실행한다. 모든 쿼리 토큰에 대해, 페이지 패치 임베딩에 대한 최대 내적(dot-product)을 취하고 합산한다. top-k 페이지를 반환한다.

4. **합성(Synthesize).** 쿼리와 top-5 페이지 이미지로 Qwen3-VL-30B를 호출한다. 프롬프트: "제공된 페이지만 사용해 답하라. 각 주장을 (doc_id, page)로 인용하고 영역(그림, 표, 문단)을 명명하라."

5. **증거 영역.** 답변을 후처리하여 인용된 영역을 추출한다. VLM이 bounding box를 방출하면(Qwen3-VL은 그렇다), 뷰어에서 오버레이로 렌더링한다.

6. **OCR 폴백.** 방정식이 빽빽한 것으로 식별된 페이지(이미지 분산에 대한 휴리스틱)에 대해, Nougat 또는 dots.ocr을 실행하고 OCR 텍스트를 이미지와 함께 추가 채널로 전달한다.

7. **평가(Eval).** ViDoRe v3(검색 nDCG@5)과 M3DocVQA(다중 페이지 QA 정확도)를 실행한다. 같은 합성기로 같은 코퍼스에 대해 OCR 후 텍스트 파이프라인도 실행한다. 콘텐츠 유형 × 접근법 행렬을 만든다.

8. **UI.** 먼저 Streamlit 프로토타입; 페이지별 증거 영역 오버레이를 갖춘 Next.js 15 프로덕션(production) 뷰어.

## 라이브러리로 써보기 (Use It)

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 산출물 (Ship It)

`outputs/skill-doc-qa.md`은 결과물을 기술한다. 특정 코퍼스에 맞춰 튜닝되고 ViDoRe v3에서 OCR 후 텍스트 베이스라인과 비교 평가된, 비전 우선 멀티모달 문서 QA 시스템이다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 정확도 | OCR 텍스트 베이스라인 및 공개된 리더보드 대비 벤치마크 수치 |
| 20 | 증거 영역 근거화(grounding) | 실제로 답변 범위를 담고 있는 인용된 영역의 비율 |
| 20 | 저장 공간과 지연 시간 엔지니어링 | DocPruner 압축 비율, 인덱스 p95, 답변 p95 |
| 20 | 다중 페이지 추론 | 손으로 레이블링한 100개 질문 다중 페이지 셋에서의 정확도 |
| 15 | 출처 검사 UX | 뷰어 명료성, 오버레이 충실도, 나란히 비교 도구 |
| **100** | | |

## 연습 문제 (Exercises)

1. 같은 코퍼스에서 ColQwen2.5-v0.2 대 ColQwen3-omni를 측정한다. 한쪽이 맞히고 다른 쪽이 놓치는 페이지는 어떤 것인가? 유형별로 라우팅하기 위해 인덱스에 "content class" 태그를 추가한다.

2. 임베딩을 공격적으로(75%, 90%) 가지치기한다. 압축 절벽(compression cliff)을 찾는다. ViDoRe nDCG@5가 OCR 베이스라인 아래로 떨어지는 지점이다.

3. 하이브리드를 만든다. OCR 후 텍스트와 ColQwen을 병렬로 실행하고, RRF로 융합하고, 크로스 인코더(cross-encoder)로 재순위(rerank)한다. 하이브리드가 둘 중 어느 쪽 단독보다 나은가? 어디서 가장 도움이 되는가?

4. Qwen3-VL-30B를 더 작은 VLM(Qwen2.5-VL-7B)으로 교체한다. 달러당 정확도 곡선을 측정한다.

5. 손글씨 노트 지원을 추가한다. 손글씨 코퍼스를 렌더링하고, ColQwen으로 임베딩하고, 검색을 측정한다. 손글씨 OCR 파이프라인과 비교한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 늦은 상호작용(Late interaction) | "ColPali 스타일 검색" | 쿼리 토큰이 페이지 패치에 독립적으로 점수를 매김; MaxSim이 집계 |
| 다중 벡터(Multi-vector) | "패치별 임베딩" | 각 문서가 단일 풀링 벡터가 아니라 여러 벡터를 가짐 |
| MaxSim | "늦은 상호작용 채점" | 모든 쿼리 토큰에 대해, 문서 벡터에 대한 최대 유사도를 취함; 합산 |
| DocPruner | "패치 압축" | 무시할 만한 정확도 손실로 패치의 50%를 유지하는 2026 가지치기 |
| ViDoRe v3 | "문서 검색 벤치마크" | 시각 문서 검색을 측정하는 2026 표준 |
| 증거 영역(Evidence region) | "인용된 bounding box" | 답변 범위를 국소화하는 출처 페이지의 bbox |
| OCR 폴백(OCR fallback) | "방정식 채널" | 방정식이나 표가 많은 페이지에 대해 비전과 함께 사용되는 텍스트 파이프라인 |

## 더 읽을거리 (Further Reading)

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali): 레퍼런스 늦은 상호작용 문서 검색
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449): 기초가 되는 방법론 논문
- [ColQwen family on Hugging Face](https://huggingface.co/vidore): 프로덕션 준비된 체크포인트
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952): 다중 페이지 멀티모달 RAG 베이스라인
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html): 레퍼런스 서빙 스택
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors): 대안 인덱스
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html): 대안 매니지드 인덱스
- [Nougat OCR](https://github.com/facebookresearch/nougat): 방정식 처리가 가능한 OCR 폴백
