# ColPali와 비전 네이티브 문서 RAG

> 전통적인 RAG는 PDF를 텍스트로 파싱하고 청크(chunk)로 분할한 뒤, 청크를 임베딩(embedding)하고 벡터(vector)를 저장한다. 모든 단계에서 신호가 사라진다. OCR은 차트 데이터를 떨어뜨리고, 청킹은 표 행을 끊으며, 텍스트 임베딩은 그림을 무시한다. ColPali(Faysse et al., 2024년 7월)는 더 단순한 질문을 던졌다. 왜 텍스트를 추출하는가? PaliGemma로 페이지 이미지를 직접 임베딩하고, 검색에는 ColBERT 스타일의 지연 상호작용(late interaction)을 쓰며, 문서가 담은 레이아웃, 그림, 폰트, 서식 신호를 모두 유지하라. 발표된 벤치마크(benchmark)를 보면, 시각적으로 풍부한 문서에서 텍스트-RAG보다 종단간 정확도가 20~40% 더 높다. ColQwen2, ColSmol, VisRAG가 이 패턴을 확장했다. 이 레슨은 비전 네이티브 RAG 명제를 읽고 작은 ColPali 같은 인덱서를 만든다.

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180분

## 학습 목표 (Learning Objectives)

- 이중 인코더(bi-encoder) 검색(문서당 하나의 벡터)과 지연 상호작용 검색(문서당 여러 벡터)의 차이를 설명하기.
- ColBERT의 MaxSim 연산과, ColPali가 그것을 텍스트 토큰(token)에서 이미지 패치로 일반화하는 방식을 설명하기.
- 작은 ColPali 같은 인덱서를 만들기: 페이지 → 패치 임베딩 → 질의 항(query-term) 임베딩에 대한 MaxSim → 상위 k개 페이지.
- 청구서 / 재무 보고서 활용 사례에서 ColPali + Qwen2.5-VL 생성기와 텍스트-RAG + GPT-4를 비교하기.

## 문제 (The Problem)

PDF에 텍스트-RAG를 쓰면 문서의 대부분을 버린다. 재무 보고서의 3분기 매출 성장은 보통 차트에 있고, 의료 보고서의 소견은 주석이 달린 이미지에 있으며, 법률 계약의 서명란은 텍스트 사실이 아니라 레이아웃 사실이다.

텍스트-RAG 파이프라인(pipeline):

1. PDF → OCR / pdftotext를 통한 텍스트.
2. 텍스트 → 300~500 토큰 청크.
3. 청크 → 이중 인코더 임베딩(하나의 벡터).
4. 사용자 질의 → 임베딩 → 코사인 유사도 → 상위 k개 청크.
5. 청크 + 질의 → LLM.

다섯 개의 손실 단계. 차트는 포착되지 않는다. 표는 청크에 걸쳐 끊긴다. 다단 레이아웃은 평평해진다. 그림 주석은 사라진다.

ColPali의 해법은 OCR을 건너뛰고 페이지 이미지를 직접 임베딩하는 것이다. 검색에 ColBERT 스타일의 지연 상호작용을 써서 모델이 질의 시점에 세밀한 패치에 어텐션(attention)하게 한다.

## 개념 (The Concept)

### ColBERT (2020)

ColBERT(Khattab & Zaharia, arXiv:2004.12832)는 텍스트 검색 방법이다. 문서당 하나의 벡터 대신 토큰당 하나의 벡터를 생성한다. 질의 시점에:

- 질의 토큰은 자신의 임베딩을 갖는다(N_q개 벡터).
- 문서 토큰은 임베딩을 갖는다(N_d개 벡터, 보통 캐시됨).
- 점수 = 질의 토큰에 대한, 문서 토큰에 대한 코사인 유사도의 최댓값의 합: Σ_i max_j cos(q_i, d_j).

이것이 MaxSim 연산이다. 각 질의 토큰이 자신에게 가장 잘 맞는 문서 토큰을 "고른다". 최종 점수는 그 합이다.

장점은 강한 회상과 항(term) 수준 의미 처리다. 단점은 문서당 N_d개 벡터라 저장 비용이 비싸다는 것이다.

### ColPali

ColPali(Faysse et al., arXiv:2407.01449)는 ColBERT 패턴을 이미지에 적용한다.

- 각 페이지는 PaliGemma(ViT + 언어)가 패치 임베딩으로 인코딩한다: 페이지당 N_p개 벡터.
- 각 사용자 질의(텍스트)는 질의 토큰 임베딩으로 인코딩된다: N_q개 벡터.
- 점수 = Σ_i max_j cos(q_i, p_j), 즉 질의 텍스트 토큰과 페이지 이미지 패치에 대한 MaxSim.
- 총 점수로 상위 k개 페이지를 검색한다.

문서 입력 시점에는 모든 페이지를 PaliGemma로 임베딩하고 모든 패치 임베딩을 저장한다. 질의 시점에는 질의 토큰을 임베딩하고, 저장된 모든 페이지 임베딩에 대해 MaxSim을 계산해 상위 k개 페이지를 반환한다.

장점은 시각적으로 풍부한 문서에서 종단간으로 텍스트-RAG를 20~40% 이긴다는 것이다. 각 패치 벡터가 국소 레이아웃과 내용을 포착한다.

단점은 페이지당 N_p개 패치 × 4바이트 부동소수점 × D차원 벡터라 저장이 빠르게 늘어난다는 점이다. PQ / OPQ 양자화로 완화한다.

### ColQwen2와 ColSmol

ColQwen2(illuin-tech, 2024~2025)는 PaliGemma를 Qwen2-VL로 교체한다. 기반 인코더가 나아지면서 검색도 나아진다.

ColSmol은 로컬 / 엣지 사용을 위한 더 작은 규모의 변형이다. 약 1B 파라미터(parameter)의 ColSmol 검색기는 소비자용 GPU에서 실행된다.

### VisRAG

VisRAG(Yu et al., arXiv:2410.10594)는 다른 변형이다. 패치에 대한 MaxSim 대신, 각 페이지를 VLM으로 하나의 벡터로 풀링(pooling)한 다음 이중 인코더로 검색한다. 인덱싱이 더 빠르고 저장이 더 작지만, 회상은 더 약하다.

품질 대 비용 트레이드오프(trade-off): 품질에는 ColPali, 규모에는 VisRAG.

### M3DocRAG

M3DocRAG(Cho et al., arXiv:2411.04952)는 멀티모달 검색을 다중 페이지 다중 문서 추론으로 확장한다. 여러 문서에 걸쳐 페이지를 검색하고, VLM을 위한 다중 페이지 컨텍스트를 구성한다.

### ViDoRe: 벤치마크

ColPali의 동반 벤치마크. Visual Document Retrieval Evaluation. 과제에는 재무 보고서, 과학 논문, 행정 문서, 의료 기록, 매뉴얼이 포함된다. 지표: nDCG@5.

ColPali-v1은 ViDoRe에서 약 80% nDCG@5를 기록한다; 동일 문서에 대한 텍스트-RAG는 약 50~60%를 기록한다.

### 종단간 RAG 파이프라인

비전 네이티브 RAG의 경우:

1. 입력: PDF → 페이지 이미지 → PaliGemma 인코딩 → 모든 패치 임베딩 저장.
2. 질의: 사용자 텍스트 → 질의 토큰 임베딩 → 인덱싱된 모든 페이지에 대한 MaxSim → 상위 k개 페이지.
3. 생성: 상위 k개 페이지 이미지 + 질의 → VLM(Qwen2.5-VL 또는 Claude) → 답.

어디에도 OCR이 없다. 그림, 차트, 폰트, 레이아웃이 모두 답으로 흘러 들어간다.

### 저장 계산

페이지당 729개 패치와 128차원 임베딩을 갖는 50페이지 재무 보고서:

- ColPali: 50 * 729 * 128 * 4 바이트 = 원시 약 18 MB, PQ 후 약 4 MB.
- 텍스트-RAG: 50개 청크 * 768차원 * 4 바이트 = 약 150 kB.

ColPali는 문서당 저장이 약 30배 많다. 규모가 커지면 OPQ / PQ가 이를 약 5~10배로 낮추며, 대개 견딜 만하다.

### 텍스트-RAG가 여전히 이기는 경우

- 레이아웃 신호가 없는 순수 텍스트 문서(위키 글, 채팅 로그). 텍스트-RAG가 더 단순하고 저장이 저렴하다.
- 저장이 비용을 지배하는 수백만 페이지 아카이브.
- 검색과 함께 추출 가능한 OCR 텍스트를 요구하는 엄격한 규제 요건.

2026년의 그 밖의 모든 것(재무 보고서, 과학 논문, 법률 계약, 의료 기록, UX 문서) 에서는 비전 네이티브 RAG가 이긴다.

```figure
mm-maxsim
```

## 라이브러리로 써보기 (Use It)

`code/main.py`:

- 장난감 패치 인코더: "페이지"(특성 벡터의 작은 그리드)를 패치 임베딩 배열로 매핑한다.
- MaxSim 스코어러: 질의 토큰 임베딩 집합과 페이지 패치 집합 사이의 ColBERT 스타일 점수를 계산한다.
- 장난감 페이지 5개를 인덱싱하고, 질의 3개를 실행해 점수와 함께 상위 k개를 반환한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-vision-rag-designer.md`를 생성한다. 문서 RAG 프로젝트가 주어지면, ColPali / ColQwen2 / VisRAG / 텍스트-RAG를 선택하고 저장 크기를 정한다.

## 연습 문제 (Exercises)

1. 페이지당 729개 패치, 128차원 임베딩, 4바이트 부동소수점의 200페이지 연차 보고서. 원시 저장과 PQ 압축(8배) 저장을 계산하라.

2. MaxSim은 Σ_i max_j cos(q_i, p_j)다. 이 합이 단순 평균 유사도가 포착하지 못하는 무엇을 포착하는가?

3. ColPali는 페이지를 패치 집합으로 인덱싱한다. 대신 (ColBERT처럼) 단어 수준에서 인덱싱하면 무엇이 바뀌는가? 트레이드오프는?

4. 질의당 500ms 지연 시간(latency) 예산의 100만 페이지 코퍼스에 대한 종단간 파이프라인을 설계하라. ColQwen2 / VisRAG를 고르고 정당화하라.

5. M3DocRAG(arXiv:2411.04952)를 읽으라. 다중 페이지 어텐션 패턴과 그것이 단일 페이지 ColPali 검색과 어떻게 다른지 설명하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 지연 상호작용(Late interaction) | "ColBERT 스타일" | 단일 문서 벡터가 아니라 토큰별 또는 패치별 임베딩 + MaxSim을 사용하는 검색 |
| MaxSim | "패치에 대한 최댓값" | 각 질의 토큰에 대해 유사도가 가장 높은 문서 토큰을 고름; 질의 전반에 걸쳐 합산 |
| 이중 인코더(Bi-encoder) | "단일 벡터" | 문서당 하나의 벡터; 더 빠르지만 세밀함을 잃음 |
| 다중 벡터(Multi-vector) | "문서당 여러 벡터" | 문서 / 페이지당 N_p개 벡터 저장; 저장 비용이 늘지만 회상이 개선됨 |
| 패치 임베딩(Patch embedding) | "페이지 특성" | VLM 인코더에서 나온 이미지 패치당 하나의 벡터, 페이지별로 캐시됨 |
| ViDoRe | "비전 문서 벤치" | 시각 문서 검색을 위한 ColPali의 벤치마크 모음 |
| PQ 양자화(PQ quantization) | "곱 양자화(Product quantization)" | 벡터 유사도를 유지하면서 저장을 약 8배 줄이는 압축 |

## 더 읽을거리 (Further Reading)

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
