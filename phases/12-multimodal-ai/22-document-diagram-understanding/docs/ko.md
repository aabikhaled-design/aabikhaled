# 문서와 다이어그램 이해

> 문서는 사진이 아니다. PDF, 과학 논문, 청구서, 손으로 쓴 양식에는 레이아웃, 표, 다이어그램, 각주, 머리글처럼 일반적인 이미지 이해가 포착하지 못하는 의미 구조가 있다. VLM 이전의 스택은 파이프라인(pipeline)이었다. Tesseract OCR + LayoutLMv3 + 표 추출 휴리스틱이다. VLM 물결은 이를 OCR 없는 모델(Donut(2022), Nougat(2023), DocLLM(2023))로 대체했고, 이들은 구조화된 마크업을 직접 내보낸다. 2026년에 이르러 프런티어는 그저 "페이지 이미지를 2576px 네이티브로 Claude Opus 4.7에 공급"하는 것이고, 구조화된 마크업 출력은 덤으로 따라온다. 이 레슨은 문서 AI의 3시대 궤적을 읽는다.

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180분

## 학습 목표 (Learning Objectives)

- 문서 AI의 세 시대를 설명하기: OCR 파이프라인, OCR 없는 방식, VLM 네이티브.
- LayoutLMv3의 세 입력 스트림을 설명하기: 텍스트, 레이아웃(bbox), 이미지 패치, 통합 마스킹과 함께.
- Donut(OCR 없음, 이미지 → 마크업), Nougat(과학 논문 → LaTeX), DocLLM(레이아웃 인식 생성형), PaliGemma 2(VLM 네이티브)를 비교하기.
- 새로운 과제(청구서, 과학 논문, 손으로 쓴 양식, 중국어 영수증)에 맞는 문서 모델을 고르기.

## 문제 (The Problem)

"이 PDF를 이해하라"는 보기보다 어렵다. 정보는 다음에 자리한다:

- 텍스트 내용(신호의 90%).
- 레이아웃(머리글, 각주, 사이드바, 2단 형식).
- 표(행, 열, 병합된 셀).
- 그림과 다이어그램.
- 손으로 쓴 주석.
- 폰트와 타이포그래피(제목 대 본문).

원시 OCR은 텍스트를 덤프하고 나머지를 잃는다. 청구서를 다루는 시스템은 "합계: $1,245"가 각주가 아니라 오른쪽 아래에서 왔음을 알아야 한다.

## 개념 (The Concept)

### 시대 1: OCR 파이프라인 (2021년 이전)

고전적 스택:

1. PDF → 페이지당 이미지.
2. Tesseract(또는 상용 OCR)가 단어별 경계 상자(bounding box)와 함께 텍스트를 추출한다.
3. 레이아웃 분석기가 블록(머리글, 표, 단락)을 식별한다.
4. 표 구조 인식기가 표를 파싱한다.
5. 도메인 규칙 + 정규식이 필드를 추출한다.

깔끔한 인쇄 텍스트에는 동작한다. 손글씨, 기울어진 스캔, 복잡한 표, 비영어 문자에서는 무너진다. 모든 실패 양식은 맞춤형 예외 경로를 요구한다.

### TrOCR (2021)

TrOCR(Li et al., arXiv:2109.10282)는 Tesseract의 고전적 CNN-CTC를 합성 + 실제 텍스트 이미지로 학습한 트랜스포머(transformer) 인코더-디코더로 대체했다. 손글씨와 다국어 텍스트에서 깔끔한 승리다. 여전히 파이프라인(검출기 후 TrOCR 후 레이아웃)이지만, OCR 단계가 극적으로 개선되었다.

### 시대 2: OCR 없는 방식 (2022~2023)

최초의 OCR 없는 모델들은 이렇게 말했다. 검출을 완전히 건너뛰고, 이미지 픽셀을 구조화된 출력으로 직접 매핑하라.

Donut(Kim et al., arXiv:2111.15664):
- 인코더-디코더 트랜스포머, 인코더는 Swin-B.
- 출력은 양식 이해를 위한 JSON, 요약을 위한 마크다운, 또는 임의의 과제 특화 스키마.
- OCR 없음, 레이아웃 없음, 검출 없음.

Nougat(Blecher et al., arXiv:2308.13418):
- 과학 논문에 특화해 학습했다.
- 출력은 LaTeX / 마크다운.
- 수식, 다단 레이아웃, 그림을 처리한다.
- 모든 arXiv 파서가 호출하는 모델.

이들은 만능이 아니라 전문가다. 과학 논문에 대한 Donut은 실패하고, 청구서에 대한 Nougat은 실패한다.

### LayoutLMv3 (2022)

다른 트랙이다. LayoutLMv3(Huang et al., arXiv:2204.08387)는 OCR을 유지하되 레이아웃 이해를 더한다:

- 세 입력 스트림: OCR 텍스트 토큰(token), 토큰별 2D 경계 상자, 이미지 패치.
- 세 가지 모달리티 전반의 마스킹 학습 목표(마스킹된 텍스트, 마스킹된 패치, 마스킹된 레이아웃).
- 다운스트림: 분류(classification), 엔티티 추출, 표 QA.

LayoutLMv3는 OCR 기반 문서 이해의 정점이다. 양식과 청구서에 강하다. 상류에 OCR이 필요하다. 표준화된 문서 벤치마크(benchmark)에서 VLM 이전 최고의 정확도다.

### DocLLM (2023)

DocLLM(Wang et al., arXiv:2401.00908)은 LayoutLM의 생성형 형제다. 레이아웃 토큰을 조건으로 자유 형식 답변을 생성한다. 문서에 대한 QA에 더 좋다; 여전히 OCR 입력에 의존한다.

### 시대 3: VLM 네이티브 (2024년 이후)

2024년 VLM은 파이프라인 전체를 대체할 만큼 충분히 좋아졌다. 전체 페이지 이미지를 고해상도로 VLM에 공급하고, 질문하고, 답을 받는다.

- LLaVA-NeXT 336-타일 AnyRes는 작은 문서에 동작한다.
- Qwen2.5-VL 동적 해상도는 2048+ 픽셀을 네이티브로 처리한다.
- Claude Opus 4.7은 2576px 문서를 지원한다.
- PaliGemma 2(2025년 4월)는 문서 + 손글씨에 특화해 학습한다.

VLM 네이티브와 OCR 파이프라인 사이의 격차는 빠르게 좁혀졌다. 2026년에 이르러 VLM 네이티브가 다음에서 이긴다:

- 장면 텍스트(손글씨 + 인쇄, 혼합 문자).
- 병합된 셀이 있는 복잡한 표.
- 텍스트에 삽입된 수학 수식.
- 텍스트 주석이 있는 그림.

OCR 파이프라인은 여전히 다음에서 이긴다:

- 페이지당 지연 시간(latency)이 중요한 대규모 순수 스캔 작업.
- 파이프라인 신뢰성(결정론적 실패 대 VLM 환각).
- 감사 가능한 OCR 출력이 필요한 규제 환경.

### Claude 4.7 / GPT-5 프런티어

2576픽셀 네이티브 입력에서, 프런티어 VLM은 인간에 가까운 정확도로 문서를 이해한다. 2026년 초의 벤치마크 수치:

- DocVQA: Claude 4.7 약 95.1, PaliGemma 2 약 88.4, Nougat 약 77.3, 파이프라인 LayoutLMv3 약 83.
- ChartQA: Claude 4.7 약 92.2, GPT-4V 약 78.
- VisualMRC: Claude 4.7 약 94.

클로즈드 모델 격차는 대부분 해상도와 기반 LLM 규모다. 7B 오픈 모델은 몇 점 뒤처지지만 따라잡고 있다.

### 수학 수식과 LaTeX 출력

과학 논문은 수식에 대한 정확한 LaTeX 출력이 필요하다. Nougat은 이것을 학습했다. LaTeX 목표로 학습한 VLM(Qwen2.5-VL-Math, Nougat 파생물)은 사용 가능한 LaTeX를 생성한다. 명시적 LaTeX 학습 없이는, VLM은 읽을 수 있지만 부정확한 전사(transcription)를 생성한다.

2026년 과학 논문 파이프라인의 경우: PDF에 Nougat을 체이닝하고, 까다로운 페이지에는 VLM을 쓴다.

### 손글씨

여전히 가장 어려운 하위 과제다. 인쇄 + 손글씨 혼합(의사 메모, 채워진 양식)은 OCR 파이프라인이 여전히 비용에서 VLM을 이기는 지점이다. 손글씨 전용 VLM은 개선되고 있다(Claude 4.7, PaliGemma 2).

### 2026년 레시피

새로운 문서 AI 프로젝트의 경우:

- 대규모 순수 인쇄 청구서: LayoutLMv3 + 규칙, 비용 효율적.
- 혼합 문서(과학 + 손글씨 + 양식): VLM 네이티브(PaliGemma 2 또는 Qwen2.5-VL).
- 전체 arXiv 입력: 수학에는 Nougat, 그림에는 VLM.
- 규제: OCR 파이프라인 + 교차 점검을 위한 VLM 검증기.

```figure
mm-doc-layout
```

## 라이브러리로 써보기 (Use It)

`code/main.py`:

- 장난감 레이아웃 인식 토크나이저(tokenizer): (텍스트, bbox) 쌍이 주어지면 LayoutLMv3 스타일 입력을 생성한다.
- Donut 스타일 과제 스키마 생성기: 양식을 위한 JSON 템플릿.
- OCR 파이프라인, Donut, Nougat, VLM 네이티브에 걸친 페이지당 토큰 예산 비교.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-document-ai-stack-picker.md`를 생성한다. 문서 AI 프로젝트(도메인, 규모, 품질, 규제)가 주어지면, OCR 파이프라인, OCR 없는 전문가, VLM 네이티브 중에서 선택한다.

## 연습 문제 (Exercises)

1. 당신의 프로젝트는 하루 1천만 건의 청구서다. 정확도를 잃지 않으면서 페이지당 비용을 최소화하는 스택은 무엇인가?

2. LayoutLMv3가 양식 QA에서 순수 CLIP-VLM을 능가하지만 장면 텍스트에서는 미달하는 이유는? bbox 스트림은 무엇을 포기하는가?

3. Nougat은 LaTeX를 생성한다. VLM 네이티브 출력이 LaTeX 충실도에서 Nougat을 이기는 테스트 케이스와, Nougat이 이기는 케이스를 제안하라.

4. PaliGemma 2 논문(Google, 2024)을 읽으라. PaliGemma 1 대비 문서 정확도를 끌어올린 핵심 학습 데이터 추가는 무엇이었는가?

5. 규제 안전 하이브리드를 설계하라: 주(primary)는 OCR 파이프라인, 보조(secondary)는 VLM 교차 점검. 불일치를 어떻게 해소하는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| OCR 파이프라인(OCR pipeline) | "Tesseract 스타일" | 단계별 스택: 검출 -> OCR -> 레이아웃 -> 규칙; 결정론적이지만 취약 |
| OCR 없는 방식(OCR-free) | "Donut 스타일" | 명시적 OCR을 건너뛰는 이미지-출력 트랜스포머; 단일 모델 |
| 레이아웃 인식(Layout-aware) | "LayoutLM" | 입력에 토큰별 bbox 좌표 포함; 모달리티 전반의 통합 마스킹 |
| VLM 네이티브(VLM-native) | "프런티어 VLM" | 페이지 이미지를 고해상도로 Claude/GPT/Qwen VLM에 직접 공급; 파이프라인 없음 |
| DocVQA | "문서 벤치마크" | 문서 VQA 표준; 가장 많이 인용되는 점수 |
| 마크업 출력(Markup output) | "LaTeX / MD" | 자유 형식 텍스트 대신 구조화된 출력 형식; 다운스트림 자동화를 가능하게 함 |

## 더 읽을거리 (Further Reading)

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
