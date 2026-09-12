# Machine Translation

> 번역은 30년 동안 NLP 연구에 돈을 댄 작업이며, 지금도 계속 돈을 대고 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75분

## 문제 (The Problem)

모델이 한 언어로 된 문장을 읽고 다른 언어로 된 문장을 만들어낸다. 길이가 달라진다. 어순이 달라진다. 어떤 원문 단어는 여러 개의 목표 단어에 대응하고, 그 반대도 마찬가지다. 관용구는 일대일 대응을 거부한다. "I miss you"는 프랑스어로 "tu me manques"인데, 직역하면 "당신이 나에게 부족하다"이다. 어떤 단어 수준 정렬(alignment)도 이를 견뎌내지 못한다.

기계 번역(machine translation)은 NLP가 인코더-디코더(encoder-decoder), 어텐션(attention), 트랜스포머(transformer), 그리고 결국 LLM 패러다임 전체를 발명하도록 강제한 작업이다. 모든 전진은 번역 품질을 측정할 수 있었고 사람과 기계 사이의 격차가 끈질겼기 때문에 이루어졌다.

이 레슨은 역사 강의를 건너뛰고 2026년의 실전 파이프라인(pipeline)을 가르친다. 사전 학습(pretraining)된 다국어 인코더-디코더(NLLB-200 또는 mBART), 서브워드 토큰화(subword tokenization), 빔 서치(beam search), BLEU 및 chrF 평가, 그리고 발견되지 않은 채 프로덕션(production)에 배포되곤 하는 몇 가지 실패 양상이 그것이다.

## 개념 (The Concept)

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

현대 MT는 병렬 텍스트(parallel text)로 학습된 트랜스포머 인코더-디코더다. 인코더(encoder)는 원문을 해당 언어의 토큰화 방식으로 읽는다. 디코더(decoder)는 인코더의 출력을 크로스 어텐션(cross-attention, 레슨 10)으로 사용하면서 목표 언어를 한 번에 서브워드 하나씩 생성한다. 디코딩은 그리디 디코딩(greedy-decoding)의 함정을 피하려고 빔 서치를 쓴다. 출력은 디토큰화(detokenize)되고, 트루케이싱(truecasing)이 해제되며, 참조(reference)와 비교해 채점된다.

세 가지 운영상의 선택이 실제 MT 품질을 좌우한다.

- **토크나이저(Tokenizer).** 혼합 언어 코퍼스(corpus)로 학습된 SentencePiece BPE. 언어 간 공유 어휘(shared vocabulary)가 바로 NLLB에서 제로샷(zero-shot) 언어쌍을 가능하게 만드는 요소다.
- **모델 크기(Model size).** NLLB-200 distilled 600M은 노트북에 들어간다. NLLB-200 3.3B가 공개된 프로덕션 기본값이다. 54.5B는 연구의 상한선이다.
- **디코딩(Decoding).** 일반 콘텐츠에는 빔 너비(beam width) 4-5. 너무 짧은 출력을 피하기 위한 길이 페널티(length penalty). 용어 일관성이 필요할 때는 제약 디코딩(constrained decoding).

```figure
seq2seq-alignment
```

## 직접 만들기 (Build It)

### Step 1: 사전 학습된 MT 호출

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

여기서 세 가지가 중요하다. `src_lang`은 토크나이저에게 어떤 문자 체계와 분절(segmentation)을 적용할지 알려준다. `forced_bos_token_id`는 디코더에게 어떤 언어를 생성할지 알려준다. 둘 다 NLLB에 특화된 트릭이다. mBART와 M2M-100은 각자의 관례를 사용하며 서로 호환되지 않는다.

### Step 2: BLEU와 chrF

BLEU는 출력과 참조 사이의 n-gram 중복을 측정한다. 네 가지 참조 n-gram 크기(1-4), 정밀도(precision)의 기하 평균, 너무 짧은 출력에 대한 간결성 페널티(brevity penalty)로 구성된다. 점수는 [0, 100] 범위다. 흔히 쓰인다. 해석하기는 짜증스럽다. BLEU 30은 "쓸 만한" 수준, 40은 "좋은" 수준, 50은 "탁월한" 수준이며, 1 BLEU 미만의 차이는 잡음(noise)이다.

chrF는 문자 수준 F-점수(F-score)를 측정한다. BLEU가 일치를 과소 계산하는 형태론적으로 풍부한 언어에 더 민감하다. 종종 BLEU와 나란히 보고된다.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

항상 `sacrebleu`를 사용하라. 토큰화를 정규화(normalization)하여 논문 간에 점수가 비교 가능하도록 만든다. BLEU 계산을 직접 만드는 것은 오해를 부르는 벤치마크(benchmark)가 생기는 지름길이다.

### 3단계 평가 계층 구조 (2026)

현대 MT 평가는 서로 보완하는 세 가지 지표 계열을 사용한다. 최소 두 가지를 갖춰 배포하라.

- **휴리스틱(Heuristic)** (BLEU, chrF). 빠르고, 참조 기반이며, 해석 가능하지만, 패러프레이즈(paraphrase)에 둔감하다. 레거시 비교와 회귀(regression) 탐지에 사용하라.
- **학습 기반(Learned)** (COMET, BLEURT, BERTScore). 사람의 판단으로 학습된 신경망 모델이다. 번역을 원문 및 참조와 의미적 유사성으로 비교한다. COMET은 2023년 이후 MT 연구와 가장 높은 연관성을 가지며, 품질이 중요한 경우 2026년 프로덕션 기본값이다.
- **LLM-as-judge** (참조 없음). 대형 모델에게 유창성, 충실성, 어조, 문화적 적절성을 기준으로 번역을 채점하도록 프롬프트(prompt)한다. 평가 기준(rubric)이 잘 설계되면 GPT-4-as-judge는 사람과의 일치율이 약 80%에 이른다. 참조가 존재하지 않는 개방형 콘텐츠에 사용하라.

2026년 실전 스택: BLEU와 chrF에는 `sacrebleu`, COMET에는 `unbabel-comet`, 그리고 사람을 향한 최종 신호에는 프롬프트된 LLM을 사용한다. 프로덕션 데이터에서 신뢰하기 전에 모든 지표를 사람이 레이블링한 50-100개 예시로 보정(calibrate)하라.

참조 없는 지표(COMET-QE, BLEURT-QE, LLM-as-judge)는 참조 없이 번역을 평가할 수 있게 해주는데, 이는 참조 번역이 존재하지 않는 롱테일(long-tail) 언어쌍에서 중요하다.

### Step 3: 프로덕션에서 무엇이 망가지는가

위의 실전 파이프라인은 80%의 경우 유창하게 번역하고 나머지 20%는 조용히 실패한다. 명명된 실패 양상:

- **환각(Hallucination).** 모델이 원문에 없던 내용을 지어낸다. 익숙하지 않은 도메인 어휘에서 흔하다. 증상: 출력은 유창하지만 원문이 진술하지 않은 사실을 주장한다. 완화책: 도메인 용어에 대한 제약 디코딩, 규제 콘텐츠에 대한 사람의 검토, 입력보다 훨씬 긴 출력에 대한 모니터링.
- **타깃 이탈 생성(Off-target generation).** 모델이 잘못된 언어로 번역한다. NLLB는 희귀 언어쌍에서 놀랄 만큼 이 문제에 취약하다. 완화책: `forced_bos_token_id`를 검증하고 항상 출력에 대해 언어 식별(language-ID) 모델 검사를 수행하며 디코딩하라.
- **용어 표류(Terminology drift).** "Sign up"이 문서 1에서는 "s'inscrire"가 되고 문서 2에서는 "créer un compte"가 된다. UI 텍스트와 사용자 대면 문자열에서는 일관성이 원시 품질보다 더 중요하다. 완화책: 용어집 제약 디코딩(glossary-constrained decoding) 또는 사후 편집 사전(post-edit dictionary).
- **격식 불일치(Formality mismatch).** 프랑스어 "tu" 대 "vous", 일본어의 공손 수준. 모델은 학습 데이터에서 더 흔했던 형태를 고른다. 고객 대면 콘텐츠에서는 보통 잘못된 선택이다. 완화책: 모델이 지원한다면 격식 토큰(formality token)으로 프롬프트 접두사를 붙이거나, 격식체 전용 코퍼스로 작은 모델을 파인튜닝(fine-tuning)하라.
- **짧은 입력에서의 길이 폭발(Length explosion on short input).** 매우 짧은 입력 문장은 종종 지나치게 긴 번역을 만드는데, 길이 페널티가 원문 토큰 약 5개 미만에서 급격히 떨어지기 때문이다. 완화책: 원문 길이에 비례하는 하드 최대 길이 상한(hard max-length cap).

### Step 4: 도메인을 위한 파인튜닝

사전 학습된 모델은 일반주의자다. 법률, 의료, 게임 대사 번역은 도메인 병렬 데이터로 파인튜닝하면 측정 가능한 이득을 얻는다. 그 레시피는 이국적이지 않다.

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

수천 개의 고품질 병렬 예시가 수십만 개의 잡음 섞인 웹 스크래핑 예시를 이긴다. 학습 데이터의 품질이 프로덕션에서 가장 큰 단일 레버다.

## 라이브러리로 써보기 (Use It)

2026년 MT 프로덕션 스택:

| 사용 사례 | 권장 출발점 |
|---------|---------------------------|
| 임의-대-임의, 200개 언어 | `facebook/nllb-200-distilled-600M` (노트북) 또는 `nllb-200-3.3B` (프로덕션) |
| 영어 중심, 고품질, 50개 언어 | `facebook/mbart-large-50-many-to-many-mmt` |
| 짧은 실행, 저렴한 추론, 영어-프랑스어/독일어/스페인어 | Helsinki-NLP / Marian 모델 |
| 지연 시간이 중요한 브라우저 측 | ONNX 양자화 Marian (~50 MB) |
| 최대 품질, 비용 지불 의향 있음 | 번역 프롬프트를 사용한 GPT-4 / Claude / Gemini |

2026년 현재 LLM은 여러 언어쌍에서, 특히 관용적 콘텐츠와 긴 컨텍스트(context)에서 특화된 MT 모델을 능가한다. 트레이드오프(trade-off)는 토큰당 비용과 지연 시간(latency)이다. 컨텍스트 길이, 문체 일관성, 또는 프롬프팅을 통한 도메인 적응이 처리량(throughput)보다 더 중요할 때 LLM을 선택하라.

## 산출물 (Ship It)

`outputs/skill-mt-evaluator.md`로 저장하라:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 연습 문제 (Exercises)

1. **Easy.** `nllb-200-distilled-600M`을 사용해 5문장짜리 영어 문단을 프랑스어로 번역한 뒤 다시 영어로 번역하라. 왕복 결과가 원본에 얼마나 가까운지 측정하라. 단어 선택의 표류와 함께 의미 보존이 나타나는 것을 볼 수 있을 것이다.
2. **Medium.** `fasttext lid.176` 또는 `langdetect`를 사용해 번역 출력에 대한 언어 식별 검사를 구현하라. MT 호출에 통합하여 타깃 이탈 생성이 반환되기 전에 잡히도록 하라.
3. **Hard.** 당신이 선택한 5,000쌍 도메인 코퍼스로 `nllb-200-distilled-600M`을 파인튜닝하라. 파인튜닝 전후로 홀드아웃(held-out) 세트에 대한 BLEU를 측정하라. 어떤 종류의 문장이 개선되고 어떤 종류가 퇴보했는지 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| BLEU | 번역 점수 | 간결성 페널티를 적용한 n-gram 정밀도. [0, 100]. |
| chrF | 문자 F-점수 | 문자 수준 F-점수. 형태론적으로 풍부한 언어에 더 민감하다. |
| NMT | 신경망 MT | 병렬 텍스트로 학습된 트랜스포머 인코더-디코더. 2017년 이후의 기본값. |
| NLLB | No Language Left Behind | Meta의 200개 언어 MT 모델 계열. |
| Constrained decoding | 제어된 출력 | 특정 토큰이나 n-gram이 출력에 나타나도록 / 나타나지 않도록 강제한다. |
| Hallucination | 지어낸 내용 | 원문이 뒷받침하지 않는 모델 출력. |

## 더 읽을거리 (Further Reading)

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672): NLLB 논문.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/): 왜 `sacrebleu`가 BLEU를 보고하는 유일하게 올바른 방법인지.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/): chrF 논문.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation): 실전 파인튜닝 안내.
