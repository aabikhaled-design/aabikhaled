# Capstone 15: 헌법적 안전 하네스 + 레드팀 사격장 (Constitutional Safety Harness + Red-Team Range)

> Anthropic의 Constitutional Classifiers, Meta의 Llama Guard 4, Google의 ShieldGemma-2, NVIDIA의 Nemotron 3 Content Safety, 그리고 다국어 커버리지를 위한 X-Guard가 2026 안전 분류기(safety-classifier) 스택을 정의했다. garak, PyRIT, NVIDIA Aegis, 그리고 promptfoo는 표준 적대적(adversarial) 평가 도구가 되었다. NeMo Guardrails v0.12는 이것들을 프로덕션(production) 파이프라인으로 묶는다. 이 캡스톤(capstone)은 그 모든 것을 함께 연결한다: 타깃(target) 앱을 둘러싼 계층화된 안전 하네스(safety harness), 6개 이상의 공격 계열(attack family)을 돌리는 자율 레드팀 에이전트(red-team agent), 그리고 측정 가능한 무해성(harmlessness) 차이를 만들어내는 헌법적 자기 비평(constitutional self-critique) 실행.

**Type:** Capstone
**Languages:** Python (safety pipeline, red team), YAML (policy configs)
**Prerequisites:** Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 18 (ethics, safety, alignment)
**Phases exercised:** P10 · P11 · P13 · P14 · P18
**Time:** 25시간

## 문제 (Problem)

2026년 LLM 안전의 최전선은 분류기가 작동하는지 여부(대략 작동한다)가 아니라, 과도 거부(over-refusing)하거나 명백한 구멍을 남기지 않으면서 프로덕션 앱 주위에 이 분류기들을 어떻게 올바르게 조합하느냐다. Llama Guard 4는 영어 정책 위반을 처리한다. X-Guard(132개 언어)는 다국어 탈옥(jailbreak)을 처리한다. ShieldGemma-2는 이미지 기반 프롬프트 인젝션(prompt injection)을 잡는다. NVIDIA Nemotron 3 Content Safety는 엔터프라이즈 범주를 다룬다. Anthropic의 Constitutional Classifiers는 서빙이 아니라 학습 중에 쓰이는 별도 접근법이다.

공격 진화도 중요하다. PAIR와 TAP는 탈옥 발견을 자동화한다. GCG는 그래디언트 기반(gradient-based) 접미사 공격을 실행한다. 멀티턴(multi-turn)과 코드 스위치(code-switch) 공격은 에이전트 메모리를 악용한다. 배포된 모든 LLM은 레드팀 사격장(red-team range)이 필요하다. garak과 PyRIT가 표준 드라이버다. 여기에 문서화된 완화책(mitigation)과 CVSS로 채점된 발견(finding)을 더한다.

이 캡스톤에서는 타깃 애플리케이션(8B 인스트럭션 튜닝(instruction-tuned) 모델이거나 다른 캡스톤의 RAG 챗봇 중 하나)을 강화하고, 6개 이상의 공격 계열을 거기에 실행하며, 전후(before/after) 무해성 측정을 만들어낸다.

## 개념 (Concept)

안전 파이프라인은 다섯 계층이다. **입력 정제(Input sanitize)**: 폭 0 문자 제거, base64/rot13 디코드, 유니코드 정규화. **정책 계층(Policy layer)**: NeMo Guardrails v0.12 레일(rail)(도메인 외, 독성, PII 추출). **분류기 게이트(Classifier gate)**: 입력에 Llama Guard 4, 비영어에 X-Guard, 이미지 입력에 ShieldGemma-2. **모델(Model)**: 타깃 LLM. **출력 필터(Output filter)**: 출력에 Llama Guard 4, Presidio PII 스크럽(scrub), 해당되는 경우 인용 강제. **HITL 계층**: 고위험으로 표시된 출력은 Slack 큐로 간다.

레드팀 사격장은 스케줄러 위에서 돌아간다. PAIR와 TAP는 자율적으로 탈옥을 발견한다. GCG는 그래디언트 기반 접미사 공격을 실행한다. ASCII / base64 / rot13 인코딩 공격. 멀티턴 공격(페르소나 채택, 메모리 악용). 코드 스위치 공격(영어를 스와힐리어나 태국어와 섞기). 각 실행은 CVSS 채점과 공개(disclosure) 타임라인이 담긴 구조화된 발견 파일을 생성한다.

헌법적 자기 비평 실행은 학습 시점(training-time) 개입이다. 1천 개의 유해 시도 프롬프트를 가져와, 모델이 응답을 초안하게 하고, 작성된 헌법(do-not-harm 규칙)에 비추어 비평한 뒤, 비평 루프(critique loop)로 재학습한다. 홀드아웃(held-out) 평가에서 전후 무해성 차이를 측정한다.

## 아키텍처 (Architecture)

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 스택 (Stack)

- 안전 분류기: Llama Guard 4, ShieldGemma-2, NVIDIA Nemotron 3 Content Safety, X-Guard
- 가드레일 프레임워크: NeMo Guardrails v0.12 + OPA
- 레드팀 드라이버: garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo
- 탈옥 에이전트: PAIR (Chao et al., 2023), Tree-of-Attacks (TAP), GCG suffix
- 헌법적 학습: Anthropic 스타일 자기 비평 루프 + 비평에 대한 SFT
- PII 스크럽: Presidio
- 타깃: 8B 인스트럭션 튜닝 모델 또는 다른 캡스톤의 RAG 챗봇 중 하나

```figure
cf-safety-stack
```

## 직접 만들기 (Build It)

1. **타깃 설정.** vLLM에 8B 인스트럭션 튜닝 모델을 세운다(또는 다른 캡스톤의 RAG 챗봇을 재사용한다). 이것이 테스트 대상 앱이다.

2. **안전 파이프라인 래핑.** 타깃 주위에 다섯 계층 파이프라인을 연결한다. 각 계층이 개별적으로 관측 가능한지(Langfuse에서 계층별 스팬) 검증한다.

3. **분류기 커버리지.** Llama Guard 4, X-Guard(다국어), ShieldGemma-2(이미지)를 로드한다. 베이스라인을 세우기 위해 각각을 작은 레이블된 세트에 실행한다.

4. **레드팀 스케줄러.** garak, PyRIT, PAIR 에이전트, TAP 에이전트, GCG 러너, 멀티턴 공격자, 코드 스위치 공격자를 스케줄링한다. 각각은 별도 큐에서 돌아간다.

5. **공격 스위트.** 여섯 가지 공격 계열: (1) PAIR 자동 탈옥, (2) TAP tree-of-attacks, (3) GCG 그래디언트 접미사, (4) ASCII / base64 / rot13 인코딩, (5) 멀티턴 페르소나, (6) 다국어 코드 스위치. 계열별 성공률을 보고한다.

6. **헌법적 자기 비평.** 1천 개의 유해 시도 프롬프트를 큐레이션한다. 각각에 대해 타깃이 응답을 초안한다. 비평자(critic) LLM이 작성된 헌법("해를 끼치지 말 것", "증거를 인용할 것", "불법 요청을 거부할 것")에 비추어 채점한다. 비평자가 반대하는 프롬프트는 다시 작성되고, 타깃은 비평으로 개선된 쌍(pair)에 파인튜닝(fine-tuning)한다. 홀드아웃 평가에서 전후 무해성을 측정한다.

7. **과도 거부 측정.** 무해한 프롬프트 스위트(예: XSTest)에서 거짓 양성(false-positive) 비율을 추적한다. 타깃은 무해한 질문에 대해 도움을 유지해야 한다.

8. **CVSS 채점.** 성공한 각 탈옥에 대해 CVSS 4.0(공격 벡터, 복잡도, 영향)으로 채점한다. 공개 타임라인과 완화 계획을 만든다.

9. **사격장 자동화.** 위의 모든 것을 cron으로 돌린다. 발견은 큐에 기록된다. 과도 거부 회귀(regression) 경보가 Slack으로 발사된다.

## 라이브러리로 써보기 (Use It)

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 산출물 (Ship It)

`outputs/skill-safety-harness.md`가 결과물이다. 프로덕션급 계층화된 안전 파이프라인과 전후 무해성 차이를 갖는 재현 가능한 레드팀 사격장.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | 공격 표면 커버리지 | 6개 이상 공격 계열 동작, 2개 이상 언어 |
| 20 | 참 양성 / 거짓 양성 트레이드오프 | 공격 차단율 대 XSTest 무해 통과율 |
| 20 | 자기 비평 차이 | 홀드아웃 평가에서 전후 무해성 |
| 20 | 문서화 및 공개 | 타임라인을 갖는 CVSS 채점 발견 |
| 15 | 자동화 및 반복성 | 모든 것이 경보와 함께 cron으로 실행됨 |
| **100** | | |

## 연습 문제 (Exercises)

1. RAG 챗봇에 garak의 프롬프트 인젝션 플러그인을 실행하고, 출력 필터 계층 유무에 따른 공격 성공률을 비교한다.

2. 일곱 번째 공격 계열을 추가한다: 검색된 문서를 통한 간접 프롬프트 인젝션(indirect prompt injection). 필요한 추가 방어를 측정한다.

3. "도움과 함께 거부(refuse-with-help)" 모드를 구현한다: 가드레일이 차단할 때, 타깃이 단호한 거부 대신 더 안전한 관련 답변을 제안한다. XSTest 차이를 측정한다.

4. 다국어 커버리지 격차: X-Guard가 저성능을 보이는 언어를 찾는다. 그것을 겨냥한 파인튜닝 데이터셋을 제안한다.

5. 30B 모델에 헌법적 자기 비평을 실행하고 차이가 확장되는지 측정한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| Layered safety | "심층 방어(defense in depth)" | 입력, 게이트, 출력, HITL에서의 다중 가드레일 |
| Llama Guard 4 | "Meta의 안전 분류기" | 2026 레퍼런스 입출력 콘텐츠 분류기 |
| PAIR | "탈옥 에이전트" | LLM 주도 탈옥 발견에 관한 논문 (Chao et al.) |
| TAP | "Tree-of-Attacks" | PAIR의 트리 탐색(tree-search) 변형 |
| GCG | "greedy coordinate gradient" | 그래디언트 기반 적대적 접미사 공격 |
| Constitutional self-critique | "Anthropic 스타일 학습" | 타깃 초안 -> 비평자 채점 -> 재작성 -> 재학습 |
| XSTest | "무해 프로브 세트" | 과도 거부 회귀를 위한 벤치마크 |
| CVSS 4.0 | "심각도 점수" | 안전 발견을 위한 표준 취약점 채점 |

## 더 읽을거리 (Further Reading)

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers): 학습 시점 레퍼런스
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/): 2026 입출력 분류기
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b): 이미지 + 멀티모달 안전
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/): 엔터프라이즈 레퍼런스
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848): 132개 언어 다국어 안전
- [garak](https://github.com/NVIDIA/garak): NVIDIA 레드팀 툴킷
- [PyRIT](https://github.com/Azure/PyRIT): Microsoft 레드팀 프레임워크
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/): 레일 프레임워크
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419): 탈옥 에이전트 논문
