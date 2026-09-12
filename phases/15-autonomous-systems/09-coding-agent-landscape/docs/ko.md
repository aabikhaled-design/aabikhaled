# 자율 코딩 에이전트 지형(2026)

> SWE-bench Verified는 3년이 채 안 되는 기간에 4%에서 80.9%로 올랐다. 동일한 Claude Sonnet 4.5가 SWE-agent v1에서는 43.2%, Cline autonomous에서는 59.8%를 기록했다. 이제 모델 주위의 스캐폴딩(scaffolding)이 모델 자체만큼이나 중요하다는 뜻이다. OpenHands(이전 명칭 OpenDevin)는 가장 활발한 MIT 라이선스 플랫폼이며, 그 CodeAct 루프는 JSON 도구 호출 대신 샌드박스(sandbox)에서 Python 액션을 직접 실행한다. 화려한 수치 뒤에는 방법론적 문제가 숨어 있다. SWE-bench Verified의 500개 과제 중 161개는 단 1~2줄 변경만 필요하며, SWE-bench Pro(10줄 이상 과제)에서는 동일한 프런티어 모델들이 23~59%에 머문다.

**Type:** Learn
**Languages:** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Prerequisites:** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45분

## 문제 (The Problem)

"어떤 코딩 에이전트가 가장 좋은가"는 잘못된 질문이다. 올바른 질문은 이것이다. 내 업무와 맞아떨어지는 과제 분포에서, 프로덕션(production)에서 실제로 돌릴 스캐폴딩까지 함께 놓고 봤을 때, 어떤 종단 간(end-to-end) 신뢰성을 얻는가?

2022년에서 2026년 사이, 이 분야는 스캐폴딩, 곧 검색(retrieval) 층과 플래너(planner), 샌드박스, 편집-검증 루프, 피드백 형식이 핵심을 떠받친다는 사실을 배웠다. Claude Sonnet 4.5는 SWE-agent v1에서 SWE-bench Verified 43.2%를 기록했고, 동일한 모델이 Cline의 autonomous 스캐폴드 안에서는 59.8%를 기록했다. 같은 가중치(weight)인데 절대 16.6포인트의 차이다. 기반 모델(base model)은 하나의 부품이고, 루프가 곧 제품이다.

이와 짝을 이루는 문제는, 벤치마크(benchmark) 포화(saturation)가 퇴보를 가린다는 것이다. SWE-bench Verified는 포화에 가깝고, 쉬운 과제 꼬리(500개 중 161개가 2줄 이하 필요)가 최상위 점수를 끌어올린다. 실제 품질은 SWE-bench Pro(10줄 이상 변경) 같은 분포에서 더 잘 측정되며, 거기서는 같은 선두 모델들도 여전히 23~59%에 머문다.

## 개념 (The Concept)

### SWE-bench, 한 문단으로

SWE-bench(Jimenez et al.)는 정답 패치(ground-truth patch)가 있는 실제 GitHub 이슈들을 가져와, 에이전트(agent)에게 테스트 스위트를 통과시키는 패치를 만들도록 요구한다. SWE-bench Verified(OpenAI, 2024)는 모호하거나 깨진 과제들을 제거한, 사람이 큐레이션한 500개 과제 부분집합이다. SWE-bench Pro는 더 어려운 후속작으로, 10줄 이상 변경이 필요한 과제들이며, 현재 프런티어 에이전트들이 23~59%에 머무는 곳이다.

### 2022 → 2026 곡선이 실제로 보여주는 것

- **2022년**: 연구용 모델들이 원본 SWE-bench에서 약 4%.
- **2024년**: GPT-4 + Devin 스타일 스캐폴딩이 약 14%, SWE-agent가 약 12%.
- **2025년**: Aider와 SWE-agent 안의 Claude 3.5/3.7 Sonnet이 40~55% 범위로 진입.
- **2026년**: Claude Sonnet 4.5와 프런티어 경쟁자들이 SWE-bench Verified에서 70~80%+. Epoch AI의 리더보드가 이를 실시간으로 추적한다.

이 기울기는 세 가지 복합적 원천에서 나왔다. 더 나은 기반 모델, 더 나은 스캐폴딩(CodeAct, 반성(reflection), 검증자(verifier) 루프), 그리고 더 나은 벤치마크(노이즈를 제거한 Verified).

### CodeAct vs JSON 도구 호출

OpenHands(All-Hands-AI, arXiv:2407.16741, 이전 명칭 OpenDevin)는 특정한 아키텍처적 베팅을 했다. 모델이 JSON 도구 호출을 내보내면 호스트가 이를 디코딩해 실행하는 방식 대신, 모델이 Python 코드를 내보내고 Jupyter 스타일 커널이 이를 샌드박스에서 실행한다. 에이전트는 단일 액션 안에서 파일들을 순회하고, 도구들을 연쇄하고, 자신의 예외를 잡아낼 수 있다.

트레이드오프(trade-off):

- **JSON 도구 호출**: 모든 액션이 한 턴이다. 감사하기 쉽다. 합성성(compositionality)이 제한적이다. 각 호출이 명시적 검증기를 거치므로 기본적으로 안전하다.
- **CodeAct**: 하나의 액션이 프로그램 전체가 될 수 있다. 합성적이다. 견고화된 샌드박스가 필요하다(OpenHands는 Docker 격리를 사용한다). 실패 양상에는 샌드박스 런타임이 허용하는 모든 것이 포함된다.

두 아키텍처 모두 프로덕션에서 쓰인다. CodeAct는 오픈 플랫폼(OpenHands, smolagents)에서 지배적이다. JSON 도구 호출은 제공자가 실행기를 통제하는 관리형 서비스(Anthropic Managed Agents, OpenAI Assistants)에서 여전히 지배적이다.

### 2026 지형의 스캐폴드들

| 스캐폴드 | 라이선스 | 실행 모델 | 주목할 특성 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 내 CodeAct | 가장 활발한 오픈 플랫폼; 이벤트 스트림 재생 가능 |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | 최초의 종단 간 SWE-bench 스캐폴드 |
| Aider | Apache-2 | 로컬 저장소에서 diff 기반 편집 | 최소 스캐폴드, 강한 회귀 안정성 |
| Cline | Apache-2 | 도구 정책을 갖춘 VS Code 에이전트 | Sonnet 4.5에서 최고 점수를 낸 오픈 스캐폴드 |
| Devin (Cognition) | 독점 | 관리형 VM + 플래너 | 최초의 "AI 소프트웨어 엔지니어" 제품 범주 |
| Claude Code | 독점 | 권한 모드 + 루틴 | Lesson 10에서 에이전트 루프를 상세히 다룬다 |

### 스캐폴딩이 지배하는 이유

코딩 실행은 장기 지평(long-horizon) 궤적이다(Lesson 1). 신뢰성은 스텝들에 걸쳐 복리로 누적된다. 스캐폴딩이 점수를 벌어들이는 세 곳:

1. **검색(Retrieval)**: 읽어야 할 올바른 파일을 찾는 것이 조용한 병목이다. SWE-agent의 ACI, OpenHands의 파일 인덱스, Aider의 저장소 맵(repo-map)이 모두 이 문제를 공략한다.
2. **검증자 루프**: 테스트를 돌리고, 스택 트레이스를 읽고, 재시도하는 것은 SWE-bench에서 10포인트 이상의 차이를 낸다.
3. **실패 격리(Failure containment)**: 오류 시 롤백하는 샌드박스는 피해가 복리로 불어나는 것을 막는다. 검증자 루프가 있는 같은 모델과 없는 같은 모델은 서로 다른 두 제품처럼 보인다.

### 벤치마크 포화와 실제 분포

OpenHands 저자들과 Epoch AI 모두 SWE-bench Verified에 쉬운 꼬리가 있음을 지적한다. 500개 과제 중 161개는 1~2줄 변경만 필요하다. 높은 점수는 부분적으로 이 꼬리에 의해 견인된다. SWE-bench Pro는 10줄 이상 변경으로 제한하며, 프런티어 시스템에서도 23~59% 범위의 점수에 그친다. 현장의 프로덕션 분포는 거의 확실히 Verified보다 Pro에 더 가깝다.

에이전트 선택에 대한 함의: 자신의 버그 백로그(backlog)에서 Pro 같은 부분집합을 돌려라. 중요한 점수는 실제로 출하하는 작업을 대표하는 과제들에서의 점수다.

```figure
a5-scaffold-delta
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 고정된 미니 과제 분포에서 두 개의 장난감 에이전트 스캐폴드를 비교한다:

1. 턴당 한 액션을 취하는 **JSON 도구 호출** 스캐폴드.
2. 액션당 작은 Python 스니펫을 내보낼 수 있는 **CodeAct** 스캐폴드.

둘 다 스텁(stub) "모델"(결정론적 규칙)을 사용하므로, 비교는 스캐폴드를 모델 품질로부터 분리한다. 출력은 CodeAct 스캐폴드가 더 큰 액션당 폭발 반경(blast radius)을 대가로 더 적은 턴에 더 많은 과제를 해결함을 보여준다.

## 산출물 (Ship It)

`outputs/skill-scaffold-audit.md`는 채택 전에 제안된 코딩 에이전트 스캐폴드를 감사하는 데 도움을 준다: 검색 품질, 검증자 존재 여부, 샌드박스 격리, 벤치마크-대-분포 적합도.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 같은 과제 집합에서 각 스캐폴드는 몇 턴이 걸리는가? 각각의 액션당 폭발 반경은 무엇인가?

2. OpenHands 논문(arXiv:2407.16741)을 읽어라. 이 논문은 복잡한 과제에서 CodeAct가 JSON 도구 호출을 이긴다고 주장한다. 논문이 인정하는 실패 양상 하나를 찾아내고, 그 양상이 프로덕션에서 언제 지배적이 될지를 한 문장으로 써라.

3. 두 파일에 걸쳐 10줄 이상 변경이 필요한 과제 하나를 자신의 버그 백로그에서 골라라. (a) JSON 도구 호출과 (b) CodeAct 하에서 프런티어 모델의 종단 간 성공 확률을 추정하라. 그 차이를 정당화하라.

4. SWE-bench Verified에는 단일 파일, 1~2줄짜리 과제가 161개 있다. 이들을 제외하는 점수를 구성하라. 리더보드는 어떻게 뒤섞이는가?

5. "Introducing SWE-bench Verified"(OpenAI)를 읽어라. 모호한 과제를 제거하는 데 쓰인 구체적 방법론을 설명하고, 그 큐레이션이 놓칠 범주 하나를 지목하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| SWE-bench | "코딩 벤치마크" | 정답 패치와 테스트 스위트가 있는 실제 GitHub 이슈들 |
| SWE-bench Verified | "정제된 부분집합" | 사람이 큐레이션한 500개 과제, 쉬운 꼬리가 존재함 |
| SWE-bench Pro | "더 어려운 부분집합" | 10줄 이상 변경; 프런티어가 23~59%에 머묾 |
| CodeAct | "코드-as-액션" | 에이전트가 Python을 내보내고; Jupyter 스타일 커널이 샌드박스에서 실행 |
| JSON 도구 호출 | "함수 호출" | 각 액션이 실행 전에 검증되는 구조화된 JSON 페이로드 |
| 스캐폴드 (Scaffold) | "에이전트 프레임워크" | 기반 모델 주위의 검색 + 플래너 + 실행기 + 검증자 루프 |
| ACI (Agent-Computer Interface) | "SWE-agent의 형식" | 사람용 셸이 아니라 LLM 사용성을 위해 설계된 명령어 집합 |
| 검증자 루프 (Verifier loop) | "테스트-후-재시도" | 테스트를 돌리고, 출력을 읽고, 패치를 수정; 모델 외 신뢰성 향상 중 가장 큰 폭 |

## 더 읽을거리 (Further Reading)

- [Jimenez et al.: SWE-bench](https://www.swebench.com/) 원본 벤치마크와 방법론.
- [OpenAI: Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) 큐레이션된 부분집합이 어떻게 만들어졌는지.
- [Wang et al.: OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) CodeAct 아키텍처와 이벤트 스트림 설계.
- [Epoch AI: SWE-bench leaderboard](https://epoch.ai/benchmarks) 실시간 추적 점수.
- [Anthropic: Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) 장기 지평 코딩 에이전트 신뢰성 프레이밍.
