# 사례 연구와 2026년 최첨단 (Case Studies and the 2026 State of the Art)

> 각각 다중 에이전트(multi-agent) 엔지니어링의 서로 다른 단면을 보여주는, 처음부터 끝까지 공부할 세 가지 프로덕션(production) 등급 참조. **Anthropic의 Research 시스템**(오케스트레이터-워커(orchestrator-worker), 15배 토큰(token), 단일 에이전트(agent) Opus 4 대비 +90.2%, 레인보우 배포(rainbow deployment))은 정석 감독자(supervisor) 사례다. **MetaGPT / ChatDev**(소프트웨어 엔지니어링을 위한 SOP 인코딩 역할 특화; ChatDev의 "소통적 탈환각(communicative dehallucination)"; DAG를 통해 >1000 에이전트로 확장하는 MacNet 확장, arXiv:2406.07155)은 정석 역할 분해 사례다. **OpenClaw / Moltbook**(원래 Peter Steinberger의 Clawdbot, 2025년 11월; 두 번 이름 변경; 2026년 3월까지 247k GitHub 스타; 로컬 ReAct 루프 에이전트; 출시 며칠 만에 약 230만 에이전트 계정을 가진 에이전트 전용 소셜 네트워크 Moltbook, 2026-03-10 Meta에 인수)은 집단 규모에서 무슨 일이 일어나는지 보여준다: 창발적 경제 활동, 프롬프트 주입(prompt-injection) 위험, 국가 수준 규제(중국이 2026년 3월 정부 컴퓨터에서 OpenClaw를 제한). **2026년 4월 프레임워크 지형:** LangGraph와 CrewAI가 프로덕션을 선도한다. AG2는 커뮤니티 AutoGen 계승이다. Microsoft AutoGen은 유지보수 모드다(Microsoft Agent Framework로 병합, 2026년 2월 RC). OpenAI Agents SDK는 프로덕션 Swarm 후계자다. Google ADK(2025년 4월)는 A2A 네이티브 진입자다. 이제 모든 주요 프레임워크가 MCP 지원을 출하하고, 대부분 A2A를 출하한다. 이 레슨은 각 사례를 처음부터 끝까지 읽고 공통 패턴을 추려, 다음 프로덕션 시스템에 맞는 참조를 고르도록 돕는다.

**Type:** Learn (capstone)
**Languages:** —
**Prerequisites:** all of Phase 16 (Lessons 01-24)
**Time:** ~90분

## 문제 (Problem)

다중 에이전트 엔지니어링은 젊은 학문이다. 프로덕션 참조는 적고, 각각은 그 공간의 서로 다른 부분을 다룬다. 하나씩 읽는 것은 유용하다. 한 묶음으로 비교하는 것은 더 유용하다. 이 레슨은 2026년의 세 정석 사례 연구를 처음부터 끝까지 읽는 독서 목록으로 다루고, 공통 패턴을 못 박으며, 프레임워크 지형을 매핑해 마케팅이 아니라 지식에서 프레임워크를 고르도록 돕는다.

## 개념 (Concept)

### Anthropic Research 시스템

프로덕션 감독자-워커 사례. Claude Opus 4가 계획하고 종합한다. Claude Sonnet 4 서브에이전트들이 병렬로 연구한다. 발표된 엔지니어링 게시물: https://www.anthropic.com/engineering/multi-agent-research-system.

핵심 측정 결과:

- 내부 연구 평가에서 단일 에이전트 Opus 4 대비 **+90.2%** 개선.
- **BrowseComp 분산의 80%**가 **토큰 사용량만으로** 설명됨: 다중 에이전트가 이기는 이유는 대체로 각 서브에이전트가 신선한 컨텍스트 윈도우(context window)를 받기 때문이다.
- 단일 에이전트 대비 **쿼리당 15배 토큰**.
- 에이전트가 장기 실행되고 상태를 가지므로 **레인보우 배포**.

성문화된 설계 교훈:

1. **노력을 쿼리 복잡도에 맞춰 조정하라.** 단순 → 3-10회 도구 호출을 가진 1개 에이전트. 중간 → 3개 에이전트. 복잡한 연구 → 10개 이상의 서브에이전트.
2. **먼저 넓게, 그다음 좁게.** 서브에이전트들이 넓은 검색을 한다. 리드가 종합한다. 후속 서브에이전트들이 표적화된 심층 작업을 한다.
3. **레인보우 배포.** 진행 중인 에이전트가 끝날 때까지 옛 런타임(runtime) 버전을 살려 두라.
4. **검증은 선택 사항이 아니다.** 명시적 검증자 역할 없이는 시스템이 환각(hallucination)하는 것이 관찰되었다.

이것이 프로덕션 규모에서 감독자-워커 토폴로지(topology)(Phase 16 · 05)의 참조 사례다.

### MetaGPT / ChatDev

프로덕션 SOP 역할 분해 사례. arXiv:2308.00352 (MetaGPT)과 arXiv:2307.07924 (ChatDev)을 다룬다.

MetaGPT는 소프트웨어 엔지니어링 SOP를 역할 프롬프트(prompt)로 인코딩한다: 제품 관리자(Product Manager), 아키텍트(Architect), 프로젝트 관리자(Project Manager), 엔지니어(Engineer), QA 엔지니어(QA Engineer). 논문의 프레이밍: `Code = SOP(Team)`. 각 역할은 좁고 특화된 프롬프트를 가진다. 역할 간 핸드오프는 구조화된 산출물(PRD 문서, 아키텍처 문서, 코드)을 운반한다.

ChatDev의 기여: **소통적 탈환각**. 에이전트들이 답하기 전에 구체사항을 요청한다. 디자이너 에이전트가 UI를 스케치하기 전에 추측하는 대신 프로그래머에게 어떤 언어가 의도되었는지 묻는다. 논문은 이것이 다중 에이전트 파이프라인(pipeline)에서 환각을 측정 가능하게 줄인다고 보고한다.

MacNet (arXiv:2406.07155)은 ChatDev를 **DAG를 통해 >1000 에이전트로** 확장한다. 각 DAG 노드는 역할 특화다. 간선은 핸드오프 계약을 인코딩한다. 라우팅(routing)이 명시적이고 오프라인 계산 가능하기 때문에 이 규모가 가능하다.

설계 교훈:

1. **구조가 크기보다 더 중요하다.** 빡빡한 5역할 SOP 팀이 50개 에이전트의 비구조화된 그룹을 이긴다.
2. **문서로 된 핸드오프 계약.** 역할 간 전달되는 산출물은 스키마를 따른다.
3. **소통적 탈환각**은 저렴하면서 핵심적인 패턴이다.
4. **DAG가 채팅보다 더 멀리 확장된다.** 흐름을 알 수 있을 때, 그것을 인코딩하라.

이것이 역할 특화(Phase 16 · 08)와 구조화된 토폴로지(Phase 16 · 15)의 참조 사례다.

### OpenClaw / Moltbook 생태계

프로덕션 집단 규모 사례. 타임라인:

- **2025년 11월:** Clawdbot(Peter Steinberger의 로컬 ReAct 루프 코딩 에이전트) 출시.
- **2025년 12월 – 2026년 3월:** 두 번 이름 변경(Clawdbot → OpenClaw → OpenClaw로 계속).
- **2026년 2월:** Moltbook이 같은 프리미티브 위에서 에이전트 전용 소셜 네트워크로 출시. 며칠 만에 약 230만 에이전트 계정.
- **2026년 3월 (2026-03-10):** Meta가 Moltbook 인수.
- **2026년 3월:** 중국이 정부 컴퓨터에서 OpenClaw 제한.
- **2026년 3월:** OpenClaw가 247k GitHub 스타 돌파.

이것이 수백만 개의 에이전트를 공유 기반 위에 올렸을 때 다중 에이전트가 보이는 모습이다:

- **창발적 경제 활동.** 에이전트들이 토큰 지불을 사용해 서로 사고, 팔고, 서비스한다.
- **집단 규모의 프롬프트 주입 위험.** 바이럴 에이전트 프로필 속의 악의적 프롬프트 하나가 몇 시간 내에 수천 건의 에이전트 간 상호작용으로 전파된다.
- **국가 수준 규제 대응.** 출시 몇 주 내에 규제가 생태계에 도달한다.

이 사례의 설계 교훈은 부분적으로 기술적이고, 부분적으로 거버넌스적이다:

1. **집단 규모의 다중 에이전트는 새로운 영역이다.** 개별 시스템 모범 사례(검증, 역할 명확성)는 여전히 적용되지만 충분하지 않다.
2. **프롬프트 주입은 새로운 XSS다.** 에이전트 프로필과 에이전트 간 메시지를 기본적으로 신뢰할 수 없는 입력으로 다뤄라.
3. **규제가 설계 주기보다 빠르다.** 이에 대비하라.
4. **오픈소스 + 바이럴 규모는 복리로 불어난다.** 약 4개월 만에 247k 스타는 이례적이다. 배포 폭증 부하에 대비해 설계하라.

생태계 세부 사항은 [OpenClaw Wikipedia](https://en.wikipedia.org/wiki/OpenClaw)와 CNBC / Palo Alto Networks 보도를 보라. 기술적 토대로는, Clawdbot / OpenClaw 저장소가 로컬 ReAct 루프를 노출하고, Moltbook의 공개 게시물이 그 위의 소셜 그래프 아키텍처를 드러낸다.

### 2026년 4월 프레임워크 지형

| 프레임워크 | 상태 | 적합 용도 | 비고 |
|---|---|---|---|
| **LangGraph** (LangChain) | 프로덕션 선두 | 구조화된 그래프 + 체크포인팅 + 인간 개입(human-in-the-loop) | 프로덕션 권장 기본값 |
| **CrewAI** | 프로덕션 선두 | Sequential/Hierarchical 프로세스를 가진 역할 기반 크루 | 역할 분해에 강함 |
| **AG2** | 커뮤니티 유지보수 | GroupChat + 발화자 선택 | AutoGen v0.2 계승 |
| **Microsoft AutoGen** | 유지보수 모드 (2026년 2월) |: | Microsoft Agent Framework RC로 병합 |
| **Microsoft Agent Framework** | RC (2026년 2월) | 오케스트레이션 패턴 + 엔터프라이즈 통합 | 신규 진입자. 주시 |
| **OpenAI Agents SDK** | 프로덕션 | Swarm 후계자 | 도구 반환 핸드오프 패턴 |
| **Google ADK** | 프로덕션 (2025년 4월) | A2A 네이티브 | Google Cloud 통합 |
| **Anthropic Claude Agent SDK** | 프로덕션 | 단일 에이전트 + Research 확장 | Research 시스템 게시물 참조 |

이제 모든 주요 프레임워크가 **MCP** 지원을 출하하고, 대부분 **A2A**를 출하한다. 프로토콜 호환성은 더 이상 차별화 요소가 아니다.

### 세 사례 전반의 공통 패턴

1. **오케스트레이터 + 워커**(Anthropic의 명시적 감독자, MetaGPT의 PM-as-supervisor, OpenClaw의 개별 에이전트 + 네트워크 효과).
2. **구조화된 핸드오프 계약**(Anthropic 서브에이전트 과제 기술, MetaGPT PRD/아키텍처 문서, OpenClaw A2A 산출물).
3. **일급 역할로서의 검증**(Anthropic의 검증자, MetaGPT의 QA 엔지니어, OpenClaw의 네트워크 내 검증자).
4. **확장은 단지 더 많은 에이전트가 아니라 토폴로지 + 기반이다**(레인보우 배포, MacNet DAG, 집단 규모 기반).
5. **비용은 실질적이며 공개된다**(15배 토큰, MetaGPT의 역할별 예산, Moltbook의 상호작용당 가격).
6. **보안 태세는 명시적이다**(Anthropic의 샌드박싱, MetaGPT의 역할 제한, OpenClaw의 알려진 공격 표면으로서의 프롬프트 주입).

### 다음 프로젝트를 위한 참조 선택

- **프로덕션 연구 / 지식 과제 → Anthropic Research.** 신선한 컨텍스트 서브에이전트가 이긴다.
- **엔지니어링 / 도구 체인 워크플로 → MetaGPT / ChatDev.** 역할 + SOP + 핸드오프 계약.
- **네트워크 효과 소셜 제품 → OpenClaw / Moltbook.** 기반 + 창발적 경제.
- **고전적 엔터프라이즈 자동화 → CrewAI 또는 LangGraph**(프로덕션 선두, 안정적 런타임).

### 2026년 최첨단 요약

2026년 4월 이 분야의 위치:

- **프레임워크가 수렴하고 있다.** MCP + A2A 지원은 기본 요건이다. 핸드오프 의미론이 남은 설계 선택이다.
- **평가가 단단해지고 있다.** SWE-bench Pro, MARBLE, STRATUS 완화 벤치마크. Pro가 현재의 오염 저항적 현실 점검이다.
- **프로덕션 실패율이 측정 가능하다**(Cemri 2025 MAST; 실제 MAS에서 41-86.7%). 이 분야는 "데모에서는 멋져 보임" 시대를 벗어났다.
- **비용이 핵심 엔지니어링 제약이다.** 과제당 토큰 비용, 상호작용당 벽시계 시간, 레인보우 배포 오버헤드. 다중 에이전트는 정확도로 이기지만 비용으로 진다. 그 거래가 비즈니스 결정이다.
- **규제는 배경 우려가 아니라 단기 입력이다.** 관할권이 개별 배포 주기보다 빠르게 움직이고 있다.

```figure
a5-orchestrator-scale
```

## 라이브러리로 써보기 (Use It)

`outputs/skill-case-study-mapper.md`는 제안된 다중 에이전트 시스템 설계를 읽고 가장 가까운 사례 연구에 매핑해, 그 사례 연구가 이미 시험한 설계 결정을 드러내는 스킬(skill)이다.

## 산출물 (Ship It)

2026년 프로덕션 다중 에이전트를 위한 시작 규칙:

- **밑바닥이 아니라 사례 연구에서 시작하라.** Anthropic Research / MetaGPT / OpenClaw 중 가장 가까운 것을 골라 적응시켜라.
- **MCP + A2A를 채택하라.** 프레임워크 간 이식성은 가치 있다. 프로토콜 지원은 공짜다.
- **SWE-bench Pro 또는 내부 Pro 등가물에 대해 측정하라.** Verified는 오염되었다.
- **검증 세금을 지불하라.** 독립적 검증자는 토큰 예산의 약 20-30%가 들지만 측정 가능한 정확성을 산다.
- **장기 실행 에이전트를 레인보우 배포하라.** 여러 시간짜리 에이전트 실행이 일상적이라고 예상하라.
- **WMAC 2026과 MAST 후속 연구를 읽어라.** 이 학문은 빠르게 움직이고 있다.

## 연습 문제 (Exercises)

1. Anthropic Research 시스템 게시물을 처음부터 끝까지 읽어라. Opus 4를 더 작은 모델(예: Haiku 4)로 교체하면 바뀔 세 가지 설계 결정을 식별하라.
2. MetaGPT 3-4절(arXiv:2308.00352)을 읽어라. 자신의 도메인(소프트웨어가 아닌)에서 SOP 하나를 역할 프롬프트로 인코딩하라. 그 SOP는 몇 개의 역할을 함의하는가?
3. ChatDev (arXiv:2307.07924)을 읽어라. "소통적 탈환각"의 메커니즘을 식별하라. 그 메커니즘을 기존 다중 에이전트 시스템 중 하나에 구현하라.
4. OpenClaw와 Moltbook을 읽어라. 5개 에이전트 시스템에서는 나타나지 않을, 집단 규모에서 창발한 특정 실패 모드 하나를 골라라. 그 모드를 어떻게 방어 설계하겠는가?
5. 현재 진행 중인 다중 에이전트 프로젝트를 골라라. 세 사례 연구 중 어느 것이 가장 가까운 참조인가? 그 사례 연구의 설계 결정 중 아직 채택하지 *않은* 것은 무엇인가? 이번 분기에 채택할 것 하나를 적어라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Anthropic Research | "감독자 참조" | Claude Opus 4 + Sonnet 4 서브에이전트. 15배 토큰. 단일 에이전트 대비 +90.2%. |
| MetaGPT | "프롬프트로서의 SOP" | 소프트웨어 엔지니어링을 위한 역할 분해. `Code = SOP(Team)`. |
| ChatDev | "역할로서의 에이전트" | 디자이너 / 프로그래머 / 검토자 / 테스터. 소통적 탈환각. |
| MacNet | "DAG를 통한 ChatDev 확장" | arXiv:2406.07155. 명시적 DAG 라우팅을 통한 1000+ 에이전트. |
| OpenClaw | "로컬 ReAct 루프 에이전트" | Steinberger의 프로젝트. 2026년 3월까지 247k 스타. |
| Moltbook | "에이전트 전용 소셜 네트워크" | 230만 에이전트 계정. 2026년 3월 Meta에 인수. |
| 레인보우 배포(Rainbow deploy) | "여러 버전 동시 실행" | 진행 중인 장기 실행 에이전트를 위해 옛 런타임 버전을 살려 둠. |
| 소통적 탈환각(Communicative dehallucination) | "답하기 전에 물어보기" | 에이전트들이 추측하는 대신 동료에게 구체사항을 요청. |
| WMAC 2026 | "AAAI 워크숍" | 다중 에이전트 협응을 위한 2026년 4월 커뮤니티 초점. |

## 더 읽을거리 (Further Reading)

- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) 감독자-워커 프로덕션 참조
- [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) SOP 역할 분해
- [ChatDev: Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) 소통적 탈환각
- [MacNet: scaling role-based agents to 1000+](https://arxiv.org/abs/2406.07155) DAG 기반 확장
- [OpenClaw on Wikipedia](https://en.wikipedia.org/wiki/OpenClaw): 생태계 개요
- [WMAC 2026](https://multiagents.org/2026/) — AAAI 2026 Bridge Program Workshop on Multi-Agent Coordination
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/workflows-agents): 프로덕션 선두
- [CrewAI docs](https://docs.crewai.com/en/introduction): 역할 기반 프레임워크
