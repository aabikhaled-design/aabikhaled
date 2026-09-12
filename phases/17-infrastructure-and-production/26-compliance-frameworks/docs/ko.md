# 컴플라이언스: SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001

> 다중 프레임워크(multi-framework) 커버리지는 2026년 엔터프라이즈 거래의 기본 요건(table stakes)이다. **EU AI Act**: 2024년 8월 1일부터 발효. 대부분의 고위험(high-risk) 요건은 2026년 8월 2일에 집행된다. 고위험 시스템 의무 위반에 대해 최대 €15M 또는 전 세계 연 매출 3%의 벌금(Art. 99(4)); 금지된 AI 관행(prohibited AI practices)에는 최대 €35M 또는 7%(Art. 99(3)). EU 사용자에게 서비스하면 전 세계적으로 적용된다. **Colorado AI Act**: 2026년 6월 30일 발효(SB25B-004에 의해 2026년 2월에서 연기됨)(고위험 시스템에 대한 영향 평가(impact assessment), AI 결정에 대한 이의 제기권. Virginia는 신용/고용/주택/교육에 대해 유사하다. **SOC 2 Type II**: 사실상의 B2B AI 요건(핀테크에는 Type I이 아니라 Type II). **GDPR**: 문서화된 가장 큰 AI 특화 벌금은 Clearview AI에 대한 €30.5M(네덜란드 DPA, 2024년 9월); 이탈리아 Garante가 2024년 12월 OpenAI에 €15M을 부과했다(2026년 3월 항소심에서 뒤집힘). 추론 시점의 실시간 PII 수정(redaction)이 방어 가능한 표준이다; 사후 처리(post-processing) 정리만으로는 충분하지 않다. **HIPAA**: 헬스케어에 구속됨) BAA 없이는 외부 AI 서비스에 PHI를 보낼 수 없다. **PCI-DSS**: AI 상호작용 계층 커버리지는 자동이 아니라 설정 + 계약 합의를 요구한다. **ISO 42001**: 떠오르는 AI 거버넌스 표준, ISO 27001과 나란히 조달(procurement) 요건으로 성장 중. 참조 프로파일: OpenAI는 SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, ChatGPT 결제 구성요소에 대한 PCI-DSS를 유지한다. 교차 프레임워크 매핑(cross-framework mapping)은 감사 피로를 줄인다: 접근 제어는 ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a)에 걸쳐 매핑된다.

**Type:** Learn
**Languages:** (Python optional — compliance is policy + process, not code)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- LLM 제품에 관련된 2026년 일곱 프레임워크를 열거하고 각각을 고객 세그먼트(segment)에 대응시키기.
- EU AI Act 집행 일정(2024년 8월 발효; 고위험 집행 2026년 8월)과 2단계 벌금 상한(고위험 의무에 €15M / 3%, 금지된 관행에 €35M / 7%)을 인용하기.
- 사후 처리 PII 정리가 GDPR에 충분하지 않은 이유를 설명하고 실시간 추론 계층 수정을 방어 가능한 표준으로 명명하기.
- 교차 프레임워크 제어 매핑(예: 접근 제어가 ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)에 매핑됨)을 설명하기.

## 문제 (The Problem)

엔터프라이즈 고객의 조달 부서가 SOC 2 Type II, GDPR, HIPAA BAA, ISO 27001, 그리고 "EU AI Act 컴플라이언스 진술서"를 요구한다. 팀이 갖춘 건 SOC 2 Type I뿐이다. Type II까지 6개월 남았고 GDPR 제30조 기록(records)은 시작도 안 했다.

다중 프레임워크 커버리지는 LLM 문제가 아니라 엔터프라이즈-SaaS 문제이며, 여기에 LLM 특화 덧입힘(overlay)이 붙는다. 2026년 조달 팀은 PDF가 아니라 프레임워크당 행 하나, 제어(control)당 열 하나를 가진 매트릭스를 원한다.

## 개념 (The Concept)

### 일곱 프레임워크

| 프레임워크 | 범위 | LLM 특화 요건 |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS 기준선 | 6~12개월에 걸쳐 감사되는 프로세스 제어 |
| HIPAA | 미국 헬스케어 | BAA 필수; 서명된 합의 없이 PHI가 인프라를 떠날 수 없음 |
| GDPR | EU 사용자 | 실시간 PII 수정; 정보 주체 권리; 제30조 기록 |
| PCI-DSS | 결제 데이터 | 결제를 다루는 AI에 대한 설정 + 계약 |
| EU AI Act | EU 사용자 서비스 | 위험 등급 분류; 고위험 시스템: 적합성 평가(conformity assessment), 문서화, 로깅 |
| Colorado AI Act | CO 거주자 서비스 | 영향 평가; 이의 제기권 |
| ISO 42001 | AI 거버넌스 | 떠오름; ISO 27001과 짝을 이룸 |

### EU AI Act 일정

- 2024년 8월 1일: 발효.
- 2025년 2월 2일: 금지된 AI 관행 집행.
- 2026년 8월 2일: 고위험 시스템 집행(적합성 평가, 문서화, 로깅).
- 2027년 8월: 조화 법령(harmonized legislation) 하의 제품에 든 고위험 시스템.

위험 등급: 허용 불가(Unacceptable)(금지), 고위험(High-risk)(적합성 + 로깅), 제한 위험(Limited-risk)(투명성), 최소 위험(Minimal-risk)(제약 없음). 대부분의 B2B LLM SaaS는 제한 위험이다; 고위험은 고용, 신용, 교육, 법 집행, 이주(migration), 필수 서비스에 발동한다.

벌금(제99조): 고위험 시스템 의무 위반에 최대 €15M 또는 전 세계 연 매출 3%(Art. 99(4)); 금지된 AI 관행에 최대 €35M 또는 7%(Art. 99(3)); 둘 중 더 높은 것이 적용된다.

### GDPR: 실시간 수정이 표준이다

사후 처리 정리(LLM이 본 후에 PII를 수정)는 방어 가능한 자세가 아니다. 모델이 이미 데이터를 봤기 때문이다. 실시간 추론 계층 수정이 2026년 표준이다:

- LLM 호출 전 개체 인식(entity recognition).
- 일관된 토큰화(Mesh 방식)가 의미를 보존한다.
- 수정된 프롬프트 + 동의된 옵트인(opt-in) 원시 데이터만 저장한다.

최근 집행: Clearview AI에 대한 €30.5M(네덜란드 DPA, 2024년 9월)이 현재까지 문서화된 가장 큰 AI 특화 GDPR 벌금이다; OpenAI에 대한 €15M(이탈리아 Garante, 2024년 12월)은 가장 큰 LLM 특화 벌금이지만, 2026년 3월 항소심에서 뒤집혔고 판결은 추가 검토 중이다. 사후 처리 주장은 감사에서 실패해 왔다.

### HIPAA: BAA는 선택이 아니다

서명된 BAA(Business Associate Agreement) 없이는 외부 AI 서비스에 PHI를 보낼 수 없다. 세 하이퍼스케일러 LLM 플랫폼(Bedrock, Azure OpenAI, Vertex) 모두 BAA를 제공한다. OpenAI 직접 API는 BAA를 제공한다. Anthropic 직접 API는 BAA를 제공한다. PHI를 보내기 전에 확인하라.

### SOC 2 Type II

Type I: 제어가 설계되고 문서화됨.
Type II: 제어가 6~12개월에 걸쳐 효과적으로 작동함.

2026년 B2B 조달은 기본적으로 Type II를 요구한다. Type I은 시작일 뿐이다. Type II가 관문이다.

흔한 감사 동인: 접근 로그(누가 무엇을 봤는가), 변경 관리(어떻게 배포됐는가), 위험 평가(분기마다), 인시던트 대응(테스트됐는가?). Phase 17 · 25의 감사 로그는 직접 재사용 가능하다.

### 교차 프레임워크 매핑

하나의 접근 제어 정책이 여러 프레임워크 제어를 충족한다:

| 제어 | 프레임워크 |
|---------|-----------|
| 접근 로깅 | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| 변경 관리 | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach-notification scope |
| 전송 중 암호화 | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| 시크릿 관리 | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

컴플라이언스 도구(Drata, Vanta, Secureframe)가 이 매핑을 자동화한다. 규모가 커지면 비용을 들일 가치가 있다.

### ISO 42001: 떠오름

2023년 말 발행. ISO 27001과 나란히 조달 요건으로 성장 중. 위험 관리, 데이터 품질, 투명성, 사람 감독(human oversight)을 포함한 AI 거버넌스 프레임워크.

### OpenAI의 참조 프로파일

OpenAI는 SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, ChatGPT 결제 구성요소에 대한 PCI-DSS를 유지한다. 이 구성이 2026년 엔터프라이즈 기본 요건에 대략 해당한다.

### 기억해 둘 숫자들

- EU AI Act 벌금: 최대 €15M / 3%(고위험 의무, Art. 99(4)); 최대 €35M / 7%(금지된 관행, Art. 99(3)).
- EU AI Act 고위험 집행: 2026년 8월 2일.
- 문서화된 가장 큰 AI 특화 GDPR 벌금: €30.5M, Clearview AI(네덜란드 DPA, 2024년 9월).
- 가장 큰 LLM 특화 GDPR 벌금: €15M, OpenAI(이탈리아 Garante, 2024년 12월; 2026년 3월 항소심에서 뒤집힘).
- SOC 2 Type II 기간: 작동된 제어 6~12개월.
- Colorado AI Act 발효일: 2026년 6월 30일(SB25B-004에 의해 2026년 2월에서 연기됨).

```figure
i4-control-matrix
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 Python으로 된 컴플라이언스 매핑 스프레드시트다. 제어가 주어지면 그것이 충족하는 프레임워크들을 나열한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-compliance-matrix.md`를 만든다. 고객 세그먼트와 지역이 주어지면 필요한 프레임워크와 제어를 명시한다.

## 연습 문제 (Exercises)

1. 첫 엔터프라이즈 고객이 SOC 2 Type II, HIPAA BAA, EU AI Act 진술서를 요구한다. 거래를 따내기 위한 최소 실행 가능 컴플라이언스 자세는 무엇인가?
2. 세 가지 가상의 LLM 제품을 EU AI Act 위험 등급으로 분류하라. 고위험에서 무엇이 바뀌는가?
3. 실수로 BAA 없이 프로바이더에 PHI를 보냈다. 인시던트 대응을 단계별로 밟아보라.
4. 미들마켓(mid-market) AI 벤더에게 ISO 42001이 "2026년에 필요한지" 논증하라.
5. LLM 감사 로그 필드(Phase 17 · 25)를 적어도 세 프레임워크 제어에 매핑하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| SOC 2 Type II | "감사된 제어" | 6~12개월에 걸쳐 작동하며 독립적으로 입증된 제어 |
| HIPAA BAA | "헬스케어 계약" | Business Associate Agreement; PHI에 필수 |
| GDPR | "EU 프라이버시" | 실시간 PII 수정이 방어 가능한 2026 표준 |
| EU AI Act | "EU AI 규칙" | 고위험 집행 2026년 8월; €15M / 3%(고위험 의무): €35M / 7%(금지된 관행) |
| Colorado AI Act | "미국 AI 주법" | 2026년 6월 30일 발효(SB25B-004에 의해 연기); 영향 평가 |
| ISO 42001 | "AI 거버넌스" | AI 위험 + 투명성을 위한 떠오르는 프레임워크 |
| ISO 27001 | "보안 ISMS" | 정보 보안 관리 시스템(Information Security Management System) 기준선 |
| 적합성 평가 (Conformity assessment) | "EU AI 문서 패키지" | 고위험 요건: 문서, 테스트, 로깅 |
| 교차 프레임워크 매핑 (Cross-framework mapping) | "하나의 제어, 여러 프레임" | 단일 정책이 여러 프레임워크 제어를 충족 |

## 더 읽을거리 (Further Reading)

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile.
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard.
