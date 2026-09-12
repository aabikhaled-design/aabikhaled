# 보안: 시크릿, API 키 순환, 감사 로그, 가드레일

> 중앙집중식 볼트(vault)(HashiCorp Vault, AWS Secrets Manager, Azure Key Vault)를 통해 시크릿 난립(secret sprawl)을 제거하라. 자격 증명(credential)을 설정 파일, VCS 안의 env 파일, 스프레드시트에 절대 저장하지 마라. 정적 키 대신 IAM 역할(role)을 쓰고, CI/CD에는 OIDC를 쓴다. AI 게이트웨이(gateway) 패턴이 2026년의 해법이다: 앱 → 게이트웨이 → 모델 프로바이더(provider). 게이트웨이가 런타임에 볼트에서 자격 증명을 가져온다. 볼트에서 순환(rotate)하면 모든 앱이 몇 분 안에 받아간다. 재배포(redeploy)도, "새 키 누가 가졌어"라는 슬랙 메시지도 없다. 순환 정책 ≤90일; 모든 커밋에서 TruffleHog / GitGuardian / Gitleaks로 스캔. 제로 트러스트(zero-trust): MFA, SSO, RBAC/ABAC, 단기 토큰(short-lived token), 디바이스 자세(device posture). PII 스크러빙(scrubbing)은 개체 인식(entity recognition)을 사용해 전달 전에 PHI/PII를 마스킹한다. 일관된 토큰화(consistent tokenization)(Mesh 방식)는 민감 값을 안정적인 자리표시자(placeholder)에 매핑하여 LLM이 코드/관계 의미를 보존하게 한다. 네트워크 이그레스(egress): LLM 서비스를 전용 VPC/VNet 서브넷에 두고 `api.openai.com`, `api.anthropic.com` 등만 화이트리스트(whitelist)하고, 그 외 모든 아웃바운드는 차단한다. 2026년 인시던트 동인: 손상된 CI/CD 자격 증명을 통한 Vercel 공급망 공격(supply-chain attack)이 수천 개 고객 배포에 걸쳐 env 변수를 유출했다.

**Type:** Learn
**Languages:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 네 가지 시크릿 관리 안티패턴(VCS 안의 설정 파일, 하드코딩된 env, 스프레드시트, 정적 키)을 열거하고 그 대체물을 명명하기.
- AI 게이트웨이가 볼트에서 가져오는 패턴을 2026년 프로덕션 표준으로 설명하기.
- 일관된 토큰화(같은 값 → 같은 자리표시자)를 갖춘 PII 스크러버를 구현하여 의미가 살아남게 하기.
- 2026년 Vercel 공급망 인시던트와 그것이 CI/CD 자격 증명 위생(hygiene)에 대해 가르친 바를 명명하기.

## 문제 (The Problem)

인턴이 API 키가 든 `.env`를 커밋한다. 빠르게 삭제한다. 키는 이미 git 히스토리에 있다. GitGuardian 스캔이 잡아내고, 순환 절차라고는 "팀에 슬랙하고, 40개 설정 파일을 갱신하고, 모든 서비스를 재배포한다"가 전부다. 8시간 후, 서비스의 절반은 라이브고 절반은 배포 윈도를 기다린다.

별개로, 사용자 프롬프트에 "내 SSN은 123-45-6789야"가 포함된다. 프롬프트가 OpenAI로 간다. BAA는 있지만 내부 정책은 전달 전에 PII를 마스킹하는 것이다. 하지 않았다.

별개로, EKS 클러스터의 LLM 파드가 임의의 인터넷 호스트에 도달할 수 있다. 누군가 공격자가 통제하는 도메인으로의 DNS 조회를 통해 데이터를 유출(exfil)한다. 아무것도 막지 못했다.

LLM 서비스의 보안은 이 세 벡터를 모두 다뤄야 한다. 볼트 기반 자격 증명. PII 스크러빙. 네트워크 이그레스 필터링. 감사 로그.

## 개념 (The Concept)

### 중앙집중식 볼트 + IAM 역할 가져오기

**볼트**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. 단일 진실 출처(source of truth).

**IAM 역할**: 앱/게이트웨이가 정적 키가 아니라 자신의 IAM 신원으로 인증한다. 볼트는 토큰의 수명 동안 시크릿을 반환한다.

**AI 게이트웨이 패턴**: 게이트웨이가 요청 시점에 볼트에서 `OPENAI_API_KEY`를 가져온다. 볼트에서 순환하면 다음 요청이 새 키를 받는다. 재배포 없음.

### 순환 정책 ≤ 90일

모든 API 키, 볼트 루트 토큰, CI/CD 자격 증명. 가능한 곳에는 자동 순환. 수동 순환은 로그를 남기고 추적한다.

### 시크릿 스캔

- **TruffleHog**: 커밋에 대한 정규식 + 엔트로피.
- **GitGuardian**: 상용, 높은 정확도.
- **Gitleaks**: OSS, CI에서 실행.

모든 커밋에서 실행한다. 새 시크릿이 탐지되면 PR을 차단한다.

### 제로 트러스트 자세

- 모든 계정에 MFA 필수.
- SAML/OIDC를 통한 SSO.
- 세밀한 접근을 위한 RBAC(역할 기반) 또는 ABAC(속성 기반).
- 단기 토큰(일이 아니라 시간 단위).
- 디바이스 자세: 디스크 암호화가 된 회사 기기만.

### PII / PHI 스크러빙

프롬프트가 인프라를 떠나기 전에:

1. 개체 인식(spaCy NER, Presidio, 상용).
2. 일치한 개체를 마스킹: `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`.
3. 일관된 토큰화(Mesh 방식): 같은 값이 같은 자리표시자에 매핑되어 LLM이 관계를 보존한다.
4. LLM 응답을 위한 선택적 역매핑.

정적 정규식 필터는 기본 패턴을 잡고, NER은 더 많이 잡는다. 둘 다 쓴다.

### 입력 + 출력 가드레일

입력: 알려진 탈옥(jailbreak), 금지 주제를 차단; 사용자별 속도 제한(rate-limit).

출력: 유출된 시크릿에 대한 정규식 스크럽(API 키 패턴, 거부 맥락의 이메일 패턴), 정책 위반에 대한 분류기(classifier).

### 네트워크 이그레스 화이트리스트

LLM 서비스를 전용 서브넷에 둔다:
- 화이트리스트: `api.openai.com`, `api.anthropic.com`, 벡터 DB 엔드포인트, 볼트 엔드포인트.
- 그 외 모든 것: 드롭(drop).
- 허용 목록 전용 리졸버를 통한 DNS(DNS 터널링 유출 회피).

### 감사 로그 (Audit log)

모든 LLM 호출에 대한 불변(immutable) 로그:
- 타임스탬프.
- 사용자 / 테넌트(tenant).
- 프롬프트 해시(개인정보 보호를 위해 원시 프롬프트가 아님).
- 모델 + 버전.
- 토큰 수.
- 비용.
- 응답 해시.
- 모든 가드레일 발동.

규제 요건에 따라 보관한다(SOC 2 1년, HIPAA 6년).

### 2026년 Vercel 인시던트

공급망 공격: 손상된 CI/CD 자격 증명이 수천 개 고객 배포에 걸쳐 env 변수를 유출했다. 교훈: CI/CD 자격 증명은 프로덕션과 동등하다. 볼트에 저장하라. 권한 범위를 좁게 잡아라. 공격적으로 순환하라.

### 기억해 둘 숫자들

- 순환 정책: ≤ 90일.
- 모든 커밋에서 스캔: TruffleHog / GitGuardian / Gitleaks.
- Vercel 2026: CI/CD 자격 증명 손상 → 수천 개 고객 env 변수 유출.
- 감사 로그 보관: SOC 2 = 1년, HIPAA = 6년.

```figure
i4-vault-rotation
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 일관된 토큰화를 갖춘 토이(toy) PII 스크러버와 추가 전용(append-only) 감사 로그를 구현한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-llm-security-plan.md`를 만든다. 규제 범위와 현재 상태가 주어지면 볼트 마이그레이션, 스크러버, 이그레스, 감사 로그를 계획한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 같은 SSN을 참조하는 두 프롬프트를 보내라. 둘 다 같은 자리표시자를 받는지 확인하라.
2. OpenAI + Anthropic + Weaviate를 호출하는 vLLM-on-EKS 배포를 위한 네트워크 이그레스 정책을 설계하라.
3. git 히스토리에서 키를 발견했다(2년 됨). 올바른 대응은 무엇인가: 키 순환, 히스토리 스크럽, 아니면 둘 다? 정당화하라.
4. 감사 로그가 하루 10GB씩 증가한다. 보관 계층(핫 30일, 웜 12개월, 콜드 6년)을 설계하라.
5. 역토큰화(실제 값을 LLM 응답에 다시 대입)가 자리표시자를 보이게 유지하는 것 대비 복잡성을 감수할 가치가 있는지 논증하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 볼트 (Vault) | "시크릿 저장소" | 중앙집중식 자격 증명 관리 서비스 |
| IAM 역할 (IAM role) | "신원 기반 인증" | 앱이 맡는 역할; 단기 자격 증명을 반환 |
| CI/CD용 OIDC (OIDC for CI/CD) | "클라우드 발급 토큰" | CI에 정적 키 없음: OIDC를 통한 신원 |
| TruffleHog / GitGuardian / Gitleaks | "시크릿 스캐너" | 커밋 시점 시크릿 탐지 |
| RBAC / ABAC | "접근 제어" | 역할 기반 vs 속성 기반 |
| PII 스크러빙 (PII scrubbing) | "데이터 마스킹" | 민감 개체 제거 또는 토큰화 |
| 일관된 토큰화 (Consistent tokenization) | "안정적 자리표시자" | 같은 값 → 매번 같은 토큰 |
| Mesh 방식 (Mesh approach) | "Mesh 토큰화" | 의미를 보존하는 토큰화 패턴 |
| 이그레스 화이트리스트 (Egress whitelist) | "아웃바운드 허용 목록" | 허용된 도메인에만 도달 가능 |
| 감사 로그 (Audit log) | "불변 히스토리" | 컴플라이언스를 위한 추가 전용 기록 |

## 더 읽을거리 (Further Reading)

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
