# LMCache KV 오프로딩을 갖춘 vLLM 프로덕션 스택(vLLM Production Stack with LMCache KV Offloading)

> vLLM의 production-stack은 레퍼런스 쿠버네티스(Kubernetes) 배포다. 라우터, 엔진, 관측성(observability)이 함께 연결되어 있다. LMCache는 KV 캐시를 GPU 메모리에서 빼내어 쿼리와 엔진 전반에서 재사용하는 KV 오프로딩(offloading) 계층이다(CPU DRAM, 그다음 디스크/Ceph). vLLM 0.11.0 KV Offloading Connector(2026년 1월)는 이것을 Connector API(v0.9.0+)를 통해 비동기적이고 플러그형(pluggable)으로 만든다. 오프로드 지연 시간(latency)은 사용자에게 노출되지 않는다. LMCache는 공유 프리픽스(prefix)가 없어도 가치가 있다. GPU가 KV 슬롯을 소진하면, 선점된(preempted) 요청을 프리필(prefill)을 재계산하는 대신 CPU에서 복원할 수 있다. 4개의 a3-highgpu-4g에 걸친 16x H100(80GB HBM)에서 발표된 벤치마크: KV 캐시가 HBM을 초과하면, 네이티브 CPU 오프로드와 LMCache 둘 다 처리량을 상당히 개선한다. 낮은 KV 풋프린트에서는 모든 구성이 작은 오버헤드로 베이스라인과 일치한다.

**Type:** Learn
**Languages:** Python (stdlib, toy KV-spill simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- vLLM production-stack 계층을 다이어그램으로 그리기: 라우터, 엔진, KV 오프로드, 관측성.
- KV Offloading Connector API(v0.9.0+)와 0.11.0 비동기 경로가 오프로드 지연 시간을 어떻게 숨기는지 설명하기.
- LMCache CPU-DRAM이 도움이 되는 경우(KV > HBM)와 오버헤드를 더하는 경우(KV가 HBM에 들어갈 만큼 작음)를 정량화하기.
- 배포 제약이 주어졌을 때 네이티브 vLLM CPU 오프로드와 LMCache 커넥터 중에서 고르기.

## 문제 (The Problem)

vLLM 서빙은 동시성이 오를 때마다 선점(preemption) 이벤트와 함께 GPU의 HBM이 100% 차는 모습을 보인다. 요청이 축출(evict)되고 재큐잉되며 같은 2K 토큰 프롬프트를 1분에 네 번 재프리필한다. GPU 연산이 중복 프리필에 쓰인다. 굿풋(goodput)이 원시 처리량보다 한참 아래다.

GPU를 더 추가하는 것은 선형적으로 비용이 든다. HBM을 더 추가하는 것은 불가능하다. 하지만 CPU DRAM은 싸다. 한 소켓에 512 GB+가 HBM보다 자릿수 단위로 나쁜 지연 시간으로 있지만, "일시적으로 따뜻한" KV 캐시에는 괜찮다.

LMCache는 선점된 요청이 빠르게 복구되도록 KV 캐시를 CPU DRAM으로 빼내고, 엔진 전반의 반복되는 프리픽스가 각 엔진이 재프리필하지 않고 캐시를 공유하게 한다.

## 개념 (The Concept)

### vLLM production-stack

`github.com/vllm-project/production-stack`은 레퍼런스 쿠버네티스 배포다:

- **라우터(Router)**: 캐시 인식(Phase 17 · 11). KV 이벤트를 소비한다.
- **엔진(Engines)**: vLLM 워커. GPU당 하나 또는 TP/PP 그룹당 하나.
- **KV 캐시 오프로드**: LMCache 배포 또는 네이티브 커넥터.
- **관측성(Observability)**: Prometheus 스크레이프, Grafana 대시보드, OTel 트레이스.
- **컨트롤 플레인(Control plane)**: 서비스 디스커버리, 설정, 롤링 업데이트.

Helm 차트 + 오퍼레이터(operator)로 출시된다.

### KV Offloading Connector API (v0.9.0+)

vLLM 0.9.0은 플러그형 KV 캐시 백엔드를 위한 Connector API를 도입했다. 엔진이 블록을 커넥터로 오프로드하면 커넥터가 이를 저장한다(RAM, 디스크, 오브젝트 스토리지, LMCache). 요청에 블록이 필요해지면 커넥터가 다시 로드한다.

vLLM 0.11.0(2026년 1월)은 비동기 오프로드 경로를 추가한다. 오프로드가 백그라운드에서 일어날 수 있어 일반적인 경우 엔진이 여기에 블로킹되지 않는다. 종단 간 지연 시간과 처리량은 여전히 워크로드 형태, KV 캐시 적중률, 시스템 압박에 의존한다. vLLM 자체 노트는 커스텀 커널 오프로드가 낮은 적중률에서 처리량을 저하시킬 수 있고, 비동기 스케줄링이 추측 디코딩(speculative decoding)과 알려진 상호작용 문제가 있음을 지적한다.

### 네이티브 CPU 오프로드 대 LMCache

**네이티브 vLLM CPU 오프로드**: 엔진 로컬. KV 블록을 호스트 RAM에 저장한다. 구현이 빠르고, 네트워크 홉(hop)이 없다. 엔진을 가로지르지 않는다.

**LMCache 커넥터**: 클러스터 규모. 블록을 공유 LMCache 서버(CPU DRAM + Ceph/S3 티어)에 저장한다. 블록이 어느 엔진에서든 접근 가능하다. 16x H100 벤치마크 발표됨.

단일 엔진에 HBM 압박이 있을 때 네이티브를 고르라. 여러 엔진이 프리픽스를 공유할 때(공통 시스템 프롬프트가 있는 RAG, 공유 템플릿이 있는 멀티 테넌트) LMCache를 고르라.

### 벤치마크 동작

4개의 a3-highgpu-4g에 걸친 16x H100(80 GB HBM) 테스트:

- 낮은 KV 풋프린트(짧은 프롬프트, 낮은 동시성): 모든 구성이 베이스라인과 일치, LMCache가 ~3-5% 오버헤드 추가.
- 중간 풋프린트: LMCache가 엔진 전반의 프리픽스 재사용에서 도움을 주기 시작.
- KV가 HBM 초과: 네이티브 CPU 오프로드와 LMCache 둘 다 처리량을 상당히 개선. 엔진 간 공유 덕분에 LMCache가 더 큰 이득.

### LMCache가 결정적인 경우

- 시스템 프롬프트가 테넌트 전반에 공유되는 멀티 테넌트 서빙.
- 문서 청크가 쿼리 전반에 반복되는 RAG.
- 베이스 모델 KV 재사용이 중복 작업을 줄이는, 같은 베이스의 파인튜닝 변형(LoRA).
- 선점 중심 워크로드: CPU에서 복원이 재프리필보다 저렴.

### 켜면 안 되는 경우

- 작은 HBM 압박: 이득 없이 오버헤드를 지불한다.
- 짧은 컨텍스트(<1K 토큰): 전송 시간 > 재프리필.
- 단일 테넌트 단일 프롬프트 워크로드: 포착할 재사용이 없다.

### 분리형 서빙과의 통합

Phase 17 · 17 분리형 서빙 + LMCache는 복합적이다: 프리필 풀에서 디코드 풀로의 KV 전송이 쓰이지 않으면 LMCache에 안착하고, 이후 쿼리가 LMCache에서 가져온다. Phase 17 · 11 캐시 인식 라우터는 로컬 또는 LMCache 공유 캐시가 일치하는 엔진으로 라우팅할 수 있다.

### 기억해야 할 숫자들

- vLLM 0.9.0: Connector API 출시.
- vLLM 0.11.0(2026년 1월): 비동기 오프로드 경로. 종단 간 지연 시간 영향은 워크로드, KV 적중률, 시스템 압박에 의존(절대적 보장 아님).
- 16x H100 벤치마크: KV 풋프린트가 HBM을 초과할 때 LMCache가 도움.
- 작은 HBM 압박: 이득 없이 3-5% 오버헤드.

```figure
zero-sharding
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 LMCache가 있을 때와 없을 때 선점 중심 워크로드를 시뮬레이션한다. 회피된 재프리필, 처리량 이득, 손익분기 HBM 사용률을 보고한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-vllm-stack-decider.md`를 생성한다. 워크로드 형태와 vLLM 배포가 주어지면 네이티브 대 LMCache 대 둘 다 아님을 결정한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 어느 HBM 사용률에서 LMCache가 이득이 되기 시작하는가?
2. 한 테넌트가 6K 토큰 시스템 프롬프트를 시간당 200 쿼리 전반에 공유한다. 테넌트당 예상 LMCache 절감을 계산하라.
3. LMCache 서버는 단일 장애점(single point of failure)이다. HA 전략(레플리카, 네이티브로 폴백)을 설계하라.
4. LMCache가 회전 디스크의 Ceph에 저장한다. 70B FP8에서 4K 토큰 KV(500 MB)에 대해, 읽기 시간 대 재프리필은 무엇인가?
5. vLLM 0.11.0 비동기 경로가 "공짜"인지 논하라. 오버헤드는 어디에 숨는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| Production-stack | "레퍼런스 배포" | vLLM의 쿠버네티스 Helm 차트 + 오퍼레이터 |
| Connector API | "KV 백엔드 인터페이스" | vLLM 0.9.0+ 플러그형 KV 스토어 인터페이스 |
| 네이티브 CPU 오프로드 (Native CPU offload) | "엔진 로컬 스필" | 같은 엔진의 호스트 RAM에 KV 저장 |
| LMCache | "클러스터 KV 캐시" | CPU DRAM + 디스크의 엔진 간 KV 캐시 서버 |
| 0.11.0 async | "논블로킹 오프로드" | 엔진 스트림 뒤에 숨겨진 오프로드 |
| 선점 (Preemption) | "공간을 만들려고 축출" | HBM이 가득 찼을 때 KV 캐시 셔플 |
| 프리픽스 재사용 (Prefix reuse) | "같은 시스템 프롬프트" | 여러 쿼리가 시작 부분을 공유; 캐시 적중 |
| Ceph 티어 (Ceph tier) | "디스크 티어" | 캐시 계층에서 DRAM 아래의 내구성 스토리지 |

## 더 읽을거리 (Further Reading)

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack): Helm 차트 + 오퍼레이터.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache): Connector 구현.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases): 비동기 경로 세부 사항.
