# vLLM 서빙 내부 구조: PagedAttention, 연속 배칭(Continuous Batching), 청크 프리필(Chunked Prefill)

> 2026년 vLLM의 지배력은 단일 트릭이 아니라 세 가지 복리적 기본값에 기반한다. PagedAttention은 항상 켜져 있다. 연속 배칭(continuous batching)은 디코드 반복 사이에 새 요청을 활성 배치(batch)에 주입한다. 청크 프리필(chunked prefill)은 긴 프롬프트를 잘게 썰어 디코드 토큰(token)이 결코 굶지 않게 한다. 셋 다 켜면 하나의 H100 SXM5에서 Llama 3.3 70B FP8이 동시 128에서 초당 2,200~2,400 토큰을 밀어낸다. vLLM 자체 기본값보다 약 25% 위, 순진한 PyTorch 루프의 3~4배다. 이 레슨은 스케줄러와 어텐션 커널을 직접 도식화할 수 있는 수준에서 읽고, vLLM이 하는 방식으로 프리필과 디코드를 스케줄링하는 장난감 연속 배처를 `code/main.py`에서 만드는 것으로 끝맺는다.

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- PagedAttention을 KV 캐시 할당자로 설명하기: 블록, 블록 테이블, 그리고 프로덕션(production) 부하에서 단편화(fragmentation)가 왜 4% 미만으로 유지되는지.
- 반복 수준에서 연속 배칭을 도식화하기: 완료된 시퀀스가 어떻게 배치를 떠나고 새 시퀀스가 배수(drain) 없이 합류하는지.
- 청크 프리필을 한 문장으로 설명하고 그것이 어느 지연 시간 지표를 보호하는지 명명하기(힌트: 평균 처리량이 아니라 TTFT 꼬리다).
- 모든 최적화를 한꺼번에 켜는 팀을 무는 2026년 vLLM v0.18.0의 함정을 명명하기.

## 문제 (The Problem)

순진한 PyTorch 서빙 루프는 한 번에 한 요청을 돌린다: 토크나이즈, 프리필, EOS까지 디코드, 반환. 사용자 한 명에는 이게 작동한다. 백 명에서는 인내심 있는 사람들의 대기열이 된다. 명백한 해법(정적 배칭(static batching))은 모든 요청을 윈도우 내 가장 긴 프롬프트에 맞춰 패딩(padding)하고, 모든 디코드를 가장 긴 예상 출력에 맞춰 패딩하며, 전체 배치를 가장 느린 시퀀스에서 멈춰 세운다. 결코 쓰지 않는 패딩에 비용을 내고, 빠른 요청은 느린 요청을 기다린다.

vLLM은 세 문제를 한꺼번에 해결한다. PagedAttention은 고전적 연속(contiguous) 할당이 하는 방식으로 KV 캐시 단편화가 GPU 메모리의 60~80%를 먹는 것을 막는다. 연속 배칭은 각 디코드 반복 사이에 요청이 배치에 합류하고 떠나게 하여, 배치가 항상 실제 작업으로 가득 차게 한다. 청크 프리필은 32k 토큰 프롬프트를 디코드와 교차되는 약 512 토큰 슬라이스로 쪼개어, 긴 프롬프트가 GPU의 모든 디코드 토큰을 얼리지 않게 한다.

2026년 프로덕션 기본값은 셋 다 켜는 것이다. 각각이 무엇을 하는지 이해해야 하는데, 실패 모드가 모두 모델이 아니라 스케줄러에 있기 때문이다.

## 개념 (The Concept)

### 가상 메모리 시스템으로서의 PagedAttention

KV 캐시는 시퀀스당 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`다. 8192 토큰에서 Llama 3.3 70B의 경우, 이는 BF16에서 시퀀스당 약 1.25GB다. 모든 요청에 대해 8192 슬롯을 미리 예약하지만 평균 요청이 1500 토큰만 쓴다면, 예약한 HBM의 약 82%를 낭비한다. 고전적 배칭은 이 낭비를 치른다.

PagedAttention은 OS 가상 메모리에서 아이디어를 빌린다. KV 캐시는 시퀀스당 연속적이지 않다. 고정 크기 블록(기본 16 토큰)으로 할당된다. 각 시퀀스는 자신의 논리적 토큰 위치를 물리적 블록 ID로 매핑하는 블록 테이블을 갖는다. 시퀀스가 할당된 블록을 넘어 자라면, 블록 하나가 더 추가된다. 끝나면, 그 블록들은 풀로 되돌아간다.

단편화는 60~80%(고전적)에서 4% 미만(PagedAttention)으로 떨어진다. 플래그로 PagedAttention을 켜는 것이 아니라, vLLM이 출하하는 유일한 할당자다. 손잡이는 `--gpu-memory-utilization`(기본 0.9)으로, 가중치와 활성값을 로드한 후 KV 블록을 위해 얼마나 많은 HBM을 예약할지 vLLM에 알려준다.

### 반복 수준에서의 연속 배칭

옛 "동적 배칭(dynamic batching)"은 배치를 채우기 위해 윈도우(가령 10ms)를 기다린 뒤, 모든 시퀀스가 끝날 때까지 프리필 + 디코드 + 디코드 + 디코드를 돌렸다. 빠른 시퀀스는 일찍 떠나 유휴 상태로 앉아 있었고 GPU는 느린 것들을 마무리했다.

연속 배칭은 각 디코드 스텝 사이에 작동한다. 실행 중인 시퀀스 집합을 `RUNNING` 리스트라 하자. 각 반복에서:

1. 방금 EOS나 max_tokens에 도달한 `RUNNING` 내의 시퀀스는 제거된다.
2. 스케줄러는 대기 큐를 본다. 빈 KV 블록이 있으면, 새 시퀀스(프리필 또는 재개)를 받아들인다.
3. 순방향 패스(forward pass)가 이제 `RUNNING`에 있는 무엇이든 위에서 돌아가며, 시퀀스당 하나의 새 토큰을 방출한다.

배치 크기는 결코 고정된 수로 패딩되지 않는다. 출력의 서로 다른 위치에 있는 시퀀스들이 하나의 융합된(fused) 순방향을 공유한다. 2026년 vLLM에서 이것은 `V1 scheduler`라 불린다. 핵심 불변식: 스케줄러는 요청당 한 번이 아니라 디코드 반복당 한 번 돈다.

### 청크 프리필은 TTFT 꼬리를 보호한다

프리필은 연산 바운드(compute-bound)다. Llama 3.3 70B에서 32k 토큰 프롬프트는 하나의 H100에서 순수 프리필만 약 800ms 걸린다. 프리필이 도는 동안, 배치 내 다른 모든 시퀀스의 디코드 토큰은 기다린다. 서빙 루프에서, 한 긴 프롬프트의 첫 토큰 지연 시간(TTFT)이 수십 명의 다른 사용자에게는 토큰 간 지연 시간(ITL) 끊김이 된다.

청크 프리필은 프리필을 고정 크기 청크(기본 512 토큰)로 나누고 각 청크를 하나의 단위로 스케줄링한다. 청크 사이에 스케줄러는 디코드 시퀀스를 한 토큰 전진시킬 수 있다. 작은 절대 프리필 지연 시간 손실(청크당 몇 ms)을 훨씬 낮은 디코드 시간 지터(jitter)와 맞바꾼다. 혼합 부하에서 P99 ITL은 공개된 벤치마크(benchmark)에서 약 50ms에서 약 15ms로 떨어진다.

### 세 기본값은 상호작용한다

세 기능 모두 서로를 가정한다. PagedAttention은 스케줄러에게 트레이드할 세밀한 KV 자원을 준다. 연속 배칭은 새 시퀀스를 받아들이는 것이 전역 재섞기를 강제하지 않도록 그 세밀한 자원을 필요로 한다. 청크 프리필은 같은 `RUNNING` 리스트에서 스케줄러가 내리는 결정이다. 별도 시스템이 아니라 하나의 추가 스케줄러 정책일 뿐이다.

모든 플래그를 알 필요는 없다. 스케줄러가 무엇을 최적화하는지 알아야 한다: 청크 프리필 슬라이싱에 종속된, KV 블록 예산 하에서의 굿풋(goodput).

### 2026년 v0.18.0 함정

vLLM v0.18.0에서는 `--enable-chunked-prefill`을 드래프트 모델(draft-model) 추측 디코딩(`--speculative-model`)과 결합할 수 없다. 문서화된 예외는 V1 스케줄러에서의 N-gram GPU 추측 디코딩이다. 릴리스 노트를 읽지 않고 모든 플래그를 켜는 팀은 부드러운 회귀(regression)가 아니라 시작 시 런타임 오류를 얻는다. 추측 디코딩 이득이 청크 프리필을 켤 만한 가치가 있었다면, 그 선택을 재고하라. 2026년의 올바른 답은 종종 컴파일되지 않는 드래프트 모델 + 청크 프리필이 아니라 청크 프리필 없는 EAGLE-3다.

### 기억해야 할 숫자

- Llama 3.3 70B FP8, H100 SXM5, 동시 128, 셋 다 켬: 초당 2,200~2,400 토큰.
- 같은 모델, 기본 vLLM(청크 프리필 없음): 약 1,800 토큰/초.
- 같은 모델, 순진한 PyTorch 순방향 루프: 약 600 토큰/초.
- 프로덕션 부하에서 PagedAttention 하의 KV 단편화 낭비: <4%.
- 혼합 부하에서 P99 ITL: 청크 프리필로 약 15ms, 없으면 약 50ms.

### 스케줄러는 어떻게 생겼는가

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py`는 가짜 토큰 수와 가짜 순방향 지연 시간을 가진 stdlib Python에서 정확히 이 루프다. 실행하면 긴 프리필 동안 청크 프리필이 어떻게 디코드 시퀀스를 살려 두는지 보여준다.

```figure
tensor-parallel
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 토글 가능한 기능을 가진 vLLM 스타일 스케줄러를 시뮬레이션한다. 실행하여 다음을 보라:

- `NAIVE` 모드: 한 번에 한 요청, 배칭 없음.
- `STATIC` 모드: 패딩하고 기다림, 고전적 배칭.
- `CONTINUOUS` 모드: 반복 수준의 수용과 해제.
- `CONTINUOUS + CHUNKED` 모드: 디코드와 교차된 프리필 슬라이스.

출력은 총 처리량(가상 초당 토큰), TTFT 평균, P99 ITL을 보여준다. `CONTINUOUS + CHUNKED` 행이 혼합 트래픽에서 지배해야 한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-vllm-scheduler-reader.md`를 만들어낸다. 서빙 설정(배치 크기, KV 메모리 사용률, 청크 프리필 크기, 추측 설정)이 주어지면, 세 기본값 중 어느 것이 병목인지 명명하고 무엇을 튜닝할지 알려주는 스케줄러 진단을 만들어낸다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 짧고 긴 요청이 섞인 워크로드에서 `STATIC`을 `CONTINUOUS`와 비교하라. 처리량 격차는 어디서 오는가: 프리필 효율, 디코드 효율, 아니면 꼬리 지연 시간?
2. 장난감 스케줄러를 수정해 `--max-num-batched-tokens`를 추가하라. Llama 3.3 70B FP8을 돌리는 H100에 올바른 값은 무엇인가? (힌트: 그것은 순수 HBM이 아니라 KV 블록 크기와 빈 블록 수의 함수다.)
3. vLLM v0.18.0 릴리스 노트를 다시 읽어라. 어느 플래그 조합이 상호 배타적인가? 나열하라.
4. 평균 1,500 출력 토큰, 표준편차 600 토큰을 가진 1,000개 요청 트레이스에 대해, (a) 8192 최대에서 요청별 연속 할당, (b) 16 토큰 블록의 PagedAttention 하에서 KV 캐시 단편화 낭비를 계산하라.
5. 청크 프리필이 왜 단독으로는 P99 ITL을 돕지만 처리량은 돕지 않는지 한 문단으로 설명하라. 실제로 처리량 이득은 어디서 오는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| PagedAttention | "KV 트릭" | KV 캐시를 위한 고정 크기 블록 할당자. 단편화 <4% |
| Block table | "페이지 테이블" | 논리적 토큰 위치에서 물리적 KV 블록으로의 시퀀스별 맵 |
| Continuous batching | "동적 배칭, 제대로 한" | 모든 디코드 반복마다 내려지는 수용/해제 결정 |
| Chunked prefill | "프리필 분할" | 긴 프리필을 디코드와 교차되는 512 토큰 슬라이스로 쪼갬 |
| TTFT | "첫 토큰 시간" | 프리필 + 큐 + 네트워크. 긴 프롬프트에서는 프리필이 지배 |
| ITL | "토큰 간 지연 시간" | 연속 디코드 토큰 사이의 시간. 배치 크기가 지배 |
| Goodput | "SLO를 충족하는 처리량" | 모든 요청이 여전히 TTFT와 ITL 목표를 친 초당 토큰 |
| V1 scheduler | "새 스케줄러" | vLLM의 2026년 스케줄러. N-gram 추측 디코드가 청크 프리필 호환 경로 |
| `--gpu-memory-utilization` | "메모리 손잡이" | 가중치와 활성값 후 KV 블록을 위해 예약된 HBM 비율 |

## 더 읽을거리 (Further Reading)

- [vLLM documentation: Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) 청크 프리필과 추측 디코딩 호환성에 관한 공식 출처.
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html): 2026년 릴리스 주기와 버전별 동작.
- [vLLM Blog: PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) 할당자를 어떻게 생각해야 하는지를 여전히 정의하는 원본 글.
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180): 단편화 분석과 스케줄러 설계.
- [Aleksa Gordic: Inside vLLM](https://www.aleksagordic.com/blog/vllm) 플레임 그래프를 곁들인 상세한 V1 스케줄러 안내.
