# Capstone 07: 종단 간 파인튜닝 파이프라인 (Data to SFT to DPO to Serve)

> 직접 모은 데이터로 학습하고, 직접 정한 선호로 DPO 정렬하고, 양자화(quantize)하고, 추측 디코딩(speculative-decode)하고, 측정 가능한 100만 토큰당 $로 서빙하는 8B 모델. 2026년의 오픈 스택은 Axolotl v0.8, TRL 0.15, 반복(iteration)을 위한 Unsloth, 양자화를 위한 GPTQ/AWQ/GGUF, 서빙을 위한 EAGLE-3를 갖춘 vLLM 0.7이다. 캡스톤(capstone)은 전체 파이프라인을 재현 가능하게 실행하고(YAML 입력, 서빙되는 엔드포인트 출력) 2026 Model Openness Framework 하에서 모델 카드(model card)를 발행하는 것이다.

**Type:** Capstone
**Languages:** Python (pipeline), YAML (configs), Bash (scripts)
**Prerequisites:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 hours

## 문제 (Problem)

2026년의 진지한 모든 AI 팀은 파인튜닝(fine-tuning) 파이프라인을 손닿는 곳에 둔다. 프런티어 베이스 모델을 출시해서가 아니라, 다운스트림 적응(도메인 SFT, 레이블링된 선호에 대한 DPO, 추측 디코딩을 위한 증류된 드래프트(draft), EAGLE-3로의 서빙)이 측정 가능한 성과가 나오는 곳이기 때문이다. Axolotl v0.8은 멀티 GPU SFT 설정을 처리한다. TRL 0.15는 DPO와 GRPO를 처리한다. Unsloth는 빠른 단일 GPU 반복을 가능하게 한다. EAGLE-3를 갖춘 vLLM 0.7은 품질 손실 없이 디코드 처리량(throughput)을 2-3배 끌어올린다. 도구는 작동한다. 기교는 YAML, 데이터 위생, 그리고 평가 규율에 있다.

8B 베이스(Llama 3.3, Qwen3, 또는 Gemma 3)를 작업별 데이터로 SFT한 뒤 DPO를 거치게 하고, 서빙을 위해 양자화하고, lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro에 대해 이득을 측정한다. 2026 Model Openness Framework 하에서 모델 카드를 만든다. 핵심은 재현성이다. 한 명령이 전체 파이프라인을 처음부터 끝까지 재실행한다.

## 개념 (Concept)

파이프라인에는 다섯 단계가 있다. **데이터(Data)**: 중복 제거(MinHash / Datatrove), 품질 필터(Nemotron-CC 스타일 분류기), PII 제거, 공개 벤치마크(benchmark) 오염에 대한 분할 위생 검사. **SFT**: Axolotl YAML, 8xH100에서 ZeRO-3, 코사인 스케줄, 패킹된 시퀀스, 2-3 에폭(epoch). **DPO 또는 GRPO**: TRL 설정, 1 에폭, 사람이 레이블링했거나 모델이 심판한 선호 쌍, 베타 튜닝. **양자화(Quantize)**: 배포 유연성을 위한 GPTQ + AWQ + GGUF. **서빙(Serve)**: EAGLE-3 추측 헤드를 갖춘 vLLM 0.7(또는 SpecForge를 갖춘 SGLang), K8s 배포, 큐 대기(queue-wait)에 대한 HPA.

절제(ablation)가 결과물이다. 세 개의 작업별 벤치마크에서 SFT 단독 대 SFT+DPO 대 SFT+GRPO. 서빙 지표: 배치(batch) 1 / 8 / 32에서 초당 토큰(token), EAGLE-3 수락 비율(acceptance rate), 100만 토큰당 $. 안전성 평가: Llama Guard 4 통과율. 모델 카드: 편향(bias) 평가, 재현성 시드(seed), 데이터 라이선싱.

## 아키텍처 (Architecture)

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 스택 (Stack)

- 데이터: 중복 제거를 위한 Datatrove, 품질을 위한 Nemotron-CC 분류기, PII를 위한 Presidio
- 베이스: Llama 3.3 8B, Qwen3 14B, 또는 Gemma 3 12B
- SFT: ZeRO-3, Flash Attention 3, 패킹된 시퀀스를 갖춘 Axolotl v0.8
- 선호 튜닝: DPO 또는 GRPO를 위한 TRL 0.15; 단일 GPU 반복을 위한 Unsloth
- 양자화: GPTQ(Marlin), AWQ, llama.cpp를 통한 GGUF
- 서빙: EAGLE-3 추측 디코딩을 갖춘 vLLM 0.7(또는 SpecForge를 갖춘 SGLang 0.4)
- 평가: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- 안전성 평가: Llama Guard 4, ShieldGemma-2
- 인프라: Kubernetes + NVIDIA device plugin, 큐 대기 지표에 대한 HPA
- 관측성(observability): 학습을 위한 W&B, 추론(inference)을 위한 Langfuse

```figure
ce-finetune-stages
```

## 직접 만들기 (Build It)

1. **데이터 파이프라인.** 원시 코퍼스(corpus)에 Datatrove 중복 제거를 실행한다. Nemotron-CC 스타일 품질 분류기를 적용한다. Presidio가 PII를 제거한다. 명시적 시드로 train/val 분할을 쓴다.

2. **오염 검사.** 모든 검증 분할에 대해, MMLU-Pro, MT-Bench-v2, RewardBench-2 테스트 셋에 대해 MinHash를 계산한다. 어떤 겹침이든 거부한다.

3. **Axolotl SFT.** ZeRO-3, FA3, 시퀀스 패킹을 갖춘 YAML. 8xH100에서 2-3 에폭. W&B에 로깅한다.

4. **TRL DPO / GRPO.** SFT 체크포인트를 받아, 선호 쌍에 대해 DPO 1 에폭(또는 수학/코드에 검증 가능한 보상(reward)을 갖춘 GRPO)을 실행한다. 베타를 스윕(sweep)한다.

5. **양자화.** 세 가지 양자화를 만든다. GPTQ-INT4-Marlin, AWQ-INT4, llama.cpp를 위한 GGUF-Q4_K_M. 크기와 공칭 처리량을 기록한다.

6. **추측 디코딩으로 서빙.** Red Hat Speculators를 통해 학습된 EAGLE-3 드래프트 헤드를 갖춘 vLLM 0.7 설정. 배치 1 / 8 / 32에서 수락 비율과 꼬리 지연 시간(tail latency)을 측정한다. 같은 평가에서 Anthropic / OpenAI 대비 100만 토큰당 $를 보고한다.

7. **평가 행렬(matrix).** base, SFT 단독, SFT+DPO, SFT+GRPO에 대해 lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro를 실행한다. 표를 만든다.

8. **안전성 평가.** dev 셋에서 Llama Guard 4 통과율. ShieldGemma-2 출력 필터.

9. **모델 카드.** MOF 2026 템플릿: 데이터, 학습, 평가, 안전성, 라이선스, 그리고 YAML과 커밋 SHA가 담긴 재현성 섹션.

## 라이브러리로 써보기 (Use It)

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 산출물 (Ship It)

`outputs/skill-finetuning-pipeline.md`은 결과물을 기술한다. 단일 명령이 데이터를 SFT를 거쳐 DPO를 거쳐 양자화를 거쳐 서빙을 거쳐 평가까지 실행하고, 모델 카드와 서빙되는 엔드포인트를 내놓는다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 베이스 대비 평가 차이 | 목표 작업(MMLU-Pro, MT-Bench-v2, 작업별)에서의 측정된 이득 |
| 20 | 파이프라인 재현성 | 한 명령이 동일한 시드로 처음부터 끝까지 재실행 |
| 20 | 데이터 위생 | 중복 제거 비율, PII 제거 커버리지, 오염 검사 통과 |
| 20 | 서빙 효율성 | bs=1/8/32에서 초당 토큰, EAGLE-3 수락 비율, 100만 토큰당 $ |
| 15 | 모델 카드 + 안전성 평가 | 2026 MOF 완전성 + Llama Guard 4 통과율 |
| **100** | | |

## 연습 문제 (Exercises)

1. 같은 작업별 벤치마크에서 SFT 단독 대 SFT+DPO 대 SFT+GRPO를 실행한다. 어떤 선호 방법이 이기고 얼마나 차이가 나는지 보고한다.

2. Llama 3.3 8B를 Qwen3 14B로 교체한다. 맞춰진 품질에서 100만 토큰당 $를 측정한다.

3. 도메인 데이터 대 일반 ShareGPT에서 EAGLE-3 수락 비율을 측정한다. 그 차이와 그것이 지연 시간 예산에 의미하는 바를 보고한다.

4. 오염 1%를 주입(MMLU-Pro 답을 학습 데이터에 유출)하고 평가를 재실행한다. MMLU-Pro 정확도가 비현실적으로 뛰는 것을 관찰한다. 이것을 잡아내는 오염 검사 CI 게이트를 만든다.

5. 전체 파인튜닝의 대안으로 LoRA SFT를 추가한다. 10배 낮은 메모리에서 품질 격차를 측정한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| Axolotl | "SFT 트레이너" | SFT, DPO, 증류를 위한 통합 YAML 기반 트레이너 |
| TRL | "선호 튜너" | LLM에 대한 DPO, GRPO, PPO를 위한 Hugging Face 라이브러리 |
| GRPO | "그룹 상대 정책 최적화" | 검증 가능한 보상을 갖춘 DeepSeek R1의 RL 레시피 |
| EAGLE-3 | "추측 디코딩 드래프트" | N개 토큰을 앞서 예측하는 드래프트 헤드; vLLM이 타깃 모델로 검증 |
| MOF | "Model Openness Framework" | 데이터, 코드, 라이선스에 대해 모델 출시를 등급 매기는 2026 표준 |
| 오염 검사(Contamination check) | "분할 위생" | 학습으로의 테스트셋 유출을 MinHash 기반으로 탐지 |
| 수락 비율(Acceptance rate) | "EAGLE / MTP 지표" | 타깃 모델이 수락하는 드래프트된 토큰의 비율 |

## 더 읽을거리 (Further Reading)

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/): 레퍼런스 SFT / DPO 트레이너
- [TRL documentation](https://huggingface.co/docs/trl): DPO 및 GRPO 레퍼런스 구현
- [Unsloth](https://github.com/unslothai/unsloth): 단일 GPU 반복 레퍼런스
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948): GRPO 방법론
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai): 레퍼런스 서빙 스택
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge): 대안 추측 디코딩 트레이너
- [Model Openness Framework 2026](https://isocpp.org/): 오픈 출시 등급 표준
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): 정전(canonical)에 해당하는 평가 러너
