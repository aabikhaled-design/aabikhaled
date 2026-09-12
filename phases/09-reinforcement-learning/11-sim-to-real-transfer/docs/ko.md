# 시뮬레이션-실제 전이 (Sim-to-Real Transfer)

> 시뮬레이터에서 학습되었으나 하드웨어에서 실패하는 정책은 시뮬레이터를 암기한 정책이다. 도메인 무작위화(domain randomization), 도메인 적응(domain adaptation), 시스템 식별(system identification)이 학습된 제어기가 현실 격차(reality gap)를 건너게 만드는 세 가지 도구다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45분

## 문제 (The Problem)

실제 로봇을 학습시키는 것은 느리고, 위험하고, 비싸다. 이족 보행 로봇은 걷는 법을 배우는 데 수백만 학습(training) 에피소드가 걸린다; 한 번이라도 넘어지는 실제 이족 로봇은 하드웨어를 부순다. 시뮬레이션은 무제한 리셋, 결정론적 재현성, 병렬 환경, 그리고 물리적 손상 없음을 제공한다.

하지만 시뮬레이터는 틀렸다. 베어링은 MuJoCo 모델보다 마찰이 더 크다. 카메라에는 시뮬레이터가 포함하지 않는 렌즈 왜곡이 있다. 모터에는 99%의 시뮬레이션 모델이 건너뛰는 지연, 백래시(backlash), 포화(saturation)가 있다. 바람, 먼지, 가변 조명이 멸균된 렌더링에 학습된 정책을 망친다. **현실 격차**, 곧 시뮬레이션 분포와 실제 분포 사이의 체계적 차이는 로보틱스 배포(deployment) RL의 핵심 문제다.

여기서 필요한 것은 *시뮬레이션-실제 분포 이동(distribution shift)에 견고한* 정책이다. 세 가지 역사적 접근이 있다. 시뮬레이터를 무작위화하거나(도메인 무작위화), 약간의 실제 데이터로 정책을 적응시키거나(도메인 적응 / 파인튜닝(fine-tuning)), 실제 시스템의 파라미터(parameter)를 식별해 일치시킨다(시스템 식별). 2026년에 지배적 레시피는 셋 모두를 대규모 병렬 시뮬레이션(Isaac Sim, Isaac Lab, GPU의 Mujoco MJX)과 결합한다.

## 개념 (The Concept)

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**도메인 무작위화(Domain Randomization, DR).** Tobin et al. 2017, Peng et al. 2018. 학습 중에, 실제 로봇에서 다를 수 있는 모든 시뮬레이션 파라미터를 무작위화한다: 질량, 마찰 계수, 모터 PD 게인(gain), 센서 노이즈, 카메라 위치, 조명, 텍스처, 접촉 모델. 정책은 "오늘 어떤 시뮬레이션 안에 있는지"에 대한 조건부 분포를 학습하고 전체 범위에 걸쳐 일반화한다. 실제 로봇이 학습 봉투(training envelope) 안에 들면, 정책이 작동한다.

- **장점:** 실제 데이터가 필요 없다. 하나의 레시피, 많은 로봇.
- **단점:** 과도하게 무작위화된 학습은 "보편적"이지만 지나치게 신중한 정책을 만든다. 너무 많은 노이즈 ≈ 너무 많은 정규화(regularization).

**시스템 식별(System Identification, SI).** 학습 전에 시뮬레이터의 파라미터를 실세계 데이터에 맞춘다. 실제 로봇에서 팔-관절 마찰을 측정할 수 있다면, 그것을 시뮬레이션에 넣는다. 그런 다음 그 값을 기대하는 정책을 학습한다. 실제 시스템에 대한 접근이 필요하지만 현실 격차를 직접 줄인다.

- **장점:** 정밀하고 노이즈가 낮은 학습 목표.
- **단점:** 잔여 모델 오차가 정책에 보이지 않는다; 식별되지 않은 작은 효과(예: 모터 데드밴드)가 여전히 배포를 깨뜨린다.

**도메인 적응(Domain Adaptation).** 시뮬레이션에서 학습하고, 소량의 실제 데이터로 파인튜닝한다. 두 가지 종류:

- **Real2Sim2Real:** 실제 롤아웃(rollout)을 사용해 잔여 시뮬레이터 `f(s, a, z) - f_sim(s, a)`를 학습하고, 보정된 시뮬레이션에서 학습한다. 많은 실제 데이터 없이 격차를 메운다.
- **관측 적응:** 학습된 특성 추출기(예: GAN 픽셀-투-픽셀)를 통해 실제 관측 → 시뮬레이션-유사 관측으로 매핑하는 정책을 학습한다. 제어기는 시뮬레이션에 남는다.

**특권 학습 / 교사-학생(privileged learning / teacher-student).** Miki et al. 2022 (ANYmal 사족 로봇). 특권 정보(ground truth 마찰, 지형 높이, IMU 드리프트)에 접근하는 *교사(teacher)*를 시뮬레이션에서 학습한다. 실제 센서 관측만 보는 *학생(student)*을 증류(distill)한다. 학생은 이력으로부터 특권 특성을 추론하는 법을 배우며, 물리 파라미터에 걸쳐 견고하다.

**대규모 병렬 시뮬레이션.** 2024–2026. Isaac Lab, Mujoco MJX, Brax 모두 단일 GPU에서 수천 개의 병렬 로봇을 실행한다. 4,096개의 병렬 휴머노이드를 가진 PPO는 수 시간 만에 수년치 경험을 수집한다. 학습 분포가 넓어지면서 "현실 격차"가 줄어든다; 그 4,096개 환경 각각이 다른 무작위화 파라미터를 가질 때 DR은 거의 공짜가 된다.

**실세계 2026 레시피(사족 보행 예시):**

1. 중력, 마찰, 모터 게인, 적재량을 도메인 무작위화한 대규모 병렬 시뮬레이션.
2. 특권 정보(지형 지도, 몸체 속도 ground truth)로 학습된 교사 정책.
3. 고유수용감각(proprioception, 다리 관절 인코더)만 사용해 교사로부터 증류된 학생 정책.
4. 실제 IMU에 대한 오토인코더(autoencoder)를 통한 선택적 관측 적응.
5. 배포. 10개 이상의 환경에서 제로샷(zero-shot). 실패하면, 안전 제약 PPO로 수 분간 실세계 파인튜닝을 한다.

```figure
f3-reality-gap
```

## 직접 만들기 (Build It)

이 레슨의 코드는 *노이즈가 있는* 전이를 가진 GridWorld에서 도메인 무작위화의 작은 시연이다. "시뮬레이션"에서 무작위화된 미끄럼 확률을 경험하는 정책을 학습하고, 학습 중에 본 적 없는 미끄럼 수준으로 "실제"에서 평가한다. 그 형태는 MuJoCo-대-하드웨어 전이에 직접 매핑된다.

### 1단계: 파라미터화된 시뮬레이션

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip`은 시뮬레이터가 노출하는 파라미터다. 실제 로보틱스에서는 마찰, 질량, 모터 게인처럼 시뮬레이션과 실제 사이에 이동하는 무엇이든 될 수 있다.

### 2단계: DR로 학습

각 에피소드 시작 시, `slip ~ Uniform[0.0, 0.4]`를 샘플링한다. PPO / Q-러닝(Q-learning) / 무엇이든 학습한다. 많은 에피소드 동안 이렇게 한다.

### 3단계: "실제" 미끄럼에 제로샷 평가

`slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`에서 평가한다. 처음 네 개는 학습 지지(support) 안에 있고; `0.5`와 `0.7`은 밖에 있다. DR로 학습된 정책은 지지 안에서 거의 최적에 머물고 밖에서는 우아하게 저하되어야 한다. 고정-미끄럼으로 학습된 정책은 학습 미끄럼 밖에서 취약할 것이다.

### 4단계: 좁은 학습과 비교

`slip = 0.0`만으로 두 번째 정책을 학습한다. 같은 `slip` 스윕(sweep)에서 평가한다. 실제 미끄럼 > 0이 되자마자 파국적 하락을 보게 될 것이다.

## 함정 (Pitfalls)

- **너무 많은 무작위화.** `slip ∈ [0, 0.9]`로 학습하면 정책이 너무 위험 회피적이어서 최적 경로를 결코 시도하지 않는다. "무엇이든 일어날 수 있다"가 아니라 *기대되는* 실세계 분포에 맞춰라.
- **너무 적은 무작위화.** 얇은 조각으로 학습하면 정책이 전혀 일반화할 수 없다. 정책이 개선되면서 분포를 넓히는 적응적 커리큘럼(Automatic Domain Randomization)을 사용하라.
- **잘못 식별된 파라미터 공간.** 잘못된 것을 무작위화하면(실제 격차가 모터 지연일 때 카메라 색조) DR이 도움이 안 된다. 먼저 실제 로봇을 프로파일링하라.
- **특권 정보 누출.** 관측만이 아니라 전역 상태를 행동에 사용하는 교사는 따라잡을 수 없는 학생을 만들 수 있다. 교사의 정책이 관측 이력이 주어졌을 때 학생에 의해 실현 가능하도록 보장하라.
- **시뮬레이션-시뮬레이션 전이 실패.** 정책이 더 어려운 시뮬레이션 변형에 견고하지 않으면, 실세계에도 견고하지 않을 것이다. 배포 전에 항상 보류된 시뮬레이션 변형에서 테스트하라.
- **실세계 안전 봉투 없음.** 시뮬레이션에서 작동하고 저수준 안전 차폐(safety shield) 없이 "실제에서 작동"하는 정책도 여전히 하드웨어를 부술 수 있다. 비학습 제어기에 속도 제한, 토크 제한, 관절 제한을 추가하라.

## 라이브러리로 써보기 (Use It)

2026년 시뮬레이션-실제 스택:

| 도메인 | 스택 |
|--------|-------|
| 다리 보행 (ANYmal, Spot, 휴머노이드) | Isaac Lab + DR + 특권 교사 / 학생 |
| 조작 (정교한 손, 집기-놓기) | Isaac Lab + DR + 비전용 DR-GAN |
| 자율주행 | CARLA / NVIDIA DRIVE Sim + DR + 실제 파인튜닝 |
| 드론 레이싱 | RotorS / Flightmare + DR + 온라인 적응 |
| 손가락/손 안 조작 | OpenAI Dactyl (전례 없는 규모의 DR) |
| 산업용 팔 | MuJoCo-Warp + SI + 작은 실제 파인튜닝 |

모든 규모의 제어에서, 워크플로는 일관된다: 시뮬레이션을 최대한 맞추고, 맞출 수 없는 것은 무작위화하고, 거대한 정책을 학습하고, 증류하고, 안전 차폐와 함께 배포한다.

## 산출물 (Ship It)

`outputs/skill-sim2real-planner.md`로 저장하라:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 연습 문제 (Exercises)

1. **쉬움.** 고정-미끄럼 GridWorld(slip=0.0)에서 Q-러닝 에이전트(agent)를 학습하라. slip ∈ {0.0, 0.1, 0.3, 0.5}에서 평가하라. slip 대 리턴을 그려라.
2. **중간.** `slip ~ Uniform[0, 0.3]`을 샘플링하는 DR Q-러닝 에이전트를 학습하라. 같은 스윕을 평가하라. slip=0.5(분포 밖)에서 DR이 얼마나 이득을 주는가?
3. **어려움.** 커리큘럼을 구현하라: slip=0.0으로 시작하고, 정책이 최적의 90%에 도달할 때마다 DR 범위를 넓힌다. slip=0.3 제로샷에 도달하기까지의 총 환경 스텝을 고정 DR 베이스라인(baseline)과 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| 현실 격차 | "시뮬레이션-실제 차이" | 학습과 배포 물리/센싱 사이의 분포 이동. |
| 도메인 무작위화 (DR) | "무작위 시뮬레이션에 걸쳐 학습" | 정책이 일반화하도록 학습 중 시뮬레이션 파라미터를 무작위화. |
| 시스템 식별 (SI) | "실제를 측정하고 시뮬레이션을 맞춤" | 실제 물리 파라미터를 추정; 시뮬레이션을 일치하도록 설정. |
| 도메인 적응 | "실제 데이터로 파인튜닝" | 시뮬레이션 학습 후 작은 실세계 파인튜닝; 관측이나 동역학을 적응시킬 수 있음. |
| 특권 정보 | "교사를 위한 ground truth" | 시뮬레이션만 가진 정보; 학생은 관측 이력으로부터 추론해야 함. |
| 교사/학생 | "특권 -> 관측 가능으로 증류" | 지름길로 학습된 교사; 학생은 그것 없이 모방하는 법을 배움. |
| ADR | "자동 도메인 무작위화" | 정책이 개선되면서 DR 범위를 넓히는 커리큘럼. |
| Real2Sim | "실제 데이터로 격차 메우기" | 시뮬레이션이 실제 롤아웃을 모방하도록 잔차를 학습. |

## 더 읽을거리 (Further Reading)

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907): 원조 DR 논문(로보틱스용 비전).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537): 동역학을 위한 DR, 사족 보행.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113): Dactyl, 대규모 ADR.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822): ANYmal을 위한 교사-학생.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470): 2025–2026 배포를 이끄는 대규모 병렬 시뮬레이션.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113): ADR 커리큘럼 방법.
- [Sutton & Barto (2018). Ch. 8: Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) 현대 시뮬레이션-실제 파이프라인을 떠받치는 Dyna 틀(계획 + 롤아웃에 모델 사용).
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303): 벤치마크 결과와 함께하는 시뮬레이션-실제 방법의 분류법.
