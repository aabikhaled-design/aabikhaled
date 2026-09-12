# GPU 설정과 클라우드 (GPU Setup & Cloud)

> 학습 목적이라면 CPU에서 훈련해도 괜찮다. 하지만 제대로 훈련하려면 GPU가 필요하다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45분

## 학습 목표 (Learning Objectives)

- `nvidia-smi`와 PyTorch의 CUDA API로 로컬 GPU 가용성 검증하기
- 무료 클라우드 기반 실험을 위해 T4 GPU가 달린 Google Colab 구성하기
- CPU와 GPU에서 행렬 곱셈(matrix multiplication)을 벤치마크(benchmark)해 속도 향상 측정하기
- fp16 어림 규칙으로 VRAM에 들어가는 가장 큰 모델 크기 추정하기

## 문제 (The Problem)

phase 1~3의 대부분 레슨은 CPU에서 문제없이 돌아간다. 하지만 CNN, 트랜스포머(transformer), LLM(phase 4 이후)을 훈련하기 시작하면 GPU 가속이 필요하다. CPU에서 8시간 걸리는 훈련이 GPU에서는 10분이면 끝난다.

선택지는 세 가지다. 로컬 GPU, 클라우드 GPU, 또는 (무료인) Google Colab.

## 개념 (The Concept)

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## 직접 만들기 (Build It)

### 옵션 1: 로컬 NVIDIA GPU

GPU가 있는지 확인한다.

```bash
nvidia-smi
```

CUDA가 포함된 PyTorch를 설치한다.

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 옵션 2: Google Colab

1. [colab.research.google.com](https://colab.research.google.com)으로 이동한다
2. Runtime > Change runtime type > T4 GPU
3. `!nvidia-smi`를 실행해 검증한다

이 강의의 노트북을 Colab에 바로 업로드할 수 있다.

### 옵션 3: 클라우드 GPU

Lambda Labs, RunPod, Vast.ai의 경우:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### GPU가 없다고? 괜찮다.

대부분의 레슨은 CPU에서 돌아간다. GPU가 필요한 레슨은 그렇다고 명시하고 Colab 링크를 함께 넣는다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 직접 만들기: GPU vs CPU 벤치마크

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 연습 문제 (Exercises)

1. 위 벤치마크를 실행하고 CPU와 GPU 시간을 비교하라
2. GPU가 없다면 Google Colab에서 실행하고 비교하라
3. GPU 메모리가 얼마나 있는지 확인하고 들어갈 수 있는 가장 큰 모델을 추정하라(어림 규칙: fp16은 파라미터(parameter)당 2바이트)

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| CUDA | "GPU 프로그래밍" | GPU에서 코드를 실행하게 해 주는 NVIDIA의 병렬 컴퓨팅 플랫폼 |
| VRAM | "GPU 메모리" | GPU에 있는 비디오 RAM. 시스템 RAM과 별개이며 모델 크기를 제한한다. |
| fp16 | "반정밀도(half precision)" | 16비트 부동소수점. fp32 대비 절반의 메모리를 쓰면서 정확도 손실은 최소화한다 |
| 텐서 코어(Tensor Core) | "빠른 행렬 하드웨어" | 행렬 곱셈에 특화된 GPU 코어. 일반 코어보다 4~8배 빠르다 |
