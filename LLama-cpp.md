# Introduction:
- Opensource c++ Library for running LLM Locally on your hardware.
- Run LLM Inference without needing cloud API inference Engine.
- It is like vLLM but not same as vLLM.
- Ollama is using llama.cpp in backend.

Ollama = llama.cpp + node RESET API Wrapper + Model download management + Simple CLI + Modelfiles <br>

vLLM = Own Enginer ( Paged Attention )


# Kimit k2.6 Model Deployment With Llama

## Nvidia H200 GPU

Dockerfile
```
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/ggml-org/llama.cpp.git

WORKDIR /app/llama.cpp

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda"

RUN cmake --build build --target llama-server -j 8

EXPOSE 8080

ENTRYPOINT ["./build/bin/llama-server"]

```

docker-compose.yaml
```
services:
  kimi:
    build: .

    container_name: kimi-server

    restart: unless-stopped

    security_opt:
      - seccomp=unconfined

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    ports:
      - "8080:8080"

    volumes:
      - /Data-2/models/Kimi-k2.6-GUFF/BF16:/models
    ulimits:
      memlock: -1
      stack: 67108864

    command: >
      -m /models/Kimi-K2.6-BF16-00001-of-00046.gguf
      --host 0.0.0.0
      --port 8080
      -c 2048
      -t 64
      -np 1
#      -m /models/Kimi-K2.6-BF16-00001-of-00046.gguf
#      --host 0.0.0.0
#      --port 8080
#      -ngl 999
#      -c 16384
#      -t 32
#      --parallel 8
#      --mlock
#      --tensor-split 20,20,20,20,20,20,0,0

```

## AMD MI3000 GPU
Dockerfile
```
FROM rocm/dev-ubuntu-22.04:6.4

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    hipblas-dev \
    rocblas-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/ggml-org/llama.cpp.git

WORKDIR /app/llama.cpp

#RUN git checkout b3407

#RUN cmake -B build \
#    -DGGML_HIP=ON \
#    -DAMDGPU_TARGETS=gfx942 \
#    -DCMAKE_BUILD_TYPE=Release
RUN cmake -B build \
    -DGGML_HIP=ON \
    -DCMAKE_C_COMPILER=hipcc \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DAMDGPU_TARGETS=gfx942 \
    -DCMAKE_BUILD_TYPE=Release

RUN cmake --build build -j 4
#RUN cmake --build build --target llama-server --verbose -j 4

EXPOSE 8080

ENTRYPOINT ["./build/bin/llama-server"]

```

docker-compose.yaml
```
services:
  kimi:
    build: .

    container_name: kimi-server

    restart: unless-stopped

    devices:
      - /dev/kfd
      - /dev/dri

    group_add:
      - video
      - render

    security_opt:
      - seccomp=unconfined

    environment:
      - HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      - HSA_OVERRIDE_GFX_VERSION=11.0.0

    ports:
      - "8080:8080"

    volumes:
      - /Data/models/Kimi-k2.6-GUFF/BF16:/models
    ulimits:
      memlock: -1
      stack: 67108864

    command: >
      -m /models/Kimi-K2.6-BF16-00001-of-00046.gguf
      --host 0.0.0.0
      --port 8080
      -ngl 999
      -c 16384
      -t 32
      --parallel 8
      --mlock
      --tensor-split 20,20,20,20,20,20,0,0
```

### Issue on both servers:
```
kimi-server  | 0.01.288.293 I device_info:
kimi-server  | 0.01.543.501 I   - CUDA0   : NVIDIA H200 (143166 MiB, 110773 MiB free)
kimi-server  | 0.02.086.769 I   - CUDA1   : NVIDIA H200 (143166 MiB, 141115 MiB free)
kimi-server  | 0.02.419.331 I   - CUDA2   : NVIDIA H200 (143166 MiB, 141181 MiB free)
kimi-server  | 0.02.688.877 I   - CUDA3   : NVIDIA H200 (143166 MiB, 142203 MiB free)
kimi-server  | 0.03.093.934 I   - CUDA4   : NVIDIA H200 (143166 MiB, 142637 MiB free)
kimi-server  | 0.03.274.253 I   - CUDA5   : NVIDIA H200 (143166 MiB, 142637 MiB free)
kimi-server  | 0.03.457.917 I   - CUDA6   : NVIDIA H200 (143166 MiB, 111227 MiB free)
kimi-server  | 0.03.660.713 I   - CUDA7   : NVIDIA H200 (143166 MiB, 111227 MiB free)
kimi-server  | 0.03.660.747 I   - CPU     : Intel(R) Xeon(R) Platinum 8480+ (2063699 MiB, 2063699 MiB free)
kimi-server  | 0.03.660.922 I system_info: n_threads = 64 (n_threads_batch = 64) / 224 | CUDA : ARCHS = 900 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | AMX_INT8 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | 
kimi-server  | 0.03.661.014 I srv          init: using 223 threads for HTTP server
kimi-server  | 0.03.661.152 I srv         start: binding port with default address family
kimi-server  | 0.03.662.472 I srv  llama_server: loading model
kimi-server  | 0.03.662.476 I srv    load_model: loading model '/models/Kimi-K2.6-BF16-00001-of-00046.gguf'
kimi-server  | 0.03.662.552 I common_init_result: fitting params to device memory ...
kimi-server  | 0.03.662.554 I common_init_result: (for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)
```
