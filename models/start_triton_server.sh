#!/usr/bin/bash

MAX_SESSION_SHARE_COUNT=100

tritonserver  \
  --model-repository=/models/Generic \
  --model-repository=/models/AlphaPept \
  --model-repository=/models/Prosit \
  --model-repository=/models/Deeplc \
  --model-repository=/models/ms2pip \
  --allow-grpc=true \
  --grpc-port=8500 \
  --allow-http=true \
  --http-port=8501 \
  --allow-metrics=true \
  --allow-cpu-metrics=true \
  --allow-gpu-metrics=true \
  --metrics-port=8502 \
  --log-info=true \
  --log-warning=true \
  --log-error=true \
  --rate-limit "execution_count" \
  --cuda-memory-pool-byte-size 0:536870912 \