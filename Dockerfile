# CUDA 12.8 / torch 2.8 base: ships sm_120 (Blackwell) kernels plus sm_70-sm_100.
# Do not downgrade to cuda12.4/torch2.6 — that build lacks sm_120 and dies with
# "no kernel image is available" on RTX 50-series / B200 GPUs.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV DIMER_TASK_TYPE=tabular_regression
COPY train.py ./
CMD ["python", "train.py"]
