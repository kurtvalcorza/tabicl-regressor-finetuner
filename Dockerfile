# CUDA 12.8 / torch 2.8 base: ships sm_120 (Blackwell) kernels plus sm_70-sm_100.
# Do not downgrade to cuda12.4/torch2.6 — that build lacks sm_120 and dies with
# "no kernel image is available" on RTX 50-series / B200 GPUs.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Bake the base checkpoint at a pinned revision so fine-tuning is reproducible
# and never silently pulls a moved "main" (external-review #8).
ARG BASE_MODEL=tabicl-regressor-v2-20260212.ckpt
ARG BASE_MODEL_REVISION=4dcd344ece2c00be9e831fdd35bed57b5ad83e19
RUN python -c "import shutil; from huggingface_hub import hf_hub_download; shutil.copyfile(hf_hub_download('jingang/TabICL', '${BASE_MODEL}', revision='${BASE_MODEL_REVISION}'), '/app/${BASE_MODEL}')"
ENV DIMER_TASK_TYPE=tabular_regression \
    DIMER_BASE_MODEL_PATH=/app/tabicl-regressor-v2-20260212.ckpt \
    DIMER_BASE_MODEL_REVISION=4dcd344ece2c00be9e831fdd35bed57b5ad83e19
COPY train.py ./
CMD ["python", "train.py"]
