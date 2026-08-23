# tabicl-regressor-finetuner

DIMER fine-tuner for the TabICLv2 regressor pipeline. It fine-tunes
[`jingang/TabICL`](https://huggingface.co/jingang/TabICL) (regressor checkpoint) through
`tabicl.FinetunedTabICLRegressor` on a validated tabular-regression dataset, then writes the
fine-tuner artifacts and a `result.json` with metrics and provenance (see [Outputs](#outputs)).
See the [model card](https://github.com/kurtvalcorza/tabicl-regressor-pipeline/blob/main/MODEL_CARD.md)
for the model's provenance, checksums, and licence.

- Runs as a **CUDA GPU** Kubernetes Job — GPU-only, no zero-shot fallback (see below).
- DIMER builds the root `Dockerfile` into an image and runs `train.py`.
- `dimer-pipeline.json` at the repo root defines the workbench preprocessing and fine-tuning
  fields. DIMER reads it from the repo root to render them; it is **not** copied into the image,
  and manifest↔runtime parity is enforced by the test suite (CI), not by the build.
- Pairs with `tabicl-regressor-dataset-validator`.

## Fine-tuning is GPU-only

This component is a **fine-tuner**, not a zero-shot server. `FinetunedTabICLRegressor` requires
CUDA; if no GPU is available the run fails with a clear message rather than silently downgrading a
fine-tuning request into zero-shot inference. `DIMER_TRAIN_DEVICE` is honored (a bare index like
`0` is normalized to `cuda:0`); any non-CUDA request is rejected. Because TabICL stays an
in-context learner after fine-tuning, the served artifact carries **both** the fine-tuned
checkpoint and the training context — see [Outputs](#outputs). Predictions are never clipped and
negative targets are valid.

Other key behavior:

- pins `tabicl[finetune]==2.1.1`; base image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
  (ships sm_120/Blackwell kernels — do not downgrade to cuda12.4/torch2.6).
- base checkpoint `tabicl-regressor-v2-20260212.ckpt`, pinned by HF revision + SHA-256 and
  verified at fit time; a DIMER-provided base overrides it.
- excludes non-finite targets, then draws a validation holdout when `val.csv` is absent and caps
  training to `max_train_rows`.
- best-checkpoint selection on `eval_metric` (`mae` / `mse` / `r2`); reports MAE, MSE, RMSE, R².
- scores `test.csv` when present, after selection (never used for selection).
- reloads the fine-tuned checkpoint from `artifact.json` + the context table and predicts before
  reporting success.

## Outputs

`train.py` materializes the DIMER fine-tuner artifact layout under
`DIMER_OUTPUT_DIR/tabicl_regressor/`:

- `checkpoints/best.ckpt` — the fine-tuned TabICLv2 checkpoint (the `modelArtifact` DIMER exports).
- `training_context.parquet` — the in-context support table; the **second file** required to serve
  the model.
- `artifact.json` — the self-contained inference contract (schema, inference block, categorical
  encoders, digests).
- `evaluation/report.json` — validation / optional test metrics.
- `logs/run-summary.json` — the run summary.
- `progress/epoch_*.json` — best-effort per-epoch telemetry (written **post-fit, not live**; TabICL
  exposes no per-epoch hook).

`result.json` carries the metrics and `provenance`, plus an `artifacts` object with `modelArtifact`
(`best.ckpt`), `trainingContext` (`training_context.parquet`), `manifest`, `evaluationReport`, and
`logArtifact` entries — each `{path, name, contentType, sizeBytes}`, `path` relative to the `/data`
mount (`fine-tuning/<run_id>/...`), which is how the backend resolves it
(`workbench.domain._resolve_result_artifact`). `modelArtifact` is **mandatory** — without it
export-to-repository and model-download report *"No model artifact found"*. Unlike a single-file
model, TabICL needs **both** `modelArtifact` and `trainingContext` at serve time.

## DIMER Pipeline Builder integration

The repository builds directly with the **repository root as the Docker build context**; the
Dockerfile invokes root-level `train.py`. Root layout: `Dockerfile`, `train.py`, `requirements.txt`,
`README.md`, `dimer-pipeline.json`.

- Every `datasetPreprocessing` key is read from `DIMER_PREPROCESSING_ARGS_JSON` and every
  `modelFinetuning` key from `DIMER_HYPERPARAMETERS_JSON`, 1:1 — no manifest control is silently
  ignored (guarded by `tests/test_train.py::test_manifest_matches_env_consumption`).
- `model_id` is intentionally **not** a manifest parameter; the DIMER Base Model selection is
  authoritative.
- The image bakes `DIMER_TASK_TYPE=tabular_regression` as the fallback for DIMER Custom / Other
  pipelines.
