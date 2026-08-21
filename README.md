# tabicl-regressor-finetuner

DIMER GPU fine-tuner for TabICLv2 regression using `tabicl.FinetunedTabICLRegressor`.

The artifact contains the fine-tuned checkpoint plus the training/support context required for TabICL inference. Validation and optional test metrics are written to `result.json` with provenance hashes.

Pairs with `tabicl-regressor-dataset-validator`. Full documentation lives in `tabicl-regressor-pipeline`.

## DIMER Pipeline Builder integration

The repository builds directly with the **repository root as the Docker build
context**; the Dockerfile invokes root-level `train.py`. Root layout:
`Dockerfile`, `train.py`, `requirements.txt`, `README.md`, `dimer-pipeline.json`.

- Every `datasetPreprocessing` key is read from `DIMER_PREPROCESSING_ARGS_JSON`
  and every `modelFinetuning` key from `DIMER_HYPERPARAMETERS_JSON`, 1:1 — no
  manifest control is silently ignored (guarded by
  `tests/test_train.py::test_manifest_matches_env_consumption`).
- `model_id` is intentionally **not** a manifest parameter; the DIMER Base Model
  selection is authoritative.
- The image bakes `DIMER_TASK_TYPE=tabular_regression` as the fallback for
  DIMER Custom / Other pipelines.
