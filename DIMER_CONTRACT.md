# DIMER fine-tuner contract

This repository owns the TabICLv2 regression fine-tuning worker. DIMER Workbench changes are outside this repository.

## Normal execution

Without `GPU_BURST_MODE`, the container delegates to `train.main()` and uses the existing `DIMER_DATASET_DIR`, `DIMER_OUTPUT_DIR`, `DIMER_RESULT_PATH`, preprocessing, hyperparameter, and callback environment variables.

## On-prem GPU transport compatibility

When `GPU_BURST_MODE=true`, the container entrypoint uses bounded local scratch (default `/dev/shm/dimer`), stages the dataset from `GPU_BURST_S3_BUCKET` and `GPU_BURST_DATASET_PREFIX`, runs the unchanged training core, verifies declared artifacts, and publishes the full output tree. `GPU_BURST_RESULT_KEY` is uploaded last so a visible result means the declared artifact bundle has been persisted. The completion callback runs after publication.

`DIMER_BURST_MAX_SCRATCH_BYTES` bounds the staged dataset plus generated output (default 4 GiB). A publication failure changes the run outcome to failure rather than reporting success for data that exists only in ephemeral scratch.

## TabICL output is a bundle

Serving (bundle format `tabicl-dimer-regressor-v2`) requires at minimum:

- `checkpoints/best.safetensors`
- `training_context.parquet`
- `artifact.json`

The checkpoint remains exposed as `modelArtifact` for current compatibility, but it is not sufficient by itself to reconstruct TabICL inference.

#### Bundle format v1 → v2 (2026-10-24)

- **v2 (current, written by the worker and the tutorial notebook):** `artifact.json` has `artifactFormat`
  `tabicl-dimer-regressor-v2`, and its `checkpoint`, `inference.modelPath` and `payloadFiles` all name
  `checkpoints/best.safetensors`, whose SHA-256 and size are in `digests.checkpointSha256` and `sizes.checkpoint`. The
  file holds the network's `state_dict` as tensors and its `config` as sorted JSON string metadata (`tabicl_config`); it
  is loaded without `torch.load` (`SafetensorsTabICLRegressor`, which builds `TabICL(**config)` and calls
  `load_state_dict(strict=True)`). The worker's `result.json` `modelArtifact` names `best.safetensors` with its `sha256`.
- **Conversion:** tabicl 2.1.1's fine-tuning loop writes `best.ckpt` itself with `torch.save`. Right after training it is
  read once with `torch.load(weights_only=True)`, only `config` and `state_dict` are kept, and the pickle is deleted; only
  the safetensors file is packaged. A `pretrained` bundle carries the pinned safetensors conversion of the base checkpoint.
- **v1 (deprecated, still readable):** `tabicl-dimer-regressor-v1` bundles carry a pickle `checkpoints/best.ckpt`. Every
  consumer in the TabICL repositories still accepts them, loading the checkpoint only with `torch.load(weights_only=True)`
  and emitting a `DeprecationWarning`, until the v1 release artifacts are re-issued. Nothing loads with
  `weights_only=False`.
- **Shared code:** the conversion, the shim and the format checks live in `safetensors_shim.py` of
  `kurtvalcorza/tabicl-regressor-pipeline` (canonical); the worker vendors it byte for byte as `tabicl_safetensors_shim.py`
  and pins its SHA-256 in a test.

## Base-model boundary

The default base is the repository-pinned TabICLv2 regression checkpoint. An explicit mounted `DIMER_BASE_MODEL_PATH` override is honored and recorded by digest/source. The worker does not currently translate `DIMER_MODEL_CONFIG_JSON` into a checkpoint path, so the platform must provide a concrete mounted-path handoff before describing selectable base models as authoritative for this worker.

## DIMER-side follow-ups (documentation only)

The platform should preserve all declared bundle components across export/deployment, consume artifact roles generically rather than relying on a single `best.pt`, provision the CUDA path required by this worker, provide resolved validator preprocessing values during dataset validation, and define a concrete selected-base-model handoff if multiple bases are exposed.

No `dimer-backend` changes are included here.
