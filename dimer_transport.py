"""Transport adapter for DIMER GPU-burst fine-tuning jobs.

Normal executions keep the existing /data contract.  GPU_BURST_MODE is the
current on-prem compatibility path where DIMER provides S3/MinIO coordinates
but no /data mount.  In that mode we stage the dataset into bounded scratch and
publish the complete TabICL model bundle before result.json becomes visible.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_SCRATCH_ROOT = "/dev/shm/dimer"
DEFAULT_MAX_SCRATCH_BYTES = 4 << 30


def burst_mode_enabled() -> bool:
    return os.getenv("GPU_BURST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    value = default if not raw else int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _safe_relative_key(prefix: str, key: str) -> Path | None:
    normalized_prefix = prefix.rstrip("/") + "/"
    if not key.startswith(normalized_prefix):
        raise ValueError(f"S3 object {key!r} is outside configured prefix {normalized_prefix!r}")
    relative = key[len(normalized_prefix):]
    if not relative or relative.endswith("/"):
        return None
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe GPU-burst dataset object key: {key!r}")
    return Path(*posix.parts)


def _s3_client():
    import boto3

    kwargs: dict[str, Any] = {}
    endpoint = os.getenv("S3_ENDPOINT_URL", "").strip()
    region = os.getenv("GPU_BURST_S3_REGION", "").strip() or os.getenv("AWS_DEFAULT_REGION", "").strip()
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def _scratch_limit() -> int:
    return _positive_int_env("DIMER_BURST_MAX_SCRATCH_BYTES", DEFAULT_MAX_SCRATCH_BYTES)


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def stage_burst_dataset(dataset_dir: Path) -> dict[str, Any]:
    bucket = os.getenv("GPU_BURST_S3_BUCKET", "").strip()
    prefix = os.getenv("GPU_BURST_DATASET_PREFIX", "").strip()
    if not bucket or not prefix:
        raise ValueError("GPU burst mode requires GPU_BURST_S3_BUCKET and GPU_BURST_DATASET_PREFIX")

    client = _s3_client()
    objects: list[tuple[str, Path, int]] = []
    total_bytes = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            relative = _safe_relative_key(prefix, key)
            if relative is None:
                continue
            size = int(item.get("Size") or 0)
            if size < 0:
                raise ValueError(f"S3 object {key!r} reported an invalid size {size}")
            total_bytes += size
            if total_bytes > _scratch_limit():
                raise ValueError(
                    f"GPU-burst dataset requires {total_bytes:,} bytes, exceeding "
                    f"DIMER_BURST_MAX_SCRATCH_BYTES={_scratch_limit():,}"
                )
            objects.append((key, relative, size))

    if not objects:
        raise FileNotFoundError(f"GPU-burst dataset prefix s3://{bucket}/{prefix} contains no files")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for key, relative, _size in objects:
        destination = dataset_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(destination))

    return {"fileCount": len(objects), "totalBytes": total_bytes}


def prepare_burst_runtime() -> dict[str, Any] | None:
    """Configure scratch paths and stage the dataset before importing train.py."""
    if not burst_mode_enabled():
        return None

    run_id = os.getenv("DIMER_RUN_ID", "").strip()
    if not run_id:
        raise ValueError("GPU burst mode requires DIMER_RUN_ID")
    scratch_root = Path(os.getenv("DIMER_BURST_SCRATCH_ROOT", DEFAULT_SCRATCH_ROOT))
    dataset_dir = scratch_root / "dataset"
    output_dir = scratch_root / "fine-tuning" / run_id
    result_path = output_dir / "result.json"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # train.py resolves these paths at import time, so set them before importing it.
    os.environ["DIMER_DATASET_DIR"] = str(dataset_dir)
    os.environ["DIMER_OUTPUT_DIR"] = str(output_dir)
    os.environ["DIMER_RESULT_PATH"] = str(result_path)

    staged = stage_burst_dataset(dataset_dir)
    return {
        "scratchRoot": scratch_root,
        "datasetDir": dataset_dir,
        "outputDir": output_dir,
        "resultPath": result_path,
        **staged,
    }


def _result_target() -> tuple[str, str]:
    bucket = os.getenv("GPU_BURST_S3_BUCKET", "").strip()
    result_key = os.getenv("GPU_BURST_RESULT_KEY", "").strip()
    if not bucket or not result_key:
        raise ValueError("GPU burst mode requires GPU_BURST_S3_BUCKET and GPU_BURST_RESULT_KEY")
    return bucket, result_key


def _validated_artifacts(payload: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    root = output_dir.parent.parent
    resolved: dict[str, Path] = {}
    for role, descriptor in artifacts.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"artifact {role!r} must be an object")
        raw_path = str(descriptor.get("path") or "")
        posix = PurePosixPath(raw_path)
        if not raw_path or posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"artifact {role!r} has unsafe path {raw_path!r}")
        local_path = root.joinpath(*posix.parts)
        if not local_path.is_file():
            raise FileNotFoundError(f"declared artifact {role!r} does not exist: {local_path}")
        expected_size = descriptor.get("sizeBytes")
        if expected_size is not None and int(expected_size) != local_path.stat().st_size:
            raise ValueError(
                f"declared artifact {role!r} size mismatch: "
                f"result={expected_size}, file={local_path.stat().st_size}"
            )
        resolved[role] = local_path
    return resolved


def publish_result_only(result_path: Path) -> dict[str, Any]:
    bucket, result_key = _result_target()
    if not result_path.is_file():
        raise FileNotFoundError(f"result.json does not exist: {result_path}")
    _s3_client().upload_file(str(result_path), bucket, result_key)
    return {"bucket": bucket, "resultKey": result_key}


def publish_success_bundle(
    *,
    payload: dict[str, Any],
    dataset_dir: Path,
    output_dir: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Publish the complete output tree, then result.json as the commit marker."""
    bucket, result_key = _result_target()
    resolved_artifacts = _validated_artifacts(payload, output_dir)
    used_bytes = _tree_bytes(dataset_dir) + _tree_bytes(output_dir)
    if used_bytes > _scratch_limit():
        raise ValueError(
            f"GPU-burst scratch usage {used_bytes:,} bytes exceeds "
            f"DIMER_BURST_MAX_SCRATCH_BYTES={_scratch_limit():,}"
        )

    client = _s3_client()
    result_parent = PurePosixPath(result_key).parent.as_posix().rstrip("/")
    uploaded: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == result_path:
            continue
        relative = path.relative_to(output_dir).as_posix()
        key = f"{result_parent}/{relative}"
        client.upload_file(str(path), bucket, key)
        uploaded.append(key)

    # Compatibility alias for DIMER code that still exposes a single model key.
    # The canonical deployable TabICL artifact remains the declared multi-file bundle.
    legacy_model_key = os.getenv("GPU_BURST_MODEL_KEY", "").strip()
    model_path = resolved_artifacts.get("modelArtifact")
    if legacy_model_key and model_path is not None:
        natural_key = f"{result_parent}/{model_path.relative_to(output_dir).as_posix()}"
        if legacy_model_key != natural_key:
            client.upload_file(str(model_path), bucket, legacy_model_key)
            uploaded.append(legacy_model_key)

    # result.json is uploaded last so its presence means the declared bundle is durable.
    client.upload_file(str(result_path), bucket, result_key)
    uploaded.append(result_key)
    return {"bucket": bucket, "resultKey": result_key, "uploaded": uploaded}
