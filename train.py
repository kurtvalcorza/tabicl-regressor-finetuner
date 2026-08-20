"""DIMER fine-tuner for TabICLv2 regression."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TEMPLATE_NAME = "tabicl-regressor-finetuner"
BASE_MODEL = "tabicl-regressor-v2-20260212.ckpt"
BASE_MODEL_REPO = "jingang/TabICL"
# Pin the base checkpoint by HF revision + SHA-256 so fine-tuning never silently
# picks up a moved "main". Overridable for a future base-model bump.
BASE_MODEL_REVISION = os.getenv("DIMER_BASE_MODEL_REVISION", "4dcd344ece2c00be9e831fdd35bed57b5ad83e19")
BASE_MODEL_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
TABICL_VERSION = "2.1.1"
DATASET_DIR = Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset"))
OUTPUT_DIR = Path(os.getenv("DIMER_OUTPUT_DIR", "/data/output"))
RESULT_PATH = Path(os.getenv("DIMER_RESULT_PATH", "/data/results/result.json"))
DONE_CALLBACK = os.getenv("DIMER_DONE_CALLBACK", "").strip()
MAX_FEATURES = 2_000
MIN_TRAIN_ROWS = 50

# Numeric/limit configuration. Module-level DEFAULTS (plain literals so import
# never fails); values used are (re)loaded from the environment inside the
# protected run() via _load_limits(), so a malformed platform value produces a
# structured failure result.json instead of an import-time crash.
CALLBACK_TIMEOUT_SECONDS = 10.0
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1 << 30
MAX_SINGLE_CSV_BYTES = 512 << 20
PREDICT_BATCH_ROWS = 8192  # bound single predict() calls to avoid avoidable OOM


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else float(raw)


def _load_limits() -> None:
    """Re-read numeric limits from the environment inside the protected path."""
    global CALLBACK_TIMEOUT_SECONDS, MAX_ARCHIVE_UNCOMPRESSED_BYTES, MAX_SINGLE_CSV_BYTES, PREDICT_BATCH_ROWS
    CALLBACK_TIMEOUT_SECONDS = _float_env("DIMER_CALLBACK_TIMEOUT_SECONDS", 10.0)
    MAX_ARCHIVE_UNCOMPRESSED_BYTES = _int_env("DIMER_MAX_ARCHIVE_UNCOMPRESSED_BYTES", 1 << 30)
    MAX_SINGLE_CSV_BYTES = _int_env("DIMER_MAX_SINGLE_CSV_BYTES", 512 << 20)
    PREDICT_BATCH_ROWS = max(1, _int_env("DIMER_PREDICT_BATCH_ROWS", 8192))


def log(message: str) -> None:
    print(f"[{TEMPLATE_NAME}] {message}", flush=True)


def _json_env(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def notify_done_callback() -> dict[str, Any]:
    if not DONE_CALLBACK:
        return {"attempted": False}
    parsed = urlparse(DONE_CALLBACK)
    if parsed.scheme not in {"http", "https"}:
        return {"attempted": False, "error": f"unsupported callback scheme {parsed.scheme!r}"}
    try:
        response = requests.post(DONE_CALLBACK, timeout=CALLBACK_TIMEOUT_SECONDS)
        return {"attempted": True, "ok": response.ok, "statusCode": response.status_code}
    except requests.RequestException as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def _normalize_member(name: str) -> str | None:
    if not name or name.endswith("/"):
        return None
    normalized = name.replace("\\", "/")
    parts = Path(normalized).parts
    # Reject absolute paths and parent-directory traversal BEFORE stripping
    # anything (Path() has already collapsed any leading "./").
    if normalized.startswith("/") or ".." in parts:
        raise ValueError(f"unsafe archive member: {name}")
    if len(parts) > 1 and parts[0].lower() in {"dataset", "datasets"}:
        parts = parts[1:]
    return "/".join(parts) if parts else None


def _zip_and_member(stem: str) -> tuple[zipfile.ZipFile | None, Any | None]:
    zips = sorted(DATASET_DIR.glob("*.zip"))
    if len(zips) > 1:
        raise ValueError(f"multiple dataset zip files found: {[p.name for p in zips]}")
    if not zips:
        matches = sorted(p for p in DATASET_DIR.rglob("*.csv") if p.stem.lower() == stem.lower())
        if len(matches) > 1:
            raise ValueError(f"multiple {stem}.csv candidates found: {[str(p) for p in matches]}")
        if not matches:
            return None, None
        if matches[0].stat().st_size > MAX_SINGLE_CSV_BYTES:
            raise ValueError(f"{matches[0].name} exceeds the configured CSV size limit")
        return None, matches[0]

    zf = zipfile.ZipFile(zips[0])
    total = sum(int(info.file_size) for info in zf.infolist())
    if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        zf.close()
        raise ValueError("dataset archive exceeds the configured uncompressed-size limit")
    matches = []
    for info in zf.infolist():
        logical = _normalize_member(info.filename)
        if logical and Path(logical).suffix.lower() == ".csv" and Path(logical).stem.lower() == stem.lower():
            matches.append(info)
    if len(matches) > 1:
        names = [info.filename for info in matches]
        zf.close()
        raise ValueError(f"multiple {stem}.csv candidates found: {names}")
    if not matches:
        zf.close()
        return None, None
    if matches[0].file_size > MAX_SINGLE_CSV_BYTES:
        zf.close()
        raise ValueError(f"{matches[0].filename} exceeds the configured CSV size limit")
    return zf, matches[0]


def _read_csv(stem: str) -> pd.DataFrame | None:
    zf, source = _zip_and_member(stem)
    if source is None:
        return None
    try:
        if zf is not None:
            with zf.open(source) as handle:
                return pd.read_csv(handle)
        return pd.read_csv(source)
    finally:
        if zf is not None:
            zf.close()


def _fit_categorical_encoder(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, list[str]]:
    """Ordinal maps for non-numeric feature columns.

    FinetunedTabICLRegressor fails on raw string/categorical features
    (upstream soda-inria/tabicl#118), so object/string/category/bool feature
    columns are ordinal-encoded before fine-tuning. The returned map is
    {column: [category, ...]}; a value's code is its index in that list, and
    any missing or unseen value maps to len(categories) — a dedicated unknown
    bucket. Numeric (non-bool) columns are passed through untouched.
    """
    encoders: dict[str, list[str]] = {}
    for col in feature_columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            continue
        encoders[col] = sorted({str(v) for v in series.dropna().unique()})
    return encoders


def _apply_categorical_encoder(frame: pd.DataFrame, encoders: dict[str, list[str]]) -> pd.DataFrame:
    """Apply persisted ordinal maps; missing/unseen values go to the unknown bucket."""
    if not encoders:
        return frame
    out = frame.copy()
    for col, categories in encoders.items():
        if col not in out.columns:
            continue
        index = {cat: code for code, cat in enumerate(categories)}
        unknown = len(categories)
        out[col] = pd.Series(
            [unknown if pd.isna(v) else index.get(str(v), unknown) for v in frame[col]],
            index=frame.index,
            dtype="int64",
        )
    return out


def _clean_frame(frame: pd.DataFrame, target: str, drop_columns: list[str]) -> pd.DataFrame:
    if target not in frame.columns:
        raise KeyError(f"target column {target!r} not found")
    out = frame.drop(columns=[c for c in drop_columns if c in frame.columns]).copy()
    numeric = pd.to_numeric(out[target], errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    out = out.loc[finite].copy()
    out[target] = numeric.loc[finite].astype(float)
    return out.reset_index(drop=True)


def _prepare_frames(pre: dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str, list[str]]:
    target = str(pre.get("target_column") or "target").strip()
    drop_columns = [c.strip() for c in str(pre.get("drop_columns") or "").split(",") if c.strip()]
    if target in drop_columns:
        raise ValueError(f"target_column {target!r} must not appear in drop_columns")
    validation_split = float(pre.get("validation_split") if pre.get("validation_split") is not None else 0.2)
    max_train_rows = int(pre.get("max_train_rows") or 10_000)
    max_train_rows = max(100, min(max_train_rows, 50_000))

    train = _read_csv("train")
    if train is None:
        raise FileNotFoundError("no train.csv found")
    train = _clean_frame(train, target, drop_columns)
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(f"only {len(train)} usable training rows; need at least {MIN_TRAIN_ROWS}")
    if train[target].nunique() < 2:
        raise ValueError("regression target must contain at least 2 distinct finite values")

    feature_columns = [c for c in train.columns if c != target]
    if not feature_columns:
        raise ValueError("no feature columns remain")
    if len(feature_columns) > MAX_FEATURES:
        raise ValueError(f"{len(feature_columns)} features exceeds configured limit {MAX_FEATURES}")

    val = _read_csv("val")
    if val is not None:
        val = _clean_frame(val, target, drop_columns)
        if set(val.columns) != set(train.columns):
            raise ValueError("val.csv schema does not match train.csv after preprocessing")
        val = val[train.columns]
    else:
        split = min(max(validation_split, 0.05), 0.4)
        train, val = train_test_split(train, test_size=split, random_state=seed)
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)

    if len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=seed).reset_index(drop=True)

    test = _read_csv("test")
    if test is not None:
        test = _clean_frame(test, target, drop_columns)
        if set(test.columns) != set(train.columns):
            raise ValueError("test.csv schema does not match train.csv after preprocessing")
        test = test[train.columns].reset_index(drop=True)

    return train, val, test, target, feature_columns


def _batched(predict_fn, X: pd.DataFrame) -> np.ndarray:
    """Call a predict function in row batches to bound peak memory on large frames."""
    if len(X) <= PREDICT_BATCH_ROWS:
        return np.asarray(predict_fn(X))
    parts = [
        np.asarray(predict_fn(X.iloc[i:i + PREDICT_BATCH_ROWS]))
        for i in range(0, len(X), PREDICT_BATCH_ROWS)
    ]
    return np.concatenate(parts, axis=0)


def _regression_metrics(model, frame: pd.DataFrame, target: str) -> dict[str, Any]:
    X = frame.drop(columns=[target])
    y = frame[target].to_numpy(dtype=float)
    pred = _batched(model.predict, X).astype(float)
    mse = float(mean_squared_error(y, pred))
    return {
        "rows": int(len(frame)),
        "mae": float(mean_absolute_error(y, pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y, pred)),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_digest() -> dict[str, Any] | None:
    zips = sorted(DATASET_DIR.glob("*.zip"))
    if zips:
        return {"file": zips[0].name, "sha256": _sha256(zips[0])}
    csvs = sorted(DATASET_DIR.rglob("*.csv"))
    if not csvs:
        return None
    h = hashlib.sha256()
    for path in csvs:
        h.update(path.relative_to(DATASET_DIR).as_posix().encode("utf-8"))
        h.update(b"\0")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return {"files": [p.relative_to(DATASET_DIR).as_posix() for p in csvs], "sha256": h.hexdigest()}


def run() -> int:
    _load_limits()
    hp = _json_env("DIMER_HYPERPARAMETERS_JSON")
    pre = _json_env("DIMER_PREPROCESSING_ARGS_JSON")
    seed = int(hp.get("seed") or 0)
    train, val, test, target, feature_columns = _prepare_frames(pre, seed)

    # Ordinal-encode categorical feature columns (upstream #118: fine-tuning
    # fails on raw strings). Fit on the training context; keep a raw val copy
    # so the reload smoke can exercise the persisted encoder end-to-end.
    categorical_encoders = _fit_categorical_encoder(train, feature_columns)
    if categorical_encoders:
        log(f"Ordinal-encoding categorical features: {sorted(categorical_encoders)}")
    val_raw = val
    train = _apply_categorical_encoder(train, categorical_encoders)
    val = _apply_categorical_encoder(val, categorical_encoders)
    if test is not None:
        test = _apply_categorical_encoder(test, categorical_encoders)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("TabICLv2 fine-tuning requires a CUDA GPU in this DIMER pipeline")

    from tabicl import FinetunedTabICLRegressor, TabICLRegressor

    # Base-model handoff. This pipeline is fixed to the pinned TabICLv2 regressor
    # checkpoint by default, but DIMER can override it by mounting its selected
    # Base Model and setting DIMER_BASE_MODEL_PATH. A provided path is used as-is
    # and its SHA-256 recorded; the pinned-download path is SHA-verified so a
    # moved "main" cannot silently change the default base.
    from huggingface_hub import hf_hub_download
    provided = os.getenv("DIMER_BASE_MODEL_PATH", "").strip()
    if provided and Path(provided).exists():
        base_ckpt = Path(provided)
        base_source = "provided-path"
    else:
        base_ckpt = Path(hf_hub_download(BASE_MODEL_REPO, BASE_MODEL, revision=BASE_MODEL_REVISION))
        base_source = "pinned-download"
    base_model_sha256 = _sha256(base_ckpt)
    base_matches_pinned = base_model_sha256 == BASE_MODEL_SHA256
    if base_source == "pinned-download" and not base_matches_pinned:
        raise RuntimeError(
            f"base checkpoint sha256 {base_model_sha256} does not match the pinned "
            f"{BASE_MODEL_SHA256} at revision {BASE_MODEL_REVISION}"
        )
    if base_source == "provided-path" and not base_matches_pinned:
        log(
            f"Using a DIMER-provided base checkpoint (sha256 {base_model_sha256}) that "
            f"differs from the pinned default {BASE_MODEL_SHA256}."
        )

    epochs = int(hp.get("epochs") or 30)
    learning_rate = float(hp.get("learning_rate") or 1e-5)
    weight_decay = float(hp.get("weight_decay") if hp.get("weight_decay") is not None else 0.01)
    patience = int(hp.get("patience") or 8)
    time_limit = float(hp.get("time_limit_seconds") or 1800)
    eval_metric = str(hp.get("eval_metric") or "mae")
    n_ft = int(hp.get("n_estimators_finetune") or 2)
    n_val = int(hp.get("n_estimators_validation") or 2)
    n_inf = int(hp.get("n_estimators_inference") or 8)

    artifact_dir = OUTPUT_DIR / "tabicl_regressor"
    ckpt_dir = artifact_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = train.drop(columns=[target]), train[target]
    X_val, y_val = val.drop(columns=[target]), val[target]

    model = FinetunedTabICLRegressor(
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_estimators_finetune=n_ft,
        n_estimators_validation=n_val,
        n_estimators_inference=n_inf,
        early_stopping=True,
        patience=patience,
        time_limit=time_limit,
        eval_metric=eval_metric,
        model_path=str(base_ckpt),
        allow_auto_download=False,
        device="cuda",
        random_state=seed,
        verbose=True,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, output_dir=str(ckpt_dir))

    best_ckpt = ckpt_dir / "best.ckpt"
    if not best_ckpt.exists():
        raise RuntimeError("TabICL fine-tuning completed without producing checkpoints/best.ckpt")

    val_metrics = _regression_metrics(model, val, target)
    test_metrics = _regression_metrics(model, test, target) if test is not None and len(test) else None

    # Build a portable inference artifact: fine-tuned checkpoint + the ICL context table.
    context_path = artifact_dir / "training_context.parquet"
    train.to_parquet(context_path, index=False)
    checkpoint_sha256 = _sha256(best_ckpt)
    training_context_sha256 = _sha256(context_path)
    # artifact.json is the complete inference contract: everything a serving
    # process needs to rebuild the exact model that produced the reported metrics.
    manifest = {
        "artifactFormat": "tabicl-dimer-regressor-v1",
        "checkpoint": "checkpoints/best.ckpt",
        "trainingContext": "training_context.parquet",
        "targetColumn": target,
        "featureColumns": feature_columns,
        "baseCheckpoint": BASE_MODEL,
        "baseModelRevision": BASE_MODEL_REVISION,
        "baseModelSha256": base_model_sha256,
        "baseModelSource": base_source,
        "baseMatchesPinned": base_matches_pinned,
        "tabiclVersion": TABICL_VERSION,
        "inference": {
            "class": "TabICLRegressor",
            "modelPath": "checkpoints/best.ckpt",
            "nEstimators": n_inf,
            "randomState": seed,
            "device": "cuda",
            "allowAutoDownload": False,
            "categoricalEncoders": categorical_encoders,
            "encoding": "ordinal; code = index in categoricalEncoders[col]; missing/unseen = len(list)",
            "procedure": (
                "TabICLRegressor(model_path, **inference); "
                "apply categoricalEncoders to X; "
                "fit(context[featureColumns], context[targetColumn]); "
                "predict(X[featureColumns])"
            ),
        },
        "digests": {
            "checkpointSha256": checkpoint_sha256,
            "trainingContextSha256": training_context_sha256,
        },
    }
    (artifact_dir / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Prove the artifact is a self-contained inference contract: reconstruct the
    # model using ONLY artifact.json + the context table, then predict.
    served = json.loads((artifact_dir / "artifact.json").read_text(encoding="utf-8"))
    inference = served["inference"]
    context = pd.read_parquet(artifact_dir / served["trainingContext"])
    ctx_features = context[served["featureColumns"]]
    ctx_target = context[served["targetColumn"]]
    reloaded = TabICLRegressor(
        model_path=str(artifact_dir / inference["modelPath"]),
        allow_auto_download=inference["allowAutoDownload"],
        n_estimators=inference["nEstimators"],
        random_state=inference["randomState"],
        device=inference["device"],
    )
    reloaded.fit(ctx_features, ctx_target)
    smoke_rows = min(8, len(val_raw))
    if smoke_rows:
        smoke_X = _apply_categorical_encoder(val_raw, inference.get("categoricalEncoders", {}))
        _ = reloaded.predict(smoke_X[served["featureColumns"]].iloc[:smoke_rows])

    # Keep only best.ckpt in the served artifact; drop intermediate epoch checkpoints.
    pruned_bytes = 0
    for stale in sorted(ckpt_dir.glob("epoch*.ckpt")):
        pruned_bytes += stale.stat().st_size
        stale.unlink()
    if pruned_bytes:
        log(f"Pruned {pruned_bytes} bytes of intermediate epoch checkpoints")

    payload = {
        "successful": True,
        "message": f"TabICLv2 fine-tuning succeeded on {len(train)} rows; validation MAE {val_metrics['mae']:.4f}.",
        "metrics": {
            "trainRows": int(len(train)),
            "val": val_metrics,
            "test": test_metrics,
            "featureCount": len(feature_columns),
            "device": "cuda",
            "mode": "fine-tune",
        },
        "artifacts": {"modelDir": str(artifact_dir), "checkpoint": str(best_ckpt), "trainingContext": str(context_path)},
        "provenance": {"baseModel": BASE_MODEL, "baseModelRevision": BASE_MODEL_REVISION, "baseModelSha256": base_model_sha256, "baseModelSource": base_source, "baseMatchesPinned": base_matches_pinned, "tabiclVersion": TABICL_VERSION, "fineTunedCheckpointSha256": checkpoint_sha256, "trainingContextSha256": training_context_sha256, "artifactDigestSha256": hashlib.sha256((checkpoint_sha256 + training_context_sha256).encode("utf-8")).hexdigest(), "dataset": _dataset_digest()},
        "metadata": {"template": TEMPLATE_NAME, "taskType": "tabular_regression", "targetColumn": target, "seed": seed, "epochs": epochs, "learningRate": learning_rate, "evalMetric": eval_metric},
    }
    write_result(payload)
    log(f"Callback: {json.dumps(notify_done_callback(), sort_keys=True)}")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:  # noqa: BLE001
        payload = {"successful": False, "message": f"TabICLv2 fine-tuning failed: {exc}", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "metadata": {"template": TEMPLATE_NAME, "taskType": "tabular_regression"}}
        try:
            write_result(payload)
            notify_done_callback()
        except Exception as write_exc:  # noqa: BLE001
            log(f"Failed to persist crash result: {write_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
