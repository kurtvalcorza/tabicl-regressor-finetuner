# ruff: noqa: E501  -- long docstrings and messages are kept readable
"""Safetensors serving for TabICLv2 regressors (tabicl==2.1.1) and the DIMER serving-bundle formats.

CANONICAL SOURCE: kurtvalcorza/tabicl-regressor-pipeline, src/tabicl_regressor_pipeline/safetensors_shim.py.
Vendored byte for byte, never edited there, into kurtvalcorza/tabicl-regressor-finetuner as
tabicl_safetensors_shim.py; that repository's tests pin this file's SHA-256. To re-sync after a change here, run
``python tools/print_shim_sha256.py`` in this repository, copy this file over the vendored one, and update the pinned
digest in the worker's tests/test_vendored_shim.py to the printed value.

This module depends only on the standard library at import time (tabicl, torch and safetensors are imported where they
are used), so it can be carried by a notebook or vendored by a worker without the rest of this package.

- ``convert_checkpoint`` reads a digest-checked tabicl ``.ckpt`` once with ``torch.load(weights_only=True)`` and writes
  its ``state_dict`` as safetensors with its ``config`` as sorted JSON string metadata (byte-reproducible).
- ``convert_finetuned_checkpoint`` does the same for the ``best.ckpt``/``epoch{N}.ckpt`` tabicl's fine-tuning loop
  writes itself with ``torch.save``, and removes the pickle.
- ``SafetensorsTabICLRegressor`` overrides only ``TabICLRegressor._load_model``: the network is built from the file's
  config and loaded with ``load_state_dict(strict=True)``; ``torch.load`` is never called.
- Bundle format v2 (``tabicl-dimer-regressor-v2``) carries ``checkpoints/best.safetensors``. v1 bundles
  (``checkpoints/best.ckpt``) stay readable through ``LegacyCheckpointTabICLRegressor``: ``weights_only=True`` only,
  with a ``DeprecationWarning``, until the v1 artifacts are re-issued.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ARTIFACT_FORMAT = "tabicl-dimer-regressor-v2"
LEGACY_ARTIFACT_FORMAT = "tabicl-dimer-regressor-v1"
SUPPORTED_ARTIFACT_FORMATS = (ARTIFACT_FORMAT, LEGACY_ARTIFACT_FORMAT)
BUNDLE_CHECKPOINT = "checkpoints/best.safetensors"
LEGACY_BUNDLE_CHECKPOINT = "checkpoints/best.ckpt"
CONVERSION_FORMAT = "tabicl-state-dict-safetensors-v1"
CONFIG_METADATA_KEY = "tabicl_config"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_safetensors(data: bytes) -> bytes:
    """Re-emit a safetensors header as sorted, compact JSON padded with spaces to 8 bytes; the data is unchanged.

    ``safetensors.torch.save`` (0.8.0) writes ``__metadata__`` in hash-map order, so equal inputs can give different
    bytes. Tensor offsets are relative to the end of the header, so re-ordering the header keeps them valid."""
    size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + size])
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return len(text).to_bytes(8, "little") + text + data[8 + size :]


def convert_checkpoint(source: str | Path, destination: str | Path, *, expected_sha256: str, source_name: str) -> dict[str, Any]:
    """Convert a tabicl ``.ckpt`` to safetensors once; return the source and converted digests.

    The source must match ``expected_sha256`` before it is read, and is read with
    ``torch.load(map_location="cpu", weights_only=True)``. The ``state_dict`` tensors are made contiguous CPU clones
    and the ``config`` is stored as sorted JSON under ``CONFIG_METADATA_KEY``."""
    source, destination = Path(source), Path(destination)
    source_sha = sha256_file(source)
    if source_sha != expected_sha256:
        raise ValueError(f"{source.name}: sha256 {source_sha} != pinned {expected_sha256}; refusing to deserialise it")
    import torch
    from safetensors.torch import save

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"{source.name}: not a tabicl checkpoint (needs 'config' and 'state_dict')")
    config = dict(checkpoint["config"])
    tensors = {key: value.detach().to("cpu").contiguous().clone() for key, value in checkpoint["state_dict"].items()}
    metadata = {
        "format": CONVERSION_FORMAT,
        CONFIG_METADATA_KEY: json.dumps(config, sort_keys=True, separators=(",", ":")),
        "source_file": source_name,
        "source_sha256": source_sha,
    }
    data = canonical_safetensors(save(tensors, metadata=metadata))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, destination)
    return {"source_sha256": source_sha, "converted_sha256": hashlib.sha256(data).hexdigest(), "tensors": len(tensors), "bytes": len(data)}


def convert_finetuned_checkpoint(source: str | Path, destination: str | Path | None = None, *, remove_source: bool = True) -> dict[str, Any]:
    """Convert a ``.ckpt`` that tabicl's fine-tuning loop just wrote (``best.ckpt``/``epoch{N}.ckpt``) to safetensors.

    tabicl 2.1.1 writes its fine-tuning checkpoints itself with ``torch.save`` (``config``, ``state_dict`` and the
    optimizer/scheduler state). The file is this process's own output, so its observed SHA-256 is the expected one; it is
    read once with ``torch.load(weights_only=True)``, only ``config`` and ``state_dict`` are kept, and the pickle is
    removed unless ``remove_source=False``. Returns ``convert_checkpoint``'s facts plus the destination path."""
    source = Path(source)
    target = Path(destination) if destination is not None else source.with_suffix(".safetensors")
    facts = convert_checkpoint(source, target, expected_sha256=sha256_file(source), source_name=source.name)
    if remove_source:
        source.unlink()
    return {**facts, "path": str(target)}


def load_converted_network(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Build the TabICL network from a converted file's config and load its tensors with ``strict=True``."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    from tabicl._model.tabicl import TabICL

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != CONVERSION_FORMAT or CONFIG_METADATA_KEY not in metadata:
        raise ValueError(f"{path}: not a {CONVERSION_FORMAT} file")
    config = json.loads(metadata[CONFIG_METADATA_KEY])
    network = TabICL(**config)
    network.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    network.eval()
    return network, config


def load_legacy_checkpoint_network(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """DEPRECATED: build the network from a v1 pickle checkpoint (``checkpoints/best.ckpt``) for old bundles only.

    Read with ``torch.load(map_location="cpu", weights_only=True)``, never ``weights_only=False``; the network is built
    from its ``config`` and loaded with ``strict=True``. Kept until the v1 release artifacts are re-issued as v2."""
    import torch
    from tabicl._model.tabicl import TabICL

    warnings.warn(
        f"{Path(path).name}: loading a pickle checkpoint from a {LEGACY_ARTIFACT_FORMAT} bundle (weights_only=True). "
        f"This layout is deprecated; re-issue the bundle as {ARTIFACT_FORMAT} with {BUNDLE_CHECKPOINT}.",
        DeprecationWarning,
        stacklevel=2,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"{path}: not a tabicl checkpoint (needs 'config' and 'state_dict')")
    config = dict(checkpoint["config"])
    network = TabICL(**config)
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    network.eval()
    return network, config


_SHIM_CLASSES: dict[str, type] = {}


def safetensors_regressor_class() -> type:
    """``TabICLRegressor`` with ``_load_model`` reading the converted safetensors file named by ``model_path``."""
    if "regressor" not in _SHIM_CLASSES:
        from tabicl import TabICLRegressor

        class SafetensorsTabICLRegressor(TabICLRegressor):
            def _load_model(self) -> None:
                self.model_path_ = Path(self.model_path)
                self.model_, self.model_config_ = load_converted_network(self.model_path_)

        _SHIM_CLASSES["regressor"] = SafetensorsTabICLRegressor
    return _SHIM_CLASSES["regressor"]


def legacy_checkpoint_regressor_class() -> type:
    """DEPRECATED: ``TabICLRegressor`` whose ``_load_model`` reads a v1 pickle checkpoint via ``load_legacy_checkpoint_network``."""
    if "legacy" not in _SHIM_CLASSES:
        from tabicl import TabICLRegressor

        class LegacyCheckpointTabICLRegressor(TabICLRegressor):
            def _load_model(self) -> None:
                self.model_path_ = Path(self.model_path)
                self.model_, self.model_config_ = load_legacy_checkpoint_network(self.model_path_)

        _SHIM_CLASSES["legacy"] = LegacyCheckpointTabICLRegressor
    return _SHIM_CLASSES["legacy"]


def safetensors_finetuned_regressor_class() -> type:
    """``FinetunedTabICLRegressor`` whose inner estimator is the safetensors shim (the pretrained load goes through it)."""
    if "finetuned" not in _SHIM_CLASSES:
        from tabicl import FinetunedTabICLRegressor

        class SafetensorsFinetunedTabICLRegressor(FinetunedTabICLRegressor):
            def _create_inner_estimator(self, *args: Any, **kwargs: Any):
                inner = super()._create_inner_estimator(*args, **kwargs)
                return safetensors_regressor_class()(**inner.get_params())

        _SHIM_CLASSES["finetuned"] = SafetensorsFinetunedTabICLRegressor
    return _SHIM_CLASSES["finetuned"]


def serving_regressor_class(model_path: str | Path) -> type:
    """The estimator class for a weights file: the safetensors shim for ``.safetensors``, else the deprecated v1 loader."""
    return safetensors_regressor_class() if Path(model_path).suffix == ".safetensors" else legacy_checkpoint_regressor_class()


def check_bundle_layout(manifest: Mapping[str, Any]) -> str:
    """The checkpoint member a bundle format allows: v2 names ``BUNDLE_CHECKPOINT`` in ``checkpoint``,
    ``inference.modelPath`` and ``payloadFiles``; a v1 bundle names a ``.ckpt`` (deprecated, ``weights_only=True`` only).
    Returns the format; anything else is refused before a checkpoint is opened."""
    fmt = manifest.get("artifactFormat")
    checkpoint = str(manifest.get("checkpoint"))
    model_path = (manifest.get("inference") or {}).get("modelPath")
    payload = manifest.get("payloadFiles")
    if fmt == ARTIFACT_FORMAT:
        if checkpoint != BUNDLE_CHECKPOINT or model_path not in (None, BUNDLE_CHECKPOINT) or (payload is not None and BUNDLE_CHECKPOINT not in payload):
            raise ValueError(f"{ARTIFACT_FORMAT} bundles carry {BUNDLE_CHECKPOINT} (checkpoint, inference.modelPath, payloadFiles); got checkpoint={checkpoint!r}, modelPath={model_path!r}")
    elif fmt == LEGACY_ARTIFACT_FORMAT:
        if Path(checkpoint).suffix != ".ckpt":
            raise ValueError(f"{LEGACY_ARTIFACT_FORMAT} bundles carry a .ckpt checkpoint; got {checkpoint!r}")
    else:
        raise ValueError(f"Unsupported artifactFormat: {fmt} (supported: {list(SUPPORTED_ARTIFACT_FORMATS)})")
    return fmt
