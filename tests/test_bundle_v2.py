"""Bundle format v2: the worker packages checkpoints/best.safetensors, never a pickle; v1 bundles stay readable.

The run() test drives the real bundle-writing code on CPU with a stand-in for tabicl's fine-tuner that writes a tiny
TabICL network in the exact layout tabicl 2.1.1's ``_save_checkpoint`` uses (``best.ckpt`` and ``epoch{N}.ckpt`` via
``torch.save``). Skips without tabicl/torch/safetensors (the light CI job).
"""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
tabicl = pytest.importorskip("tabicl")

import tabicl_safetensors_shim as shim  # noqa: E402
import train  # noqa: E402

KIND = "regressor"
TINY = {"max_classes": 0, "embed_dim": 16, "col_num_blocks": 1, "col_nhead": 2, "col_num_inds": 8, "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2, "icl_num_blocks": 1, "icl_nhead": 2}


def _tiny_checkpoint(path: Path, seed: int = 0) -> None:
    from tabicl._model.tabicl import TabICL

    torch.manual_seed(seed)
    net = TabICL(**TINY)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for p in net.parameters():
        p.grad = torch.randn_like(p) * 1e-3
    opt.step()
    torch.save({"config": dict(TINY), "state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "optimizer_state": opt.state_dict(), "scheduler_state": None, "curr_step": 1, "epoch": 1, "best_metric": -0.5}, path)


def _refuse(*args, **kwargs):
    raise AssertionError("torch.load on the safetensors path")


class FakeFinetuner:
    """Stands in for FinetunedTabICLRegressor: writes tabicl's checkpoint files, then serves the tiny network."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, X_val=None, y_val=None, output_dir=None):
        out = Path(output_dir)
        _tiny_checkpoint(out / "best.ckpt")
        shutil.copy2(out / "best.ckpt", out / "epoch1.ckpt")
        self._inner = shim.legacy_checkpoint_regressor_class()(model_path=str(out / "best.ckpt"), n_estimators=2, random_state=0, device="cpu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._inner.fit(X, y)
        return self

    def predict(self, X):
        return self._inner.predict(X)


def _dataset(root: Path) -> None:
    rng = np.random.default_rng(0)
    for name, n in (("train", 320), ("val", 60)):
        X = rng.normal(size=(n, 4))
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        df["color"] = np.array(["a", "b"])[rng.integers(0, 2, size=n)]
        df["target"] = X[:, 0] + X[:, 1]
        df.to_csv(root / f"{name}.csv", index=False)


def test_run_packages_best_safetensors_only_and_reloads_through_the_shim(tmp_path, monkeypatch):
    ds, out, res = tmp_path / "dataset", tmp_path / "output", tmp_path / "results" / "result.json"
    ds.mkdir()
    _dataset(ds)
    monkeypatch.setattr(train, "DATASET_DIR", ds)
    monkeypatch.setattr(train, "OUTPUT_DIR", out)
    monkeypatch.setattr(train, "RESULT_PATH", res)
    monkeypatch.setattr(train, "_resolve_train_device", lambda: "cpu")
    monkeypatch.setattr(train, "_resolve_base_model", lambda: (tmp_path / "base.ckpt", train.BASE_MODEL_SHA256, "pinned-baked", True, train.BASE_MODEL_REVISION))
    monkeypatch.setattr(tabicl, "FinetunedTabICLRegressor", FakeFinetuner)
    reloads = []
    real_reload = train._reload_bundle

    def spy(artifact_dir, device, rows):
        assert torch.load is not _refuse
        out_pred = real_reload(artifact_dir, device, rows)
        reloads.append(out_pred)
        return out_pred

    monkeypatch.setattr(train, "_reload_bundle", spy)
    monkeypatch.setenv("DIMER_HYPERPARAMETERS_JSON", '{"epochs":1,"seed":0}')
    payload = train.run()
    assert payload["successful"] is True and reloads and reloads[0] is not None
    bundle = out / f"tabicl_{KIND}"
    assert sorted(p.name for p in bundle.rglob("*.ckpt")) == []
    manifest = json.loads((bundle / "artifact.json").read_text())
    assert manifest["artifactFormat"] == f"tabicl-dimer-{KIND}-v2"
    assert manifest["checkpoint"] == manifest["inference"]["modelPath"] == "checkpoints/best.safetensors"
    assert "checkpoints/best.safetensors" in manifest["payloadFiles"]
    on_disk = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file() and p.name != "artifact.json")
    assert manifest["payloadFiles"] == on_disk  # the pipeline's verify_artifact_bundle allowlist holds
    weights = bundle / "checkpoints" / "best.safetensors"
    assert manifest["digests"]["checkpointSha256"] == train._sha256(weights) and manifest["sizes"]["checkpoint"] == weights.stat().st_size
    model = payload["artifacts"]["modelArtifact"]
    assert model["name"] == "best.safetensors" and model["path"].endswith("checkpoints/best.safetensors") and model["sha256"] == train._sha256(weights)
    assert payload["provenance"]["fineTunedCheckpointSha256"] == model["sha256"] and payload["provenance"]["fineTunedCheckpointFormat"] == "safetensors"
    assert shim.check_bundle_layout(manifest) == shim.ARTIFACT_FORMAT


def test_the_converted_checkpoint_predicts_exactly_like_the_ckpt_with_torch_load_refused(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "bundle" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    _tiny_checkpoint(ckpt_dir / "best.ckpt", seed=3)
    shutil.copy2(ckpt_dir / "best.ckpt", ckpt_dir / "epoch1.ckpt")
    keep = tmp_path / "kept.ckpt"  # outside the bundle
    shutil.copy2(ckpt_dir / "best.ckpt", keep)
    path, facts = train._convert_best_checkpoint(ckpt_dir)
    assert path.name == "best.safetensors" and sorted(p.name for p in ckpt_dir.iterdir()) == ["best.safetensors"]
    assert facts["source_sha256"] == train._sha256(keep) and facts["removed_pickle_bytes"] == keep.stat().st_size
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=[f"f{i}" for i in range(4)])
    y = X["f0"] * 2.0
    stock = tabicl.TabICLRegressor(model_path=str(keep), allow_auto_download=False, n_estimators=2, random_state=0, device="cpu").fit(X[:40], y[:40])
    expected = stock.predict(X[40:])
    monkeypatch.setattr(torch, "load", _refuse)
    got = shim.serving_regressor_class(path)(model_path=str(path), n_estimators=2, random_state=0, device="cpu").fit(X[:40], y[:40]).predict(X[40:])
    monkeypatch.undo()
    assert np.array_equal(got, expected)


def test_a_v1_bundle_stays_readable_with_weights_only_and_a_deprecation_warning(tmp_path, monkeypatch):
    ckpt = tmp_path / "checkpoints" / "best.ckpt"
    ckpt.parent.mkdir()
    _tiny_checkpoint(ckpt)
    manifest = {"artifactFormat": f"tabicl-dimer-{KIND}-v1", "checkpoint": "checkpoints/best.ckpt", "inference": {"modelPath": "checkpoints/best.ckpt"}}
    assert shim.check_bundle_layout(manifest) == shim.LEGACY_ARTIFACT_FORMAT
    calls = []
    real = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    X = pd.DataFrame(np.random.default_rng(2).normal(size=(30, 3)), columns=["a", "b", "c"])
    y = X["a"] * 2.0
    model = shim.serving_regressor_class(ckpt)(model_path=str(ckpt), n_estimators=2, random_state=0, device="cpu")
    assert type(model).__name__ == "LegacyCheckpointTabICLRegressor"
    with pytest.warns(DeprecationWarning, match=f"tabicl-dimer-{KIND}-v1"):
        model.fit(X, y)
    assert calls and all(call.get("weights_only") is True for call in calls)
