"""End-to-end GPU fine-tune integration test.

Skipped unless a CUDA GPU and tabicl are available, so it is inert on the
standard CPU CI runners and runnable on a self-hosted GPU runner or locally
(e.g. inside the built image, which bakes the pinned base checkpoint).
"""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tabicl")
if not torch.cuda.is_available():
    pytest.skip("requires a CUDA GPU", allow_module_level=True)

import train


def test_finetune_end_to_end(tmp_path, monkeypatch):
    ds = tmp_path / "dataset"
    ds.mkdir()
    rng = np.random.default_rng(0)
    w = np.array([1.5, -2.0, 0.7, 0.0, 3.0, -1.0])

    def make(n):
        X = rng.normal(size=(n, 6))
        y = X @ w + rng.normal(scale=0.5, size=n) + 5.0
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        df["color"] = np.array(["a", "b", "c"])[rng.integers(0, 3, size=n)]  # categorical
        df["target"] = y
        return df

    make(200).to_csv(ds / "train.csv", index=False)
    make(60).to_csv(ds / "val.csv", index=False)
    out = tmp_path / "output"
    res = tmp_path / "results" / "result.json"
    monkeypatch.setattr(train, "DATASET_DIR", ds)
    monkeypatch.setattr(train, "OUTPUT_DIR", out)
    monkeypatch.setattr(train, "RESULT_PATH", res)
    monkeypatch.setenv("DIMER_HYPERPARAMETERS_JSON", '{"epochs":1,"time_limit_seconds":300,"seed":0}')

    # main(), not run(), owns result persistence and process exit semantics.
    assert train.main() == 0
    payload = json.loads(res.read_text())
    assert payload["successful"] is True
    assert payload["provenance"]["baseModelRevision"]
    assert payload["provenance"]["baseModelSha256"]
    assert payload["provenance"]["baseModelSource"] in ("pinned-baked", "pinned-download")
    assert payload["provenance"]["baseMatchesPinned"] is True
    art = json.loads((out / "tabicl_regressor" / "artifact.json").read_text())
    assert "inference" in art
    assert "categoricalEncoders" in art["inference"]
    ckpts = sorted((out / "tabicl_regressor" / "checkpoints").glob("*.ckpt"))
    assert [p.name for p in ckpts] == ["best.ckpt"]
