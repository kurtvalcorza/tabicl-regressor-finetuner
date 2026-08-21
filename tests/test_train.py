import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("train", Path(__file__).parents[1] / "train.py")
train = importlib.util.module_from_spec(SPEC)
sys.modules["train"] = train
SPEC.loader.exec_module(train)


@pytest.fixture(autouse=True)
def _reset_limits():
    train._load_limits()
    yield


def test_load_limits_rejects_malformed_env(monkeypatch):
    monkeypatch.setenv("DIMER_MAX_SINGLE_CSV_BYTES", "not-an-int")
    with pytest.raises(ValueError):
        train._load_limits()


def test_main_writes_failure_on_malformed_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DIMER_MAX_SINGLE_CSV_BYTES", "not-an-int")
    monkeypatch.setattr(train, "RESULT_PATH", tmp_path / "result.json")
    assert train.main() == 1
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["successful"] is False


def test_invalid_timeout_does_not_corrupt_global(monkeypatch):
    monkeypatch.setenv("DIMER_CALLBACK_TIMEOUT_SECONDS", "-5")
    prior = train.CALLBACK_TIMEOUT_SECONDS
    with pytest.raises(ValueError):
        train._load_limits()
    assert train.CALLBACK_TIMEOUT_SECONDS == prior
    assert train.CALLBACK_TIMEOUT_SECONDS > 0


def test_batched_predict_chunks_and_matches(monkeypatch):
    monkeypatch.setattr(train, "PREDICT_BATCH_ROWS", 3)
    X = pd.DataFrame({"a": range(10)})
    calls = []

    def fake(sub):
        calls.append(len(sub))
        return sub["a"].to_numpy()

    out = train._batched(fake, X)
    assert list(out) == list(range(10))
    assert calls == [3, 3, 3, 1]


def test_resolve_base_model_missing_provided_path_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("DIMER_BASE_MODEL_PATH", str(tmp_path / "nope.ckpt"))
    monkeypatch.delenv("TABICL_BAKED_BASE_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="does not exist"):
        train._resolve_base_model()


def test_resolve_base_model_custom_provided_provenance(tmp_path, monkeypatch):
    f = tmp_path / "custom.ckpt"
    f.write_bytes(b"not the pinned checkpoint")
    monkeypatch.setenv("DIMER_BASE_MODEL_PATH", str(f))
    monkeypatch.delenv("TABICL_BAKED_BASE_MODEL", raising=False)
    path, sha, source, matches, rev = train._resolve_base_model()
    assert path == f
    assert source == "dimer-provided"
    assert matches is False
    assert rev is None  # a custom base must not claim the pinned revision


def test_resolve_base_model_baked_mismatch_errors(tmp_path, monkeypatch):
    f = tmp_path / "baked.ckpt"
    f.write_bytes(b"wrong baked bytes")
    monkeypatch.delenv("DIMER_BASE_MODEL_PATH", raising=False)
    monkeypatch.setenv("TABICL_BAKED_BASE_MODEL", str(f))
    with pytest.raises(RuntimeError, match="pinned"):
        train._resolve_base_model()


def test_clean_frame_keeps_negative_targets():
    frame = pd.DataFrame({"x": [1, 2, 3], "target": [-5.0, 0.0, 2.5]})
    out = train._clean_frame(frame, "target", [])
    assert out["target"].tolist() == [-5.0, 0.0, 2.5]


def test_clean_frame_drops_nonfinite_targets():
    frame = pd.DataFrame({"x": range(5), "target": [1.0, np.nan, np.inf, -np.inf, 2.0]})
    out = train._clean_frame(frame, "target", [])
    assert out["target"].tolist() == [1.0, 2.0]


def test_metrics_do_not_clip_predictions():
    class Model:
        def predict(self, X):
            return np.array([-2.0, 1.0])
    frame = pd.DataFrame({"x": [0, 1], "target": [-3.0, 2.0]})
    metrics = train._regression_metrics(Model(), frame, "target")
    assert metrics["mae"] == 1.0


def test_categorical_encoder_ordinal_numeric_passthrough_and_unknown():
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["b", "a", "b"],
            "flag": [True, False, True],
            "target": [0.5, 1.5, 2.5],
        }
    )
    enc = train._fit_categorical_encoder(frame, ["num", "cat", "flag"])
    assert "num" not in enc  # numeric passthrough
    assert enc["cat"] == ["a", "b"]  # sorted categories
    assert enc["flag"] == ["False", "True"]

    out = train._apply_categorical_encoder(frame, enc)
    assert list(out["cat"]) == [1, 0, 1]  # b=1, a=0
    assert list(out["flag"]) == [1, 0, 1]  # True=1, False=0
    assert list(out["num"]) == [1.0, 2.0, 3.0]  # unchanged

    # unseen category and missing value both fall into the unknown bucket (len)
    infer = pd.DataFrame({"num": [9.0, 9.0], "cat": ["z", None], "flag": [True, None]})
    enc_infer = train._apply_categorical_encoder(infer, enc)
    assert list(enc_infer["cat"]) == [2, 2]  # z unseen -> 2, None -> 2
    assert list(enc_infer["flag"]) == [1, 2]  # True -> 1, None -> 2 (unknown)


def test_categorical_encoder_empty_when_all_numeric():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4], "target": [0.1, 0.2]})
    enc = train._fit_categorical_encoder(frame, ["a", "b"])
    assert enc == {}
    assert train._apply_categorical_encoder(frame, enc) is frame  # no-op passthrough


def test_normalize_member_rejects_traversal_and_absolute():
    assert train._normalize_member("train.csv") == "train.csv"
    assert train._normalize_member("./train.csv") == "train.csv"
    assert train._normalize_member("dataset/train.csv") == "train.csv"
    for hostile in ("../train.csv", "../../etc/passwd", "/train.csv", "dataset/../secret.csv"):
        with pytest.raises(ValueError, match="unsafe archive member"):
            train._normalize_member(hostile)


def test_prepare_frames_rejects_target_in_drop_columns():
    with pytest.raises(ValueError, match="must not appear in drop_columns"):
        train._prepare_frames({"target_column": "target", "drop_columns": "target,x"}, 0)


def test_manifest_matches_env_consumption():
    # Every DIMER manifest control must map 1:1 to a value train.py actually
    # reads from the env JSON, and nothing may be read that the manifest does
    # not declare. Guards against a silently-ignored (or undeclared) parameter.
    import re

    root = Path(__file__).parents[1]
    manifest = json.loads((root / "dimer-pipeline.json").read_text())
    src = (root / "train.py").read_text()
    hp_keys = set(re.findall(r"hp\.get\(\s*[\"']([^\"']+)[\"']", src))
    pre_keys = set(re.findall(r"pre\.get\(\s*[\"']([^\"']+)[\"']", src))
    manifest_hp = set(manifest["modelFinetuning"])
    manifest_pre = set(manifest["datasetPreprocessing"])
    assert manifest_hp <= hp_keys, f"manifest hyperparameters not consumed: {manifest_hp - hp_keys}"
    assert manifest_pre <= pre_keys, f"manifest preprocessing keys not consumed: {manifest_pre - pre_keys}"
    assert hp_keys <= manifest_hp, f"consumed but undeclared hyperparameters: {hp_keys - manifest_hp}"
    assert pre_keys <= manifest_pre, f"consumed but undeclared preprocessing keys: {pre_keys - manifest_pre}"
    assert "model_id" not in manifest_hp | manifest_pre


def test_normalize_device_string_honors_dimer_assignment():
    # DIMER_TRAIN_DEVICE contract: honor the assigned GPU; normalize the bare-index
    # pitfall ("0" -> "cuda:0"); reject non-CUDA (no CPU fine-tune path).
    assert train._normalize_device_string("") == "cuda"
    assert train._normalize_device_string("cuda") == "cuda"
    assert train._normalize_device_string("0") == "cuda:0"
    assert train._normalize_device_string("1") == "cuda:1"
    assert train._normalize_device_string("cuda:1") == "cuda:1"
    for bad in ("cpu", "mps", "gpu"):
        with pytest.raises(RuntimeError, match="not supported"):
            train._normalize_device_string(bad)


def test_resolve_task_type_chain(monkeypatch):
    # taskType precedence: DIMER metadata -> baked DIMER_TASK_TYPE env -> literal.
    monkeypatch.delenv("DIMER_TASK_TYPE", raising=False)
    assert train._resolve_task_type({}) == "tabular_regression"
    monkeypatch.setenv("DIMER_TASK_TYPE", "baked_custom")
    assert train._resolve_task_type({}) == "baked_custom"
    assert train._resolve_task_type({"taskType": "from_metadata"}) == "from_metadata"
