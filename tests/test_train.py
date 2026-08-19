import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("train", Path(__file__).parents[1] / "train.py")
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


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
