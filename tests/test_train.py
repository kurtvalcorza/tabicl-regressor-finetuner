import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

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
