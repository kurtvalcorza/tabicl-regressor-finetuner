"""CPU checks that the pinned tabicl exposes the API surface train.py relies on.

Skipped automatically when tabicl is not installed (the fast unit-test job),
and exercised by the tabicl-integration CI job which installs
tabicl[finetune]==2.1.1.
"""
import inspect
from importlib.metadata import version as _pkg_version

import pytest

pytest.importorskip("tabicl")


def test_tabicl_version_is_pinned():
    assert _pkg_version("tabicl") == "2.1.1"


def test_finetuned_regressor_api_surface():
    from tabicl import FinetunedTabICLRegressor

    params = set(inspect.signature(FinetunedTabICLRegressor.__init__).parameters)
    required = {
        "model_path", "allow_auto_download", "checkpoint_version", "epochs",
        "learning_rate", "weight_decay", "n_estimators_finetune",
        "n_estimators_validation", "n_estimators_inference", "early_stopping",
        "patience", "time_limit", "eval_metric", "device", "random_state",
        "verbose",
    }
    assert required <= params, f"missing from FinetunedTabICLRegressor: {required - params}"


def test_plain_regressor_api_surface():
    from tabicl import TabICLRegressor

    params = set(inspect.signature(TabICLRegressor.__init__).parameters)
    required = {"model_path", "allow_auto_download", "n_estimators", "random_state", "device"}
    assert required <= params, f"missing from TabICLRegressor: {required - params}"


def test_finetuned_regressor_instantiates_with_train_kwargs():
    from tabicl import FinetunedTabICLRegressor

    # Same kwargs train.py passes; construction only (no fit, no download).
    FinetunedTabICLRegressor(
        epochs=1,
        n_estimators_finetune=2,
        n_estimators_validation=2,
        n_estimators_inference=8,
        allow_auto_download=False,
        device="cpu",
        random_state=0,
    )
