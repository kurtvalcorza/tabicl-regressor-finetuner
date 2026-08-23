from __future__ import annotations

from pathlib import Path

import pytest

import dimer_transport


class FakePaginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, **_kwargs):
        return self.pages


class FakeS3:
    def __init__(self, objects=None):
        self.objects = objects or {}
        self.uploads = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator([
            {"Contents": [{"Key": key, "Size": len(body)} for key, body in self.objects.items()]}
        ])

    def download_file(self, bucket, key, destination):
        Path(destination).write_bytes(self.objects[key])

    def upload_file(self, source, bucket, key):
        self.uploads.append((source, bucket, key))


def _burst_env(monkeypatch):
    monkeypatch.setenv("GPU_BURST_MODE", "true")
    monkeypatch.setenv("DIMER_RUN_ID", "r1")
    monkeypatch.setenv("GPU_BURST_S3_BUCKET", "dimer")
    monkeypatch.setenv("GPU_BURST_DATASET_PREFIX", "workbench/sessions/s1/dataset/")
    monkeypatch.setenv("GPU_BURST_RESULT_KEY", "workbench/sessions/s1/fine-tuning/r1/result.json")
    monkeypatch.setenv("GPU_BURST_MODEL_KEY", "workbench/sessions/s1/fine-tuning/r1/artifacts/best.pt")


def test_non_burst_mode_does_not_rewrite_runtime(monkeypatch):
    monkeypatch.delenv("GPU_BURST_MODE", raising=False)
    assert dimer_transport.prepare_burst_runtime() is None


def test_burst_runtime_stages_dataset_and_rewrites_paths(tmp_path, monkeypatch):
    _burst_env(monkeypatch)
    monkeypatch.setenv("DIMER_BURST_SCRATCH_ROOT", str(tmp_path))
    prefix = "workbench/sessions/s1/dataset/"
    fake = FakeS3({prefix + "train.csv": b"x,target\n1,1.0\n2,2.0\n"})
    monkeypatch.setattr(dimer_transport, "_s3_client", lambda: fake)

    runtime = dimer_transport.prepare_burst_runtime()

    assert runtime is not None
    assert runtime["fileCount"] == 1
    assert Path(runtime["datasetDir"], "train.csv").exists()
    assert Path(runtime["outputDir"]) == tmp_path / "fine-tuning" / "r1"
    assert Path(runtime["resultPath"]) == tmp_path / "fine-tuning" / "r1" / "result.json"


def test_unsafe_burst_dataset_key_is_rejected(tmp_path, monkeypatch):
    _burst_env(monkeypatch)
    monkeypatch.setenv("DIMER_BURST_SCRATCH_ROOT", str(tmp_path))
    prefix = "workbench/sessions/s1/dataset/"
    fake = FakeS3({prefix + "../escape.csv": b"bad"})
    monkeypatch.setattr(dimer_transport, "_s3_client", lambda: fake)

    with pytest.raises(ValueError, match="unsafe GPU-burst dataset object key"):
        dimer_transport.prepare_burst_runtime()


def test_success_bundle_uploads_result_last(tmp_path, monkeypatch):
    _burst_env(monkeypatch)
    fake = FakeS3()
    monkeypatch.setattr(dimer_transport, "_s3_client", lambda: fake)

    root = tmp_path
    dataset_dir = root / "dataset"
    output_dir = root / "fine-tuning" / "r1"
    artifact = output_dir / "tabicl_regressor" / "checkpoints" / "best.ckpt"
    context = output_dir / "tabicl_regressor" / "training_context.parquet"
    manifest = output_dir / "tabicl_regressor" / "artifact.json"
    result = output_dir / "result.json"
    dataset_dir.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    (dataset_dir / "train.csv").write_bytes(b"dataset")
    artifact.write_bytes(b"checkpoint")
    context.write_bytes(b"context")
    manifest.write_text("{}\n", encoding="utf-8")
    result.write_text("{\"successful\":true}\n", encoding="utf-8")

    def descriptor(path, name):
        return {
            "path": path.relative_to(root).as_posix(),
            "name": name,
            "contentType": "application/octet-stream",
            "sizeBytes": path.stat().st_size,
        }

    payload = {
        "successful": True,
        "artifacts": {
            "modelArtifact": descriptor(artifact, "best.ckpt"),
            "trainingContext": descriptor(context, "training_context.parquet"),
            "manifest": descriptor(manifest, "artifact.json"),
        },
    }

    status = dimer_transport.publish_success_bundle(
        payload=payload,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        result_path=result,
    )

    uploaded_keys = [key for _source, _bucket, key in fake.uploads]
    assert uploaded_keys[-1] == "workbench/sessions/s1/fine-tuning/r1/result.json"
    assert "workbench/sessions/s1/fine-tuning/r1/tabicl_regressor/checkpoints/best.ckpt" in uploaded_keys
    assert "workbench/sessions/s1/fine-tuning/r1/tabicl_regressor/training_context.parquet" in uploaded_keys
    assert "workbench/sessions/s1/fine-tuning/r1/tabicl_regressor/artifact.json" in uploaded_keys
    assert "workbench/sessions/s1/fine-tuning/r1/artifacts/best.pt" in uploaded_keys
    assert status["resultKey"] == uploaded_keys[-1]


def test_declared_missing_artifact_fails_before_result_publish(tmp_path, monkeypatch):
    _burst_env(monkeypatch)
    fake = FakeS3()
    monkeypatch.setattr(dimer_transport, "_s3_client", lambda: fake)
    output_dir = tmp_path / "fine-tuning" / "r1"
    dataset_dir = tmp_path / "dataset"
    result = output_dir / "result.json"
    output_dir.mkdir(parents=True)
    dataset_dir.mkdir()
    result.write_text("{}", encoding="utf-8")
    payload = {
        "successful": True,
        "artifacts": {
            "modelArtifact": {
                "path": "fine-tuning/r1/tabicl_regressor/checkpoints/best.ckpt",
                "name": "best.ckpt",
                "sizeBytes": 1,
            }
        },
    }

    with pytest.raises(FileNotFoundError, match="declared artifact"):
        dimer_transport.publish_success_bundle(
            payload=payload,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            result_path=result,
        )
    assert fake.uploads == []
