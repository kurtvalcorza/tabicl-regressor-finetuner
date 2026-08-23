"""Container entrypoint preserving normal train.py behavior plus GPU-burst I/O."""
from __future__ import annotations

import json
from pathlib import Path

import dimer_transport


def _callback(train) -> None:
    try:
        train.log(f"Callback: {json.dumps(train.notify_done_callback(), sort_keys=True)}")
    except Exception as exc:  # noqa: BLE001
        train.log(f"Callback failed: {exc}")


def _write(train, payload: dict) -> bool:
    try:
        train.write_result(payload)
        return True
    except Exception as exc:  # noqa: BLE001
        train.log(f"Failed to persist result: {exc}")
        return False


def _burst_main(runtime: dict) -> int:
    # Import only after prepare_burst_runtime() has redirected DIMER_* paths;
    # train.py resolves them at module import time.
    import train

    payload = train._crash_payload()
    rc = 1
    try:
        payload = train.run()
        rc = 0 if payload.get("successful") else 1
    except Exception as exc:  # noqa: BLE001
        payload = train._crash_payload(exc)
        rc = 1

    wrote = _write(train, payload)
    if wrote:
        try:
            if rc == 0:
                status = dimer_transport.publish_success_bundle(
                    payload=payload,
                    dataset_dir=Path(runtime["datasetDir"]),
                    output_dir=Path(runtime["outputDir"]),
                    result_path=Path(runtime["resultPath"]),
                )
                train.log(f"GPU-burst publish: {json.dumps(status, sort_keys=True)}")
            else:
                status = dimer_transport.publish_result_only(Path(runtime["resultPath"]))
                train.log(f"GPU-burst failure result publish: {json.dumps(status, sort_keys=True)}")
        except Exception as publish_exc:  # noqa: BLE001
            # A successful model that was not durably published is not a successful
            # DIMER run. Rewrite the local result as failure and make one best-effort
            # attempt to publish that failure envelope before notifying DIMER.
            train.log(f"GPU-burst publication failed: {publish_exc}")
            payload = train._crash_payload(publish_exc)
            payload["message"] = f"TabICLv2 fine-tuning output publication failed: {publish_exc}"
            rc = 1
            if _write(train, payload):
                try:
                    dimer_transport.publish_result_only(Path(runtime["resultPath"]))
                except Exception as result_exc:  # noqa: BLE001
                    train.log(f"GPU-burst failure result publication failed: {result_exc}")

    _callback(train)
    return rc


def main() -> int:
    if not dimer_transport.burst_mode_enabled():
        import train

        return train.main()

    try:
        runtime = dimer_transport.prepare_burst_runtime()
        assert runtime is not None
    except Exception as stage_exc:  # noqa: BLE001
        # prepare_burst_runtime sets DIMER_OUTPUT_DIR / DIMER_RESULT_PATH before
        # staging, so the ordinary train failure envelope can still be persisted.
        import train

        payload = train._crash_payload(stage_exc)
        payload["message"] = f"TabICLv2 fine-tuning dataset staging failed: {stage_exc}"
        if _write(train, payload):
            try:
                dimer_transport.publish_result_only(train.RESULT_PATH)
            except Exception as result_exc:  # noqa: BLE001
                train.log(f"GPU-burst staging failure result publication failed: {result_exc}")
        _callback(train)
        return 1

    return _burst_main(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
