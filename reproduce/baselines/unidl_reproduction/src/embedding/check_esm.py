from __future__ import annotations

from pathlib import Path


def esm_status(model_path: str | Path | None = None) -> dict:
    status = {"esm_importable": False, "local_model_exists": False, "model_path": None, "loadable": False}
    try:
        import esm  # noqa: F401
        status["esm_importable"] = True
    except Exception:
        status["esm_importable"] = False

    if model_path is not None:
        path = Path(model_path)
        status["model_path"] = str(path)
        status["local_model_exists"] = path.exists()
        status["loadable"] = status["esm_importable"] and status["local_model_exists"]
    return status
