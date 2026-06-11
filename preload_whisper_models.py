from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
CACHE = ROOT / "cache"


def configure_cache() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("HF_HOME", str(MODELS / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS / "huggingface" / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main() -> None:
    configure_cache()
    from faster_whisper import WhisperModel

    model_root = MODELS / "whisper"
    model_root.mkdir(parents=True, exist_ok=True)
    WhisperModel("base", device="cpu", compute_type="int8", download_root=str(model_root))
    print(f"Whisper base model is ready in: {model_root}")


if __name__ == "__main__":
    main()
