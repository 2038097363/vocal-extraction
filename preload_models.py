from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import math
import struct
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
MODELS = ROOT / "models"


def configure_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("TORCH_HOME", str(MODELS))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE / "numba"))
    os.environ.setdefault("PIP_CACHE_DIR", str(CACHE / "pip"))
    os.environ.setdefault("DEMUCS_CACHE", str(MODELS))


def create_silent_wav(path: Path) -> None:
    sample_rate = 44_100
    frames = sample_rate * 12
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        chunks = []
        for index in range(frames):
            value = int(1000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            frame = struct.pack("<hh", value, value)
            chunks.append(frame)
        wav.writeframes(b"".join(chunks))


def preload(model: str) -> None:
    with tempfile.TemporaryDirectory(prefix="demucs_preload_") as tmp:
        tmp_path = Path(tmp)
        wav_path = tmp_path / "silence.wav"
        out_path = tmp_path / "out"
        create_silent_wav(wav_path)
        print(f"Preloading {model}...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "demucs.separate",
                "-n",
                model,
                "--two-stems",
                "vocals",
                "--float32",
                "--out",
                str(out_path),
                str(wav_path),
            ],
            check=True,
        )


def main() -> None:
    configure_cache()
    preload("htdemucs")
    preload("htdemucs_ft")
    print(f"Models/cache directory: {MODELS}")


if __name__ == "__main__":
    main()
