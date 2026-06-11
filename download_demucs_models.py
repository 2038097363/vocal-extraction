from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models" / "hub" / "checkpoints"
BASE_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer"
FILES = [
    "955717e8-8726e21a.th",
    "f7e0c4bc-ba3fe64a.th",
    "d12395a8-e57c48e6.th",
    "92cfc3b6-ef3bcb9c.th",
    "04573f0d-f3cf25b2.th",
]


def configure_cache() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))
    os.environ.setdefault("DEMUCS_CACHE", str(ROOT / "models"))
    MODELS.mkdir(parents=True, exist_ok=True)


def expected_hash_prefix(filename: str) -> str:
    return filename.rsplit("-", 1)[1].split(".", 1)[0]


def sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid(path: Path, filename: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    actual = sha256_prefix(path)
    expected = expected_hash_prefix(filename)
    if actual.startswith(expected):
        return True
    print(f"Hash mismatch for {filename}: expected {expected}, got {actual[:8]}")
    path.unlink(missing_ok=True)
    return False


def download(filename: str) -> None:
    target = MODELS / filename
    if valid(target, filename):
        print(f"Already OK: {filename}")
        return

    url = f"{BASE_URL}/{filename}"
    part = target.with_suffix(target.suffix + ".part")
    part.unlink(missing_ok=True)

    for attempt in range(1, 6):
        try:
            print(f"Downloading {filename} (attempt {attempt}/5)")
            with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as output:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                last_report = time.monotonic()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 5:
                        if total:
                            percent = done / total * 100
                            print(f"  {percent:5.1f}%  {done / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB")
                        else:
                            print(f"  {done / 1024 / 1024:.1f} MB")
                        last_report = now

            if valid(part, filename):
                part.replace(target)
                print(f"Saved: {target}")
                return
            part.unlink(missing_ok=True)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"Download failed: {exc}")
            part.unlink(missing_ok=True)
            time.sleep(3 * attempt)

    raise RuntimeError(f"Failed to download {filename}")


def main() -> None:
    configure_cache()
    for filename in FILES:
        download(filename)
    print(f"All requested Demucs models are ready in: {MODELS}")


if __name__ == "__main__":
    main()
