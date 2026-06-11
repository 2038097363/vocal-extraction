from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from voice_extractor import FORMATS, OutputFormat, clean_name, configure_local_cache, write_transcripts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从视频中提取并分离人声。")
    parser.add_argument("input", type=Path, help="输入视频文件")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument(
        "-f",
        "--format",
        choices=[fmt.ext for fmt in FORMATS.values()],
        default="flac",
        help="输出格式: flac, wav, mp3, m4a, ogg, opus",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=("htdemucs", "htdemucs_ft"),
        default="htdemucs",
        help="Demucs 模型",
    )
    parser.add_argument("--sample-rate", choices=("44100", "48000"), help="重采样到指定采样率")
    parser.add_argument("--transcribe", action="store_true", help="同时识别人声文字，输出 TXT 和 SRT")
    parser.add_argument("--whisper-model", default="base", choices=("tiny", "base", "small", "medium"), help="文字识别模型")
    parser.add_argument("--language", choices=("zh", "en", "ja", "ko"), help="指定识别语言；不填则自动识别")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")
    return parser.parse_args()


def require_dependencies(transcribe: bool) -> None:
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    if importlib.util.find_spec("demucs") is None:
        missing.append("demucs")
    if transcribe and importlib.util.find_spec("faster_whisper") is None:
        missing.append("faster-whisper")
    if missing:
        raise SystemExit("缺少运行依赖: " + ", ".join(missing) + "。请先运行 Install.bat。")


def select_format(ext: str) -> OutputFormat:
    for fmt in FORMATS.values():
        if fmt.ext == ext:
            return fmt
    raise ValueError(ext)


def next_output_path(output_dir: Path, source_stem: str, ext: str) -> Path:
    base = clean_name(source_stem) + "_vocals"
    candidate = output_dir / f"{base}.{ext}"
    index = 2
    while candidate.exists():
        candidate = output_dir / f"{base}_{index}.{ext}"
        index += 1
    return candidate


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_vocals_file(root: Path) -> Path:
    candidates = sorted(root.rglob("vocals.wav"))
    if not candidates:
        raise FileNotFoundError("没有找到 Demucs 输出的人声文件 vocals.wav")
    return candidates[0]


def transcribe(audio_path: Path, txt_path: Path, srt_path: Path, model_name: str, language: str | None) -> None:
    from faster_whisper import WhisperModel

    model_root = Path("models") / "whisper"
    model_root.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(model_root))
    segments_iter, info = model.transcribe(str(audio_path), language=language, vad_filter=True, beam_size=5)
    segments = list(segments_iter)
    if getattr(info, "language", None):
        print(f"识别语言: {info.language}")
    write_transcripts(segments, txt_path, srt_path)


def main() -> None:
    configure_local_cache()
    args = parse_args()
    require_dependencies(args.transcribe)

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"输入文件不存在: {input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fmt = select_format(args.format)
    output_path = next_output_path(args.output_dir, input_path.stem, fmt.ext)

    temp_root = Path(tempfile.mkdtemp(prefix="voice_extract_"))
    try:
        source_wav = temp_root / "source_audio.wav"
        demucs_out = temp_root / "demucs"

        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                str(input_path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "2",
                "-c:a",
                "pcm_f32le",
                str(source_wav),
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "demucs.separate",
                "-n",
                args.model,
                "--two-stems",
                "vocals",
                "--float32",
                "--out",
                str(demucs_out),
                str(source_wav),
            ]
        )
        vocals = find_vocals_file(demucs_out)

        if args.transcribe:
            txt_path = output_path.with_suffix(".txt")
            srt_path = output_path.with_suffix(".srt")
            transcribe(vocals, txt_path, srt_path, args.whisper_model, args.language)
            print(f"TXT: {txt_path}")
            print(f"SRT: {srt_path}")

        encode_cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(vocals), "-vn"]
        if args.sample_rate:
            encode_cmd.extend(["-ar", args.sample_rate])
        encode_cmd.extend(fmt.ffmpeg_args)
        encode_cmd.append(str(output_path))
        run(encode_cmd)

        print(f"完成: {output_path}")
    finally:
        if args.keep_temp:
            print(f"临时文件已保留: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
