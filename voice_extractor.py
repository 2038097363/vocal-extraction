# -*- coding: utf-8 -*-
from __future__ import annotations

import ctypes
import importlib.util
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter import font as tkfont
from tkinter.scrolledtext import ScrolledText


APP_TITLE = "视频人声提取工具"
DEFAULT_SAMPLE_RATE = "保持模型默认"
AUTO_LANGUAGE = "自动识别"


@dataclass(frozen=True)
class OutputFormat:
    label: str
    ext: str
    ffmpeg_args: tuple[str, ...]
    lossless: bool
    summary: str
    best_for: str
    note: str


FORMATS: dict[str, OutputFormat] = {
    "FLAC - 无损压缩": OutputFormat(
        "FLAC - 无损压缩",
        "flac",
        ("-c:a", "flac", "-compression_level", "8"),
        True,
        "无损压缩音频。质量接近 WAV，但文件通常更小。",
        "首选推荐：存档、后期剪辑、继续降噪或二次处理。",
        "不会再引入有损编码损失；文件比 WAV 小，但比 MP3/M4A 大。",
    ),
    "WAV - 32-bit float 无损": OutputFormat(
        "WAV - 32-bit float 无损",
        "wav",
        ("-c:a", "pcm_f32le"),
        True,
        "未压缩的 32-bit float 音频。体积最大，处理余量最好。",
        "专业后期、混音、音频编辑软件继续处理。",
        "质量保留最直接，但文件会明显更大。",
    ),
    "MP3 - 320 kbps": OutputFormat(
        "MP3 - 320 kbps",
        "mp3",
        ("-c:a", "libmp3lame", "-b:a", "320k"),
        False,
        "高码率 MP3。有损压缩，但兼容性非常好。",
        "普通播放、发给别人、兼容老设备或老软件。",
        "体积较小；不建议作为后期母版。",
    ),
    "M4A - AAC 320 kbps": OutputFormat(
        "M4A - AAC 320 kbps",
        "m4a",
        ("-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart"),
        False,
        "AAC 编码，通常同体积下比 MP3 更细腻。",
        "手机、剪映、Premiere、Final Cut、常见播放器。",
        "属于有损压缩；适合交付和日常使用。",
    ),
    "OGG - Vorbis q10": OutputFormat(
        "OGG - Vorbis q10",
        "ogg",
        ("-c:a", "libvorbis", "-q:a", "10"),
        False,
        "Vorbis 的最高质量档之一，开放格式。",
        "网页、游戏、开源工作流或不依赖 Apple/MP3 生态的场景。",
        "兼容性不如 MP3/M4A；仍然是有损压缩。",
    ),
    "OPUS - 256 kbps VBR": OutputFormat(
        "OPUS - 256 kbps VBR",
        "opus",
        ("-c:a", "libopus", "-b:a", "256k", "-vbr", "on"),
        False,
        "现代高效率编码，语音表现很好。",
        "语音内容、网络分享、需要小体积但希望音质不错的场景。",
        "部分剪辑软件兼容性一般；交给别人前建议确认能打开。",
    ),
}


VIDEO_FILETYPES = (
    ("视频文件", "*.mp4 *.mkv *.mov *.avi *.webm *.flv *.m4v *.wmv *.mpeg *.mpg *.ts"),
    ("所有文件", "*.*"),
)

LANGUAGE_OPTIONS = {
    AUTO_LANGUAGE: None,
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "韩文": "ko",
}


def enable_high_dpi() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def configure_local_cache() -> None:
    root = app_dir()
    cache_dir = root / "cache"
    models_dir = root / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("TORCH_HOME", str(models_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache_dir / "pip"))
    os.environ.setdefault("DEMUCS_CACHE", str(models_dir))
    os.environ.setdefault("HF_HOME", str(models_dir / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(models_dir / "huggingface" / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def clean_name(value: str) -> str:
    bad_chars = '<>:"/\\|?*'
    result = "".join("_" if ch in bad_chars else ch for ch in value).strip()
    return result or "output"


def srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_transcripts(segments: list[object], txt_path: Path, srt_path: Path) -> None:
    text_lines: list[str] = []
    srt_blocks: list[str] = []

    for index, segment in enumerate(segments, start=1):
        text = getattr(segment, "text", "").strip()
        if not text:
            continue
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", start))
        text_lines.append(text)
        srt_blocks.append(f"{index}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")

    txt_path.write_text("\n".join(text_lines).strip() + "\n", encoding="utf-8")
    srt_path.write_text("\n".join(srt_blocks).strip() + "\n", encoding="utf-8")


class VoiceExtractorApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1000x760")
        self.root.minsize(900, 680)

        self.input_path = StringVar()
        self.output_dir = StringVar(value=str(app_dir() / "outputs"))
        self.output_format = StringVar(value="FLAC - 无损压缩")
        self.model = StringVar(value="htdemucs")
        self.sample_rate = StringVar(value=DEFAULT_SAMPLE_RATE)
        self.keep_temp = BooleanVar(value=False)
        self.transcribe_text = BooleanVar(value=False)
        self.whisper_model = StringVar(value="base")
        self.whisper_language = StringVar(value=AUTO_LANGUAGE)
        self.is_running = False
        self.log_queue: queue.Queue[tuple[str, str | int]] = queue.Queue()

        self._configure_style()
        self._build_ui()
        self.root.after(100, self._drain_log_queue)

    def _configure_style(self) -> None:
        self.root.option_add("*Font", "{Segoe UI} 13")
        self.root.option_add("*TCombobox*Listbox.font", "{Segoe UI} 13")

        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family="Segoe UI", size=13)
            except Exception:
                pass
        try:
            tkfont.nametofont("TkHeadingFont").configure(family="Segoe UI", size=14, weight="bold")
        except Exception:
            pass

        try:
            self.root.tk.call("tk", "scaling", 1.35)
        except Exception:
            pass

        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except Exception:
            pass
        self.title_font = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self.bold_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self.button_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self.log_font = tkfont.Font(family="Consolas", size=12)
        style.configure("TFrame", background="#f6f7f9")
        style.configure("TLabel", background="#f6f7f9", foreground="#1f2933")
        style.configure("Title.TLabel", font=self.title_font, foreground="#111827")
        style.configure("Hint.TLabel", foreground="#55616f")
        style.configure("TButton", padding=(16, 10), font=self.button_font)
        style.configure("Accent.TButton", padding=(20, 12), font=self.button_font)
        style.configure("TEntry", padding=(6, 6))
        style.configure("TCombobox", padding=(6, 6))
        style.configure("TLabelframe", background="#f6f7f9")
        style.configure("TLabelframe.Label", background="#f6f7f9", foreground="#111827", font=self.bold_font)

    def _build_ui(self) -> None:
        self.root.configure(bg="#f6f7f9")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(26, 20, 26, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="打开说明文件夹", command=self.open_docs_folder).grid(row=0, column=1, sticky="e")

        content = ttk.Frame(self.root, padding=(26, 10, 26, 20))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(2, weight=1)

        form = ttk.LabelFrame(content, text="任务设置", padding=16)
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="视频文件").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Entry(form, textvariable=self.input_path).grid(row=0, column=1, sticky="ew", padx=12)
        ttk.Button(form, text="选择", command=self.choose_input).grid(row=0, column=2, sticky="e")

        ttk.Label(form, text="输出目录").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(form, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=12)
        ttk.Button(form, text="选择", command=self.choose_output_dir).grid(row=1, column=2, sticky="e")

        ttk.Label(form, text="输出格式").grid(row=2, column=0, sticky="w", pady=8)
        ttk.Combobox(form, textvariable=self.output_format, values=list(FORMATS), state="readonly").grid(
            row=2,
            column=1,
            sticky="ew",
            padx=12,
        )

        ttk.Label(form, text="分离模型").grid(row=3, column=0, sticky="w", pady=8)
        ttk.Combobox(form, textvariable=self.model, values=("htdemucs", "htdemucs_ft"), state="readonly").grid(
            row=3,
            column=1,
            sticky="ew",
            padx=12,
        )

        ttk.Label(form, text="采样率").grid(row=4, column=0, sticky="w", pady=8)
        ttk.Combobox(
            form,
            textvariable=self.sample_rate,
            values=(DEFAULT_SAMPLE_RATE, "44100", "48000"),
            state="readonly",
        ).grid(row=4, column=1, sticky="ew", padx=12)

        ttk.Checkbutton(form, text="保留临时文件", variable=self.keep_temp).grid(
            row=5,
            column=1,
            sticky="w",
            padx=12,
            pady=(10, 4),
        )

        transcript = ttk.LabelFrame(content, text="文字识别", padding=16)
        transcript.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        transcript.columnconfigure(1, weight=1)

        ttk.Checkbutton(transcript, text="同时识别人声中的文字，输出 TXT 和 SRT", variable=self.transcribe_text).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 10),
        )
        ttk.Label(transcript, text="识别模型").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(
            transcript,
            textvariable=self.whisper_model,
            values=("tiny", "base", "small", "medium"),
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=12)
        ttk.Label(transcript, text="识别语言").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Combobox(
            transcript,
            textvariable=self.whisper_language,
            values=tuple(LANGUAGE_OPTIONS),
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=12)

        action_row = ttk.Frame(transcript)
        action_row.grid(row=1, column=2, rowspan=2, sticky="se", padx=(16, 0))
        self.start_button = ttk.Button(
            action_row,
            text="开始提取人声",
            style="Accent.TButton",
            command=self.start,
        )
        self.start_button.grid(row=0, column=0, sticky="e")

        log_frame = ttk.LabelFrame(content, text="处理日志", padding=12)
        log_frame.grid(row=2, column=0, sticky="nsew", pady=(18, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(
            log_frame,
            height=13,
            wrap="word",
            state="disabled",
            font=self.log_font,
            relief="solid",
            borderwidth=1,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def open_docs_folder(self) -> None:
        docs = app_dir() / "docs"
        docs.mkdir(exist_ok=True)
        try:
            os.startfile(docs)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"无法打开说明文件夹：{exc}")

    def choose_input(self) -> None:
        path = filedialog.askopenfilename(title="选择视频文件", filetypes=VIDEO_FILETYPES)
        if path:
            self.input_path.set(path)
            if not self.output_dir.get():
                self.output_dir.set(str(Path(path).parent / "outputs"))

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir.set(path)

    def start(self) -> None:
        if self.is_running:
            return

        input_path = Path(self.input_path.get()).expanduser()
        output_dir = Path(self.output_dir.get()).expanduser()
        transcribe = self.transcribe_text.get()
        if not input_path.is_file():
            messagebox.showerror(APP_TITLE, "请先选择一个有效的视频文件。")
            return
        if not self._dependencies_ok(transcribe):
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        fmt = FORMATS[self.output_format.get()]
        output_path = self._next_output_path(output_dir, input_path.stem, fmt.ext)

        self.is_running = True
        self.start_button.configure(state="disabled")
        self._clear_log()
        self._log(f"输入视频: {input_path}")
        self._log(f"输出音频: {output_path}")
        self._log(f"输出格式: {fmt.label}")
        if transcribe:
            self._log(f"文字识别: 开启 / 模型 {self.whisper_model.get()} / 语言 {self.whisper_language.get()}")
        self._log("开始处理。耗时取决于视频长度、电脑性能和所选模型。")

        worker = threading.Thread(
            target=self._run_job,
            args=(
                input_path,
                output_path,
                fmt,
                self.model.get(),
                self.sample_rate.get(),
                transcribe,
                self.whisper_model.get(),
                LANGUAGE_OPTIONS[self.whisper_language.get()],
            ),
            daemon=True,
        )
        worker.start()

    def _dependencies_ok(self, transcribe: bool) -> bool:
        missing: list[str] = []
        if shutil.which("ffmpeg") is None:
            missing.append("ffmpeg")
        if importlib.util.find_spec("demucs") is None:
            missing.append("demucs")
        if transcribe and importlib.util.find_spec("faster_whisper") is None:
            missing.append("faster-whisper")
        if not missing:
            return True

        messagebox.showerror(
            APP_TITLE,
            "缺少运行依赖: "
            + ", ".join(missing)
            + "\n\n请先双击 Install.bat 安装环境，然后用 启动工具.vbs 启动本工具。",
        )
        return False

    def _next_output_path(self, output_dir: Path, source_stem: str, ext: str) -> Path:
        base = clean_name(source_stem) + "_vocals"
        candidate = output_dir / f"{base}.{ext}"
        index = 2
        while candidate.exists():
            candidate = output_dir / f"{base}_{index}.{ext}"
            index += 1
        return candidate

    def _run_job(
        self,
        input_path: Path,
        output_path: Path,
        fmt: OutputFormat,
        model: str,
        sample_rate: str,
        transcribe: bool,
        whisper_model: str,
        language: str | None,
    ) -> None:
        temp_root: Path | None = None
        try:
            temp_root = Path(tempfile.mkdtemp(prefix="voice_extract_"))
            source_wav = temp_root / "source_audio.wav"
            demucs_out = temp_root / "demucs"

            total_steps = 4 if transcribe else 3
            self._log(f"\n[1/{total_steps}] 正在从视频中提取高精度临时音频...")
            self._run_command(
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

            self._log(f"\n[2/{total_steps}] 正在用 Demucs 分离人声...")
            self._run_command(
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
                    str(demucs_out),
                    str(source_wav),
                ]
            )

            vocals = self._find_vocals_file(demucs_out)

            if transcribe:
                txt_path = output_path.with_suffix(".txt")
                srt_path = output_path.with_suffix(".srt")
                self._log(f"\n[3/{total_steps}] 正在识别人声文字...")
                self._transcribe(vocals, txt_path, srt_path, whisper_model, language)
                self._log(f"TXT 已输出: {txt_path}")
                self._log(f"SRT 已输出: {srt_path}")

            self._log(f"\n[{total_steps}/{total_steps}] 正在编码最终音频...")
            encode_cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(vocals), "-vn"]
            if sample_rate != DEFAULT_SAMPLE_RATE:
                encode_cmd.extend(["-ar", sample_rate])
            encode_cmd.extend(fmt.ffmpeg_args)
            encode_cmd.append(str(output_path))
            self._run_command(encode_cmd)

            if fmt.lossless:
                self._log("\n完成。已输出无损格式，人声分离阶段之外没有有损压缩。")
            else:
                self._log("\n完成。已按高码率输出有损格式；如需最大保真，请改用 FLAC 或 WAV。")
            self._log(f"音频位置: {output_path}")
            self.log_queue.put(("done", 0))
        except subprocess.CalledProcessError as exc:
            self._log(f"\n处理失败，命令退出码: {exc.returncode}")
            self.log_queue.put(("error", "处理失败，请看日志中的错误信息。"))
        except Exception as exc:
            self._log(f"\n处理失败: {exc}")
            self.log_queue.put(("error", str(exc)))
        finally:
            if temp_root and temp_root.exists():
                if self.keep_temp.get():
                    self._log(f"\n临时文件已保留: {temp_root}")
                else:
                    shutil.rmtree(temp_root, ignore_errors=True)

    def _transcribe(
        self,
        audio_path: Path,
        txt_path: Path,
        srt_path: Path,
        whisper_model: str,
        language: str | None,
    ) -> None:
        from faster_whisper import WhisperModel

        model_root = app_dir() / "models" / "whisper"
        model_root.mkdir(parents=True, exist_ok=True)
        model = WhisperModel(
            whisper_model,
            device="cpu",
            compute_type="int8",
            download_root=str(model_root),
        )
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
            beam_size=5,
        )
        segments = list(segments_iter)
        detected = getattr(info, "language", None)
        probability = getattr(info, "language_probability", None)
        if detected:
            if probability is not None:
                self._log(f"识别语言: {detected} / 置信度 {probability:.2f}")
            else:
                self._log(f"识别语言: {detected}")
        write_transcripts(segments, txt_path, srt_path)

    def _find_vocals_file(self, demucs_out: Path) -> Path:
        candidates = sorted(demucs_out.rglob("vocals.wav"))
        if not candidates:
            raise FileNotFoundError("没有找到 Demucs 输出的人声文件 vocals.wav")
        return candidates[0]

    def _run_command(self, cmd: list[str]) -> None:
        printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
        self._log(f"$ {printable}")
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            self._log(line.rstrip())
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

    def _log(self, message: str) -> None:
        self.log_queue.put(("log", message))

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", str(payload) + "\n")
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
                elif kind == "done":
                    self.is_running = False
                    self.start_button.configure(state="normal")
                    messagebox.showinfo(APP_TITLE, "处理完成。")
                elif kind == "error":
                    self.is_running = False
                    self.start_button.configure(state="normal")
                    messagebox.showerror(APP_TITLE, str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)


def main() -> None:
    enable_high_dpi()
    configure_local_cache()
    root = Tk()
    VoiceExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
