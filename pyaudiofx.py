#!/usr/bin/env python3

from __future__ import annotations

# ─── stdlib ────────────────────────────────────────────────────────────────
import importlib.util, json, sys, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import List
import threading
import time

# ─── 3rd-party ─────────────────────────────────────────────────────────────
import numpy as np
import sounddevice as sd
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QFont, QColor, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QLabel,
    QDialog,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QHBoxLayout,
    QVBoxLayout,
    QComboBox,
    QDoubleSpinBox,
    QWidget,
    QToolButton,
)

from PyQt6.Qsci import QsciScintilla, QsciLexerPython
import qdarktheme
from tqdm import tqdm
from pedalboard import Pedalboard
from pedalboard.io import AudioFile, AudioStream

# ─── constants ─────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
SETTINGS_PATH = HERE / "pyfx_state.json"
DEFAULT_WAV = HERE / "default.wav"
DEFAULT_SETTINGS = {
    "last_audio": str(DEFAULT_WAV),
    "batch_root": str(Path.home()),
    "batch_suffix": "processed",
    "batch_workers": 4,
    "tail_seconds": 0,
}
DRACULA = {
    "bg": "#282a36",
    "current": "#44475a",
    "fg": "#f8f8f2",
    "comment": "#6272a4",
    "cyan": "#8be9fd",
    "green": "#50fa7b",
    "orange": "#ffb86c",
    "pink": "#ff79c6",
    "purple": "#bd93f9",
    "yellow": "#f1fa8c",
}


def apply_dracula(editor: QsciScintilla, lexer: QsciLexerPython) -> None:
    """Colourise the editor + lexer with Dracula palette."""
    editor.setPaper(QColor(DRACULA["bg"]))
    editor.setColor(QColor(DRACULA["fg"]))
    editor.setCaretForegroundColor(QColor(DRACULA["fg"]))
    editor.setCaretWidth(2)
    editor.setCaretLineVisible(True)
    editor.setCaretLineBackgroundColor(QColor(DRACULA["current"]))
    editor.setSelectionBackgroundColor(QColor(DRACULA["current"]))
    editor.setSelectionForegroundColor(QColor(DRACULA["fg"]))

    editor.setMarginsBackgroundColor(QColor(DRACULA["bg"]))
    editor.setMarginsForegroundColor(QColor(DRACULA["comment"]))
    editor.setMarginWidth(0, "00000")  # line numbers

    lexer.setColor(QColor(DRACULA["fg"]), QsciLexerPython.Default)
    lexer.setColor(QColor(DRACULA["comment"]), QsciLexerPython.Comment)
    lexer.setColor(QColor(DRACULA["comment"]), QsciLexerPython.CommentBlock)
    lexer.setColor(QColor(DRACULA["purple"]), QsciLexerPython.Number)
    lexer.setColor(QColor(DRACULA["yellow"]), QsciLexerPython.DoubleQuotedString)
    lexer.setColor(QColor(DRACULA["yellow"]), QsciLexerPython.SingleQuotedString)
    lexer.setColor(QColor(DRACULA["yellow"]), QsciLexerPython.TripleSingleQuotedString)
    lexer.setColor(QColor(DRACULA["yellow"]), QsciLexerPython.TripleDoubleQuotedString)
    lexer.setColor(QColor(DRACULA["pink"]), QsciLexerPython.Keyword)
    lexer.setColor(QColor(DRACULA["green"]), QsciLexerPython.ClassName)
    lexer.setColor(QColor(DRACULA["green"]), QsciLexerPython.FunctionMethodName)
    lexer.setColor(QColor(DRACULA["pink"]), QsciLexerPython.Operator)
    lexer.setColor(QColor(DRACULA["orange"]), QsciLexerPython.Decorator)
    lexer.setFont(editor.font())


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def load_user_module(code: str) -> ModuleType:
    """Hot-reload user script into a module."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_user_fx.py")
    Path(tmp.name).write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("user_fx", tmp.name)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    sys.modules["user_fx"] = mod
    return mod


def load_settings() -> dict:
    try:
        return {**DEFAULT_SETTINGS, **json.loads(SETTINGS_PATH.read_text())}
    except Exception:
        return DEFAULT_SETTINGS.copy()


def save_settings(data: dict) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        tqdm.write(f"⚠️  Could not save settings: {exc}")


def extract_board(mod: ModuleType) -> Pedalboard | None:
    """
    Expect a global called **effected** that is a list/tuple of Pedalboard plugins.
    Any other style (old `board` var or `process()` function) is no longer supported.
    """
    if not hasattr(mod, "effected"):
        raise AttributeError(
            "Current script must define a global list/tuple named 'effected', e.g.\n\n"
            "from pedalboard import Reverb\n"
            "effected = [Reverb(room_size=0.6)]"
        )
    return Pedalboard(mod.effected)


def load_audio_from_path(path: str) -> np.ndarray:
    """
    Load audio from a file, ensuring it is always 2D (channels, samples).
    """
    with AudioFile(path, "r") as f:
        audio = f.read(f.frames)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=1)
        sr = f.samplerate

    return audio.T, sr


# -----------------------------------------------------------------------------
# Worker objects
# -----------------------------------------------------------------------------


class AudioWorker(QObject):
    finished = pyqtSignal(object, int)  # ndarray, sr
    error = pyqtSignal(str)

    def __init__(self, path: Path, module: ModuleType, tail_seconds: int):
        super().__init__()
        self.path = path
        self.module = module
        self.tail_seconds = tail_seconds

    def run(self):
        try:
            audio, sr = load_audio_from_path(str(self.path))
            if self.tail_seconds > 0:
                tail = np.zeros(
                    (int(self.tail_seconds * sr), audio.shape[1]), dtype=audio.dtype
                )
                audio = np.concatenate((audio, tail), axis=0)
            processed = extract_board(self.module)(audio, sr)
            self.finished.emit(np.asarray(processed), int(sr))
        except Exception as e:
            self.error.emit(str(e))


class BatchWorker(QObject):
    progress = pyqtSignal(int, int)  # done, total
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        root: Path,
        module: ModuleType,
        suffix: str,
        workers: int,
        batch_tail_seconds: int,
    ):
        super().__init__()
        self.root = root
        self.module = module
        self.suffix = suffix
        self.workers = workers
        self.batch_tail_seconds = batch_tail_seconds

    def _process_one(self, path: Path):
        try:
            audio, sr = load_audio_from_path(str(path))  # ← CHANGED
            # if tail samples, add them to the end of the audio, the shape is 42000, 2
            if self.batch_tail_seconds > 0:
                tail = np.zeros(
                    (self.batch_tail_seconds * sr, audio.shape[1]), dtype=audio.dtype
                )
                audio = np.concatenate((audio, tail), axis=0)
            board = extract_board(self.module)
            processed = board(audio, sr)
            out_path = path.with_stem(f"{path.stem}_{self.suffix}").with_suffix(".wav")
            # write with pedalboard
            with AudioFile(str(out_path), "w", sr, 2) as f:
                f.write(processed.T)
        except Exception as exc:
            tqdm.write(f"❌ {path}: {exc}")

    def run(self):
        try:
            files = [
                p
                for p in self.root.rglob("*")
                if p.suffix.lower() in {".wav", ".aiff", ".flac"}
            ]
            total = len(files)
            if total == 0:
                self.error.emit("No audio files found.")
                return

            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = {ex.submit(self._process_one, f): f for f in files}
                with tqdm(total=total, desc="Batch") as bar:
                    for fut in as_completed(futures):
                        bar.update(1)
                        done = bar.n
                        self.progress.emit(done, total)

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


# -----------------------------------------------------------------------------
# Batch dialog
# -----------------------------------------------------------------------------


class BatchDialog(QDialog):
    def __init__(self, parent: "PyFXLab"):
        super().__init__(parent)
        self.setWindowTitle("Batch Process")
        self.setModal(True)
        self.setWindowIcon(parent.windowIcon())
        self.parent = parent
        s = parent.settings

        self.root_edit = QLineEdit(str(Path.home()))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)

        root_row = QHBoxLayout()
        root_row.addWidget(self.root_edit)
        root_row.addWidget(browse_btn)

        self.suffix_edit = QLineEdit("processed")
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 64)
        self.workers_spin.setValue(4)

        # Add Tail Samples Amount
        self.tail_seconds_spin = QDoubleSpinBox()
        self.tail_seconds_spin.setRange(0, 2147483647)
        self.tail_seconds_spin.setDecimals(2)
        self.tail_seconds_spin.setValue(parent.tail_seconds)  # convert to seconds
        self.tail_seconds_spin.setSingleStep(0.01)
        self.tail_seconds_spin.setSuffix(" sec")

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._begin)

        self.root_edit.setText(s["batch_root"])
        self.suffix_edit.setText(s["batch_suffix"])
        self.workers_spin.setValue(s["batch_workers"])
        self.tail_seconds_spin.setValue(s.get("tail_seconds", 0))

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Root folder:"))
        layout.addLayout(root_row)
        layout.addWidget(QLabel("Process name suffix:"))
        layout.addWidget(self.suffix_edit)
        layout.addWidget(QLabel("Concurrent workers:"))
        layout.addWidget(self.workers_spin)
        layout.addWidget(QLabel("Tail seconds to append:"))
        layout.addWidget(self.tail_seconds_spin)
        layout.addStretch(1)
        layout.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose folder", self.root_edit.text()
        )
        if d:
            self.root_edit.setText(d)

    def _begin(self):
        root = Path(self.root_edit.text())
        if not root.is_dir():
            QMessageBox.warning(self, "Invalid folder", "Root folder does not exist.")
            return
        if self.parent.current_module is None:
            QMessageBox.warning(
                self, "Not compiled", "Compile the script before batch processing."
            )
            return

        self.start_btn.setEnabled(False)  # optional: disable UI
        self.worker_thread = QThread(self)

        self.worker = BatchWorker(
            root,
            self.parent.current_module,
            self.suffix_edit.text(),
            self.workers_spin.value(),
            int(self.tail_seconds_spin.value()),
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)

        # Close dialog *after* the thread has fully exited
        self.worker_thread.finished.connect(self.accept)

        self.worker_thread.start()

        self.parent.settings.update(
            batch_root=str(root),
            batch_suffix=self.suffix_edit.text(),
            batch_workers=self.workers_spin.value(),
            tail_seconds=self.tail_seconds_spin.value(),
        )

        # set parent tail seconds
        self.parent.tail_seconds_spin.setValue(self.tail_seconds_spin.value())

        save_settings(self.parent.settings)
        # self.accept()  # close dialog

    def _on_done(self):
        QMessageBox.information(self.parent, "Batch", "Batch processing complete.")

    def _on_error(self, msg: str):
        QMessageBox.critical(self.parent, "Batch error", msg)


class LiveEngine:
    """Manage a realtime audio stream that passes input → board → output."""

    def __init__(self):
        self.stream: AudioStream | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self, input_device_idx: int, board: Pedalboard, sr: int = 48_000):
        """
        Begin live monitoring on the chosen input device.

        Parameters
        ----------
        input_device_idx : int
            Index from sounddevice.query_devices().
        board : Pedalboard
            The effect chain to run live audio through.
        sr : int
            Desired sample-rate for the stream. 48 kHz is a safe default.
        """
        self.stop()  # shut down any existing stream first

        if input_device_idx == -1:
            return  # “None” selected in the UI – nothing to do

        input_name = AudioStream.input_device_names[input_device_idx]
        output_name = AudioStream.default_output_device_name

        # Create the AudioStream (doesn't start until __enter__ is called)
        self.stream = AudioStream(
            input_device_name=input_name,
            output_device_name=output_name,
            sample_rate=sr,
            buffer_size=512,
            plugins=board,
        )

        def _run_stream(as_instance: AudioStream):
            with as_instance:
                while as_instance.running:  # keep thread alive
                    time.sleep(0.1)  # sleep briefly to avoid busy-waiting

        self._thread = threading.Thread(
            target=_run_stream, args=(self.stream,), daemon=True
        )
        self._thread.start()
        tqdm.write("🔈 Live-monitoring started via AudioStream.")

    def stop(self):
        """Stop and clean up the live stream (if one is running)."""
        if self.stream is not None:
            self.stream.close()  # also stops if running
            self.stream = None
            tqdm.write("⏹️  Live-monitoring stopped.")
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------


class PyFXLab(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.setWindowTitle("pyAudioFX")
        self.setWindowIcon(QIcon("dticon3.png"))
        self.resize(1220, 700)

        self.audio_path: Path | None = None
        self.processed: np.ndarray | None = None
        self.sr: int = 0
        self.current_module: ModuleType | None = None  # last successful compile

        # Editor --------------------------------------------------------------
        self.editor = QsciScintilla()
        self.editor.setUtf8(True)
        font = QFont("Fira Code", 11)
        self.editor.setFont(font)

        lexer = QsciLexerPython(self.editor)
        apply_dracula(self.editor, lexer)
        self.editor.setLexer(lexer)

        self.editor.setText(
            """\
from pedalboard import Chorus, Reverb\n
effected = [Chorus(), Reverb(room_size=0.6)]\n
info = {
    "prompt": ["reverb", "a good amount of reverb", "60% reverb"] # instruct insert 'add reverb', 'remove reverb' etc
}


"""
        )

        # Top Toolbar ---------------------------------------------------------
        top_tool_bar = QToolBar(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, top_tool_bar)

        compile_act = QAction("⚙️ Compile", self)
        compile_act.triggered.connect(self.compile_and_process)
        top_tool_bar.addAction(compile_act)

        self.indicator = QLabel("🔴")  # compile status
        top_tool_bar.addWidget(self.indicator)

        # Left Toolbar --------------------------------------------------------
        left_tool_bar = QToolBar(self)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, left_tool_bar)

        self.live = LiveEngine()  # <-- keep a single engine instance

        # input-device combo
        self.input_box = QLabel(" 🎙️ Input:")
        self.device_combo = QComboBox()
        self.device_combo.addItem("None", -1)
        for idx, dev in enumerate(AudioStream.input_device_names):
            self.device_combo.addItem(f"{idx}: {dev}", idx)

        self.device_combo.setMaximumWidth(200)

        left_tool_bar.addWidget(self.input_box)
        left_tool_bar.addWidget(self.device_combo)

        self.device_combo.currentIndexChanged.connect(self._on_device_change)

        left_tool_bar.addSeparator()

        load_audio_act = QAction("📂 Load Audio…", self)
        load_audio_act.triggered.connect(self.load_audio)

        left_tool_bar.addAction(load_audio_act)

        open_script_act = QAction("📖 Open Script", self)
        save_script_act = QAction("💾 Save Script", self)
        batch_act = QAction("📊 Batch", self)
        play_dry_act = QAction("▶️ Dry", self)
        play_wet_act = QAction("▶️ Wet", self)
        stop_audio_act = QAction("⏹️ Stop", self)

        # tail seconds spinbox (label to the left of spinbox)
        tail_row = QWidget()
        tail_layout = QHBoxLayout(tail_row)
        tail_layout.setContentsMargins(0, 0, 0, 0)
        tail_layout.setSpacing(2)

        self.tail_seconds_label = QLabel("Tail Seconds:")
        self.tail_seconds_spin = QDoubleSpinBox()
        self.tail_seconds_spin.setRange(0, 2147483647)
        self.tail_seconds_spin.setDecimals(2)
        self.tail_seconds_spin.setValue(self.settings.get("tail_seconds", 0))
        self.tail_seconds_spin.setSingleStep(0.01)
        self.tail_seconds_spin.setSuffix(" sec")
        self.tail_seconds_spin.setToolTip("Tail seconds to append to processed audio")
        self.tail_seconds_spin.setMaximumWidth(100)

        tail_layout.addWidget(self.tail_seconds_label)
        tail_layout.addWidget(self.tail_seconds_spin)
        left_tool_bar.addWidget(tail_row)

        self.tail_seconds_spin.setValue(self.settings.get("tail_seconds", 0))

        # set settings on tail seconds spinbox change, set the dict key tail_seconds

        def on_tail_seconds_change(value: float):
            self.settings["tail_seconds"] = int(value)
            save_settings(self.settings)

        self.tail_seconds_spin.valueChanged.connect(on_tail_seconds_change)

        # --- Transport controls ----------------------------------------------------
        transport = QWidget()  # acts as a horizontal row
        t_layout = QHBoxLayout(transport)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(2)  # tighten spacing a little

        # Helper to turn an action into a QToolButton and wire it up quickly
        def _make_btn(action: QAction, slot):
            btn = QToolButton()
            btn.setDefaultAction(action)  # text/icon come from the QAction
            btn.setAutoRaise(True)
            btn.pressed.connect(slot)  # fire on *press* not release
            t_layout.addWidget(btn)
            return btn

        _make_btn(play_dry_act, self.play_dry_audio)
        _make_btn(play_wet_act, self.play_audio)
        _make_btn(stop_audio_act, self.stop_audio)

        # ------------- Export button ------------------------------------------------
        export_act = QAction("💾", self)
        export_act.setEnabled(False)  # stays disabled until audio processed
        export_act.triggered.connect(self.export_audio)
        self._export_act = export_act  # save so we can (de)activate later
        _make_btn(export_act, self.export_audio)

        left_tool_bar.addWidget(transport)

        left_tool_bar.addSeparator()

        batch_act.triggered.connect(self.open_batch_dialog)
        left_tool_bar.addAction(batch_act)

        # add open save that gets triggered on trigger
        open_script_act.triggered.connect(self.open_script)
        left_tool_bar.addAction(open_script_act)

        save_script_act.triggered.connect(self.save_script)
        left_tool_bar.addAction(save_script_act)

        play_dry_act.setEnabled(False)
        play_wet_act.setEnabled(False)
        batch_act.setEnabled(False)
        self._play_dry_act = play_dry_act
        self._play_wet_act = play_wet_act
        self._batch_act = batch_act

        self.setCentralWidget(self.editor)

        self._load_startup_audio()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    # ---------- Script I/O ----------
    def open_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Python script", str(Path.home()), "Python (*.py)"
        )
        if path:
            self.editor.setText(Path(path).read_text(encoding="utf-8"))
            self.indicator.setText("🔴")  # needs compile

    def save_script(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Python script", str(Path.home() / "my_fx.py"), "Python (*.py)"
        )
        if path:
            Path(path).write_text(self.editor.text(), encoding="utf-8")

    # ---------- Audio ----------
    def load_audio(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Choose audio file",
            str(Path.home()),
            "Audio files (*.wav *.aiff *.flac)",
        )
        if path_str:
            self.audio_path = Path(path_str)
            self.settings["last_audio"] = path_str
            save_settings(self.settings)
            self.statusBar().showMessage(f"Loaded {self.audio_path.name}")
            self._play_wet_act.setEnabled(False)
            self._batch_act.setEnabled(False)
            self.indicator.setText("🔴")
            self.processed = None
            self._play_dry_act.setEnabled(True)
            self._export_act.setEnabled(False)

    def stop_audio(self):
        """Stop any currently playing audio."""
        sd.stop()

    def export_audio(self):
        """Write the last processed buffer to a user-chosen .wav file."""
        if self.processed is None:
            QMessageBox.warning(
                self, "Nothing to export", "Process some audio first (⚙️ Compile)."
            )
            return

        # Suggest a sensible default file name
        suggested = self.audio_path.with_stem(
            f"{self.audio_path.stem}_processed"
        ).with_suffix(".wav")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export processed audio",
            str(suggested),
            "WAV (*.wav)",
        )
        if not out_path:
            return  # user cancelled

        # Ensure shape = (ch, N) before writing
        data = np.asarray(self.processed, dtype=np.float32).T

        try:
            with AudioFile(out_path, "w", self.sr, data.shape[0]) as f:
                f.write(data)
            QMessageBox.information(self, "Export complete", f"Saved to:\n{out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    # ---------- Compile & preview ----------
    def compile_and_process(self):
        if self.audio_path is None:
            QMessageBox.warning(self, "No Audio", "Please load an audio file first.")
            return
        try:
            module = load_user_module(self.editor.text())
            self.indicator.setText("🔴")  # processing…
        except Exception as e:
            self.indicator.setText("🔴")
            QMessageBox.critical(self, "Compile Error", str(e))
            return

        self.thread = QThread()
        self.worker = AudioWorker(
            self.audio_path, module, self.settings["tail_seconds"]
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_processed)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.start()
        self.statusBar().showMessage("Processing…")

    def on_processed(self, audio: object, sr: int):
        self.processed = audio  # type: ignore[assignment]
        self.sr = sr
        self.current_module = load_user_module(
            self.editor.text()
        )  # save compiled module
        self.indicator.setText("🟢")
        self._play_wet_act.setEnabled(True)
        self._play_dry_act.setEnabled(True)
        self._batch_act.setEnabled(True)
        self._export_act.setEnabled(True)
        if self.device_combo.currentData() != -1:  # live mode is on → restart
            self.live.start(
                self.device_combo.currentData(), extract_board(self.current_module)
            )
        self.statusBar().showMessage(
            f"Processing complete – ready to play: {self.audio_path.name}"
        )

    def on_error(self, msg: str):
        self.indicator.setText("🔴")
        QMessageBox.critical(self, "Processing Error", msg)
        self._export_act.setEnabled(False)
        self.statusBar().clearMessage()

    # ---------- Batch ----------
    def open_batch_dialog(self):
        dlg = BatchDialog(self)
        dlg.exec()

    def _update_batch_status(self, done: int, total: int):
        self.statusBar().showMessage(f"Batch: {done}/{total} files done")

    def play_audio(self):
        if self.processed is None:
            return
        # Ensure audio is in correct format for sounddevice
        audio = np.asarray(self.processed, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=1)
        elif audio.ndim > 1 and audio.shape[0] < audio.shape[1]:
            # If channels are first dimension, transpose to (samples, channels)
            audio = audio.T

        # Clip to prevent distortion
        audio = np.clip(audio, -1.0, 1.0)

        # Play asynchronously
        sd.play(audio, samplerate=self.sr)

    def play_dry_audio(self):
        if self.audio_path is None:
            return
        try:
            audio, sr = load_audio_from_path(str(self.audio_path))  # ← CHANGED
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=1)
            sd.play(np.clip(audio, -1.0, 1.0), samplerate=sr)
        except Exception as e:
            QMessageBox.warning(self, "Playback Error", f"Could not play audio: {e}")

    def _load_startup_audio(self) -> None:
        cand = Path(self.settings["last_audio"])
        if cand.is_file():
            self.audio_path = cand
        elif DEFAULT_WAV.is_file():
            self.audio_path = DEFAULT_WAV
            self.settings["last_audio"] = str(DEFAULT_WAV)
        else:
            self.audio_path = None
        if self.audio_path:
            self.statusBar().showMessage(f"Loaded {self.audio_path.name}")
            self._play_dry_act.setEnabled(True)

    def _on_device_change(self, _idx: int):
        dev_idx = self.device_combo.currentData()
        if dev_idx == -1:  # "None" selected
            self.live.stop()
            return

        if self.current_module is None:
            QMessageBox.information(
                self, "No board yet", "Compile a script first, then enable live input."
            )
            self.device_combo.setCurrentIndex(0)  # back to "None"
            return

        board = extract_board(self.current_module)
        if board is None:
            QMessageBox.warning(
                self, "No board", "Current script doesn't define 'effected' or 'board'."
            )
            self.device_combo.setCurrentIndex(0)
            return

        self.live.start(dev_idx, board, sr=48000)


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------


def main():
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    win = PyFXLab()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
