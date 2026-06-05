#!/usr/bin/env python3
"""
DeepFake Studio v1.0
Forensic Deepfake Research & Testing Platform
PyQT5 | InsightFace (roop) | SimSwap | SadTalker | FOMM

Author : GM Insider Threat Team
Purpose: Authorized forensic deepfake generation for research and detection training

Dependencies
────────────
pip install PyQt5 opencv-python-headless numpy insightface onnxruntime
Optional GPU:  pip install onnxruntime-gpu
Optional enhance: pip install gfpgan
External repos (clone separately):
  SimSwap  → https://github.com/neuralchen/SimSwap
  SadTalker → https://github.com/OpenTalker/SadTalker
  FOMM     → https://github.com/AliaksandrSiarohin/first-order-model

Model files (download separately):
  inswapper_128.onnx  → InsightFace face swapper model
  GFPGANv1.4.pth      → Optional face enhancement
"""

import sys
import os
import cv2
import json
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QFileDialog, QProgressBar,
    QComboBox, QSpinBox, QCheckBox, QGroupBox, QTextEdit,
    QSplitter, QStatusBar, QToolBar, QAction, QMessageBox,
    QLineEdit, QFormLayout, QDialog, QDialogButtonBox, QSizePolicy,
    QTableWidget, QTableWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QUrl, QMimeData
)
from PyQt5.QtGui import (
    QPixmap, QImage, QFont, QColor, QTextCursor,
    QDragEnterEvent, QDropEvent
)

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS & STYLESHEET
# ═══════════════════════════════════════════════════════════════════

APP_NAME = "DeepFake Studio"
VERSION  = "1.0.0"

# Color palette
DARK_BG      = "#1e1e2e"
PANEL_BG     = "#2a2a3e"
INPUT_BG     = "#16213e"
ACCENT       = "#7c3aed"
ACCENT_HOVER = "#6d28d9"
ACCENT2      = "#06b6d4"
TEXT         = "#e2e8f0"
TEXT_DIM     = "#94a3b8"
SUCCESS      = "#10b981"
WARNING      = "#f59e0b"
ERROR        = "#ef4444"
BORDER       = "#374151"

# File filters
IMG_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
VID_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.webm)"
AUD_FILTER = "Audio (*.wav *.mp3 *.m4a *.ogg *.flac)"
ALL_FILTER = (
    "All Media (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv *.webm);;"
    + IMG_FILTER + ";;" + VID_FILTER
)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 18px;
    padding: 10px 10px 10px 10px;
    font-weight: bold;
    color: {TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {ACCENT2};
    font-size: 13px;
}}
QPushButton {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
    min-width: 80px;
}}
QPushButton:hover  {{ background-color: {ACCENT_HOVER}; }}
QPushButton:pressed {{ background-color: #5b21b6; }}
QPushButton:disabled {{ background-color: #4b5563; color: #6b7280; }}
QPushButton#secondary {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    color: {TEXT};
}}
QPushButton#secondary:hover {{ background-color: #1e293b; }}
QPushButton#danger {{
    background-color: {ERROR};
    color: white;
}}
QPushButton#danger:hover {{ background-color: #dc2626; }}
QPushButton#small {{
    padding: 4px 10px;
    min-width: 30px;
    font-size: 12px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {PANEL_BG};
    top: -1px;
}}
QTabBar::tab {{
    background: {INPUT_BG};
    color: {TEXT_DIM};
    padding: 7px 16px;
    border-radius: 4px 4px 0 0;
    margin-right: 3px;
    font-size: 12px;
}}
QTabBar::tab:selected {{
    background: {ACCENT};
    color: white;
    font-weight: bold;
}}
QTabBar::tab:hover:!selected {{ background: #1e293b; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    color: {TEXT};
    padding: 6px 10px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; padding-right: 8px; }}
QComboBox QAbstractItemView {{
    background-color: {INPUT_BG};
    color: {TEXT};
    selection-background-color: {ACCENT};
    border: 1px solid {BORDER};
}}
QProgressBar {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    text-align: center;
    color: {TEXT};
    height: 22px;
    font-weight: bold;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT}, stop:1 {ACCENT2});
    border-radius: 4px;
}}
QTextEdit {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
QScrollBar:vertical {{
    background: {PANEL_BG};
    width: 10px;
    border-radius: 5px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QSplitter::handle {{ background: {BORDER}; width: 2px; height: 2px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {INPUT_BG};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: none;
}}
QStatusBar {{
    background: {PANEL_BG};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
    font-size: 12px;
}}
QToolBar {{
    background: {PANEL_BG};
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 4px;
}}
QToolBar QToolButton {{
    background: transparent;
    color: {TEXT};
    border-radius: 4px;
    padding: 4px 10px;
}}
QToolBar QToolButton:hover {{ background: {INPUT_BG}; }}
QLabel#header {{
    font-size: 17px;
    font-weight: bold;
    color: {ACCENT2};
    padding: 4px 0;
}}
QLabel#subheader {{
    font-size: 12px;
    color: {TEXT_DIM};
}}
QDialog {{
    background-color: {DARK_BG};
    color: {TEXT};
}}
"""


# ═══════════════════════════════════════════════════════════════════
#  BACKEND PIPELINE CLASSES
# ═══════════════════════════════════════════════════════════════════

def _cpu_args(args: tuple) -> tuple:
    """
    Redirect any cuda device argument to CPU.
    Handles: str 'cuda', str 'cuda:0',
             torch.device('cuda'), torch.device('cuda:0')
    Must be module-level so _force_cpu() closures can reference it.
    """
    if not args:
        return args
    try:
        import torch
        arg = args[0]
        if isinstance(arg, str) and "cuda" in arg:
            return ("cpu",) + args[1:]
        if isinstance(arg, torch.device) and arg.type == "cuda":
            return (torch.device("cpu"),) + args[1:]
    except Exception:
        pass
    return args


class InsightFaceBackend:
    """
    roop-style one-click face swap via InsightFace + inswapper_128.onnx.

    Setup:
        pip install insightface onnxruntime          # CPU
        pip install insightface onnxruntime-gpu      # GPU
        Download: inswapper_128.onnx from huggingface/ezioruan/inswapper_128
    """
    def __init__(self):
        self.face_app = None
        self.swapper  = None
        self.loaded   = False

    def load(self, model_path: str, use_gpu: bool = False):
        # ── Step 1: verify imports only ───────────────────────────
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            # Surface the real missing dependency, not a generic message
            raise RuntimeError(
                f"InsightFace import failed — a dependency is missing or broken.\n\n"
                f"Root cause:\n  {type(exc).__name__}: {exc}\n\n"
                f"Fix options:\n"
                f"  1) pip install insightface onnxruntime albumentations onnx scikit-learn\n"
                f"  2) Make sure the app is launched from the SAME venv where insightface is installed:\n"
                f"     .venv\\Scripts\\activate  →  python deepfake_studio.py\n"
                f"  3) Verify with: python -c \"from insightface.app import FaceAnalysis\""
            ) from exc

        # ── Step 2: load models (real errors now surface clearly) ─
        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if use_gpu else ["CPUExecutionProvider"]
            )
            self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.face_app.prepare(
                ctx_id=0 if use_gpu else -1,
                det_size=(640, 640)
            )
            # insightface 1.0.x: get_model accepts a full file path directly
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"inswapper model not found: '{model_path}'\n"
                    f"Download from: https://huggingface.co/ezioruan/inswapper_128.onnx\n"
                    f"Then set the path in the InsightFace Swap tab."
                )
            self.swapper = insightface.model_zoo.get_model(
                model_path, download=False, download_zip=False
            )
            self.swapper.prepare(ctx_id=0 if use_gpu else -1)
            self.loaded = True
        except Exception as exc:
            raise RuntimeError(
                f"InsightFace model load failed:\n\n{exc}\n\n"
                f"Checklist:\n"
                f"  • inswapper_128.onnx path is correct (use the … button)\n"
                f"  • buffalo_l downloads automatically to ~/.insightface/models/\n"
                f"  • If GPU checked but no CUDA: uncheck 'Use GPU'\n"
                f"  • onnxruntime version: {__import__('onnxruntime').__version__}"
            ) from exc

    def get_source_face(self, source_img: np.ndarray):
        """Return the largest detected face from the source image."""
        if not self.loaded:
            raise RuntimeError("InsightFace model not loaded.")
        faces = self.face_app.get(source_img)
        if not faces:
            raise ValueError("No face detected in source image.")
        return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]), reverse=True)[0]

    def swap_frame(
        self,
        frame: np.ndarray,
        source_face,
        face_index: int = -1,
        enhance: bool = True,
    ) -> np.ndarray:
        """
        Swap face(s) in `frame` with `source_face`.
        face_index -1  →  swap all detected faces.
        """
        if not self.loaded:
            raise RuntimeError("InsightFace model not loaded.")
        target_faces = self.face_app.get(frame)
        if not target_faces:
            return frame
        result = frame.copy()
        faces_to_swap = (
            [target_faces[face_index]] if 0 <= face_index < len(target_faces)
            else target_faces
        )
        for face in faces_to_swap:
            result = self.swapper.get(result, face, source_face, paste_back=True)
        if enhance:
            result = self._try_enhance(result)
        return result

    @staticmethod
    def _try_enhance(img: np.ndarray) -> np.ndarray:
        """Attempt GFPGAN face restoration; silently skip if unavailable."""
        try:
            from gfpgan import GFPGANer
            enhancer = GFPGANer(model_path="GFPGANv1.4.pth", upscale=1)
            _, _, enhanced = enhancer.enhance(img, paste_back=True)
            return enhanced
        except Exception:
            return img


class SimSwapBackend:
    """
    Identity-preserving face swap via SimSwap subprocess calls.

    Setup:
        git clone https://github.com/neuralchen/SimSwap
        cd SimSwap && pip install -r requirements.txt
        Download pretrained models per SimSwap README.
    """
    def __init__(self, simswap_dir: str = ""):
        self.dir = simswap_dir

    def _validate(self):
        if not self.dir or not os.path.isdir(self.dir):
            raise RuntimeError(
                f"SimSwap directory not found: '{self.dir}'\n"
                "Set the correct path in the SimSwap tab."
            )

    @staticmethod
    def _run(cmd, cwd):
        """Run subprocess and raise with full stderr on failure."""
        result = subprocess.run(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Subprocess exited with code {result.returncode}.\n\n"
                f"--- Output ---\n{result.stdout[-3000:]}"
            )
        return result

    def swap_image(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        crop_size: int = 224,
        use_mask: bool = True,
    ) -> str:
        self._validate()
        cmd = [
            sys.executable, "test_one_image.py",
            "--pic_a_path", source_path,
            "--pic_b_path", target_path,
            "--output_path", output_path,
            "--crop_size", str(crop_size),
        ]
        if use_mask:
            cmd.append("--use_mask")
        self._run(cmd, self.dir)
        return output_path

    def swap_video(
        self,
        source_path: str,
        target_video: str,
        output_path: str,
        crop_size: int = 224,
        use_mask: bool = True,
    ) -> str:
        self._validate()
        tmp = os.path.join(self.dir, "tmp_deepfake_studio")
        os.makedirs(tmp, exist_ok=True)
        cmd = [
            sys.executable, "test_video_swapsingle.py",
            "--pic_a_path", source_path,
            "--video_path", target_video,
            "--output_path", output_path,
            "--crop_size", str(crop_size),
            "--temp_path", tmp,
        ]
        if use_mask:
            cmd.append("--use_mask")
        self._run(cmd, self.dir)
        return output_path


class SadTalkerBackend:
    """
    Audio-driven talking head generation via SadTalker.

    Setup:
        git clone https://github.com/OpenTalker/SadTalker
        cd SadTalker && pip install -r requirements.txt
        bash scripts/download_models.sh
    """
    def __init__(self, sadtalker_dir: str = ""):
        self.dir = sadtalker_dir

    def _validate(self):
        if not self.dir or not os.path.isdir(self.dir):
            raise RuntimeError(
                f"SadTalker directory not found: '{self.dir}'\n"
                "Set the correct path in the SadTalker tab."
            )

    @staticmethod
    def _run(cmd, cwd):
        result = subprocess.run(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Subprocess exited with code {result.returncode}.\n\n"
                f"--- Output ---\n{result.stdout[-3000:]}"
            )
        return result

    def generate(
        self,
        source_image: str,
        driven_audio: str,
        output_dir: str,
        size: int = 256,
        preprocess: str = "crop",
        still: bool = False,
        enhancer: str = "gfpgan",
    ) -> str:
        self._validate()
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            sys.executable, "inference.py",
            "--driven_audio",  driven_audio,
            "--source_image",  source_image,
            "--result_dir",    output_dir,
            "--size",          str(size),
            "--preprocess",    preprocess,
            "--enhancer",      enhancer,
        ]
        if still:
            cmd.append("--still")
        self._run(cmd, self.dir)
        outputs = sorted(
            Path(output_dir).glob("*.mp4"),
            key=lambda p: p.stat().st_mtime
        )
        if not outputs:
            raise RuntimeError("SadTalker produced no output video.")
        return str(outputs[-1])


class FOMMBackend:
    """
    Video-driven face reenactment via First Order Motion Model.

    Setup:
        git clone https://github.com/AliaksandrSiarohin/first-order-model
        cd first-order-model && pip install -r requirements.txt
        Download vox-cpk.pth.tar from the FOMM releases.
    """
    def __init__(self, fomm_dir: str = ""):
        self.dir = fomm_dir

    def _validate(self):
        if not self.dir or not os.path.isdir(self.dir):
            raise RuntimeError(
                f"FOMM directory not found: '{self.dir}'\n"
                "Set the correct path in the FOMM tab."
            )

    @staticmethod
    def _run(cmd, cwd):
        result = subprocess.run(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FOMM demo.py exited with code {result.returncode}.\n\n"
                f"--- Output (last 3000 chars) ---\n{result.stdout[-3000:]}"
            )
        return result

    def animate(
        self,
        source_image: str,
        driving_video: str,
        output_path: str,
        config_path: str = "config/vox-256.yaml",
        checkpoint: str = "vox-cpk.pth.tar",
        relative: bool = True,
        adapt_scale: bool = True,
    ) -> str:
        self._validate()

        # Resolve checkpoint to absolute path — relative paths break when
        # demo.py is run with cwd=fomm_dir and the .tar is elsewhere
        ckpt_abs = checkpoint if os.path.isabs(checkpoint) else \
            str(Path(self.dir) / checkpoint)
        if not os.path.isfile(ckpt_abs):
            raise FileNotFoundError(
                f"FOMM checkpoint not found: '{ckpt_abs}'\n"
                f"Download vox-cpk.pth.tar from:\n"
                f"  https://github.com/AliaksandrSiarohin/first-order-model/releases\n"
                f"Then use the … button in the FOMM tab to point to it."
            )

        # Resolve output path to absolute — demo.py may not respect relative paths
        out_abs = str(Path(output_path).resolve())

        cmd = [
            sys.executable, "demo.py",
            "--config",        config_path,
            "--checkpoint",    ckpt_abs,
            "--source_image",  source_image,
            "--driving_video", driving_video,
            "--result_video",  out_abs,
        ]
        if relative:
            cmd.append("--relative")
        if adapt_scale:
            cmd.append("--adapt_scale")

        self._run(cmd, self.dir)
        return out_abs


# ═══════════════════════════════════════════════════════════════════
#  PROCESSING WORKER (QThread)
# ═══════════════════════════════════════════════════════════════════

class ProcessingWorker(QThread):
    """Runs the selected deepfake pipeline off the main thread."""
    progress      = pyqtSignal(int)                # 0–100
    log_msg       = pyqtSignal(str, str)           # (message, level)
    frame_preview = pyqtSignal(object)             # np.ndarray
    finished      = pyqtSignal(str)               # output path
    error         = pyqtSignal(str)               # error message

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._stop  = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            alg = self.config.get("algorithm", "insightface")
            dispatch = {
                "insightface": self._run_insightface,
                "simswap":     self._run_simswap,
                "sadtalker":   self._run_sadtalker,
                "fomm":        self._run_fomm,
            }
            fn = dispatch.get(alg)
            if fn:
                fn()
            else:
                self.error.emit(f"Unknown algorithm: {alg}")
        except Exception as exc:
            self.error.emit(str(exc))

    # ── InsightFace ───────────────────────────────────────────────

    def _run_insightface(self):
        cfg = self.config
        self.log_msg.emit("Loading InsightFace model…", "info")
        backend = InsightFaceBackend()
        backend.load(cfg["model_path"], use_gpu=cfg.get("use_gpu", False))
        self.log_msg.emit("Model loaded ✓", "success")

        src_img = cv2.imread(cfg["source_path"])
        if src_img is None:
            raise ValueError(f"Cannot read source image: {cfg['source_path']}")

        self.log_msg.emit("Extracting source face embedding…", "info")
        src_face = backend.get_source_face(src_img)
        self.log_msg.emit(
            f"Source face detected  (confidence: {src_face.det_score:.3f})",
            "success"
        )

        enhance   = cfg.get("enhance", True)
        face_idx  = cfg.get("face_index", -1)
        tgt_path  = cfg["target_path"]
        out_path  = cfg["output_path"]

        if Path(tgt_path).suffix.lower() in VIDEO_EXTS:
            self._insightface_video(backend, src_face, tgt_path, out_path, enhance, face_idx)
        else:
            self.log_msg.emit("Processing single image…", "info")
            target = cv2.imread(tgt_path)
            if target is None:
                raise ValueError(f"Cannot read target image: {tgt_path}")
            result = backend.swap_frame(target, src_face, face_idx, enhance)
            cv2.imwrite(out_path, result)
            self.frame_preview.emit(result)
            self.progress.emit(100)
            self.log_msg.emit(f"Saved → {out_path}", "success")
            self.finished.emit(out_path)

    def _insightface_video(
        self, backend, src_face, video_path, out_path, enhance, face_idx
    ):
        cap    = cv2.VideoCapture(video_path)
        total  = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        self.log_msg.emit(f"Processing {total} frames @ {fps:.1f} fps…", "info")

        for i in range(total):
            if self._stop:
                self.log_msg.emit("Processing stopped by user.", "warn")
                break
            ret, frame = cap.read()
            if not ret:
                break
            result = backend.swap_frame(frame, src_face, face_idx, enhance)
            writer.write(result)
            if i % 5 == 0:
                self.frame_preview.emit(result.copy())
            self.progress.emit(int((i + 1) / total * 100))

        cap.release()
        writer.release()
        self.log_msg.emit(f"Video saved → {out_path}", "success")
        self.finished.emit(out_path)

    # ── SimSwap ───────────────────────────────────────────────────

    def _run_simswap(self):
        cfg = self.config
        self.log_msg.emit("Starting SimSwap pipeline…", "info")
        backend = SimSwapBackend(cfg.get("simswap_dir", ""))
        src, tgt, out = cfg["source_path"], cfg["target_path"], cfg["output_path"]

        if Path(tgt).suffix.lower() in VIDEO_EXTS:
            self.log_msg.emit("SimSwap: video mode…", "info")
            backend.swap_video(src, tgt, out, cfg.get("crop_size", 224), cfg.get("use_mask", True))
        else:
            self.log_msg.emit("SimSwap: image mode…", "info")
            backend.swap_image(src, tgt, out, cfg.get("crop_size", 224), cfg.get("use_mask", True))

        self.progress.emit(100)
        self.log_msg.emit(f"SimSwap complete → {out}", "success")
        self.finished.emit(out)

    # ── SadTalker ─────────────────────────────────────────────────

    def _run_sadtalker(self):
        cfg = self.config
        self.log_msg.emit("Starting SadTalker pipeline…", "info")
        backend  = SadTalkerBackend(cfg.get("sadtalker_dir", ""))
        out_dir  = str(Path(cfg["output_path"]).parent)
        result   = backend.generate(
            source_image = cfg["source_path"],
            driven_audio = cfg.get("audio_path", ""),
            output_dir   = out_dir,
            size         = cfg.get("size", 256),
            preprocess   = cfg.get("preprocess", "crop"),
            still        = cfg.get("still_mode", False),
            enhancer     = cfg.get("enhancer", "gfpgan"),
        )
        self.progress.emit(100)
        self.log_msg.emit(f"SadTalker complete → {result}", "success")
        self.finished.emit(result)

    # ── FOMM ──────────────────────────────────────────────────────

    def _run_fomm(self):
        cfg = self.config
        self.log_msg.emit("Starting FOMM pipeline…", "info")
        backend = FOMMBackend(cfg.get("fomm_dir", ""))
        result  = backend.animate(
            source_image  = cfg["source_path"],
            driving_video = cfg["target_path"],
            output_path   = cfg["output_path"],
            config_path   = cfg.get("fomm_config", "config/vox-256.yaml"),
            checkpoint    = cfg.get("fomm_checkpoint", "vox-cpk.pth.tar"),
            relative      = cfg.get("relative", True),
            adapt_scale   = cfg.get("adapt_scale", True),
        )
        self.progress.emit(100)
        self.log_msg.emit(f"FOMM complete → {result}", "success")
        self.finished.emit(result)


# ═══════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════

# ── Drag-and-drop media zone ───────────────────────────────────────

class MediaDropZone(QLabel):
    """
    Clickable + drag-and-drop thumbnail panel.
    Shows image preview or first video frame after loading.
    """
    file_dropped = pyqtSignal(str)

    def __init__(self, placeholder: str = "Drop Media Here", images_only: bool = False):
        super().__init__()
        self._path        = ""
        self._placeholder = placeholder
        self._images_only = images_only
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(165)
        self.setMinimumWidth(190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._apply_empty_style()

    # ── State helpers ─────────────────────────────────────────────

    def _apply_empty_style(self):
        self.setText(f"🖼  {self._placeholder}\n\nDrag & Drop  or  Click to Browse")
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {BORDER};
                border-radius: 10px;
                color: {TEXT_DIM};
                background: {INPUT_BG};
                font-size: 12px;
            }}
        """)

    def _apply_loaded_style(self):
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px solid {ACCENT};
                border-radius: 10px;
                background: {INPUT_BG};
            }}
        """)

    def load_path(self, path: str):
        self._path = path
        ext = Path(path).suffix.lower()
        frame = None
        if ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
        else:
            frame = cv2.imread(path)
        if frame is not None:
            self._show_frame(frame)

    def _show_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.width() - 16, self.height() - 16,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pix)
        self._apply_loaded_style()

    def update_frame(self, frame: np.ndarray):
        """Called during live processing for live output preview."""
        self._show_frame(frame)

    def clear(self):
        self._path = ""
        self.clear()
        self._apply_empty_style()

    @property
    def path(self) -> str:
        return self._path

    # ── Drag & Drop ───────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet(f"""
                QLabel {{
                    border: 2px dashed {ACCENT};
                    border-radius: 10px;
                    background: {PANEL_BG};
                    color: {TEXT};
                }}
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, _event):
        if not self._path:
            self._apply_empty_style()
        else:
            self._apply_loaded_style()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.load_path(path)
            self.file_dropped.emit(path)

    def mousePressEvent(self, _event):
        filt = IMG_FILTER if self._images_only else ALL_FILTER
        path, _ = QFileDialog.getOpenFileName(self, "Select Media", "", filt)
        if path:
            self.load_path(path)
            self.file_dropped.emit(path)


# ── Import panel ──────────────────────────────────────────────────

class ImportPanel(QGroupBox):
    """Source face, target media, and drive audio import."""
    source_changed = pyqtSignal(str)
    target_changed = pyqtSignal(str)
    audio_changed  = pyqtSignal(str)

    def __init__(self):
        super().__init__("📂  Media Import")
        self._audio_path = ""
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setSpacing(16)

        # ── Source ────────────────────────────────────────────────
        src_col = QVBoxLayout()
        src_hdr = QLabel("Source Face / Portrait")
        src_hdr.setStyleSheet(f"color: {ACCENT2}; font-weight: bold; font-size: 12px;")
        src_hint = QLabel("InsightFace / SimSwap / SadTalker / FOMM")
        src_hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.src_drop = MediaDropZone("Source Image", images_only=True)
        self.src_drop.file_dropped.connect(self.source_changed)
        src_col.addWidget(src_hdr)
        src_col.addWidget(src_hint)
        src_col.addWidget(self.src_drop)

        # ── Target ────────────────────────────────────────────────
        tgt_col = QVBoxLayout()
        tgt_hdr = QLabel("Target Video / Image")
        tgt_hdr.setStyleSheet(f"color: {ACCENT2}; font-weight: bold; font-size: 12px;")
        tgt_hint = QLabel("InsightFace / SimSwap / FOMM driving video")
        tgt_hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.tgt_drop = MediaDropZone("Target Media", images_only=False)
        self.tgt_drop.file_dropped.connect(self.target_changed)
        tgt_col.addWidget(tgt_hdr)
        tgt_col.addWidget(tgt_hint)
        tgt_col.addWidget(self.tgt_drop)

        # ── Audio ─────────────────────────────────────────────────
        aud_col = QVBoxLayout()
        aud_hdr = QLabel("Drive Audio")
        aud_hdr.setStyleSheet(f"color: {ACCENT2}; font-weight: bold; font-size: 12px;")
        aud_hint = QLabel("SadTalker only (.wav/.mp3/.m4a)")
        aud_hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.aud_btn  = QPushButton("🎵  Browse Audio")
        self.aud_btn.setObjectName("secondary")
        self.aud_name = QLabel("No audio selected")
        self.aud_name.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.aud_name.setWordWrap(True)
        self.aud_btn.clicked.connect(self._browse_audio)
        aud_col.addWidget(aud_hdr)
        aud_col.addWidget(aud_hint)
        aud_col.addWidget(self.aud_btn)
        aud_col.addWidget(self.aud_name)
        aud_col.addStretch()

        root.addLayout(src_col, 2)
        root.addLayout(tgt_col, 2)
        root.addLayout(aud_col, 1)

    def _browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio", "", AUD_FILTER)
        if path:
            self._audio_path = path
            self.aud_name.setText(Path(path).name)
            self.audio_changed.emit(path)

    def reset(self):
        self.src_drop._path = ""
        self.src_drop._apply_empty_style()
        self.src_drop.setText(f"🖼  Source Image\n\nDrag & Drop  or  Click to Browse")
        self.tgt_drop._path = ""
        self.tgt_drop._apply_empty_style()
        self.tgt_drop.setText(f"🖼  Target Media\n\nDrag & Drop  or  Click to Browse")
        self._audio_path = ""
        self.aud_name.setText("No audio selected")

    @property
    def source_path(self) -> str:
        return self.src_drop.path

    @property
    def target_path(self) -> str:
        return self.tgt_drop.path

    @property
    def audio_path(self) -> str:
        return self._audio_path


# ── Algorithm parameter tab panels ────────────────────────────────

class InsightFaceParams(QWidget):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        form.setSpacing(12)
        form.setContentsMargins(12, 12, 12, 12)

        # Model path
        self.model_edit = QLineEdit("inswapper_128.onnx")
        self.model_btn  = QPushButton("…")
        self.model_btn.setObjectName("small")
        self.model_btn.setFixedWidth(32)
        self.model_btn.clicked.connect(self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_edit)
        model_row.addWidget(self.model_btn)

        self.face_idx = QSpinBox()
        self.face_idx.setRange(-1, 9)
        self.face_idx.setValue(-1)
        self.face_idx.setToolTip("-1 swaps all detected faces; 0 = first face, 1 = second…")

        self.use_gpu = QCheckBox("Use GPU  (requires onnxruntime-gpu + CUDA)")
        self.enhance = QCheckBox("Face Enhancement  (GFPGAN — requires GFPGANv1.4.pth)")
        self.enhance.setChecked(True)

        hint = QLabel("-1 = all detected faces  |  0, 1, 2… = specific face by size rank")
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")

        form.addRow("inswapper_128.onnx:", model_row)
        form.addRow("Face Index:", self.face_idx)
        form.addRow("", hint)
        form.addRow("", self.use_gpu)
        form.addRow("", self.enhance)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX Model", "", "ONNX (*.onnx)")
        if path:
            self.model_edit.setText(path)

    def get_config(self) -> dict:
        return {
            "algorithm":  "insightface",
            "model_path": self.model_edit.text(),
            "face_index": self.face_idx.value(),
            "use_gpu":    self.use_gpu.isChecked(),
            "enhance":    self.enhance.isChecked(),
        }


class SimSwapParams(QWidget):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        form.setSpacing(12)
        form.setContentsMargins(12, 12, 12, 12)

        self.dir_edit = QLineEdit("./SimSwap")
        self.dir_btn  = QPushButton("…")
        self.dir_btn.setObjectName("small")
        self.dir_btn.setFixedWidth(32)
        self.dir_btn.clicked.connect(self._browse_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(self.dir_btn)

        self.crop_size = QComboBox()
        self.crop_size.addItems(["224", "512"])
        self.use_mask  = QCheckBox("Use Hair / Background Mask")
        self.use_mask.setChecked(True)

        hint = QLabel("Requires: git clone https://github.com/neuralchen/SimSwap")
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")

        form.addRow("SimSwap Dir:", dir_row)
        form.addRow("Crop Size:", self.crop_size)
        form.addRow("", self.use_mask)
        form.addRow("", hint)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "SimSwap Repository Directory")
        if d:
            self.dir_edit.setText(d)

    def get_config(self) -> dict:
        return {
            "algorithm":   "simswap",
            "simswap_dir": self.dir_edit.text(),
            "crop_size":   int(self.crop_size.currentText()),
            "use_mask":    self.use_mask.isChecked(),
        }


class SadTalkerParams(QWidget):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        form.setSpacing(12)
        form.setContentsMargins(12, 12, 12, 12)

        self.dir_edit = QLineEdit("./SadTalker")
        self.dir_btn  = QPushButton("…")
        self.dir_btn.setObjectName("small")
        self.dir_btn.setFixedWidth(32)
        self.dir_btn.clicked.connect(self._browse_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(self.dir_btn)

        self.size_combo    = QComboBox()
        self.size_combo.addItems(["256", "512"])
        self.preprocess    = QComboBox()
        self.preprocess.addItems(["crop", "resize", "full", "extcrop", "extfull"])
        self.enhancer      = QComboBox()
        self.enhancer.addItems(["gfpgan", "RestoreFormer", "none"])
        self.still_mode    = QCheckBox("Still Mode  (minimize head motion)")

        hint = QLabel("Import a drive audio in the Media Import panel above.")
        hint.setStyleSheet(f"color: {WARNING}; font-size: 11px;")

        form.addRow("SadTalker Dir:", dir_row)
        form.addRow("Output Size:", self.size_combo)
        form.addRow("Preprocess:", self.preprocess)
        form.addRow("Face Enhancer:", self.enhancer)
        form.addRow("", self.still_mode)
        form.addRow("", hint)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "SadTalker Repository Directory")
        if d:
            self.dir_edit.setText(d)

    def get_config(self) -> dict:
        return {
            "algorithm":    "sadtalker",
            "sadtalker_dir": self.dir_edit.text(),
            "size":          int(self.size_combo.currentText()),
            "preprocess":    self.preprocess.currentText(),
            "enhancer":      self.enhancer.currentText(),
            "still_mode":    self.still_mode.isChecked(),
        }


class FOMMParams(QWidget):
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        form.setSpacing(12)
        form.setContentsMargins(12, 12, 12, 12)

        # FOMM dir
        self.dir_edit = QLineEdit("./first-order-model")
        self.dir_btn  = QPushButton("…")
        self.dir_btn.setObjectName("small")
        self.dir_btn.setFixedWidth(32)
        self.dir_btn.clicked.connect(self._browse_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(self.dir_btn)

        # Config
        self.cfg_edit = QLineEdit("config/vox-256.yaml")

        # Checkpoint
        self.ckpt_edit = QLineEdit("vox-cpk.pth.tar")
        self.ckpt_btn  = QPushButton("…")
        self.ckpt_btn.setObjectName("small")
        self.ckpt_btn.setFixedWidth(32)
        self.ckpt_btn.clicked.connect(self._browse_ckpt)
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self.ckpt_edit)
        ckpt_row.addWidget(self.ckpt_btn)

        self.relative    = QCheckBox("Relative Keypoints  (recommended)")
        self.relative.setChecked(True)
        self.adapt_scale = QCheckBox("Adapt Scale")
        self.adapt_scale.setChecked(True)

        hint = QLabel("Target input = driving video (not a static image).")
        hint.setStyleSheet(f"color: {WARNING}; font-size: 11px;")

        form.addRow("FOMM Dir:", dir_row)
        form.addRow("Config YAML:", self.cfg_edit)
        form.addRow("Checkpoint:", ckpt_row)
        form.addRow("", self.relative)
        form.addRow("", self.adapt_scale)
        form.addRow("", hint)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "FOMM Repository Directory")
        if d:
            self.dir_edit.setText(d)

    def _browse_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Checkpoint File", "", "Checkpoint (*.tar *.pth *.pth.tar)")
        if path:
            self.ckpt_edit.setText(path)

    def get_config(self) -> dict:
        return {
            "algorithm":        "fomm",
            "fomm_dir":         self.dir_edit.text(),
            "fomm_config":      self.cfg_edit.text(),
            "fomm_checkpoint":  self.ckpt_edit.text(),
            "relative":         self.relative.isChecked(),
            "adapt_scale":      self.adapt_scale.isChecked(),
        }


# ── Algorithm panel (tab container) ───────────────────────────────

class AlgorithmPanel(QGroupBox):
    TABS: List = [
        ("🔄  InsightFace Swap",  InsightFaceParams),
        ("🎭  SimSwap",           SimSwapParams),
        ("🗣  SadTalker",         SadTalkerParams),
        ("🎬  FOMM",              FOMMParams),
    ]

    def __init__(self):
        super().__init__("⚙️  Algorithm & Parameters")
        layout       = QVBoxLayout(self)
        self.tabs    = QTabWidget()
        self.panels: List = []
        for label, cls in self.TABS:
            panel = cls()
            self.panels.append(panel)
            self.tabs.addTab(panel, label)
        layout.addWidget(self.tabs)

    def get_config(self) -> dict:
        return self.panels[self.tabs.currentIndex()].get_config()

    @property
    def current_tab_label(self) -> str:
        return self.TABS[self.tabs.currentIndex()][0]


# ── Preview panel ─────────────────────────────────────────────────

class PreviewPanel(QGroupBox):
    """Side-by-side Source | Target | Output preview."""

    def __init__(self):
        super().__init__("🖥  Preview")
        layout = QHBoxLayout(self)
        layout.setSpacing(12)
        self._cells = {}
        for key, title in [("source", "Source"), ("target", "Target"), ("output", "Output ✨")]:
            col  = QVBoxLayout()
            hdr  = QLabel(title)
            hdr.setAlignment(Qt.AlignCenter)
            hdr.setStyleSheet(f"color: {ACCENT2}; font-weight: bold; font-size: 12px;")
            cell = QLabel("—")
            cell.setAlignment(Qt.AlignCenter)
            cell.setMinimumHeight(190)
            cell.setStyleSheet(
                f"background: {INPUT_BG}; border-radius: 6px; "
                f"color: {TEXT_DIM}; font-size: 12px;"
            )
            cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            col.addWidget(hdr)
            col.addWidget(cell)
            self._cells[key] = cell
            layout.addLayout(col)

    def set_frame(self, key: str, frame: np.ndarray):
        cell = self._cells.get(key)
        if cell is None:
            return
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            cell.width() or 300,
            cell.height() or 190,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        cell.setPixmap(pix)

    def load_path(self, key: str, path: str):
        ext = Path(path).suffix.lower()
        if ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.set_frame(key, frame)
        else:
            img = cv2.imread(path)
            if img is not None:
                self.set_frame(key, img)

    def reset(self):
        for cell in self._cells.values():
            cell.setPixmap(QPixmap())
            cell.setText("—")


# ── Export panel ──────────────────────────────────────────────────

class ExportPanel(QGroupBox):
    run_clicked  = pyqtSignal()
    stop_clicked = pyqtSignal()

    def __init__(self):
        super().__init__("💾  Export")
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # Row 1 — output path
        row1 = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Output file path — e.g. ./output/result.mp4")
        self.out_btn  = QPushButton("…")
        self.out_btn.setObjectName("secondary")
        self.out_btn.setFixedWidth(36)
        self.out_btn.clicked.connect(self._browse_output)
        row1.addWidget(QLabel("Output:"))
        row1.addWidget(self.out_edit)
        row1.addWidget(self.out_btn)
        root.addLayout(row1)

        # Row 2 — format + progress
        row2 = QHBoxLayout()
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["MP4 (H264)", "AVI", "MOV", "PNG", "JPEG"])
        self.fmt_combo.setFixedWidth(140)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        row2.addWidget(QLabel("Format:"))
        row2.addWidget(self.fmt_combo)
        row2.addWidget(self.progress, 1)
        root.addLayout(row2)

        # Row 3 — actions
        row3 = QHBoxLayout()
        self.run_btn  = QPushButton("▶  Run Pipeline")
        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_clicked)
        self.stop_btn.clicked.connect(self.stop_clicked)
        row3.addStretch()
        row3.addWidget(self.stop_btn)
        row3.addWidget(self.run_btn)
        root.addLayout(row3)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", "",
            "Video / Image (*.mp4 *.avi *.mov *.png *.jpg)"
        )
        if path:
            self.out_edit.setText(path)

    @property
    def output_path(self) -> str:
        return self.out_edit.text().strip()

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def set_progress(self, val: int):
        self.progress.setValue(val)

    def reset(self):
        self.out_edit.clear()
        self.progress.setValue(0)
        self.set_running(False)


# ── Log panel ─────────────────────────────────────────────────────

class LogPanel(QGroupBox):
    """Color-coded scrollable console log."""
    _COLORS = {
        "info":    TEXT,
        "success": SUCCESS,
        "warn":    WARNING,
        "error":   ERROR,
    }

    def __init__(self):
        super().__init__("📋  Processing Log")
        layout = QVBoxLayout(self)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(120)
        self.console.setMaximumHeight(160)

        btn_row = QHBoxLayout()
        clr_btn = QPushButton("Clear Log")
        clr_btn.setObjectName("secondary")
        clr_btn.setFixedWidth(90)
        clr_btn.clicked.connect(self.console.clear)
        sav_btn = QPushButton("Save Log")
        sav_btn.setObjectName("secondary")
        sav_btn.setFixedWidth(90)
        sav_btn.clicked.connect(self._save_log)
        btn_row.addStretch()
        btn_row.addWidget(sav_btn)
        btn_row.addWidget(clr_btn)

        layout.addWidget(self.console)
        layout.addLayout(btn_row)

    def log(self, message: str, level: str = "info"):
        color = self._COLORS.get(level, TEXT)
        ts    = datetime.now().strftime("%H:%M:%S")
        icon  = {"info": "ℹ", "success": "✔", "warn": "⚠", "error": "✖"}.get(level, "·")
        html  = (
            f'<span style="color:{TEXT_DIM};">[{ts}]</span> '
            f'<span style="color:{color};">{icon}  {message}</span>'
        )
        self.console.append(html)
        self.console.moveCursor(QTextCursor.End)

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Log", "", "Text (*.txt *.log)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.console.toPlainText())
            self.log(f"Log saved → {path}", "success")


# ═══════════════════════════════════════════════════════════════════
#  SETTINGS DIALOG
# ═══════════════════════════════════════════════════════════════════

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setSpacing(12)

        self.tmp_dir = QLineEdit(tempfile.gettempdir())
        tmp_btn = QPushButton("…")
        tmp_btn.setObjectName("small")
        tmp_btn.setFixedWidth(32)
        tmp_btn.clicked.connect(self._browse_tmp)
        tmp_row = QHBoxLayout()
        tmp_row.addWidget(self.tmp_dir)
        tmp_row.addWidget(tmp_btn)
        form.addRow("Temp Directory:", tmp_row)

        self.auto_preview = QCheckBox("Live preview update during video processing")
        self.auto_preview.setChecked(True)
        form.addRow("", self.auto_preview)

        self.preview_every = QSpinBox()
        self.preview_every.setRange(1, 30)
        self.preview_every.setValue(5)
        self.preview_every.setSuffix("  frames")
        form.addRow("Preview every:", self.preview_every)

        layout.addLayout(form)
        layout.addSpacing(12)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse_tmp(self):
        d = QFileDialog.getExistingDirectory(self, "Temp Directory")
        if d:
            self.tmp_dir.setText(d)

    def get_settings(self) -> dict:
        return {
            "tmp_dir":       self.tmp_dir.text(),
            "auto_preview":  self.auto_preview.isChecked(),
            "preview_every": self.preview_every.value(),
        }


# ═══════════════════════════════════════════════════════════════════
#  DEPENDENCY CHECKER
# ═══════════════════════════════════════════════════════════════════

class PipWorker(QThread):
    """Runs a pip install command and streams output line-by-line."""
    output = pyqtSignal(str)
    done   = pyqtSignal(bool)   # True = success

    def __init__(self, cmd: str):
        super().__init__()
        self.cmd = cmd

    def run(self):
        try:
            proc = subprocess.Popen(
                self.cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True
            )
            for line in proc.stdout:
                self.output.emit(line.rstrip())
            proc.wait()
            self.done.emit(proc.returncode == 0)
        except Exception as exc:
            self.output.emit(f"Error: {exc}")
            self.done.emit(False)


class DepCheckWorker(QThread):
    """Import-tests packages and emits per-row results."""
    result = pyqtSignal(int, str, str, str)   # row, status, version, note
    done   = pyqtSignal()

    def __init__(self, checks: list):
        super().__init__()
        self.checks = checks   # [(display, import_name, req_ver_prefix, fix_cmd), …]

    def run(self):
        for i, (_, import_name, req_ver, _) in enumerate(self.checks):
            status, version, note = self._check(import_name, req_ver)
            self.result.emit(i, status, version, note)
        self.done.emit()

    @staticmethod
    def _check(import_name: str, req_ver):
        try:
            __import__(import_name, fromlist=[""])          # handles 'moviepy.editor'
            top_ver = getattr(
                __import__(import_name.split(".")[0]),
                "__version__", "?"
            )
            if req_ver and not top_ver.startswith(req_ver):
                return "warn", top_ver, f"Need v{req_ver}x  (got {top_ver})"
            return "ok", top_ver, "✔  OK"
        except (ImportError, ModuleNotFoundError) as exc:
            return "fail", "—", str(exc).split("\n")[0]


class DependencyCheckerDialog(QDialog):
    """
    Four-tab dependency scanner with live pip-install fix runner.
    Opens from toolbar or auto-triggers before pipeline run.
    """

    # (display_name, import_name, required_version_prefix_or_None, fix_command)
    CHECKS = {
        "🔄  InsightFace": [
            ("insightface",    "insightface",    None,  "pip install insightface"),
            ("onnxruntime",    "onnxruntime",    None,  "pip install onnxruntime"),
            ("albumentations", "albumentations", "1.",  'pip install "albumentations==1.3.1"'),
            ("onnx",           "onnx",           None,  "pip install onnx"),
            ("scikit-image",   "skimage",        None,  "pip install scikit-image"),
            ("opencv",         "cv2",            None,  "pip install opencv-python-headless"),
        ],
        "🎭  SimSwap": [
            ("torch",          "torch",          None,  "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"),
            ("torchvision",    "torchvision",    None,  "pip install torchvision"),
            ("moviepy 1.x",    "moviepy.editor", "1.",  'pip install "moviepy==1.0.3" imageio-ffmpeg'),
            ("imageio-ffmpeg", "imageio_ffmpeg", None,  "pip install imageio-ffmpeg"),
            ("insightface",    "insightface",    None,  "pip install insightface"),
        ],
        "🗣  SadTalker": [
            ("torch",          "torch",          None,  "pip install torch"),
            ("face_alignment", "face_alignment", None,  "pip install face-alignment"),
            ("scipy",          "scipy",          None,  "pip install scipy"),
            ("kornia",         "kornia",         None,  "pip install kornia"),
            ("imageio",        "imageio",        None,  "pip install imageio"),
        ],
        "🎬  FOMM": [
            ("torch",          "torch",          None,  "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"),
            ("torchvision",    "torchvision",    None,  "pip install torchvision"),
            ("pyyaml",         "yaml",           None,  "pip install pyyaml"),
            ("face_alignment", "face_alignment", None,  "pip install face-alignment"),
            ("scikit-image",   "skimage",        None,  "pip install scikit-image"),
        ],
    }

    # status → (icon, colour)
    _STYLE = {
        "ok":   ("✔", SUCCESS),
        "warn": ("⚠", WARNING),
        "fail": ("✖", ERROR),
    }
    COL_ST, COL_PKG, COL_VER, COL_NOTE, COL_FIX = range(5)

    def __init__(self, highlight_alg: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔍  Dependency Checker")
        self.setMinimumSize(860, 560)
        self._workers:    dict = {}
        self._pip_worker: Optional[PipWorker] = None
        self._tables:     dict = {}
        self._build()
        # Auto-select the tab matching the active algorithm
        for i, key in enumerate(self.CHECKS):
            if highlight_alg and highlight_alg.lower() in key.lower():
                self.tabs.setCurrentIndex(i)
                break

    # ── UI construction ───────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 10)

        hdr = QLabel("🔍  Dependency Status  —  Per Algorithm")
        hdr.setStyleSheet(f"color:{ACCENT2}; font-size:14px; font-weight:bold;")
        hint = QLabel(
            "▶ Check  scans imports.   ⚠ = wrong version.   "
            "🔧 Fix  runs pip install.   📋 copies the command."
        )
        hint.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        root.addWidget(hdr)
        root.addWidget(hint)

        self.tabs = QTabWidget()
        for alg_name, checks in self.CHECKS.items():
            tab, table = self._make_tab(alg_name, checks)
            self._tables[alg_name] = table
            short = alg_name.split("  ", 1)[-1]
            self.tabs.addTab(tab, short)
        root.addWidget(self.tabs, 1)

        # pip output console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(115)
        self.console.setPlaceholderText("pip install output will appear here…")
        root.addWidget(self.console)

        # Bottom row
        btns = QHBoxLayout()
        self.check_all_btn = QPushButton("▶  Check All")
        self.check_all_btn.clicked.connect(self._check_all)
        self.fix_tab_btn = QPushButton("🔧  Fix Missing on This Tab")
        self.fix_tab_btn.setObjectName("secondary")
        self.fix_tab_btn.clicked.connect(self._fix_tab_missing)
        close_btn = QPushButton("Close")
        close_btn.setObjectName("secondary")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(self.check_all_btn)
        btns.addWidget(self.fix_tab_btn)
        btns.addStretch()
        btns.addWidget(close_btn)
        root.addLayout(btns)

    def _make_tab(self, alg_name: str, checks: list):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        table = QTableWidget(len(checks), 5)
        table.setHorizontalHeaderLabels(["", "Package", "Version", "Notes", "Fix Command"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(self.COL_ST,  32)
        table.setColumnWidth(self.COL_PKG, 115)
        table.setColumnWidth(self.COL_VER, 75)
        table.setColumnWidth(self.COL_NOTE, 210)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setStyleSheet(f"""
            QTableWidget {{
                background:{INPUT_BG}; border:1px solid {BORDER};
                border-radius:6px; gridline-color:{BORDER};
            }}
            QHeaderView::section {{
                background:{PANEL_BG}; color:{TEXT_DIM};
                padding:5px; border:none; font-size:11px;
            }}
            QTableWidget::item {{ padding:3px 6px; }}
        """)

        for row, (display, _, _, fix_cmd) in enumerate(checks):
            # Status label
            st = QLabel("·")
            st.setAlignment(Qt.AlignCenter)
            st.setStyleSheet(f"color:{TEXT_DIM}; font-size:16px;")
            table.setCellWidget(row, self.COL_ST, st)

            # Package name
            pi = QTableWidgetItem(display)
            pi.setForeground(QColor(TEXT))
            table.setItem(row, self.COL_PKG, pi)

            # Version
            vi = QTableWidgetItem("—")
            vi.setForeground(QColor(TEXT_DIM))
            vi.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, self.COL_VER, vi)

            # Notes
            ni = QTableWidgetItem("not checked")
            ni.setForeground(QColor(TEXT_DIM))
            table.setItem(row, self.COL_NOTE, ni)

            # Fix widget
            fw = QWidget()
            fw.setStyleSheet("background:transparent;")
            fl = QHBoxLayout(fw)
            fl.setContentsMargins(4, 0, 4, 0)
            fl.setSpacing(3)
            lbl = QLabel(fix_cmd)
            lbl.setStyleSheet(
                f"color:{TEXT_DIM}; font-family:Consolas,monospace; font-size:10px;"
            )
            lbl.setWordWrap(False)
            cp = QPushButton("📋")
            cp.setObjectName("small")
            cp.setFixedWidth(28)
            cp.setToolTip("Copy command to clipboard")
            cp.clicked.connect(lambda _, c=fix_cmd: QApplication.clipboard().setText(c))
            rn = QPushButton("🔧")
            rn.setObjectName("small")
            rn.setFixedWidth(28)
            rn.setToolTip("Run pip install now")
            rn.clicked.connect(lambda _, c=fix_cmd: self._run_pip(c))
            fl.addWidget(lbl, 1)
            fl.addWidget(cp)
            fl.addWidget(rn)
            table.setCellWidget(row, self.COL_FIX, fw)

        lay.addWidget(table, 1)

        chk_btn = QPushButton(
            f"▶  Check {alg_name.split('  ', 1)[-1]} Dependencies"
        )
        chk_btn.clicked.connect(
            lambda _, n=alg_name, t=table, c=checks: self._check_tab(n, t, c)
        )
        lay.addWidget(chk_btn)
        return tab, table

    # ── Check logic ───────────────────────────────────────────────

    def _check_tab(self, alg_name: str, table: QTableWidget, checks: list):
        for row in range(table.rowCount()):
            st = table.cellWidget(row, self.COL_ST)
            if st:
                st.setText("…"); st.setStyleSheet(f"color:{TEXT_DIM}; font-size:14px;")
            ni = table.item(row, self.COL_NOTE)
            if ni:
                ni.setText("checking…"); ni.setForeground(QColor(TEXT_DIM))

        w = DepCheckWorker(checks)
        w.result.connect(lambda r, s, v, n, t=table: self._update_row(t, r, s, v, n))
        w.start()
        self._workers[alg_name] = w

    def _check_all(self):
        for alg_name, checks in self.CHECKS.items():
            t = self._tables.get(alg_name)
            if t:
                self._check_tab(alg_name, t, checks)

    def _update_row(
        self, table: QTableWidget, row: int,
        status: str, version: str, note: str
    ):
        icon, color = self._STYLE.get(status, ("·", TEXT_DIM))
        st = table.cellWidget(row, self.COL_ST)
        if st:
            st.setText(icon)
            st.setStyleSheet(
                f"color:{color}; font-size:15px; font-weight:bold;"
            )
        vi = table.item(row, self.COL_VER)
        if vi:
            vi.setText(version); vi.setForeground(QColor(color))
        ni = table.item(row, self.COL_NOTE)
        if ni:
            ni.setText(note); ni.setForeground(QColor(color))

    # ── Pip fix runner ────────────────────────────────────────────

    def _run_pip(self, cmd: str):
        if self._pip_worker and self._pip_worker.isRunning():
            self.console.append("⚠  Another install is running — please wait.")
            return
        self.console.clear()
        self.console.append(f"▶  Running: {cmd}\n{'─'*60}")
        self._pip_worker = PipWorker(cmd)
        self._pip_worker.output.connect(self.console.append)
        self._pip_worker.done.connect(self._on_pip_done)
        self._pip_worker.start()
        self.fix_tab_btn.setEnabled(False)
        self.check_all_btn.setEnabled(False)

    def _on_pip_done(self, success: bool):
        if success:
            self.console.append(f"\n{'─'*60}\n✔  Install complete — click ▶ Check to verify.")
        else:
            self.console.append(f"\n{'─'*60}\n✖  Install failed — see output above.")
        self.fix_tab_btn.setEnabled(True)
        self.check_all_btn.setEnabled(True)

    def _fix_tab_missing(self):
        idx      = self.tabs.currentIndex()
        alg_name = list(self.CHECKS.keys())[idx]
        checks   = self.CHECKS[alg_name]
        table    = self._tables[alg_name]

        cmds, seen = [], set()
        for row, (_, _, _, fix_cmd) in enumerate(checks):
            st = table.cellWidget(row, self.COL_ST)
            if st and st.text() in ("✖", "⚠") and fix_cmd not in seen:
                cmds.append(fix_cmd)
                seen.add(fix_cmd)

        if not cmds:
            self.console.append(
                "ℹ  No ✖/⚠ packages found on this tab.\n"
                "   Run ▶ Check first, then retry."
            )
            return
        self._run_pip(" && ".join(cmds))


# ═══════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════

class DeepFakeStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{VERSION}  ·  Forensic Testing")
        self.setMinimumSize(1120, 860)
        self._worker:   Optional[ProcessingWorker] = None
        self._settings: dict = {
            "tmp_dir":       tempfile.gettempdir(),
            "auto_preview":  True,
            "preview_every": 5,
        }
        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()
        self._wire_signals()
        self.log("DeepFake Studio initialized — authorized forensic use only.", "info")

    # ── Build UI ──────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(10)

        # Header bar
        hdr_row = QHBoxLayout()
        hdr = QLabel(f"🎭  {APP_NAME}  —  Forensic Deepfake Research Platform")
        hdr.setObjectName("header")
        hdr.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {ACCENT2};")
        badge = QLabel("⚠  Authorized Use Only")
        badge.setStyleSheet(
            f"background: {ERROR}20; color: {ERROR}; border: 1px solid {ERROR}; "
            f"border-radius: 5px; padding: 2px 10px; font-size: 11px; font-weight: bold;"
        )
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()
        hdr_row.addWidget(badge)
        root.addLayout(hdr_row)

        # Top splitter: Import | Algorithm
        top_split = QSplitter(Qt.Horizontal)
        self.import_panel = ImportPanel()
        self.algo_panel   = AlgorithmPanel()
        top_split.addWidget(self.import_panel)
        top_split.addWidget(self.algo_panel)
        top_split.setSizes([480, 560])
        root.addWidget(top_split)

        # Preview
        self.preview_panel = PreviewPanel()
        root.addWidget(self.preview_panel)

        # Bottom row — Export + Log side by side
        bottom_split = QSplitter(Qt.Horizontal)
        bottom_split.setChildrenCollapsible(False)

        self.export_panel = ExportPanel()
        self.log_panel    = LogPanel()

        bottom_split.addWidget(self.export_panel)
        bottom_split.addWidget(self.log_panel)
        bottom_split.setSizes([480, 560])

        root.addWidget(bottom_split)

    def _build_toolbar(self):
        tb = self.addToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))

        actions = [
            ("🗒  New Session",  self._new_session),
            ("📂  Load Config",  self._load_config),
            ("💾  Save Config",  self._save_config),
            None,
            ("⚙️  Settings",    self._open_settings),
            ("🔍  Deps",        self._open_dep_checker),
            None,
            ("❓  Help",        self._show_help),
            ("📋  About",       self._show_about),
        ]
        for item in actions:
            if item is None:
                tb.addSeparator()
            else:
                label, slot = item
                act = QAction(label, self)
                act.triggered.connect(slot)
                tb.addAction(act)

    def _build_statusbar(self):
        sb = self.statusBar()
        self._status_algo = QLabel("Algorithm: InsightFace Swap")
        self._status_algo.setStyleSheet(f"color: {ACCENT2}; padding-right: 12px;")
        sb.addPermanentWidget(self._status_algo)
        sb.showMessage("Ready")

    # ── Signal Wiring ─────────────────────────────────────────────

    def _wire_signals(self):
        self.import_panel.source_changed.connect(self._on_source_changed)
        self.import_panel.target_changed.connect(self._on_target_changed)
        self.export_panel.run_clicked.connect(self._run_pipeline)
        self.export_panel.stop_clicked.connect(self._stop_pipeline)
        self.algo_panel.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_source_changed(self, path: str):
        self.preview_panel.load_path("source", path)
        self._set_status(f"Source: {Path(path).name}")

    def _on_target_changed(self, path: str):
        self.preview_panel.load_path("target", path)
        self._set_status(f"Target: {Path(path).name}")

    def _on_tab_changed(self, idx: int):
        label = self.algo_panel.TABS[idx][0].strip().split("  ", 1)[-1]
        self._status_algo.setText(f"Algorithm: {label}")

    # ── Pipeline Execution ────────────────────────────────────────

    def _run_pipeline(self):
        ip  = self.import_panel
        ep  = self.export_panel
        cfg = self.algo_panel.get_config()
        alg = cfg["algorithm"]

        # Validation
        errors = []
        if not ip.source_path:
            errors.append("• Source image is required for all algorithms.")
        if alg in ("insightface", "simswap", "fomm") and not ip.target_path:
            errors.append("• Target image/video is required for this algorithm.")
        if alg == "sadtalker" and not ip.audio_path:
            errors.append("• Drive audio file is required for SadTalker.")
        if not ep.output_path:
            errors.append("• Output file path must be set.")
        if errors:
            QMessageBox.warning(self, "Missing Inputs", "\n".join(errors))
            return

        # ── Pre-flight dependency check ───────────────────────────
        if not self._pre_check_deps(alg):
            return   # user cancelled after seeing missing deps

        # Ensure output directory exists
        out_dir = Path(ep.output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build full config
        cfg.update({
            "source_path": ip.source_path,
            "target_path": ip.target_path,
            "audio_path":  ip.audio_path,
            "output_path": ep.output_path,
        })

        # Create and start worker
        self._worker = ProcessingWorker(cfg)
        self._worker.progress.connect(ep.set_progress)
        self._worker.log_msg.connect(self.log)
        self._worker.frame_preview.connect(self._on_frame_preview)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        ep.set_running(True)
        ep.set_progress(0)
        self._set_status(f"Processing  [{alg.upper()}]…")
        self.log(f"Pipeline started  ►  {alg.upper()}  |  {Path(ep.output_path).name}", "info")
        self._worker.start()

    def _stop_pipeline(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.log("Stop signal sent — finishing current frame…", "warn")
        self.export_panel.set_running(False)
        self._set_status("Stopped.")

    def _on_frame_preview(self, frame: np.ndarray):
        if self._settings.get("auto_preview", True):
            self.preview_panel.set_frame("output", frame)

    def _on_finished(self, out_path: str):
        self.export_panel.set_running(False)
        self.export_panel.set_progress(100)
        self._set_status(f"Done  →  {out_path}")
        self.log(f"Pipeline complete  ✔  →  {out_path}", "success")
        self.preview_panel.load_path("output", out_path)
        QMessageBox.information(
            self, "Pipeline Complete",
            f"Output successfully saved:\n\n{out_path}"
        )

    def _on_error(self, message: str):
        self.export_panel.set_running(False)
        self._set_status(f"Error: {message[:60]}…")
        self.log(f"PIPELINE ERROR: {message}", "error")
        QMessageBox.critical(self, "Pipeline Error", message)

    # ── Toolbar Actions ───────────────────────────────────────────

    def _new_session(self):
        reply = QMessageBox.question(
            self, "New Session",
            "Clear all inputs, previews, and log?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.import_panel.reset()
            self.preview_panel.reset()
            self.export_panel.reset()
            self.log_panel.console.clear()
            self._set_status("New session started.")
            self.log("New session initialized.", "info")

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session Config", "", "JSON Config (*.json)"
        )
        if not path:
            return
        cfg = self.algo_panel.get_config()
        cfg.update({
            "source_path": self.import_panel.source_path,
            "target_path": self.import_panel.target_path,
            "audio_path":  self.import_panel.audio_path,
            "output_path": self.export_panel.output_path,
            "_version":    VERSION,
            "_saved":      datetime.now().isoformat(),
        })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        self.log(f"Config saved → {path}", "success")
        self._set_status(f"Config saved: {Path(path).name}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session Config", "", "JSON Config (*.json)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        # Restore media paths
        if cfg.get("source_path") and os.path.exists(cfg["source_path"]):
            self.import_panel.src_drop.load_path(cfg["source_path"])
            self.import_panel.source_changed.emit(cfg["source_path"])
        if cfg.get("target_path") and os.path.exists(cfg["target_path"]):
            self.import_panel.tgt_drop.load_path(cfg["target_path"])
            self.import_panel.target_changed.emit(cfg["target_path"])
        if cfg.get("output_path"):
            self.export_panel.out_edit.setText(cfg["output_path"])

        # Restore algorithm tab
        alg_tab = {
            "insightface": 0, "simswap": 1, "sadtalker": 2, "fomm": 3
        }.get(cfg.get("algorithm", "insightface"), 0)
        self.algo_panel.tabs.setCurrentIndex(alg_tab)

        self.log(f"Config loaded ← {Path(path).name}", "success")
        self._set_status(f"Config loaded: {Path(path).name}")

    def _open_dep_checker(self, highlight_alg: str = ""):
        """Open the Dependency Checker dialog, pre-focused on the active algorithm."""
        if not highlight_alg:
            highlight_alg = self.algo_panel.get_config().get("algorithm", "")
        dlg = DependencyCheckerDialog(highlight_alg=highlight_alg, parent=self)
        dlg.exec_()

    def _pre_check_deps(self, alg: str) -> bool:
        """
        Quick synchronous import check for critical deps of the chosen algorithm.
        Returns True if all critical deps pass (pipeline can proceed).
        Warns with a dialog listing failures but still allows the user to proceed.
        """
        # Map algorithm → critical (import_name, display_name, fix_cmd)
        CRITICAL = {
            "insightface": [
                ("insightface",   "insightface",   "pip install insightface"),
                ("onnxruntime",   "onnxruntime",   "pip install onnxruntime"),
            ],
            "simswap": [
                ("torch",         "torch",         "pip install torch"),
                ("moviepy.editor","moviepy 1.x",   'pip install "moviepy==1.0.3"'),
            ],
            "sadtalker": [
                ("torch",         "torch",         "pip install torch"),
                ("face_alignment","face_alignment","pip install face-alignment"),
            ],
            "fomm": [
                ("torch",         "torch",         "pip install torch"),
                ("torchvision",   "torchvision",   "pip install torchvision"),
            ],
        }
        checks = CRITICAL.get(alg, [])
        failures = []
        for imp, display, fix in checks:
            try:
                __import__(imp, fromlist=[""])
            except (ImportError, ModuleNotFoundError):
                failures.append(f"  ✖  {display}  →  {fix}")

        if failures:
            msg = (
                f"Missing dependencies for {alg.upper()}:\n\n"
                + "\n".join(failures)
                + "\n\nOpen 🔍 Deps in the toolbar to install them.\n\n"
                "Proceed anyway?"
            )
            reply = QMessageBox.warning(
                self, "Dependency Warning", msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            return reply == QMessageBox.Yes
        return True

    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self._settings = dlg.get_settings()
            self.log("Settings updated.", "info")

    def _show_help(self):
        QMessageBox.information(
            self, "Quick Start — DeepFake Studio",
            "WORKFLOW\n"
            "────────────────────────────────────\n"
            "1. Import source face (all algorithms)\n"
            "2. Import target video/image (InsightFace/SimSwap/FOMM)\n"
            "   — or —  import drive audio (SadTalker)\n"
            "3. Select algorithm tab and configure repo/model paths\n"
            "4. Set output file path\n"
            "5. Click  ▶ Run Pipeline\n\n"
            "ALGORITHM GUIDE\n"
            "────────────────────────────────────\n"
            "InsightFace Swap  →  fastest, one-click, needs inswapper_128.onnx\n"
            "SimSwap           →  identity-preserving, better hair/bg masking\n"
            "SadTalker         →  portrait image + audio → talking head video\n"
            "FOMM              →  source image animated by driving video\n\n"
            "SESSIONS\n"
            "────────────────────────────────────\n"
            "Use  Save/Load Config  to persist your paths between sessions.\n\n"
            "⚠  For authorized forensic testing only."
        )

    def _show_about(self):
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"{APP_NAME}  v{VERSION}\n\n"
            "Forensic Deepfake Research & Testing Platform\n"
            "GM Insider Threat Team\n\n"
            "Backends:\n"
            "  • InsightFace  (github.com/deepinsight/insightface)\n"
            "  • SimSwap      (github.com/neuralchen/SimSwap)\n"
            "  • SadTalker    (github.com/OpenTalker/SadTalker)\n"
            "  • FOMM         (github.com/AliaksandrSiarohin/first-order-model)\n\n"
            "⚠  Authorized use only. Do not distribute output media."
        )

    # ── Helpers ───────────────────────────────────────────────────

    def log(self, message: str, level: str = "info"):
        self.log_panel.log(message, level)

    def _set_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # High-DPI must be set BEFORE QApplication is created
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        pass  # Qt6 handles HiDPI automatically — attributes removed

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    window = DeepFakeStudio()
    window.show()
    sys.exit(app.exec_())