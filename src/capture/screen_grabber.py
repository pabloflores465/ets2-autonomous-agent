"""
Captura de pantalla usando MSS (Multiple ScreenShots).
Captura la ventana de Parsec en macOS a 720p.
"""

import time

import mss
import numpy as np


class ScreenGrabber:
    """Captura frames de la pantalla usando MSS."""

    def __init__(self, config: dict):
        cfg = config["capture"]
        self.monitor = cfg.get("monitor", 1)
        region = cfg["region"]
        self.width = region["width"]
        self.height = region["height"]
        self.region = {
            "left": region["left"],
            "top": region["top"],
            "width": self.width,
            "height": self.height,
        }
        self._sct = mss.mss()

    def capture(self) -> np.ndarray:
        """Captura un frame y lo devuelve como array RGB (H, W, 3)."""
        img = self._sct.grab(self.region)
        # MSS devuelve BGRA → convertir a RGB
        frame = np.array(img)
        frame = frame[:, :, :3]  # quitar alpha
        frame = frame[:, :, ::-1]  # BGR → RGB
        return frame

    def capture_bgr(self) -> np.ndarray:
        """Captura en formato BGR para OpenCV."""
        img = self._sct.grab(self.region)
        frame = np.array(img)
        return frame[:, :, :3]  # BGRA → BGR

    def benchmark(self, n_frames: int = 100) -> dict:
        """Mide FPS y latencia de captura."""
        times = []
        for _ in range(n_frames):
            t0 = time.perf_counter()
            self.capture()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
        return {
            "n_frames": n_frames,
            "avg_ms": np.mean(times),
            "max_ms": np.max(times),
            "min_ms": np.min(times),
            "fps": 1000 / np.mean(times),
        }

    def close(self):
        self._sct.close()
