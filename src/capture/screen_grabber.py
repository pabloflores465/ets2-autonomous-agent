"""
Captura de pantalla usando MSS (Multiple ScreenShots).
Detecta automáticamente la ventana de ETS2 o Parsec en macOS.
Si no hay ventana disponible, devuelve frame negro.
"""

import time
from typing import Optional

import mss
import numpy as np

# Nombres de ventana a buscar (prioridad)
WINDOW_NAMES = [
    "Euro Truck Simulator 2",
    "Parsec",
    "ETS2",
]


def find_game_window() -> Optional[dict]:
    """
    Busca la ventana del juego en macOS usando Quartz.
    Returns dict con {left, top, width, height} o None.
    """
    try:
        import Quartz

        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )

        for target_name in WINDOW_NAMES:
            for w in windows:
                name = w.get("kCGWindowName", "") or ""
                owner = w.get("kCGWindowOwnerName", "") or ""
                layer = w.get("kCGWindowLayer", 0)
                bounds = w.get("kCGWindowBounds", {})

                # Solo ventanas de capa 0 (app principal)
                if layer != 0:
                    continue

                if target_name.lower() in name.lower() or target_name.lower() in owner.lower():
                    x = int(bounds.get("X", 0))
                    y = int(bounds.get("Y", 0))
                    w_px = int(bounds.get("Width", 0))
                    h_px = int(bounds.get("Height", 0))

                    if w_px < 640 or h_px < 400:
                        continue

                    return {"left": x, "top": y, "width": w_px, "height": h_px}

    except ImportError:
        pass
    except Exception:
        pass

    return None


class ScreenGrabber:
    """
    Captura frames de la ventana del juego.
    Solo captura ETS2 o Parsec. Si no hay ventana, devuelve frame negro.
    """

    def __init__(self, config: dict, auto_detect: bool = True):
        cfg = config["capture"]
        self._auto_detect = auto_detect
        self._sct = mss.mss()
        self._last_detect_time = 0
        self._detect_interval = 2.0

        # Dimensiones por defecto para frame negro
        self._default_w = cfg.get("width", 1280)
        self._default_h = cfg.get("height", 720)

        # Detectar ventana inicial
        self.region: Optional[dict] = None
        self.width = self._default_w
        self.height = self._default_h
        self._detect_window()

    @property
    def has_window(self) -> bool:
        """True si hay ventana de juego/Parsec detectada."""
        return self.region is not None

    def _detect_window(self) -> bool:
        """Detecta ventana del juego. Returns True si la encontró."""
        if not self._auto_detect:
            return False

        window = find_game_window()
        if window:
            self.region = window
            self.width = window["width"]
            self.height = window["height"]
            return True

        self.region = None
        return False

    def _maybe_redetect(self):
        """Re-detecta ventana periódicamente para seguir resize/move."""
        now = time.time()
        if now - self._last_detect_time < self._detect_interval:
            return

        self._last_detect_time = now
        self._detect_window()

    def _make_black_frame(self) -> np.ndarray:
        """Devuelve frame negro en formato RGB."""
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _make_black_frame_bgr(self) -> np.ndarray:
        """Devuelve frame negro en formato BGR."""
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def capture(self) -> np.ndarray:
        """Captura un frame RGB. Negro si no hay ventana."""
        self._maybe_redetect()

        if self.region is None:
            return self._make_black_frame()

        try:
            img = self._sct.grab(self.region)
            frame = np.array(img)
            frame = frame[:, :, :3]
            frame = frame[:, :, ::-1]  # BGR → RGB
            return frame
        except Exception:
            self.region = None
            return self._make_black_frame()

    def capture_bgr(self) -> np.ndarray:
        """Captura en formato BGR. Negro si no hay ventana."""
        self._maybe_redetect()

        if self.region is None:
            return self._make_black_frame_bgr()

        try:
            img = self._sct.grab(self.region)
            frame = np.array(img)
            return frame[:, :, :3]  # BGRA → BGR
        except Exception:
            self.region = None
            return self._make_black_frame_bgr()

    def get_window_info(self) -> dict:
        """Devuelve info actual de la ventana capturada."""
        return {
            "left": self.region["left"] if self.region else 0,
            "top": self.region["top"] if self.region else 0,
            "width": self.width,
            "height": self.height,
            "auto_detected": self._auto_detect,
            "has_window": self.has_window,
        }

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
