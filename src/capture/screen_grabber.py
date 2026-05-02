"""
Captura de pantalla usando MSS (Multiple ScreenShots).
Detecta automáticamente la ventana de ETS2 o Parsec en macOS.
Soporta resize de ventana del juego sin romperse.
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
    Returns dict con {x, y, width, height} o None.
    """
    try:
        import Quartz

        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )

        # Buscar por nombre exacto primero
        for target_name in WINDOW_NAMES:
            for w in windows:
                name = w.get("kCGWindowName", "") or ""
                owner = w.get("kCGWindowOwnerName", "") or ""
                layer = w.get("kCGWindowLayer", 0)
                bounds = w.get("kCGWindowBounds", {})

                # Solo ventanas de capa 0 (app principal)
                if layer != 0:
                    continue

                # Match por nombre de ventana o owner
                if target_name.lower() in name.lower() or target_name.lower() in owner.lower():
                    x = int(bounds.get("X", 0))
                    y = int(bounds.get("Y", 0))
                    w_px = int(bounds.get("Width", 0))
                    h_px = int(bounds.get("Height", 0))

                    # Filtrar ventanas muy pequeñas (menús, tooltips)
                    if w_px < 640 or h_px < 400:
                        continue

                    return {"left": x, "top": y, "width": w_px, "height": h_px}

    except ImportError:
        pass  # Quartz no disponible, fallback a config
    except Exception:
        pass

    return None


class ScreenGrabber:
    """
    Captura frames de la ventana del juego.
    Detecta automáticamente ETS2/Parsec o usa config como fallback.
    Actualiza posición si la ventana se mueve (resize-safe).
    """

    def __init__(self, config: dict, auto_detect: bool = True):
        cfg = config["capture"]
        self._config_region = cfg["region"]
        self._auto_detect = auto_detect
        self._sct = mss.mss()
        self._last_detect_time = 0
        self._detect_interval = 2.0  # re-detectar cada 2s

        # Detectar ventana inicial
        self.region = self._detect_or_fallback()
        self.width = self.region["width"]
        self.height = self.region["height"]

    def _detect_or_fallback(self) -> dict:
        """Detecta ventana del juego o usa config."""
        if self._auto_detect:
            window = find_game_window()
            if window:
                return window

        # Fallback a config
        return {
            "left": self._config_region["left"],
            "top": self._config_region["top"],
            "width": self._config_region["width"],
            "height": self._config_region["height"],
        }

    def _maybe_redetect(self):
        """Re-detecta ventana periódicamente para seguir resize/move."""
        now = time.time()
        if now - self._last_detect_time < self._detect_interval:
            return

        self._last_detect_time = now
        window = find_game_window()
        if window:
            # Solo actualizar si cambió
            if (
                window["left"] != self.region["left"]
                or window["top"] != self.region["top"]
                or window["width"] != self.region["width"]
                or window["height"] != self.region["height"]
            ):
                self.region = window
                self.width = window["width"]
                self.height = window["height"]

    def capture(self) -> np.ndarray:
        """Captura un frame y lo devuelve como array RGB (H, W, 3)."""
        self._maybe_redetect()
        img = self._sct.grab(self.region)
        # MSS devuelve BGRA → convertir a RGB
        frame = np.array(img)
        frame = frame[:, :, :3]  # quitar alpha
        frame = frame[:, :, ::-1]  # BGR → RGB
        return frame

    def capture_bgr(self) -> np.ndarray:
        """Captura en formato BGR para OpenCV."""
        self._maybe_redetect()
        img = self._sct.grab(self.region)
        frame = np.array(img)
        return frame[:, :, :3]  # BGRA → BGR

    def get_window_info(self) -> dict:
        """Devuelve info actual de la ventana capturada."""
        return {
            "left": self.region["left"],
            "top": self.region["top"],
            "width": self.width,
            "height": self.height,
            "auto_detected": self._auto_detect,
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
