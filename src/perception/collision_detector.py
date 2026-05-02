"""
Detección visual de colisiones sin telemetría.
Usa optical flow (motion stop) + damage overlay (flash rojo en bordes).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CollisionInfo:
    """Información de colisión detectada."""

    collision_detected: bool
    method: str  # "optical_flow" | "red_flash" | "both" | "none"
    motion_magnitude: float  # magnitud promedio del flujo óptico
    red_flash_score: float  # proporción de bordes con rojo súbito
    confidence: float  # 0.0 - 1.0


class CollisionDetector:
    """
    Detecta colisiones visualmente.
    - Optical flow: caída brusca de movimiento (>70% reducción) = colisión
    - Red flash: píxeles rojos súbitos en bordes de pantalla = daño
    """

    def __init__(self, config: dict = None):
        self.prev_gray: np.ndarray | None = None
        self.motion_history: list = []
        self.motion_window = 5  # frames de historial

        # Umbrales
        self.motion_collapse_ratio = 0.30  # si el flujo cae al 30% del promedio → colisión
        self.red_flash_threshold = 0.05  # 5% de bordes rojos = daño
        self.border_width_pct = 0.10  # 10% del ancho/alto como borde

    def detect(self, frame: np.ndarray) -> CollisionInfo:
        """
        Detecta colisión en el frame actual.
        Debe llamarse una vez por frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Optical Flow ---
        motion_magnitude = 0.0
        motion_collision = False

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion_magnitude = float(np.mean(mag))

            # Historial de movimiento
            self.motion_history.append(motion_magnitude)
            if len(self.motion_history) > self.motion_window:
                self.motion_history.pop(0)

            # Colisión por colapso de flujo
            if len(self.motion_history) >= self.motion_window:
                recent_avg = np.mean(self.motion_history[-3:])
                historical_avg = np.mean(self.motion_history[:-1])
                if historical_avg > 0.5:  # solo si había movimiento antes
                    if recent_avg < historical_avg * self.motion_collapse_ratio:
                        motion_collision = True

        self.prev_gray = gray

        # --- Red Flash (bordes) ---
        red_score = self._detect_red_flash(frame)

        # --- Decisión ---
        if motion_collision and red_score > self.red_flash_threshold:
            return CollisionInfo(
                collision_detected=True,
                method="both",
                motion_magnitude=motion_magnitude,
                red_flash_score=red_score,
                confidence=min(1.0, (red_score * 10 + (1.0 if motion_collision else 0)) / 2),
            )
        elif motion_collision:
            return CollisionInfo(
                collision_detected=True,
                method="optical_flow",
                motion_magnitude=motion_magnitude,
                red_flash_score=red_score,
                confidence=0.7,
            )
        elif red_score > self.red_flash_threshold:
            return CollisionInfo(
                collision_detected=True,
                method="red_flash",
                motion_magnitude=motion_magnitude,
                red_flash_score=red_score,
                confidence=0.6,
            )

        return CollisionInfo(
            collision_detected=False,
            method="none",
            motion_magnitude=motion_magnitude,
            red_flash_score=red_score,
            confidence=0.0,
        )

    def _detect_red_flash(self, frame: np.ndarray) -> float:
        """Detecta flash rojo de daño en bordes de pantalla."""
        h, w = frame.shape[:2]
        bw = int(w * self.border_width_pct)
        bh = int(h * self.border_width_pct)

        # Extraer bordes como lista de píxeles
        border_pixels = []
        # Top
        if bh > 0:
            border_pixels.append(frame[0:bh, :].reshape(-1, 3))
        # Bottom
        if bh > 0:
            border_pixels.append(frame[h - bh : h, :].reshape(-1, 3))
        # Left
        if bw > 0 and (h - 2 * bh) > 0:
            border_pixels.append(frame[bh : h - bh, 0:bw].reshape(-1, 3))
        # Right
        if bw > 0 and (h - 2 * bh) > 0:
            border_pixels.append(frame[bh : h - bh, w - bw : w].reshape(-1, 3))

        if not border_pixels:
            return 0.0

        borders = np.concatenate(border_pixels)

        # Convertir a HSV para detectar rojo
        # reshape a (N, 1, 3) para cvtColor
        borders_reshaped = borders.reshape(-1, 1, 3).astype(np.uint8)
        hsv = cv2.cvtColor(borders_reshaped, cv2.COLOR_BGR2HSV)
        hsv = hsv.reshape(-1, 3)

        # Rojo intenso (daño)
        mask1 = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 150, 100]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)

        score = red_pixels / len(borders) if len(borders) > 0 else 0.0
        return score

    def reset(self):
        """Reinicia el historial (tras colisión resuelta)."""
        self.motion_history.clear()
        self.prev_gray = None
