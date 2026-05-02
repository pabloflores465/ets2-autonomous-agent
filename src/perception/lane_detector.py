"""
Detección de carriles mediante CV clásico (Canny + HoughLinesP).
Con fallback para caminos de terracería sin líneas pintadas.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class LaneType(Enum):
    PAINTED = "painted"  # Líneas de carril detectadas
    DIRT = "dirt"  # Terracería, sin líneas
    UNKNOWN = "unknown"


@dataclass
class LaneInfo:
    """Información de carril detectado."""

    lane_type: LaneType
    offset_px: float  # Desplazamiento lateral desde centro (px)
    offset_norm: float  # -1.0 (izq total) a 1.0 (der total)
    angle_deg: float  # Ángulo de las líneas respecto a vertical
    confidence: float  # 0.0 - 1.0
    left_line: np.ndarray | None = None
    right_line: np.ndarray | None = None


class LaneDetector:
    """
    Detecta líneas de carril usando Canny + HoughLinesP.
    Con fallback a modo terracería cuando no hay líneas.
    """

    def __init__(self, config: dict = None):
        # Parámetros de Canny
        self.canny_low = 50
        self.canny_high = 150

        # Parámetros de Hough
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 30
        self.hough_min_line_length = 60
        self.hough_max_line_gap = 40

        # Región de interés (mitad inferior del frame frontal)
        self.roi_top_pct = 0.55  # desde 55% de altura
        self.roi_bottom_pct = 0.95

        # Umbrales
        self.min_lines_for_painted = 2
        self.expected_lane_width_px = 300  # ancho típico de carril a 720p

    def detect(self, frame: np.ndarray) -> LaneInfo:
        """
        Detecta carriles en el frame frontal.
        Args:
            frame: imagen BGR completa
        Returns:
            LaneInfo con tipo de carril y offset
        """
        h, w = frame.shape[:2]

        # ROI: mitad inferior del frame (donde está la carretera)
        roi_y1 = int(h * self.roi_top_pct)
        roi_y2 = int(h * self.roi_bottom_pct)
        roi = frame[roi_y1:roi_y2, :]

        # Preprocesamiento
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )

        if lines is None or len(lines) < self.min_lines_for_painted:
            # Terracería: sin líneas
            return self._dirt_mode(w, roi_y1)

        return self._analyze_lines(lines, h, w, roi_y1)

    def _analyze_lines(
        self, lines: np.ndarray, frame_h: int, frame_w: int, roi_y1: int
    ) -> LaneInfo:
        """Clasifica líneas en izquierda/derecha y calcula offset."""
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue  # línea vertical pura, saltar
            slope = (y2 - y1) / (x2 - x1)

            # Filtrar líneas casi horizontales
            if abs(slope) < 0.4:
                continue

            # Línea izquierda: pendiente negativa (en sistema imagen)
            # Línea derecha: pendiente positiva
            (x1 + x2) / 2
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

        # Calcular línea promedio por lado
        left_avg = self._average_line(left_lines, frame_h, frame_w)
        right_avg = self._average_line(right_lines, frame_h, frame_w)

        # Calcular offset desde el centro
        offset = self._calculate_offset(left_avg, right_avg, frame_w)
        offset_norm = np.clip(offset / (frame_w / 4), -1.0, 1.0)

        # Ángulo promedio
        angles = []
        if left_avg is not None:
            dx = left_avg[2] - left_avg[0]
            dy = left_avg[3] - left_avg[1]
            if dy != 0:
                angles.append(np.degrees(np.arctan2(dx, dy)))
        if right_avg is not None:
            dx = right_avg[2] - right_avg[0]
            dy = right_avg[3] - right_avg[1]
            if dy != 0:
                angles.append(np.degrees(np.arctan2(dx, dy)))

        angle_deg = np.mean(angles) if angles else 0.0

        confidence = 1.0
        if left_avg is None or right_avg is None:
            confidence = 0.6  # solo un lado detectado

        return LaneInfo(
            lane_type=LaneType.PAINTED,
            offset_px=offset,
            offset_norm=offset_norm,
            angle_deg=angle_deg,
            confidence=confidence,
            left_line=np.array(left_avg) if left_avg is not None else None,
            right_line=np.array(right_avg) if right_avg is not None else None,
        )

    def _average_line(self, lines: list, frame_h: int, frame_w: int) -> tuple | None:
        """Promedia una lista de líneas en una sola línea extendida."""
        if not lines:
            return None

        # Extender cada línea a y=0 (top) y y=frame_h (bottom)
        x_bottom = []
        x_top = []
        for line in lines:
            x1, y1, x2, y2 = line
            if y2 == y1:
                continue
            slope = (x2 - x1) / (y2 - y1)
            intercept = x1 - slope * y1
            x_bottom.append(intercept + slope * frame_h)
            x_top.append(intercept)

        if not x_bottom:
            return None

        x1_avg = int(np.mean(x_top))
        y1_avg = 0
        x2_avg = int(np.mean(x_bottom))
        y2_avg = frame_h

        return (x1_avg, y1_avg, x2_avg, y2_avg)

    def _calculate_offset(
        self, left_line: tuple | None, right_line: tuple | None, frame_w: int
    ) -> float:
        """Calcula offset desde el centro del carril."""
        center_x = frame_w / 2

        if left_line is not None and right_line is not None:
            # Carril entre ambas líneas
            left_bottom = left_line[0]  # x en y=bottom
            right_bottom = right_line[0]
            lane_center = (left_bottom + right_bottom) / 2
            return lane_center - center_x

        elif left_line is not None:
            # Solo línea izquierda: estimar centro de carril
            left_bottom = left_line[0]
            lane_center = left_bottom + self.expected_lane_width_px / 2
            return lane_center - center_x

        elif right_line is not None:
            # Solo línea derecha
            right_bottom = right_line[0]
            lane_center = right_bottom - self.expected_lane_width_px / 2
            return lane_center - center_x

        return 0.0

    def _dirt_mode(self, frame_w: int, roi_y1: int) -> LaneInfo:
        """Modo terracería: sin líneas detectadas."""
        return LaneInfo(
            lane_type=LaneType.DIRT,
            offset_px=0.0,
            offset_norm=0.0,
            angle_deg=0.0,
            confidence=0.3,  # baja confianza
        )

    def draw_lanes(self, frame: np.ndarray, lane_info: LaneInfo) -> np.ndarray:
        """Dibuja líneas detectadas sobre el frame (debug)."""
        vis = frame.copy()
        h, w = frame.shape[:2]
        roi_y1 = int(h * self.roi_top_pct)

        if lane_info.left_line is not None:
            x1, y1, x2, y2 = lane_info.left_line
            cv2.line(vis, (x1, y1 + roi_y1), (x2, y2 + roi_y1), (255, 0, 0), 3)
        if lane_info.right_line is not None:
            x1, y1, x2, y2 = lane_info.right_line
            cv2.line(vis, (x1, y1 + roi_y1), (x2, y2 + roi_y1), (0, 0, 255), 3)

        # Centro
        center_x = w // 2
        cv2.line(vis, (center_x, h), (center_x, int(h * 0.5)), (0, 255, 255), 1)

        # Offset text
        cv2.putText(
            vis,
            f"Lane: {lane_info.lane_type.value} "
            f"offset={lane_info.offset_px:.0f}px "
            f"conf={lane_info.confidence:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )

        return vis
