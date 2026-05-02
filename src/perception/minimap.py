"""
Procesamiento del minimapa de ETS2.
Extrae la ruta GPS (línea roja) y la orientación del camión (flecha azul).
Elementos del minimapa:
  - Flecha AZUL (grande): posición y orientación del camión
  - Flechas VERDES (pequeñas): waypoints de dirección
  - Línea ROJA: ruta GPS a seguir
"""

from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


class GPSDirection(Enum):
    STRAIGHT = "straight"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    UNKNOWN = "unknown"


class MinimapProcessor:
    """Procesa el minimapa para extraer vector GPS."""

    def __init__(self, config: dict):
        cfg = config["perception"]["minimap"]["roi"]
        self.roi_pct = (
            cfg["left_pct"], cfg["top_pct"],
            cfg["right_pct"], cfg["bottom_pct"],
        )

    def process(self, frame: np.ndarray) -> Tuple[GPSDirection, float, Tuple[float, float]]:
        """
        Procesa el minimapa.
        Args:
            frame: imagen BGR completa
        Returns:
            (GPSDirection, turn_intensity, truck_center_xy)
        """
        h, w = frame.shape[:2]
        x1 = int(self.roi_pct[0] * w)
        y1 = int(self.roi_pct[1] * h)
        x2 = int(self.roi_pct[2] * w)
        y2 = int(self.roi_pct[3] * h)

        if x2 <= x1 or y2 <= y1:
            return GPSDirection.UNKNOWN, 0.0, (0.0, 0.0)

        roi = frame[y1:y2, x1:x2]
        direction, intensity = self._analyze_route(roi)
        truck_center = self._detect_truck_arrow(roi)
        return direction, intensity, truck_center

    def _analyze_route(self, roi: np.ndarray) -> Tuple[GPSDirection, float]:
        """
        Detecta línea roja y calcula dirección.
        Steps:
        1. Filtro rojo en HSV
        2. Skeletonize
        3. Calcular vector principal
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Rojos (dos rangos en HSV por wrap-around del hue)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        # Limpieza morfológica
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        if cv2.countNonZero(red_mask) < 50:
            return GPSDirection.UNKNOWN, 0.0

        # Skeletonize
        skeleton = self._skeletonize(red_mask)

        # Encontrar contornos principales
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GPSDirection.UNKNOWN, 0.0

        # Tomar el contorno más grande
        largest = max(contours, key=cv2.contourArea)

        # Calcular vector principal (PCA simple)
        if len(largest) < 5:
            return GPSDirection.UNKNOWN, 0.0

        points = largest.reshape(-1, 2).astype(np.float32)
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        principal = eigenvectors[:, np.argmax(eigenvalues)]

        # Ángulo del vector principal
        angle_rad = np.arctan2(principal[1], principal[0])
        angle_deg = np.degrees(angle_rad)

        # Determinar dirección
        if -30 <= angle_deg <= 30:
            direction = GPSDirection.STRAIGHT
            intensity = 1.0 - abs(angle_deg) / 30
        elif angle_deg < -30:
            direction = GPSDirection.TURN_RIGHT
            intensity = min(1.0, abs(angle_deg) / 90)
        else:
            direction = GPSDirection.TURN_LEFT
            intensity = min(1.0, angle_deg / 90)

        return direction, intensity

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """Adelgaza una máscara binaria a 1px de ancho."""
        skeleton = np.zeros(mask.shape, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        size = np.size(mask)
        while not done:
            eroded = cv2.erode(mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(mask, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            mask = eroded.copy()
            zeros = size - cv2.countNonZero(mask)
            if zeros == size:
                done = True
        return skeleton

    def _detect_truck_arrow(self, roi: np.ndarray) -> Tuple[float, float]:
        """
        Detecta la flecha azul del camión en el minimapa.
        Returns:
            (center_x_normalized, center_y_normalized) relativas a la ROI
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, w = roi.shape[:2]

        # Azules en HSV (H: 100-130 en OpenCV)
        lower_blue = np.array([100, 80, 80])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Limpieza
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0.5, 0.5)  # centro por defecto

        # Buscar contorno más grande que sea razonablemente "flecha"
        # (no muy pequeño, no demasiado grande)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            roi_area = w * h
            # La flecha del camión ocupa ~2-8% del minimapa
            if 0.005 * roi_area < area < 0.12 * roi_area:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    candidates.append((area, cx, cy))

        if candidates:
            # Tomar el contorno más grande (la flecha azul)
            _, cx, cy = max(candidates, key=lambda x: x[0])
            return (cx / w, cy / h)

        return (0.5, 0.5)

    def get_roi_coords(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Devuelve coordenadas de la ROI del minimapa."""
        h, w = frame_shape[:2]
        return (
            int(self.roi_pct[0] * w),
            int(self.roi_pct[1] * h),
            int(self.roi_pct[2] * w),
            int(self.roi_pct[3] * h),
        )
