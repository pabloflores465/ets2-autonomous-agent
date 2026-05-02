"""
Procesamiento del minimapa de ETS2.
Extrae la ruta GPS (línea roja), las flechas verdes (dirección de giro)
y la posición del camión (flecha azul).
Elementos del minimapa:
  - Flecha AZUL (grande): posición y orientación del camión → solo referencia
  - Flechas VERDES (pequeñas): waypoints de dirección → indican hacia dónde girar
  - Línea ROJA: ruta GPS a seguir
"""

from enum import Enum

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
            cfg["left_pct"],
            cfg["top_pct"],
            cfg["right_pct"],
            cfg["bottom_pct"],
        )
        # Suavizado temporal de dirección
        self._angle_history: list[float] = []
        self._max_history = 5
        # Memoria de última dirección de flechas verdes
        self._last_green_dir: GPSDirection = GPSDirection.UNKNOWN

    def process(self, frame: np.ndarray) -> tuple[GPSDirection, float, tuple[float, float]]:
        """
        Procesa el minimapa.
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

        # Posición del camión (flecha azul) — solo referencia
        truck_center = self._detect_truck_arrow(roi)

        # Detectar dirección desde flechas VERDES (prioridad alta)
        green_dir, green_int = self._detect_green_arrows(roi, truck_center)

        # Detectar dirección desde línea ROJA (fallback si no hay verdes)
        red_dir, red_int = self._analyze_route(roi)

        # Combinar: verdes tienen prioridad, rojas como respaldo
        if green_dir != GPSDirection.UNKNOWN and green_int > 0.3:
            direction = green_dir
            intensity = green_int
        elif red_dir != GPSDirection.UNKNOWN:
            direction = red_dir
            intensity = red_int
        else:
            direction = GPSDirection.UNKNOWN
            intensity = 0.0

        return direction, intensity, truck_center

    # ─── Flechas VERDES (dirección de giro) ───

    def _detect_green_arrows(self, roi: np.ndarray, truck_pos: tuple[float, float]) -> tuple[GPSDirection, float]:
        """
        Detecta las pequeñas flechas verdes que indican dirección de giro.
        Args:
            roi: región del minimapa
            truck_pos: (cx_norm, cy_norm) posición del camión
        Returns (direction, intensity).
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, w = roi.shape[:2]

        # Verde en HSV (H 35-95, amplio para capturar distintos tonos)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Limpieza
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos verdes
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GPSDirection.UNKNOWN, 0.0

        # Posición del camión en píxeles dentro de la ROI
        truck_cx = truck_pos[0] * w
        truck_cy = truck_pos[1] * h

        # Filtrar: buscar pequeños triángulos/óvalos (flechas)
        arrow_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.001 * w * h or area > 0.05 * w * h:
                continue

            moments = cv2.moments(cnt)
            if moments["m00"] == 0:
                continue

            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

            # Distancia al camión (las flechas relevantes están cerca)
            dist = np.sqrt((cx - truck_cx)**2 + (cy - truck_cy)**2)
            arrow_candidates.append((dist, area, cx, cy, cnt))

        if not arrow_candidates:
            return GPSDirection.UNKNOWN, 0.0

        # Ordenar por cercanía al camión
        arrow_candidates.sort(key=lambda c: c[0])

        # Tomar la más cercana
        _, _, cx, cy, cnt = arrow_candidates[0]

        # Encontrar la punta (punto más alejado del centroide)
        cnt_pts = cnt.reshape(-1, 2)
        if len(cnt_pts) < 5:
            return GPSDirection.UNKNOWN, 0.0

        centroid = np.mean(cnt_pts, axis=0)
        dists = np.linalg.norm(cnt_pts - centroid, axis=1)
        tip = cnt_pts[np.argmax(dists)]

        # Vector del centroide a la punta
        tip_vec = tip - centroid
        tip_angle = np.degrees(np.arctan2(tip_vec[1], tip_vec[0]))

        # En minimapa de ETS2: arriba es adelante
        # tip_angle ~0° = punta derecha → TURN_RIGHT
        # tip_angle ~±180° = punta izquierda → TURN_LEFT
        # tip_angle ~-90° = punta arriba → STRAIGHT
        direction = GPSDirection.UNKNOWN
        intensity = 0.0

        if -60 <= tip_angle <= 60:
            direction = GPSDirection.TURN_RIGHT
            intensity = min(1.0, abs(tip_angle) / 60) if abs(tip_angle) > 10 else 0.3
        elif abs(tip_angle) >= 120:
            direction = GPSDirection.TURN_LEFT
            intensity = min(1.0, (180 - abs(tip_angle)) / 60) if abs(tip_angle) < 170 else 0.3
        else:
            direction = GPSDirection.STRAIGHT
            intensity = 0.2

        self._last_green_dir = direction
        return direction, max(intensity, 0.0)

    # ─── Línea ROJA (ruta GPS) ───

    def _analyze_route(self, roi: np.ndarray) -> tuple[GPSDirection, float]:
        """
        Detecta línea roja y calcula dirección.
        Fallback cuando no hay flechas verdes.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        kernel = np.ones((3, 3), np.uint8)
        # Abrir: elimina puntos pequeños (red dot), mantiene línea fina
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Cerrar: rellena huecos en la línea
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        if cv2.countNonZero(red_mask) < 50:
            return GPSDirection.UNKNOWN, 0.0

        skeleton = self._skeletonize(red_mask)

        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GPSDirection.UNKNOWN, 0.0

        # Filtrar contornos pequeños (red dot si sobrevivió, iconos)
        large_contours = [c for c in contours if cv2.contourArea(c) > 30]
        if not large_contours:
            return GPSDirection.UNKNOWN, 0.0

        largest = max(large_contours, key=cv2.contourArea)
        if len(largest) < 5:
            return GPSDirection.UNKNOWN, 0.0

        points = largest.reshape(-1, 2).astype(np.float32)
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        principal = eigenvectors[:, np.argmax(eigenvalues)]

        angle_rad = np.arctan2(principal[1], principal[0])
        angle_deg = np.degrees(angle_rad)

        # Suavizado temporal
        self._angle_history.append(angle_deg)
        if len(self._angle_history) > self._max_history:
            self._angle_history.pop(0)
        angle_deg = np.mean(self._angle_history)

        direction = GPSDirection.UNKNOWN
        intensity = 0.0

        if -45 <= angle_deg <= 45:
            direction = GPSDirection.STRAIGHT
            intensity = 1.0 - abs(angle_deg) / 45
        elif angle_deg < -45:
            direction = GPSDirection.TURN_RIGHT
            intensity = min(1.0, (abs(angle_deg) - 45) / 45)
        else:
            direction = GPSDirection.TURN_LEFT
            intensity = min(1.0, (angle_deg - 45) / 45)

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

    # ─── Flecha AZUL (posición del camión) ───

    def _detect_truck_arrow(self, roi: np.ndarray) -> tuple[float, float]:
        """
        Detecta la flecha azul del camión en el minimapa.
        Returns (center_x_normalized, center_y_normalized).
        Solo se usa como referencia de posición, NO para dirección.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, w = roi.shape[:2]

        lower_blue = np.array([100, 80, 80])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0.5, 0.5)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            roi_area = w * h
            if 0.005 * roi_area < area < 0.12 * roi_area:
                moments = cv2.moments(cnt)
                if moments["m00"] > 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                    candidates.append((area, cx, cy))

        if candidates:
            _, cx, cy = max(candidates, key=lambda x: x[0])
            return (cx / w, cy / h)

        return (0.5, 0.5)

    def get_roi_coords(self, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        """Devuelve coordenadas de la ROI del minimapa."""
        h, w = frame_shape[:2]
        return (
            int(self.roi_pct[0] * w),
            int(self.roi_pct[1] * h),
            int(self.roi_pct[2] * w),
            int(self.roi_pct[3] * h),
        )
