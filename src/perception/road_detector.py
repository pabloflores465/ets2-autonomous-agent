"""
Detector de camino por color.
Funciona en terracería (marrón) y asfalto (gris).
Usa segmentación por color HSV para encontrar los bordes del camino.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.perception.lane_detector import LaneInfo, LaneType


class RoadDetector:
    """
    Detecta el camino por su color característico.
    No necesita líneas pintadas — funciona en terracería, grava, asfalto.
    """

    def __init__(self):
        self.roi_top = 0.55
        self.roi_bottom = 0.92
        self._debug_vis: np.ndarray | None = None

    @property
    def debug_vis(self) -> np.ndarray | None:
        return self._debug_vis

    def detect(self, frame_bgr: np.ndarray) -> LaneInfo:
        h, w = frame_bgr.shape[:2]
        y1 = int(h * self.roi_top)
        y2 = int(h * self.roi_bottom)
        roi = frame_bgr[y1:y2, :]
        rh, rw = roi.shape[:2]

        if rh < 10 or rw < 10:
            return LaneInfo(
                lane_type=LaneType.UNKNOWN,
                offset_px=0.0,
                offset_norm=0.0,
                angle_deg=0.0,
                confidence=0.0,
            )

        # 1. Color de referencia del camino: tomar una franja en el centro inferior
        sample_h = int(rh * 0.3)
        sample = roi[rh - sample_h : rh, int(rw * 0.35) : int(rw * 0.65)]
        if sample.size == 0:
            return LaneInfo(
                lane_type=LaneType.UNKNOWN,
                offset_px=0.0,
                offset_norm=0.0,
                angle_deg=0.0,
                confidence=0.0,
            )

        # Calcular color promedio y desviación del camino
        mean_color = cv2.mean(sample)[:3]  # BGR
        std_color = np.std(sample.reshape(-1, 3), axis=0)

        # 2. Crear máscara para todos los píxeles con color similar al camino
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Rango amplio para capturar distintos tipos de camino
        # Gris/asfalto: H cualquiera, S baja, V media-alta
        # Marrón/terracería: H 5-35, S 20-120, V 60-200
        lower = np.array([0, 0, 60])
        upper = np.array([180, 120, 220])
        road_mask = cv2.inRange(hsv, lower, upper)

        # También incluir por cercanía al color muestreado (en BGR)
        bgr_lower = np.array(
            [max(0, mean_color[0] - std_color[0] * 2 - 30),
             max(0, mean_color[1] - std_color[1] * 2 - 30),
             max(0, mean_color[2] - std_color[2] * 2 - 30)],
            dtype=np.uint8,
        )
        bgr_upper = np.array(
            [min(255, mean_color[0] + std_color[0] * 2 + 30),
             min(255, mean_color[1] + std_color[1] * 2 + 30),
             min(255, mean_color[2] + std_color[2] * 2 + 30)],
            dtype=np.uint8,
        )

        # Máscara por rango BGR
        bgr_mask = cv2.inRange(roi, bgr_lower, bgr_upper)

        # Combinar: el camino debe estar en ambas máscaras
        final_mask = cv2.bitwise_and(road_mask, bgr_mask)

        # Limpieza
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        # 3. Encontrar bordes izquierdo y derecho del camino
        # Para cada fila, encontrar el punto más a la izquierda y derecha
        # que pertenezca al camino
        left_edges = []
        right_edges = []

        for row in range(0, rh, 5):  # muestrear cada 5 filas
            row_data = final_mask[row, :]
            road_pixels = np.where(row_data > 0)[0]
            if len(road_pixels) > rw * 0.2:  # al menos 20% de la fila es camino
                left_edges.append((road_pixels[0], row))
                right_edges.append((road_pixels[-1], row))

        if len(left_edges) < 3:
            # No se pudo detectar el camino
            return LaneInfo(
                lane_type=LaneType.DIRT,
                offset_px=0.0,
                offset_norm=0.0,
                angle_deg=0.0,
                confidence=0.2,
            )

        # 4. Calcular centro y offset
        # Tomar las últimas filas (más cercanas al camión) para el offset
        bottom_rows = left_edges[-3:]
        bottom_left = np.mean([x for x, y in bottom_rows])
        bottom_right = np.mean([x for x, y in right_edges[-3:]])

        road_center = (bottom_left + bottom_right) / 2
        frame_center = rw / 2
        offset_px = road_center - frame_center
        offset_norm = np.clip(offset_px / (rw * 0.35), -1.0, 1.0)

        # Confianza: qué tan consistente es el ancho del camino
        if len(left_edges) > 5 and len(right_edges) > 5:
            widths = [right_edges[i][0] - left_edges[i][0] for i in range(len(left_edges))]
            width_std = np.std(widths) if widths else 1.0
            width_mean = np.mean(widths) if widths else rw * 0.5
            confidence = max(0.3, min(1.0, 1.0 - width_std / (width_mean + 1)))
        else:
            confidence = 0.3

        # ── Debug visualization ──
        debug = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
        # Mostrar máscara del camino (verde semitransparente)
        mask_colored = np.zeros_like(debug)
        mask_colored[:, :, 1] = final_mask * 80  # verde
        mask_colored[:, :, 3] = final_mask * 120  # alpha
        debug = cv2.addWeighted(debug, 1.0, mask_colored, 0.5, 0)
        # Borde izquierdo
        for x, y in left_edges:
            cv2.circle(debug, (x, y), 2, (0, 255, 255), -1)
        # Borde derecho
        for x, y in right_edges:
            cv2.circle(debug, (x, y), 2, (255, 0, 255), -1)
        # Centro detectado
        cv2.line(debug, (int(road_center), 0), (int(road_center), rh), (0, 0, 255), 2)
        # Centro del frame
        cv2.line(debug, (int(frame_center), 0), (int(frame_center), rh), (255, 255, 255), 1)
        # Texto
        cv2.putText(debug, f"road_center={road_center:.0f} offset={offset_norm:.2f}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug, f"conf={confidence:.2f} left={len(left_edges)} right={len(right_edges)}",
                    (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self._debug_vis = debug

        return LaneInfo(
            lane_type=LaneType.PAINTED,  # usar como carril aunque sea dirt
            offset_px=offset_px,
            offset_norm=offset_norm,
            angle_deg=0.0,
            confidence=confidence,
        )
