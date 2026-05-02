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
        # ROI: zona del camino, EXCLUYENDO el timón (Y < 80%)
        self.roi_top = 0.50
        self.roi_bottom = 0.78
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

        # 2. Detectar bordes del camino por TRANSICIÓN de color
        #    Escanear desde el centro hacia afuera en cada fila
        #    Donde el color cambia mucho = borde del camino
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_gray = cv2.mean(sample)[0]  # brillo promedio del camino

        left_edges = []
        right_edges = []

        for row in range(0, rh, 3):
            row_data = gray[row, :]

            # Encontrar bordes: donde el brillo se desvía > 30 del promedio del camino
            dev = np.abs(row_data.astype(np.int16) - mean_gray)
            edge_mask = (dev > 30).astype(np.uint8)

            # Encontrar el primer borde a cada lado desde el centro
            center = rw // 2
            road_left = None
            road_right = None

            # Escanear desde el centro hacia la izquierda
            for x in range(center - 1, 0, -1):
                if edge_mask[x] == 1:
                    road_left = x
                    break

            # Escanear desde el centro hacia la derecha
            for x in range(center + 1, rw - 1):
                if edge_mask[x] == 1:
                    road_right = x
                    break

            # Si hay bordes a ambos lados y la distancia es razonable
            if road_left is not None and road_right is not None:
                road_width = road_right - road_left
                if rw * 0.15 < road_width < rw * 0.85:
                    left_edges.append((road_left, row))
                    right_edges.append((road_right, row))

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
        # Dibujar región del camino entre bordes izquierdo y derecho
        if len(left_edges) > 2 and len(right_edges) > 2:
            left_pts = [(x, y) for x, y in left_edges]
            right_pts = [(x, y) for x, y in reversed(right_edges)]
            pts = np.array(left_pts + right_pts, dtype=np.int32)
            if len(pts) > 2:
                overlay = debug.copy()
                cv2.fillPoly(overlay, [pts], (0, 180, 0, 80))  # verde semitransparente
                debug = cv2.addWeighted(overlay, 0.4, debug, 0.6, 0)
        # Borde izquierdo (amarillo)
        for x, y in left_edges:
            cv2.circle(debug, (x, y), 2, (0, 255, 255), -1)
        # Borde derecho (magenta)
        for x, y in right_edges:
            cv2.circle(debug, (x, y), 2, (255, 0, 255), -1)
        # Línea del centro detectado
        cv2.line(debug, (int(road_center), 0), (int(road_center), rh), (0, 0, 255), 2)
        # Centro del frame
        cv2.line(debug, (int(frame_center), 0), (int(frame_center), rh), (255, 255, 255), 1)
        cv2.putText(debug, f"offset={offset_norm:.2f} center={road_center:.0f}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(debug, f"conf={confidence:.2f} edges={len(left_edges)}",
                    (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        self._debug_vis = debug

        return LaneInfo(
            lane_type=LaneType.PAINTED,  # usar como carril aunque sea dirt
            offset_px=offset_px,
            offset_norm=offset_norm,
            angle_deg=0.0,
            confidence=confidence,
        )
