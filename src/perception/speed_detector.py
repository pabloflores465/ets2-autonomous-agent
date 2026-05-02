"""
Estimación de velocidad del camión.
Método primario: OCR del velocímetro digital en el tablero.
Método secundario: optical flow (magnitud promedio entre frames).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SpeedInfo:
    """Información de velocidad estimada."""

    speed_kmh: float  # velocidad en km/h
    method: str  # "ocr" | "optical_flow" | "none"
    confidence: float  # 0.0 - 1.0


class SpeedDetector:
    """
    Estima velocidad del camión por OCR del tablero digital.
    Fallback: optical flow.
    """

    def __init__(self, config: dict = None):
        # ROI del velocímetro digital (centro-inferior del frame)
        # En cabina ETS2, el display digital está abajo al centro
        self.roi_x1_pct = 0.40
        self.roi_y1_pct = 0.82
        self.roi_x2_pct = 0.60
        self.roi_y2_pct = 0.92

        # Optical flow (fallback)
        self.prev_gray: np.ndarray | None = None
        self.flow_scale = 50.0  # factor de conversión flujo → km/h (requiere calibración)
        self.flow_history = []
        self.flow_window = 5

    def detect(self, frame_bgr: np.ndarray) -> SpeedInfo:
        """
        Estima velocidad actual.
        """
        # Intentar OCR del tablero primero
        speed_ocr = self._ocr_speed(frame_bgr)
        if speed_ocr is not None:
            return SpeedInfo(speed_kmh=speed_ocr, method="ocr", confidence=0.8)

        # Fallback: optical flow
        speed_flow = self._optical_flow_speed(frame_bgr)
        if speed_flow is not None:
            return SpeedInfo(speed_kmh=speed_flow, method="optical_flow", confidence=0.5)

        return SpeedInfo(speed_kmh=0.0, method="none", confidence=0.0)

    def _ocr_speed(self, frame_bgr: np.ndarray) -> float | None:
        """
        Detecta dígitos del velocímetro digital usando CV simple.
        El display LCD de ETS2 muestra números blancos/grises sobre fondo oscuro.
        """
        h, w = frame_bgr.shape[:2]
        x1 = int(w * self.roi_x1_pct)
        y1 = int(h * self.roi_y1_pct)
        x2 = int(w * self.roi_x2_pct)
        y2 = int(h * self.roi_y2_pct)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame_bgr[y1:y2, x1:x2]

        # Convertir a escala de grises
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Binarizar: asumimos dígitos claros sobre fondo oscuro
        # Umbral adaptativo
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )

        # Invertir si necesario (queremos texto blanco sobre negro)
        if np.mean(thresh) > 127:
            thresh = 255 - thresh

        # Encontrar contornos que podrían ser dígitos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos con forma de dígito (alto > ancho)
        digit_candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = ch / cw if cw > 0 else 0
            area = cw * ch
            (x2 - x1) * (y2 - y1)
            # Dígito: alto 30-80% de ROI, aspect ratio ~1.5-3.0
            if 0.15 * (y2 - y1) < ch < 0.8 * (y2 - y1) and 1.2 < aspect_ratio < 4.0 and area > 50:
                digit_candidates.append((x, cnt))

        if not digit_candidates:
            return None

        # Ordenar de izquierda a derecha
        digit_candidates.sort(key=lambda c: c[0])

        # Template matching con dígitos de referencia
        digits_str = self._match_digits(thresh, digit_candidates)
        if digits_str and digits_str.isdigit():
            return float(digits_str)

        return None

    def _match_digits(self, thresh: np.ndarray, candidates: list) -> str | None:
        """
        Intenta reconocer dígitos usando template matching simple.
        Generamos templates 0-9 en la misma escala.
        """
        if not candidates:
            return None

        result = []
        for _, cnt in candidates:
            x, y, w, h = cv2.boundingRect(cnt)
            digit_img = thresh[y : y + h, x : x + w]

            # Normalizar tamaño
            digit_img = cv2.resize(digit_img, (20, 35))

            # Comparar con templates generados
            best_digit, best_score = self._classify_digit(digit_img)
            if best_score > 0.4:  # umbral de confianza
                result.append(str(best_digit))

        return "".join(result) if result else None

    def _classify_digit(self, digit_img: np.ndarray) -> tuple[int, float]:
        """
        Clasifica un dígito comparando con templates sintéticos.
        Returns (digit, score).
        """
        # Para simplicidad, usamos características geométricas:
        # - Relación de aspecto
        # - Momentos de Hu
        # - Proporción de píxeles blancos en regiones

        moments = cv2.HuMoments(cv2.moments(digit_img)).flatten()
        # Normalizar momentos
        moments = -np.sign(moments) * np.log10(np.abs(moments) + 1e-10)

        # Proporción en 7 segmentos verticales (heurística)
        h, w = digit_img.shape
        segments = []
        for i in range(7):
            y_start = int(h * i / 7)
            y_end = int(h * (i + 1) / 7)
            segment = digit_img[y_start:y_end, :]
            segments.append(np.count_nonzero(segment) / (segment.size + 1))

        # Clasificador simple por vecino más cercano con templates de referencia
        # Usamos Hu moments como feature principal
        # (Para producción real usaríamos pytesseract, pero evitamos dependencia extra)

        # Templates de referencia de Hu moments para dígitos en display LCD ETS2
        # Estos se calibrarían con capturas reales del juego
        ref_moments = {
            0: np.array([0.18, 0.05, 0.02, 0.01, -0.01, 0.005, -0.01]),
            1: np.array([0.22, 0.08, 0.03, 0.02, -0.02, 0.01, -0.02]),
            2: np.array([0.20, 0.03, 0.01, 0.005, -0.01, 0.003, -0.005]),
            3: np.array([0.19, 0.04, 0.01, 0.008, -0.01, 0.004, -0.008]),
            4: np.array([0.21, 0.06, 0.02, 0.015, -0.015, 0.008, -0.01]),
            5: np.array([0.18, 0.03, 0.01, 0.006, -0.008, 0.003, -0.006]),
            6: np.array([0.19, 0.04, 0.015, 0.01, -0.01, 0.005, -0.008]),
            7: np.array([0.20, 0.07, 0.025, 0.018, -0.018, 0.01, -0.02]),
            8: np.array([0.17, 0.02, 0.008, 0.003, -0.005, 0.002, -0.004]),
            9: np.array([0.19, 0.03, 0.012, 0.007, -0.008, 0.004, -0.007]),
        }

        best_digit = 0
        best_dist = float("inf")
        for digit, ref in ref_moments.items():
            dist = np.linalg.norm(moments[:7] - ref)
            if dist < best_dist:
                best_dist = dist
                best_digit = digit

        # Convertir distancia a score (menor distancia = mayor score)
        score = max(0.0, 1.0 - best_dist / 2.0)
        return best_digit, score

    def _optical_flow_speed(self, frame_bgr: np.ndarray) -> float | None:
        """
        Estima velocidad por optical flow (magnitud de movimiento).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # Verificar tamaño compatible (ventana pudo cambiar de tamaño)
        if self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.flow_history.clear()
            return None

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        avg_mag = float(np.mean(mag))

        self.prev_gray = gray

        # Suavizar con historial
        self.flow_history.append(avg_mag)
        if len(self.flow_history) > self.flow_window:
            self.flow_history.pop(0)

        if len(self.flow_history) >= 3:
            avg_mag = np.mean(self.flow_history)

        # Convertir a km/h (requiere calibración en pista recta)
        speed = avg_mag * self.flow_scale
        return speed

    def calibrate_flow_scale(self, real_speed_kmh: float, flow_magnitude: float):
        """Calibra el factor de conversión flujo → km/h."""
        if flow_magnitude > 0:
            self.flow_scale = real_speed_kmh / flow_magnitude

    def reset(self):
        self.prev_gray = None
        self.flow_history.clear()
