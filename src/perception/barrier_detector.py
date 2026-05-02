"""
Detector de barreras/guardarraíles mediante CV.
Detecta líneas horizontales largas (guardarraíles, barreras metálicas)
que no son detectadas por YOLO (no están en COCO).
"""

import cv2
import numpy as np

from src.perception.detector import Detection


def detect_barriers(frame_bgr: np.ndarray) -> list[Detection]:
    """
    Busca barreras/guardarraíles en los laterales del frame.
    Returns:
        Lista de Detection con class_name="barrier"
    """
    h, w = frame_bgr.shape[:2]
    detections = []

    # ROI: zona inferior, donde suelen estar las barreras
    roi_y1 = int(h * 0.6)
    roi_y2 = int(h * 0.95)
    roi = frame_bgr[roi_y1:roi_y2, :]

    # Gris + bordes
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 80)

    # Detectar líneas horizontales (guardarraíl es una línea larga horizontal)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=int(w * 0.3),  # mínimo 30% del ancho del frame
        maxLineGap=30,
    )

    if lines is None:
        return detections

    # Agrupar líneas cercanas que forman una barrera
    barrier_regions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Filtrar: solo líneas casi horizontales (pendiente baja)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < w * 0.25 or dy > dx * 0.3:
            continue  # no es horizontal

        # Posición y altura de la línea
        cy = (y1 + y2) / 2 + roi_y1
        cx = (x1 + x2) / 2
        length = dx

        barrier_regions.append((cx, cy, length))

    if not barrier_regions:
        return detections

    # Fusionar regiones cercanas
    barrier_regions.sort(key=lambda r: r[1])  # ordenar por Y
    merged = [barrier_regions[0]]
    for cx, cy, length in barrier_regions[1:]:
        last_cx, last_cy, last_length = merged[-1]
        if abs(cy - last_cy) < 20:  # misma región vertical
            # Extender
            merged[-1] = (
                (cx + last_cx) / 2,
                (cy + last_cy) / 2,
                max(length, last_length),
            )
        else:
            merged.append((cx, cy, length))

    # Crear Detection para cada barrera encontrada
    for cx, cy, length in merged:
        # Solo si está en los laterales del frame
        x_frac = cx / w
        if x_frac < 0.1 or x_frac > 0.9:
            continue  # muy al borde, probablemente no es relevante

        # Crear bbox alrededor de la barrera
        box_w = int(length)
        box_h = int(h * 0.1)
        x1 = max(0, int(cx - box_w / 2))
        y1 = max(0, int(cy - box_h / 2))
        x2 = min(w, int(cx + box_w / 2))
        y2 = min(h, int(cy + box_h / 2))

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        det = Detection(
            class_id=-1,
            class_name="barrier",
            confidence=0.5,
            bbox=bbox,
        )
        detections.append(det)

    return detections
