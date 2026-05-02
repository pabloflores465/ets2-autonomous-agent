"""
Detector de barreras/guardarraíles mediante CV.
Detecta líneas horizontales largas (guardarraíles, barreras metálicas)
que no son detectadas por YOLO (no están en COCO).

EXCLUYE la zona del capó/timón (Y > 78%) para no detectar el interior de la cabina.
"""

import cv2
import numpy as np

from src.perception.detector import Detection


def detect_barriers(frame_bgr: np.ndarray) -> list[Detection]:
    """
    Busca barreras/guardarraíles en los laterales del frame.
    Excluye zona del capó (Y > 78%) para evitar falsos positivos del interior.
    Returns:
        Lista de Detection con class_name="barrier"
    """
    h, w = frame_bgr.shape[:2]
    detections = []

    # ROI: zona media del frame, EXCLUYENDO el capó (último 22% donde está el timón/dashboard)
    # y también el cielo (primer 30%)
    roi_y1 = int(h * 0.30)
    roi_y2 = int(h * 0.75)  # excluye el capó (Y > 78%)
    if roi_y2 <= roi_y1:
        return detections
    roi = frame_bgr[roi_y1:roi_y2, :]
    rh = roi.shape[0]

    # Gris + bordes
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 80)

    # Detectar líneas horizontales
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=int(w * 0.25),
        maxLineGap=30,
    )

    if lines is None:
        return detections

    barrier_regions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < w * 0.25 or dy > dx * 0.3:
            continue

        cy = (y1 + y2) / 2 + roi_y1
        cx = (x1 + x2) / 2

        # Excluir si está en la zona del capó (timón/dashboard)
        if cy > h * 0.78:
            continue

        barrier_regions.append((cx, cy, dx))

    if not barrier_regions:
        return detections

    # Fusionar regiones cercanas
    barrier_regions.sort(key=lambda r: r[1])
    merged = [barrier_regions[0]]
    for cx, cy, length in barrier_regions[1:]:
        last_cx, last_cy, last_length = merged[-1]
        if abs(cy - last_cy) < 20:
            merged[-1] = (
                (cx + last_cx) / 2,
                (cy + last_cy) / 2,
                max(length, last_length),
            )
        else:
            merged.append((cx, cy, length))

    for cx, cy, length in merged:
        x_frac = cx / w
        if x_frac < 0.1 or x_frac > 0.9:
            continue

        box_w = int(length)
        box_h = int(h * 0.1)
        x1 = max(0, int(cx - box_w / 2))
        y1 = max(0, int(cy - box_h / 2))
        x2 = min(w, int(cx + box_w / 2))
        y2 = min(h, int(cy + box_h / 2))

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        det = Detection(class_id=-1, class_name="barrier", confidence=0.5, bbox=bbox)
        detections.append(det)

    return detections
