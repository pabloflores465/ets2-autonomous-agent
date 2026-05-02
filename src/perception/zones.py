"""
Asignación de detecciones a zonas lógicas de percepción.
Cada zona es una región del frame definida por porcentajes.
"""


import numpy as np

from src.perception.detector import Detection


class ZoneAssigner:
    """Asigna detecciones a zonas lógicas (frontal, espejos, etc)."""

    def __init__(self, config: dict, frame_width: int, frame_height: int):
        zone_cfg = config["perception"]["zones"]
        self.width = frame_width
        self.height = frame_height
        self.zones = {}
        for name, zone in zone_cfg.items():
            self.zones[name] = {
                "x1": int(zone["x1"] * frame_width),
                "y1": int(zone["y1"] * frame_height),
                "x2": int(zone["x2"] * frame_width),
                "y2": int(zone["y2"] * frame_height),
            }

    def assign(self, detections: list[Detection]) -> dict[str, list[Detection]]:
        """
        Clasifica cada detección en una o más zonas.
        Returns:
            Dict[zone_name, List[Detection]]
        """
        result = {name: [] for name in self.zones}
        for det in detections:
            for zone_name, zone in self.zones.items():
                if self._bbox_in_zone(det, zone):
                    result[zone_name].append(det)
        return result

    def _bbox_in_zone(self, det: Detection, zone: dict) -> bool:
        """Verifica si el centro del bbox está dentro de la zona."""
        cx, cy = det.center_x, det.center_y
        return zone["x1"] <= cx <= zone["x2"] and zone["y1"] <= cy <= zone["y2"]

    def get_zone_bounds(self, zone_name: str) -> dict:
        """Devuelve los límites en píxeles de una zona."""
        return self.zones.get(zone_name, {})


def draw_zones(frame: np.ndarray, assigner: ZoneAssigner) -> np.ndarray:
    """Dibuja las zonas sobre el frame (modo debug)."""
    import cv2

    colors = {
        "frontal": (0, 255, 0),
        "capo": (255, 255, 0),
        "espejo_izq": (255, 0, 0),
        "espejo_der": (0, 0, 255),
        "lateral_izq": (255, 0, 255),
        "lateral_der": (0, 255, 255),
    }
    vis = frame.copy()
    for name, zone in assigner.zones.items():
        color = colors.get(name, (128, 128, 128))
        cv2.rectangle(vis, (zone["x1"], zone["y1"]), (zone["x2"], zone["y2"]), color, 2)
        cv2.putText(
            vis, name, (zone["x1"] + 5, zone["y1"] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    return vis
