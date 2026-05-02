"""
Visualizador de debug para percepción.
Muestra en ventanas OpenCV:
  - Frame con bboxes, zonas y líneas de carril
  - Minimapa procesado
  - Panel de estado (comportamiento activo, métricas)
"""


import cv2
import numpy as np

from src.perception.detector import Detection
from src.perception.lane_detector import LaneDetector, LaneInfo
from src.perception.minimap import MinimapProcessor
from src.perception.zones import ZoneAssigner, draw_zones

# Colores por clase
CLASS_COLORS = {
    "car": (0, 255, 0),
    "truck": (0, 200, 0),
    "bus": (0, 150, 0),
    "motorcycle": (0, 255, 128),
    "person": (255, 128, 0),
    "traffic_light": (255, 255, 0),
    "stop_sign": (0, 0, 255),
}


class DebugVisualizer:
    """Muestra ventanas de debug con toda la información de percepción.
    Controles de teclado en la ventana:
      P / Space = Pausar / Reanudar
      Q        = Salir
      S        = Tomar screenshot
    """

    def __init__(self, config: dict, enabled: bool = True):
        self.enabled = enabled
        self.window_name = "ETS2 Agent - Perception"
        self.minimap_window = "ETS2 Agent - Minimap"
        self.status_window = "ETS2 Agent - Status"

        # Posiciones de ventanas
        self._window_created = False

    def show(
        self,
        frame: np.ndarray,
        bgr_frame: np.ndarray,
        detections: list[Detection],
        zones: dict[str, list[Detection]],
        zone_assigner: ZoneAssigner,
        lane_info: LaneInfo | None,
        lane_detector: LaneDetector,
        minimap_proc: MinimapProcessor,
        behavior: str,
        action_str: str,
        fps: float,
        total_ms: float,
        frame_id: int,
        traffic_light: str = "none",
        gps_direction: str = "unknown",
        collision: bool = False,
    ):
        """Renderiza todas las visualizaciones."""
        if not self.enabled:
            return

        # ── Ventana principal: frame + detecciones + zonas + carriles ──
        vis = frame.copy()

        # Dibujar zonas
        vis = draw_zones(vis, zone_assigner)

        # Dibujar detecciones
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            color = CLASS_COLORS.get(det.class_name, (128, 128, 128))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Dibujar carriles
        if lane_detector is not None and lane_info is not None:
            lane_vis = lane_detector.draw_lanes(vis, lane_info)
            vis = lane_vis

        # Header info
        h, w = vis.shape[:2]
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        cv2.putText(
            vis,
            f"FPS: {fps:.1f} | Frame: {frame_id} | Total: {total_ms:.0f}ms | Behavior: {behavior}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            vis,
            f"GPS: {gps_direction} | "
            f"Light: {traffic_light} | "
            f"Dets: {len(detections)} | "
            f"Collision: {'YES' if collision else 'no'}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            vis,
            f"Action: {action_str}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        cv2.imshow(self.window_name, vis)

        # ── Ventana de minimapa ──
        mini_vis = self._render_minimap(bgr_frame, minimap_proc, zones)
        if mini_vis is not None:
            cv2.imshow(self.minimap_window, mini_vis)

        # Crear ventanas solo la primera vez
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 960, 540)
            cv2.namedWindow(self.minimap_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.minimap_window, 360, 270)
            self._window_created = True

        # Non-blocking wait
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[VISUALIZER] Q pressed - requesting stop")
        return key

    def _render_minimap(
        self,
        frame_bgr: np.ndarray,
        minimap_proc: MinimapProcessor,
        zones: dict[str, list[Detection]],
    ) -> np.ndarray | None:
        """Renderiza el minimapa con overlay de procesamiento."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = minimap_proc.get_roi_coords(frame_bgr.shape)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame_bgr[y1:y2, x1:x2].copy()

        # Dibujar borde
        cv2.rectangle(roi, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), (0, 255, 255), 2)

        # Dibujar detecciones que caen en zona minimapa
        minimap_dets = zones.get("minimap", [])
        for det in minimap_dets:
            bx1, by1, bx2, by2 = det.bbox.astype(int)
            # Ajustar coordenadas relativas a la ROI
            bx1 = max(0, bx1 - x1)
            by1 = max(0, by1 - y1)
            bx2 = min(roi.shape[1], bx2 - x1)
            by2 = min(roi.shape[0], by2 - y1)
            cv2.rectangle(roi, (bx1, by1), (bx2, by2), (0, 255, 0), 1)

        cv2.putText(roi, "Minimap", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return roi

    def log_detections(self, detections: list[Detection], zones: dict[str, list[Detection]]):
        """Imprime detecciones en consola (modo verbose)."""
        if not self.enabled:
            return

        if not detections:
            return

        print(f"\n{'=' * 60}")
        print(f"Detections: {len(detections)}")
        print(f"{'=' * 60}")
        for zone_name, zone_dets in zones.items():
            if zone_dets:
                print(f"  [{zone_name}]:")
                for d in zone_dets:
                    print(
                        f"    - {d.class_name:15s} conf={d.confidence:.2f} "
                        f"bbox=({d.bbox[0]:.0f},{d.bbox[1]:.0f}) "
                        f"size={d.bbox[2] - d.bbox[0]:.0f}x{d.bbox[3] - d.bbox[1]:.0f}"
                    )

    def close(self):
        cv2.destroyAllWindows()
