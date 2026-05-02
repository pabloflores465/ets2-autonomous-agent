"""
Visualizador de debug para percepción.
Muestra en ventanas OpenCV:
  - Frame con bboxes, zonas y líneas de carril
  - Minimapa procesado
  - Panel de estado (comportamiento activo, métricas)
  - Flechas de dirección GPS y decisión
"""

import cv2
import numpy as np

from src.perception.detector import Detection
from src.perception.lane_detector import LaneDetector, LaneInfo
from src.perception.minimap import GPSDirection, MinimapProcessor
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

# Colores para direcciones
DIR_COLORS = {
    "straight": (0, 255, 0),    # Verde
    "turn_left": (0, 165, 255),  # Naranja
    "turn_right": (0, 165, 255), # Naranja
    "unknown": (128, 128, 128),  # Gris
}

# Colores para behaviors
BEHAVIOR_COLORS = {
    "EmergencyStop": (0, 0, 255),
    "ObstacleAvoid": (0, 128, 255),
    "TrafficLight": (0, 255, 255),
    "StopSign": (0, 0, 200),
    "YieldPedestrian": (255, 128, 0),
    "LaneFollow": (0, 255, 0),
    "Cruise": (0, 200, 0),
    "idle": (128, 128, 128),
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
        self.direction_window = "ETS2 Agent - Direction"

        # Posiciones de ventanas
        self._window_created = False
        
        # Historial de direcciones para suavizar
        self._gps_history = []
        self._action_history = []
        self._max_history = 10

    def _draw_direction_arrow(
        self,
        img: np.ndarray,
        direction: str,
        intensity: float,
        center: tuple[int, int],
        length: int = 80,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3,
        label: str = "",
    ):
        """Dibuja una flecha de dirección en la imagen."""
        cx, cy = center
        
        # Mapear dirección a ángulo
        angle_map = {
            "straight": -90,  # Arriba
            "turn_left": -135,  # Arriba-izquierda
            "turn_right": -45,  # Arriba-derecha
            "unknown": -90,
        }
        
        angle_deg = angle_map.get(direction, -90)
        # Ajustar ángulo por intensidad para giros
        if direction == "turn_left":
            angle_deg = -90 - (intensity * 45)
        elif direction == "turn_right":
            angle_deg = -90 + (intensity * 45)
        
        angle_rad = np.radians(angle_deg)
        
        # Calcular punta de flecha
        end_x = int(cx + length * np.cos(angle_rad))
        end_y = int(cy + length * np.sin(angle_rad))
        
        # Dibujar flecha
        cv2.arrowedLine(
            img,
            (cx, cy),
            (end_x, end_y),
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.3,
        )
        
        # Dibujar círculo en la base
        cv2.circle(img, (cx, cy), 8, color, -1)
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), 2)
        
        # Label
        if label:
            cv2.putText(
                img,
                label,
                (cx - 40, cy + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    def _draw_behavior_indicator(
        self,
        img: np.ndarray,
        behavior: str,
        action_str: str,
        top_left: tuple[int, int],
        width: int = 250,
        height: int = 100,
    ):
        """Dibuja un panel indicador del behavior activo."""
        x, y = top_left
        color = BEHAVIOR_COLORS.get(behavior, (128, 128, 128))
        
        # Fondo semi-transparente
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
        cv2.addWeighted(img, 0.6, overlay, 0.4, 0, img)
        
        # Borde con color del behavior
        cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
        
        # Título
        cv2.putText(
            img,
            "BEHAVIOR",
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        
        # Nombre del behavior (grande)
        cv2.putText(
            img,
            behavior.upper(),
            (x + 10, y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        
        # Acción (pequeño)
        # Truncar acción si es muy larga
        action_display = action_str[:35] + "..." if len(action_str) > 35 else action_str
        cv2.putText(
            img,
            action_display,
            (x + 10, y + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
        )

    def _draw_direction_panel(
        self,
        img: np.ndarray,
        gps_direction: str,
        gps_intensity: float,
        behavior: str,
        action_str: str,
        steer: float,
    ):
        """Dibuja panel completo de dirección con flechas grandes."""
        h, w = img.shape[:2]
        
        # Panel de fondo (lado derecho)
        panel_w = 200
        panel_h = 250
        panel_x = w - panel_w - 10
        panel_y = 10
        
        # Fondo semi-transparente
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(img, 0.7, overlay, 0.3, 0, img)
        
        # Borde
        cv2.rectangle(
            img,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (100, 100, 100),
            1,
        )
        
        # Título
        cv2.putText(
            img,
            "DIRECTION",
            (panel_x + 10, panel_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        
        # Flecha GPS (grande)
        gps_color = DIR_COLORS.get(gps_direction, (128, 128, 128))
        gps_center = (panel_x + panel_w // 2, panel_y + 80)
        self._draw_direction_arrow(
            img,
            gps_direction,
            gps_intensity,
            gps_center,
            length=50,
            color=gps_color,
            thickness=3,
            label=f"GPS: {gps_direction}",
        )
        
        # Flecha de decisión (basada en behavior y steer)
        decision_dir = "straight"
        decision_intensity = 0.5
        if steer < -5:
            decision_dir = "turn_left"
            decision_intensity = min(1.0, abs(steer) / 25)
        elif steer > 5:
            decision_dir = "turn_right"
            decision_intensity = min(1.0, steer / 25)
        
        decision_color = BEHAVIOR_COLORS.get(behavior, (0, 255, 0))
        decision_center = (panel_x + panel_w // 2, panel_y + 160)
        self._draw_direction_arrow(
            img,
            decision_dir,
            decision_intensity,
            decision_center,
            length=40,
            color=decision_color,
            thickness=2,
            label=f"BT: {decision_dir}",
        )
        
        # Info de steer
        steer_text = f"Steer: {steer:+.1f}°"
        cv2.putText(
            img,
            steer_text,
            (panel_x + 10, panel_y + 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        
        # Intensidad
        intensity_text = f"Intensity: {gps_intensity:.2f}"
        cv2.putText(
            img,
            intensity_text,
            (panel_x + 10, panel_y + 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

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
        gps_intensity: float = 0.0,
        steer: float = 0.0,
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

        # ── Panel de dirección con flechas ──
        self._draw_direction_panel(
            vis,
            gps_direction,
            gps_intensity,
            behavior,
            action_str,
            steer,
        )
        
        # ── Indicador de behavior (esquina inferior izquierda) ──
        self._draw_behavior_indicator(
            vis,
            behavior,
            action_str,
            (10, h - 110),
        )

        cv2.imshow(self.window_name, vis)

        # ── Ventana de minimapa ──
        mini_vis = self._render_minimap(bgr_frame, minimap_proc, zones)
        if mini_vis is not None:
            cv2.imshow(self.minimap_window, mini_vis)

        # Crear ventanas solo la primera vez
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)
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
