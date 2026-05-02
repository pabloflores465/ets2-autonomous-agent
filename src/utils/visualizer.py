"""
Visualizador de debug para percepción.
Muestra en ventanas OpenCV:
  - Frame con bboxes, zonas y líneas de carril
  - Minimapa procesado
  - Panel de estado (comportamiento activo, métricas)
  - Flechas de dirección GPS y decisión
  - Botones GUI: Pausar, Reanudar, Salir
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
    "barrier": (0, 0, 200),  # barrera/guardarraíl en azul oscuro
}

# Colores para direcciones
DIR_COLORS = {
    "straight": (0, 255, 0),
    "turn_left": (0, 165, 255),
    "turn_right": (0, 165, 255),
    "unknown": (128, 128, 128),
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


class Button:
    """Botón clickeable renderizado en frame OpenCV."""

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        color: tuple[int, int, int] = (60, 60, 60),
        hover_color: tuple[int, int, int] = (100, 100, 100),
        text_color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False
        self.clicked = False

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, img: np.ndarray):
        bg = self.hover_color if self.hovered else self.color
        # Fondo
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        # Borde
        border = (200, 200, 200) if self.hovered else (120, 120, 120)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), border, 1)
        # Texto centrado
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(self.label, font, scale, thickness)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(img, self.label, (tx, ty), font, scale, self.text_color, thickness)


class DebugVisualizer:
    """Ventana de debug con percepción, dirección y botones GUI."""

    def __init__(self, config: dict, enabled: bool = True):
        self.enabled = enabled
        self.window_name = "ETS2 Agent - Perception"
        self.minimap_window = "ETS2 Agent - Minimap"

        self._window_created = False

        # Estado de pausa y salida controlados por GUI
        self.paused = False
        self.quit_requested = False

        # Historial de direcciones para suavizar
        self._gps_history: list[str] = []
        self._action_history: list[str] = []
        self._max_history = 10

        # Botones (posición relativa a esquina inferior derecha)
        btn_w, btn_h = 100, 32
        self._btn_pause = Button(0, 0, btn_w, btn_h, "▶ Resume", color=(40, 130, 40))
        self._btn_quit = Button(0, 0, btn_w, btn_h, "✕ Quit", color=(140, 40, 40))
        self._buttons = [self._btn_pause, self._btn_quit]

        # Mouse callback
        self._mouse_pos = (0, 0)

    def _layout_buttons(self, img_w: int, img_h: int):
        """Posiciona botones en la esquina inferior derecha."""
        margin = 10
        btn_w, btn_h = 100, 32
        gap = 8

        # Quit a la derecha
        self._btn_quit.x = img_w - margin - btn_w
        self._btn_quit.y = img_h - margin - btn_h
        self._btn_quit.w = btn_w
        self._btn_quit.h = btn_h

        # Pause/Resume a la izquierda de Quit
        self._btn_pause.x = self._btn_quit.x - gap - btn_w
        self._btn_pause.y = img_h - margin - btn_h
        self._btn_pause.w = btn_w
        self._btn_pause.h = btn_h

        # Actualizar label según estado
        if self.paused:
            self._btn_pause.label = "▶ Resume"
            self._btn_pause.color = (40, 130, 40)
        else:
            self._btn_pause.label = "⏸ Pause"
            self._btn_pause.color = (50, 50, 160)

    def _on_mouse(self, event, x, y, flags, param):
        """Callback de mouse para hover y clics."""
        self._mouse_pos = (x, y)

        # Hover
        for btn in self._buttons:
            btn.hovered = btn.contains(x, y)

        # Click
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._btn_pause.contains(x, y):
                self.paused = not self.paused
                self._btn_pause.clicked = True
            elif self._btn_quit.contains(x, y):
                self.quit_requested = True
                self._btn_quit.clicked = True

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
        cx, cy = center
        angle_map = {
            "straight": -90,
            "turn_left": -135,
            "turn_right": -45,
            "unknown": -90,
        }
        angle_deg = angle_map.get(direction, -90)
        if direction == "turn_left":
            angle_deg = -90 - (intensity * 45)
        elif direction == "turn_right":
            angle_deg = -90 + (intensity * 45)

        angle_rad = np.radians(angle_deg)
        end_x = int(cx + length * np.cos(angle_rad))
        end_y = int(cy + length * np.sin(angle_rad))

        cv2.arrowedLine(img, (cx, cy), (end_x, end_y), color, thickness, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(img, (cx, cy), 8, color, -1)
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), 2)

        if label:
            cv2.putText(img, label, (cx - 40, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_behavior_indicator(
        self,
        img: np.ndarray,
        behavior: str,
        action_str: str,
        top_left: tuple[int, int],
        width: int = 250,
        height: int = 100,
    ):
        x, y = top_left
        color = BEHAVIOR_COLORS.get(behavior, (128, 128, 128))

        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
        cv2.addWeighted(img, 0.6, overlay, 0.4, 0, img)
        cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)

        cv2.putText(img, "BEHAVIOR", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, behavior.upper(), (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        action_display = action_str[:35] + "..." if len(action_str) > 35 else action_str
        cv2.putText(img, action_display, (x + 10, y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    def _draw_direction_panel(
        self,
        img: np.ndarray,
        gps_direction: str,
        gps_intensity: float,
        behavior: str,
        action_str: str,
        steer: float,
    ):
        h, w = img.shape[:2]
        panel_w = 200
        panel_h = 250
        panel_x = w - panel_w - 10
        panel_y = 10

        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(img, 0.7, overlay, 0.3, 0, img)
        cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 1)

        cv2.putText(img, "DIRECTION", (panel_x + 10, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        gps_color = DIR_COLORS.get(gps_direction, (128, 128, 128))
        gps_center = (panel_x + panel_w // 2, panel_y + 80)
        self._draw_direction_arrow(
            img, gps_direction, gps_intensity, gps_center, length=50, color=gps_color, thickness=3,
            label=f"GPS: {gps_direction}",
        )

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
            img, decision_dir, decision_intensity, decision_center, length=40, color=decision_color, thickness=2,
            label=f"BT: {decision_dir}",
        )

        cv2.putText(img, f"Steer: {steer:+.1f}deg", (panel_x + 10, panel_y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Intensity: {gps_intensity:.2f}", (panel_x + 10, panel_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _draw_pause_overlay(self, img: np.ndarray):
        """Overlay oscuro cuando está en pausa."""
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(img, 0.35, overlay, 0.65, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "PAUSED"
        scale = 2.0
        thickness = 4
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(img, text, ((w - tw) // 2, h // 2), font, scale, (0, 255, 255), thickness)

        sub = "Click Resume or press P/Space"
        (sw, sh), _ = cv2.getTextSize(sub, font, 0.55, 1)
        cv2.putText(img, sub, ((w - sw) // 2, h // 2 + 40), font, 0.55, (200, 200, 200), 1)

    def _draw_no_window_overlay(self, img: np.ndarray):
        """Overlay cuando no hay ventana de juego/Parsec detectada."""
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(img, 0.3, overlay, 0.7, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Waiting for ETS2 / Parsec..."
        scale = 1.2
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(img, text, ((w - tw) // 2, h // 2 - 20), font, scale, (0, 200, 255), thickness)

        sub = "Open the game or connect via Parsec"
        (sw, sh), _ = cv2.getTextSize(sub, font, 0.5, 1)
        cv2.putText(img, sub, ((w - sw) // 2, h // 2 + 20), font, 0.5, (180, 180, 180), 1)

    def _set_window_on_top(self, window_name: str):
        """Pone la ventana OpenCV always-on-top vía PyObjC (macOS)."""
        try:
            from AppKit import NSApplication, NSFloatingWindowLevel
            for win in NSApplication.sharedApplication().windows():
                if win.title() == window_name:
                    win.setLevel_(NSFloatingWindowLevel)
                    break
        except Exception:
            pass  # fallback silencioso

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
        if not self.enabled:
            return

        # Crear ventanas la primera vez (antes de imshow)
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.window_name, 800, 500)
            cv2.setMouseCallback(self.window_name, self._on_mouse)
            cv2.namedWindow(self.minimap_window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.minimap_window, 280, 210)
            self._window_created = True
            # Always-on-top
            self._set_window_on_top(self.window_name)
            self._set_window_on_top(self.minimap_window)

        vis = frame.copy()
        h, w = vis.shape[:2]

        # Detectar frame negro (sin ventana de juego)
        is_black = (vis.sum() == 0)
        if is_black:
            self._draw_no_window_overlay(vis)

        # Zonas
        vis = draw_zones(vis, zone_assigner)

        # Detecciones
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            color = CLASS_COLORS.get(det.class_name, (128, 128, 128))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Carriles
        if lane_detector is not None and lane_info is not None:
            vis = lane_detector.draw_lanes(vis, lane_info)

        # Header
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        cv2.putText(vis, f"FPS: {fps:.1f} | Frame: {frame_id} | Total: {total_ms:.0f}ms | Behavior: {behavior}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"GPS: {gps_direction} | Light: {traffic_light} | Dets: {len(detections)} | Collision: {'YES' if collision else 'no'}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Action: {action_str}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Panel de dirección
        self._draw_direction_panel(vis, gps_direction, gps_intensity, behavior, action_str, steer)

        # Behavior indicator
        self._draw_behavior_indicator(vis, behavior, action_str, (10, h - 110))

        # Botones
        self._layout_buttons(w, h)
        for btn in self._buttons:
            btn.draw(vis)

        # Overlay de pausa (encima de todo excepto botones)
        if self.paused:
            self._draw_pause_overlay(vis)
            # Redibujar botones encima del overlay
            for btn in self._buttons:
                btn.draw(vis)

        cv2.imshow(self.window_name, vis)

        # Minimapa
        mini_vis = self._render_minimap(bgr_frame, minimap_proc, zones)
        if mini_vis is not None:
            cv2.imshow(self.minimap_window, mini_vis)

        # Teclado (además de los botones)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("p"), ord(" ")):
            self.paused = not self.paused
        elif key == ord("q"):
            self.quit_requested = True

        return key

    def _render_minimap(
        self,
        frame_bgr: np.ndarray,
        minimap_proc: MinimapProcessor,
        zones: dict[str, list[Detection]],
    ) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = minimap_proc.get_roi_coords(frame_bgr.shape)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame_bgr[y1:y2, x1:x2].copy()
        cv2.rectangle(roi, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), (0, 255, 255), 2)

        for det in zones.get("minimap", []):
            bx1, by1, bx2, by2 = det.bbox.astype(int)
            bx1 = max(0, bx1 - x1)
            by1 = max(0, by1 - y1)
            bx2 = min(roi.shape[1], bx2 - x1)
            by2 = min(roi.shape[0], by2 - y1)
            cv2.rectangle(roi, (bx1, by1), (bx2, by2), (0, 255, 0), 1)

        cv2.putText(roi, "Minimap", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return roi

    def log_detections(self, detections: list[Detection], zones: dict[str, list[Detection]]):
        if not self.enabled or not detections:
            return
        print(f"\n{'=' * 60}")
        print(f"Detections: {len(detections)}")
        print(f"{'=' * 60}")
        for zone_name, zone_dets in zones.items():
            if zone_dets:
                print(f"  [{zone_name}]:")
                for d in zone_dets:
                    print(f"    - {d.class_name:15s} conf={d.confidence:.2f} bbox=({d.bbox[0]:.0f},{d.bbox[1]:.0f}) size={d.bbox[2] - d.bbox[0]:.0f}x{d.bbox[3] - d.bbox[1]:.0f}")

    def close(self):
        cv2.destroyAllWindows()
