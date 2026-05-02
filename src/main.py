"""
Loop principal de conducción autónoma en ETS2.
Pipeline: Captura → Percepción → Contexto → Decisión → Actuación
15 Hz con timeout adaptativo. Pausa/Resume con P, Quit con Q.
"""

import os
import sys
import time

import cv2
import pyautogui
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# fmt: off
# ruff: noqa: I001
from src.actuation.controller import Controller  # noqa: E402
from src.capture.screen_grabber import ScreenGrabber  # noqa: E402
from src.decision.blackboard import BB  # noqa: E402
from src.decision.behavior_tree import build_behavior_tree, get_active_action  # noqa: E402
from src.decision.context import WorldContext  # noqa: E402
from src.perception.barrier_detector import detect_barriers  # noqa: E402
from src.perception.collision_detector import CollisionDetector  # noqa: E402
from src.perception.detector import YOLODetector  # noqa: E402
from src.perception.lane_detector import LaneDetector  # noqa: E402
from src.perception.minimap import MinimapProcessor  # noqa: E402
from src.perception.speed_detector import SpeedDetector  # noqa: E402
from src.perception.traffic_light_classifier import TrafficLightClassifier  # noqa: E402
from src.perception.zones import ZoneAssigner  # noqa: E402
from src.utils.logger import CycleMetrics, SessionLogger  # noqa: E402
from src.utils.visualizer import DebugVisualizer  # noqa: E402
# fmt: on


class ETS2Agent:
    """Agente autónomo de conducción para ETS2."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.target_hz = self.config["decision"]["target_hz"]
        self.frame_budget_ms = 1000.0 / self.target_hz
        perc = self.config["perception"]

        # Módulos
        self.grabber = ScreenGrabber(self.config)
        self.detector = YOLODetector(
            model_path=perc["model"],
            device=perc["device"],
            confidence=perc["confidence"],
            iou=perc["iou"],
        )
        self.zones = ZoneAssigner(self.config, self.grabber.width, self.grabber.height)
        self.light_classifier = TrafficLightClassifier(self.config)
        self.minimap = MinimapProcessor(self.config)
        self.lane_detector = LaneDetector(perc.get("lane_detector"))
        self.collision_detector = CollisionDetector(perc.get("collision"))
        self.speed_detector = SpeedDetector(perc.get("speed"))

        self.world = WorldContext()
        self.bt = build_behavior_tree(self.world, self.config)
        self.controller = Controller(self.config)

        self.logger = SessionLogger(
            log_dir=self.config["logging"]["log_dir"],
            level=self.config["logging"]["level"],
        )

        debug_enabled = self.config.get("debug", {}).get("enabled", True)
        self.visualizer = DebugVisualizer(self.config, enabled=debug_enabled)

        self.running = False
        self.frame_id = 0
        self._frame_bgr = None

    @property
    def paused(self) -> bool:
        """Estado de pausa leído del visualizer."""
        return self.visualizer.paused

    @paused.setter
    def paused(self, value: bool):
        self.visualizer.paused = value

    def run(self):
        """Loop principal."""
        self.logger.start_session()
        self.running = True
        self.logger.log_event("INFO", "Agent started. Buttons or: P=pause Q=quit")

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0

        # Dar foco a ETS2
        self._focus_ets2()

        self.bt.setup(timeout=15)

        while self.running:
            t_start = time.perf_counter()

            # ── Quit (teclado o botón) ──
            if self.visualizer.quit_requested:
                self.logger.log_event("INFO", "Stopped via GUI button")
                self.running = False
                break

            # ── Pausa (teclado o botón) ──
            if self.paused:
                self._show_pause_overlay()
                time.sleep(0.1)
                continue

            # ── Captura ──
            try:
                frame_rgb = self.grabber.capture()
                frame_bgr = frame_rgb[:, :, ::-1]
                self._frame_bgr = frame_bgr
            except Exception as e:
                self.logger.log_event("ERROR", f"Capture: {e}")
                time.sleep(0.05)
                continue

            # ── Percepción ──
            t_per = time.perf_counter()
            detections = self.detector.detect(frame_bgr)

            # Detectar barreras/guardarraíles (no detectados por YOLO)
            barriers = detect_barriers(frame_bgr)
            detections.extend(barriers)

            zones = self.zones.assign(detections)

            if len(detections) == 0:
                self.logger.log_event("DEBUG", f"F{self.frame_id}: no objects detected")

            # Semáforo
            tl_state = None
            for det in zones.get("frontal", []):
                if det.class_name == "traffic_light":
                    tl_state = self.light_classifier.classify(frame_bgr, det).value
                    break

            # Minimapa + carril + colisión
            gps_dir, gps_int, truck_xy = self.minimap.process(frame_bgr)

            # Detectar carril cada 3 frames
            if self.frame_id % 3 == 0:
                lane_info = self.lane_detector.detect(frame_bgr)
            else:
                lane_info = None

            # Colisión cada 4 frames
            if self.frame_id % 4 == 0:
                collision_info = self.collision_detector.detect(frame_bgr)
            else:
                collision_info = None

            # Velocidad cada 3 frames
            if self.frame_id % 3 == 1:
                speed_info = self.speed_detector.detect(frame_bgr)
            else:
                speed_info = None

            per_ms = (time.perf_counter() - t_per) * 1000

            # ── Contexto ──
            self.world.update(
                detections=detections,
                zones=zones,
                gps_direction=gps_dir,
                gps_intensity=gps_int,
                truck_minimap_xy=truck_xy,
                traffic_light_state=tl_state,
                lane_info=lane_info,
                collision_info=collision_info,
                speed_info=speed_info,
            )

            # ── Decisión ──
            t_dec = time.perf_counter()
            self.bt.tick()
            action = get_active_action(self.bt)
            reverse_req = BB.reverse_requested
            if BB.collision_recovered:
                self.world.collision_recovered = True
                self.collision_detector.reset()
                BB.collision_recovered = False
                BB.reverse_requested = False
            dec_ms = (time.perf_counter() - t_dec) * 1000

            # ── Actuación ──
            t_act = time.perf_counter()
            cam_look = BB.camera_look_angle
            self.controller.execute(
                action, duration_ms=50, reverse_requested=reverse_req, camera_look_angle=cam_look
            )
            act_ms = (time.perf_counter() - t_act) * 1000

            # ── Métricas ──
            total_ms = (time.perf_counter() - t_start) * 1000
            over = total_ms > self.frame_budget_ms
            metrics = CycleMetrics(
                timestamp=time.time(),
                frame_id=self.frame_id,
                fps=1000.0 / total_ms if total_ms > 0 else 0,
                capture_ms=0,
                inference_ms=per_ms,
                decision_ms=dec_ms,
                actuation_ms=act_ms,
                total_ms=total_ms,
                detections_count=len(detections),
                active_behavior=action.behavior,
                action=str(action),
                confidence_sum=sum(d.confidence for d in detections),
                over_budget=over,
            )
            self.logger.log_cycle(metrics)

            # ── Visualización ──
            if self.visualizer.enabled and self.frame_id % 3 == 0:  # cada ~3 frames
                if len(detections) == 0:
                    self.logger.log_event("DEBUG", f"F{self.frame_id}: 0 detections")
                
                # Obtener steer de la acción actual
                current_action = BB.action
                steer_value = current_action.steer if hasattr(current_action, 'steer') else 0.0
                
                key = self.visualizer.show(
                    frame=frame_bgr,
                    bgr_frame=frame_bgr,
                    detections=detections,
                    zones=zones,
                    zone_assigner=self.zones,
                    lane_info=lane_info,
                    lane_detector=self.lane_detector,
                    minimap_proc=self.minimap,
                    behavior=action.behavior,
                    action_str=str(action),
                    fps=metrics.fps,
                    total_ms=total_ms,
                    frame_id=self.frame_id,
                    traffic_light=tl_state or "none",
                    gps_direction=gps_dir.value if gps_dir else "unknown",
                    collision=(collision_info.collision_detected if collision_info else False),
                    gps_intensity=gps_int,
                    steer=steer_value,
                )
                # Controles de teclado (botones ya manejados por visualizer)
                if key in (ord("p"), ord(" ")):
                    self.paused = not self.paused
                    state = "PAUSED" if self.paused else "RESUMED"
                    self.logger.log_event("INFO", f"Agent {state} (keyboard)")
                elif key == ord("q"):
                    self.visualizer.quit_requested = True

            # ── Timeout ──
            elapsed = (time.perf_counter() - t_start) * 1000
            if elapsed < self.frame_budget_ms:
                time.sleep((self.frame_budget_ms - elapsed) / 1000)
            else:
                self.logger.log_event("WARN", f"F{self.frame_id} over budget: {elapsed:.0f}ms")

            self.frame_id += 1

        # Shutdown
        self.controller.emergency_stop()
        self.bt.shutdown()
        self.grabber.close()
        self.visualizer.close()
        summary = self.logger.end_session()
        self.logger.log_event("INFO", f"Summary: {summary}")

    def _show_pause_overlay(self):
        """Overlay de pausa en ventana debug."""
        if not self.visualizer.enabled or self._frame_bgr is None:
            return
        vis = self._frame_bgr.copy()
        h, w = vis.shape[:2]
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(vis, 0.4, overlay, 0.6, 0, vis)
        cv2.putText(
            vis, "⏸ PAUSED", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3
        )
        cv2.putText(
            vis,
            "P/Space = resume | Q = quit",
            (w // 2 - 160, h // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.imshow(self.visualizer.window_name, vis)
        cv2.waitKey(1)

    def _focus_ets2(self):
        """Dar foco a ventana ETS2 y liberar freno de mano."""
        import subprocess

        try:
            subprocess.run(
                ["osascript", "-e", 'tell application "Euro Truck Simulator 2" to activate'],
                capture_output=True,
                timeout=3,
            )
            time.sleep(1.0)
            pyautogui.click(
                self.config["actuation"]["steering_center_x"],
                self.config["actuation"]["steering_center_y"],
            )
            # Liberar freno de mano (space) si estaba puesto de sesión anterior
            subprocess.run(
                ["osascript", "-e", 'tell application "System Events" to key code 49'],
                capture_output=True,
                timeout=1,
            )
            time.sleep(0.3)
            self.logger.log_event("INFO", "ETS2 focused + clicked + handbrake released")
        except Exception as e:
            self.logger.log_event("WARN", f"Focus failed: {e}")

    def _check_quit(self) -> bool:
        """Verifica paro manual: Ctrl+C via KeyboardInterrupt."""
        return False  # Ctrl+C ya lo maneja Python


def main():
    agent = ETS2Agent("config.yaml")
    agent.run()


if __name__ == "__main__":
    main()
