"""
Modelo del mundo (World Context).
Agrega detecciones, zonas, estado GPS, velocidad y mantiene memoria temporal.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.perception.detector import Detection
from src.perception.lane_detector import LaneInfo, LaneType
from src.perception.collision_detector import CollisionInfo
from src.perception.speed_detector import SpeedInfo
from src.perception.minimap import GPSDirection


@dataclass
class DrivingAction:
    """Acción de conducción resultante del BT."""
    behavior: str
    accelerate: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    handbrake: bool = False

    def __repr__(self):
        return (f"DrivingAction({self.behavior}, accel={self.accelerate:.2f}, "
                f"brake={self.brake:.2f}, steer={self.steer:.1f}°)")

    @staticmethod
    def idle():
        return DrivingAction("idle")


@dataclass
class WorldContext:
    """Estado agregado del mundo para la toma de decisiones."""

    detections: List[Detection] = field(default_factory=list)
    zones: Dict[str, List[Detection]] = field(default_factory=dict)

    gps_direction: GPSDirection = GPSDirection.UNKNOWN
    gps_intensity: float = 0.0
    truck_minimap_xy: Tuple[float, float] = (0.5, 0.5)

    lane_info: Optional[LaneInfo] = None
    collision_info: Optional[CollisionInfo] = None
    collision_active: bool = False
    collision_recovered: bool = False

    speed_info: Optional[SpeedInfo] = None
    current_speed_kmh: float = 0.0

    traffic_light_state: Optional[str] = None

    detection_history: List[List[Detection]] = field(default_factory=list)
    max_history: int = 5

    obstacle_frontal: bool = False
    obstacle_near: bool = False
    obstacle_emergency: bool = False
    pedestrian_in_path: bool = False
    stop_sign_ahead: bool = False
    red_light: bool = False

    empty_frames_count: int = 0

    def update(self, detections: List[Detection],
               zones: Dict[str, List[Detection]],
               gps_direction: GPSDirection,
               gps_intensity: float,
               truck_minimap_xy: Tuple[float, float],
               traffic_light_state: Optional[str],
               lane_info: Optional[LaneInfo] = None,
               collision_info: Optional[CollisionInfo] = None,
               speed_info: Optional[SpeedInfo] = None):
        self.detections = detections
        self.zones = zones
        self.gps_direction = gps_direction
        self.gps_intensity = gps_intensity
        self.truck_minimap_xy = truck_minimap_xy
        self.traffic_light_state = traffic_light_state
        self.lane_info = lane_info
        self.collision_info = collision_info
        self.collision_active = (collision_info is not None and
                                 collision_info.collision_detected)
        self.speed_info = speed_info
        self.current_speed_kmh = speed_info.speed_kmh if speed_info else 0.0

        self.detection_history.append(detections)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

        frontal = zones.get("frontal", [])
        capo = zones.get("capo", [])

        self.obstacle_frontal = len(frontal) > 0
        self.obstacle_near = len(capo) > 0
        self.obstacle_emergency = any(
            d.area > 5000 for d in capo + frontal
            if d.class_name in ("car", "truck", "bus", "motorcycle")
        )
        self.pedestrian_in_path = any(
            d.class_name == "person" for d in capo + frontal
        )
        self.stop_sign_ahead = any(
            d.class_name == "stop_sign" for d in frontal
        )
        self.red_light = (traffic_light_state == "red")

        if len(detections) == 0:
            self.empty_frames_count += 1
        else:
            self.empty_frames_count = 0

    def is_recovery_mode(self, threshold: int = 3) -> bool:
        return self.empty_frames_count >= threshold

    def is_emergency_stop(self, threshold: int = 10) -> bool:
        return self.empty_frames_count >= threshold
