import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext
from src.perception.lane_detector import LaneType
from src.perception.minimap import GPSDirection


class LaneFollow(py_trees.behaviour.Behaviour):
    """
    Seguir carril usando cámara (líneas de carril) como fuente principal.
    GPS del minimapa solo para navegación general (intersecciones).
    90% lane detection, 10% GPS.
    """

    GPS_WEIGHT = 0.1   # minimapa: solo nudge suave en curvas
    LANE_WEIGHT = 0.9  # cámara: seguimiento de carril principal
    MAX_STEER = 25.0

    def __init__(self, name: str, world: WorldContext, config: dict = None):
        super().__init__(name)
        self.world = world
        self._prev_steer = 0.0

    def update(self) -> py_trees.common.Status:
        gps = self.world.gps_direction
        gps_int = self.world.gps_intensity
        speed = self._get_speed()

        # ── Steering limit según velocidad ──
        if speed < 5:
            max_steer = 5.0
        elif speed < 30:
            max_steer = 12.0
        elif speed < 60:
            max_steer = 20.0
        else:
            max_steer = self.MAX_STEER

        # ── Steering: 90% carril (cámara) ──
        steer_lane = 0.0
        if self.world.lane_info is not None and self.world.lane_info.lane_type == LaneType.PAINTED:
            steer_lane = -self.world.lane_info.offset_norm * max_steer * self.LANE_WEIGHT

        # ── GPS: solo 10%, solo para curvas pronunciadas (intersecciones) ──
        steer_gps = 0.0
        if gps in (GPSDirection.TURN_LEFT, GPSDirection.TURN_RIGHT) and gps_int > 0.3:
            if gps == GPSDirection.TURN_LEFT:
                steer_gps = -max_steer * gps_int * self.GPS_WEIGHT
            else:
                steer_gps = max_steer * gps_int * self.GPS_WEIGHT

        steer = steer_lane + steer_gps
        steer = max(-max_steer, min(max_steer, steer))

        # Dead zone
        if abs(steer) < 0.5:
            steer = 0.0

        # ── Smoothing ──
        steer = self._prev_steer * 0.7 + steer * 0.3
        self._prev_steer = steer

        # ── Aceleración ──
        if speed < 3:
            accelerate = 1.0
        elif gps_int > 0.5 and gps != GPSDirection.STRAIGHT:
            accelerate = 0.5
        else:
            accelerate = 0.8

        BB.action = DrivingAction("lane_follow", accelerate=accelerate, brake=0.0, steer=steer)
        return py_trees.common.Status.SUCCESS

    def _get_speed(self) -> float:
        if self.world.speed_info is not None:
            return self.world.speed_info.speed_kmh
        if self.world.current_speed_kmh > 0:
            return self.world.current_speed_kmh
        return 0.0
