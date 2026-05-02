import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext
from src.perception.lane_detector import LaneType
from src.perception.minimap import GPSDirection


class LaneFollow(py_trees.behaviour.Behaviour):
    """
    Seguir camino usando cámara + GPS.
    - Si hay líneas de carril: 90% cámara, 10% GPS (conducción precisa)
    - Si es terracería (DIRT): 80% GPS, usa minimapa como guía principal
    - Si no hay info de carril: 100% GPS
    """

    MAX_STEER = 25.0

    def __init__(self, name: str, world: WorldContext, config: dict = None):
        super().__init__(name)
        self.world = world
        self._prev_steer = 0.0

    def update(self) -> py_trees.common.Status:
        gps = self.world.gps_direction
        gps_int = self.world.gps_intensity
        speed = self._get_speed()
        lane = self.world.lane_info

        # ── Steering limit según velocidad ──
        if speed < 5:
            max_steer = 5.0
        elif speed < 30:
            max_steer = 12.0
        elif speed < 60:
            max_steer = 20.0
        else:
            max_steer = self.MAX_STEER

        # ── Elegir pesos según tipo de camino ──
        if lane is not None and lane.lane_type == LaneType.PAINTED:
            # Camino con líneas: usar carril como guía principal
            gps_weight = 0.1
            lane_weight = 0.9
        elif lane is not None and lane.lane_type == LaneType.DIRT:
            # Terracería: minimapa guía, sin líneas pintadas
            gps_weight = 0.8
            lane_weight = 0.2  # usar borde del camino si es detectable
        else:
            # Sin info de carril: solo GPS
            gps_weight = 1.0
            lane_weight = 0.0

        # ── Steering por carril (si hay) ──
        steer_lane = 0.0
        if lane is not None and lane.lane_type == LaneType.PAINTED:
            steer_lane = -lane.offset_norm * max_steer * lane_weight

        # ── Steering por GPS (minimapa) ──
        steer_gps = 0.0
        if gps in (GPSDirection.TURN_LEFT, GPSDirection.TURN_RIGHT) and gps_int > 0.15:
            if gps == GPSDirection.TURN_LEFT:
                steer_gps = -max_steer * gps_int * gps_weight
            else:
                steer_gps = max_steer * gps_int * gps_weight

        steer = steer_lane + steer_gps
        steer = max(-max_steer, min(max_steer, steer))

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
