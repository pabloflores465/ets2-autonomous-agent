import py_trees
from src.decision.context import WorldContext, DrivingAction
from src.perception.minimap import GPSDirection
from src.perception.lane_detector import LaneType


class LaneFollow(py_trees.behaviour.Behaviour):
    """
    Seguir carril fusionando GPS (minimapa) + detección de líneas.
    Conducción conservadora: reduce velocidad en curvas.
    """

    GPS_WEIGHT = 0.6
    LANE_WEIGHT = 0.4
    MAX_STEER = 25.0

    CURVE_SPEED = 45.0         # km/h objetivo en curvas
    STRAIGHT_SPEED = 70.0      # km/h objetivo en recta

    def __init__(self, name: str, world: WorldContext, config: dict = None):
        super().__init__(name)
        self.world = world

    def update(self) -> py_trees.common.Status:
        gps = self.world.gps_direction
        gps_int = self.world.gps_intensity
        lane_info = self.world.lane_info

        # Steering GPS
        steer_gps = 0.0
        if gps == GPSDirection.TURN_LEFT:
            steer_gps = -self.MAX_STEER * gps_int * self.GPS_WEIGHT
        elif gps == GPSDirection.TURN_RIGHT:
            steer_gps = self.MAX_STEER * gps_int * self.GPS_WEIGHT

        # Corrección de carril
        steer_lane = 0.0
        if lane_info is not None and lane_info.lane_type == LaneType.PAINTED:
            steer_lane = -lane_info.offset_norm * self.MAX_STEER * self.LANE_WEIGHT

        steer = steer_gps + steer_lane
        steer = max(-self.MAX_STEER, min(self.MAX_STEER, steer))

        # Aceleración conservadora según curva
        is_curving = gps_int > 0.3
        speed = self._get_speed()

        if is_curving and speed > self.CURVE_SPEED:
            accelerate = 0.0     # soltar en curva
        elif gps == GPSDirection.UNKNOWN:
            accelerate = 0.3     # precaución si no hay GPS
        else:
            accelerate = 0.7     # normal en recta

        self.root.blackboard.driving_action = DrivingAction(
            "lane_follow", accelerate=accelerate, brake=0.0, steer=steer)

        return py_trees.common.Status.SUCCESS

    def _get_speed(self) -> float:
        if self.world.speed_info is not None:
            return self.world.speed_info.speed_kmh
        return 40.0  # valor por defecto conservador
