import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext
from src.perception.lane_detector import LaneType
from src.perception.minimap import GPSDirection


class LaneFollow(py_trees.behaviour.Behaviour):
    """
    Seguir carril fusionando GPS (minimapa) + detección de líneas.
    Conducción conservadora: reduce velocidad en curvas.
    """

    GPS_WEIGHT = 0.6
    LANE_WEIGHT = 0.4
    MAX_STEER = 8.0  # reducido: era 25.0

    CURVE_SPEED = 45.0  # km/h objetivo en curvas
    STRAIGHT_SPEED = 70.0  # km/h objetivo en recta

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

        # Dead zone: ignorar steering muy pequeño
        if abs(steer) < 0.3:
            steer = 0.0

        # Aceleración: siempre positivo en lane_follow, salvo que venga curva fuerte
        is_curving = gps_int > 0.6
        if is_curving:
            accelerate = 0.4  # reducir en curva cerrada
        else:
            accelerate = 1.0  # acelerar a fondo en recta

        BB.action = DrivingAction("lane_follow", accelerate=accelerate, brake=0.0, steer=steer)

        return py_trees.common.Status.SUCCESS

    def _get_speed(self) -> float:
        if self.world.speed_info is not None:
            return self.world.speed_info.speed_kmh
        return 40.0  # valor por defecto conservador
