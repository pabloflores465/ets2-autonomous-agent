import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext
from src.perception.lane_detector import LaneType
from src.perception.minimap import GPSDirection


class LaneFollow(py_trees.behaviour.Behaviour):
    """
    Seguir carril fusionando GPS (minimapa) + detección de líneas.
    Conducción suave: depende de velocidad, histéresis de dirección, sin oscilaciones.
    """

    GPS_WEIGHT = 0.7
    LANE_WEIGHT = 0.3
    MAX_STEER = 25.0  # tope absoluto

    def __init__(self, name: str, world: WorldContext, config: dict = None):
        super().__init__(name)
        self.world = world
        # Histéresis de dirección: recordar última dirección distinta de UNKNOWN/STRAIGHT
        self._prev_turn: GPSDirection | None = None
        self._prev_steer = 0.0

    def update(self) -> py_trees.common.Status:
        gps = self.world.gps_direction
        gps_int = self.world.gps_intensity
        speed = self._get_speed()

        # ── Steering limit según velocidad ──
        if speed < 5:
            # Parado o casi: reducir steering drásticamente para que avance recto
            max_steer = 5.0
        elif speed < 30:
            max_steer = 12.0
        elif speed < 60:
            max_steer = 20.0
        else:
            max_steer = self.MAX_STEER

        # ── Steering GPS con histéresis ──
        steer_gps = 0.0
        if gps in (GPSDirection.TURN_LEFT, GPSDirection.TURN_RIGHT):
            # Solo responder si intensidad es significativa
            if gps_int > 0.15:
                if gps == GPSDirection.TURN_LEFT:
                    steer_gps = -max_steer * gps_int * self.GPS_WEIGHT
                else:
                    steer_gps = max_steer * gps_int * self.GPS_WEIGHT
                self._prev_turn = gps
            else:
                # Intensidad muy baja: mantener dirección previa (histéresis)
                if self._prev_turn == GPSDirection.TURN_LEFT:
                    steer_gps = -max_steer * 0.15 * self.GPS_WEIGHT
                elif self._prev_turn == GPSDirection.TURN_RIGHT:
                    steer_gps = max_steer * 0.15 * self.GPS_WEIGHT
        else:
            # STRAIGHT o UNKNOWN: no forzar giro, permitir que corrección de carril guíe
            self._prev_turn = None

        # ── Lane correction ──
        steer_lane = 0.0
        if self.world.lane_info is not None and self.world.lane_info.lane_type == LaneType.PAINTED:
            steer_lane = -self.world.lane_info.offset_norm * max_steer * self.LANE_WEIGHT

        steer = steer_gps + steer_lane
        steer = max(-max_steer, min(max_steer, steer))

        # Dead zone amplia: ignorar steering muy pequeño (evita twitching)
        if abs(steer) < 0.5:
            steer = 0.0

        # ── Smoothing exponencial entre frames ──
        steer = self._prev_steer * 0.7 + steer * 0.3
        self._prev_steer = steer

        # ── Aceleración ──
        if speed < 3:
            # Parado: acelerar a fondo para arrancar
            accelerate = 1.0
        elif gps_int > 0.5 and gps != GPSDirection.STRAIGHT:
            # Curva cerrada: reducir velocidad
            accelerate = 0.5
        else:
            # Recta: aceleración normal
            accelerate = 0.8

        BB.action = DrivingAction("lane_follow", accelerate=accelerate, brake=0.0, steer=steer)
        return py_trees.common.Status.SUCCESS

    def _get_speed(self) -> float:
        if self.world.speed_info is not None:
            return self.world.speed_info.speed_kmh
        if self.world.current_speed_kmh > 0:
            return self.world.current_speed_kmh
        return 0.0
