import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext


class ObstacleAvoid(py_trees.behaviour.Behaviour):
    """Evitar colisión: frena y gira para esquivar obstáculo frontal/lateral."""

    STEER_AWAY = 18.0  # grados de giro evasivo

    def __init__(self, name: str, world: WorldContext, config: dict):
        super().__init__(name)
        self.world = world

    def update(self):
        if not (self.world.obstacle_near or self.world.obstacle_frontal):
            return py_trees.common.Status.FAILURE

        # Determinar dirección de esquiva según zona del obstáculo
        steer = 0.0
        zones = self.world.zones

        # Obstáculo muy cerca en capó → frenar fuerte + girar al lado libre
        capo_dets = zones.get("capo", [])
        frontal_dets = zones.get("frontal", [])

        # Revisar laterales para girar hacia el lado despejado
        lateral_izq = zones.get("lateral_izq", [])
        lateral_der = zones.get("lateral_der", [])

        if capo_dets:
            # Obstáculo inmediato: frenar + girar al lado con menos obstáculos
            steer = self._steer_to_clear_side(lateral_izq, lateral_der)
            brake = 0.8
        elif any(d.class_name in ("car", "truck", "bus") for d in frontal_dets):
            # Vehículo adelante: frenar + girar al lado despejado
            steer = self._steer_to_clear_side(lateral_izq, lateral_der)
            brake = 0.5
        else:
            # Obstáculo menor (persona, objeto): frenar leve
            steer = self._steer_to_clear_side(lateral_izq, lateral_der)
            brake = 0.3

        BB.action = DrivingAction("obstacle_avoid", accelerate=0.0, brake=brake, steer=steer)
        return py_trees.common.Status.SUCCESS

    def _steer_to_clear_side(self, left_dets: list, right_dets: list) -> float:
        """Decide dirección de giro para esquivar obstáculos."""
        if left_dets and not right_dets:
            return self.STEER_AWAY  # girar derecha
        elif right_dets and not left_dets:
            return -self.STEER_AWAY  # girar izquierda
        elif left_dets and right_dets:
            # Ambos lados ocupados: girar al que tenga objetos más lejanos
            left_max_area = max((d.area for d in left_dets), default=0)
            right_max_area = max((d.area for d in right_dets), default=0)
            if left_max_area > right_max_area:
                return self.STEER_AWAY  # girar derecha (menos peligroso)
            else:
                return -self.STEER_AWAY  # girar izquierda
        # Sin laterales: no girar, solo frenar
        return 0.0
