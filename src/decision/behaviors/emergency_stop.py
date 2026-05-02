import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext


class EmergencyStop(py_trees.behaviour.Behaviour):
    """Frenado de emergencia + giro evasivo si hay espacio."""

    STEER_AWAY = 22.0  # giro brusco para esquivar

    def __init__(self, name: str, world: WorldContext, config: dict):
        super().__init__(name)
        self.world = world

    def update(self):
        if not self.world.obstacle_emergency:
            return py_trees.common.Status.FAILURE

        # Obstáculo enorme y muy cerca: frenar máximo + girar al lado libre
        zones = self.world.zones
        lateral_izq = zones.get("lateral_izq", [])
        lateral_der = zones.get("lateral_der", [])

        steer = 0.0
        if lateral_izq and not lateral_der:
            steer = self.STEER_AWAY  # derecha
        elif lateral_der and not lateral_izq:
            steer = -self.STEER_AWAY  # izquierda
        # Ambos ocupados o ninguno → no girar, solo frenar

        BB.action = DrivingAction(
            "emergency_stop", accelerate=0.0, brake=1.0, steer=steer, handbrake=True
        )
        return py_trees.common.Status.SUCCESS
