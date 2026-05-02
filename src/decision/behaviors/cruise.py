import py_trees
from src.decision.context import WorldContext, DrivingAction


class Cruise(py_trees.behaviour.Behaviour):
    """
    Avance normal sin obstáculos.
    Conducción conservadora:
      - Acelera hasta velocidad segura (~70 km/h en recta, menos en curvas)
      - Si va muy rápido (>85 km/h), deja de acelerar
      - Prioriza estabilidad sobre velocidad
    """

    CRUISE_SPEED = 70.0       # km/h objetivo en recta
    MAX_SPEED = 85.0           # km/h máximo (el camión limita a 90)
    CURVE_SPEED = 45.0         # km/h en curvas

    def __init__(self, name: str, world: WorldContext, config: dict = None):
        super().__init__(name)
        self.world = world

    def update(self) -> py_trees.common.Status:
        speed = self._get_speed()

        # Determinar aceleración según velocidad actual
        accelerate = 1.0
        if speed > self.MAX_SPEED:
            accelerate = 0.0  # soltar acelerador
        elif speed > self.CRUISE_SPEED:
            accelerate = 0.3  # aceleración suave
        elif speed > self.CURVE_SPEED and self._is_curving():
            accelerate = 0.0  # no acelerar en curvas si vamos rápido
        elif speed < 10.0:
            accelerate = 1.0  # arranque: acelerar a fondo

        self.root.blackboard.driving_action = DrivingAction(
            "cruise", accelerate=accelerate, brake=0.0, steer=0.0)

        return py_trees.common.Status.SUCCESS

    def _get_speed(self) -> float:
        """Obtiene velocidad actual del world context."""
        if self.world.speed_info is not None:
            return self.world.speed_info.speed_kmh
        return 30.0  # asumir velocidad moderada si no hay datos

    def _is_curving(self) -> bool:
        """Determina si estamos en una curva basado en GPS intensity."""
        return self.world.gps_intensity > 0.3
