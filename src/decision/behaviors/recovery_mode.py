import py_trees
from src.decision.context import WorldContext, DrivingAction


class RecoveryMode(py_trees.behaviour.Behaviour):
    """
    Recuperación ante pérdida de detecciones.
    
    Si YOLO no detecta nada por varios frames consecutivos
    (oscuridad, niebla, túnel, cámara tapada, bug visual),
    el agente reduce velocidad progresivamente hasta detenerse.
    
    Fases:
      - 3-9 frames vacíos  → reducir a 30 km/h
      - 10+ frames vacíos  → STOP total
      - Recupera detección → volver a modo normal
    """

    CAUTION_SPEED = 30.0      # km/h en modo precaución
    CAUTION_FRAMES = 3        # frames sin detección para entrar
    STOP_FRAMES = 10           # frames sin detección para STOP

    def __init__(self, name: str, world: WorldContext, config: dict = None):
        super().__init__(name)
        self.world = world

    def update(self) -> py_trees.common.Status:
        empty = self.world.empty_frames_count

        if empty < self.CAUTION_FRAMES:
            return py_trees.common.Status.FAILURE

        # Estamos en recovery
        speed = self.world.current_speed_kmh

        if empty >= self.STOP_FRAMES:
            # STOP total
            self.root.blackboard.driving_action = DrivingAction(
                "recovery_stop", accelerate=0.0, brake=1.0, steer=0.0)
            return py_trees.common.Status.SUCCESS

        elif empty >= self.CAUTION_FRAMES:
            # Reducir velocidad
            if speed > self.CAUTION_SPEED:
                self.root.blackboard.driving_action = DrivingAction(
                    "recovery_caution", accelerate=0.0, brake=0.5, steer=0.0)
            else:
                self.root.blackboard.driving_action = DrivingAction(
                    "recovery_caution", accelerate=0.3, brake=0.0, steer=0.0)
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE
