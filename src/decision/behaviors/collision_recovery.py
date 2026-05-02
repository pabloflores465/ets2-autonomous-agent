import py_trees
from src.decision.context import WorldContext, DrivingAction


class CollisionRecovery(py_trees.behaviour.Behaviour):
    """
    Recuperación post-colisión.
    Fases:
    1. FRENAR (freno de mano + brake)
    2. ESPERAR (1s para estabilizar)
    3. REVERSA + corregir dirección (mirando GPS minimapa)
    4. AVANZAR de nuevo
    """

    REVERSE_DURATION_S = 2.0     # segundos en reverse
    STEER_CORRECTION_DEG = 20.0  # grados de corrección

    def __init__(self, name: str, world: WorldContext, config: dict = None):
        super().__init__(name)
        self.world = world
        self.phase = 0           # 0=brake, 1=wait, 2=reverse, 3=done
        self.phase_timer = 0.0
        self.last_tick_time = None

    def initialise(self):
        self.phase = 0
        self.phase_timer = 0.0

    def update(self) -> py_trees.common.Status:
        import time
        now = time.monotonic()

        # Solo activar si hay colisión detectada
        if not self.world.collision_active and self.phase == 0:
            return py_trees.common.Status.FAILURE

        if self.world.collision_recovered and self.phase != 0:
            # Ya nos recuperamos, éxito
            self.phase = 0  # reset para próxima
            return py_trees.common.Status.SUCCESS

        if self.last_tick_time is None:
            self.last_tick_time = now

        dt = now - self.last_tick_time
        self.last_tick_time = now
        self.phase_timer += dt

        if self.phase == 0:
            # Frenar fuerte
            self.root.blackboard.driving_action = DrivingAction(
                "collision_brake", accelerate=0.0, brake=1.0, steer=0.0,
                handbrake=True
            )
            if self.phase_timer > 0.5:
                self.phase = 1
                self.phase_timer = 0.0
            return py_trees.common.Status.RUNNING

        elif self.phase == 1:
            # Esperar estabilización
            self.root.blackboard.driving_action = DrivingAction(
                "collision_wait", accelerate=0.0, brake=0.0, steer=0.0,
                handbrake=False
            )
            if self.phase_timer > 1.0:
                self.phase = 2
                self.phase_timer = 0.0
            return py_trees.common.Status.RUNNING

        elif self.phase == 2:
            # Reverse + corregir dirección según GPS
            steer = self._get_reverse_steer()
            action = DrivingAction(
                "collision_reverse", accelerate=0.0, brake=0.0,
                steer=steer, handbrake=False
            )
            # Simular reverse: brake key es S, pero reverse es otra tecla
            # Marcamos en el blackboard que queremos reverse
            action = DrivingAction(
                "collision_reverse", accelerate=0.0, brake=0.0,
                steer=steer, handbrake=False
            )
            setattr(self.root.blackboard, "reverse_requested", True)
            self.root.blackboard.driving_action = action

            if self.phase_timer > self.REVERSE_DURATION_S:
                self.phase = 3
                self.phase_timer = 0.0
                setattr(self.root.blackboard, "reverse_requested", False)
                setattr(self.root.blackboard, "collision_recovered", True)
            return py_trees.common.Status.RUNNING

        elif self.phase == 3:
            # Recuperación completada
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE

    def _get_reverse_steer(self) -> float:
        """
        Calcula steering para reverse.
        Si el GPS dice girar derecha, en reverse es steer izquierdo para
        orientar el camión hacia la ruta.
        """
        from src.perception.minimap import GPSDirection
        gps = self.world.gps_direction
        intensity = self.world.gps_intensity

        if gps == GPSDirection.UNKNOWN:
            return 0.0

        # En reverse, el steering se invierte para corregir orientación
        # Queremos que la flecha azul apunte hacia la línea roja
        truck_x, _ = self.world.truck_minimap_xy
        # Si el camión está a la derecha de la ruta → reverse con steer izq
        if truck_x > 0.55:
            return -self.STEER_CORRECTION_DEG * intensity
        elif truck_x < 0.45:
            return self.STEER_CORRECTION_DEG * intensity
        return 0.0
