import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext
from src.perception.minimap import GPSDirection


class IntersectionHandler(py_trees.behaviour.Behaviour):
    """
    Maneja cruce de intersecciones.
    Pipeline:
      1. GPS indica giro fuerte → entrar modo intersección
      2. Reducir velocidad a 20 km/h
      3. Mirar lateral correspondiente (cámara + espejo)
      4. Verificar YOLO en zona lateral y espejo
      5. Si vía libre → cruzar (steering GPS)
      6. Si vehículo → esperar hasta 5s, luego avanzar lento
      7. Si timeout → avanzar con precaución

    Requiere que controller soporte camera_look (mouse mirar izq/der).
    """

    INTERSECTION_SPEED = 20.0  # km/h al aproximar
    CROSS_SPEED = 15.0  # km/h al cruzar
    LOOK_ANGLE = 60.0  # grados de giro de cámara
    LOOK_DURATION_S = 1.5  # segundos mirando cada lado
    MAX_WAIT_S = 5.0  # timeout espera

    def __init__(self, name: str, world: WorldContext, config: dict = None):
        super().__init__(name)
        self.world = world
        self.phase = 0  # 0=approach, 1=look_left, 2=look_right, 3=wait, 4=cross, 5=done
        self.phase_timer = 0.0
        self.last_tick_time = None
        self.wait_start = 0.0
        self.look_direction = "left"

    def initialise(self):
        self.phase = 0
        self.phase_timer = 0.0
        self.wait_start = 0.0

    def update(self) -> py_trees.common.Status:
        import time

        now = time.monotonic()

        # Solo activar si GPS indica giro fuerte (intersección)
        gps = self.world.gps_direction
        gps_int = self.world.gps_intensity

        is_intersection = gps in (GPSDirection.TURN_LEFT, GPSDirection.TURN_RIGHT) and gps_int > 0.5
        if not is_intersection and self.phase == 0:
            return py_trees.common.Status.FAILURE

        if self.last_tick_time is None:
            self.last_tick_time = now
        dt = now - self.last_tick_time
        self.last_tick_time = now
        self.phase_timer += dt

        if self.phase == 0:
            return self._phase_approach()
        elif self.phase == 1:
            return self._phase_look(dt, now)
        elif self.phase == 2:
            return self._phase_wait(now)
        elif self.phase == 3:
            return self._phase_cross()
        elif self.phase == 4:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE

    def _phase_approach(self) -> py_trees.common.Status:
        """Reducir velocidad al aproximar intersección."""
        BB.action = DrivingAction("intersection_approach", accelerate=0.0, brake=0.3, steer=0.0)
        if self.phase_timer > 0.5:
            self.phase = 1
            self.phase_timer = 0.0
            # Determinar hacia dónde mirar primero
            if self.world.gps_direction == GPSDirection.TURN_LEFT:
                self.look_direction = "left"
            else:
                self.look_direction = "right"
        return py_trees.common.Status.RUNNING

    def _phase_look(self, dt: float, now: float) -> py_trees.common.Status:
        """
        Mirar lateral + verificar espejo.
        La cámara se controla vía blackboard.camera_look_angle.
        """
        # Alternar mirada: primero un lado, luego el otro
        if self.phase_timer < self.LOOK_DURATION_S:
            # Mirar primer lado
            angle = self.LOOK_ANGLE if self.look_direction == "left" else -self.LOOK_ANGLE
            setattr(BB, "camera_look_angle", angle)

            # Verificar espejo de ese lado
            self._check_side(self.look_direction)
            BB.action = DrivingAction("intersection_look", accelerate=0.0, brake=0.5, steer=0.0)

        elif self.phase_timer < self.LOOK_DURATION_S * 2:
            # Mirar el otro lado
            other = "right" if self.look_direction == "left" else "left"
            angle = self.LOOK_ANGLE if other == "left" else -self.LOOK_ANGLE
            setattr(BB, "camera_look_angle", angle)

            self._check_side(other)
            BB.action = DrivingAction("intersection_look", accelerate=0.0, brake=0.5, steer=0.0)
        else:
            # Volver cámara al frente
            setattr(BB, "camera_look_angle", 0.0)

            if self._both_sides_clear():
                self.phase = 3  # cruzar
                self.phase_timer = 0.0
            else:
                self.phase = 2  # esperar
                self.phase_timer = 0.0
                self.wait_start = now

        return py_trees.common.Status.RUNNING

    def _phase_wait(self, now: float) -> py_trees.common.Status:
        """Esperar a que la vía lateral esté libre."""
        BB.action = DrivingAction("intersection_wait", accelerate=0.0, brake=1.0, steer=0.0)

        if self._both_sides_clear():
            self.phase = 3
            self.phase_timer = 0.0
            return py_trees.common.Status.RUNNING

        if (now - self.wait_start) > self.MAX_WAIT_S:
            # Timeout: avanzar con precaución
            BB.action = DrivingAction(
                "intersection_yield_timeout", accelerate=0.3, brake=0.0, steer=0.0
            )
            self.phase = 3
            self.phase_timer = 0.0

        return py_trees.common.Status.RUNNING

    def _phase_cross(self) -> py_trees.common.Status:
        """Cruzar la intersección."""
        gps = self.world.gps_direction
        gps_int = self.world.gps_intensity

        steer = 0.0
        if gps == GPSDirection.TURN_LEFT:
            steer = -20.0 * gps_int
        elif gps == GPSDirection.TURN_RIGHT:
            steer = 20.0 * gps_int

        BB.action = DrivingAction("intersection_cross", accelerate=0.5, brake=0.0, steer=steer)

        # Terminar cuando el GPS deja de indicar giro fuerte
        if gps_int < 0.3:
            self.phase = 4
            setattr(BB, "camera_look_angle", 0.0)
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.RUNNING

    def _check_side(self, side: str) -> bool:
        """
        Verifica si un lado está libre de tráfico.
        Revisa zona lateral + espejo correspondiente.
        """
        zone_name = f"lateral_{side}"
        mirror_name = f"espejo_{side}"

        lateral_dets = self.world.zones.get(zone_name, [])
        mirror_dets = self.world.zones.get(mirror_name, [])

        # Vehículos en lateral o espejo
        threats = [
            d
            for d in lateral_dets + mirror_dets
            if d.class_name in ("car", "truck", "bus", "motorcycle")
        ]

        return len(threats) == 0

    def _both_sides_clear(self) -> bool:
        return self._check_side("left") and self._check_side("right")
