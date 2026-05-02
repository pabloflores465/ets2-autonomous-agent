import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext


class Overtake(py_trees.behaviour.Behaviour):
    """
    Maniobra de rebase segura con verificación de espejos.
    Pipeline:
      1. Detectar vehículo lento delante (obstacle_frontal > 2s)
      2. Verificar espejo izquierdo + lateral izquierdo libres
      3. Señalizar izquierda (steering) + cambiar carril
      4. Acelerar para adelantar (85 km/h temporal)
      5. Verificar espejo derecho libre
      6. Retornar a carril derecho
      7. Volver a velocidad crucero

    Solo en autopista (recta, GPS STRAIGHT).
    No rebasa en curvas, intersecciones ni ciudad.
    """

    OVERTAKE_SPEED = 85.0  # km/h durante rebase
    LANE_CHANGE_STEER = 8.0  # grados para cambiar carril
    MIN_FRONTAL_TIME = 2.0  # segundos mínimo con vehículo delante
    CLEAR_DISTANCE = 100  # píxeles de margen lateral libre

    def __init__(self, name: str, world: WorldContext, config: dict = None):
        super().__init__(name)
        self.world = world
        self.phase = 0
        # 0=detect, 1=check_left, 2=change_left, 3=pass, 4=check_right, 5=change_right, 6=done
        self.phase_timer = 0.0
        self.last_tick_time = None
        self._frontal_timer = 0.0  # tiempo acumulado con obstáculo frontal

    def initialise(self):
        self.phase = 0
        self.phase_timer = 0.0
        self._frontal_timer = 0.0

    def update(self) -> py_trees.common.Status:
        import time

        now = time.monotonic()

        if self.last_tick_time is None:
            self.last_tick_time = now
        dt = now - self.last_tick_time
        self.last_tick_time = now
        self.phase_timer += dt

        # Solo activar en autopista (GPS recto, alta intensidad de confianza)
        from src.perception.minimap import GPSDirection

        gps = self.world.gps_direction
        if gps != GPSDirection.STRAIGHT and self.phase == 0:
            return py_trees.common.Status.FAILURE

        # Verificar que hay vehículo delante por tiempo suficiente
        if self.phase == 0:
            return self._phase_detect(dt)
        elif self.phase == 1:
            return self._phase_check_left()
        elif self.phase == 2:
            return self._phase_change_left()
        elif self.phase == 3:
            return self._phase_pass()
        elif self.phase == 4:
            return self._phase_check_right()
        elif self.phase == 5:
            return self._phase_return_right()
        elif self.phase == 6:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE

    def _phase_detect(self, dt: float) -> py_trees.common.Status:
        """Acumular tiempo con vehículo delante. Si suficiente, iniciar rebase."""
        if self.world.obstacle_frontal:
            self._frontal_timer += dt
        else:
            self._frontal_timer = max(0, self._frontal_timer - dt)

        if self._frontal_timer >= self.MIN_FRONTAL_TIME:
            self.phase = 1
            self.phase_timer = 0.0
            return self._phase_check_left()

        return py_trees.common.Status.FAILURE

    def _phase_check_left(self) -> py_trees.common.Status:
        """
        Verificar que carril izquierdo está libre.
        Revisa: espejo_izq + lateral_izq + zona frontal completa.
        """
        left_clear = self._check_left_lane()

        # También verificar que podemos ver detrás (espejo sin vehículos cercanos)
        mirror_dets = self.world.zones.get("espejo_izq", [])
        rear_clear = not any(
            d.class_name in ("car", "truck", "bus", "motorcycle")
            for d in mirror_dets
            if d.confidence > 0.4
        )

        if left_clear and rear_clear:
            self.phase = 2
            self.phase_timer = 0.0
            return py_trees.common.Status.RUNNING
        else:
            # Esperar a que se libere (máximo 10s)
            if self.phase_timer > 10.0:
                self.phase = 6  # abortar, timeout
                return py_trees.common.Status.RUNNING
            BB.action = DrivingAction("overtake_wait", accelerate=0.3, brake=0.0, steer=0.0)
            return py_trees.common.Status.RUNNING

    def _phase_change_left(self) -> py_trees.common.Status:
        """Cambiar al carril izquierdo."""
        # Steer suave a la izquierda
        BB.action = DrivingAction(
            "overtake_change_left", accelerate=0.8, brake=0.0, steer=-self.LANE_CHANGE_STEER
        )

        if self.phase_timer > 1.5:
            self.phase = 3
            self.phase_timer = 0.0

        return py_trees.common.Status.RUNNING

    def _phase_pass(self) -> py_trees.common.Status:
        """Acelerar para adelantar."""
        # Verificar que seguimos viendo al vehículo (que no chocamos)
        if self.world.obstacle_emergency:
            # ¡Peligro! Frenar
            BB.action = DrivingAction("overtake_abort", accelerate=0.0, brake=1.0, steer=0.0)
            self.phase = 6
            return py_trees.common.Status.RUNNING

        # Acelerar a fondo para pasar
        BB.action = DrivingAction("overtake_passing", accelerate=1.0, brake=0.0, steer=0.0)

        # Pasar cuando el vehículo ya no está en zona frontal
        # Y podemos verlo en espejo derecho (ya lo pasamos)
        if not self.world.obstacle_frontal:
            # Verificar que el vehículo está en espejo derecho (confirmación visual)
            right_mirror = self.world.zones.get("espejo_der", [])
            vehicle_in_right_mirror = any(
                d.class_name in ("car", "truck", "bus", "motorcycle")
                for d in right_mirror
                if d.confidence > 0.3
            )
            if vehicle_in_right_mirror or self.phase_timer > 4.0:
                self.phase = 4
                self.phase_timer = 0.0

        return py_trees.common.Status.RUNNING

    def _phase_check_right(self) -> py_trees.common.Status:
        """Verificar que podemos retornar al carril derecho."""
        right_clear = self._check_right_lane()

        # Verificar espejo derecho (vehículo pasó o está lejos)
        mirror_dets = self.world.zones.get("espejo_der", [])
        rear_clear = not any(
            d.class_name in ("car", "truck", "bus", "motorcycle")
            for d in mirror_dets
            if d.confidence > 0.5
        )

        if right_clear and rear_clear:
            self.phase = 5
            self.phase_timer = 0.0
            return py_trees.common.Status.RUNNING
        else:
            if self.phase_timer > 8.0:
                # Timeout: retornar de todas formas con precaución
                self.phase = 5
                self.phase_timer = 0.0
                return py_trees.common.Status.RUNNING
            BB.action = DrivingAction("overtake_wait_right", accelerate=0.5, brake=0.0, steer=0.0)
            return py_trees.common.Status.RUNNING

    def _phase_return_right(self) -> py_trees.common.Status:
        """Retornar a carril derecho."""
        BB.action = DrivingAction(
            "overtake_return", accelerate=0.5, brake=0.0, steer=self.LANE_CHANGE_STEER
        )

        if self.phase_timer > 1.5:
            self.phase = 6

        return py_trees.common.Status.RUNNING

    def _check_left_lane(self) -> bool:
        """Verifica que el carril izquierdo no tiene vehículos cercanos."""
        threats = []
        for zone_name in ("frontal", "lateral_izq"):
            for det in self.world.zones.get(zone_name, []):
                if det.class_name in ("car", "truck", "bus", "motorcycle"):
                    if det.confidence > 0.4:
                        threats.append(det)

        # Si hay vehículos en zona izquierda del frontal, verificar distancia
        for det in threats:
            # Vehículo en mitad izquierda del frame = potencial peligro
            if det.center_x < self._frame_center_x():
                return False

        return True

    def _check_right_lane(self) -> bool:
        """Verifica que el carril derecho no tiene vehículos cercanos."""
        threats = []
        for zone_name in ("frontal", "lateral_der"):
            for det in self.world.zones.get(zone_name, []):
                if det.class_name in ("car", "truck", "bus", "motorcycle"):
                    if det.confidence > 0.4:
                        threats.append(det)

        for det in threats:
            if det.center_x > self._frame_center_x():
                return False

        return True

    def _frame_center_x(self) -> float:
        """Devuelve el centro X del frame (asumiendo 1280)."""
        return 640.0
