"""
Controlador del vehículo vía pyautogui.
Mouse steering + teclado W/S/Space + camera look lateral.
Incluye logging de teclas presionadas para debug.
"""

import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


class Controller:
    """Emula mouse y teclado para controlar ETS2."""

    def __init__(self, config: dict):
        act = config["actuation"]
        self.steering_sensitivity = act["steering_sensitivity"]
        self.center_x = act["steering_center_x"]
        self.center_y = act["steering_center_y"]
        self.accel_key = act["accel_key"]
        self.brake_key = act["brake_key"]
        self.handbrake_key = act["handbrake_key"]
        self.reverse_key = act.get("reverse_key", "down")
        self.verbose = config.get("debug", {}).get("verbose_log", False)

        self._current_steer = 0.0
        self._accel_pressed = False
        self._brake_pressed = False
        self._handbrake_pressed = False
        self._reverse_pressed = False
        self._camera_look_angle = 0.0
        self._last_action_logged = ""

    def execute(
        self,
        action,
        duration_ms: int = 50,
        reverse_requested: bool = False,
        camera_look_angle: float = 0.0,
    ):
        """Ejecuta una acción de conducción."""
        # Log si cambió la acción
        if self.verbose and str(action) != self._last_action_logged:
            print(f"  [CTRL] {action}")
            self._last_action_logged = str(action)

        # Camera look
        if camera_look_angle != self._camera_look_angle:
            self._look_camera(camera_look_angle)
        else:
            self._steer(action.steer, duration_ms)

        # Reverse
        if reverse_requested:
            if not self._reverse_pressed:
                print(f"  [CTRL] keyDown({self.reverse_key}) REVERSE")
                pyautogui.keyDown(self.reverse_key)
                self._reverse_pressed = True
            self._release_key(self.accel_key, "_accel_pressed")
            self._release_key(self.brake_key, "_brake_pressed")
            return
        else:
            self._release_key(self.reverse_key, "_reverse_pressed")

        # Acelerar
        if action.accelerate > 0.3:
            if not self._accel_pressed:
                print(f"  [CTRL] keyDown({self.accel_key})")
                pyautogui.keyDown(self.accel_key)
                self._accel_pressed = True
            self._release_key(self.brake_key, "_brake_pressed")
        elif action.brake > 0.3:
            if not self._brake_pressed:
                print(f"  [CTRL] keyDown({self.brake_key})")
                pyautogui.keyDown(self.brake_key)
                self._brake_pressed = True
            self._release_key(self.accel_key, "_accel_pressed")
        else:
            self._release_key(self.accel_key, "_accel_pressed")
            self._release_key(self.brake_key, "_brake_pressed")

        # Freno de mano
        if action.handbrake:
            if not self._handbrake_pressed:
                print(f"  [CTRL] keyDown({self.handbrake_key})")
                pyautogui.keyDown(self.handbrake_key)
                self._handbrake_pressed = True
        else:
            self._release_key(self.handbrake_key, "_handbrake_pressed")

    def _release_key(self, key, attr):
        if getattr(self, attr):
            print(f"  [CTRL] keyUp({key})")
            pyautogui.keyUp(key)
            setattr(self, attr, False)

    def _look_camera(self, angle: float):
        if abs(angle) < 1:
            pyautogui.moveTo(self.center_x, self.center_y, duration=0.2)
        else:
            dx = int(angle * 8)
            pyautogui.moveRel(dx, 0, duration=0.3)
        self._camera_look_angle = angle

    def _steer(self, angle: float, duration_ms: int):
        dx = int(angle * self.steering_sensitivity)
        if abs(dx) > 0:
            pyautogui.moveRel(dx, 0, duration=duration_ms / 1000)
        else:
            pyautogui.moveTo(self.center_x, self.center_y, duration=duration_ms / 1000)
        self._current_steer = angle

    def emergency_stop(self):
        print("[CTRL] EMERGENCY STOP - releasing all keys")
        for key, flag in [
            (self.accel_key, "_accel_pressed"),
            (self.brake_key, "_brake_pressed"),
            (self.reverse_key, "_reverse_pressed"),
            (self.handbrake_key, "_handbrake_pressed"),
        ]:
            if getattr(self, flag):
                pyautogui.keyUp(key)
                setattr(self, flag, False)
