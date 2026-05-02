"""
Controlador del vehículo vía pyautogui.
Mouse steering + teclado W/S/Space + camera look lateral.
"""

import time

import pyautogui

from src.decision.context import DrivingAction

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


class Controller:
    """Emula mouse y teclado para controlar ETS2 vía Parsec."""

    def __init__(self, config: dict):
        act = config["actuation"]
        self.steering_sensitivity = act["steering_sensitivity"]
        self.center_x = act["steering_center_x"]
        self.center_y = act["steering_center_y"]
        self.accel_key = act["accel_key"]
        self.brake_key = act["brake_key"]
        self.handbrake_key = act["handbrake_key"]
        self.reverse_key = act.get("reverse_key", "down")

        self._current_steer = 0.0
        self._accel_pressed = False
        self._brake_pressed = False
        self._handbrake_pressed = False
        self._reverse_pressed = False
        self._camera_look_angle = 0.0

    def execute(self, action: DrivingAction, duration_ms: int = 50,
                reverse_requested: bool = False,
                camera_look_angle: float = 0.0):
        """Ejecuta una acción de conducción."""
        # Camera look tiene prioridad sobre steering
        if camera_look_angle != self._camera_look_angle:
            self._look_camera(camera_look_angle)
        else:
            self._steer(action.steer, duration_ms)

        # Reverse
        if reverse_requested:
            if not self._reverse_pressed:
                pyautogui.keyDown(self.reverse_key)
                self._reverse_pressed = True
            if self._accel_pressed:
                pyautogui.keyUp(self.accel_key)
                self._accel_pressed = False
            if self._brake_pressed:
                pyautogui.keyUp(self.brake_key)
                self._brake_pressed = False
            return
        else:
            if self._reverse_pressed:
                pyautogui.keyUp(self.reverse_key)
                self._reverse_pressed = False

        # Acelerar
        if action.accelerate > 0.3:
            if not self._accel_pressed:
                pyautogui.keyDown(self.accel_key)
                self._accel_pressed = True
            if self._brake_pressed:
                pyautogui.keyUp(self.brake_key)
                self._brake_pressed = False
        elif action.brake > 0.3:
            if not self._brake_pressed:
                pyautogui.keyDown(self.brake_key)
                self._brake_pressed = True
            if self._accel_pressed:
                pyautogui.keyUp(self.accel_key)
                self._accel_pressed = False
        else:
            if self._accel_pressed:
                pyautogui.keyUp(self.accel_key)
                self._accel_pressed = False
            if self._brake_pressed:
                pyautogui.keyUp(self.brake_key)
                self._brake_pressed = False

        # Freno de mano
        if action.handbrake:
            if not self._handbrake_pressed:
                pyautogui.keyDown(self.handbrake_key)
                self._handbrake_pressed = True
        else:
            if self._handbrake_pressed:
                pyautogui.keyUp(self.handbrake_key)
                self._handbrake_pressed = False

    def _look_camera(self, angle: float):
        """Gira la cámara lateralmente (mouse horizontal)."""
        if abs(angle) < 1:
            # Centrar
            pyautogui.moveTo(self.center_x, self.center_y, duration=0.2)
        else:
            dx = int(angle * 8)  # 8px por grado
            pyautogui.moveRel(dx, 0, duration=0.3)
        self._camera_look_angle = angle

    def _steer(self, angle: float, duration_ms: int):
        """Mueve el mouse proporcionalmente al ángulo de giro."""
        dx = int(angle * self.steering_sensitivity)
        if abs(dx) > 0:
            pyautogui.moveRel(dx, 0, duration=duration_ms / 1000)
        else:
            pyautogui.moveTo(self.center_x, self.center_y,
                             duration=duration_ms / 1000)
        self._current_steer = angle

    def emergency_stop(self):
        """Suelta todas las teclas y aplica freno de mano."""
        for key, flag in [(self.accel_key, '_accel_pressed'),
                           (self.brake_key, '_brake_pressed'),
                           (self.reverse_key, '_reverse_pressed')]:
            if getattr(self, flag):
                pyautogui.keyUp(key)
                setattr(self, flag, False)
        pyautogui.keyDown(self.handbrake_key)
        time.sleep(0.1)
        pyautogui.keyUp(self.handbrake_key)
