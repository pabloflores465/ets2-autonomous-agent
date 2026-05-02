"""
Controlador por teclado puro para ETS2.
Arrow keys: up=acelerar, down=frenar/reversa, left/right=steering.
PWM para steering proporcional.
"""

import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


class Controller:
    """Emula teclado con arrow keys para ETS2."""

    def __init__(self, config: dict):
        act = config["actuation"]
        self.accel_key = act["accel_key"]  # "up"
        self.brake_key = act["brake_key"]  # "down"
        self.handbrake_key = act["handbrake_key"]  # "space"
        self.reverse_key = act.get("reverse_key", "down")
        self.steer_left = act.get("steer_left", "left")
        self.steer_right = act.get("steer_right", "right")
        self.verbose = config.get("debug", {}).get("verbose_log", False)

        self._steering = 0.0  # grados actuales
        self._steer_pressed = None  # "left" | "right" | None
        self._accel = False
        self._brake = False
        self._reverse = False
        self._handbrake = False
        self._last_log = ""

    def execute(
        self,
        action,
        duration_ms: int = 50,
        reverse_requested: bool = False,
        camera_look_angle: float = 0.0,
    ):
        """
        Ejecuta acción con teclado.
        Steering usa PWM para simular giro parcial:
          - 100% = mantener tecla presionada
          - 50% = presionar 25ms, soltar 25ms (se reparte entre ciclos)
        """

        if self.verbose and str(action) != self._last_log:
            print(f"  [CTRL] {action}")
            self._last_log = str(action)

        # ── Steering (PWM binario) ──
        steer = action.steer
        pwm_duty = min(1.0, abs(steer) / 20.0)  # 0-20° → 0-100% duty
        steer_on = pwm_duty > 0.15  # umbral mínimo

        if steer_on and steer != self._steering:
            if steer < 0:
                pyautogui.keyUp(self.steer_right)
                pyautogui.keyDown(self.steer_left)
                self._steer_pressed = "left"
            else:
                pyautogui.keyUp(self.steer_left)
                pyautogui.keyDown(self.steer_right)
                self._steer_pressed = "right"
            self._steering = steer
        elif not steer_on and self._steering != 0.0:
            self._release_steer()
            self._steering = 0.0

        # ── Reverse ──
        if reverse_requested:
            if not self._reverse:
                pyautogui.keyDown(self.reverse_key)
                self._reverse = True
            self._release_key("accel")
            self._release_key("brake")
            return
        else:
            self._release_key("reverse")

        # ── Acelerar / Frenar ──
        if action.accelerate > 0.3:
            if not self._accel:
                pyautogui.keyDown(self.accel_key)
                self._accel = True
            self._release_key("brake")
        elif action.brake > 0.3:
            if not self._brake:
                pyautogui.keyDown(self.brake_key)
                self._brake = True
            self._release_key("accel")
        else:
            self._release_key("accel")
            self._release_key("brake")

        # ── Handbrake ──
        if action.handbrake:
            if not self._handbrake:
                pyautogui.keyDown(self.handbrake_key)
                self._handbrake = True
        else:
            self._release_key("handbrake")

    def _release_key(self, which: str):
        key_map = {
            "accel": (self.accel_key, "_accel"),
            "brake": (self.brake_key, "_brake"),
            "reverse": (self.reverse_key, "_reverse"),
            "handbrake": (self.handbrake_key, "_handbrake"),
        }
        if which in key_map:
            key, attr = key_map[which]
            if getattr(self, attr):
                pyautogui.keyUp(key)
                setattr(self, attr, False)

    def _release_steer(self):
        if self._steer_pressed == "left":
            pyautogui.keyUp(self.steer_left)
        elif self._steer_pressed == "right":
            pyautogui.keyUp(self.steer_right)
        self._steer_pressed = None

    def emergency_stop(self):
        print("[CTRL] EMERGENCY STOP")
        for key in [
            self.accel_key,
            self.brake_key,
            self.reverse_key,
            self.steer_left,
            self.steer_right,
            self.handbrake_key,
        ]:
            pyautogui.keyUp(key)
        self._accel = self._brake = self._reverse = self._handbrake = False
        self._steer_pressed = None
        self._steering = 0.0
