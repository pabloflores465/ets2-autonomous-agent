"""
Controlador por AppleScript key codes para ETS2 en macOS.
Arrow keys: up=126, down=125, left=123, right=124, space=49.
"""

import subprocess


class Controller:
    """Envía keystrokes a ETS2 vía osascript (System Events)."""

    KEY_CODES = {
        "up": 126,
        "down": 125,
        "left": 123,
        "right": 124,
        "space": 49,
    }

    def __init__(self, config: dict):
        act = config["actuation"]
        self.accel_key = act["accel_key"]
        self.brake_key = act["brake_key"]
        self.handbrake_key = act["handbrake_key"]
        self.reverse_key = act.get("reverse_key", "down")
        self.steer_left = act.get("steer_left", "left")
        self.steer_right = act.get("steer_right", "right")
        self.verbose = config.get("debug", {}).get("verbose_log", False)

        self._steering = 0.0
        self._steer_pressed = None
        self._accel = False
        self._brake = False
        self._reverse = False
        self._handbrake = False
        self._last_log = ""
        self._last_key = ""  # evitar spam de prints

    def execute(
        self,
        action,
        duration_ms: int = 50,
        reverse_requested: bool = False,
        camera_look_angle: float = 0.0,
    ):
        if self.verbose and str(action) != self._last_log:
            print(f"  [CTRL] {action}")
            self._last_log = str(action)

        steer = action.steer
        steer_on = abs(steer) > 0.5

        # ── Steering ──
        if steer_on:
            if steer < 0 and self._steer_pressed != "left":
                self._send_key(self.steer_left)
                self._steer_pressed = "left"
            elif steer > 0 and self._steer_pressed != "right":
                self._send_key(self.steer_right)
                self._steer_pressed = "right"
        elif self._steer_pressed is not None:
            self._steer_pressed = None  # soltar: el key code ya es press+release

        # ── Reverse ──
        if reverse_requested:
            if not self._reverse:
                self._send_key(self.reverse_key)
                self._reverse = True
            return
        elif self._reverse:
            self._reverse = False

        # ── Acelerar / Frenar ──
        if action.accelerate > 0.3:
            if not self._accel:
                self._send_key(self.accel_key)
                self._accel = True
            self._brake = False
        elif action.brake > 0.3:
            if not self._brake:
                self._send_key(self.brake_key)
                self._brake = True
            self._accel = False
        else:
            self._accel = False
            self._brake = False

        # ── Handbrake ──
        if action.handbrake:
            if not self._handbrake:
                self._send_key(self.handbrake_key)
                self._handbrake = True
        else:
            self._handbrake = False

    def _send_key(self, key_name: str):
        """Envía un key press vía osascript."""
        code = self.KEY_CODES.get(key_name)
        if code is None:
            return
        if self.verbose and key_name != self._last_key:
            print(f"  [KEY] {key_name}")
            self._last_key = key_name
        try:
            subprocess.run(
                ["osascript", "-e", f'tell application "System Events" to key code {code}'],
                capture_output=True,
                timeout=1,
            )
        except Exception:
            pass

    def emergency_stop(self):
        print("[CTRL] EMERGENCY STOP")
        self._accel = self._brake = self._reverse = self._handbrake = False
        self._steer_pressed = None
        self._steering = 0.0
