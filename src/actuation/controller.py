"""
Controlador por AppleScript key down/up para ETS2 en macOS.
Mantiene teclas presionadas entre ciclos para acelerar/girar.
"""

import subprocess


class Controller:
    """Envía key down/up a ETS2 vía osascript (System Events)."""

    KEY_CODES = {"up": 126, "down": 125, "left": 123, "right": 124, "space": 49}

    def __init__(self, config: dict):
        act = config["actuation"]
        self.accel_key = act["accel_key"]
        self.brake_key = act["brake_key"]
        self.handbrake_key = act["handbrake_key"]
        self.reverse_key = act.get("reverse_key", "down")
        self.steer_left = act.get("steer_left", "left")
        self.steer_right = act.get("steer_right", "right")
        self.verbose = config.get("debug", {}).get("verbose_log", False)

        # Teclas actualmente presionadas
        self._held = set()
        self._last_log = ""
        self._steer_smooth = 0.0  # steering suavizado

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

        desired = set()

        # ── Steering con suavizado ──
        steer = action.steer
        # Suavizado suave: 50% anterior + 50% nuevo
        self._steer_smooth = self._steer_smooth * 0.5 + steer * 0.5
        steer = self._steer_smooth

        if steer < -0.5:
            desired.add(self.steer_left)
        elif steer > 0.5:
            desired.add(self.steer_right)

        # ── Acelerar / Frenar / Reverse ──
        if reverse_requested:
            desired.add(self.reverse_key)
        elif action.accelerate > 0.3:
            desired.add(self.accel_key)
        elif action.brake > 0.3:
            desired.add(self.brake_key)

        # ── Handbrake (solo tap, no hold) ──
        if action.handbrake:
            self._tap(self.handbrake_key)

        # ── Aplicar cambios: soltar las que ya no se necesitan, presionar nuevas ──
        to_release = self._held - desired
        to_press = desired - self._held

        for key in to_release:
            self._key_up(key)
        for key in to_press:
            self._key_down(key)

        self._held = desired

    def _key_down(self, key_name: str):
        code = self.KEY_CODES.get(key_name)
        if code is None:
            return
        if self.verbose:
            print(f"  [KEY] DOWN {key_name}")
        self._osascript(f"key down {code}")

    def _key_up(self, key_name: str):
        code = self.KEY_CODES.get(key_name)
        if code is None:
            return
        if self.verbose:
            print(f"  [KEY] UP   {key_name}")
        self._osascript(f"key up {code}")

    def _tap(self, key_name: str):
        code = self.KEY_CODES.get(key_name)
        if code is None:
            return
        if self.verbose:
            print(f"  [KEY] TAP {key_name}")
        self._osascript(f"key code {code}")

    def _osascript(self, cmd: str):
        try:
            subprocess.run(
                ["osascript", "-e", f'tell application "System Events" to {cmd}'],
                capture_output=True,
                timeout=1,
            )
        except Exception:
            pass

    def emergency_stop(self):
        print("[CTRL] EMERGENCY STOP - release all")
        for key in list(self._held):
            self._key_up(key)
        self._held.clear()
