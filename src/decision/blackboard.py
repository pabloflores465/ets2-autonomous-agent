"""
Shared state entre behaviors y main loop.
Objeto Python simple, sin magia de py_trees blackboard.
"""

from src.decision.context import DrivingAction


class SharedState:
    """Estado compartido thread-safe entre behaviors y main loop."""

    def __init__(self):
        self.action: DrivingAction = DrivingAction.idle()
        self.reverse_requested: bool = False
        self.camera_look_angle: float = 0.0
        self.collision_recovered: bool = False

    def reset(self):
        self.action = DrivingAction.idle()
        self.reverse_requested = False
        self.camera_look_angle = 0.0


# Instancia única compartida
BB = SharedState()
