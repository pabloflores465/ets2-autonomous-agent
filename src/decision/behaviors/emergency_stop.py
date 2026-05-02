import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext


class EmergencyStop(py_trees.behaviour.Behaviour):
    """Frenado de emergencia ante peligro inminente."""

    def __init__(self, name: str, world: WorldContext, config: dict):
        super().__init__(name)
        self.world = world

    def update(self):
        if self.world.obstacle_emergency:
            BB.action = DrivingAction(
                "emergency_stop", accelerate=0.0, brake=1.0, steer=0.0, handbrake=True
            )
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
