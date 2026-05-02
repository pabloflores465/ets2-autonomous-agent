import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext


class TrafficLightBehavior(py_trees.behaviour.Behaviour):
    """Detención ante semáforo rojo."""

    def __init__(self, name: str, world: WorldContext, config: dict):
        super().__init__(name)
        self.world = world

    def update(self):
        if self.world.red_light:
            BB.action = DrivingAction("traffic_light_stop", accelerate=0.0, brake=1.0, steer=0.0)
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
