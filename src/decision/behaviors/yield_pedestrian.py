import py_trees

from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext


class YieldPedestrian(py_trees.behaviour.Behaviour):
    """Ceder el paso a peatones."""

    def __init__(self, name: str, world: WorldContext, config: dict):
        super().__init__(name)
        self.world = world

    def update(self):
        if self.world.pedestrian_in_path:
            BB.action = DrivingAction("yield_pedestrian", accelerate=0.0, brake=1.0, steer=0.0)
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
