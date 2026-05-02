import py_trees

from src.decision.context import DrivingAction, WorldContext


class StopSignBehavior(py_trees.behaviour.Behaviour):
    """Detención ante señal de alto."""

    def __init__(self, name: str, world: WorldContext, config: dict):
        super().__init__(name)
        self.world = world

    def update(self):
        if self.world.stop_sign_ahead:
            self.root.blackboard.driving_action = DrivingAction(
                "stop_sign", accelerate=0.0, brake=1.0, steer=0.0
            )
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
