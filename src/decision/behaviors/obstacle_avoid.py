import py_trees

from src.decision.context import DrivingAction, WorldContext


class ObstacleAvoid(py_trees.behaviour.Behaviour):
    """Evitar colisión con vehículo/obstáculo frontal."""

    def __init__(self, name: str, world: WorldContext, config: dict):
        super().__init__(name)
        self.world = world

    def update(self):
        if self.world.obstacle_near or self.world.obstacle_frontal:
            self.root.blackboard.driving_action = DrivingAction(
                "obstacle_avoid", accelerate=0.0, brake=0.6, steer=0.0
            )
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
