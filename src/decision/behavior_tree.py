"""
Árbol de Comportamiento para conducción autónoma en ETS2.
Usa py_trees 2.x con blackboard compartido vía Client.
"""

import py_trees

from src.decision.behaviors.collision_recovery import CollisionRecovery
from src.decision.behaviors.cruise import Cruise
from src.decision.behaviors.emergency_stop import EmergencyStop
from src.decision.behaviors.intersection import IntersectionHandler
from src.decision.behaviors.lane_follow import LaneFollow
from src.decision.behaviors.obstacle_avoid import ObstacleAvoid
from src.decision.behaviors.overtake import Overtake
from src.decision.behaviors.recovery_mode import RecoveryMode
from src.decision.behaviors.stop_sign import StopSignBehavior
from src.decision.behaviors.traffic_light import TrafficLightBehavior
from src.decision.behaviors.yield_pedestrian import YieldPedestrian
from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext


def build_behavior_tree(world: WorldContext, config: dict) -> py_trees.trees.BehaviourTree:
    """
    Construye el árbol de comportamiento.
    Estructura:
        Root (Selector → prioridad)
        ├── EmergencyStop
        ├── CollisionRecovery
        ├── IntersectionHandler
        ├── TrafficLightBehavior
        ├── StopSignBehavior
        ├── ObstacleAvoid
        ├── RecoveryMode
        ├── YieldPedestrian
        ├── Overtake
        ├── LaneFollow
        └── Cruise
    """
    root = py_trees.composites.Selector(name="Root", memory=False)

    emergency = EmergencyStop("EmergencyStop", world, config)
    collision = CollisionRecovery("CollisionRecovery", world, config)
    intersection = IntersectionHandler("IntersectionHandler", world, config)
    traffic_light = TrafficLightBehavior("TrafficLight", world, config)
    stop_sign = StopSignBehavior("StopSign", world, config)
    obstacle_avoid = ObstacleAvoid("ObstacleAvoid", world, config)
    recovery = RecoveryMode("RecoveryMode", world, config)
    yield_pedestrian = YieldPedestrian("YieldPedestrian", world, config)
    overtake = Overtake("Overtake", world, config)
    lane_follow = LaneFollow("LaneFollow", world, config)
    cruise = Cruise("Cruise", world, config)

    root.add_children(
        [
            emergency,
            collision,
            intersection,
            traffic_light,
            stop_sign,
            obstacle_avoid,
            recovery,
            yield_pedestrian,
            overtake,
            lane_follow,
            cruise,
        ]
    )

    return py_trees.trees.BehaviourTree(root)


def get_active_action(tree: py_trees.trees.BehaviourTree) -> DrivingAction:
    """Extrae la acción del state compartido después de un tick."""
    return BB.action
