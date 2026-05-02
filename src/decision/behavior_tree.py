"""
Árbol de Comportamiento para conducción autónoma en ETS2.
Usa py_trees 2.x con blackboard compartido vía SharedState.
"""

import py_trees

from src.decision.behaviors.cruise import Cruise
from src.decision.behaviors.emergency_stop import EmergencyStop
from src.decision.behaviors.lane_follow import LaneFollow
from src.decision.behaviors.obstacle_avoid import ObstacleAvoid
from src.decision.behaviors.stop_sign import StopSignBehavior
from src.decision.behaviors.traffic_light import TrafficLightBehavior
from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext


def build_behavior_tree(world: WorldContext, config: dict) -> py_trees.trees.BehaviourTree:
    """
    BT simplificado para pruebas locales M1:
    Solo comportamientos esenciales.
    """
    root = py_trees.composites.Selector(name="Root", memory=False)

    root.add_children(
        [
            EmergencyStop("EmergencyStop", world, config),
            TrafficLightBehavior("TrafficLight", world, config),
            StopSignBehavior("StopSign", world, config),
            ObstacleAvoid("ObstacleAvoid", world, config),
            LaneFollow("LaneFollow", world, config),
            Cruise("Cruise", world, config),
        ]
    )

    return py_trees.trees.BehaviourTree(root)


def get_active_action(tree: py_trees.trees.BehaviourTree) -> DrivingAction:
    return BB.action
