"""
Árbol de Comportamiento para conducción autónoma en ETS2.
Usa py_trees 2.x con blackboard compartido vía SharedState.

⚠ Obstacle detection desactivada temporalmente (imprecisa).
Solo: LaneFollow → Cruise → comportamientos de tráfico.
"""

import py_trees

from src.decision.behaviors.cruise import Cruise
from src.decision.behaviors.lane_follow import LaneFollow
from src.decision.behaviors.stop_sign import StopSignBehavior
from src.decision.behaviors.traffic_light import TrafficLightBehavior
from src.decision.blackboard import BB
from src.decision.context import DrivingAction, WorldContext


def build_behavior_tree(world: WorldContext, config: dict) -> py_trees.trees.BehaviourTree:
    """
    BT simplificado: solo LaneFollow + Cruise + señales.
    ObstacleAvoid y EmergencyStop desactivados por ahora.
    """
    root = py_trees.composites.Selector(name="Root", memory=False)

    root.add_children(
        [
            TrafficLightBehavior("TrafficLight", world, config),
            StopSignBehavior("StopSign", world, config),
            LaneFollow("LaneFollow", world, config),
            Cruise("Cruise", world, config),
        ]
    )

    return py_trees.trees.BehaviourTree(root)


def get_active_action(tree: py_trees.trees.BehaviourTree) -> DrivingAction:
    return BB.action
