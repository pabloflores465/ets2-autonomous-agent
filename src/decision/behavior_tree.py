"""
Árbol de Comportamiento para conducción autónoma en ETS2.
Usa py_trees para la estructura jerárquica de decisiones.
"""

import py_trees

from src.decision.behaviors.emergency_stop import EmergencyStop
from src.decision.behaviors.collision_recovery import CollisionRecovery
from src.decision.behaviors.intersection import IntersectionHandler
from src.decision.behaviors.obstacle_avoid import ObstacleAvoid
from src.decision.behaviors.traffic_light import TrafficLightBehavior
from src.decision.behaviors.stop_sign import StopSignBehavior
from src.decision.behaviors.yield_pedestrian import YieldPedestrian
from src.decision.behaviors.lane_follow import LaneFollow
from src.decision.behaviors.cruise import Cruise
from src.decision.context import WorldContext, DrivingAction


class DrivingAction:
    """Acción de conducción resultante del BT."""

    def __init__(self, behavior: str, accelerate: float = 0.0,
                 brake: float = 0.0, steer: float = 0.0,
                 handbrake: bool = False):
        self.behavior = behavior       # nombre del comportamiento activo
        self.accelerate = accelerate   # 0.0 a 1.0
        self.brake = brake             # 0.0 a 1.0
        self.steer = steer             # grados (-izq, +der)
        self.handbrake = handbrake

    def __repr__(self):
        return (f"DrivingAction({self.behavior}, accel={self.accelerate:.2f}, "
                f"brake={self.brake:.2f}, steer={self.steer:.1f}°)")

    @staticmethod
    def idle():
        return DrivingAction("idle", accelerate=0.0, brake=0.0, steer=0.0)


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
        ├── YieldPedestrian
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
    yield_pedestrian = YieldPedestrian("YieldPedestrian", world, config)
    lane_follow = LaneFollow("LaneFollow", world, config)
    cruise = Cruise("Cruise", world, config)

    root.add_children([
        emergency,
        collision,
        intersection,
        traffic_light,
        stop_sign,
        obstacle_avoid,
        yield_pedestrian,
        lane_follow,
        cruise,
    ])

    return py_trees.trees.BehaviourTree(root)


def get_active_action(tree: py_trees.trees.BehaviourTree) -> DrivingAction:
    """
    Extrae la acción del blackboard después de un tick.
    """
    bb = tree.root.blackboard
    action = getattr(bb, "driving_action", None)
    if action is None:
        return DrivingAction.idle()
    return action
