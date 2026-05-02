"""
Blackboard compartido para comunicación entre behaviors.
py_trees blackboard Client accesible sin ciclos de importación.
"""

import py_trees

BB = py_trees.blackboard.Client(name="Driving")
BB.register_key("action", access=py_trees.common.Access.WRITE)
BB.register_key("reverse_requested", access=py_trees.common.Access.WRITE)
BB.register_key("camera_look_angle", access=py_trees.common.Access.WRITE)
BB.register_key("collision_recovered", access=py_trees.common.Access.WRITE)
