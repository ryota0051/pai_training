"""把持・配置タスク: 関節角 + 物体位置 + 置き場所の位置（置き場所もエピソードごとに変える, 実験12）."""
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "gripper"]

features = {
    "observation.state": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
    "observation.environment_state": {"dtype": "float32", "shape": (5,),
                                      "names": ["object_x", "object_y", "object_z", "place_x", "place_y"]},
    "action": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
}
robot_type = "crane_plus"
repo_id = "crane_plus_pick_objpos_place"
task_name = "pick_place"
