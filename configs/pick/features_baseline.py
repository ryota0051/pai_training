"""把持・配置タスク（ベースライン）: 演習と同じく関節角のみ（environment_state も関節角）."""
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "gripper"]

features = {
    "observation.state": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
    "observation.environment_state": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
}
robot_type = "crane_plus"
repo_id = "crane_plus_pick_baseline"
task_name = "pick_place"
