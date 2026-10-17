"""把持・配置タスク: 関節角 + 俯瞰カメラ画像（物体位置・置き場所は入れない, A 案）.

物体と置き場所の位置は方策が画像から読み取る。画像の大きさは worlds/pick_place.sdf の overhead_camera と同じ。
"""
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "gripper"]

features = {
    "observation.state": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
    "observation.images.overhead": {"dtype": "video", "shape": (240, 320, 3),
                                    "names": ["height", "width", "channels"]},
    "action": {"dtype": "float32", "shape": (5,), "names": JOINT_NAMES},
}
robot_type = "crane_plus"
repo_id = "crane_plus_pick_image"
task_name = "pick_place"
