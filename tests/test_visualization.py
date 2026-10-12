from pai_training.visualization import entity_paths, feature_names

FEATURES = {
    'observation.state': {'shape': (2,), 'names': ['joint1', 'gripper']},
    'observation.environment_state': {'shape': (3,), 'names': ['object_x', 'object_y', 'object_z']},
    'action': {'shape': (2,), 'names': ['joint1', 'gripper']},
    'nameless': {'shape': (2,), 'names': None},
}


def test_feature_names_falls_back_to_indices():
    assert feature_names(FEATURES, 'action') == ['joint1', 'gripper']
    assert feature_names(FEATURES, 'nameless') == ['0', '1']


def test_entity_paths_group_command_and_state_per_joint():
    assert entity_paths('action', ['joint1', 'gripper']) == ['joints/joint1/action', 'joints/gripper/action']
    assert entity_paths('observation.state', ['joint1']) == ['joints/joint1/state']
    assert entity_paths('observation.environment_state', ['object_x']) == ['object/object_x']
    assert entity_paths('other', ['a']) == ['other/a']
