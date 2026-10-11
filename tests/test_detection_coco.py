import pytest

from pai_training.detection.coco import split_ids, to_target, flip_target


def test_to_target_normalizes_xywh_to_cxcywh():
    t = to_target([{'bbox': [10, 20, 30, 40], 'category_id': 0}], width=100, height=200)
    assert t['class_labels'].tolist() == [0]
    assert t['boxes'].tolist()[0] == pytest.approx([0.25, 0.2, 0.3, 0.2])


def test_to_target_empty():
    t = to_target([], 100, 100)
    assert t['boxes'].shape == (0, 4)


def test_flip_target_mirrors_x_center():
    t = to_target([{'bbox': [10, 20, 30, 40], 'category_id': 0}], 100, 200)
    f = flip_target(t)
    assert f['boxes'].tolist()[0] == pytest.approx([0.75, 0.2, 0.3, 0.2])


def test_split_ids_is_deterministic_and_disjoint():
    a_tr, a_va = split_ids(range(100), val_ratio=0.2, seed=0)
    b_tr, b_va = split_ids(range(100), val_ratio=0.2, seed=0)
    assert a_tr == b_tr and a_va == b_va
    assert len(a_va) == 20 and not set(a_tr) & set(a_va)
