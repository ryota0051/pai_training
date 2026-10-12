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


def test_best_per_class_picks_highest_score_of_each_label():
    from pai_training.detection.coco import best_per_class
    scores = [0.6, 0.9, 0.7, 0.55]
    labels = [0, 1, 0, 1]
    boxes = [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]]
    best = best_per_class(scores, labels, boxes)
    assert best == {0: ([2, 2, 3, 3], 0.7), 1: ([1, 1, 2, 2], 0.9)}
    assert best_per_class([], [], []) == {}


def test_recent_median_rejects_single_outlier_and_resets():
    from pai_training.detection.coco import RecentMedian
    m = RecentMedian(window=5)
    for v in [(0.10, 0.10, 0.02), (0.101, 0.10, 0.02), (0.10, 0.099, 0.02)]:
        m.update(v, episode=1)
    out = m.update((0.09, 0.13, 0.02), episode=1)          # 一部が隠れた瞬間の外れ値
    assert out == pytest.approx((0.10, 0.10, 0.02), abs=1e-3)
    assert m.update(None, episode=1) == pytest.approx((0.10, 0.10, 0.02), abs=1e-3)  # 検出なしは直前の値
    assert m.update(None, episode=2) is None                # エピソードが変わったら捨てる
