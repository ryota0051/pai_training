from pai_training.episode_discovery import find_episode_dirs


def _ep(root, *parts):
    d = root.joinpath(*parts)
    d.mkdir(parents=True)
    (d / 'x_0.db3').write_bytes(b'')
    return d


def test_finds_in_natural_order_and_skips_underscore_dirs(tmp_path):
    a = tmp_path / 'a'
    _ep(a, 'ep10'); _ep(a, 'ep2'); _ep(a, '_discarded', 'ep3')
    assert [p.name for p in find_episode_dirs([a])] == ['ep2', 'ep10']


def test_multiple_dirs_are_concatenated_in_given_order(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    _ep(a, 'ep1'); _ep(b, 'ep0')
    assert [p.parent.name + '/' + p.name for p in find_episode_dirs([a, b])] == ['a/ep1', 'b/ep0']


def test_accepts_single_path_string(tmp_path):
    _ep(tmp_path, 'ep0')
    assert len(find_episode_dirs(str(tmp_path))) == 1
