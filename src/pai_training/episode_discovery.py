"""rosbag2 エピソードディレクトリの探索（複数の収録ディレクトリをまとめて 1 つのデータセットにする）."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Union

PathLike = Union[str, Path]


def natural_key(path: Path):
    """ep2 < ep10 となるよう数字部分を数値として比較する."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(path))]


def find_episode_dirs(bag_dirs: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    """*.db3 を含むディレクトリを、bag_dirs の順に（各ディレクトリ内は自然順で）返す.

    `_` で始まるディレクトリ（`_discarded` など）配下は除外する。
    """
    if isinstance(bag_dirs, (str, Path)):
        bag_dirs = [bag_dirs]
    out: List[Path] = []
    for d in bag_dirs:
        root = Path(d).expanduser().resolve()
        found = {p.parent for p in root.rglob('*.db3')
                 if not any(part.startswith('_') for part in p.relative_to(root).parts)}
        out.extend(sorted(found, key=natural_key))
    return out
