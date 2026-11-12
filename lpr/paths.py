"""Work-directory independent path resolution for the CLI entry points.

Every script in ``plate_detection/`` can be launched two ways::

    cd plate_detection && python predict.py --weights runs/smoke_fix/checkpoints/best.pth
    python plate_detection/predict.py --weights runs/smoke_fix/checkpoints/best.pth   # repo root

The short forms in the README are written relative to the project directory, so a
bare ``runs/...`` only exists when the shell is already inside it.  ``resolve_path``
removes that ambiguity: a path that exists as given (i.e. relative to the current
directory) is used unchanged, otherwise it is interpreted relative to the project
root.  Absolute paths are always returned untouched.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """Resolve ``path`` against the cwd first, then against ``base``."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return base / p
