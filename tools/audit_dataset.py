"""CCPD2019 dataset audit: path validity, split coverage, per-folder province skew.

The training target of this project is *plate text*, so the question that matters
is not only "does every path exist" but "which province characters does the
training pool actually contain".  This script answers both, and dumps a JSON
report used by the figures in ``REPORT.md``.

    python tools/audit_dataset.py                       # full audit
    python tools/audit_dataset.py --out runs/audit/dataset_audit.json
    python tools/audit_dataset.py --no-exists           # skip os.stat pass (fast)

Checks
------
1. every ``splits/*.txt`` entry resolves to a real file (no nested-dir fudging)
2. no image on disk is orphaned (only ``ccpd_np`` is exempt: no annotations)
3. no leftover nested ``<sub>/<sub>/`` directory (the historical layout bug)
4. every filename annotation parses; province index in range; text length sane
5. province histogram per subdirectory + for each split -> the skew that makes
   the recogniser collapse to "皖"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lpr.charset import ALPHABETS, PROVINCES  # noqa: E402

# subdirectories that carry no usable annotation (pure digits, negative samples)
SKIP_DIRS = {"ccpd_np"}
NON_IMAGE_SUFFIXES = {".txt", ".json", ".bat", ".ds_store"}


def parse_labels(stem: str) -> list[int] | None:
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    try:
        labels = [int(v) for v in parts[4].split("_")]
    except ValueError:
        return None
    if not 6 <= len(labels) <= 8:
        return None
    if not 0 <= labels[0] < len(PROVINCES):
        return None
    if any(not 0 <= v < len(ALPHABETS) for v in labels[1:]):
        return None
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "CCPD2019"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "audit" / "dataset_audit.json"))
    ap.add_argument("--no-exists", action="store_true", help="skip the per-file existence pass")
    args = ap.parse_args()

    root = Path(args.data_root)
    if not root.is_dir():
        print(f"[FATAL] data root not found: {root}")
        return 2

    # ---------------------------------------------------------------- 1. layout
    subdirs = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "splits")
    nested = [d for d in subdirs if (root / d / d).is_dir()]
    print(f"[layout] data root      : {root}")
    print(f"[layout] subdirectories : {', '.join(subdirs)}")
    print(f"[layout] nested {root.name}/<sub>/<sub>/ : "
          f"{'STILL PRESENT -> ' + ', '.join(nested) if nested else 'none (fixed)'}")

    # ------------------------------------------------------- 2. inventory on disk
    disk: dict[str, set[str]] = {}
    per_dir: dict[str, Counter] = {}
    bad_annotation: dict[str, int] = defaultdict(int)
    n_images = 0
    for sub in subdirs:
        files: set[str] = set()
        prov = Counter()
        for p in (root / sub).iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() != ".jpg":
                bad_annotation[f"{sub}:non-jpg"] += 1
                continue
            files.add(p.name)
            labels = parse_labels(p.stem)
            if labels is None:
                bad_annotation[sub] += 1
                continue
            prov[PROVINCES[labels[0]]] += 1
        disk[sub] = files
        per_dir[sub] = prov
        n_images += len(files)
        top = ", ".join(f"{k} {v / max(sum(prov.values()), 1):.1%}" for k, v in prov.most_common(4))
        print(f"[disk ] {sub:16s} {len(files):7d} jpg   province: {top}")
    print(f"[disk ] total images: {n_images}")

    # -------------------------------------------------------------- 3. split files
    splits: dict[str, dict] = {}
    referenced: dict[str, set[str]] = defaultdict(set)
    for txt in sorted((root / "splits").glob("*.txt")):
        lines = [ln.strip() for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        missing = []
        folders = Counter()
        prov = Counter()
        for rel in lines:
            rel_posix = rel.replace("\\", "/")
            folder, _, name = rel_posix.rpartition("/")
            folders[folder] += 1
            if folder in disk and name in disk[folder]:
                referenced[folder].add(name)
                labels = parse_labels(Path(name).stem)
                if labels is not None:
                    prov[PROVINCES[labels[0]]] += 1
            else:
                missing.append(rel_posix)
        n_ok = len(lines) - len(missing)
        splits[txt.stem] = {
            "file": str(txt),
            "lines": len(lines),
            "resolved": n_ok,
            "missing": len(missing),
            "missing_examples": missing[:5],
            "folders": dict(folders),
            "provinces": dict(prov),
        }
        print(f"[split] {txt.name:20s} lines={len(lines):7d} resolved={n_ok:7d} "
              f"missing={len(missing):5d} folders={dict(folders)}")
        if missing:
            print(f"        e.g. missing: {missing[:3]}")

    # ------------------------------------------------------------ 4. orphan files
    orphans = {}
    for sub, files in disk.items():
        if sub in SKIP_DIRS:
            continue
        orphan = len(files - referenced.get(sub, set()))
        if orphan:
            orphans[sub] = orphan
    print(f"[cover] orphan images (on disk, in no split): "
          f"{orphans if orphans else 'none'}")

    # ------------------------------------------------------- 5. global skew summary
    all_prov = Counter()
    for prov in per_dir.values():
        all_prov.update(prov)
    tot = sum(all_prov.values()) or 1
    skew = {k: v / tot for k, v in all_prov.most_common()}
    print(f"[skew ] global province distribution: "
          + ", ".join(f"{k} {v:.2%}" for k, v in list(skew.items())[:6])
          + f" | classes present: {len(skew)}/{len(PROVINCES)}")

    report = {
        "data_root": str(root),
        "subdirs": subdirs,
        "nested_dirs_remaining": nested,
        "images_on_disk": n_images,
        "per_dir_counts": {k: len(v) for k, v in disk.items()},
        "per_dir_provinces": {k: dict(v.most_common()) for k, v in per_dir.items()},
        "bad_annotation": dict(bad_annotation),
        "splits": splits,
        "orphans": orphans,
        "global_provinces": dict(all_prov.most_common()),
        "global_province_ratio": {k: round(v, 6) for k, v in all_prov.most_common()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done ] report -> {out}")
    return 0 if not nested else 1


if __name__ == "__main__":
    raise SystemExit(main())
