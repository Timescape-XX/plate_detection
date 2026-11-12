"""CCPD2019 dataset integrity checker.

Verifies:
  1. every path listed in data/CCPD2019/splits/*.txt exists on disk
  2. every image on disk is referenced by at least one split
  3. no zero-byte / truncated (missing JPEG EOI marker) files
  4. random sample is fully decodable by PIL and filename annotations parse

Usage:
  python tools/check_dataset.py [--data-root plate_detection/data/CCPD2019] [--sample 3000]
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
]
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_name(name: str) -> dict:
    """Parse a CCPD filename into its annotation fields."""
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"too few fields: {stem}")
    area, tilt, bbox, vertices, labels = parts[0], parts[1], parts[2], parts[3], parts[4]

    def pts(field: str) -> list[tuple[int, int]]:
        return [tuple(int(v) for v in p.split("&")) for p in field.split("_")]

    return {
        "area": int(area.split("_")[0]),
        "tilt": [int(x) for x in tilt.split("_")],
        "bbox": pts(bbox),          # 2 corner points of the axis-aligned box
        "vertices": pts(vertices),  # 4 plate corners
        "labels": [int(i) for i in labels.split("_")],
    }


def jpeg_ok(path: Path) -> tuple[bool, str]:
    """Structural check: SOI at head, EOI at tail, plausible size.

    Returns (ok, reason). ``reason == "junk"`` marks AppleDouble resource-fork
    leftovers (``\\x00\\x05\\x16\\x07`` magic) that are not images at all.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"stat error {exc}"
    if size == 0:
        return False, "zero bytes"
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
            fh.seek(-2, 2)
            tail = fh.read(2)
    except OSError as exc:
        return False, f"read error {exc}"
    if head[:4] == b"\x00\x05\x16\x07":
        return False, "junk"
    if size < 512:
        return False, f"suspiciously small ({size} bytes)"
    if head[:2] != b"\xff\xd8":
        return False, f"bad SOI {head[:2]!r}"
    if tail != b"\xff\xd9":
        return False, f"bad EOI (truncated?) {tail!r}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "data" / "CCPD2019"))
    ap.add_argument("--sample", type=int, default=3000, help="number of images to fully decode")
    args = ap.parse_args()

    root = Path(args.data_root)
    splits_dir = root / "splits"
    if not root.is_dir():
        print(f"[FATAL] data root not found: {root}")
        return 2
    if not splits_dir.is_dir():
        print(f"[FATAL] splits dir not found: {splits_dir}")
        return 2

    # ---------------------------------------------------------------- disk side
    # ccpd_base is shipped one level too deep in this copy: ccpd_base/ccpd_base/*.jpg
    def resolve(rel: str) -> Path:
        p = root / rel
        if p.exists():
            return p
        parts = Path(rel).parts
        if len(parts) >= 2:
            alt = root / parts[0] / parts[0] / Path(*parts[1:])
            if alt.exists():
                return alt
        return p

    on_disk: dict[str, list[str]] = {}
    nested_dirs: set[str] = set()
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name == "splits":
            continue
        files = [f.name for f in sub.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if (sub / sub.name).is_dir():
            nested_dirs.add(sub.name)
            files += [f.name for f in (sub / sub.name).iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        on_disk[sub.name] = files

    print("== disk inventory ==")
    total_disk = 0
    for k, v in on_disk.items():
        dup = len(v) - len(set(v))
        flag = f"  [duplicate names: {dup}]" if dup else ""
        print(f"  {k:<16} {len(v):>7}{flag}")
        total_disk += len(v)
    print(f"  {'TOTAL':<16} {total_disk:>7}")
    for k in sorted(nested_dirs):
        print(f"  note: {k}/ is nested one level deeper ({k}/{k}/); handled transparently")

    def dir_of(sub_name: str) -> Path:
        base = root / sub_name
        return base / sub_name if sub_name in nested_dirs else base

    referenced: dict[str, set[str]] = {}
    missing: list[str] = []
    dup_refs: list[str] = []
    unparsable: list[str] = []
    for txt in sorted(splits_dir.glob("*.txt")):
        lines = [ln.strip() for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        bad = 0
        seen: set[str] = set()
        for rel in lines:
            if not resolve(rel).is_file():
                missing.append(rel)
                bad += 1
                continue
            split_name, fname = rel.split("/", 1)
            if rel in seen:                      # duplicate *paths* only: the same
                dup_refs.append(f"{txt.stem}:{rel}")  # basename may exist in two subsets
            seen.add(rel)
            referenced.setdefault(split_name, set()).add(fname)
            try:
                ann = parse_name(fname)
                assert len(ann["labels"]) in (7, 8), "label count"
                assert len(ann["vertices"]) == 4, "vertex count"
                assert all(0 <= v < 65 for v in ann["labels"]), "label range"
            except Exception:  # noqa: BLE001
                unparsable.append(rel)
        print(f"  split {txt.stem:<14} {len(lines):>7} listed, {bad} missing, {len(seen) - (len(lines) - bad)} dup")

    print("\n== coverage (files on disk not referenced by any split) ==")
    uncovered_total = 0
    for k, files in on_disk.items():
        used = referenced.get(k, set())
        uncovered = [f for f in set(files) if f not in used]
        uncovered_total += len(uncovered)
        print(f"  {k:<16} referenced {len(used):>7} / on-disk {len(set(files)):>7}, unreferenced {len(uncovered):>6}")

    print("\n== file structure scan (all images) ==")
    broken: list[tuple[str, str]] = []
    junk: list[str] = []
    scanned = 0
    for sub_name, files in on_disk.items():
        base = dir_of(sub_name)
        for name in files:
            ok, why = jpeg_ok(base / name)
            scanned += 1
            if ok:
                continue
            if why == "junk":
                junk.append(f"{sub_name}/{name}")
            else:
                broken.append((f"{sub_name}/{name}", why))
    print(f"  scanned {scanned} files: {len(broken)} broken, {len(junk)} non-image junk")
    for path, why in broken[:20]:
        print(f"      BROKEN {path}: {why}")
    if junk:
        print(f"      junk e.g. {junk[:3]}"
              f"{' ...' if len(junk) > 3 else ''} (AppleDouble resource forks, not usable images)")
    junk_by_dir = Counter(p.split("/")[0] for p in junk)

    print("\n== random full-decode + annotation sample ==")
    rng = random.Random(0)
    junk_set = set(junk)
    pool = [(k, f) for k, fs in on_disk.items() for f in fs if f"{k}/{f}" not in junk_set]
    sample = rng.sample(pool, min(args.sample, len(pool)))
    from PIL import Image

    decode_fail: list[tuple[str, str]] = []
    shapes: Counter = Counter()
    label_ok = label_bad = 0
    label_bad_examples: list[str] = []
    for sub_name, fname in sample:
        p = dir_of(sub_name) / fname
        try:
            with Image.open(p) as im:
                im.load()
                shapes[im.size] += 1
        except Exception as exc:  # noqa: BLE001
            decode_fail.append((str(p), repr(exc)))
            continue
        try:
            parse_name(fname)
            label_ok += 1
        except Exception:  # noqa: BLE001
            label_bad += 1
            if len(label_bad_examples) < 3:
                label_bad_examples.append(f"{sub_name}/{fname}")
    print(f"  decoded {len(sample) - len(decode_fail)}/{len(sample)} ok, {len(decode_fail)} failures")
    print(f"  annotation parse: {label_ok} ok, {label_bad} unparsable"
          + (f" (e.g. {label_bad_examples})" if label_bad_examples else ""))
    print(f"  image sizes: {dict(shapes.most_common(6))}")
    for p, why in decode_fail[:10]:
        print(f"      {p}: {why}")

    # ------------------------------------------------------------------ verdict
    # Unlabeled folders (e.g. ccpd_np) legitimately contain no annotations, so
    # unparsable filenames there are not counted as errors.
    unlabeled_dirs = {k for k in on_disk if not referenced.get(k)}
    print(f"\n  unlabeled / unreferenced folders: {sorted(unlabeled_dirs) or 'none'}")

    print("\n== verdict ==")
    problems = []
    if missing:
        problems.append(f"{len(missing)} split entries missing on disk")
    if broken:
        problems.append(f"{len(broken)} truncated/unreadable images")
    if decode_fail:
        problems.append(f"{len(decode_fail)} images failed full decode")
    if unparsable:
        problems.append(f"{len(unparsable)} labelled filenames failed annotation parsing")
    if problems:
        print("  [FAIL] " + "; ".join(problems))
        return 1
    print("  [OK] labelled data complete: every split entry exists, every image decodes")
    if junk:
        print(f"  warn: {len(junk)} AppleDouble junk files (not images) in {dict(junk_by_dir)}")
        print("        these are not part of any split and can be ignored")
    if uncovered_total:
        print(f"  note: {uncovered_total} on-disk files are referenced by no split "
              f"(spare pool incl. unlabeled folders)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
