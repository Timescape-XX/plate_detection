"""Build training / validation / test splits for the province-balanced retrain.

Why a new split is unavoidable
------------------------------
CCPD2019's official ``splits/train.txt`` is 100 % ``ccpd_base`` and 95.9 % 皖:
only **4 106** non-皖 plates total, i.e. ~137 per province.  The other
sub-directories (blur / challenge / db / fn / rotate / tilt / weather) hold
another 25 407 non-皖 plates but the official ``test.txt`` consumes all of them
as test data, so they could not be trained on.

This script therefore carves the non-base pool into train / val / test and, on
top, builds a **province-uniform diagnostic set** (31 provinces x N) so that
per-province accuracy can be measured with a sane error bar.

Outputs (under ``data/CCPD2019/splits/``)
-----------------------------------------
``train_mix.txt``   base train (100 k) + 85 % of the non-base pool   -> training
``val_base.txt``    4 k from ``ccpd_base`` val, natural 96 % 皖 prior -> legacy-comparable val
``val_mix.txt``     2 k base + 2 k non-皖 + 1 k province-uniform    -> per-step monitoring
``test_prov.txt``   31 provinces x 150 = 4 650, uniform             -> headline diagnostic
``test_mix.txt``    natural-prior holdout of the non-base pool      -> realistic robustness

Disjointness is enforced and asserted: a filename may appear in at most one
output split (the val/test members are removed from the training pool).

    python tools/make_splits.py --per-province 150 --val-base 4000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lpr.charset import PROVINCES  # noqa: E402

BASE_DIR = "ccpd_base"
# every annotated sub-directory that is NOT ccpd_base (ccpd_np has no labels)
POOL_DIRS = ["ccpd_blur", "ccpd_challenge", "ccpd_db", "ccpd_fn",
             "ccpd_rotate", "ccpd_tilt", "ccpd_weather"]


def province_of(name: str) -> str | None:
    parts = Path(name).stem.split("-")
    if len(parts) < 5:
        return None
    try:
        idx = int(parts[4].split("_")[0])
    except ValueError:
        return None
    return PROVINCES[idx] if 0 <= idx < len(PROVINCES) else None


def scan(root: Path, folder: str) -> dict[str, list[str]]:
    """``folder`` -> ``{province: [relative paths]}``."""
    out: dict[str, list[str]] = defaultdict(list)
    for p in (root / folder).glob("*.jpg"):
        prov = province_of(p.name)
        if prov is not None:
            out[prov].append(f"{folder}/{p.name}")
    for v in out.values():
        v.sort()                      # deterministic before shuffling
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "CCPD2019"))
    ap.add_argument("--per-province", type=int, default=150,
                    help="images per province in the province-uniform diagnostic set")
    ap.add_argument("--val-base", type=int, default=4000)
    ap.add_argument("--val-mix-base", type=int, default=2000)
    ap.add_argument("--val-mix-nonwan", type=int, default=2000)
    ap.add_argument("--val-mix-unif", type=int, default=1000)
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="fraction of the remaining non-base pool held out as test_mix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    splits_dir = root / "splits"
    rng = random.Random(args.seed)

    # ------------------------------------------------------------ source pools
    base_by_prov = scan(root, BASE_DIR)
    pool: dict[str, dict[str, list[str]]] = {d: scan(root, d) for d in POOL_DIRS}
    n_pool = sum(len(v) for d in pool.values() for v in d.values())
    print(f"[scan] ccpd_base annotated: {sum(len(v) for v in base_by_prov.values())}")
    for d in POOL_DIRS:
        print(f"[scan] {d:16s} {sum(len(v) for v in pool[d].values()):7d}")

    used_from_pool: set[str] = set()

    # the official split files decide what may be trained on / evaluated on
    base_val_all = [ln.strip() for ln in (splits_dir / "val.txt").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    base_train_all = [ln.strip() for ln in (splits_dir / "train.txt").read_text(
        encoding="utf-8").splitlines() if ln.strip()]

    # --------------------------------------------- 1. province-uniform diagnostic
    # Round-robin over folders so no single (hard) subset dominates the metric.
    # A province only donates at most 20 % of its images (and at least 4) to this
    # set: CCPD's long tail is brutal (藏 1, 宁 8, 琼 14, 青 15 images in total),
    # and eating the whole tail for evaluation would leave nothing to train on.
    diag: list[str] = []
    diag_by_prov: dict[str, int] = {}
    for prov in PROVINCES:
        buckets = {d: list(pool[d].get(prov, [])) for d in POOL_DIRS}
        for d in buckets:
            rng.shuffle(buckets[d])
        available = sum(len(b) for b in buckets.values())
        if prov == "皖":
            # 皖 diagnostic samples must come from ccpd_base/val.txt: taking them
            # from the whole directory would leak ccpd_base/train.txt images into
            # the test set.
            base_only = [p for p in base_val_all if province_of(Path(p).name) == "皖"]
            rng.shuffle(base_only)
            buckets[BASE_DIR] = base_only
            available = len(base_only)
        quota = min(args.per_province, max(4, int(0.2 * available)))
        got: list[str] = []
        order = list(buckets)
        k = 0
        while len(got) < quota and any(buckets.values()):
            d = order[k % len(order)]
            k += 1
            if buckets[d]:
                got.append(buckets[d].pop())
        diag.extend(got)
        diag_by_prov[prov] = len(got)
        used_from_pool.update(g for g in got if not g.startswith(BASE_DIR + "/"))
    print(f"[diag] province-uniform set: {len(diag)} images, "
          f"min {min(diag_by_prov.values())} / max {max(diag_by_prov.values())} per province")

    diag_base = [d for d in diag if d.startswith(BASE_DIR + "/")]
    diag_from_val = [d for d in diag_base if d in set(base_val_all)]
    assert len(diag_from_val) == len(diag_base), \
        "diagnostic 皖 samples must all come from ccpd_base/val.txt"

    # --------------------------------------------------- 2. base val variants
    base_val_pool = [p for p in base_val_all if p not in set(diag_base)]
    rng.shuffle(base_val_pool)
    val_base = base_val_pool[:args.val_base]
    val_mix_base = base_val_pool[args.val_base:args.val_base + args.val_mix_base]

    # ------------------------------------------------ 3. non-base pool -> splits
    prov_train: list[str] = []
    prov_val: list[str] = []
    test_mix: list[str] = []
    used_set = used_from_pool
    for d in POOL_DIRS:
        for prov, files in pool[d].items():
            rest = [f for f in files if f not in used_set]
            rng.shuffle(rest)
            n_val = max(1, int(round(0.08 * len(rest))))
            n_test = int(round(args.test_frac * len(rest)))
            prov_val.extend(rest[:n_val])
            test_mix.extend(rest[n_val:n_val + n_test])
            prov_train.extend(rest[n_val + n_test:])
    rng.shuffle(prov_val)
    rng.shuffle(test_mix)
    # val_mix: a natural-prior slice of non-皖 pool + a province-uniform slice
    val_mix_nonwan = prov_val[:args.val_mix_nonwan]
    uniform_src = [f for f in prov_val[len(val_mix_nonwan):]
                   if province_of(Path(f).name) != "皖"]
    rng.shuffle(uniform_src)
    val_mix_unif: list[str] = []
    seen: Counter = Counter()
    for f in uniform_src:            # greedily spread the uniform slice over provinces
        p = province_of(Path(f).name)
        if seen[p] < 200:
            seen[p] += 1
            val_mix_unif.append(f)
        if len(val_mix_unif) >= args.val_mix_unif:
            break

    train_mix = base_train_all + prov_train
    rng.shuffle(train_mix)
    val_mix = val_mix_base + val_mix_nonwan + val_mix_unif

    # ------------------------------------------------------- 4. write + verify
    outputs = {
        "train_mix.txt": train_mix,
        "val_base.txt": val_base,
        "val_mix.txt": val_mix,
        "test_prov.txt": diag,
        "test_mix.txt": test_mix,
    }
    overlap = 0
    seen_paths: dict[str, str] = {}
    for name, lines in outputs.items():
        if not args.dry_run:
            (splits_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        for ln in lines:
            if ln in seen_paths:
                overlap += 1
            seen_paths[ln] = name

    stats = {}
    for name, lines in outputs.items():
        cnt = Counter(province_of(Path(ln).name) for ln in lines)
        n = sum(cnt.values())
        stats[name] = {"n": n, "wan_ratio": round(cnt.get("皖", 0) / max(n, 1), 4),
                       "provinces": dict(cnt.most_common())}
        print(f"[write] {name:16s} n={n:7d}  皖={cnt.get('皖', 0) / max(n, 1):6.2%}  "
              f"classes={len(cnt)}")
    print(f"[check] cross-split duplicate paths: {overlap}")

    # effective (per-province) training counts: what the sampler will rebalance
    tr_cnt = Counter(province_of(Path(ln).name) for ln in train_mix)
    non = {k: v for k, v in tr_cnt.items() if k != "皖"}
    print(f"[check] train_mix non-皖 samples: {sum(non.values())} "
          f"(mean {sum(non.values()) / max(len(non), 1):.0f}/province, "
          f"min {min(non.values())}, max {max(non.values())})")

    out = PROJECT_ROOT / "runs" / "audit" / "split_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done ] stats -> {out}")
    return 0 if overlap == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
