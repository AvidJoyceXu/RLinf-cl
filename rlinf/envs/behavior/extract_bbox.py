"""Extract TRUE per-model geometry from the BEHAVIOR asset USDs, with no simulator.

Writes `native_bbox.json`, which `detect.py` reads for every projected box. This
replaces a `scratchpad/extract_bbox.py` that produced the same artifact on a previous
host and was never committed -- so when the machine changed, `detect` and
`detect_scope` died on a missing file and the extraction method had to be recovered
from a prose description. Hence this lives in the package.

Two attributes are read, both authored by BEHAVIOR into every object USD:

``ig:nativeBB``
    the object's native bounding-box extent, the same attribute OmniGibson reads in
    `DatasetObject.native_bbox` (dataset_object.py:358). Nothing here is nominal or
    estimated -- an earlier version of `detect` gave every task object a hardcoded
    15 cm half-extent, which made a refrigerator, a floor and a sheet of plywood the
    same size and is why solvability read 80% while the mode was broken.

``ig:offsetBaseLink``
    the offset between the base link -- which is what a scene JSON's ``root_link.pos``
    reports -- and the bbox CENTRE. `detect` currently treats position as the centre,
    so every box is displaced by this offset (for one saucepot, (-0.103, 0.0003,
    0.035)). Small for small objects, not small for furniture. It is extracted here
    because it is data we already own and were not using; see the 0815 detect debug
    log §3, suspect 1.

Method: assets ship as Fernet-encrypted usdz. The key ships with the dataset, so
decryption needs only `cryptography` and reading needs only standalone `usd-core`
(`pip install usd-core`) -- **Kit is never booted**. That is consistent with what the
mode needs: initial geometry only, since `detect` steps symbolically and never
physically.

    python -m rlinf.envs.behavior.extract_bbox            # writes the default path
    python -m rlinf.envs.behavior.extract_bbox --limit 20 # smoke test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import tempfile

from cryptography.fernet import Fernet

ASSET_ROOT = "/data/behavior-data/behavior-1k-assets/objects"
KEY_PATH = "/data/behavior-data/omnigibson.key"
OUT_PATH = "/data/behavior-data/native_bbox.json"


def _decrypt(path: str, key: bytes) -> str:
    """Decrypt one asset to a temp file and return its path. Caller unlinks."""
    data = Fernet(key).decrypt(open(path, "rb").read())
    fd, out = tempfile.mkstemp(suffix=".usdz")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return out


def _read_attrs(usdz_path: str) -> tuple[list, list]:
    """Return (nativeBB, offsetBaseLink) from the first prim that carries them.

    The usdz must be UNZIPPED first: `Usd.Stage.Open` on the package itself fails with
    `Failed to open layer`, and only the inner `.usdc` opens. OmniGibson's own
    `extracted()` helper does the same thing for the same reason.
    """
    import zipfile

    from pxr import Usd

    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(usdz_path) as z:
            inner = [n for n in z.namelist() if n.endswith((".usdc", ".usda", ".usd"))]
            if not inner:
                raise ValueError("no usd layer inside the usdz package")
            z.extract(inner[0], d)
            stage = Usd.Stage.Open(os.path.join(d, inner[0]))
            if stage is None:
                raise ValueError("USD stage would not open")
            return _scan(stage)


def _scan(stage) -> tuple[list, list]:
    bb = off = None
    for prim in stage.Traverse():
        if bb is None and prim.HasAttribute("ig:nativeBB"):
            bb = prim.GetAttribute("ig:nativeBB").Get()
        if off is None and prim.HasAttribute("ig:offsetBaseLink"):
            off = prim.GetAttribute("ig:offsetBaseLink").Get()
        if bb is not None and off is not None:
            break
    if bb is None:
        raise ValueError("no ig:nativeBB on any prim")
    return [float(v) for v in bb], ([float(v) for v in off] if off is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset-root", default=ASSET_ROOT)
    ap.add_argument("--key", default=KEY_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--limit", type=int, default=0, help="stop after N models")
    args = ap.parse_args()

    key = open(args.key, "rb").read()
    assets = sorted(glob.glob(os.path.join(args.asset_root, "*", "*", "usd",
                                           "*.usdz.encrypted")))
    if args.limit:
        assets = assets[: args.limit]
    print(f"assets: {len(assets)}", flush=True)

    # Schema is dictated by the consumer, `detect.py`:
    #   by_model[model]      -> {"bbox": [x,y,z], "offset": [x,y,z]}   (extent_for_model
    #                            reads bb["bbox"], offset_for_model reads bb["offset"])
    #   by_category[cat]     -> [x,y,z], the MEDIAN over that category's real assets
    #                            (extent_for_category iterates the vector directly)
    by_model: dict[str, dict] = {}
    per_category: dict[str, list] = {}
    failures: list[tuple[str, str]] = []

    for i, path in enumerate(assets):
        parts = path.split(os.sep)
        category, model = parts[-4], parts[-3]
        tmp = None
        try:
            tmp = _decrypt(path, key)
            bb, off = _read_attrs(tmp)
            by_model[model] = {"bbox": bb}
            if off is not None:
                by_model[model]["offset"] = off
            per_category.setdefault(category, []).append(bb)
        except Exception as ex:                      # record, never silently skip
            failures.append((model, f"{type(ex).__name__}: {ex}"))
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(assets)}  models={len(by_model)} "
                  f"failed={len(failures)}", flush=True)

    # Per-category fallback is the MEDIAN of that category's real assets, per axis --
    # measured data, never a hand-written number. (An earlier version of `detect` used
    # one hardcoded 15 cm half-extent for everything; that is the mistake this avoids.)
    by_category = {
        cat: [statistics.median(b[i] for b in boxes) for i in range(3)]
        for cat, boxes in per_category.items()
    }

    with open(args.out, "w") as f:
        json.dump({"by_model": by_model, "by_category": by_category}, f)

    n_off = sum("offset" in v for v in by_model.values())
    n_nonzero = sum(any(abs(c) > 1e-9 for c in v.get("offset", ()))
                    for v in by_model.values())
    print(f"\nmodels     : {len(by_model)}")
    print(f"categories : {len(by_category)}")
    print(f"offsets    : {n_off}  ({n_nonzero} non-zero, "
          f"{100.0 * n_nonzero / max(len(by_model), 1):.0f}%)")
    print(f"failures   : {len(failures)}")
    for m, why in failures[:10]:
        print(f"    {m}: {why}")
    print(f"written    : {args.out}")


if __name__ == "__main__":
    main()
