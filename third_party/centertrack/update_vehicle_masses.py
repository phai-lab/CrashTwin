#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def infer_scenario_from_spec_path(spec_path: str) -> str:
    stem = Path(spec_path).stem
    if stem.endswith("_vehicle_specs"):
        return stem[: -len("_vehicle_specs")]
    return stem


def update_spec_from_pred(spec_path: str, preds: dict, left_id: int = None, opp_id: int = None) -> None:
    spec = load_json(spec_path)
    scenario = spec.get("scenario") or infer_scenario_from_spec_path(spec_path)
    key = f"{scenario}_frame.png"

    pred_entry = preds.get(key)
    if not isinstance(pred_entry, dict):
        print(f"[SKIP] {spec_path}: no prediction for key {key}")
        return

    # Determine IDs to use
    lid, oid = left_id, opp_id
    if lid is None or oid is None:
        try:
            ids = sorted({int(k) for k in pred_entry.keys()})
        except Exception:
            ids = []
        if len(ids) >= 2:
            lid = ids[0] if lid is None else lid
            oid = ids[1] if oid is None else oid

    left_mass = None
    opp_mass = None
    if lid is not None:
        left_mass = (pred_entry.get(str(lid)) or {}).get("mass_kg")
    if oid is not None:
        opp_mass = (pred_entry.get(str(oid)) or {}).get("mass_kg")

    old_left = (spec.get("left") or {}).get("mass_kg")
    old_opp = (spec.get("opponent") or {}).get("mass_kg")

    if left_mass is not None:
        spec.setdefault("left", {})["mass_kg"] = float(left_mass)
    if opp_mass is not None:
        spec.setdefault("opponent", {})["mass_kg"] = float(opp_mass)

    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    print(
        f"{scenario}: left {old_left} -> {spec.get('left', {}).get('mass_kg')}, "
        f"opponent {old_opp} -> {spec.get('opponent', {}).get('mass_kg')}"
    )


def main():
    ap = argparse.ArgumentParser(description="Update *_vehicle_specs.json masses from mass_pred.json")
    ap.add_argument("--pred", required=True, help="Path to mass_pred.json")
    ap.add_argument("--spec", required=True, nargs="+", help="One or more *_vehicle_specs.json paths")
    ap.add_argument("--left-id", type=int, default=None, help="Override: predicted ID for left")
    ap.add_argument("--opponent-id", type=int, default=None, help="Override: predicted ID for opponent")
    args = ap.parse_args()

    preds = load_json(args.pred)

    for spec_path in args.spec:
        try:
            update_spec_from_pred(
                spec_path,
                preds,
                left_id=args.left_id,
                opp_id=args.opponent_id,
            )
        except Exception as e:
            print(f"[ERROR] {spec_path}: {e}")


if __name__ == "__main__":
    main()

