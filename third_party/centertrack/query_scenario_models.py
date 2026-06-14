#!/usr/bin/env python3
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple


def load_xml(xml_path: str) -> ET.Element:
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    try:
        tree = ET.parse(xml_path)
        return tree.getroot()
    except ET.ParseError as e:
        raise RuntimeError(f"Failed to parse XML: {e}")


def find_scenario(root: ET.Element, name: str) -> ET.Element:
    # Find scenario element with exact name match
    for scen in root.findall('.//scenario'):
        if scen.get('name') == name:
            return scen
    return None


def find_actor_model(scenario: ET.Element, actor_name: str) -> str:
    # Search within the scenario for an other_actor with given name
    el = scenario.find(f".//other_actor[@name='{actor_name}']")
    return el.get('model') if el is not None else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Query left_turn_vehicle and opponent_vehicle models by scenario name "
            "from merged_scenarios_42k.xml"
        )
    )
    parser.add_argument(
        "scenario",
        help="Scenario name (e.g., VV_3)",
    )
    parser.add_argument(
        "--xml",
        default="/root/CenterTrack/merged_scenarios_42k.xml",
        help="Path to the scenarios XML (default: /root/CenterTrack/merged_scenarios_42k.xml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON to stdout (in addition to optional file output)",
    )
    parser.add_argument(
        "--output",
        help="Path to save the JSON payload (e.g., /dataspace/VV_3/VV_3_vehicle_specs.json)",
    )

    args = parser.parse_args(argv)

    # Load XML; if fails, keep root=None to trigger default fallback below
    root: Optional[ET.Element]
    try:
        root = load_xml(args.xml)
    except Exception as e:
        print(f"Failed to load XML '{args.xml}': {e}\nUsing default vehicle specs.", file=sys.stderr)
        root = None

    scenario = find_scenario(root, args.scenario) if root is not None else None
    if scenario is None:
        # Provide a concise hint with available names (if XML loaded)
        if root is not None:
            names = [el.get('name') for el in root.findall('.//scenario') if el.get('name')]
            msg = f"Scenario not found: {args.scenario}"
            if names:
                # show a short preview to avoid flooding output
                preview = ", ".join(names[:10]) + (" ..." if len(names) > 10 else "")
                msg += f"\nAvailable examples: {preview}"
            print(msg, file=sys.stderr)
        # Fall back to defaults by leaving models as None
        left_model = None
        opp_model = None
    else:
        left_model = find_actor_model(scenario, 'left_turn_vehicle')
        opp_model = find_actor_model(scenario, 'opponent_vehicle')

    # Vehicle specs lookup (mass in kg; size in meters: L, W, H)
    VEHICLE_SPECS: Dict[str, Dict[str, Tuple[float, float, float] or float]] = {
        # model: { mass_kg: float, size_m: (L, W, H) }
        "vehicle.ambulance.ford": {"mass_kg": 3000.0, "size_m": (6.36, 2.35, 2.43)},
        "vehicle.carlacola.actors": {"mass_kg": 3000.0, "size_m": (8.00, 2.91, 4.05)},
        "vehicle.dodge.charger": {"mass_kg": 1920.0, "size_m": (5.01, 1.88, 1.54)},
        "vehicle.dodgecop.charger": {"mass_kg": 1920.0, "size_m": (5.24, 1.92, 1.64)},
        "vehicle.firetruck.actors": {"mass_kg": 3000.0, "size_m": (8.58, 2.90, 3.83)},
        "vehicle.fuso.mitsubishi": {"mass_kg": 3000.0, "size_m": (10.17, 3.93, 4.24)},
        "vehicle.lincoln.mkz": {"mass_kg": 1696.0, "size_m": (4.89, 1.84, 1.52)},
        "vehicle.mini.cooper": {"mass_kg": 1130.0, "size_m": (4.55, 2.10, 1.77)},
        "vehicle.nissan.patrol": {"mass_kg": 2355.0, "size_m": (5.59, 2.15, 2.06)},
        "vehicle.sprinter.mercedes": {"mass_kg": 3000.0, "size_m": (5.92, 1.99, 2.73)},
        "vehicle.taxi.ford": {"mass_kg": 1920.0, "size_m": (5.35, 1.79, 1.58)},
    }
    # 默认：普通小轿车（找不到就回落到它）
    DEFAULT_SPEC = {
        "mass_kg": 1500.0,
        "size_m": (4.50, 1.80, 1.45),
        "model_tag": "vehicle.default.sedan",
    }

    def _get_spec(model: Optional[str]):
        """Return (spec, used_default: bool, model_out: str)."""
        if not model:
            return DEFAULT_SPEC, True, DEFAULT_SPEC["model_tag"]
        spec = VEHICLE_SPECS.get(model)
        if spec is None:
            # Keep original model string for visibility, but use default numbers
            return DEFAULT_SPEC, True, model
        return spec, False, model

    def compute_radius(model: Optional[str]) -> Tuple[Optional[float], Optional[float], bool]:
        spec, used_default, _ = _get_spec(model)
        L, W, _H = spec["size_m"]
        # Planar bounding circle radius: half of the planar diagonal
        radius = 0.5 * ((L * L + W * W) ** 0.5)
        return float(spec["mass_kg"]), radius, used_default

    left_mass, left_radius, left_used_default = compute_radius(left_model)
    opp_mass, opp_radius, opp_used_default = compute_radius(opp_model)

    def get_bbox(model: Optional[str]):
        spec, _used_default, _ = _get_spec(model)
        L, W, H = spec["size_m"]
        return {"length_m": L, "width_m": W, "height_m": H}

    # Normalize model output fields (use default tag if model is None)
    _, _, left_model_out = _get_spec(left_model)
    _, _, opp_model_out = _get_spec(opp_model)

    payload = {
        "scenario": args.scenario,
        "left": {
            "model": left_model_out,
            "mass_kg": left_mass,
            "planar_radius_m": left_radius,
            "bounding_box": get_bbox(left_model),
            "message": ("default vehicle specs used" if left_used_default else "lookup ok"),
        },
        "opponent": {
            "model": opp_model_out,
            "mass_kg": opp_mass,
            "planar_radius_m": opp_radius,
            "bounding_box": get_bbox(opp_model),
            "message": ("default vehicle specs used" if opp_used_default else "lookup ok"),
        },
    }

    # Save to file if requested
    if args.output:
        # os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved vehicle specs to {args.output}")

    # Optionally also print JSON to stdout
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))

    # Signal success
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
