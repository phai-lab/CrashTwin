from __future__ import annotations


METRIC_NAMES = ("E_flow", "E_warp", "J_p", "J_H", "J_E", "S_ID", "D_ad")
LOWER_IS_BETTER = frozenset({"E_flow", "E_warp", "J_p", "J_H", "J_E", "D_ad"})
HIGHER_IS_BETTER = frozenset({"S_ID"})


def metric_direction(metric_name: str) -> str:
    if metric_name in LOWER_IS_BETTER:
        return "lower"
    if metric_name in HIGHER_IS_BETTER:
        return "higher"
    raise KeyError(f"Unknown CrashTwin metric: {metric_name}")


def is_metric_name(value: str) -> bool:
    return value in METRIC_NAMES

