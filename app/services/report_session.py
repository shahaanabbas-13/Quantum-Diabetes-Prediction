"""Session-only report data lifecycle helpers."""

from __future__ import annotations

from typing import MutableMapping, Any


def clear_sensitive_report_state(state: MutableMapping[str, Any]) -> None:
    """Clear report candidates, editable inputs, confirmations, and predictions."""
    fixed_keys = {
        "medical_report_upload",
        "report_candidates",
        "report_models_to_run",
        "last_live_model_predictions",
    }
    feature_prefixes = ("report_feature_", "confirm_")
    for key in list(state):
        if key in fixed_keys or key.startswith(feature_prefixes):
            state.pop(key, None)


def unconfirmed_or_missing_features(
    values: MutableMapping[str, Any],
    confirmations: MutableMapping[str, bool],
    features: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Return missing and unchecked fields; a caller must block prediction if present."""
    missing = [feature for feature in features if values.get(feature) is None]
    unconfirmed = [feature for feature in features if not confirmations.get(feature, False)]
    return missing, unconfirmed


def confirmed_candidate_defaults(
    candidates: list[MutableMapping[str, Any]], features: tuple[str, ...]
) -> dict[str, float | None]:
    """Return only unique, unit-compatible candidate values for user review."""
    defaults: dict[str, float | None] = {}
    for feature in features:
        matches = [item for item in candidates if item.get("feature") == feature]
        distinct_values = {float(item["value"]) for item in matches}
        compatible = all(
            "not mapped" not in str(item.get("unit", "")).lower()
            and "not stated" not in str(item.get("unit", "")).lower()
            for item in matches
        )
        defaults[feature] = (
            next(iter(distinct_values))
            if len(distinct_values) == 1 and compatible
            else None
        )
    return defaults
