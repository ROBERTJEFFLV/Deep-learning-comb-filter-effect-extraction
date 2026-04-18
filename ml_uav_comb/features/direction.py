"""Canonical direction semantics for v2 offline features."""

from __future__ import annotations

from typing import Dict


SHIFT_NONE = "shift_none"
SHIFT_UP_FREQ = "shift_up_freq"
SHIFT_DOWN_FREQ = "shift_down_freq"

SHIFT_TO_NUM: Dict[str, float] = {
    SHIFT_DOWN_FREQ: -1.0,
    SHIFT_NONE: 0.0,
    SHIFT_UP_FREQ: 1.0,
}

NUM_TO_SHIFT = {value: key for key, value in SHIFT_TO_NUM.items()}


def lag_to_shift_direction(lag: int, reliable: bool = True) -> str:
    """Map comb lag sign to an unambiguous frequency-shift label.

    Positive lag means the newer frame must be shifted toward higher frequency
    bins to align with the reference frame.
    """
    if (not reliable) or lag == 0:
        return SHIFT_NONE
    return SHIFT_UP_FREQ if lag > 0 else SHIFT_DOWN_FREQ

