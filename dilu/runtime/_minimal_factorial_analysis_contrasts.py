"""Registered averaged contrasts for the ICLR 2027 minimal factorial."""

from __future__ import annotations

import math
from typing import Mapping

CONDITIONS = tuple(f"c{p}{o}{e}" for p in (0, 1) for o in (0, 1) for e in (0, 1))


def factorial_contrasts(cells: Mapping[str, float]) -> dict[str, float]:
    values = _cells(cells)
    p_main = (
        sum(values[f"c1{o}{e}"] - values[f"c0{o}{e}"] for o in (0, 1) for e in (0, 1))
        / 4
    )
    o_main = (
        sum(values[f"c{p}1{e}"] - values[f"c{p}0{e}"] for p in (0, 1) for e in (0, 1))
        / 4
    )
    e_main = (
        sum(values[f"c{p}{o}1"] - values[f"c{p}{o}0"] for p in (0, 1) for o in (0, 1))
        / 4
    )
    po = (
        sum(
            values[f"c11{e}"]
            - values[f"c10{e}"]
            - values[f"c01{e}"]
            + values[f"c00{e}"]
            for e in (0, 1)
        )
        / 2
    )
    pe = (
        sum(
            values[f"c1{o}1"]
            - values[f"c1{o}0"]
            - values[f"c0{o}1"]
            + values[f"c0{o}0"]
            for o in (0, 1)
        )
        / 2
    )
    oe = (
        sum(
            values[f"c{p}11"]
            - values[f"c{p}10"]
            - values[f"c{p}01"]
            + values[f"c{p}00"]
            for p in (0, 1)
        )
        / 2
    )
    poe = (
        values["c111"]
        - values["c011"]
        - values["c101"]
        + values["c001"]
        - values["c110"]
        + values["c010"]
        + values["c100"]
        - values["c000"]
    )
    return {
        "P_MAIN": p_main,
        "O_MAIN": o_main,
        "E_MAIN": e_main,
        "PO_INTERACTION": po,
        "PE_INTERACTION": pe,
        "OE_INTERACTION": oe,
        "POE_INTERACTION": poe,
    }


def endpoint_contrast(cells: Mapping[str, float]) -> float:
    values = _cells(cells)
    return values["c111"] - values["c000"]


def _cells(cells: Mapping[str, float]) -> dict[str, float]:
    if set(cells) != set(CONDITIONS):
        raise ValueError("Factorial contrast requires exactly all eight conditions.")
    values = {condition: float(cells[condition]) for condition in CONDITIONS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Factorial outcomes must be finite.")
    return values


__all__ = ["CONDITIONS", "endpoint_contrast", "factorial_contrasts"]
