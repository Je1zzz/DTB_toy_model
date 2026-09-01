"""Signed bipolar gain construction and source-to-SEEG projection."""

from __future__ import annotations

import re

import numpy as np

def epileptor_source(states: np.ndarray) -> np.ndarray:
    """VEP spontaneous generator source: x2 - x1 (states 3 and 0)."""
    values=np.asarray(states)
    return values[...,3,:]-values[...,0,:]

def legacy_source_convention(states: np.ndarray) -> np.ndarray:
    values=np.asarray(states)
    return values[...,0,:]-values[...,3,:]


def _split_bipolar_name(channel: str) -> tuple[str, str]:
    if "-" not in channel:
        raise ValueError(f"Not a bipolar channel name: {channel!r}")
    left, right = channel.rsplit("-", 1)
    match = re.match(r"^(.*?\d+)$", left)
    if match is None or not right.isdigit():
        raise ValueError(f"Cannot parse bipolar channel name: {channel!r}")
    prefix = re.match(r"^(.*?)(\d+)$", left).group(1)
    return left, prefix + right


def bipolarize_gain(
    contact_gain: np.ndarray,
    contact_names: tuple[str, ...] | list[str],
    channel_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Construct signed bipolar rows aligned to the recording channels.

    For a recorded channel ``A-B``, the reference uses contact ``B - A``;
    therefore the observation row is ``G_B - G_A``.
    """

    gain = np.asarray(contact_gain, dtype=float)
    contacts = tuple(contact_names)
    lookup = {name: index for index, name in enumerate(contacts)}
    rows: list[np.ndarray] = []
    missing: list[str] = []
    for channel in channel_names:
        first, second = _split_bipolar_name(channel)
        if first not in lookup or second not in lookup:
            missing.append(channel)
            continue
        rows.append(gain[lookup[second]] - gain[lookup[first]])
    if missing:
        raise KeyError(f"Bipolar channels missing from electrodes.tsv: {missing[:5]}")
    result = np.asarray(rows, dtype=float)
    if result.ndim != 2 or result.shape[1] != gain.shape[1]:
        raise ValueError(f"Unexpected bipolar gain shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Bipolar gain contains non-finite values")
    return result


def project_to_seeg(source_activity: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Apply the linear observation operator: ``SEEG = source @ gain.T``."""

    source = np.asarray(source_activity, dtype=float)
    matrix = np.asarray(gain, dtype=float)
    if source.ndim != 2 or matrix.ndim != 2:
        raise ValueError(f"source and gain must be matrices, got {source.shape}, {matrix.shape}")
    if source.shape[1] != matrix.shape[1]:
        raise ValueError(f"source/gain region mismatch: {source.shape}, {matrix.shape}")
    output = source @ matrix.T
    if not np.isfinite(output).all():
        raise FloatingPointError("Source-to-SEEG projection produced non-finite values")
    return output
