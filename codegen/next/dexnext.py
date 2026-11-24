"""The ``.dexnext`` weight format and a NumPy reference of the Rust observer.

Mirrors ``dexhal::math::torque_observer`` (``dexhal/specs/factr2.md`` §2.4).
The architecture is fixed: input ``[q, qd, q_cmd − q]``, 2-layer LSTM with
hidden 128, MLP head 128 → 256 → dof, stateless windows of 50 samples at
100 Hz. A file is a 144-byte little-endian header followed by the ``f64``
parameters, every matrix stored transposed (input-major).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

MAGIC = b"DEXNEXT\x02"
HEADER = struct.Struct("<8sBB6s3dd3d9d")
assert HEADER.size == 144

LAYERS = 2
HIDDEN = 128
GATES = 4 * HIDDEN
MLP = 256
WINDOW = 50
SAMPLE_RATE_HZ = 100

FULL_DYNAMICS, FRICTION_RESIDUAL = 0, 1

STANDARD_GRAVITY = 9.80665


def stride_ticks(control_freq_hz: int) -> int:
    """Control ticks per network sample (``LstmObserver::stride_ticks``)."""
    return max((int(control_freq_hz) + SAMPLE_RATE_HZ // 2) // SAMPLE_RATE_HZ, 1)


@dataclass
class Header:
    dof: int
    mode: int = FULL_DYNAMICS
    gravity: Tuple[float, float, float] = (0.0, 0.0, -STANDARD_GRAVITY)
    payload_mass: float = 0.0
    payload_com: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    payload_inertia: Tuple[float, ...] = (0.0,) * 9


def shapes(dof: int):
    """``(name, PyTorch shape)`` of every tensor, in file order. Matrices are
    ``out × in`` here and stored transposed."""
    d = 3 * dof
    return [
        ("in_mean", (d,)),
        ("in_std", (d,)),
        ("w_ih0", (GATES, d)),
        ("w_hh0", (GATES, HIDDEN)),
        ("b0", (GATES,)),
        ("w_ih1", (GATES, HIDDEN)),
        ("w_hh1", (GATES, HIDDEN)),
        ("b1", (GATES,)),
        ("w_mlp", (MLP, HIDDEN)),
        ("b_mlp", (MLP,)),
        ("w_out", (dof, MLP)),
        ("b_out", (dof,)),
        ("out_mean", (dof,)),
        ("out_std", (dof,)),
    ]


def param_count(dof: int) -> int:
    return sum(int(np.prod(s)) for _, s in shapes(dof))


def file_len(dof: int) -> int:
    return HEADER.size + 8 * param_count(dof)


@dataclass
class Weights:
    """Float64 parameters in PyTorch shapes; gates ``i, f, g, o`` and
    ``b = b_ih + b_hh``."""

    in_mean: np.ndarray
    in_std: np.ndarray
    w_ih0: np.ndarray
    w_hh0: np.ndarray
    b0: np.ndarray
    w_ih1: np.ndarray
    w_hh1: np.ndarray
    b1: np.ndarray
    w_mlp: np.ndarray
    b_mlp: np.ndarray
    w_out: np.ndarray
    b_out: np.ndarray
    out_mean: np.ndarray
    out_std: np.ndarray

    @property
    def dof(self) -> int:
        return int(self.b_out.shape[0])

    def flat(self) -> np.ndarray:
        """File-order parameters (matrices transposed)."""
        parts = []
        for name, shape in shapes(self.dof):
            a = np.asarray(getattr(self, name), dtype=np.float64)
            if a.shape != shape:
                raise DexNextError(f"{name} has shape {a.shape}, expected {shape}")
            parts.append(a.T.ravel() if a.ndim == 2 else a)
        return np.concatenate(parts)

    @classmethod
    def from_flat(cls, dof: int, flat: np.ndarray) -> "Weights":
        arrays, at = {}, 0
        for name, shape in shapes(dof):
            n = int(np.prod(shape))
            a = np.asarray(flat[at : at + n], dtype=np.float64)
            arrays[name] = a.reshape(shape[::-1]).T.copy() if len(shape) == 2 else a.copy()
            at += n
        return cls(**arrays)


class DexNextError(ValueError):
    """A ``.dexnext`` file the Rust loader would reject (``InvalidObserver``)."""


def _check(header: Header, weights: Weights, flat: np.ndarray) -> None:
    h = header
    finite = (
        np.all(np.isfinite(h.gravity))
        and np.isfinite(h.payload_mass)
        and np.all(np.isfinite(h.payload_com))
        and np.all(np.isfinite(h.payload_inertia))
        and np.all(np.isfinite(flat))
    )
    if not (1 <= h.dof <= 255 and h.mode in (FULL_DYNAMICS, FRICTION_RESIDUAL)):
        raise DexNextError(f"invalid header: {header}")
    if not finite or h.payload_mass < 0.0:
        raise DexNextError("non-finite value or negative payload mass")
    if weights.dof != h.dof:
        raise DexNextError("weights are for another dof")
    if np.any(weights.in_std <= 1e-12) or np.any(weights.out_std <= 0.0):
        raise DexNextError("in_std must be > 1e-12 and out_std > 0")


def pack(header: Header, weights: Weights) -> bytes:
    """Encode a ``.dexnext`` file; raises :class:`DexNextError` if invalid."""
    flat = weights.flat()
    _check(header, weights, flat)
    h = header
    head = HEADER.pack(
        MAGIC,
        h.dof,
        h.mode,
        bytes(6),
        *map(float, h.gravity),
        float(h.payload_mass),
        *map(float, h.payload_com),
        *map(float, h.payload_inertia),
    )
    return head + flat.astype("<f8").tobytes()


def unpack(data: bytes) -> Tuple[Header, Weights]:
    """Decode and validate a ``.dexnext`` file."""
    if len(data) < HEADER.size:
        raise DexNextError("truncated header")
    f = HEADER.unpack_from(data)
    magic, dof, mode, reserved = f[:4]
    if magic != MAGIC or reserved != bytes(6):
        raise DexNextError("bad magic or non-zero reserved bytes")
    header = Header(
        dof=dof,
        mode=mode,
        gravity=tuple(f[4:7]),
        payload_mass=f[7],
        payload_com=tuple(f[8:11]),
        payload_inertia=tuple(f[11:20]),
    )
    if len(data) != file_len(dof):
        raise DexNextError("length does not match the dof")
    flat = np.frombuffer(data[HEADER.size :], dtype="<f8").astype(np.float64)
    weights = Weights.from_flat(dof, flat)
    _check(header, weights, flat)
    return header, weights


# --------------------------------------------------------------------------
# Reference evaluation
# --------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def padded_windows(u: np.ndarray) -> np.ndarray:
    """``T × WINDOW × 3·dof``: for every sample, the newest ``WINDOW`` samples
    (oldest first), left-padded with the first — the runtime fill."""
    u = np.asarray(u, dtype=np.float64)
    idx = np.maximum(np.arange(len(u))[:, None] + np.arange(WINDOW)[None, :] - (WINDOW - 1), 0)
    return u[idx]


def window_outputs(weights: Weights, u: np.ndarray) -> np.ndarray:
    """Output of every window over raw inputs ``u`` (``T × 3·dof``), batched:
    what the Rust observer reports at each sample tick."""
    w = weights
    x = (padded_windows(u) - w.in_mean) / w.in_std  # T × W × D
    n = x.shape[0]
    h = [np.zeros((n, HIDDEN)) for _ in range(LAYERS)]
    c = [np.zeros((n, HIDDEN)) for _ in range(LAYERS)]
    layers = [(w.w_ih0, w.w_hh0, w.b0), (w.w_ih1, w.w_hh1, w.b1)]
    for t in range(WINDOW):
        inp = x[:, t]
        for l, (w_ih, w_hh, b) in enumerate(layers):
            g = b + inp @ w_ih.T + h[l] @ w_hh.T
            H = HIDDEN
            c[l] = _sigmoid(g[:, H : 2 * H]) * c[l] + _sigmoid(g[:, :H]) * np.tanh(g[:, 2 * H : 3 * H])
            h[l] = _sigmoid(g[:, 3 * H :]) * np.tanh(c[l])
            inp = h[l]
    z = np.maximum(w.b_mlp + h[-1] @ w.w_mlp.T, 0.0)
    return (w.b_out + z @ w.w_out.T) * w.out_std + w.out_mean


# --------------------------------------------------------------------------
# Deterministic test data, shared bit for bit with
# ``dexhal::math::torque_observer::test_support``
# --------------------------------------------------------------------------

_MASK = (1 << 64) - 1


def lcg_units(seed: int, n: int) -> np.ndarray:
    """``n`` draws in ``[-1, 1)`` from the shared 64-bit LCG."""
    s = (seed * 0x9E3779B97F4A7C15) & _MASK | 1
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & _MASK
        out[i] = (s >> 11) / 9007199254740992.0
    return out * 2.0 - 1.0


def lcg_weights(dof: int, seed: int) -> Weights:
    """Synthetic parameters (``test_support::params``): one draw per
    parameter in file order; weights and biases ``0.1·u``, ``in_mean = u``,
    ``in_std = 0.5 + 0.4·(u + 1)``, ``out_mean = 10·u``, ``out_std = 1 + 2·(u + 1)``."""
    u = lcg_units(seed, param_count(dof))
    d = 3 * dof
    flat = 0.1 * u
    flat[:d] = u[:d]
    flat[d : 2 * d] = 0.5 + 0.4 * (u[d : 2 * d] + 1.0)
    flat[-2 * dof : -dof] = 10.0 * u[-2 * dof : -dof]
    flat[-dof:] = 1.0 + 2.0 * (u[-dof:] + 1.0)
    return Weights.from_flat(dof, flat)


def parity_inputs(dof: int, n: int, seed: int) -> np.ndarray:
    """Raw inputs ``[q, qd, q_cmd − q]`` (``n × 3·dof``) scaled 1.5 / 0.8 / 0.05
    (``test_support::parity_inputs``)."""
    u = lcg_units(seed, n * 3 * dof).reshape(n, 3 * dof)
    scale = np.repeat([1.5, 0.8, 0.05], dof)
    return u * scale
