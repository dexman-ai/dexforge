"""Train a FACTR 2 (NEXT) LSTM torque observer and export it as ``.dexnext``.

Follows ``dexhal/specs/factr2.md`` §3.2::

    python -m codegen.next.train_next data.npz -o ur5.dexnext [--mode residual]

The architecture and training setup are fixed to the paper's (arXiv
2606.12406, Table 3 and §A.2; the same for its Franka and AgileX Piper
experiments): input ``[q, qd, q_cmd − q]`` sampled at 100 Hz, windows of 50
samples, 2-layer LSTM with hidden 128 and dropout 0.1, MLP head
128 → 256 → dof, MSE, AdamW (lr 1e-3, weight decay 1e-6), gradient clip 1.0,
early stopping on validation loss (patience 20, warm-up 10, min Δ 1e-5).

The input is a free-space log recorded by the data collector (``dexzero``,
spec §3.1), an ``.npz`` with, per control tick:

* ``q``, ``qd``, ``q_cmd``, ``tau_link`` (``T × dof``; link-side N·m)
* ``tau_model`` (``T × dof``): the driver's own inverse dynamics
  ``ArmMath::inverse_dynamics(q, qd, qdd_ref)`` with the collection payload
  configured — required only for ``--mode residual``
* ``segment`` (``T``): contiguous fault- and contact-free segment id, ``-1``
  for discarded frames

and the scalars / vectors ``dof``, ``control_freq_hz``, ``gravity`` (3),
``payload_mass``, ``payload_com`` (3), ``payload_inertia`` (9).

``--mode full`` (the paper) trains on ``tau_link`` and records the collection
payload in the header. ``--mode residual`` trains on ``tau_link − tau_model``
and records no payload; at runtime the driver adds the same
``flange_inverse_dynamics`` back, so target and correction come from one
model. Both use the same network.

Export folds ``b_ih + b_hh``, casts to float64 and checks the PyTorch
float64 model against the NumPy reference of the Rust observer
(:func:`codegen.next.dexnext.window_outputs`) on held-out data before
writing.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from codegen.next import dexnext as dn  # noqa: E402

PARITY_TOLERANCE = 1e-10
DROPOUT = 0.1


class NextObserverNet(torch.nn.Module):
    """2×128 LSTM over a window, MLP head on the last step (factr2.md §3.2)."""

    def __init__(self, dof: int):
        super().__init__()
        self.dof = dof
        self.lstm = torch.nn.LSTM(3 * dof, dn.HIDDEN, num_layers=dn.LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(dn.HIDDEN, dn.MLP),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(dn.MLP, dof),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``B × W × 3·dof`` normalised windows → ``B × dof`` normalised torque."""
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


@dataclass
class Normalizers:
    in_mean: np.ndarray
    in_std: np.ndarray
    out_mean: np.ndarray
    out_std: np.ndarray


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float64).numpy().copy()


def to_weights(model: NextObserverNet, norm: Normalizers) -> dn.Weights:
    """Float64 :class:`~codegen.next.dexnext.Weights` with ``b_ih + b_hh`` folded."""
    lstm = model.lstm
    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    return dn.Weights(
        in_mean=f64(norm.in_mean),
        in_std=f64(norm.in_std),
        w_ih0=_np(lstm.weight_ih_l0),
        w_hh0=_np(lstm.weight_hh_l0),
        b0=_np(lstm.bias_ih_l0) + _np(lstm.bias_hh_l0),
        w_ih1=_np(lstm.weight_ih_l1),
        w_hh1=_np(lstm.weight_hh_l1),
        b1=_np(lstm.bias_ih_l1) + _np(lstm.bias_hh_l1),
        w_mlp=_np(model.head[0].weight),
        b_mlp=_np(model.head[0].bias),
        w_out=_np(model.head[3].weight),
        b_out=_np(model.head[3].bias),
        out_mean=f64(norm.out_mean),
        out_std=f64(norm.out_std),
    )


def from_weights(weights: dn.Weights) -> NextObserverNet:
    """Float64 :class:`NextObserverNet` holding ``weights`` (folded bias as
    ``bias_ih``, zero ``bias_hh``), e.g. to fine-tune or re-check a file."""
    model = NextObserverNet(weights.dof).to(torch.float64)
    t = torch.from_numpy
    w = weights
    with torch.no_grad():
        for l, (w_ih, w_hh, b) in enumerate([(w.w_ih0, w.w_hh0, w.b0), (w.w_ih1, w.w_hh1, w.b1)]):
            getattr(model.lstm, f"weight_ih_l{l}").copy_(t(w_ih))
            getattr(model.lstm, f"weight_hh_l{l}").copy_(t(w_hh))
            getattr(model.lstm, f"bias_ih_l{l}").copy_(t(b))
            getattr(model.lstm, f"bias_hh_l{l}").zero_()
        model.head[0].weight.copy_(t(w.w_mlp))
        model.head[0].bias.copy_(t(w.b_mlp))
        model.head[3].weight.copy_(t(w.w_out))
        model.head[3].bias.copy_(t(w.b_out))
    return model.eval()


def torch_reference(model: NextObserverNet, norm: Normalizers, u: np.ndarray) -> np.ndarray:
    """PyTorch float64 output of every runtime window over raw inputs ``u``
    (``T × 3·dof``)."""
    m = copy.deepcopy(model).to(torch.float64).eval()
    x = torch.from_numpy((dn.padded_windows(u) - norm.in_mean) / norm.in_std)
    with torch.no_grad():
        y = m(x).numpy()
    return y * norm.out_std + norm.out_mean


def export_dexnext(model: NextObserverNet, header: dn.Header, norm: Normalizers, holdout: np.ndarray) -> bytes:
    """Pack ``model`` and verify PyTorch/NumPy parity on raw ``holdout`` inputs."""
    weights = to_weights(model, norm)
    data = dn.pack(header, weights)
    _, decoded = dn.unpack(data)
    err = float(np.max(np.abs(torch_reference(model, norm, holdout) - dn.window_outputs(decoded, holdout))))
    if not err < PARITY_TOLERANCE:
        raise RuntimeError(f"PyTorch/NumPy parity {err:.3e} >= {PARITY_TOLERANCE:.0e}")
    return data


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


@dataclass
class Segment:
    """One contiguous, fault- and contact-free run at 100 Hz."""

    u: np.ndarray  # T × 3·dof raw inputs [q, qd, q_cmd − q]
    y: np.ndarray  # T × dof raw target torque


def split_segments(segment_ids: np.ndarray) -> List[slice]:
    """Contiguous runs of equal non-negative ``segment_ids`` (``-1`` = discarded)."""
    out, start = [], 0
    for i in range(1, len(segment_ids) + 1):
        if i == len(segment_ids) or segment_ids[i] != segment_ids[start]:
            if segment_ids[start] >= 0:
                out.append(slice(start, i))
            start = i
    return out


def load_segments(npz: Path, mode: int, stride: int) -> List[Segment]:
    """Downsample each segment by ``stride`` (the runtime ``stride_ticks``)
    and build inputs and targets for ``mode``."""
    d = np.load(npz)
    q, qd, q_cmd = (np.asarray(d[k], dtype=np.float64) for k in ("q", "qd", "q_cmd"))
    tau = np.asarray(d["tau_link"], dtype=np.float64)
    if mode == dn.FRICTION_RESIDUAL:
        if "tau_model" not in d:
            raise ValueError("residual mode needs tau_model (the driver's inverse dynamics) in the log")
        tau = tau - np.asarray(d["tau_model"], dtype=np.float64)
    out = []
    for s in split_segments(np.asarray(d["segment"])):
        idx = np.arange(s.start, s.stop, stride)
        u = np.concatenate([q[idx], qd[idx], q_cmd[idx] - q[idx]], axis=1)
        out.append(Segment(u, tau[idx]))
    return out


def windows(segments: Sequence[Segment]) -> Tuple[np.ndarray, np.ndarray]:
    """Every runtime window (left-padded with the segment's first sample)
    and the target at its last step."""
    xs = [dn.padded_windows(s.u) for s in segments if len(s.u)]
    ys = [s.y for s in segments if len(s.u)]
    return np.concatenate(xs), np.concatenate(ys)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    clip: float = 1.0
    patience: int = 20
    warmup: int = 10
    min_delta: float = 1e-5
    max_epochs: int = 500
    batch: int = 256
    seed: int = 0


def normalizers(segments: Sequence[Segment]) -> Normalizers:
    u = np.concatenate([s.u for s in segments])
    y = np.concatenate([s.y for s in segments])
    return Normalizers(u.mean(axis=0), np.maximum(u.std(axis=0), 1e-6), y.mean(axis=0), np.maximum(y.std(axis=0), 1e-6))


def train(train_segs: Sequence[Segment], val_segs: Sequence[Segment], cfg: TrainConfig, log=print):
    """Returns ``(model, normalizers)`` with the best validation checkpoint restored."""
    torch.manual_seed(cfg.seed)
    norm = normalizers(train_segs)
    dof = train_segs[0].y.shape[1]

    def tensors(segs):
        x, y = windows(segs)
        x = (x - norm.in_mean) / norm.in_std
        y = (y - norm.out_mean) / norm.out_std
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    xt, yt = tensors(train_segs)
    xv, yv = tensors(val_segs)
    model = NextObserverNet(dof)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = torch.nn.MSELoss()

    best, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(perm), cfg.batch):
            b = perm[i : i + cfg.batch]
            loss = mse(model(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(mse(model(xv), yv))
        log(f"epoch {epoch:4d}  val {v:.6f}")
        if v < best - cfg.min_delta:
            best, best_state, stale = v, copy.deepcopy(model.state_dict()), 0
        elif epoch >= cfg.warmup:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, norm


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, help="free-space log (.npz, see above)")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--mode", choices=("full", "residual"), default="full")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    meta = np.load(a.data)
    dof = int(meta["dof"])
    control_hz = int(meta["control_freq_hz"])
    residual = a.mode == "residual"
    header = dn.Header(
        dof=dof,
        mode=dn.FRICTION_RESIDUAL if residual else dn.FULL_DYNAMICS,
        gravity=tuple(float(g) for g in meta["gravity"]),
        # Full dynamics keeps the mounted tool in the target; the residual
        # strips it, so the runtime differential uses Payload::NONE.
        payload_mass=0.0 if residual else float(meta["payload_mass"]),
        payload_com=(0.0,) * 3 if residual else tuple(map(float, meta["payload_com"])),
        payload_inertia=(0.0,) * 9 if residual else tuple(map(float, np.ravel(meta["payload_inertia"]))),
    )

    stride = dn.stride_ticks(control_hz)
    try:
        segs = load_segments(a.data, header.mode, stride)
    except ValueError as e:
        p.error(str(e))
    segs = [s for s in segs if len(s.u)]
    if not segs:
        p.error("no usable segments")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(segs))
    n_val = max(1, int(round(a.val_fraction * len(segs)))) if len(segs) > 1 else 0
    val = [segs[i] for i in order[:n_val]] or segs
    trn = [segs[i] for i in order[n_val:]] or segs

    model, norm = train(trn, val, TrainConfig(max_epochs=a.max_epochs, seed=a.seed))
    holdout = max(val, key=lambda s: len(s.u)).u[:2000]
    data = export_dexnext(model, header, norm, holdout)
    a.output.write_bytes(data)
    print(f"wrote {a.output} ({len(data)} bytes, stride {stride} at {control_hz} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
