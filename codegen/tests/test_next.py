"""Tests for ``codegen/next`` (FACTR 2 learned torque observer tooling)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from codegen.next import dexnext as dn  # noqa: E402

FIXTURE = _REPO / "dexhal" / "tests" / "fixtures" / "next" / "parity.expected"
HEADER = dn.Header(
    dof=6,
    mode=dn.FRICTION_RESIDUAL,
    gravity=(0.1, -0.2, -9.7),
    payload_mass=1.5,
    payload_com=(0.0, 0.01, 0.08),
    payload_inertia=(0.01, 0, 0, 0, 0.02, 0, 0, 0, 0.03),
)


# --------------------------------------------------------------------------
# Format and NumPy reference (NumPy only)
# --------------------------------------------------------------------------


def test_layout_matches_rust():
    assert dn.HEADER.size == 144
    lstm = (18 * 512 + 128 * 512 + 512) + (2 * 128 * 512 + 512)
    head = 128 * 256 + 256 + 256 * 6 + 6
    assert dn.param_count(6) == 2 * 18 + lstm + head + 2 * 6
    assert len(dn.pack(dn.Header(dof=6), dn.lcg_weights(6, 1))) == dn.file_len(6)


def test_matrices_are_stored_transposed():
    w = dn.lcg_weights(2, 3)
    flat = w.flat()
    d = 6
    # Column k of W_ih0 (all 512 gate rows) is contiguous.
    assert flat[2 * d + 4 * 512 + 7] == w.w_ih0[7, 4]
    np.testing.assert_array_equal(dn.Weights.from_flat(2, flat).flat(), flat)


def test_pack_unpack_round_trip():
    w = dn.lcg_weights(6, 5)
    header, w2 = dn.unpack(dn.pack(HEADER, w))
    assert header == HEADER
    np.testing.assert_array_equal(w2.flat(), w.flat())


def _patch(data: bytes, at: int, value: bytes) -> bytes:
    return data[:at] + value + data[at + len(value) :]


def test_corrupt_files_are_rejected():
    good = dn.pack(dn.Header(dof=6), dn.lcg_weights(6, 2))
    param = lambda i: dn.HEADER.size + 8 * i  # noqa: E731
    in_std, out_std = 18, dn.param_count(6) - 6
    bad = [
        _patch(good, 7, b"\x01"),  # version 1
        _patch(good, 9, b"\x02"),  # mode
        _patch(good, 12, b"\x01"),  # reserved
        good[:-8],
        good + bytes(8),
        _patch(good, 16, struct.pack("<d", float("nan"))),  # gravity
        _patch(good, 40, struct.pack("<d", -1.0)),  # payload mass
        _patch(good, param(1000), struct.pack("<d", float("inf"))),
        _patch(good, param(in_std + 3), struct.pack("<d", 1e-13)),
        _patch(good, param(out_std + 2), struct.pack("<d", 0.0)),
    ]
    for data in bad:
        with pytest.raises(dn.DexNextError):
            dn.unpack(data)


def test_stride_rounds_half_up():
    assert [dn.stride_ticks(f) for f in (1000, 500, 250, 150, 149, 100, 30)] == [10, 5, 3, 2, 1, 1, 1]


def test_window_outputs_match_a_straight_loop():
    """The batched reference equals running each left-padded window alone."""
    w = dn.lcg_weights(2, 7)
    u = dn.parity_inputs(2, 60, 1)
    batched = dn.window_outputs(w, u)
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    H = dn.HIDDEN
    for n in (0, 1, 30, 59):
        window = [u[max(n - 49 + j, 0)] for j in range(dn.WINDOW)]
        h, c = [np.zeros(H), np.zeros(H)], [np.zeros(H), np.zeros(H)]
        for row in window:
            inp = (row - w.in_mean) / w.in_std
            for l, (w_ih, w_hh, b) in enumerate([(w.w_ih0, w.w_hh0, w.b0), (w.w_ih1, w.w_hh1, w.b1)]):
                g = b + w_ih @ inp + w_hh @ h[l]
                c[l] = sig(g[H : 2 * H]) * c[l] + sig(g[:H]) * np.tanh(g[2 * H : 3 * H])
                h[l] = sig(g[3 * H :]) * np.tanh(c[l])
                inp = h[l]
        y = (w.b_out + w.w_out @ np.maximum(w.b_mlp + w.w_mlp @ h[1], 0.0)) * w.out_std + w.out_mean
        np.testing.assert_allclose(batched[n], y, rtol=0, atol=1e-12)
    # The first 50 samples share the padded prefix, so an unchanged input repeats.
    still = np.repeat(u[:1], 3, axis=0)
    out = dn.window_outputs(w, still)
    np.testing.assert_allclose(out[0], out[2], rtol=0, atol=1e-13)


# --------------------------------------------------------------------------
# PyTorch training and export (``pip install .[next]``)
# --------------------------------------------------------------------------


@pytest.fixture
def T():
    pytest.importorskip("torch")
    from codegen.next import train_next

    return train_next


def test_pytorch_reproduces_the_fixture(T):
    from codegen.next import make_fixtures as M

    committed = np.fromfile(FIXTURE, dtype="<f8").reshape(-1, 6)
    assert committed.shape == (M.SAMPLES, 6)
    assert np.max(np.abs(M.expected() - committed)) < T.PARITY_TOLERANCE


def test_export_matches_pytorch(T):
    import torch

    torch.manual_seed(1)
    model = T.NextObserverNet(6).eval()
    rng = np.random.default_rng(0)
    norm = T.Normalizers(rng.uniform(-0.1, 0.1, 18), rng.uniform(0.5, 2.0, 18), np.zeros(6), np.ones(6))
    data = T.export_dexnext(model, dn.Header(dof=6), norm, dn.parity_inputs(6, 80, 2))
    _, weights = dn.unpack(data)
    np.testing.assert_array_equal(weights.w_hh1, model.lstm.weight_hh_l1.detach().double().numpy())
    np.testing.assert_array_equal(weights.w_out, model.head[3].weight.detach().double().numpy())
    b = (model.lstm.bias_ih_l0 + model.lstm.bias_hh_l0).detach().double().numpy()
    np.testing.assert_allclose(weights.b0, b, rtol=0, atol=1e-7)


def test_from_weights_round_trips(T):
    w = dn.lcg_weights(3, 4)
    model = T.from_weights(w)
    norm = T.Normalizers(w.in_mean, w.in_std, w.out_mean, w.out_std)
    np.testing.assert_array_equal(T.to_weights(model, norm).flat(), w.flat())


def synthetic_log(path: Path, seconds: float = 6.0, hz: int = 500) -> None:
    """A 2-DOF 'arm' with viscous + Coulomb friction and gravity, two segments."""
    t = np.arange(0.0, seconds, 1.0 / hz)
    q = np.stack([0.8 * np.sin(1.3 * t), 0.5 * np.sin(0.7 * t + 1.0)], axis=1)
    qd = np.gradient(q, t, axis=0)
    qdd = np.gradient(qd, t, axis=0)
    tau_model = 3.0 * np.sin(q) + 0.1 * qdd  # the driver's rigid-body dynamics
    tau = tau_model + 0.4 * qd + 0.2 * np.tanh(20.0 * qd)
    seg = np.zeros(len(t), dtype=np.int64)
    seg[len(t) // 2 :] = 1
    seg[len(t) // 2 - 10 : len(t) // 2 + 10] = -1
    np.savez(
        path,
        tick=np.arange(len(t)),
        t=t,
        q=q,
        qd=qd,
        q_cmd=q + 0.01 * qd,
        qdd_ref=qdd,
        tau_link=tau,
        tau_model=tau_model,
        segment=seg,
        dof=2,
        control_freq_hz=hz,
        gravity=np.array([0.0, 0.0, -9.81]),
        payload_mass=0.5,
        payload_com=np.array([0.0, 0.0, 0.05]),
        payload_inertia=np.zeros(9),
    )


def test_train_cli_writes_a_loadable_observer(T, tmp_path):
    log = tmp_path / "log.npz"
    out = tmp_path / "arm.dexnext"
    synthetic_log(log)
    assert T.main([str(log), "-o", str(out), "--max-epochs", "2"]) == 0
    header, _ = dn.unpack(out.read_bytes())
    assert header.dof == 2 and header.mode == dn.FULL_DYNAMICS
    assert header.gravity == (0.0, 0.0, -9.81)
    assert header.payload_mass == 0.5  # full dynamics keeps the training payload


def test_training_reduces_the_loss(T):
    rng = np.random.default_rng(0)
    segs = []
    for _ in range(4):
        u = rng.normal(size=(200, 6))
        segs.append(T.Segment(u, np.stack([u[:, 0] + 0.5 * u[:, 2], np.tanh(u[:, 1])], axis=1)))
    losses = []
    T.train(
        segs[:3],
        segs[3:],
        T.TrainConfig(max_epochs=8, batch=64),
        log=lambda m: losses.append(float(m.split()[-1])),
    )
    assert losses[-1] < 0.5 * losses[0]


def test_segments_skip_discarded_frames(T):
    ids = np.array([-1, 0, 0, 0, -1, -1, 1, 1, 2, -1])
    assert T.split_segments(ids) == [slice(1, 4), slice(6, 8), slice(8, 9)]


def test_inputs_are_position_velocity_and_command_error(T, tmp_path):
    log = tmp_path / "log.npz"
    synthetic_log(log)
    d = np.load(log)
    segs = T.load_segments(log, dn.FULL_DYNAMICS, 5)
    first = T.split_segments(d["segment"])[0]
    idx = np.arange(first.start, first.stop, 5)
    q, qd = d["q"][idx], d["qd"][idx]
    np.testing.assert_array_equal(segs[0].u, np.concatenate([q, qd, d["q_cmd"][idx] - q], axis=1))
    np.testing.assert_array_equal(segs[0].y, d["tau_link"][idx])


def test_residual_targets_subtract_the_logged_model_torque(T, tmp_path):
    log = tmp_path / "log.npz"
    synthetic_log(log)
    d = np.load(log)
    segs = T.load_segments(log, dn.FRICTION_RESIDUAL, 5)
    first = T.split_segments(d["segment"])[0]
    idx = np.arange(first.start, first.stop, 5)
    friction = d["tau_link"] - d["tau_model"]
    np.testing.assert_array_equal(segs[0].y, friction[idx])

    out = tmp_path / "arm.dexnext"
    assert T.main([str(log), "-o", str(out), "--mode", "residual", "--max-epochs", "1"]) == 0
    header, _ = dn.unpack(out.read_bytes())
    # The logged model torque already includes the collection payload.
    assert header.mode == dn.FRICTION_RESIDUAL and header.payload_mass == 0.0


def test_residual_mode_needs_the_model_torque(T, tmp_path):
    log = tmp_path / "log.npz"
    synthetic_log(log)
    stripped = tmp_path / "stripped.npz"
    np.savez(stripped, **{k: v for k, v in np.load(log).items() if k != "tau_model"})
    with pytest.raises(SystemExit):
        T.main([str(stripped), "-o", str(tmp_path / "x.dexnext"), "--mode", "residual"])
