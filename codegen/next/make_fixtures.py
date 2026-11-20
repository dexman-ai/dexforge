"""Regenerate the Rust/PyTorch parity fixture (factr2.md §6, criterion 1).

    python -m codegen.next.make_fixtures [--check]

The weights are not committed: Rust (``torque_observer::test_support::params``)
and Python (:func:`codegen.next.dexnext.lcg_weights`) draw the same
parameters bit for bit from a shared LCG, and the same for the inputs
(``parity_inputs``). This writes ``dexhal/tests/fixtures/next/parity.expected``
(``SAMPLES × dof`` little-endian ``f64``): PyTorch float64 ``model.eval()``
output of every window. The Rust test
``dexhal::math::torque_observer::tests::pytorch_parity`` runs the observer in
a 1 kHz loop over the same inputs and requires ``max_abs_err < 1e-10``.
``--check`` recomputes and compares instead of writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from codegen.next import dexnext as dn  # noqa: E402
from codegen.next.train_next import PARITY_TOLERANCE, Normalizers, from_weights, torch_reference  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "dexhal" / "tests" / "fixtures" / "next" / "parity.expected"
DOF = 6
# Mirrors PARITY_SEED, PARITY_INPUT_SEED and PARITY_SAMPLES in torque_observer.rs.
WEIGHT_SEED = 2026
INPUT_SEED = 59
SAMPLES = 100


def expected() -> np.ndarray:
    weights = dn.lcg_weights(DOF, WEIGHT_SEED)
    u = dn.parity_inputs(DOF, SAMPLES, INPUT_SEED)
    norm = Normalizers(weights.in_mean, weights.in_std, weights.out_mean, weights.out_std)
    got = torch_reference(from_weights(weights), norm, u)
    err = float(np.max(np.abs(got - dn.window_outputs(weights, u))))
    assert err < PARITY_TOLERANCE, f"numpy parity {err:.3e}"
    return got


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    got = expected()
    if a.check:
        committed = np.fromfile(FIXTURE, dtype="<f8").reshape(-1, DOF)
        err = float(np.max(np.abs(got - committed))) if committed.shape == got.shape else float("inf")
        print(f"{FIXTURE.name}: max_abs_err {err:.3e}")
        return 0 if err < PARITY_TOLERANCE else 1
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_bytes(got.astype("<f8").tobytes())
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
