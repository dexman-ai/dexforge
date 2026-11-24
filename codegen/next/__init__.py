"""FACTR 2 (NEXT) learned torque observer tooling.

* :mod:`codegen.next.dexnext` — the fixed-architecture ``.dexnext`` v2 weight
  format and a NumPy reference of the stateless window evaluated by
  ``dexhal::math::torque_observer`` (NumPy only).
* :mod:`codegen.next.train_next` — PyTorch training and ``.dexnext`` export
  (``pip install .[next]``) from a free-space log recorded by ``dexzero``.
* :mod:`codegen.next.make_fixtures` — regenerates the Rust/PyTorch parity
  fixture ``dexhal/tests/fixtures/next/parity.expected``.

See ``dexhal/specs/factr2.md``.
"""
