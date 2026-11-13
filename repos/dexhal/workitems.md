# dexhal work items

Open work to make dexhal the complete, single driver contract. Ordered by
priority within each group. Tags mark work that reaches outside dexhal:
**[codegen]** dexforge generators, **[driver]** OEM driver crates,
**[dexzero]** the DexZero repository.

## Decisions taken

- **dexhal owns the driver contract.** OEM drivers depend on dexhal only.
  DexZero adopts dexhal's trait, manifest and buffer layouts instead of
  defining its own.
- **OEM inputs** are a URDF, a driver manifest and a hardware API / protocol
  spec. Rigid-body code is generated from the URDF; the four I/O methods are
  generated from the hardware spec.

## Open decisions

- **Plugin boundary (item 26):** a separate `dexhal-plugin` crate that is
  allowed `unsafe`, or lifting `#![forbid(unsafe_code)]` inside dexhal. The
  separate crate is the proposed answer.

## P0 — correctness of the hot path

The kernel calls `inverse_kinematics` every control cycle and writes the result
straight to `target_q`.

1. **Seeded calibrated IK.** Replace `ik_nearest`'s solve-and-polish-every-branch
   with: pick the nominal branch nearest the seed, then polish only that one on
   the calibrated chain.
2. **Continuity guard in `ik_nearest`.** Return `None` (or a flagged result)
   when the nearest solution is further from the seed than a configurable
   joint-step bound, instead of silently jumping branches.
3. **Calibrated `ik` fixes.** Polish before truncating to `max_solutions`,
   re-deduplicate after the polish, and take the closure tolerance from the
   caller's policy rather than a private constant.
4. **Zero-copy state / command views.** Add safe offset-based views over
   `&[f64]` / `&mut [f64]` for `CoreState` and `CoreCommand`; keep the copying
   functions only as helpers. Required, since dexhal's layouts are the only
   ones.
5. **Safe manifest defaults.** Make velocity and effort limits mandatory inputs
   to the manifest constructor (no infinite defaults), sourced from the URDF,
   and add acceleration limits.

## P0 — make the surface real

6. **[codegen] Emit the full `ArmModel`.** Generate `NAME`, `JOINT_LIMITS`,
   `HAS_DYNAMICS` and the dynamics hooks from the URDF, so no OEM writes them by
   hand. Until then no arm reports limits or dynamics through `Arm`.
7. **[codegen] Emit a nominal chain per arm.** Each driver exports its nominal
   `KinematicChain` in driver frames, plus the DH-to-driver frame adapters where
   the vendor calibration is DH-based (UR).
8. **[driver] Real-driver tests.** Test `Arm` against an actual driver for a
   6-DOF and a 7-DOF arm: chain FK equals generated FK, calibrated IK closes,
   dynamics add up. Depends on 6 and 7.
9. **[dexzero] Adopt dexhal's contract.** DexZero deletes its own copies of the
   trait, manifest and buffer structs and re-exports dexhal's;
   `driver_contract.md` is rewritten to point at dexhal.
10. **Make the host model optional in `ArmDriver`.** `arm()` returns `Option`,
    so a driver without a dexhal model (the sim driver) can implement the trait
    and the kernel falls back to its own math.

## P0 — OEM workflow

11. **Manifest as a data file.** Define a schema (the `schema.yaml` files in the
    DexZero drivers are the starting point) and generate or load `ArmManifest`
    from it.
12. **[codegen] Driver template and agent prompt.** A skeleton crate plus a
    prompt, alongside the IK translation prompts in `codegen/`, for generating
    `connect`, `disconnect`, `read_state` and `write_command` from an API spec.
13. **Conformance test kit.** A reusable suite an OEM runs against their driver:
    buffer lengths match the manifest, unsupported fields are `NAN`, no
    allocation in the data plane, FK agrees with hardware-reported TCP.

## P1 — missing capability

14. **Inverse dynamics.** Add an `ArmModel` hook and
    `Arm::inverse_dynamics(q, qd, qdd)` including payload; expose it on
    `ArmMath`. Needed by the kernel's torque-residual observer.
15. **Pseudo-inverse on the erased interface.** Expose the Jacobian
    pseudo-inverse (padded) on `ArmMath` / `ArmDriver`.
16. **Tool-frame quantities.** Add a tool-frame Jacobian and helpers to move
    wrenches and diagonal stiffness between base and tool frames.
17. **Calibration and gravity in the control plane.** Define a flat,
    `MAX_DOF`-padded calibration format and add `set_calibration` /
    `clear_calibration` / `set_gravity` to `ArmMath` and `ArmDriver`.
18. **Calibration self-check.** Compare host FK with hardware-reported
    `tcp_pose` and report the residual.
19. **UR calibration file handling.** Test with real `calibration.conf` values,
    including large cancelling `d` terms on the parallel axes; fix precision if
    needed.

## P1 — evidence for the perf promise

20. **Benchmarks.** Measure calibrated FK / Jacobian, IK polish and dynamics
    against the uncalibrated path; record the numbers in the README.
21. **Embedded build in CI.** Build dexhal for `thumbv7em-none-eabihf`.
22. **Declare the minimum Rust version** (1.82), or remove the features that
    require it (inline `const` asserts, `Option::is_none_or`).

## P1 — contract plumbing

23. **Contract versioning policy.** Define what bumps `CONTRACT_VERSION` and how
    the kernel rejects a mismatched driver.
24. **Resolve method-name clashes** between `Arm`'s inherent methods and
    `ArmMath` (`set_payload`, `jacobian`, `dynamics`).
25. **Split `arm.rs`** into model, arm and contract modules.
26. **Plugin boundary.** DexZero loads drivers as C-ABI plugins; provide an
    export macro that wraps an `ArmDriver` in that ABI. See *Open decisions*.

## P2 — cleanup

27. **Move `identify.rs` out of the real-time crate** into host tooling or
    behind a feature flag.
28. **Harden identification.** Add position / rotation weighting, noise and
    outlier tests, and an observability report; correct the README claim about
    recovering driver frames.
29. **Remove duplicated math.** Reuse `kinematics.rs` / `ssik_core.rs` helpers
    and the existing refiners where they fit.
30. **Collapse the parallel APIs.** Deprecate the static TCP / payload methods on
    `ArmModel` in favour of `Arm`, and merge `Payload` and `RigidPayload`. Needs
    a driver-compatible migration.
31. **Analytic payload Coriolis** in place of finite differences, and remove the
    redundant chain evaluation in `Arm::dynamics`.
32. **Update the top-level README**, which still documents the old API.
