# dexhal work items

Open work to make dexhal the complete, single driver contract. Ordered by
priority within each group. Tags mark work that reaches outside dexhal:
**[codegen]** dexforge generators, **[driver]** OEM driver crates,
**[dexzero]** the DexZero repository. Items are prefixed **Done** or
**Postponed**; unprefixed items are open.

## Decisions taken

- **dexhal owns the driver contract.** OEM drivers depend on dexhal only.
  DexZero adopts dexhal's trait, manifest and buffer layouts instead of
  defining its own.
- **OEM inputs** are a URDF, a driver manifest and a hardware API / protocol
  spec. Rigid-body code is generated from the URDF; the four I/O methods are
  generated from the hardware spec.
- **dexhal is the contract, the common math and the tests.** Each driver owns
  its IK and its dynamics (`dexforge-ur` is the reference partition). dexhal
  holds no solver and no arm-specific code or data; its tests run on a
  stand-in arm (`src/math/testing.rs`).
- **No backward compatibility** is kept while the contract settles.

## Open decisions

- Closed: **Plugin boundary (item 26):** drivers are loaded via a separate `dexhal-plugin` crate that is
  allowed `unsafe`. The only unsafe component in dexzero.

## P0 — correctness of the hot path

The kernel calls `inverse_kinematics` every control cycle and writes the result
straight to `target_q`.

1. Done: **Seeded calibrated IK.** `ik_nearest` asks the driver's solve for the
   one solution nearest the seed and polishes only that. Was: Replace `ik_nearest`'s solve-and-polish-every-branch
   with: pick the nominal branch nearest the seed, then polish only that one on
   the calibrated chain.
2. Done: **Continuity guard in `ik_nearest`.** `ik_max_joint_step` (default
   π/2), settable on `Arm`, `ArmMath` and `ArmDriver`. Was: Return `None` (or a flagged result)
   when the nearest solution is further from the seed than a configurable
   joint-step bound, instead of silently jumping branches.
3. Done: **Calibrated `ik` fixes.** Polish before truncating to `max_solutions`,
   re-deduplicate after the polish, and take the closure tolerance from the
   caller's policy rather than a private constant.
4. Done: **Zero-copy state / command views.** `CoreStateView(Mut)`,
   `CoreCommandView(Mut)`, offsets from `offset_of!`. Was: Add safe offset-based views over
   `&[f64]` / `&mut [f64]` for `CoreState` and `CoreCommand`; keep the copying
   functions only as helpers. Required, since dexhal's layouts are the only
   ones.
5. Done: **Safe manifest defaults.** `JointRatings` is a required argument
   of `ArmManifest::for_model`; `acceleration_limits` added. Taking the values
   from the URDF is part of item 6. Was: Make velocity and effort limits mandatory inputs
   to the manifest constructor (no infinite defaults), sourced from the URDF,
   and add acceleration limits.

## P0 — make the surface real

6. Postponed: **[codegen] Emit the full `ArmModel`.** Generate `NAME`, `JOINT_LIMITS`,
   `HAS_DYNAMICS` and the dynamics hooks from the URDF, so no OEM writes them by
   hand. Until then no arm reports limits or dynamics through `Arm`.
7. Postponed: **[codegen] Emit a nominal chain per arm.** Each driver exports its nominal
   `KinematicChain` in driver frames, plus the DH-to-driver frame adapters where
   the vendor calibration is DH-based (UR).
8. Postponed: **[driver] Real-driver tests.** Test `Arm` against an actual driver for a
   6-DOF and a 7-DOF arm: chain FK equals generated FK, calibrated IK closes,
   dynamics add up. Depends on 6 and 7.
9. Done: **[dexzero] Adopt dexhal's contract.** DexZero deleted its own copies of the
   trait, manifest and buffer structs and uses dexhal's directly (`dexzero.md`,
   `compliance.md`, `api.md`, `ipc.md`).
10. Postponed: **Make the host model optional in `ArmDriver`.** `arm()` returns `Option`,
    so a driver without a dexhal model can implement the trait and the kernel
    falls back to its own math. (DexZero's simulator takes its model from the
    simulated plugin.)

## P0 — OEM workflow

11. Done: **Manifest as a data file.** `ArmManifest::from_text`, documented by
    `examples/manifest.yaml`. Was: Define a schema (the `schema.yaml` files in the
    DexZero drivers are the starting point) and generate or load `ArmManifest`
    from it.
12. Postponed: **[codegen] Driver template and agent prompt.** A skeleton crate plus a
    prompt, alongside the IK translation prompts in `codegen/`, for generating
    `connect`, `disconnect`, `read_state` and `write_command` from an API spec.
13. Postponed: **Conformance test kit.** A reusable suite an OEM runs against their driver:
    buffer lengths match the manifest, unsupported fields are `NAN`, no
    allocation in the data plane, FK agrees with hardware-reported TCP.

## P1 — missing capability

14. Done: **Inverse dynamics.** `ArmModel::flange_inverse_dynamics` (the
    driver's generated routine), `Arm::inverse_dynamics`,
    `ArmDriver::inverse_dynamics`. Was: Add an `ArmModel` hook and
    `Arm::inverse_dynamics(q, qd, qdd)` including payload; expose it on
    `ArmMath`. Needed by the kernel's torque-residual observer.
15. Done: **Pseudo-inverse on the erased interface.** Expose the Jacobian
    pseudo-inverse (padded) on `ArmMath` / `ArmDriver`.
16. Done: **Tool-frame quantities.** `jacobian_tool` / `tool_jacobian` and the
    helpers in `math::spatial`. Was: Add a tool-frame Jacobian and helpers to move
    wrenches and diagonal stiffness between base and tool frames.
17. Done: **Calibration and gravity in the control plane.** `CalibrationData`.
    Was: Define a flat,
    `MAX_DOF`-padded calibration format and add `set_calibration` /
    `clear_calibration` / `set_gravity` to `ArmMath` and `ArmDriver`.
18. Done: **Calibration self-check.** `Arm::fk_residual`,
    `ArmDriver::calibration_residual`. Was: Compare host FK with hardware-reported
    `tcp_pose` and report the residual.
19. Done in dexhal: **Large cancelling DH offsets.** A chain whose `d` terms
    are hundreds of metres and cancel across parallel axes reproduces the
    nominal kinematics to 5e-12; no precision fix was needed.
    Open: **[driver] UR calibration file handling.** Parse `calibration.conf`
    in `dexforge-ur` and test it with a real robot's file (none was available),
    where the axes are only nearly parallel.

## P1 — evidence for the perf promise

20. Postponed: **Benchmarks.** Measure calibrated FK / Jacobian, IK polish and dynamics
    Postponed: against the uncalibrated path; record the numbers in the README.
21. Postponed: **Embedded build in CI.** Build dexhal for `thumbv7em-none-eabihf`.
22. Done: **Declare the minimum Rust version** (1.82), or remove the features that
    require it (inline `const` asserts, `Option::is_none_or`).

## P1 — contract plumbing

23. Postponed: **Contract versioning policy.** Define what bumps `CONTRACT_VERSION` and how
    the kernel rejects a mismatched driver. (DexZero's `boot::validate` checks
    `major` and `minor` equality against `CONTRACT_VERSION`.)
24. Done: **Resolve method-name clashes** (`ArmMath` queries with a typed twin
    end in `_padded`, its setters are `apply_*`). Was: between `Arm`'s inherent methods and
    `ArmMath` (`set_payload`, `jacobian`, `dynamics`).
25. Done: **Split `arm.rs`** into `model`, `arm` and `contract` (with
    `contract::manifest` and `contract::wire`), plus `spatial` and `payload`
    for the shared math.
26. **[dexhal] [driver] Plugin boundary (`dexhal-plugin`, DexZero Gap G1).**
    DexZero loads drivers as C-ABI shared-library plugins (`dexzero.md` §3).
    Provide a `dexhal-plugin` crate and export macro that wraps an `ArmDriver`
    and satisfies the six kernel boundary requirements:
    1. Version handshake (plugin ABI and `ContractVersion`) before any other call.
    2. FFI-safe manifest representation (`ArmManifest` currently holds
       `&'static str` in `name` and `AuxField::label`, and `from_text` is
       generic over the model type).
    3. Construction of the driver from configuration parameters (e.g. robot IP
       address; `connect()` currently takes no arguments).
    4. Independent instances of the arm's host model (`Box<dyn ArmMath>`): the
       planner thread and the simulator each need their own model instance
       because the real-time executor thread owns the driver (`ArmDriver` and
       `ArmMath` are now `Send`).
    5. No unwinding across the C-ABI boundary; errors returned as values.
    6. Data-plane calls keep dexhal's zero-allocation, non-blocking guarantees.

## P2 — cleanup

27. Postponed: **Move `identify.rs` out of the real-time crate** into host tooling or
    behind a feature flag.
28. Done: **Harden identification.** Add position / rotation weighting, noise and
    outlier tests, and an observability report; correct the README claim about
    recovering driver frames.
29. Done: **Remove duplicated math.** `chain` / `identify` use
    `kinematics::Transform`; the `Pose` copy of `se3_inv`, the local 3×3
    helpers and limit / distance loops are gone. `KinematicChain::refine_ik`
    stays (analytic Jacobian). Was: Reuse `kinematics.rs` / `ssik_core.rs` helpers
    and the existing refiners where they fit.
30. Done: **Collapse the parallel APIs.** The static TCP / payload methods are
    removed from `ArmModel`; `RigidPayload` is merged into `Payload`. Was:
    Deprecate the static TCP / payload methods on
    `ArmModel` in favour of `Arm`, and merge `Payload` and `RigidPayload`. Needs
    a driver-compatible migration.
31. Done: **Analytic payload Coriolis** in place of finite differences, and remove the
    redundant chain evaluation in `Arm::dynamics`.
32. **Update the top-level README**, which still documents the old API
    (`Xarm7::gravity(&q, &payload, ..)` and friends no longer exist).

## Follow-ups from the items above

33. **[driver] [codegen] Adopt the new model surface.** In each driver's
    `arm.rs` (and the `codegen/scaffold.py` template): declare `NAME` and
    `JOINT_LIMITS`, and wire the four
    dynamics hooks to `generated_dynamics` with `HAS_DYNAMICS = true`
    (`dexforge-ur` does not compile `generated_dynamics.rs` yet). Part of 6.
34. **[driver] Manifest and `ArmDriver` per arm.** A `manifest.yaml` and the
    four I/O methods for each driver.
35. Done: **[dexzero] Kernel use of the continuity bound.** `inverse_kinematics`
    returns `None` when the solution is more than π/2 from the seed; DexZero's
    scrubber interpolates waypoints densely enough to stay inside the bound and
    falls back to a damped least-squares step (`compliance.md` §7).

## DexZero contract alignment (from `dexzero/workitems.md` G1–G20)

36. Done: **Add `limits.jerk` (`jerk_limits`) and `has_static_inertials` to
    `ArmManifest` (G2, G18).** `limits.jerk` is parsed as an optional positive
    per-joint list (`NAN` when omitted so the kernel applies its ramp-time
    fallback); `has_static_inertials` defaults to `ArmModel::HAS_DYNAMICS` and
    can be set to `false` in `manifest.yaml` when static URDF inertials are
    unavailable or unsuitable for `HostImpedance`.
37. Done: **Expand `CoreCommand` with `task_frame: [f64; 7]` and document
    Cartesian wire frames (G3).** `CoreCommand` (96 × `f64`) carries
    `task_frame` for delegated native compliance (`NativeCartesianImpedance`,
    `NativeCartesianAdmittance`), and Cartesian wire quantities are defined at
    the TCP origin in base axes.
38. Done: **Expand `CoreState` with `timestamp`, `robot_mode`, `fault_code`, and
    `native_reflex_fired` (G4, G11).** `CoreState` (56 × `f64`) carries the
    hardware timestamp / sample counter for dynamic $k$-deep latency alignment
    plus status, fault, and native reflex flags.
39. Done: **Align README and `contract` rustdoc with DexZero (G17) and add
    `Send` bounds to `ArmDriver` / `ArmMath` (G1).** Removed outdated
    comparisons against DexZero's retired local contract.
40. **[driver] [codegen] Seeded 7-DOF IK locks the redundant joint to the seed
    (G19).** The four 7-DOF solvers (`franka_panda`, `xarm7`, `rizon4`,
    `rizon10`) currently fix the redundant joint to a 16-value grid. When
    `q_seed` is provided, lock or parameterize the redundant joint at the
    seed's value so seeded path solving does not step across grid samples.
41. **Remaining contract semantics and limits (G5–G10, G12–G16, G20):**
    - **G5 / G6:** Specify sign and gravity/payload conventions for
      `CoreState::torques` / `wrench` and `JOINT_TORQUE` mode in driver specs.
    - **G7:** Optional explicit active-mode selector in `CoreCommand`.
    - **G8:** Optional native compliance limits (max stiffness, min damping/mass,
      $\Delta x_{\max}$) and Cartesian speed/acceleration limits in `ArmManifest`.
    - **G9:** Add joint mass to `CoreCommand` or remove `native_joint_admittance`.
    - **G10:** Optional command-side aux schema if drivers need a tail after
      `CoreCommand`.
    - **G12–G16, G20:** Analog outputs / gripper manifest limits, driver-side
      command timeout specification, enable/disable/brake lifecycle hooks,
      vendor command response buffer, optional `target_qdd`, and redundant
      null-space behavior under native Cartesian control.
