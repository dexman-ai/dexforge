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

The kernel calls `ik_one` (`IkKind::Fast`) every control cycle and writes the result
straight to `target_q`.

1. Done: **Seeded calibrated IK.** `ik_one` asks the driver for one
   seeded solution and polishes only that. Was: Replace `ik_nearest`'s solve-and-polish-every-branch
   with: pick the nominal branch nearest the seed, then polish only that one on
   the calibrated chain.
2. Done: **Continuity guard in `ik_one`.** `ik_max_joint_step` (default
   π/2, enforced per candidate after polishing; can only be tightened; the
   7-DOF lock joint is further bounded by `IK_LOCK_MAX_STEP` = π/8), settable on `Arm`, `ArmMath` and `ArmDriver`. Was: Return `None` (or a flagged result)
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

6. Done: **[codegen] Emit the full `ArmModel`.** Generate `NAME`, `JOINT_LIMITS`,
   `HAS_DYNAMICS` and the dynamics hooks from the URDF, so no OEM writes them by
   hand. Until then no arm reports limits or dynamics through `Arm`.
7. Done: **[codegen] Emit a nominal chain per arm.** Each driver exports its nominal
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
12. Done: **[codegen] Driver template and agent prompt.** A skeleton crate plus a
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
19. Done: **Large cancelling DH offsets.** A chain whose `d` terms
    are hundreds of metres and cancel across parallel axes reproduces the
    nominal kinematics to 5e-12; no precision fix was needed.
    Done: **[driver] UR calibration file handling.** `ur5::calibration` parses
    the `[mounting]` deltas of `calibration.conf` into a calibrated chain /
    `CalibrationData`; tested with synthetic files, including large cancelling
    deltas on nearly parallel axes (no real robot's file was available).

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
26. Done: **[dexhal] [driver] Plugin boundary (`dexhal-plugin`, from `dexzero/workitems.md` G1; DexZero item 5).**
    DexZero loads drivers built later by third parties as C-ABI shared-library
    plugins (`dexzero.md` §3), so this is a prerequisite for every real arm
    (DexZero item 5). Provide a `dexhal-plugin` crate and export macro that
    wraps an `ArmDriver` and solves the four things the contract does not cover
    today (and satisfies the six kernel boundary requirements):
    - *The manifest is not FFI-safe.* `ArmManifest` holds `&'static str` fields
      (in `name` and `AuxField::label`), and `from_text` is generic over the
      model type.
    - *Drivers have no configuration input.* `connect()` takes no arguments and
      nothing in the contract carries the robot's address (e.g. robot IP
      address).
    - *The host model cannot be used from a second thread.* The math is reached
      through `&self` of the driver, which the real-time loop borrows mutably
      every cycle. The planner and the simulator each need an independent model
      instance (`Box<dyn ArmMath>`; `ArmDriver` and `ArmMath` now have `Send`
      bounds, item 53, but the plugin factory still needs to export independent
      model instances).
    - *Versioning.* A handshake that rejects a mismatched plugin (plugin ABI and
      `ContractVersion` checked before any other call; see item 23).
    - *C-ABI safety and real-time guarantees.* No unwinding across the C-ABI
      boundary (errors returned as values), and data-plane calls keep dexhal's
      zero-allocation, non-blocking guarantees.
    - **Recommendation (based on DexZero requirements):**
      - *Manifest FFI:* Keep `ArmManifest` in `dexhal` unchanged or add fixed
        inline UTF-8 byte arrays (`[u8; 32]`) in a `#[repr(C)]` FFI descriptor
        in `dexhal-plugin` (avoiding heap strings and keeping `#![no_std]` in
        `dexhal`). Have the plugin build its `ArmManifest` inside the plugin via
        `ArmManifest::from_text::<Model, DOF, MAX>(MANIFEST)` and expose it over
        C-ABI, so the kernel never needs the generic `Model` type.
      - *Driver configuration:* Pass configuration at plugin construction time
        (`create_driver(config_json_or_kv: *const u8, len: usize)`) rather than
        changing `ArmDriver::connect(&mut self)`. In `dexzero.md` §3, `arms.toml`
        supplies per-arm driver config (such as robot IP / interface) at boot;
        storing it in the driver struct at construction keeps `connect()`
        signature-stable and parameter-free.
      - *Independent `ArmMath` instances:* Export a separate C-ABI factory
        function `create_math() -> *mut c_void` (wrapping `Arm::<Model, DOF, MAX>::new()`)
        alongside `create_driver`. That directly satisfies `dexzero.md` §3
        requirement 4 so the executor thread owns the driver while the planner
        thread and simulator each get their own `Box<dyn ArmMath>`.
    - **Done:** crate `dexhal/plugin` (`dexhal-plugin`): `#[repr(C)]` `ManifestDesc`, a vtable with ABI / `ContractVersion` handshake, `create_driver(config)` and an independent `create_math()`, panics caught at the boundary, `export_plugin!` for the driver side and `Plugin::load` / `LoadedDriver` / `LoadedMath` for the host. The mock driver is the reference export (`mock/rust/src/plugin.rs`, tests in `mock/rust/tests/plugin.rs`).

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
32. Done: **Update the top-level README**, which still documents the old API
    (`Xarm7::gravity(&q, &payload, ..)` and friends no longer exist).
    - **Done:** the README documents the current `Arm` / `ArmDriver` API.

## Follow-ups from the items above

33. Done: **[driver] [codegen] Adopt the new model surface.** In each driver's
    `arm.rs` (and the `codegen/scaffold.py` template): declare `NAME` and
    `JOINT_LIMITS`, and wire the four
    dynamics hooks to `generated_dynamics` with `HAS_DYNAMICS = true`
    (`dexforge-ur` does not compile `generated_dynamics.rs` yet). Part of 6.
    - **Done:** every driver declares `NAME` / `JOINT_LIMITS` and wires `generated_dynamics` with `HAS_DYNAMICS = true` (including `dexforge-ur`).
34. **[driver] Manifest and `ArmDriver` per arm.** A `manifest.yaml` and the
    four I/O methods for each driver.
    Manifests are done for every driver. Open: the four I/O methods are still
    `TODO(agent)` stubs; they need each vendor's SDK / protocol and hardware
    to test against.
35. Done: **[dexzero] Kernel use of the continuity bound.** `ik_one`
    returns `None` when no solution is within π/2 per joint of the seed
    (`IK_DEFAULT_MAX_JOINT_STEP`; 7-DOF lock joint π/8, checked on every candidate after polishing); DexZero's
    scrubber interpolates waypoints densely enough to stay inside the bound and
    falls back to a damped least-squares step (`compliance.md` §7).

## DexZero contract alignment (from `dexzero/workitems.md` G1–G20)

Each item below originates from the gap analysis in `c:\code\dexzero\workitems.md`
(`G1`–`G20`; `G1` is tracked as item 26 above and item 53 below).

36. Done: **[dexhal] Add `limits.jerk` to `ArmManifest` (from `dexzero/workitems.md` G2; DexZero items 11, 16, 17).**
    `limits.jerk` (`ArmManifest::jerk_limits: [f64; MAX_DOF]`) is parsed as an
    optional positive per-joint list in `manifest.yaml`. The scrubber's
    closed-form quintic jerk verifier and the reactive generator both need
    per-joint jerk limits. Fallback when omitted (`NAN` in `jerk_limits`): a
    kernel setting derives jerk from the acceleration limit and a ramp time
    (DexZero items 11, 16, 17).
37. Done: **[dexhal] Define Cartesian wire frames and expand `CoreCommand` / `write_command` (from `dexzero/workitems.md` G3).**
    `wrench`, `tcp_twist`, `target_twist`, `target_wrench`, `task_stiffness_k`,
    `task_damping_b`, `task_mass_m`, and `reflex_threshold` are defined at the
    TCP in base axes, and `CoreCommand` (96 × `f64`) has a `task_frame: [f64; 7]`
    slot so delegated native compliance (URScript `force_mode`, Franka;
    `NativeCartesianImpedance`, `NativeCartesianAdmittance`) can stream the
    compliance task frame, diagonal $(M, D, K)$, and $F_{\text{virt}}$ alongside
    $(q_{\text{ref}}, \dot{q}_{\text{ref}})$, and real-time MIT-mode torque
    drivers (AgileX Piper) can stream the 5-tuple
    `(q_ref, qd_ref, kp, kd, tau_ff)`. Interim in DexZero: `compliance.md` §3–5
    (`(primary_route, fallback_host_route)`).
38. Done: **[dexhal] Add status/fault code and `native_reflex_fired` to `CoreState` (from `dexzero/workitems.md` G4; DexZero items 10, 26, 33).**
    Added `robot_mode` (enabled, protective stop, emergency stop), `fault_code`,
    and `native_reflex_fired` / `contract_broken` flag reported by controller
    scripts (such as URScript) to `CoreState` (56 × `f64`). Previously
    `clear_faults` existed, but a fault was visible only as an `Err(Fault(..))`
    from a call. Needed for the fault state machine and reflex handling in
    DexZero items 10, 26, and 33.
39. Done: **[dexhal] Semantics of `torques` and `wrench` (from `dexzero/workitems.md` G5).**
    Define in the contract the exact semantics of `CoreState::torques` and
    `CoreState::wrench`: sign convention, motor side vs. external, and whether
    gravity and payload are included or already subtracted. Interim in DexZero:
    `compliance.md` §6.
    - **Recommendation (based on DexZero requirements — keep the driver a dumb pass-through):**
      Standardize on raw, uncompensated signals matching `compliance.md` §6, and
      declare any hardware-level pre-processing in `manifest.yaml` so the driver
      never runs coordinate transforms, Jacobians, or dynamics:
      - `CoreState::torques`: link-side joint effort in the positive joint
        direction (including gravity and payload). A dumb driver that only reads
        motor phase current $i$ simply writes $k_t \cdot n_{\text{gear}} \cdot i$
        (one constant multiply per joint, or sets `sensing.torque_constants` in
        `manifest.yaml` and passes raw motor current/effort straight through);
        the kernel's torque-residual observer (Route 6) computes
        $r = \tau_{\text{model}}(q_{\text{actual}}, \dot{q}_{\text{actual}}, \ddot{q}_{\text{ref}}) - \tau_{\text{measured}}$
        and $F_{\text{ext}} = (J^T)^+ r$ on the host (`compliance.md` §2, §6).
      - `CoreState::wrench`: raw wrench `[Fx, Fy, Fz, Tx, Ty, Tz]` that the
        environment / payload applies *onto* the tool (including tool/payload
        weight when untared). Rather than forcing a dumb driver with a wrist F/T
        sensor to run `fk(q)` to rotate sensor readings into base axes or
        re-reference them to the TCP, add `sensing.wrench_frame: base | flange | tcp`
        (default `base`) and `sensing.wrench_is_gravity_compensated: bool`
        (default `false`) to `ArmManifest`. The driver copies the 6 sensor floats
        straight into `CoreState::wrench`, and the kernel (which already evaluates
        `fk(q_actual)` and payload gravity every cycle in `compliance.md` §6)
        rotates, shifts to the TCP, and subtracts payload weight on the host.
    - **Done:** `sensing.torque_constants` (`ArmManifest::link_torques`), `sensing.wrench_frame`, documented `CoreState::torques` / `wrench` semantics, and `ArmDriver::external_wrench` / `Arm::external_wrench`.
40. Done: **[dexhal] Gravity compensation in `JOINT_TORQUE` mode (from `dexzero/workitems.md` G6; DexZero item 35).**
    Whether the hardware controller adds gravity compensation to
    `CoreCommand::target_torques` in `JOINT_TORQUE` mode is not declared by the
    contract or `ArmManifest`. Interim in DexZero: the kernel sends total torque
    evaluated at $(q_{\text{actual}}, \dot{q}_{\text{actual}})$. Blocks DexZero
    item 35 on hardware.
    - **Recommendation (based on DexZero requirements):** Add a boolean
      capability flag `native.gravity_compensation: bool` (default `false`) to
      `ArmManifest` (`native_gravity_compensation` in `manifest.yaml`) rather
      than picking a single fixed convention for all arms. Industrial torque
      controllers (e.g., Franka `libfranka`, Flexiv RDK) automatically add
      their own internal gravity and friction compensation to commanded torques
      and expect *additional* task/impedance torque only, whereas bare actuator
      / MIT-mode buses (e.g., Unitree Z1, AgileX Piper) expect total torque or
      explicit $\tau_{\text{ff}}$. With this flag:
      - When `native_gravity_compensation == false`: the kernel includes
        $\tau_{\text{model}}(q_{\text{actual}}, \dot{q}_{\text{actual}})$ in
        `target_torques` (current `compliance.md` §4 behavior).
      - When `native_gravity_compensation == true`: the kernel omits arm link
        gravity from `target_torques` (and includes payload gravity only when
        `set_payload` returned `Applied::Host`), allowing `HostImpedance`
        (Route 3) and `NativeJointImpedance` (Route 4) to work even on arms with
        `has_static_inertials == false` whose controller compensates gravity
        internally.
    - **Done:** `native.gravity_compensation`.
41. Done: **[dexhal] Active-mode selector in `CoreCommand` (from `dexzero/workitems.md` G7).**
    There is currently no active-mode selector in `CoreCommand`. The mode is
    implied by which targets are finite; precedence when several are finite is
    undefined, and a controller for which a mode switch is not real-time safe
    cannot say so in `ArmManifest`.
    - **Recommendation (based on DexZero requirements — keep `write_command` a dumb switch):**
      Add both an explicit `control_mode: f64` slot to `CoreCommand` and a
      `native.realtime_mode_switch: bool` (default `false`) field to
      `ArmManifest`:
      - Without `CoreCommand::control_mode`, a driver must inspect combinations
        of `is_finite()` across `target_q`, `target_qd`, `target_torques`,
        `target_pose`, `joint_stiffness_k`, and `task_stiffness_k` to guess what
        packet to send (especially since Route 4 MIT mode and Route 2 populate
        multiple target arrays at once). With an explicit `CoreCommand::control_mode`
        slot (`NAN` = idle/hold), `write_command` is a single `match` on one tag
        that copies the corresponding slice onto the wire.
      - In DexZero (`compliance.md` §3), the kernel switches between
        `primary_route` and `fallback_host_route` at `configure_compliance`
        cycle boundaries. Adding `native.realtime_mode_switch: bool` (default
        `false`) to `ArmManifest` lets the *kernel* reject live mode switches
        (`NOT_AT_REST`) unless the arm is at rest, so the driver never has to
        implement real-time mode-transition state machines.
    - **Done:** `CoreCommand::control_mode` / `ActiveMode` (`active_mode()` on the views) and `native.realtime_mode_switch`.
42. Done: **[dexhal] Compliance and Cartesian limits in `ArmManifest` (from `dexzero/workitems.md` G8; DexZero item 30).**
    `ArmManifest` has no compliance limits (maximum stiffness, minimum damping
    or mass the controller executes stably, and maximum spring force /
    deflection clamp $\Delta x_{\max}$), and no Cartesian speed or acceleration
    limit. Interim in DexZero: configured bounds (DexZero item 30).
    - **Recommendation (based on DexZero requirements):** Add optional `[f64; 6]`
      (`NAN` when omitted) compliance bounds and `[f64; 2]` (`[linear, angular]`,
      `NAN` when omitted) Cartesian motion bounds to `ArmManifest`:
      - `limits.cartesian_velocity: [linear_m_s, angular_rad_s]` and
        `limits.cartesian_acceleration: [linear_m_s2, angular_rad_s2]`, matching
        DexZero's `(linear, angular)` `speed_limit` / `accel_limit` tuple in
        `api.md` §4 (`ipc.md` §3 `cmd.speed_limit` / `cmd.accel_limit`).
      - Optional `compliance:` section in `manifest.yaml` with 6-element
        `[X, Y, Z, RX, RY, RZ]` arrays: `max_stiffness_k`, `min_damping_b`,
        `min_mass_m`, and `max_wrench` (from which DexZero derives
        $\Delta x_{\max} = F_{\max} / K$ in `compliance.md` §4–5). Keeping them
        optional (`NAN` when omitted) lets the kernel fall back to its existing
        latency-derived and conservative defaults (`compliance.md` §5) when an
        OEM does not specify them.
    - **Done:** `limits.cartesian_velocity` / `cartesian_acceleration` and the `compliance:` section (`ComplianceLimits`).
43. Done: **[dexhal] Resolve `native_joint_admittance` vs. `CoreCommand` fields (from `dexzero/workitems.md` G9).**
    `ArmManifest::native_joint_admittance` has no corresponding command fields
    in `CoreCommand` (there is `joint_stiffness_k` and `joint_damping_b`, but no
    joint mass / inertia). Either add the command fields or drop the flag.
    Interim in DexZero: not routed.
    - **Recommendation (based on DexZero requirements):** **Drop
      `native_joint_admittance`** from `ArmManifest` (and `manifest.yaml`)
      rather than adding `joint_mass_m` to `CoreCommand`. DexZero's compliance
      architecture (`compliance.md` §2) has 7 routes and never routes joint-space
      admittance (admittance is always integrated in Cartesian task space via
      Route 2, 5, or 6, while joint-space delegation uses MIT impedance Route 4).
      No supported or planned robot exposes a native joint-admittance controller,
      so removing the unused flag simplifies the manifest and avoids widening
      `CoreCommand` by 8 `f64` slots every cycle.
    - **Done:** `native.joint_admittance` removed.
44. Done: **[dexhal] Command-side tail schema (from `dexzero/workitems.md` G10).**
    There is no command-side tail: `ArmManifest::command_len()` is fixed at
    `CoreCommand::LEN` although `CoreCommandView` / `CoreCommandViewMut` expose
    `tail()`, and the manifest has an `AuxSchema` for state only.
    - **Recommendation (based on DexZero requirements):** Keep `command_len()`
      fixed at `CoreCommand::LEN` and **remove `tail()` from `CoreCommandView` /
      `CoreCommandViewMut`** (rather than adding a `cmd_aux` schema), unless a
      concrete driver requires per-cycle custom command floats. In DexZero
      (`ipc.md` §1–4, `api.md` §7), the real-time `cmd` shared-memory channel has
      no aux command pass-through; all standard real-time control inputs (including
      `task_frame` and MIT 5-tuple) are now first-class fields in `CoreCommand`,
      and non-standard vendor features go through the non-real-time
      `handle_vendor_command` (`ctl_req` op `VENDOR`). Removing the unused
      `CoreCommandView::tail()` eliminates the inconsistency without dead schema
      complexity.
    - **Done:** command `tail()` removed; `command_len()` is `CoreCommand::LEN`.
45. Done: **[dexhal] Add hardware timestamp or monotonic sample counter to `CoreState` (from `dexzero/workitems.md` G11; DexZero item 37).**
    Added `CoreState::timestamp: f64` (hardware timestamp in seconds or
    monotonic sample counter, `NAN` when not reported). A static
    `state_staleness_cycles` count cannot describe a link with variable
    network/controller delay $k$. Required for dynamic $k$-deep history lookup
    and state rebasing (DexZero item 37).
46. Done: **[dexhal] Peripherals: digital I/O precision, analog output, and gripper manifest limits (from `dexzero/workitems.md` G12).**
    The digital I/O bitmask (`io_digital_in`, `io_digital_out_mask`,
    `io_digital_out_state`) is an `f64`, exact for 53 bits only, which the
    contract should state across all digital I/O fields (`io_digital_in` rustdoc
    notes this); there is no analog output in `CoreCommand`; and `ArmManifest`
    has no gripper range (width) or force limit.
    - **Recommendation (based on DexZero requirements):**
      - *Digital I/O:* Keep `f64` bitmasks (preserving the homogeneous `f64`
        buffer invariant in `dexhal` and `ipc.md` §1) and explicitly document the
        53-bit (`0..53`) exact integer range on `io_digital_out_mask` and
        `io_digital_out_state` as well as `io_digital_in`.
      - *Gripper limits in `ArmManifest`:* Add optional `gripper.max_width: f64`
        (m) and `gripper.max_force: f64` (N) (`NAN` when omitted) to
        `ArmManifest` so DexZero can validate `gripper(width, speed, max_force)`
        and per-waypoint `gripper` lanes (`api.md` §4, §7) synchronously against
        hardware bounds when `has_integrated_gripper` is `true`.
      - *Analog output:* Add `io_analog_out: [f64; 4]` (`NAN` when not commanded)
        to `CoreCommand` symmetric with `CoreState::io_analog_in: [f64; 4]`.
    - **Done:** 53-bit range documented on all digital I/O fields, `CoreCommand::io_analog_out`, `gripper.max_width` / `max_force`.
47. Done: **[dexhal] Driver behavior when commands stop (from `dexzero/workitems.md` G13).**
    Driver behavior when commands stop arriving (kernel crash, missed cycles) is
    unspecified. A driver-side timeout and its action belong in the contract.
    - **Recommendation (based on DexZero requirements — no timer threads in the driver):**
      Do **not** require the Rust `ArmDriver` to spawn a background watchdog
      thread or synthesize braking trajectories in software (which would violate
      the synchronous pass-through model):
      - On normal stops or client disconnects (`api.md` §4.3), the DexZero kernel
        stays alive and streams a jerk-limited `stop()` and holds position.
      - If the kernel process itself crashes or freezes, `write_command` is no
        longer called, so no Rust code in the in-process driver plugin runs
        anyway. The only protection that can act is the **hardware controller's
        or motor drive's own built-in command-stream watchdog** (e.g., CAN MIT
        watchdog, UR RTDE watchdog, `libfranka` packet timeout).
      - Therefore, `connect()` only configures the hardware's native watchdog
        register (if the hardware has one), and when `write_command` receives
        `CoreCommand::IDLE` (all `NAN`), the driver forwards the hardware's
        native hold / zero-damped command (`kp = 0, kd = kd_safe, tau = 0` or
        hold current position). Add `latency.hardware_watchdog_ms: f64` (`NAN`
        when the hardware has none) to `ArmManifest` so the capability is
        declared statically.
    - **Done:** `latency.hardware_watchdog_ms` and the idle / watchdog contract on `CoreCommand` and `connect`.
48. Done: **[dexhal] Lifecycle beyond `connect` / `disconnect` (from `dexzero/workitems.md` G14).**
    `ArmDriver` lifecycle is currently `connect` / `disconnect` (plus
    `clear_faults`) only: there are no `enable`, `disable`, or brake-release
    hooks.
    - **Recommendation (based on DexZero requirements — keep the lifecycle flat):**
      Do **not** add separate `enable()`, `disable()`, or `release_brakes()`
      state-machine methods to `ArmDriver`. Keep the lifecycle to `connect()`,
      `disconnect()`, and `clear_faults()`, matching `api.md` §2, §6 and
      `ipc.md` §4:
      - Specify in the contract that `connect()` powers on, releases brakes, and
        enables the arm ready for `write_command`, and `disconnect()` disables /
        stops it cleanly.
      - If the hardware enters a protective stop or disabled state at runtime, it
        reports that via `CoreState::robot_mode` / `fault_code` (`G4`), and the
        existing `clear_faults()` hook re-enables the drives. Any vendor-specific
        commissioning action (e.g., manual brake release) goes through
        `handle_vendor_command`.
    - **Done:** lifecycle documented on `connect` / `disconnect` / `clear_faults`.
49. Done: **[dexhal] Return data from `handle_vendor_command` (from `dexzero/workitems.md` G15).**
    `ArmDriver::handle_vendor_command(&mut self, cmd_name: &str, payload: &[u8]) -> Result<(), ArmError>`
    cannot return data to the caller.
    - **Recommendation (based on DexZero requirements):** Change the signature to
      write into a caller-supplied mutable byte slice and return the number of
      bytes written:
      `fn handle_vendor_command(&mut self, cmd_name: &str, payload: &[u8], response: &mut [u8]) -> Result<usize, ArmError>`.
      This preserves `dexhal`'s `#![no_std]` and zero-allocation invariants, is
      trivially C-ABI safe for `dexhal-plugin` (item 26), and maps directly onto
      DexZero's fixed-size `ctl_rsp.values` (`104` slots in `ipc.md` §4).
    - **Done:** signature changed as recommended (`ArmError::BufferTooSmall` when the reply does not fit).
50. Done: **[dexhal] Acceleration on the wire (`target_qdd`) (from `dexzero/workitems.md` G16).**
    There is no `target_qdd` in `CoreCommand` although the trajectory generator
    produces joint acceleration. Low priority.
    - **Recommendation (based on DexZero requirements):** Add
      `pub target_qdd: [f64; MAX_DOF]` (`NAN` when not commanded) to
      `CoreCommand`. In DexZero (`dexzero.md` §5.4, `compliance.md` §4), the
      executor's quintic evaluator and reactive generator already compute
      `reference.qdd` every cycle (and use it in `HostImpedance` inverse
      dynamics and `state.qdd_ref` in `ipc.md` §3). Adding `target_qdd` lets
      position/velocity controllers that accept feedforward acceleration (e.g.,
      UR `servoj` / `speedj` acceleration parameter, Flexiv, or high-rate CAN
      drives) use it directly without finite-differencing `target_qd`.
    - **Done:** `CoreCommand::target_qdd` (`CoreCommand` is 109 × f64; contract 0.3.0).
51. Done: **[dexhal] Documentation alignment with DexZero (from `dexzero/workitems.md` G17).**
    Updated the README's "Driver contract" section (formerly "Differences from
    the DexZero spec") and the `contract` module docs (formerly "it mirrors
    DexZero's driver contract") so they no longer define dexhal relative to a
    retired DexZero-local contract that no longer exists.
52. Done: **[dexhal] Add `has_static_inertials: bool` to `ArmManifest` (from `dexzero/workitems.md` G18; DexZero items 9, 29, 35).**
    Some arms obscure URDF mass/inertia data (e.g., Flexiv) or rely on
    proprietary internal friction models (e.g., UR harmonic drives). Added
    `has_static_inertials: bool` to `ArmManifest` (parsed from `manifest.yaml`,
    defaulting to `ArmModel::HAS_DYNAMICS`). When `false`, skip SymPy inverse
    dynamics generation, disable `HostImpedance` (Route 3), and either delegate
    compliance natively (`force_mode`) or stream runtime dynamics matrices
    $(M, C, G)$ via `CoreState` aux fields into shared memory for Python
    Operational Space Control (DexZero items 9, 29, 35).
53. Done: **[dexhal] `Send` bounds and independent host-model instances for `ArmDriver` / `ArmMath` (from `dexzero/workitems.md` G1, host-model thread-safety sub-item).**
    Added `Send` bounds to `ArmDriver` and `ArmMath` so the real-time executor
    thread can own `Box<dyn ArmDriver>` while the planner and simulator threads
    each hold an independent `Box<dyn ArmMath>` (`Arm<Model, DOF, MAX>`). The
    remaining C-ABI plugin boundary work for G1 is tracked in item 26.
54. Done: **[dexhal] [driver] [codegen] Redundant arms: seeded 7-DOF IK locks the redundant joint to the seed instead of a fixed grid (from `dexzero/workitems.md` G19; DexZero items 15, 19).**
    All four 7-joint solvers in dexforge (`franka_panda`, `xarm7`, `rizon4`,
    `rizon10`) lock one joint to 16 fixed sample values and refine with that
    joint held. A seeded solve therefore cannot keep the seed's value of that
    joint. From an arbitrary posture the first solve moves it to the nearest
    sample, which is joint motion with no TCP motion; along a path the solution
    changes sample in steps of a sixteenth of the joint's range, which is inside
    the continuity bound and so is not rejected. Proposal: a seeded solve locks
    the joint at the seed's value. Interim in DexZero: `compliance.md` §7
    (DexZero items 15, 19).
    - **Recommendation (based on DexZero requirements):**
      - When `q_seed: Some(seed)` is passed to a 7-DOF `solve()` (which is what
        `Arm::ik_nearest` and `ArmDriver::inverse_kinematics` did then), compute
        the 6R sub-chain geometry online at the locked angle `q_lock = seed[LOCKED_JOINT]`
        first (or run the analytical 6R solve at `seed[LOCKED_JOINT]` followed by
        full 7-DOF damped least-squares / Levenberg-Marquardt polish with diagonal
        joint weights $W$ as required by `compliance.md` §7 and `dexzero.md` §5.2),
        falling back to the 16-sample sweep only when `q_seed` is `None` (unseeded
        global search) or when the seed's locked angle yields no reachable branch.
      - In `refine7`, allow full 7-column Jacobian LM refinement weighted toward
        `q_seed` when seeded, so that even if a grid sample is used as an initial
        guess, polishing converges onto a solution that preserves the seed's
        redundant posture rather than pinning the locked joint to the coarse grid.
    - **Done:** superseded by item 61. `ik_one` with `IkKind::Near` holds the locked joint at `seed[LOCKED_JOINT]` in every stage (seed-lock polish, then the two nearest samples re-polished at the seed value, then masked DLS from the seed); `IkKind::Fast` starts with full-joint DLS and may move it. `ik_all` always runs the full sweep. Shared by every joint-lock arm through `dexhal::math::single::seven_dof_one` (tests `seeded_solve_keeps_the_seed_locked_joint` in `rizon4` / `rizon10`, Near/Fast path tracking in each joint-lock crate's `rust/tests/ik_path.rs`, shared checks in `dexhal/conformance/ik_path.rs`).
55. Done: **[dexhal] Null space under native Cartesian control (from `dexzero/workitems.md` G20).**
    For a redundant arm the contract neither declares what a native Cartesian
    impedance or admittance controller does with the null space nor lets the
    kernel command a posture. Interim in DexZero: left to the controller.
    - **Recommendation (based on DexZero requirements):** Reuse the existing
      `CoreCommand::target_q: [f64; MAX_DOF]` (and `joint_stiffness_k` /
      `joint_damping_b` when supported) as the optional desired null-space
      posture $q_{\text{nominal}}$ during `NativeCartesianImpedance` (Route 1) and
      pose-commanded `NativeCartesianAdmittance` (Route 2), and add a boolean
      capability `native.nullspace_posture: bool` (default `false`) to
      `ArmManifest`:
      - In DexZero (`compliance.md` §2, §4), the executor already has the
        7-DOF reference joint posture `reference.q` ($q_{\text{nominal}}$) every
        cycle. Once item 41 (`G7`) adds an explicit `control_mode` selector to
        `CoreCommand`, setting `target_q = reference.q` alongside `target_pose`
        in `NativeCartesianImpedance` is unambiguous and requires zero new wire
        slots in `CoreCommand`.
      - Controllers that accept a null-space posture target in Cartesian
        impedance mode (e.g., Franka `CartesianImpedance` with nullspace
        stiffness, Flexiv) set `native.nullspace_posture: true` and read
        `target_q`; controllers that manage redundancy internally leave it `false`
        and ignore `target_q`.
    - **Done:** `native.nullspace_posture`; `ActiveMode::CartesianImpedance` / `CartesianAdmittance` document `target_q` as the posture.
56. Done: **[dexhal] Validate `has_static_inertials` against `model.dynamics` (`ArmModel::HAS_DYNAMICS`).**
    `ArmManifest::from_text` previously accepted `has_static_inertials: true` even
    when `A::HAS_DYNAMICS == false` (`manifest.model.dynamics == false`), leaving
    a contradictory manifest that claimed host-side inverse dynamics were
    trustworthy for control while `ArmMath::dynamics_padded` and
    `ArmMath::inverse_dynamics_padded` returned `None`. `ArmManifest::from_text`
    (and `dexhal_plugin::ManifestDesc::to_manifest`) now reject
    `has_static_inertials: true` and `has_residual_inertials: true` with
    `ManifestErrorKind::InvalidValue` when `!A::HAS_DYNAMICS`.
57. Done: **[dexhal] [dexzero] Decouple dynamics usability for Route 3 (`HostImpedance`) vs. Route 6 (`HostAdmittanceTorque`).**
    Added `has_residual_inertials: bool` to `ArmManifest` (parsed from
    `manifest.yaml`, defaulting to `has_static_inertials`, which defaults to
    `ArmModel::HAS_DYNAMICS`) alongside `has_static_inertials: bool`, plus
    `ArmManifest::has_impedance_dynamics()` (for Route 3 `HostImpedance`) and
    `ArmManifest::has_residual_dynamics()` (for Route 6 `HostAdmittanceTorque`),
    so disabling Route 3 on a position/velocity-controlled arm does not
    automatically disable Route 6 torque-residual wrench estimation.
58. Done: **[dexhal] [driver] [dexzero] Optional vendor runtime dynamics override via `CoreState` `aux`.**
    Added `DynamicsSource` (`Model` default, `Vendor` / `aux` override) and
    `dynamics_source: DynamicsSource` to `ArmManifest`, canonical `aux_dynamics`
    field labels (`gravity_torque` (`DOF`), `coriolis_torque` (`DOF`),
    `mass_matrix` (`DOF * DOF` row-major)), `AuxSchema::offset_of` / `slice` /
    `has_vendor_dynamics`, and `ArmManifest::aux_dynamics(state_buffer)`.
    `ArmManifest::from_text` (and `ManifestDesc::to_manifest`) validate that
    when `dynamics_source: vendor` is set, all three `aux` fields are declared
    with exact sizes (`ManifestErrorKind::InvalidAux` otherwise).

## P1 — missing capability (cont.)
59. Done: **[dexhal] [codegen] FACTR 2 (NEXT) Virtual Torque Sensor Integration** (see [`specs/factr2.md`](specs/factr2.md)).
    `math::torque_observer::LstmObserver<DOF>` (`no_std`, no allocation,
    `forbid(unsafe_code)`) decodes a validated `.dexnext` file in place and
    runs the fixed FACTR 2 network (Table 3 / A.2: inputs `[q, q̇, q_cmd − q]`,
    2 × 128 LSTM, 256-wide MLP head, stateless 50-sample window at 100 Hz,
    left-padded with the first sample after a reset), holding its output
    between samples. The 49 prefix frames of the next window are spread over
    the ticks of the 100 Hz period (bit-identical to a full evaluation);
    `step` returns `None` for the first `stride − 1` ticks after a reset.
    `Arm::set_torque_observer` / `clear_torque_observer` /
    `reset_torque_observer` / `payload_torque` / `step_free_torque` add the
    payload differential $\tau_{\text{payload}}(P_{\text{runtime}}) -
    \tau_{\text{payload}}(P_{\text{train}})$ and, in `FrictionResidual`, the
    model torque; `set_gravity` rejects a gravity that differs from the
    `FullDynamics` training gravity. `ArmMath` gains
    `step_free_torque_padded` / `reset_torque_observer_state` /
    `payload_torque_padded`; `ArmDriver::step_free_torque` /
    `reset_torque_observer` fall back to model or vendor `aux` dynamics.
    `DynamicsSource::Learned` (`dynamics_source: learned`) and
    `ArmManifest::has_torque_observer` (set by the driver; enables
    `has_residual_dynamics()`); `ModelError::InvalidObserver`;
    `CONTRACT_VERSION` 0.4. `dexhal-plugin` ABI 3: flags `LEARNED_DYNAMICS`
    / `TORQUE_OBSERVER`, error code 6, vtable entries `math_step_free_torque`,
    `math_reset_torque_observer`, `math_payload_torque`. The mock driver takes
    `MockDriver::with_torque_observer` or plugin config `observer=<path>`.
    `codegen/next`: `.dexnext` format and NumPy reference (`dexnext.py`),
    `train_next.py` (training, export, parity check; residual targets use the
    logged `tau_model`, the driver's own inverse dynamics, not pinocchio),
    `make_fixtures.py`. Acceptance: PyTorch float64 parity < 1e-10 over 100
    windows stepped at 1 kHz (`torque_observer::tests::pytorch_parity`;
    weights and inputs come from a shared LCG, only
    `tests/fixtures/next/parity.expected` is committed); plugin round trip and
    rejections (`mock/rust/tests/plugin.rs`); zero allocation in the driver
    tick and across the plugin, and WCET < 1 ms (`mock/rust/tests/observer_rt.rs`:
    `LstmObserver::step` at 1 kHz ~0.16 ms mean / ~0.24 ms worst, a whole
    window at 100 Hz ~1.7 ms worst of 10 ms, on a desktop x86-64);
    `thumbv7em-none-eabihf` build.
    Deviations from the spec: weights stay in the `&'static [u8]` (≈ 1.9 MB
    for 6 joints, stored transposed), so drivers use `include_bytes!` or
    `Box::leak`; `FrictionResidual` is our extension (the paper trains full
    torque); head dropout placement (after the ReLU) is assumed; a third vtable
    entry `math_payload_torque` serves the
    vendor `aux` path; `FrictionResidual` is rejected whenever
    `!HAS_DYNAMICS`. Data collection (spec §3.1, including logging
    `tau_model`), kernel wiring and acceptance criterion 4 (payload / contact
    isolation in the `dexzero` sim) are **[dexzero]** work.
60. Done: **[dexhal] [dexzero] Consolidate stateless SE(3), weighted DLS, and Khatib operational-space projectors in `dexhal::math`.**
    Moved all stateless `#![no_std]` spatial, kinematic, and operational-space
    linear algebra out of `dexzero` (`kernel/src/math.rs` and `sim/src/lib.rs`)
    into `dexhal::math` (`chain.rs`, `spatial.rs`):
    `rotation_exp`, antipodal-safe `rotation_error` / `pose_error`,
    `se3_offset`, `se3_lerp`, `se3_distance`, `transform_point` /
    `inverse_transform_point`, `transform_from_xyz_quat` /
    `transform_to_xyz_quat`, `rotate_spatial` / `rotate_spatial_inv`,
    `jacobian_twist`, `jacobian_transpose_wrench`, `mat6_vec`,
    `pseudo_inverse_transpose_torque`, `diag_jt_gain_j`,
    `damped_least_squares`, `weighted_damped_least_squares`,
    `dynamically_consistent_pseudo_inverse`, `nullspace_project_torque`,
    `solve_mass_matrix`, and `payload_wrench_at_tcp`. IK stays
    `ArmMath::ik_one` / `ik_all` (the driver's closed-form solve with its
    bounded polish, see item 61). `kinematics::dls_ik` is used internally by
    the 7-DOF `Fast` stage; a caller needing a generic DLS IK for a model
    without a closed-form solver (`MockArm`) builds it from
    `damped_least_squares`. `dexzero` (`sim` and `kernel`) now imports zero
    `nalgebra` items and owns strictly time, trajectories, PD control laws, and
    safety state machines.
61. Done: **[dexhal] [driver] [codegen] Unified IK entry points and shared IK code.**
    - **API.** `Arm::ik_all(target, q_seed, ..)` returns every solution (no
      early exit; joint-lock arms run the full sweep), sorted nearest-first to
      the seed. `Arm::ik_one(target, q_seed, kind, policy)` returns one
      solution and exits early; `kind: IkKind` is `Fast` (default) or `Near`.
      6-DOF: `Fast` runs a seeded analytic `refine6` first and falls back to the
      enumeration; `Near` ranks the enumeration by seed distance (nearest).
      7-DOF: `Fast` = full-joint DLS, then seed-lock polish, then the two
      nearest lock samples pinned to the sample value; `Near` = seed-lock
      polish (20 iterations), then the two nearest samples with the lock at
      the seed value, then masked DLS from the seed with the lock at the seed
      value (150 iterations). Every stage returns its first accepted solution,
      so 7-DOF `Near` is nearest only by stage order. The servo path
      (`sim_driver.py`, README examples) uses `Fast`.
    - **Contract / plugin.** `ArmMath::ik_one(.., kind)` / `ik_all(.., out)`;
      plugin ABI 4 (`math_ik_one` with a `u8` kind code, `math_ik_all`). A
      short `ik_all` buffer receives as many (nearest) solutions as fit.
    - **Shared code.** `math::single` (`six_dof_one`, `seven_dof_one`),
      `math::closed_form` (`finish6`, analytic `refine6`) for UR5 / UR3e / Z1 /
      Piper, `math::jointlock` (spherical-inner joint-lock 7R) for xArm7 /
      Franka, shared Raghavan–Roth helpers for Piper / Rizon, and
      `dexforge-ur`'s `drivers/ur_common.rs` (UR5 / UR3e differ only in a
      `Model` of geometry literals). No driver uses finite differences.
    - **Perf.** `dexhal/perf/arm_timing.rs` reports mean, robust max (10
      slowest seeds re-run 100× each) and hit rate for `ik_all`, `ik_one`
      (`Fast`) and `dls_ik`; results in the root README.
