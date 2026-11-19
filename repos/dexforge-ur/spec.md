# UR arm driver spec (dexhal driver in `dexforge-ur`, for dexzero and dexforge)

Revision 7 reworks the register contract again (B3, D8, B4, C3, C10), marked **[r7]**:
- Per-axis reflex thresholds move to doubles 42–47 on every motion command. Block 3 is retired.
- Reflex frame rotation moves into the force-detection block (2).
- Tare FT and gripper calibration become bits in the fixed int 31, echoed in output int 40. Block 6 is retired.
- Gripper open/closed counts are no longer sent at runtime. They come from the manifest.

Revision 6 reworks the register contract (B3, D8), marked **[r6]**:
- Payload, TCP and gravity become one standstill-only block.
- Tare moves into the force-detection block.
- The reflex action and watchdog limit get fixed int registers.
- `cmd_timestamp` becomes a wrapping µs int, so admittance gains and frame fit in one block and nothing is staged.
- The gripper registers are renumbered, and calibration counts are restored through a config block.
- `rqcontrol.script` changes accordingly.

Revision 5 added D8, multiplexing the RTDE registers between motion commands and a config command. As a result, B3 was rewritten to fit the real 24–47 register range, and C6, B2, B4 and WI-23/26–29/31/32/34/38 were updated. These changes are marked **[r5]**.

Revision 4 restructured r3 into three parts and reconciled them with the review decisions:
- **Part A** — what the driver must do, independent of dexware.
- **Part B** — how dexware's implementation is ported or rewritten to get there.
- **Part C** — the numbered work items.

Work items are renumbered; a mapping from the r3 numbers is at the end of Part C.

---

# Part A — UR driver: goals, features and design

## A1. Goals

A UR arm under dexzero and dexforge should behave as a "smart arm" in the dexzero.md §1 sense: the controller does the fast things and the host does the planning.

- **G1. Interruptible tracking.** The arm follows a continuously replaced reference (dense or sparse) and never runs a blocking move. Any cycle can retarget, stop or switch mode.
- **G2. Immediate force response.** The controller detects contact and runs reflexes on its own 2 ms loop, without the host in the loop.
- **G3. Native compliance.** Admittance (M-D-K in a task frame, with an optional wrench bias) also runs on the controller's 2 ms loop. Host compliance stays available as a fallback.
- **G4. A measurable contract.**
  - Every state sample says which command it reflects.
  - Every controller-side override (reflex, protective stop, speed scaling) is visible to the kernel.
  - No event is lost, even when a state frame is.
- **G5. End effector.** Robotiq 2F grippers in SI units, as a static per-variant capability.
- **G6. Multi-model.** One driver implementation for all e-series models (CB3 optional), with per-model kinematics and dynamics data only.
- **G7. Safe lifecycle.** Connecting and disconnecting are idempotent and cause no surprise motion: no gripper motion, no brake cycling, no automatic reconnect.

Non-goals:
- Safety-rated behavior. UR's safety configuration stays the safety layer; script reflexes are a functional layer.
- UR `force_mode()` (see D4).
- PolyScope X, until it is verified.

## A2. Architecture

```
dexzero kernel ──(CoreCommand / CoreState)──► UR plugin (UrDriver<Model>, one thread per arm)
   UR plugin ──RTDE 30004── output recipe 500 Hz (125 Hz CB3) + input registers (versioned register contract)
   UR plugin ──30003────── load no_op.script, then main.script (urcontrol + optional rqcontrol), constants injected
   UR plugin ──29999────── Dashboard: remote-mode check, power on, brake release, unlock protective stop
   main.script ──63352──── Robotiq gripper (rqlib socket), thread inside main.script (gripper variants only)
```

Crate layout in `dexforge-ur`:
- **`src/ur/`** is shared by all models: RTDE, Dashboard, script loader, register contract, the generic `UrDriver<M: UrModel>`, state and command mapping, gripper conversion, and `calibration.conf` parsing.
- **`src/drivers/<model>/`** holds model data only: generated kinematics, dynamics and IK, a generated `NOMINAL_DH`, the URDF and tests.
- **Manifests and plugins** come one per variant (`<model>[_<end effector>]`).

## A3. Features

| Area | Feature |
|---|---|
| Motion modes | `JointPos` (servo with velocity feedforward), `JointVel`, `CartesianPose` (servo with twist feedforward), `CartesianTwist`, `CartesianWrench` (zero-stiffness admittance with a wrench bias), `CartesianAdmittance` (native M-D-K) |
| Reference rate | Dense (500 Hz) or sparse waypoints (about 100–125 Hz) interpolated in the script; chosen per deployment (D1) |
| Force | Contact detection; per-axis reflexes STOP, YIELD and RETRACT in a reflex frame; latched reporting; native tare |
| State | Joint and TCP state, raw wrench, command echo, speed scaling, reflex and contact flags, standard fault codes, a `diag.*` block |
| Configuration | Native payload, TCP, gravity and tare, applied through an acknowledged register handshake |
| Gripper | Robotiq 2F width, speed and force in SI units; calibrated count mapping; object detection |
| Lifecycle | Dashboard-based power on and brake release on demand; safe hold on disconnect; `clear_faults` |
| Watchdogs | Script watchdog on command freshness, plus the RTDE input watchdog |

## A4. Contract decisions (dexhal and dexzero)

**C1. Command timestamp and echo.**
- `CoreCommand.cmd_timestamp` is set by the kernel. It is the kernel-clock time at which the reference applies, and it is strictly increasing.
- The driver passes it through unchanged. The script uses it for exactly three things:
  - **Watchdog:** a change of the value counts as a fresh command.
  - **Echo:** `CoreState.cmd_timestamp_echo` reports the newest value applied when the sample was produced.
  - **Interpolation timing:** in waypoint mode, segment durations come from differences between timestamps (D1).
- The kernel uses the echo to measure the round trip every cycle instead of inferring it with `find_by_time`.
- **Wire encoding** [r6]. On the wire (RTDE int register) the value is `floor(cmd_timestamp·10⁶) mod 2³¹`: a non-negative µs counter that wraps about every 35.8 min.
  - **Echo.** The driver keeps a small ring that maps each wire value it sent to the exact `f64` `cmd_timestamp`, and rebuilds `cmd_timestamp_echo` from it. An echo that isn't in the ring is reported as NaN.
  - **Uniqueness.** Within the ring, values are unique because kernel ticks are at least 1 ms apart.
  - **Differences in the script.** The script takes timestamp differences as `d = a − b; if d < 0: d = d + 2³¹`. Both values are in [0, 2³¹), so the subtraction can't overflow.
  - **Why µs rather than ms.** ms would also tell ticks apart, but µs gives waypoint interpolation sub-ms segment timing.
- NaN rules:
  - NaN in the command: the driver substitutes a counter.
  - NaN in the state: no echo is supported.

**C2. Promoted state fields.** These go into `CoreState` because control needs them:
- `cmd_timestamp_echo`.
- `speed_scaling`: the 0–1 fraction of commanded speed the controller is actually applying, covering the pendant slider and reduced mode; NaN when unknown.

Everything else goes into diagnostics (C9).

**C3. Reflex semantics.**
- **Disabled.** A NaN `reflex_action_code` means no native reflex. A NaN threshold component disables that axis. `in_contact` is still reported either way.
- **Reflex frame.** The kernel sends `reflex_frame` (or reuses `task_frame`) in config block 2 [r7], applied only at standstill. The controller rotates the *wrench* into that frame and compares it per axis, so the kernel no longer rotates or `abs`-es a threshold box.
- **`native_reflex_fired` codes:**

  | Code | Meaning |
  |---|---|
  | 0 | Armed or idle |
  | 1 | A commanded native reflex fired, and the controller has taken over motion |
  | 2 | A controller-internal protective force limit fired, not commanded by the kernel. It is treated as a reflex trigger and never as `CONTRACT_BROKEN`. |

- **Latch.**
  - The first sample after takeover reports the code, and its `timestamp` is the trigger time.
  - While latched, motion targets are ignored and the action keeps executing: STOP holds, YIELD stays compliant.
  - Response duration reported in `reflex_response_duration`.
- **Acknowledge.** The latch clears after one command with `control_mode` Idle or a NaN reflex action. dexzero already sends that when it aborts with `reason=REFLEX`. On UR the reflex action has a dedicated register that is valid in every packet, config cycles included (B3), so the acknowledgement is never delayed. [r6]
- **Re-arm.** Triggers are level-based: a new command that starts above threshold fires again at once. Leaving contact takes a reflex-free command or RETRACT.
- **Actions:**
  - **STOP:** stop and hold.
  - **YIELD:** zero-stiffness admittance (D4).
  - **RETRACT:** native STOP, and the kernel plans the retract.

**C4. A single wrench pipeline.**
- **One bias authority.** `tare_wrench_native()` uses the `Applied` / host-fallback pattern. A native tare calls `zero_ftsensor()` in the script, and the kernel resets its own bias.
  - There is no startup tare.
  - Without native tare, native reflexes are disabled.
- **The same conditioning on both sides.** The script and the kernel `Conditioner` both use a per-axis EMA with the same `compliance.native_filter_tau`, declared in the manifest. Neither models inertia.
- **Payload must match.** Native reflexes require native payload, because `actual_TCP_force` gravity compensation uses the PolyScope payload.

**C5. Delivery of state frames (nothing lost).**
- **Events can't be lost.** Reflex and fault flags are latched (C3), and contact duration accumulates in the controller.
- **Default: kernel clock.** The kernel keeps its own timer. `read_state` returns the newest sample. Samples skipped because of clock drift or jitter are delivered through a bounded catch-up ring, which dexzero ingests into its k-deep history in timestamp order. The driver counts `diag.dropped_samples`, which is only non-zero if that ring overflows.
- **Option: state-paced.** A driver may declare `latency.state_paced`. Its `read_state` can then block, up to a deadline, for the next hardware sample, and dexzero may tick on its arrival.
  - Each arm has its own executor thread, so this is compatible with multiple arms (D3).
  - Frames are then lost only on a kernel overrun, which is counted.

**C6. Limits.**
- The kernel owns limits.
  - Manifest limits are tightened per command (`speed_limit` / `accel_limit`, which already exist) or per session (sticky, a dexzero feature).
  - Widening beyond the manifest is not allowed.
- The script has only backstop clamps. They are at least as wide as the active kernel limits, so the script never silently clips a valid reference. [r5] The clamps can be changed at runtime through the limits config block (D8): the driver sets them from the manifest at connect, and updates them when session-level limits change.
- `speed_scaling < 1` is handled by the kernel: it either time-scales the reference or faults.

**C7. Faults.** dexhal adopts the standard `fault_code` set:
- 0 none
- 1 fault
- 2 protective stop
- 3 emergency stop
- 4 disabled
- 5 contract broken
- 6 comm lost

The UR mapping:

| UR condition | Code |
|---|---|
| Protective stop | 2 |
| E-stop | 3 |
| Not powered or brakes engaged | 4 |
| Script stopped or watchdog fired | 6 |
| Wrong register-contract version | 5 |

**C8. Lifecycle.**
- **Dashboard only.** On UR, power on, brake release and power off exist only through the Dashboard or the pendant, and e-series needs remote-control mode for them. "Engage brakes" means arm power off.
- **`connect`** is idempotent. It powers on and releases the brakes only if the arm isn't `RUNNING`, refuses when the safety mode isn't normal, and every step has a timeout.
- **`disconnect`** stops motion and the script and leaves the arm powered and holding ("safe hold"). Power off is opt-in, through the manifest's `lifecycle.disconnect_power_off`.
- **No automatic behavior.** Nothing reconnects automatically, and lifecycle actions never move the gripper.

**C9. Diagnostics.**
- A standard named `aux` block rather than a new wire struct:
  - `applied_mode`, `override_active`, `override_steps`
  - `stale_count`, `dropped_samples`, `cycle_overrun`
  - `executed_cmd[6]`, `filtered_wrench[6]`, `contact_duration`
  - `joint_temp[n]`, `joint_current[n]`
  - `controller_version`, `register_contract_version`
- NaN where a field isn't supported.
- Recorded and exposed by dexzero, never used for control.

**C10. Gripper.**
- **SI units only.** The driver converts to and from Robotiq counts using per-variant constants and the calibrated `open`/`closed` counts:

  | Quantity | Conversion |
  |---|---|
  | Width | `max_width·(closed − pos)/(closed − open)` |
  | Speed | 20–150 mm/s → 0–255 |
  | Force | 20–235 N → 0–255, a target only |

- **Endpoints.** A target ≥ `max_width` maps to `open`, and ≤ 0 maps to `closed`.
- **Hold.** NaN means hold the last value.
- **Measured force.** The state's `gripper_force` is the measured force, which is NaN for a Robotiq.
- **Static presence.** Presence and model are fixed per manifest variant. `connect` fails if the declared gripper is missing.
- **Calibration** is explicit, through a vendor command, and runs only on request — never at connect.
  - The counts come from the manifest's gripper section and are loaded into the script as constants. Nothing is persisted or restored at runtime. [r7]
  - Commands are rejected while the gripper is uncalibrated (`open == closed`).

**C11. Kinematics.**
- DH is static per model: generated from UR's published description files and baked in at codegen time.
- Per-robot calibration (`calibration.conf`) is applied through `set_calibration`. The analytic IK seeds, and dexhal refines on the calibrated chain.
- No runtime-parametric IK.

**C12. Plugins.** `PluginDriver::create_math()` takes no config, so either:
- each variant is its own cdylib, or
- `create_math(config)` is added, so one plugin can serve all models.

## A5. Control design decisions

**D1. Dense stream versus sparse waypoints ("smart mode").**

One script supports both. The deployment picks the mode with manifest `control_freq_hz` and the kernel configuration:
- **Dense mode (500 Hz).** The kernel streams `target_q`/`target_qd` every 2 ms, and the script servos with feedforward.
- **Waypoint mode (about 100–125 Hz).** The kernel streams time-stamped waypoints. The script runs one-waypoint-buffered cubic Hermite segments, with durations from `cmd_timestamp` differences, at 500 Hz.

| Criterion | Waypoint mode compared with dense |
|---|---|
| Responsiveness | Slightly worse: one extra waypoint period (about 8–10 ms). Small next to a 10–90 Hz policy. |
| Accuracy and precision | The same for smooth references; worse for high-bandwidth references. |
| Robustness | Better: it tolerates host and network jitter, so the host doesn't need to be real-time. |
| Force (keeping contact while tracing) | Not decided by the rate. Native admittance runs at 2 ms in both modes. Host admittance is weak in dense mode and unusable in waypoint mode. |

**Conclusion.** Native admittance (D2) is the core requirement. Waypoint mode is a deployment option, not a goal in itself.

**D2. Native admittance in the script.**
- **Law.** A per-axis M-D-K integrator in `task_frame`, driven by the conditioned wrench minus `target_wrench`. Its deflection is added to the servo target.
- **Gains.** Per-axis gains come from registers.
- **Wrench mode.** `CartesianWrench` is the same law with K = 0.
- **Interim law.** Until M-D-K is implemented, dexware's damper law applies (velocity = gain · deadband(F)), and `native.cartesian_admittance` stays false, so dexzero uses its host routes.
- **Why native.** The host admittance loop (`(n_s + n_c + 1)/f` ≈ 8 ms) limits K/M. UR's current-based joint torques make torque-residual sensing poor.

**D3. Joint-space versus Cartesian commands.**
- **IK cost doesn't depend on the command space.** Cartesian admittance needs one `get_inverse_kin` per cycle while it is active, whichever space the command is in.
- **What joint commands still buy:**
  - no mismatch between kernel and controller IK
  - no branch flips
  - zero IK cost while compliance is off
- **Both servos stay.**
  - Joint servo (30): used without Cartesian compliance.
  - Cartesian servo (31): `get_inverse_kin` with `qnear = q`, joint wrap, a blend weight clamped to at most 1, and rotation speed limited independently of translation. The kernel sends it while Cartesian compliance is active.
- **No joint-only script.**

**D4. Never use UR `force_mode()`.**
- **Why.** Entering or leaving it forces a stop and causes a click and shake.
- **What to use instead.** Every compliant behavior (admittance, wrench/guiding, YIELD) runs on the speed-control path (`speedj`/`speedl` plus velocity-from-force), so the arm is never in force mode.
- **Cost.** Wrench tracking is only as good as the admittance law, not UR's internal force loop. That's enough for light contact; it's no substitute for precise force regulation.
- **`freedrive_mode()`** isn't used either, until hardware shows it switches in without a stop.

**D5. Servo law with feedforward.**
- **Law.** `v = target_qd + corr(target_q − q)`, where `corr` is the `sqrt(a·e)` law with a linear region near zero, capped by the acceleration limit.
- **Why feedforward.** Without it, a ramp at speed `v` lags by `v²/a` (about 0.2 rad at 1 rad/s and 5 rad/s²). That breaks the kernel's fixed-delay assumption, shows up as tracking error, and gives a false residual.

**D6. Multi-arm timing.**
- dexzero runs one executor thread and timer per arm. `sync.rs` shares only a start barrier and the min phase-rate bus.
- Both C5 options work. Under state pacing, the arms' phases drift apart by about 1 tick per 10 s at 500 Hz (100 ppm). §6 step arbitration then aligns ticks by `state.timestamp` instead of by tick index.

**D7. Watchdogs.**
- **Script watchdog.** It counts cycles without a change of `cmd_timestamp`, and stops at the limit in the fixed watchdog register (B3). The default is 250 cycles (500 ms) = `hardware_watchdog_ms`. Because the kernel timestamp is unique per tick, this is a true link watchdog. The limit is valid and changeable in every packet.
- **RTDE input watchdog.** Also armed, where the controller version supports it.

**D8. Register multiplexing: motion commands and config commands.** [r5]

*Why multiplex.* RTDE exposes only 48 input doubles and 48 input ints (0–47).
- To coexist with fieldbus/PLC users of the lower range, the driver uses the upper range 24–47, as dexware does. That leaves 24 doubles and 24 ints.
- One cycle can't carry the motion target, feedforward and wrench bias, and also the reflex parameters, admittance gains, task frame, payload, TCP, gravity, limits and detector parameters.

*The approach.* `cmd_type` selects how the payload registers are read:
- **[r6] Fixed int registers, valid in every packet whatever the command type:**
  - `cmd_timestamp`, as a wrapping µs integer (C1)
  - `cmd_type`
  - the gripper ints
  - the reflex action
  - the watchdog cycles
  - the command flags: tare and gripper calibration requests (B3) [r7]
- **All 24 doubles are payload.** Motion and config both reinterpret the whole double range.
- **Motion commands** (0–50) carry only what changes every tick: the target, feedforward, wrench bias and the per-axis reflex thresholds [r7].
- **Config command** (`cmd_type` 90): the doubles and a few ints are reinterpreted as one config block, selected by a block id and carrying a sequence number. Block layout is in B3.

*Why these fields are fixed.*
- **Reflex action.** The kernel can enable, disable or acknowledge a reflex in any cycle, including a config cycle, with no scheduling delay. The latch acknowledgement (C3) never waits on the config queue.
- **Watchdog cycles.** The watchdog limit stays valid and changeable whatever command type is being sent.
- **Command flags.** A tare or calibration request is an edge on a fixed bit, so it is seen on any cycle and doesn't wait for the config queue. The standstill gate still applies. [r7]

*How the script handles a config cycle:*
- **Motion continues.** A config cycle is not a motion command, so the active motion mode keeps running on its last targets:
  - Dense mode extrapolates `target_q += target_qd·dt` for that cycle.
  - Waypoint mode continues its segment; segment timing uses only the timestamps of motion packets.
  - Idle stays idle.

  The motion target is at most one cycle stale. The reflex thresholds keep their last values, because doubles 42–47 carry block payload on a config cycle. [r7]
- **Fixed registers.** A config cycle counts as fresh for the watchdog, because its `cmd_timestamp` changed. The timestamp is echoed, and the reflex action and watchdog cycles are applied, as on any cycle.
- **Atomic apply.** The block is applied at the start of the cycle, before the force pipeline, so it takes effect within that cycle and atomically (one RTDE packet). [r6] Each block fits in one packet, so nothing is staged and there is no commit.
- **Standstill gating.** Blocks marked "standstill" (B3) are accepted only when every `|actual_qd|` is below the injected `STANDSTILL_EPS` and the active command is Idle or a position hold. Otherwise they are nacked with 2.
- **Ack or nack.** The script reports `config_ack_seq` and `config_result`:
  - 0 OK
  - 1 unknown block
  - 2 not allowed now (mode or motion)
  - 3 invalid values

  A nack never changes the active configuration.

*How the driver schedules config:*
- **What becomes config:**
  - the dexhal native setters: `set_payload_native`, `set_tcp_offset_native` and `set_gravity_native` go to block 1; `tare_wrench_native` sets command flag bit 0 [r7]
  - vendor commands: limits, detector parameters, test injection
  - *slow* `CoreCommand` fields: the reflex frame (block 2) and the admittance M/D/K and `task_frame` (block 4). The driver keeps a shadow copy of what it last sent and emits a config block only when these change (bitwise compare, NaN-safe). Because dexzero rewrites them every tick, that diff is what keeps them off the motion registers. The reflex thresholds are not in this group: they ride on every motion command. [r7]
  - The reflex action is not among them; it goes in its fixed register on every write.
- **Setters during motion.** The driver checks standstill against the latest state before queuing a standstill-gated setter, and returns an error at once if the arm is moving. The script's check is the backstop.
- **Order.** A config block a motion command depends on (new reflex frame, new admittance parameters) is sent *before* that command's first motion packet.
  - The first target of the new command is therefore one cycle late, which the kernel sees through the echo.
  - The reflex action for the new command is sent together with that config block, in its fixed register, so the action never runs against stale thresholds.
- **Rate cap.** While a motion mode is active, at most one config cycle in every two, so motion is never more than one cycle stale. Idle/stop always preempts queued config.
- **Typical latency.** [r6] Every change, including a full admittance change (gains plus frame), takes a single config cycle, about 2–4 ms.
- **Setter results.** A setter returns `Applied` once its block is queued. The ack is reported in `diag.config_ack_seq`. A nack of a kernel-originated block is a contract violation, so it raises `fault_code` 5. Vendor-command nacks are returned as errors.
- **Connect.** At connect, the full configuration is sent as config commands before the first motion command:
  1. payload, TCP and gravity
  2. force detection, with tare
  3. limits

  The reflex action is −1 from the start. Load-time injection is kept only for things that can never change at runtime (B4).

*Alternatives rejected:*
- **Multiple RTDE input recipes** don't add registers. A second packet in the same controller cycle overwrites the first before the script reads it.
- **Secondary-interface programs (port 30002)** interrupt the running main script.
- **Fixed per-function registers** would need about 70 doubles.

## A6. Constraints and risks

- **Script cycle budget is 2 ms.**
  - Watch IK, M-D-K and interpolation cost; count overruns in `diag.cycle_overrun`.
  - Hardware must show zero overruns with compliance active.
- **URSim has no usable force signal.** Add a test-only wrench-injection config block (D8), enabled by an injected constant that is false in production, so reflexes and admittance can be tested in simulation.
- **Register budget.** The upper range gives 24 input doubles and 24 input ints per cycle.
  - Doubles: motion commands use all 24, including the reflex thresholds in 42–47 [r7]. Config blocks use at most 24 (admittance uses exactly 24).
  - Ints: 8 are fixed (24–31) and 2 belong to the config header (32–33) (B3). [r7]
  - Any new per-tick field needs a spare motion int (32–47) or a fixed int. Anything slower goes into a config block. [r7]
- **Remote-control mode is required on e-series** for Dashboard power commands. `connect` checks it and fails clearly.
- **Minimum PolyScope version.** Pin a minimum PolyScope 5 version, chosen in testing, for the register range, the `rtde_set_watchdog` support and the RTDE recipe fields.
- **CB3.** It runs at 125 Hz, which needs `TIME_SLICE` injected. Support it or drop it explicitly.
- **One source for the register map.** The script constants and the Rust table come from one source, checked by a test.
- **The gripper script is hardware-tested.** Changes to it are limited to register renumbering, the int timestamp and calibration on request (B4). Verify them on hardware.

---

# Part B — Porting dexware

## B1. What dexware has

Sources (`C:\code\dexware\dexware\arm\ur`):
- **Python:** `connection.py` (RTDE, script upload), `arm\robot.py` (server loop, command and state), `arm\data.py`.
- **Scripts:** `realtime\{urcontrol, urcommon, rqcontrol, rqlib, main, no_op}.script`.
- **Register map:** `realtime\config.xml`.

How it runs:
- **Loop pacing.** robot.py's server loop is paced by the RTDE state stream. It writes a command every 2 ms, even when the command is unchanged. Goals change at up to about 90 Hz (spacemouse teleop, gated by the RealSense camera).
- **The script.** `urcontrol.script` recomputes `ur_movej`/`ur_movel`/`ur_speedj`/`ur_speedl` from the current state every cycle. These are proven interruptible tracking servos.
- **Admittance.** Velocity-from-force: a damper only, in base or TCP axes, with a radial deadband.
- **Contact and reflexes.** Contact detection and responses: yield, backoff, stop, disabled. Plus a force safety stop.
- **Gripper.** A Robotiq thread (`rqcontrol.script`) shares the input timestamp.
- **Cmd 40** is `speedl(0)` plus velocity-from-force. dexware never calls `force_mode()`.

## B2. Disposition

| dexware component | Disposition | Target |
|---|---|---|
| `connection.py` RTDE client | Rewrite in Rust | `src/ur/rtde` (C5 delivery, drop counting, watchdog) |
| Script upload, `no_op` handshake | Rewrite | `src/ur/script` (constant injection, contract-version check) |
| No Dashboard use | New | `src/ur/dashboard` (C8) |
| robot.py server loop | Replaced by the dexzero executor | The driver is a plain `ArmDriver` |
| robot.py command and state mapping | Rewrite | `src/ur/command`, `src/ur/state` (C1–C3, C7, C9) |
| `config.xml` register map | Revise into contract v2 | `src/ur/registers` plus generated script constants (B3) |
| `urcontrol.script` dispatch and servos | Keep the structure, modify | Feedforward (D5), Cartesian fixes (D3), watchdog key (D7), waypoint mode (D1) |
| `urcontrol.script` admittance | Keep as the interim law, then replace | M-D-K (D2) |
| `urcontrol.script` contact and reflexes | Rewrite the logic | Per-axis EMA, reflex frame, latch codes (C3, C4) |
| `urcontrol.script` safety stop | Modify | Report as code 2; no auto-resume |
| Payload-inertia hack, startup `zero_ftsensor()` | Drop | C4 |
| `urcommon.script` helpers | Keep | `joints_sub` wrap etc. |
| `rqcontrol.script` and `rqlib.script` | Keep, with small changes | Register renumbering, int timestamp, calibration on request (B4) |
| `main.script` | Keep | Assembled per variant (gripper thread only on gripper variants) |
| Python gripper conversions | Rewrite in the driver | C10 |
| `ArmLimits` per command (double 25–28 every cycle) | Move to config | Kernel limits, and script backstops through the limits config block (C6, D8) |
| Reflex and admittance fields per cycle (double 35–46, int 28–30) | Split | Reflex action, watchdog limit and tare/calibrate requests go to fixed ints; per-axis thresholds go to double 42–47 on motion commands; reflex frame and admittance go to config blocks sent on change (D8) [r7] |

## B3. Register contract v2

- **Versioned.** The script echoes the contract version.
- **Upper range only (24–47)**, multiplexed by `cmd_type` (D8). [r5]
- **Fixed ints grouped at the bottom of each range.** [r6] All fixed ints sit in one contiguous block (24–31), gripper included. [r7] The gripper outputs are compacted after the counts are removed. `rqcontrol.script` is updated to match (B4).
- **Integer timestamp.** [r6] `cmd_timestamp` and its echoes are wrapping µs integers (C1). That frees every input double for payload.

Inputs: fixed ints, valid in every packet

| Register | Contents |
|---|---|
| int 24 | `cmd_timestamp`, µs mod 2³¹ (C1) |
| int 25 | `cmd_type`: 0 stop, 20 speedj, 21 speedl, 30 joint servo, 31 Cartesian servo, 40 wrench (K = 0 admittance), 50 Cartesian admittance (D2), 90 config |
| int 26 | reflex action: −1 disabled, otherwise the dexhal action code (C3). Also acknowledges the latch. |
| int 27 | watchdog limit in cycles. A value ≤ 0 means the injected default (250 at 500 Hz) (D7). |
| int 28 | gripper target position (counts) |
| int 29 | gripper speed (counts) |
| int 30 | gripper force (counts) |
| int 31 | command flags [r7]: bit 0 tare FT, bit 1 calibrate gripper (gripper variants only). Each acts on a 0→1 change. The host sets the bit, waits for the echo (output int 40), then clears it. A request refused because the arm isn't at standstill sets config result 2 and isn't applied. |

Inputs: motion commands (`cmd_type` 0–50)

| Register | Contents |
|---|---|
| double 24–29 | `target_q`, or rotation-vector pose / twist |
| double 30–35 | `target_qd`, or twist feedforward |
| double 36–41 | `target_wrench` bias (modes 40 and 50; otherwise ignored) |
| double 42–47 | per-axis reflex thresholds (6) [r7]. A NaN component disables that axis (C3). Config cycles don't read them. |
| int 32–47 | spare |

Inputs: config command (`cmd_type` 90)

| Register | Contents |
|---|---|
| int 32 | block id |
| int 33 | sequence number (wraps; echoed in the ack) |
| double 24–47 | block payload (up to 24 values) |
| int 34–47 | spare |

Config blocks (each fits in one packet, so nothing is staged):

| Id | Block | Payload | Flags (int 33) | Allowed when |
|---|---|---|---|---|
| 1 | Payload, TCP, gravity | mass, CoG (3), inertia (6), TCP pose as a rotation vector (6), gravity direction (3) = 19 → `set_target_payload`, `set_tcp`, `set_gravity` | — | Standstill. A change of wrench compensation should be followed by a tare (command flag bit 0, [r7]). Until it is, the script reports `diag.tare_pending`. |
| 2 | Force detection | filter tau, contact force and torque thresholds, safety force and torque limits, reflex frame rotation (3) = 8 | — | Standstill. Tare is command flag bit 0, not a block field [r7] |
| 3 | Admittance | M (6), D (6), K (6), `task_frame` pose (6) = 24 | — | Any mode; applied with no jump, keeping the integrator state |
| 4 | Limits (backstop clamps) | joint speed (6), joint accel (6), linear speed and accel, rotational speed and accel = 16 | — | Any mode |
| 5 | Test wrench injection | wrench (6), added to `get_tcp_force()` until cleared | — | Only if `TEST_INJECTION` is true; otherwise nack 1 |

"Standstill" means every `|actual_qd|` is below `STANDSTILL_EPS` and the active command is Idle or a position hold; otherwise the script nacks with 2.

Outputs: ints

| Register | Contents |
|---|---|
| int 24 | `cmd_timestamp` echo (arm loop) |
| int 25 | stale count |
| int 26 | `native_reflex_fired` code |
| int 27 | applied `cmd_type` this cycle (90 on a config cycle) |
| int 28 | override steps |
| int 29 | override type |
| int 30 | cycle overruns |
| int 31 | contract version |
| int 32 | config acknowledgement sequence |
| int 33 | config result: 0 ok, 1 unknown block, 2 not allowed now, 3 invalid values |
| int 34 | flags: bit 0 `tare_pending` |
| int 35 | gripper `cmd_timestamp` echo (the gripper thread's applied command) |
| int 36 | gripper actual position |
| int 37 | gripper target position |
| int 38 | gripper object detected |
| int 39 | gripper in motion |
| int 40 | command flags echo [r7]: bit 0 tare applied, bit 1 calibration applied. Set when the script has acted on the request. Cleared when the host clears the request bit. |
| int 41–47 | spare |

Outputs: doubles

| Register | Contents |
|---|---|
| double 24 | contact duration |
| double 25 | reflex response duration (time since the latch, C3) |
| double 26–31 | executed command |
| double 32–37 | filtered wrench |
| double 38–47 | spare |

**Where gripper calibration lives.** [r7] The `open`/`closed` counts come from the manifest's gripper section. The script loads them as load-time constants, so the runtime neither stores nor sends them. Command flag bit 1 reruns the existing calibration routine on the gripper thread. The new counts apply for the session. To make a new calibration permanent, update the manifest.

## B4. Script changes

**`urcontrol.script`:**
- **Load-time constants** (static only): `TIME_SLICE`, `TEST_INJECTION`, `STANDSTILL_EPS`, the default watchdog limit, waypoint mode, the contract version, the gripper `open`/`closed` counts from the manifest [r7], and safe default clamps used until the first limits block arrives. Everything that can change at runtime goes through config blocks or the fixed ints instead (D8). [r6]
- **Integer timestamp** [r6]. Read `cmd_timestamp` from input int 24 and echo it to output int 24. Do the wrap-safe difference (C1). `no_op.script` resets the echo to −1.
- **Watchdog.** Key it on the change of `cmd_timestamp` (D7).
- **Joint servo (30).** Feedforward (D5); keep `joints_sub` wrap handling.
- **Cartesian servo (31).** Twist feedforward, `qnear = q`, joint wrap, a blend weight clamped to at most 1, and independent rotation limiting (D3).
- **Waypoint mode.** A one-segment Hermite buffer, selected by an injected constant (D1).
- **Wrench (40) and admittance (50).** Speed-control path only, never `force_mode()` (D2, D4).
- **Force pipeline.** Raw `get_tcp_force()` → per-axis EMA → rotate into the reflex frame (block 2) → per-axis threshold (double 42–47 of the motion command) → latched action (C3).
  - The radial 3 N / 0.5 Nm detector drives only `in_contact` and the contact duration.
- **Safety force limit.** Latch it as code 2 and remove the 50-step auto-resume.
- **Remove** the payload-inertia hack and the startup tare.
- **Config command (90).** Dispatch on block id; check standstill and mode gating; apply; write the ack and result. While handling it, continue the active motion mode on its last targets (D8). A second script is never uploaded. [r6]
- **Fixed ints on every cycle.** Apply the reflex action, the watchdog limit and the command flags whatever the command type. [r6, r7]
- **Command flags** [r7]. A 0→1 edge on bit 0 runs `zero_ftsensor()` at standstill, and clears `tare_pending`. A 0→1 edge on bit 1 starts the calibration routine on the gripper thread. Both are echoed in output int 40.
- **Count cycle overruns.**

**`rqcontrol.script`** (hardware-tested; changes kept small and mechanical) [r6]:
- **Registers.** Renumber them to B3: inputs int 28–30; outputs int 35–39. Read the shared timestamp from input int 24 instead of double 24, and echo it to output int 35 instead of output double 27.
- **Calibration on request.** Remove the top-level `if True:` calibration. The thread's `_open`/`_close` start from the manifest constants. When the script flags a request from command flag bit 1, the thread runs the existing calibration routine unchanged and keeps the new counts for the session. [r7]
- **Unchanged.** The polling loop, the single sync inside the Robotiq call, and object detection stay as they are. The socket calls block, so there is no busy loop. The known quirks are documented, not fixed.
- **Validate on hardware** before merging (WI-38).

## B5. dexware flaws not to port

1. **Connect retry doesn't retry.** `connect()` decrements `retries` once, sleeps, and carries on unconnected.
2. **E-stop can be refused.** `send_command(None)` raises "Must read states before sending" if a read is pending, so the stop isn't sent.
3. **The watchdog keys on a new goal timestamp**, not on link activity. A goal held for more than 0.5 s should stop the arm; confirm this on hardware. `Command.timestamp` defaults to 0 outside `robot.py.make_command`.
4. [Removed]
5. **Gripper calibration** runs at every connect and auto-reconnect and drops a held object. With no gripper attached it blocks the main script, and `connect` waits with no timeout.
6. **Unbounded recursive reconnect** reloads the script mid-task.
7. **Read-only mode and clock sync:**
   - Read-only mode leaves `_hw_clock_offset` as `None`, which crashes `make_state`.
   - The 4 ms clock-sync check can loop forever on a slow link.
   - `_read_raw` busy-spins on the host.
8. **`close()` leaves the script running** until its watchdog fires.
9. **`ur_movel` problems:**
   - no joint wrap
   - no `qnear` seed for the IK
   - rotation speed isn't limited when translation dominates
   - `w` can exceed 1
10. **Edge-triggered safety and contact responses.** Motion resumes into the obstacle after 50 steps, and the safety check uses the EMA, about 20 ms late.
11. **BACKOFF and STOP do nothing** when `response_duration` is 0, which is the default. BACKOFF only retraces 10 ms.
12. **The payload-inertia hack** uses mass only, no CoM, and finite differences, and depends on the controller version.
13. **Gain mismatch.** `BASE_ADMITTANCE_TORQUE` is 5 in the script but documented as 10.
14. **`get_capabilities` omits `JOINT_LINEAR`,** and `ur_movej` ignores admittance.
15. **Gripper reporting quirks** :
    - `in_motion` is never set for speed-0 commands
    - `grasp_force` is always 0
    - division by zero if `open == close`, on the Python side
16. **Smaller issues:**
    - `torques` actually holds amps
    - `cmd_length` is the stale count
    - `logging.basicConfig` runs at import
    - no recipe fallback for older controllers
    - the script cycle budget isn't monitored
17. **`TIME_SLICE` = 2 ms is hardcoded,** so the script is e-series only.
18. **The `cartesian_admittance` model mismatch.** dexware's law can't honor dexzero's Route 2 `CartesianAdmittance`:
    - It is a damper only, with no mass and no commanded stiffness.
    - Only base or TCP axes, not an arbitrary `task_frame`.
    - It has a deadband.
    - It has no `target_wrench`.

    So `native.cartesian_admittance` stays false until D2's M-D-K is in place.

## B6. Current state of `dexforge-ur`

- **Wrong model.** The model is the CB3 UR5 (`d1 = 0.089159`), while the 500 Hz manifest, the script and the README's URSim setup all mean a UR5e (`d1 = 0.1625`).
- **The manifest over-claims:**
  - `native.cartesian_admittance`
  - `payload`/`tcp_offset` while the methods return `NotSupported`
- **The manifest under-claims:**
  - wrench sensing
  - contact detection and reflexes
  - Cartesian twist and wrench modes
  - the hardware watchdog
- **Wrong or missing values:**
  - `has_static_inertials: true` contradicts dexzero's docs for UR
  - `sensing.torque` with no `torque_constants`
  - the latency is likely understated; measure it
- **Hand-written `NOMINAL_DH`** in `calibration.rs` duplicates the geometry in `arm.rs`.
- **Missing implementation:** the driver is a stub, and there is no transport, no plugin and no `cdylib`.
- **Verified, no issue:** the base and flange frames match the controller, and `calibration.rs` proves DH equals FK.

---

# Part C — Work items

Phases:

1. **Phase 1 (minimum viable).** Transport, lifecycle, joint modes in dense mode, state, watchdog, faults, plugin, UR5e. The only contract prerequisite is WI-1.
2. **Phase 2.** Native contact and reflexes, native tare and payload, the gripper, frame catch-up.
3. **Phase 3.** Cartesian modes, wrench mode, native M-D-K admittance.
4. **Phase 4.** Waypoint mode, more models, state pacing.

## dexhal contract (`C:\code\dexforge\dexhal`)

**WI-1: Command timestamp and echo** (C1, phase 1).
- Add `CoreCommand.cmd_timestamp` and `CoreState.cmd_timestamp_echo`, with the NaN rules.
- Update the wire layout, views and mock, and bump the version.

**WI-2: Reflex semantics** (C3, phase 2).
- Document NaN as disabled.
- Add `reflex_frame[7]`, or define reuse of `task_frame`.
- Define the `native_reflex_fired` codes and the latch, acknowledge and re-arm rules.

**WI-3: State delivery** (C5, phase 2).
- Define the catch-up ring: an aux block of skipped samples (timestamp, `q`, `qd`, wrench), bounded, and drained on each read.
- Add the optional `latency.state_paced` flag, with a blocking-read deadline and a timeout error.
- Document `diag.dropped_samples`.

**WI-4: Standard fault codes** (C7, phase 1). Adopt the dexzero `driver_fault` set; each driver documents its native-to-standard mapping.

**WI-5: Native tare and filter constant** (C4, phase 2).
- Add `tare_wrench_native()` with the `Applied` pattern.
- Add manifest `compliance.native_filter_tau`.

**WI-6: `CoreState.speed_scaling`** (C2, phase 1).

**WI-7: Diagnostics block** (C9, phase 1). Define the standard `diag.*` aux names.

**WI-8: Gripper semantics** (C10, phase 2): SI units, NaN means hold, measured versus target force, open and close endpoints.

**WI-9: Lifecycle wording** (C8, phase 1): `connect` is idempotent, `disconnect` leaves a safe hold, and the manifest adds `lifecycle.disconnect_power_off`.

**WI-10: Multi-model plugins** (C12, phase 1). Decide between one cdylib per variant and `create_math(config)`.

## dexzero (`C:\code\dexzero`)

**WI-11: Use the command echo** (phase 1).
- Measure the round trip from `cmd_timestamp_echo` when it is finite.
- Report measured against manifest latency.

**WI-12: Ingest catch-up samples** (phase 2). Merge the WI-3 ring into the k-deep history in timestamp order, and count overflows as overruns.

**WI-13: Optional state-paced ticking** (phase 4).
- When a driver declares `state_paced` and the config enables it, the arm's executor ticks on the return of `read_state`, and a timeout counts as comm lost.
- §6 multi-arm arbitration aligns ticks by `state.timestamp` (D6).

**WI-14: Reflex threshold fix** (phase 2).
- Stop rotating and `abs`-ing threshold boxes ([executor.rs](C:/code/dexzero/kernel/src/executor.rs:2332)); this also stops NaN axes turning into 0.
- Send thresholds in the command frame with `reflex_frame`.

**WI-15: Reflex reporting** (phase 2).
- Raise `CONTRACT_BROKEN` only for code 1 without a valid wrench, never for code 2.
- Rely on the abort for the acknowledgement.

**WI-16: Single tare authority** (phase 2).
- Use `tare_wrench_native()` when tare is applied natively.
- Require native payload when native reflexes are on.
- Use `native_filter_tau` in the `Conditioner` when native reflexes are active.

**WI-17: Speed scaling and session limits** (phase 2).
- Time-scale the reference or fault when `speed_scaling < 1`.
- Add sticky session-level limit tightening.

**WI-18: Waypoint streaming** (phase 4).
- A per-arm option to stream time-stamped waypoints at the manifest `control_freq_hz` instead of the 500 Hz dense reference.
- `cmd_timestamp` is the time the reference applies (D1).

**WI-19: Docs** (phase 2). Fix the [compliance.md](C:/code/dexzero/docs/specs/compliance.md) §8 line that says no field reports a native reflex. Document:
- the reflex frame
- the acknowledgement
- NaN as disabled
- catch-up ingestion

**WI-20: UR arm configuration** (phase 1). Add `[[arm]]` params (IP, model, end effector) following WI-10, and an integration test against URSim.

## dexforge-ur (`C:\code\dexforge\repos\dexforge-ur`)

**WI-21: Refresh the URDF and regenerate** (C11, phase 1).
- **Import from UR:** take `default_kinematics.yaml`, `physical_parameters.yaml`, `joint_limits.yaml` and `ur.urdf.xacro` from UR's official ROS2 description repo, and generate a URDF per model with xacro.
- **Regenerate per model** with `C:\code\dexforge\codegen\scaffold.py` and `ssik build <urdf>`: `generated_kinematics.rs`, `generated_dynamics.rs`, `ik_solver.rs`, and the test data.
- **Generate `NOMINAL_DH`** from `default_kinematics.yaml`, and keep the DH-vs-FK test as the guard.
- **Re-run** the DH-vs-FK, dynamics parity, IK and performance tests.
- **Models:** UR5e first, then UR3e, UR10e, UR16e, UR20 and UR30 as needed. Decide whether CB3 UR5 is kept or dropped.

**WI-22: Multi-model restructure** (A2, phase 1).
- Create `src/ur/` and the generic `UrDriver<M: UrModel>`.
- Strip `drivers/<model>/` down to model data.
- Add per-variant manifests and plugins, and per-model re-exports in `lib.rs`.

**WI-23: Register contract v2** (B3, D8, phase 1, extended in phases 2–3).
- A versioned table in `src/ur/registers`, upper range 24–47 only: the fixed, motion and config layouts, and the config-block table.
- Script constants generated from the same table, including block ids and payload offsets.
- A test that no layout exceeds 24 doubles or 24 ints, and that the fixed registers never overlap the multiplexed ones.

**WI-24: RTDE client** (phase 1).
- **Protocol:** handshake and version check.
- **Recipes:** 500 Hz output (125 Hz on CB3) and the input recipe; fail on missing fields; per-version recipes.
- **Reading:** a non-allocating receive path; the catch-up ring (WI-3), filled in phase 2; a blocking read with a deadline (phase 4).
- **Watchdog:** `rtde_set_watchdog` where available.

**WI-25: Dashboard client** (C8, phase 1).
- **Commands:** remote-control check, `robotmode`, `safetystatus`, `power on`, `brake release`, `unlock protective stop`, `close safety popup`, `stop`, `power off`.
- **Behavior:** every call has a timeout and returns a typed error.

**WI-26: Script loader** (phase 1).
- Load `no_op`, then the assembled `main.script` per variant.
- Inject the static constants only: `TIME_SLICE`, default clamps, default watchdog limit, `STANDSTILL_EPS`, `TEST_INJECTION`, waypoint mode, contract version, gripper `open`/`closed` counts (from the manifest, [r7]).
- Verify the contract version from the echo.

**WI-27: `urcontrol.script` phase 1 port** (B4).
- Watchdog on `cmd_timestamp`.
- Joint servo with feedforward (D5).
- `speedj`/`speedl`/stop as they are.
- The echo and override diagnostics, and cycle-overrun counting.
- Integer µs timestamp and echo, applied wrap-safe (C1); the fixed reflex-action, watchdog-limit and command-flag ints applied on every cycle.
- Config command dispatch (90), with blocks 1, 2 and 5: standstill and mode gating, ack and result, `tare_pending`, and continued motion during a config cycle (D8).
- Remove the startup tare and the payload-inertia hack.

**WI-28: Lifecycle** (C8, phase 1).
- **`connect`:**
  - Remote-control check; power on and brake release only if needed, then wait for `RUNNING`; refuse if the safety mode isn't normal.
  - RTDE setup and script load.
  - Send the initial config blocks and wait for each ack: block 1 (payload, TCP, gravity), block 2 (force detection and reflex frame), and block 5 (limits). No tare at connect (C4). Keep the reflex action at −1; arm the watchdogs.
  - Wait for the first echo. Every step has a timeout.
- **`disconnect`:** Idle, wait for standstill, stop the script, close RTDE; power off only per the manifest.
- **`clear_faults`:** unlock the protective stop, close the popup, reload the script if it stopped, re-arm, and wait for the echo.
- No gripper motion and no automatic reconnect.

**WI-29: `write_command`** (phase 1, with modes extended in phase 3).
- **Mode dispatch:**

  | ActiveMode | Script command | Phase |
  |---|---|---|
  | Idle | 0 | 1 |
  | `JointPos` | 30 | 1 |
  | `JointVel` | 20 | 1 |
  | `CartesianPose` | 31 | 3 |
  | `CartesianTwist` | 21 | 3 |
  | `CartesianWrench` | 40 | 3 |
  | `CartesianAdmittance` | 50 | 3, after WI-34 passes |
  | anything else | `NotSupported` | |

- **Every write (motion or config) sends the fixed ints:** the µs `cmd_timestamp`, `cmd_type`, the reflex action, the watchdog limit, the gripper ints and the command flags. A motion write also sends the target, feedforward, wrench bias and reflex thresholds. The I/O goes in the RTDE standard fields.
- **Config scheduling** (D8):
  - Keep a shadow of the slow fields (reflex frame → block 2; admittance M/D/K and `task_frame` → block 4) and queue a config block when one changes. The thresholds aren't shadowed; they go on every motion write. [r7]
  - Send dependent blocks before the command's first motion packet.
  - Cap config to one cycle in every two during motion; Idle preempts.
  - Drain the queue of setter and vendor-command blocks.
- **Constraints:** non-allocating, including the queue (fixed capacity); one RTDE input packet per call.

**WI-30: `read_state`** (phase 1).
- **Motion and force:**
  - `q`/`qd` from `actual_q`/`actual_qd`.
  - `tcp_pose` as a quaternion; `tcp_twist`.
  - `wrench` is the raw `actual_TCP_force` (base frame, gravity-compensated).
  - `torque` is NaN unless current-to-torque constants are declared.
- **Gripper and I/O:** gripper width and object; the I/O fields.
- **Contract fields:**
  - `in_contact` and `native_reflex_fired`
  - `timestamp` (RTDE), `cmd_timestamp_echo`, `speed_scaling`
  - `robot_mode`, `fault_code` (C7)
- **Diagnostics:** the `diag.*` block.

**WI-31: Native contact and reflexes** (C3, C4, phase 2).
- The script's force pipeline, the reflex frame and the latch codes. Thresholds come from double 42–47 on every motion command, the frame from block 2, and the action from fixed int 26 on every cycle. [r7]
- The safety limit as code 2 with no auto-resume.
- Test-only wrench injection (block 7).
- Declare `native.contact_detection`, `native.reflexes` and `compliance.native_filter_tau`.

**WI-32: Configuration plane** (D8, phase 2).
- **Native setters:**
  - Payload, TCP and gravity share block 1. The driver merges each setter with the shadow copy of the other two, and refuses with an error when the arm isn't at standstill.
  - Tare is command flag bit 0 (B3), with the same standstill gate. [r7]
  - Results are reported in `diag.config_ack_seq`, and a nack raises `fault_code` 5.
- **Limits:** session-level limits map to block 5.
- **TCP sync:** keep PolyScope's TCP equal to the kernel TCP.
- **Vendor commands:** Dashboard passthrough, gripper calibrate (command flag bit 1), read `calibration.conf` (feeds `set_calibration`), contract version, force-detection parameters (block 2), test injection (block 7). These wait for the ack and return a nack as an error.

**WI-33: Gripper** (C10, B4, phase 2).
- **Script:** the `rqcontrol.script` changes from B4: renumbered registers, int timestamp echo, calibration on request. Validate on hardware.
- **Driver:**
  - SI-to-count conversion with calibrated endpoints; NaN hold; reject while uncalibrated.
  - The counts come from the manifest. Nothing is persisted or restored at runtime. [r7]
  - A presence check at connect.

**WI-34: Cartesian servo and native M-D-K admittance** (D2, D3, D4, phase 3).
- **Cartesian servo (31):** with the D3 fixes.
- **Wrench (40):** K = 0 admittance with a wrench bias.
- **Admittance (50):** the M-D-K integrator in `task_frame`. Gains and frame come together in block 4, applied in one cycle with no jump. Speed path only.
- Declare `native.cartesian_admittance: true` only after it passes dexzero's latency-injection and stability tests, and the hardware shows no cycle overruns.

**WI-35: Waypoint interpolation mode** (D1, phase 4). Script-side Hermite interpolation of time-stamped waypoints with a one-waypoint buffer. The manifest variant declares its `control_freq_hz`.

**WI-36: Manifests per variant** (phase 1, updated per phase).

| Section | Settings |
|---|---|
| `control_modes` | as WI-29 per phase; `realtime: true`; `native.realtime_mode_switch: true` |
| `native` | `cartesian_admittance` false until WI-34; `contact_detection`/`reflexes` from phase 2; `payload`/`tcp_offset`/`gravity` true once WI-32 lands |
| `sensing` | `wrench: true`, `wrench_frame: base`, `wrench_is_gravity_compensated: true`; `torque` per WI-30; `gripper` per variant |
| inertials | `has_static_inertials: false`, `has_residual_inertials: true` |
| `latency` | measured `staleness`/`dispatch`; `hardware_watchdog_ms: 500`; `state_paced` from phase 4 |
| `compliance` | `max_wrench`, `native_filter_tau` |
| `gripper` | `max_width`, `max_force`, conversion constants per gripper model, `open` and `closed` counts [r7] |
| `limits` | joint and `cartesian_*` |
| `lifecycle` | `disconnect_power_off: false` |
| `aux` | `diag.*`, catch-up ring |

**WI-37: Plugin export** (C12, phase 1).
- `crate-type = ["rlib", "cdylib"]` and a `dexhal-plugin` dependency.
- A `plugin.rs` per variant, modelled on [the mock's](C:/code/dexforge/dexhal/mock/rust/src/plugin.rs).
- `create(config)` parses IP and the startup payload/TCP. The gripper counts come from the manifest. [r7]
- Keep `forbid(unsafe_code)`.

**WI-38: Tests** (every phase).
- **Offline:**
  - the register table matches the generated script constants, and the register budget isn't exceeded
  - unit conversions
  - the quaternion ↔ rotation-vector round trip
  - the catch-up ring ordering
  - config scheduling: shadow diffing including NaN, dependent-block ordering, the rate cap, Idle preemption
  - µs timestamp wrap: the encoding, ring reconstruction of the echo across the 2³¹ wrap, and the script's wrap-safe difference
- **Against URSim e-series:**
  - connect, disconnect and reconnect with no gripper motion
  - each mode tracks its stream with lag at most the declared latency
  - a lost link stops the arm through the watchdog
  - reflex codes, acknowledgement and re-arm, using wrench injection
  - each config block: ack, the nack codes (including standstill rejection), `tare_pending` after block 1, motion continuity across config cycles (no visible step in `actual_q`)
  - command flags: tare and calibration request and echo, refusal when not at standstill, and no re-trigger while the bit stays set [r7]
  - reflex thresholds on motion commands take effect in the same cycle, and hold their values through config cycles [r7]
  - the reflex action and its acknowledgement work during config cycles
  - protective stop → `fault_code` → `clear_faults`
  - the echo-measured latency matches the manifest
  - no dropped samples over 10 minutes
- **On hardware:**
  - gripper behavior unchanged after the `rqcontrol.script` changes: commands, object detection, timestamp echo, counts loaded from the manifest, calibration on request (command flag bit 1)
  - force thresholds and the timing of the native-versus-kernel trigger
  - admittance stability and the cycle-overrun budget
  - a held goal for more than 0.5 s, against the dexware watchdog behavior (B5.3)

## Renumbering from r3

| r3 | r4 |
|---|---|
| WI-1 … WI-10 | WI-1 … WI-10 (WI-3 is redefined as catch-up plus optional state pacing) |
| WI-11 | WI-11 |
| WI-12 (state-paced executor) | WI-12 (catch-up ingestion) + WI-13 (optional, phase 4) |
| WI-13 … WI-15 | WI-14 … WI-16 |
| WI-16 | WI-19 |
| WI-17 | WI-17 |
| WI-18 | WI-20 |
| — | WI-18 (waypoint streaming, new) |
| WI-19, WI-20 | WI-21, WI-22 |
| WI-21, WI-22, WI-23 | WI-24, WI-25, WI-23 |
| WI-24 (script port) | WI-26 (loader) + WI-27 (phase 1 port) + WI-31 / WI-34 / WI-35 |
| WI-25 | WI-33 |
| WI-26, WI-27, WI-28 | WI-28, WI-29, WI-30 |
| WI-29, WI-30, WI-31 | WI-31, WI-33, WI-32 |
| WI-32, WI-33, WI-34 | WI-36, WI-37, WI-38 |
| WI-35 (optional, phase 4) | WI-34 (core, phase 3) |
