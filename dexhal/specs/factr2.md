# Engineering Spec: FACTR 2 (NEXT) Virtual Torque Sensor Integration

## 1. Objective & Architecture Summary

Implement **NEXT (Neural External Torque Estimation)** from FACTR 2 (<https://jasonjzliu.com/factr2/>, [arXiv:2606.12406](https://arxiv.org/abs/2606.12406)) as a `#![no_std]`, zero-allocation virtual torque sensor inside **`dexhal`**, loaded by the **driver** at construction and consumed transparently by **`dexzero`** via Compliance Route 6 (`HostAdmittanceTorque`) with **zero `dexzero` config or IPC changes**.

* **Offline (`dexforge` / Python):** Collect ~10 minutes of free-space trajectories via `dexzero`, train a compact recurrent model (LSTM) in `float64` to predict free-space link torque $\hat{\tau}_f$ (or unmodeled friction / actuator residual $\hat{\tau}_{\text{fric}}$), and export flat binary weights (`.dexnext`).
* **Driver & HAL (`dexhal` + `dexhal-plugin`):** The OEM driver owns loading its `.dexnext` weights at construction (`PluginDriver::create(config)` — either compiled-in nominal weights via `include_bytes!` or a per-robot file path from the driver's own `config` blob) into `dexhal::math::Arm` and reports `has_residual_dynamics() == true` through `ArmManifest`. On each control tick, `ArmDriver::step_free_torque` advances the observer and applies differential runtime payload compensation to output predicted free-space link torque $\hat{\tau}_{\text{free}}$.
* **Kernel (`dexzero`):** Sees an ordinary Route 6 arm (`manifest.has_torque_sensing && manifest.has_residual_dynamics()`), calls `driver.step_free_torque(...)` instead of stateless `driver.inverse_dynamics(...)` in the Route 6 observer path, subtracts `manifest.link_torques(state.torques())` ($r = \hat{\tau}_{\text{free}} - \tau_{\text{measured}}$), projects through `jacobian_pseudo_inverse(q_actual)` to $\hat{F}_{\text{ext}}$, and feeds Route 6 admittance and contact reflexes (`STOP`, `YIELD`, `RETRACT`).

---

## 2. Network Structure & Binary Weight Format (`.dexnext`)

### 2.1 Paper Alignment (fixed architecture)

The architecture and training setup are fixed to the configuration FACTR 2 selects in its ablation (arXiv:2606.12406 Table 3 and Appendix A.2) and uses for both its Franka and AgileX Piper experiments. Nothing is configurable in the file except `dof`, the operating mode and the training-time gravity / payload:

1. **Input $(q, \dot{q}, \Delta q_d)$ (`3 * DOF`):** the ablation (MLP / GRU / LSTM × inputs $\{q,\ (q, \dot{q}),\ (q, \dot{q}, \Delta q_d)\}$ × history $\{10, 25, 50\}$) selects the joint command error $\Delta q_d = q_{\text{cmd}} - q$ alongside $(q, \dot{q})$: contact $L_1$ error drops from $1.12$ to $0.547\text{ N}\cdot\text{m}$ ($0.414$ in free space) without differentiating $\ddot{q}$. On the AgileX Piper the free-space $L_1$ error is $0.018\text{ N}\cdot\text{m}$ vs. $0.435$ for URDF inverse dynamics. $\Delta q_d = 0$ when no command is given.
2. **Stateless window, $W = 50$ at $100\text{ Hz}$ (0.5 s):** every sample the LSTM re-runs from $(h, c) = (0, 0)$ over the newest 50 samples, so sustained contact or out-of-distribution states cannot corrupt a carried cell state. After a reset the window is left-padded with the first sample (training pads identically).
3. **Backbone & head:** 2-layer LSTM, $H = 128$ (dropout 0.1 in training), 2-layer MLP head $128 \to 256 \to \text{DOF}$ with ReLU. The paper does not ablate $H$ or the head width; these are its Table 3 values.
4. **Control-loop budget (500–1000 Hz):** a window is 50 LSTM frames ($\approx 1.76\text{ ms}$ in the paper, $\approx 1.5\text{ ms}$ in `dexhal` on one desktop core), more than a 1 kHz cycle. `dexhal` keeps the result exact but **spreads the work** over the ticks of each 100 Hz period: the first 49 frames of the next window only use samples that are already known, so they are advanced a few per tick, and the sample tick runs just the final frame and the head (≈ 6 frames, ≈ 0.2 ms at 1 kHz). At a 100 Hz control rate the whole window runs every tick (≈ 1.5 ms of a 10 ms cycle).

### 2.2 Topology & Math (`f64` precision throughout)

* **Input normalization (element-wise z-score):** $u_t = [q, \dot{q}, \Delta q_d]$, $x_t = (u_t - \mu_{\text{in}}) / \sigma_{\text{in}}$.

* **Recurrent core (2 standard LSTM layers, $H = 128$, gate order `[i, f, g, o]` matching PyTorch `nn.LSTM`):**
  For layer $\ell \in \{0, 1\}$ with layer input $z_t^{(0)} = x_t$ and $z_t^{(1)} = h_t^{(0)}$:

$$\begin{aligned}
[i_t, f_t, g_t, o_t]^T &= W_{ih}^{(\ell)} z_t^{(\ell)} + b_{\text{gates}}^{(\ell)} + W_{hh}^{(\ell)} h_{t-1}^{(\ell)} \\
c_t^{(\ell)} &= \sigma(f_t) \odot c_{t-1}^{(\ell)} + \sigma(i_t) \odot \tanh(g_t) \\
h_t^{(\ell)} &= \sigma(o_t) \odot \tanh(c_t^{(\ell)})
\end{aligned}$$

  *(where $b_{\text{gates}}^{(\ell)} = b_{ih}^{(\ell)} + b_{hh}^{(\ell)}$ is pre-summed in `f64` on export).*

* **Head and denormalization** on the last frame of the window: $m = \max(0, W_{\text{mlp}} h^{(1)} + b_{\text{mlp}})$ ($256$ wide), $y = W_{\text{out}} m + b_{\text{out}}$,

$$\hat{\tau}_{\text{nn}} = \sigma_{\text{out}} \odot y + \mu_{\text{out}} \in \mathbb{R}^{\text{DOF}}$$

* **Operating modes (`ObserverMode`)** — the same network; the mode only changes the training target and what the runtime adds back:
  * `ObserverMode::FullDynamics` (`0`, the paper): $\hat{\tau}_{\text{nn}}$ predicts total free-space link torque $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) + \tau_{\text{fric}}$. Does **not** require `ArmModel::HAS_DYNAMICS`, provided the arm stays mounted with the base-frame gravity vector $\mathbf{g}$ used during training.
  * `ObserverMode::FrictionResidual` (`1`, a `dexhal` extension): $\hat{\tau}_{\text{nn}}$ predicts only the residual $\tau_{\text{measured}} - \tau_{\text{rigid}}(q, \dot{q}, \ddot{q}, \mathbf{g})$. Requires `ArmModel::HAS_DYNAMICS`; adapts to base-frame gravity changes (wall/ceiling mounts) and full-rate $\ddot{q}_{\text{ref}}$ changes between network samples.

### 2.3 Sign Convention & Units Alignment with `dexhal`

Per [`CoreState::torques`](../src/math/contract/wire.rs) and `dexzero`'s `compliance.md` §6:

1. **Units (Link-Side N·m):** `CoreState::torques` carries raw motor current/effort when `manifest.torque_constants` (`sensing.torque_constants` in `manifest.yaml`) is finite, and link-side N·m otherwise. Both data collection and runtime inference **must** operate on link-side torque $\tau_{\text{measured}} = \text{manifest.link\_torques}(\text{state.torques()})$ in N·m.
2. **Sign Convention:** `CoreState::torques` is the link-side joint effort applied *by the actuator* in the positive joint direction.
   * FACTR 2 Eq. (2) writes $\hat{\tau}_{\text{ext}}^{\text{paper}} = \tau_m - \hat{\tau}_f$ (additional motor effort required to oppose contact, i.e. $-J^T F_{\text{env}\to\text{tool}}$), whereas `dexzero` Route 6 (`compliance.md` §6) defines the residual as:

$$r = \hat{\tau}_{\text{free}} - \tau_{\text{measured}}, \qquad \hat{F}_{\text{ext}} = (J^T)^+ r$$

   so that $\hat{F}_{\text{ext}}$ is the wrench the **environment applies onto the tool** (matching `CoreState::wrench` in Route 5). In free space ($\tau_{\text{ext}} = 0 \implies \tau_{\text{measured}} = \tau_{\text{free}}$), the training target is identical; at runtime, `dexzero` must keep $r = \hat{\tau}_{\text{free}} - \tau_{\text{measured}}$.

### 2.4 Packed Binary Format (`.dexnext`, version 2)

Little-endian byte stream, parsed via `from_le_bytes` (no `unsafe` casts, so arbitrary `&[u8]` buffers from `include_bytes!` or a file work under `#![forbid(unsafe_code)]`). The weights are read in place from the `&'static [u8]`; nothing is copied.

1. **Header (144 bytes):**
   * `0..8` `magic` = `b"DEXNEXT\x02"`
   * `8` `dof: u8` (must equal the driver's `DOF`)
   * `9` `mode: u8` (`0 = FullDynamics`, `1 = FrictionResidual`)
   * `10..16` reserved, zero
   * `16..40` `gravity: [f64; 3]` (base-frame gravity in m/s² during training; enforced in `FullDynamics` mode)
   * `40..48` `trained_payload_mass: f64` (kg; `0.0` for a bare flange, or the tool mounted during collection and NOT subtracted from the targets)
   * `48..72` `trained_payload_com: [f64; 3]` (flange frame, m)
   * `72..144` `trained_payload_inertia: [f64; 9]` (row-major 3×3 about the CoM in flange axes, kg·m²)
2. **Parameters** (contiguous `f64`; let $D = 3 \cdot \text{dof}$, $G = 512$). Every matrix is stored **transposed** (input-major: for each input column, all output rows), so the runtime accumulates `acc[r] += Wᵀ[k][r] · x[k]` over a fixed row count, which vectorizes without reassociation:
   * `in_mean[D]`, `in_std[D]` (`in_std > 1e-12`)
   * `W_ih0ᵀ[D][G]`, `W_hh0ᵀ[128][G]`, `b0[G]`
   * `W_ih1ᵀ[128][G]`, `W_hh1ᵀ[128][G]`, `b1[G]`
   * `W_mlpᵀ[128][256]`, `b_mlp[256]`, `W_outᵀ[256][dof]`, `b_out[dof]`
   * `out_mean[dof]`, `out_std[dof]` (`out_std > 0`)

   $241{,}670$ parameters (≈ 1.9 MB) for 6 joints (`next_param_count`). The exact length follows from `dof`; trailing or truncated bytes, non-finite values, non-zero reserved bytes and invalid payloads are rejected.

---

## 3. PyTorch Data Collection & Training (`dexforge`)

### 3.1 Data Collection (`dexzero`)

Collection runs against a live kernel and belongs to `dexzero`; `dexforge` only consumes its log (schema in `codegen/next/train_next.py`).

1. Connect to a running `dexzero` kernel via `dexzero.connect(arm_name)`.
2. Verify preconditions and record training-time tool/payload state:
   * `get_capability_manifest()["has_torque_sensing"] == True` (from `ArmManifest::has_torque_sensing`, `sensing.torque` in `manifest.yaml`).
   * `get_configuration()["freedrive"] == False` and workspace unobstructed (no external contact).
   * Record `manifest = get_capability_manifest()` (`control_freq_hz`, `torque_constants`, `dof`), `get_gravity()`, and `get_payload()` (`mass`, `com`, `inertia` of whatever end-effector / gripper is physically mounted during collection, e.g. `Payload::NONE` for bare flange or the empty gripper's mass/CoM/inertia).
3. Command ~10 minutes of coverage-oriented free-space trajectories (`move(..., interpolation='joint')` and Cartesian workspace sweeps, following FACTR 2 Appendix A.3):
   * Phase 1: Single-joint sweeps across each joint's limit range at slow and fast speeds.
   * Phase 2: Multi-joint coupled sinusoidal / Cartesian trajectories covering diverse postures and velocities.
4. Stream full-rate samples directly from the `state` ZeroChannel ring (or `McapRecorder`, avoiding slow Python sleep-loop polling on `read_state()` that drops or aliases ticks):
   * Log `(tick, timestamp, q, qd, q_cmd, qdd_ref, tau_link)` using latency-compensated `state.q`, `state.qd`, commanded `state.q_cmd`, reference `state.qdd_ref`, and scaled link torque $\tau_{\text{link}} = \text{manifest.link\_torques}(\text{state.torques})$.
   * For `FrictionResidual` training, also log `tau_model` $= \tau_{\text{rigid}} + \tau_{\text{payload}}(P_{\text{train}})$ from the driver's own `ArmMath::inverse_dynamics(q, qd, qdd_ref)` with the collection payload configured, so the training target and the runtime `flange_inverse_dynamics` correction come from the same model (no URDF / external dynamics library in training).
   * Split into contiguous fault-free / contact-free segments, discarding frames (with a safety margin) where `kernel_fault != 0`, `driver_fault != 0`, or `in_contact == 1.0`.

### 3.2 Training & Export (`codegen/next/train_next.py`)

```python
class NextObserverNet(torch.nn.Module):
    def __init__(self, dof: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(3 * dof, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, dof)
        )

    def forward(self, x):  # B × 50 × 3·dof normalised windows
        out, _ = self.lstm(x)
        return self.head(out[:, -1])
```

* **Resampling:** downsample each contiguous segment to 100 Hz with the integer stride $k = \text{round}(\text{control\_freq\_hz} / 100)$ (the runtime `stride_ticks`) and build $u = [q, \dot{q}, q_{\text{cmd}} - q]$.
* **Supervision target & training-time payload:**
  * `FullDynamics` (`--mode full`, default): $\tau_{\text{target}} = \tau_{\text{link}}$ with the mounted tool in place; `trained_payload = P_train` is recorded so the runtime adds the exact differential $\tau_{\text{payload}}(P_{\text{runtime}}) - \tau_{\text{payload}}(P_{\text{train}})$.
  * `FrictionResidual` (`--mode residual`): $\tau_{\text{target}} = \tau_{\text{link}} - \tau_{\text{model}}$, where $\tau_{\text{model}}$ is the logged driver inverse dynamics including $P_{\text{train}}$ (§3.1); `trained_payload = Payload::NONE`.
* **Training (FACTR 2 Table 3):** every length-50 window (left-padded with the segment's first sample, like the runtime), zero initial state, MSE on the normalized torque at the last step, AdamW (`lr = 1e-3`, `weight_decay = 1e-6`), gradient clip `1.0`, batch 256, early stopping on validation loss (patience 20, warm-up 10 epochs, min Δ `1e-5`). Dropout placement in the head is not specified by the paper; it is applied after the ReLU.
* **Export:** cast to `float64`, fold `b_ih + b_hh`, check PyTorch `float64` (`model.eval()`) against the NumPy reference of the runtime window (`dexnext.window_outputs`) on held-out data (`max_abs_err < 1e-10`), and write `.dexnext`.

---

## 4. Rust `#![no_std]` Implementation (`dexhal` & `dexhal-plugin`)

### 4.1 New Module: `dexhal::math::torque_observer`

Constraints: `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap allocation, bounded loops, `libm::{exp, tanh}`. `Arm<A, DOF, MAX>` keeps its three parameters; the dimensions are crate constants:

```rust
pub const NEXT_LAYERS: usize = 2;
pub const NEXT_HIDDEN: usize = 128;
pub const NEXT_GATES: usize = 4 * NEXT_HIDDEN;
pub const NEXT_MLP: usize = 256;
pub const NEXT_WINDOW: usize = 50;
pub const NEXT_SAMPLE_RATE_HZ: u32 = 100;
pub const NEXT_MAGIC: [u8; 8] = *b"DEXNEXT\x02";
pub const NEXT_HEADER_LEN: usize = 144;
pub const fn next_param_count(dof: usize) -> usize;
pub const fn next_file_len(dof: usize) -> usize;

pub enum ObserverMode { FullDynamics = 0, FrictionResidual = 1 }

pub struct LstmObserver<const DOF: usize> {
    // &'static weight bytes + decoded normalisation, mode, trained gravity / payload
    // ring of the newest 50 normalised samples ([[[f64; DOF]; 3]; 50])
    // LSTM state of the next window's prefix and how many prefix frames it holds
    // tick within the stride period, warm-up flag, held output
}
```

About 12 KiB of state for six joints; the 1.9 MB of weights stay in the file bytes. `ModelError::InvalidObserver` (in [`ModelError`](../src/math/chain.rs), mapped in `dexhal-plugin`'s error codes) reports every rejection.

* **`from_bytes(bytes: &'static [u8]) -> Result<Self, ModelError>`:** validates the magic, `dof == DOF`, mode, zero reserved bytes, exact length, finiteness, `in_std > 1e-12`, `out_std > 0` and the trained payload.
* **`stride_ticks(control_freq_hz) -> usize`:** `round(control_freq_hz / 100)`, at least 1.
* **`step(&mut self, u: &[[f64; DOF]; 3], stride: usize) -> Option<[f64; DOF]>`:** one control tick with $u = [q, \dot{q}, q_{\text{cmd}} - q]$. Every `stride` ticks the input is sampled into the ring and a window is evaluated; the output is held in between. Scheduling (bit-identical to evaluating the whole window at the sample tick):
  * *Sample tick:* finish any remaining prefix frames, push the sample, run the final frame on the prefix state, evaluate the head, then restart the prefix from zero.
  * *Every tick:* advance `ceil(remaining / ticks_left)` frames of the next window's prefix (the 49 newest samples, oldest first), so the prefix is complete by the next sample tick.
  * *After a reset:* the first sample fills the ring; its 49-frame prefix and final frame are spread the same way and the first output appears on tick `stride − 1` (`None` before). The second window shares that padded prefix.
  * `None` also for non-finite input (the observer is not advanced).
* **`reset(&mut self)`:** forget the window (called on connect, fault clear, freedrive toggle or a reference snap).
* **`mode()`, `trained_gravity()`, `trained_payload()`, `last_output()`.**

### 4.2 Integration into `Arm`, `ArmMath`, `ArmDriver`, `ArmManifest`, and `dexhal-plugin`

Follow `dexhal`'s non-clashing method naming rules (`workitems.md` item 24: `Arm` uses inherent names; `ArmMath` uses `*_padded` for queries and `apply_*` / `remove_*` / `reset_*` for mutators):

1. **`Arm<A, DOF, MAX>` (`dexhal/src/math/arm.rs`):**
   * Add `observer: Option<LstmObserver<DOF>>` to `Arm<A, DOF, MAX>` (`None` in `Arm::new()`; `Arm` does **not** hold a value `self.model`, as `A` is a zero-sized marker trait parameter `PhantomData<fn() -> A>`).
   * Configuration & status methods:
     * `pub fn set_torque_observer(&mut self, bytes: &[u8]) -> Result<(), ModelError>`: Parses `LstmObserver::<DOF>::from_bytes(bytes)?`. Rejects `ObserverMode::FrictionResidual` with `ModelError::InvalidObserver` when `!A::HAS_DYNAMICS`. Rejects `ObserverMode::FullDynamics` with `ModelError::InvalidObserver` if `(obs.trained_gravity - self.gravity)` differs by more than `1e-3` m/s² (and `set_gravity` likewise rejects changing gravity away from `trained_gravity` while a `FullDynamics` observer is installed).
     * `pub fn clear_torque_observer(&mut self)`
     * `pub fn reset_torque_observer(&mut self)`
     * `pub fn has_torque_observer(&self) -> bool`
     * `pub fn torque_observer_mode(&self) -> Option<ObserverMode>`
   * Stateful real-time estimation method (separate from `&self` stateless `inverse_dynamics`, which returns `Option<[f64; DOF]>` for the planner/simulator):
     ```rust
     pub fn step_free_torque(
         &mut self,
         q: &[f64; DOF],
         qd: &[f64; DOF],
         qdd: &[f64; DOF],
         q_cmd: Option<&[f64; DOF]>,
         control_freq_hz: u32,
     ) -> Option<[f64; DOF]>
     ```
     1. Compute **differential payload torque** `tau_payload_delta` between the current runtime `self.payload` ($P_{\text{runtime}}$, set via `set_payload`) and the observer's `trained_payload` ($P_{\text{train}}$, `Payload::NONE` when absent or stripped during training):
        * Because `payload_inverse_dynamics` (`M_p(q) qdd + C_p(q, qd) qd + G_p(q)`) is linear in the 10 rigid-body inertial parameters `(m, m*c, I_flange)` — or directly via at most two evaluations sharing a single `let (pose, jac) = self.flange_fk_jacobian(q)` pass when `self.payload != trained_payload`:

$$\Delta\tau_{\text{payload}} = \tau_{\text{payload}}(q, \dot{q}, \ddot{q}, P_{\text{runtime}}, \mathbf{g}) - \tau_{\text{payload}}(q, \dot{q}, \ddot{q}, P_{\text{train}}, \mathbf{g})$$

        * This handles every training/test combination cleanly:
          * Trained bare-flange ($P_{\text{train}} = 0$), test with gripper + object ($P_{\text{runtime}}$): adds $+\tau_{\text{payload}}(P_{\text{runtime}})$.
          * Trained with gripper mounted ($P_{\text{train}} = P_{\text{gripper}}$), test holding object ($P_{\text{runtime}} = P_{\text{gripper+obj}}$): adds only the grasped object's torque.
          * Trained with gripper mounted ($P_{\text{train}} = P_{\text{gripper}}$), test after removing gripper ($P_{\text{runtime}} = 0$): subtracts $\tau_{\text{payload}}(P_{\text{gripper}})$.
          * Same payload at train and test ($P_{\text{runtime}} == P_{\text{train}}$): skips `flange_fk_jacobian` and adds `0`.
     2. If `Some(obs) = &mut self.observer`:
        * Pack `u = [q, qd, q_cmd - q]` (`0` when `q_cmd` is `None`).
        * `let mut tau = obs.step(&u, LstmObserver::<DOF>::stride_ticks(control_freq_hz))?;` (`None` while warming up after a reset, at most `stride − 1` ticks).
        * If `obs.mode == ObserverMode::FrictionResidual`: add `A::flange_inverse_dynamics(q, qd, qdd, &self.gravity)` every tick (so inertial/Coriolis/gravity terms update at full `control_freq_hz` even between decimated LSTM evaluations).
        * Add `tau_payload_delta` and return `Some(tau)`.
     3. If `self.observer` is `None`:
        * Delegate to `self.inverse_dynamics(q, qd, qdd)` (which returns `None` when `!A::HAS_DYNAMICS`, or `Some(A::flange_inverse_dynamics(...) + tau_payload(P_runtime))`).

2. **`ArmManifest`, `ArmMath`, & `ArmDriver` (`dexhal/src/math/contract/`):**
   * **Driver-owned loading (`PluginDriver::create(config)`):** Every driver already loads its own `manifest.yaml` (`ArmManifest::from_text`) and parses its opaque per-arm `config: &[u8]` blob from `arms.toml` (where driver-specific parameters like IP address or per-robot calibration paths live). The driver loads `.dexnext` weights inside `Driver::new` / `PluginDriver::create(config)` (either compiled-in nominal weights via `include_bytes!("nominal.dexnext")` or `std::fs::read(observer_path)` from the driver's own config) and calls `arm.set_torque_observer(&bytes)?`.
   * **`ArmManifest` (`DynamicsSource::Learned` or `has_torque_observer: bool`):**
     * Add `DynamicsSource::Learned` (`dynamics_source: learned` in `manifest.yaml`, or `manifest.has_torque_observer = true` set by the driver when an observer is loaded at construction).
     * `ArmManifest::has_residual_dynamics(&self)` returns `true` when `(self.model.dynamics && self.has_residual_inertials) || self.dynamics_source != DynamicsSource::Model || self.has_torque_observer` (while `has_impedance_dynamics()` for closed-loop torque Route 3 stays unchanged: `(self.model.dynamics && self.has_static_inertials) || self.dynamics_source == DynamicsSource::Vendor`).
     * Because `manifest()` is queried by `dexzero` *after* `PluginDriver::create(config)`, `dexzero`'s existing boot check (`manifest.has_torque_sensing && manifest.has_residual_dynamics()`) automatically selects Route 6 (`HostAdmittanceTorque`) with zero new route-selection rules or `config` IPC fields in `dexzero`.
   * **Minimal `ArmMath` / `ArmDriver` surface (real-time tick + reset only; no weight-loading IPC across the contract):**
     * On `ArmMath`:
       * `fn reset_torque_observer_state(&mut self)` (no-op when `observer` is `None`)
       * `fn step_free_torque_padded(&mut self, q: &[f64; MAX_DOF], qd: &[f64; MAX_DOF], qdd: &[f64; MAX_DOF], q_cmd: Option<&[f64; MAX_DOF]>, control_freq_hz: u32) -> Option<[f64; MAX_DOF]>` (unpads inputs via `unpad`, calls `Arm::step_free_torque`, pads result via `pad`).
     * On `ArmDriver`:
       * `fn reset_torque_observer(&mut self) { self.arm_mut().reset_torque_observer_state(); }` (also called automatically inside `clear_faults` / `connect` if overridden, or by the kernel on state snaps)
       * `fn step_free_torque(&mut self, state_buffer: &[f64], q_actual: &[f64; MAX_DOF], qd_actual: &[f64; MAX_DOF], qdd_ref: &[f64; MAX_DOF], q_cmd: Option<&[f64; MAX_DOF]>) -> Option<[f64; MAX_DOF]>`: handles `DynamicsSource::Vendor` (`manifest.aux_dynamics(state_buffer)`) when no learned observer overrides it, otherwise delegates to `self.arm_mut().step_free_torque_padded(..., self.manifest().control_freq_hz)`.

3. **`dexhal-plugin` (`dexhal/plugin/src/{lib,export,host}.rs`):**
   * Only two runtime calls cross `PluginVTable` (`math_step_free_torque` and `math_reset_torque_observer`, plus the `has_torque_observer` manifest flag bit on `ManifestDesc::flags`); weights never cross the C-ABI boundary.
   * Map `ModelError::InvalidObserver` in `model_error_code` / `model_error_from_code`, bump `ABI_VERSION`, and bump `CONTRACT_VERSION`.

---

## 5. Integration with `dexzero` Kernel

Because the driver loads `.dexnext` itself in `PluginDriver::create(config)` and reports `manifest.has_residual_dynamics() == true` (plus `manifest.has_torque_observer` if deadband tightening is desired), **`dexzero` needs no knowledge of `.dexnext` files, no new `arms.toml` kernel keys, no `ctl_req` ops, and no `ipc.md` layout changes**:

1. **Boot & Route Selection (`compliance.md` §2–3) — Unchanged:**
   * Existing Route 6 (`HostAdmittanceTorque`) eligibility (`manifest.has_torque_sensing && manifest.has_residual_dynamics() && position_mode`) works as-is because `manifest.has_residual_dynamics()` is already `true` when the driver has an observer installed (even on arms with `has_static_inertials: false` or `model.dynamics == false`).
   * Optionally, when `manifest.has_torque_observer` is `true`, the kernel's force conditioner can use its tight contact deadband (comparable to Route 5 wrist F/T) instead of the coarse unmodeled-friction deadband.

2. **Real-Time Executor Tick (`compliance.md` §4 & §6):**
   * In the Route 6 / torque-residual wrench estimation path, replace the stateless `driver.inverse_dynamics(&q_actual, &qd_actual, &reference.qdd)` call with `driver.step_free_torque(...)` every control cycle whenever `manifest.has_torque_sensing && manifest.has_residual_dynamics()` (even when all axes are currently rigid, keeping the observer's sliding window warm and `state.wrench` + contact reflexes continuously active):
     ```rust
     let tau_meas = manifest.link_torques(state.torques()); // scale raw current if torque_constants set
     if let Some(tau_free) = driver.step_free_torque(
         &state_buf,
         &q_actual,
         &qd_actual,
         &reference.qdd,
         Some(&last_commanded_q),
     ) {
         let r = sub_dof(&tau_free, &tau_meas, dof); // r = tau_free - tau_measured
         if let Some(j_pinv) = driver.jacobian_pseudo_inverse(&q_actual) {
             // jacobian_pseudo_inverse returns [[f64; 6]; MAX_DOF] (DOF x 6):
             // f_ext = (J^T)^+ * r = (J^+)^T * r -> f_ext[c] = sum_j j_pinv[j][c] * r[j]
             let f_ext_raw = transpose_pinv_mul(&j_pinv, &r, dof);
             let f_ext = wrench_conditioner.process(f_ext_raw); // LPF + tare_wrench() bias + deadband
             write_state_wrench(&mut shm_state, &f_ext);
         }
     }
     ```
   * **Payload compensation in joint space vs. Cartesian space:** In Route 5 (`CoreState::wrench`), a wrist F/T sensor sits *distal* to the joints and measures tool/payload weight directly, so `driver.external_wrench` subtracts `self.payload` weight in Cartesian space. In Route 6, `CoreState::torques` measures joint torques, and `tau_free` already accounts for the full runtime payload ($\hat{\tau}_{\text{nn}}(P_{\text{train}}) + \tau_{\text{payload}}(P_{\text{runtime}}) - \tau_{\text{payload}}(P_{\text{train}})$) in joint space—including gravity, Coriolis, and $\ddot{q}$ inertial forces—before subtracting `tau_meas`. Therefore, Route 6's Cartesian `wrench_conditioner` must **not** run `driver.external_wrench`'s Cartesian payload-weight subtraction a second time; it applies only frame rotation, low-pass filtering, `tare_wrench()` static bias removal, and deadband.
   * Call `driver.reset_torque_observer()` on `connect`, `CLEAR_FAULTS`, `SET_FREEDRIVE` transitions, and whenever `q_ref` snaps after a reflex (`STOP`, `YIELD`, `RETRACT`) — which is a harmless no-op on drivers without a learned observer.

---

## 6. Acceptance Criteria

1. **Numerical Parity Test (`dexhal`):** `LstmObserver::from_bytes` + `step` in a 1 kHz loop (stride 10, warm-up included) over 100 windows matches PyTorch `float64` (`model.eval()`) inference within `< 1e-10` max absolute error (`torque_observer::tests::pytorch_parity`; Rust and Python draw the same weights and inputs from a shared LCG, only the PyTorch outputs are committed), and the spread schedule is bit-identical across strides 1, 3 and 10.
2. **Plugin & Manifest Round-Trip Test (`dexhal-plugin`):** A driver constructed with `.dexnext` weights reports `manifest().has_residual_dynamics() == true` across `ManifestDesc`, and `step_free_torque` / `reset_torque_observer` (plus `ModelError::InvalidObserver` rejection on malformed `.dexnext`, wrong `DOF`, `FrictionResidual` when `!HAS_DYNAMICS`, and mismatched `FullDynamics` gravity) work cleanly through `LoadedDriver`.
3. **Real-Time & `#![no_std]` Safety (`dexhal` & `dexzero`):** `cargo build --manifest-path dexhal/Cargo.toml --target thumbv7em-none-eabihf` succeeds under `#![forbid(unsafe_code)]`; counting allocator confirms **0 bytes allocated** during `step_free_torque` and the 1 kHz executor tick; benchmarked WCET of `step_free_torque`, warm-up included, fits within the 1 kHz cycle budget (and a whole window within the 10 ms cycle at 100 Hz).
4. **Payload & Contact Isolation (`dexzero` sim):** Attaching a simulated payload with `set_payload()` (including when $P_{\text{runtime}} \neq P_{\text{train}}$) produces `||f_ext|| < deadband` during free-space motion, while applying a 2 N external TCP force populates `state.wrench`, triggers `in_contact == true` / reflex thresholds, and drives Route 6 admittance yield.