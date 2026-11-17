# Engineering Spec: FACTR 2 (NEXT) Virtual Torque Sensor Integration

## 1. Objective & Architecture Summary

Implement **NEXT (Neural External Torque Estimation)** from FACTR 2 (<https://jasonjzliu.com/factr2/>, [arXiv:2606.12406](https://arxiv.org/abs/2606.12406)) as a `#![no_std]`, zero-allocation virtual torque sensor inside **`dexhal`**, loaded by the **driver** at construction and consumed transparently by **`dexzero`** via Compliance Route 6 (`HostAdmittanceTorque`) with **zero `dexzero` config or IPC changes**.

* **Offline (`dexforge` / Python):** Collect ~10 minutes of free-space trajectories via `dexzero`, train a compact recurrent model (LSTM) in `float64` to predict free-space link torque $\hat{\tau}_f$ (or unmodeled friction / actuator residual $\hat{\tau}_{\text{fric}}$), and export flat binary weights (`.dexnext`).
* **Driver & HAL (`dexhal` + `dexhal-plugin`):** The OEM driver owns loading its `.dexnext` weights at construction (`PluginDriver::create(config)` — either compiled-in nominal weights via `include_bytes!` or a per-robot file path from the driver's own `config` blob) into `dexhal::math::Arm` and reports `has_residual_dynamics() == true` through `ArmManifest`. On each control tick, `ArmDriver::step_free_torque` advances the observer and applies differential runtime payload compensation to output predicted free-space link torque $\hat{\tau}_{\text{free}}$.
* **Kernel (`dexzero`):** Sees an ordinary Route 6 arm (`manifest.has_torque_sensing && manifest.has_residual_dynamics()`), calls `driver.step_free_torque(...)` instead of stateless `driver.inverse_dynamics(...)` in the Route 6 observer path, subtracts `manifest.link_torques(state.torques())` ($r = \hat{\tau}_{\text{free}} - \tau_{\text{measured}}$), projects through `jacobian_pseudo_inverse(q_actual)` to $\hat{F}_{\text{ext}}$, and feeds Route 6 admittance and contact reflexes (`STOP`, `YIELD`, `RETRACT`).

---

## 2. Network Structure & Binary Weight Format (`.dexnext`)

### 2.1 Paper Alignment vs. Hard-Real-Time Constraints

FACTR 2's reference NEXT model (arXiv:2606.12406 Appendix A.1–A.2) makes four empirical choices that directly impact accuracy, drift, and worst-case execution time (WCET):

1. **Input features $(q, \dot{q}, \Delta q_d)$ (`3 * DOF`):** Using joint tracking / command error $\Delta q_d = q_{\text{cmd}} - q_{\text{actual}}$ alongside $(q, \dot{q})$ drops contact $L_1$ error from $1.12\text{ N}\cdot\text{m}$ to $0.414\text{ N}\cdot\text{m}$ (free-space) / $0.547\text{ N}\cdot\text{m}$ (contact) without noisy $\ddot{q}$ differentiation. (On AgileX Piper, free-space $L_1$ error drops to $0.018\text{ N}\cdot\text{m}$ vs. $0.435\text{ N}\cdot\text{m}$ for URDF inverse dynamics.)
2. **Stateless sliding-window evaluation ($W = 50$ at $100\text{ Hz}$, i.e. $0.5\text{ s}$ lookback) vs. stateful streaming:** Rather than carrying $(h_t, c_t)$ indefinitely across long trajectories—where sustained external contact or out-of-distribution states can corrupt or drift $c_t$—NEXT resets $(h, c) = (0, 0)$ and runs over a rolling ring buffer of the last $W$ observations.
3. **Backbone & readout:** The paper uses a 2-layer LSTM ($H = 128$) + 2-layer MLP head ($256$), taking $\approx 1.76\text{ ms}$ per windowed pass on one desktop core and therefore evaluated at $100\text{ Hz}$ (zero-order-held between ticks).
4. **Control-loop budget in `dexhal` (500–1000 Hz):** A $1.76\text{ ms}$ pass exceeds a $1\text{ ms}$ (1 kHz) hard-real-time cycle budget. `dexhal` therefore supports **both** evaluation strategies under a bounded, stack-allocated ceiling (`OBSERVER_MAX_LAYERS = 2`, `OBSERVER_MAX_HIDDEN = 64`, `OBSERVER_MAX_MLP_HIDDEN = 64`, `OBSERVER_MAX_WINDOW = 50`, `OBSERVER_MAX_IN_DIM = 3 * MAX_DOF = 24`):
   * **`EvalMode::SlidingWindow` (Default / Recommended, faithful to FACTR 2):** Maintains a fixed-capacity ring buffer of recent inputs sampled every `stride_ticks` (e.g. `stride_ticks = 5` at 500 Hz $\rightarrow$ 100 Hz effective sample rate, `window_len = 50` $\rightarrow$ 0.5 s lookback). With $L \in \{1, 2\}$, $H \in \{32, 64\}$, $W \le 50$ in `f64`, a full window pass is $\sim 3\times 10^5$–$8\times 10^5$ FLOPs ($\approx 80$–$250\,\mu\text{s}$), well inside a 1–2 ms cycle, holding $\hat{\tau}_{\text{nn}}$ across intermediate ticks while still updating analytical rigid-body / payload terms every tick.
   * **`EvalMode::StatefulStream` (`window_len == 1`):** Advances $(h_t, c_t)$ one step per tick ($\approx 2$–$10\,\mu\text{s}$ for $H \in \{32, 64\}$) for ultra-tight embedded budgets (e.g. `thumbv7em-none-eabihf` MCUs), paired with explicit state resets on fault clear or trajectory discontinuity.

### 2.2 Topology & Math (`f64` precision throughout)

* **Input Feature Schema (`InputFeatures`):**
  * `PositionVelocityCommandError` (`0`, default, $D_{\text{in}} = 3 \cdot \text{DOF}$): $u_t = [q, \dot{q}, \Delta q_d]$ where $\Delta q_d = q_{\text{cmd}} - q_{\text{actual}}$ (at rest / before the first command, $\Delta q_d = 0$).
  * `PositionVelocityAcceleration` (`1`, $D_{\text{in}} = 3 \cdot \text{DOF}$): $u_t = [q, \dot{q}, \ddot{q}_{\text{ref}}]$ for pure torque-controlled modes where no joint position target $q_{\text{cmd}}$ is active.
  * `PositionVelocity` (`2`, $D_{\text{in}} = 2 \cdot \text{DOF}$): $u_t = [q, \dot{q}]$.

* **Input Normalization (element-wise z-score):**

$$x_t = \frac{u_t - \mu_{\text{in}}}{\sigma_{\text{in}}}$$

* **Recurrent Core ($L \in \{1, 2\}$ standard LSTM layers, hidden size $H \le 64$, gate order `[i, f, g, o]` matching PyTorch `nn.LSTM`):**
  For layer $\ell \in \{0, \dots, L-1\}$ with layer input $z_t^{(0)} = x_t$ and $z_t^{(\ell)} = h_t^{(\ell-1)}$:

$$\begin{aligned}
[i_t, f_t, g_t, o_t]^T &= W_{ih}^{(\ell)} z_t^{(\ell)} + b_{\text{gates}}^{(\ell)} + W_{hh}^{(\ell)} h_{t-1}^{(\ell)} \\
c_t^{(\ell)} &= \sigma(f_t) \odot c_{t-1}^{(\ell)} + \sigma(i_t) \odot \tanh(g_t) \\
h_t^{(\ell)} &= \sigma(o_t) \odot \tanh(c_t^{(\ell)})
\end{aligned}$$

  *(where $b_{\text{gates}}^{(\ell)} = b_{ih}^{(\ell)} + b_{hh}^{(\ell)}$ is pre-summed in `f64` on export).*

* **Readout Head ($M \in \{0, 1\}$ hidden MLP layers of width $D_{\text{mlp}} \le 64$ with ReLU, plus linear output + denormalization):**
  * If $D_{\text{mlp}} = 0$ (linear head): $y_t = W_{\text{out}} h_t^{(L-1)} + b_{\text{out}}$.
  * If $D_{\text{mlp}} > 0$ (2-layer MLP head, matching FACTR 2): $m_t = \max(0, W_{\text{mlp}} h_t^{(L-1)} + b_{\text{mlp}})$, $y_t = W_{\text{out}} m_t + b_{\text{out}}$.

$$\hat{\tau}_{\text{nn}} = \sigma_{\text{out}} \odot y_t + \mu_{\text{out}} \in \mathbb{R}^{\text{DOF}}$$

* **Operating Modes (`ObserverMode`):**
  * `ObserverMode::FullDynamics` (`0`): $\hat{\tau}_{\text{nn}}$ predicts total bare-arm free-space link torque $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) + \tau_{\text{fric}}$. Does **not** require `ArmModel::HAS_DYNAMICS` (`model.dynamics == false` or `has_residual_inertials == false` is supported), provided the arm remains mounted with the same base-frame gravity vector $\mathbf{g}$ used during training.
  * `ObserverMode::FrictionResidual` (`1`): $\hat{\tau}_{\text{nn}}$ predicts only the unmodeled actuator/transmission residual $\tau_{\text{measured}} - \tau_{\text{rigid}}(q, \dot{q}, \ddot{q}, \mathbf{g})$. Requires `ArmModel::HAS_DYNAMICS == true` (or `DynamicsSource::Vendor` on `ArmDriver`); automatically adapts to arbitrary base-frame `gravity` changes (wall/ceiling mounts) and full-rate $\ddot{q}_{\text{ref}}$ changes between decimated LSTM ticks.

### 2.3 Sign Convention & Units Alignment with `dexhal`

Per [`CoreState::torques`](../src/math/contract/wire.rs) and `dexzero`'s `compliance.md` §6:

1. **Units (Link-Side N·m):** `CoreState::torques` carries raw motor current/effort when `manifest.torque_constants` (`sensing.torque_constants` in `manifest.yaml`) is finite, and link-side N·m otherwise. Both data collection and runtime inference **must** operate on link-side torque $\tau_{\text{measured}} = \text{manifest.link\_torques}(\text{state.torques()})$ in N·m.
2. **Sign Convention:** `CoreState::torques` is the link-side joint effort applied *by the actuator* in the positive joint direction.
   * FACTR 2 Eq. (2) writes $\hat{\tau}_{\text{ext}}^{\text{paper}} = \tau_m - \hat{\tau}_f$ (additional motor effort required to oppose contact, i.e. $-J^T F_{\text{env}\to\text{tool}}$), whereas `dexzero` Route 6 (`compliance.md` §6) defines the residual as:

$$r = \hat{\tau}_{\text{free}} - \tau_{\text{measured}}, \qquad \hat{F}_{\text{ext}} = (J^T)^+ r$$

   so that $\hat{F}_{\text{ext}}$ is the wrench the **environment applies onto the tool** (matching `CoreState::wrench` in Route 5). In free space ($\tau_{\text{ext}} = 0 \implies \tau_{\text{measured}} = \tau_{\text{free}}$), the training target is identical; at runtime, `dexzero` must keep $r = \hat{\tau}_{\text{free}} - \tau_{\text{measured}}$.

### 2.4 Packed Binary Format (`.dexnext`)

Little-endian byte stream, parsed safely via `from_le_bytes` slices (no `unsafe` pointer/alignment casts, so arbitrary `&[u8]` buffers from `include_bytes!` or IPC work under `#![forbid(unsafe_code)]`):

1. **Header (168 bytes):**
   * `magic: [u8; 8]` = `b"DEXNEXT\x01"`
   * `dof: u8` (`1..=MAX_DOF`)
   * `num_layers: u8` (`1..=OBSERVER_MAX_LAYERS`)
   * `hidden_dim: u16` (`1..=OBSERVER_MAX_HIDDEN`)
   * `mlp_dim: u16` (`0` for linear head, or `1..=OBSERVER_MAX_MLP_HIDDEN`)
   * `input_dim: u8` (`2 * dof` or `3 * dof`)
   * `features: u8` (`0 = PositionVelocityCommandError`, `1 = PositionVelocityAcceleration`, `2 = PositionVelocity`)
   * `mode: u8` (`0 = FullDynamics`, `1 = FrictionResidual`)
   * `eval_mode: u8` (`0 = SlidingWindow`, `1 = StatefulStream`)
   * `window_len: u16` (`1..=OBSERVER_MAX_WINDOW`; must be `1` when `eval_mode == StatefulStream`)
   * `reserved: [u8; 4]` (zeroed)
   * `sample_rate_hz: f64` (nominal observer tick rate in Hz, e.g. `100.0` or `500.0`; finite and `> 0.0`)
   * `gravity: [f64; 3]` (base-frame gravity vector `[gx, gy, gz]` in m/s² active during training; validated in `FullDynamics` mode)
   * `trained_payload_mass: f64` (kg; `0.0` when collected bare-flange, or mass of the mounted tool/gripper present during free-space collection and NOT already subtracted from training targets)
   * `trained_payload_com: [f64; 3]` (flange-frame CoM `[x, y, z]` in m)
   * `trained_payload_inertia: [f64; 9]` (row-major 3×3 rotational inertia about `trained_payload_com` in flange axes, kg·m²)
   * `padding: [u8; 8]` (zeroed to 8-byte boundary)
2. **Payload (contiguous little-endian `f64` values):**
   * `in_mean[input_dim]`, `in_std[input_dim]` (all finite, `in_std > 1e-12`)
   * For each LSTM layer $\ell \in 0..\text{num\_layers}$ (let $D_\ell = \text{input\_dim}$ if $\ell = 0$ else $\text{hidden\_dim}$):
     * `w_ih[4 * hidden_dim * D_l]` (row-major `[4 * H, D_l]`, gate order `i, f, g, o`)
     * `w_hh[4 * hidden_dim * hidden_dim]` (row-major `[4 * H, H]`)
     * `b_gates[4 * hidden_dim]` (`b_ih + b_hh`)
   * If `mlp_dim > 0`:
     * `w_mlp[mlp_dim * hidden_dim]` (row-major `[mlp_dim, H]`)
     * `b_mlp[mlp_dim]`
   * Readout head (let $D_{\text{head}} = \text{mlp\_dim}$ if $\text{mlp\_dim} > 0$ else $\text{hidden\_dim}$):
     * `w_out[dof * D_head]` (row-major `[dof, D_head]`)
     * `b_out[dof]`
   * `out_mean[dof]`, `out_std[dof]` (all finite, `out_std > 0.0`)

Exact byte length is deterministic from header dimensions; trailing or truncated bytes are rejected.

---

## 3. PyTorch Data Collection & Training (`dexforge`)

### 3.1 Data Collection (`codegen/next/collect_next_data.py`)

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
   * Split into contiguous fault-free / contact-free segments, discarding frames (with a safety margin) where `kernel_fault != 0`, `driver_fault != 0`, or `in_contact == 1.0`.

### 3.2 Training & Export (`codegen/next/train_next.py`)

```python
class NextObserverNet(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dof: int, num_layers: int = 2, mlp_dim: int = 64):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            in_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = (
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, mlp_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mlp_dim, dof),
            )
            if mlp_dim > 0
            else torch.nn.Linear(hidden_dim, dof)
        )

    def forward(self, x, hx=None):
        out, hx = self.lstm(x, hx)
        return self.head(out), hx
```

* **Resampling / Stride:** Downsample contiguous trajectory segments to target `sample_rate_hz` (e.g. $100\text{ Hz}$ via integer stride $k = \text{round}(\text{control\_freq\_hz} / \text{sample\_rate\_hz})$, matching runtime `stride_ticks`) and construct $\Delta q_d = q_{\text{cmd}} - q$.
* **Supervision Target & Training-Time Payload Normalization:**
  * Either subtract $\tau_{\text{payload}}(q, \dot{q}, \ddot{q}_{\text{ref}}, P_{\text{train}}, \mathbf{g})$ during training and store `trained_payload = Payload::NONE` in the `.dexnext` header, **or** train directly on $\tau_{\text{link}}$ with the mounted tool/gripper in place and record `trained_payload = P_train` in the `.dexnext` header so runtime computes the exact differential payload torque $\tau_{\text{payload}}(P_{\text{runtime}}) - \tau_{\text{payload}}(P_{\text{train}})$:
    * `FullDynamics`: $\tau_{\text{target}} = \tau_{\text{link}}$ (with `trained_payload = P_train` stored in header).
    * `FrictionResidual`: $\tau_{\text{target}} = \tau_{\text{link}} - \tau_{\text{rigid}}(q, \dot{q}, \ddot{q}_{\text{ref}}, \mathbf{g}) - \tau_{\text{payload}}(q, \dot{q}, \ddot{q}_{\text{ref}}, P_{\text{train}}, \mathbf{g})$ (with `trained_payload = Payload::NONE` stored in header, since $P_{\text{train}}$ is already stripped out of $\tau_{\text{target}}$).
* **Training Regime:**
  * `SlidingWindow` (default, $W = 50$): train with zero initial hidden state `(h_0, c_0) = 0` on length-$W$ windows, supervising the final step $\hat{\tau}_t$ with MSE ($L_2$) loss on normalized target torque, AdamW (`lr = 1e-3`, `weight_decay = 1e-6`, gradient clip `1.0`, early stopping patience `20`), matching FACTR 2 Table 3.
  * `StatefulStream`: TBPTT over contiguous sub-sequences ($T = 250$) with hidden state carried across consecutive chunks and random zero-state starts so reset transients remain bounded.
* **Export:** Cast all weights to `np.float64`, fold `b_ih + b_hh` into `b_gates`, validate numerical parity between PyTorch `float64` (`model.eval()`) and a pure-NumPy step/window reference on a held-out trajectory (`max_abs_err < 1e-12`), and write `.dexnext`.

---

## 4. Rust `#![no_std]` Implementation (`dexhal` & `dexhal-plugin`)

### 4.1 New Module: `dexhal::math::torque_observer`

Constraints: `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap allocation, bounded loops, `libm::{exp, tanh}`.

Because `Arm<A, DOF, MAX>` is already deployed across every OEM driver with **three type/const parameters** (`Arm<Ur5, 6, 16>`, etc.) and `ArmMath` / `dexhal-plugin` erase `DOF` behind `MAX_DOF = 8`, `LstmObserver` **must not** add `H` or `IN_DIM` const generics to `Arm` (which would break all existing `Arm` signatures and prevent loading runtime `.dexnext` files with different hidden widths or feature modes). Instead, `LstmObserver<const DOF: usize>` uses fixed stack ceilings and reads the active dimensions (`num_layers`, `hidden_dim`, `mlp_dim`, `input_dim`, `window_len`) from the `.dexnext` header:

```rust
pub const OBSERVER_MAX_LAYERS: usize = 2;
pub const OBSERVER_MAX_HIDDEN: usize = 64;
pub const OBSERVER_MAX_MLP_HIDDEN: usize = 64;
pub const OBSERVER_MAX_WINDOW: usize = 50;
pub const OBSERVER_MAX_IN_DIM: usize = 3 * MAX_DOF; // 24

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObserverMode {
    FullDynamics = 0,
    FrictionResidual = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFeatures {
    PositionVelocityCommandError = 0,
    PositionVelocityAcceleration = 1,
    PositionVelocity = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    SlidingWindow = 0,
    StatefulStream = 1,
}

pub struct LstmObserver<const DOF: usize> {
    mode: ObserverMode,
    features: InputFeatures,
    eval_mode: EvalMode,
    num_layers: usize,
    hidden_dim: usize,
    mlp_dim: usize,
    input_dim: usize,
    window_len: usize,
    sample_rate_hz: f64,
    trained_gravity: [f64; 3],
    trained_payload: Payload,
    // Normalization & network weights (active prefixes used up to header dims)
    in_mean: [f64; OBSERVER_MAX_IN_DIM],
    in_std: [f64; OBSERVER_MAX_IN_DIM],
    w_ih0: [[f64; OBSERVER_MAX_IN_DIM]; 4 * OBSERVER_MAX_HIDDEN],
    w_ih_upper: [[[f64; OBSERVER_MAX_HIDDEN]; 4 * OBSERVER_MAX_HIDDEN]; OBSERVER_MAX_LAYERS - 1],
    w_hh: [[[f64; OBSERVER_MAX_HIDDEN]; 4 * OBSERVER_MAX_HIDDEN]; OBSERVER_MAX_LAYERS],
    b_gates: [[f64; 4 * OBSERVER_MAX_HIDDEN]; OBSERVER_MAX_LAYERS],
    w_mlp: [[f64; OBSERVER_MAX_HIDDEN]; OBSERVER_MAX_MLP_HIDDEN],
    b_mlp: [f64; OBSERVER_MAX_MLP_HIDDEN],
    w_out: [[f64; OBSERVER_MAX_HIDDEN]; DOF], // indexed up to mlp_dim (if > 0) else hidden_dim
    b_out: [f64; DOF],
    out_mean: [f64; DOF],
    out_std: [f64; DOF],
    // Runtime state (zero-allocation ring buffer + recurrent state + held output)
    history: [[f64; OBSERVER_MAX_IN_DIM]; OBSERVER_MAX_WINDOW],
    history_len: usize,
    history_head: usize,
    tick_counter: u32,
    h: [[f64; OBSERVER_MAX_HIDDEN]; OBSERVER_MAX_LAYERS],
    c: [[f64; OBSERVER_MAX_HIDDEN]; OBSERVER_MAX_LAYERS],
    last_tau_nn: [f64; DOF],
}
```

*(Note: Add `ModelError::InvalidObserver` to [`ModelError`](../src/math/chain.rs) — which is the error type used by `Arm` and `ArmMath`, automatically converting to `ArmError::Model(ModelError::InvalidObserver)` via `From<ModelError> for ArmError` — and map it in `dexhal-plugin`'s `model_error_code` / `model_error_from_code`.)*

* **Methods on `LstmObserver<DOF>`:**
  * `from_bytes(bytes: &[u8]) -> Result<Self, ModelError>`: Validates magic, `dof == DOF`, dimension bounds against `OBSERVER_MAX_*` (note `w_out` column bound `max(OBSERVER_MAX_HIDDEN, OBSERVER_MAX_MLP_HIDDEN)`), exact byte length, finiteness of all parameters, `sample_rate_hz > 0.0`, `in_std > 1e-12`, and `out_std > 0.0`.
  * `reset(&mut self)`: Zeroes `h`, `c`, `history`, `history_len`, `history_head`, `tick_counter`, and `last_tau_nn` (called on boot, fault clear, freedrive toggle, or state discontinuity).
  * `step_raw(&mut self, u: &[f64], stride_ticks: u32) -> [f64; DOF]`:
    * `StatefulStream`: normalizes `u`, advances $(h, c)$ across `0..num_layers`, evaluates MLP/linear readout, denormalizes, updates and returns `last_tau_nn`.
    * `SlidingWindow`: every `stride_ticks` ticks (including the first tick after `reset`), pushes normalized `x` into `history` (left-padding with the initial observation while `history_len < window_len` to avoid startup transients), runs the LSTM from `(h, c) = (0, 0)` across `window_len` frames, evaluates the readout head at the final frame, updates `last_tau_nn`, and returns `last_tau_nn` (returning the held `last_tau_nn` on intermediate ticks when `stride_ticks > 1`).

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
        * Pack `u` according to `obs.features` (`[q, qd, q_cmd - q]` with `0` when `q_cmd` is `None`, `[q, qd, qdd]`, or `[q, qd]`).
        * Compute `stride_ticks = ((control_freq_hz as f64) / obs.sample_rate_hz).round().max(1.0) as u32`.
        * `let mut tau = obs.step_raw(&u, stride_ticks);`
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
   * Map `ModelError::InvalidObserver` in `model_error_code` / `model_error_from_code`, bump `ABI_VERSION` (`1 -> 2`), and bump `CONTRACT_VERSION` (`0.3.0 -> 0.4.0`).

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

1. **Numerical Parity Test (`dexhal`):** `LstmObserver::from_bytes` + `step_raw` (for both `SlidingWindow` and `StatefulStream`, 1-layer linear and 2-layer MLP-head models) over a 1,000-step trajectory matches PyTorch `float64` (`model.eval()`) inference within `< 1e-12` max absolute error.
2. **Plugin & Manifest Round-Trip Test (`dexhal-plugin`):** A driver constructed with `.dexnext` weights reports `manifest().has_residual_dynamics() == true` across `ManifestDesc`, and `step_free_torque` / `reset_torque_observer` (plus `ModelError::InvalidObserver` rejection on malformed `.dexnext`, wrong `DOF`, `FrictionResidual` when `!HAS_DYNAMICS`, and mismatched `FullDynamics` gravity) work cleanly through `LoadedDriver`.
3. **Real-Time & `#![no_std]` Safety (`dexhal` & `dexzero`):** `cargo build --manifest-path dexhal/Cargo.toml --target thumbv7em-none-eabihf` succeeds under `#![forbid(unsafe_code)]`; counting allocator confirms **0 bytes allocated** during `step_free_torque` and the 1 kHz executor tick; benchmarked WCET of `step_free_torque` fits within the 1 kHz cycle budget.
4. **Payload & Contact Isolation (`dexzero` sim):** Attaching a simulated payload with `set_payload()` (including when $P_{\text{runtime}} \neq P_{\text{train}}$) produces `||f_ext|| < deadband` during free-space motion, while applying a 2 N external TCP force populates `state.wrench`, triggers `in_contact == true` / reflex thresholds, and drives Route 6 admittance yield.