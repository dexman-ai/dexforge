//! Learned free-space torque observer (FACTR 2, "NEXT").
//!
//! A fixed LSTM maps a sliding window of joint state to the torque the arm
//! needs in free space, so `τ_measured − τ_free` is the external torque
//! (Route 6). The architecture is the one the paper selects in its ablation
//! (arXiv 2606.12406, Table 3 and §A.2) and uses for both the Franka and the
//! AgileX Piper:
//!
//! * input `[q, qd, q_cmd − q]` (`3·DOF`), normalised per channel;
//! * 2-layer LSTM, hidden size [`NEXT_HIDDEN`];
//! * MLP head `Linear(128, 256) → ReLU → Linear(256, DOF)`, de-normalised;
//! * window of [`NEXT_WINDOW`] samples at [`NEXT_SAMPLE_RATE_HZ`], evaluated
//!   statelessly: every sample the network re-runs from a zero state over
//!   the newest window (left-padded with the first sample after a reset).
//!
//! [`ObserverMode`] selects what the network predicts: the full free-space
//! torque (as in the paper) or the residual on top of the driver's rigid-body
//! model ([`ArmModel::flange_inverse_dynamics`](crate::math::arm::ArmModel)).
//!
//! # Scheduling
//!
//! A window costs [`NEXT_WINDOW`] LSTM frames. At a control rate above
//! 100 Hz the network is sampled every [`LstmObserver::stride_ticks`] ticks
//! and the work is spread over the ticks in between: the first 49 frames of
//! the next window (all already-known samples) are advanced a few per tick,
//! and the sample tick only runs the final frame and the head. The result is
//! bit-identical to running the whole window at the sample tick. After a
//! reset the first output appears on the last tick of the first stride
//! period ([`LstmObserver::step`] returns `None` before).
//!
//! # `.dexnext` format (version 2)
//!
//! Little-endian; a 144-byte header followed by `f64` parameters.
//!
//! | bytes | field |
//! |---|---|
//! | 0..8 | magic `DEXNEXT\x02` |
//! | 8 | `dof` (`u8`) |
//! | 9 | mode: 0 full dynamics, 1 friction residual |
//! | 10..16 | reserved, zero |
//! | 16..40 | gravity the model was trained under (3 × `f64`) |
//! | 40..48 | trained payload mass |
//! | 48..72 | trained payload centre of mass (flange frame) |
//! | 72..144 | trained payload inertia about the CoM (row-major 3×3) |
//!
//! Parameters, every matrix stored transposed (input-major: for each input
//! column, all output rows), with `G = 4·128` LSTM gates in PyTorch order
//! `i, f, g, o` and `b = b_ih + b_hh`:
//!
//! `in_mean[3D]`, `in_std[3D]`,
//! `W_ih0ᵀ[3D][G]`, `W_hh0ᵀ[128][G]`, `b0[G]`,
//! `W_ih1ᵀ[128][G]`, `W_hh1ᵀ[128][G]`, `b1[G]`,
//! `W_mlpᵀ[128][256]`, `b_mlp[256]`, `W_outᵀ[256][D]`, `b_out[D]`,
//! `out_mean[D]`, `out_std[D]` ([`next_param_count`]).
//!
//! The weights are read in place from the `&'static` bytes; nothing is
//! copied or allocated.

use crate::math::chain::ModelError;
use crate::math::payload::Payload;

/// LSTM layers.
pub const NEXT_LAYERS: usize = 2;
/// LSTM hidden size.
pub const NEXT_HIDDEN: usize = 128;
/// LSTM gate rows per layer (`i, f, g, o`).
pub const NEXT_GATES: usize = 4 * NEXT_HIDDEN;
/// Hidden width of the MLP head.
pub const NEXT_MLP: usize = 256;
/// Samples per evaluation window.
pub const NEXT_WINDOW: usize = 50;
/// Rate (Hz) the network samples joint state at.
pub const NEXT_SAMPLE_RATE_HZ: u32 = 100;
/// `.dexnext` magic (format version 2).
pub const NEXT_MAGIC: [u8; 8] = *b"DEXNEXT\x02";
/// `.dexnext` header length in bytes.
pub const NEXT_HEADER_LEN: usize = 144;

const H: usize = NEXT_HIDDEN;
const G: usize = NEXT_GATES;
const M: usize = NEXT_MLP;
const PREFIX: usize = NEXT_WINDOW - 1;

/// What a learned observer predicts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ObserverMode {
    /// The whole free-space torque, gravity included (the paper's setup);
    /// bound to the gravity it was trained under.
    FullDynamics = 0,
    /// The residual over the driver's rigid-body inverse dynamics (friction
    /// and model error); gravity-independent.
    FrictionResidual = 1,
}

/// Number of `f64` parameters in a `.dexnext` file for `dof` joints.
pub const fn next_param_count(dof: usize) -> usize {
    Layout::new(dof).end
}

/// Exact `.dexnext` file length in bytes for `dof` joints.
pub const fn next_file_len(dof: usize) -> usize {
    NEXT_HEADER_LEN + 8 * next_param_count(dof)
}

/// Parameter offsets (in `f64`s) of every tensor.
#[derive(Clone, Copy)]
struct Layout {
    in_mean: usize,
    in_std: usize,
    w_ih0: usize,
    w_hh0: usize,
    b0: usize,
    w_ih1: usize,
    w_hh1: usize,
    b1: usize,
    w_mlp: usize,
    b_mlp: usize,
    w_out: usize,
    b_out: usize,
    out_mean: usize,
    out_std: usize,
    end: usize,
}

impl Layout {
    const fn new(dof: usize) -> Layout {
        let d = 3 * dof;
        let in_mean = 0;
        let in_std = in_mean + d;
        let w_ih0 = in_std + d;
        let w_hh0 = w_ih0 + d * G;
        let b0 = w_hh0 + H * G;
        let w_ih1 = b0 + G;
        let w_hh1 = w_ih1 + H * G;
        let b1 = w_hh1 + H * G;
        let w_mlp = b1 + G;
        let b_mlp = w_mlp + H * M;
        let w_out = b_mlp + M;
        let b_out = w_out + M * dof;
        let out_mean = b_out + dof;
        let out_std = out_mean + dof;
        Layout {
            in_mean,
            in_std,
            w_ih0,
            w_hh0,
            b0,
            w_ih1,
            w_hh1,
            b1,
            w_mlp,
            b_mlp,
            w_out,
            b_out,
            out_mean,
            out_std,
            end: out_std + dof,
        }
    }
}

#[derive(Clone, Copy)]
struct LstmState {
    h: [[f64; H]; NEXT_LAYERS],
    c: [[f64; H]; NEXT_LAYERS],
}

const ZERO_STATE: LstmState = LstmState {
    h: [[0.0; H]; NEXT_LAYERS],
    c: [[0.0; H]; NEXT_LAYERS],
};

#[inline(always)]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + libm::exp(-x))
}

/// `acc += col · x`, `col` being `R` little-endian `f64`s.
#[inline(always)]
fn axpy<const R: usize, const B: usize>(acc: &mut [f64; R], col: &[u8; B], x: f64) {
    const { assert!(B == 8 * R) };
    for (r, a) in acc.iter_mut().enumerate() {
        let w = [
            col[8 * r],
            col[8 * r + 1],
            col[8 * r + 2],
            col[8 * r + 3],
            col[8 * r + 4],
            col[8 * r + 5],
            col[8 * r + 6],
            col[8 * r + 7],
        ];
        *a += f64::from_le_bytes(w) * x;
    }
}

fn read_f64(bytes: &[u8], i: usize) -> f64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&bytes[8 * i..8 * i + 8]);
    f64::from_le_bytes(w)
}

/// The network: weights in place plus the decoded normalisation.
#[derive(Clone, Copy)]
struct Net<const DOF: usize> {
    params: &'static [u8],
    in_mean: [[f64; DOF]; 3],
    in_std: [[f64; DOF]; 3],
    out_mean: [f64; DOF],
    out_std: [f64; DOF],
}

impl<const DOF: usize> Net<DOF> {
    const L: Layout = Layout::new(DOF);

    #[inline(always)]
    fn col<const B: usize>(&self, at: usize) -> &[u8; B] {
        <&[u8; B]>::try_from(&self.params[8 * at..8 * at + B]).expect("validated length")
    }

    fn vector<const N: usize>(&self, at: usize) -> [f64; N] {
        let mut v = [0.0; N];
        for (i, x) in v.iter_mut().enumerate() {
            *x = read_f64(self.params, at + i);
        }
        v
    }

    fn normalise(&self, u: &[[f64; DOF]; 3]) -> [[f64; DOF]; 3] {
        let mut x = [[0.0; DOF]; 3];
        for c in 0..3 {
            for i in 0..DOF {
                x[c][i] = (u[c][i] - self.in_mean[c][i]) / self.in_std[c][i];
            }
        }
        x
    }

    fn cell(h: &mut [f64; H], c: &mut [f64; H], g: &[f64; G]) {
        for j in 0..H {
            let i = sigmoid(g[j]);
            let f = sigmoid(g[H + j]);
            let z = libm::tanh(g[2 * H + j]);
            let o = sigmoid(g[3 * H + j]);
            c[j] = f * c[j] + i * z;
            h[j] = o * libm::tanh(c[j]);
        }
    }

    /// Advance `s` by one normalised input frame.
    fn frame(&self, s: &mut LstmState, x: &[[f64; DOF]; 3]) {
        let l = Self::L;
        let mut g: [f64; G] = self.vector(l.b0);
        for (k, v) in x.iter().flatten().enumerate() {
            axpy(&mut g, self.col::<{ 8 * G }>(l.w_ih0 + k * G), *v);
        }
        for k in 0..H {
            axpy(&mut g, self.col::<{ 8 * G }>(l.w_hh0 + k * G), s.h[0][k]);
        }
        let [h0, h1] = &mut s.h;
        let [c0, c1] = &mut s.c;
        Self::cell(h0, c0, &g);

        let mut g: [f64; G] = self.vector(l.b1);
        for k in 0..H {
            axpy(&mut g, self.col::<{ 8 * G }>(l.w_ih1 + k * G), h0[k]);
        }
        for k in 0..H {
            axpy(&mut g, self.col::<{ 8 * G }>(l.w_hh1 + k * G), h1[k]);
        }
        Self::cell(h1, c1, &g);
    }

    /// The MLP head on the top layer's hidden state, de-normalised.
    fn head(&self, h: &[f64; H]) -> [f64; DOF] {
        let l = Self::L;
        let mut z: [f64; M] = self.vector(l.b_mlp);
        for (k, hk) in h.iter().enumerate() {
            axpy(&mut z, self.col::<{ 8 * M }>(l.w_mlp + k * M), *hk);
        }
        let mut y: [f64; DOF] = self.vector(l.b_out);
        for (k, zk) in z.iter().enumerate() {
            let zk = zk.max(0.0);
            for (r, yr) in y.iter_mut().enumerate() {
                *yr += read_f64(self.params, l.w_out + k * DOF + r) * zk;
            }
        }
        for i in 0..DOF {
            y[i] = y[i] * self.out_std[i] + self.out_mean[i];
        }
        y
    }
}

/// A loaded `.dexnext` observer and its sliding-window state.
///
/// Holds about 12 KiB of state for six joints (the window and one LSTM
/// state); the weights stay in the `&'static` file bytes.
#[derive(Clone)]
pub struct LstmObserver<const DOF: usize> {
    net: Net<DOF>,
    mode: ObserverMode,
    trained_gravity: [f64; 3],
    trained_payload: Payload,
    /// The newest [`NEXT_WINDOW`] normalised samples; `newest` indexes the latest.
    ring: [[[f64; DOF]; 3]; NEXT_WINDOW],
    newest: usize,
    /// LSTM state after the first `pre_done` frames of the next window.
    pre: LstmState,
    pre_done: usize,
    /// Tick within the stride period; 0 is a sample tick.
    phase: usize,
    started: bool,
    warming: bool,
    output: Option<[f64; DOF]>,
}

impl<const DOF: usize> core::fmt::Debug for LstmObserver<DOF> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LstmObserver")
            .field("dof", &DOF)
            .field("mode", &self.mode)
            .field("trained_gravity", &self.trained_gravity)
            .field("trained_payload", &self.trained_payload)
            .field("output", &self.output)
            .finish_non_exhaustive()
    }
}

impl<const DOF: usize> LstmObserver<DOF> {
    /// Validate and load a `.dexnext` file (see the [module docs](self)).
    ///
    /// [`ModelError::InvalidObserver`] unless the magic, `DOF`, mode and
    /// reserved bytes match, the length is exactly [`next_file_len`], every
    /// value is finite, the input scales exceed `1e-12`, the output scales
    /// are positive and the trained payload is valid.
    pub fn from_bytes(bytes: &'static [u8]) -> Result<Self, ModelError> {
        let bad = Err(ModelError::InvalidObserver);
        if bytes.len() != next_file_len(DOF)
            || bytes[0..8] != NEXT_MAGIC
            || usize::from(bytes[8]) != DOF
            || bytes[10..16].iter().any(|b| *b != 0)
        {
            return bad;
        }
        let mode = match bytes[9] {
            0 => ObserverMode::FullDynamics,
            1 => ObserverMode::FrictionResidual,
            _ => return bad,
        };
        let header = |i: usize| read_f64(&bytes[16..NEXT_HEADER_LEN], i);
        let trained_gravity = [header(0), header(1), header(2)];
        let mut inertia = [0.0; 9];
        for (k, v) in inertia.iter_mut().enumerate() {
            *v = header(7 + k);
        }
        if !trained_gravity.iter().all(|g| g.is_finite()) {
            return bad;
        }
        let Ok(trained_payload) =
            Payload::from_contract(header(3), [header(4), header(5), header(6)], inertia)
        else {
            return bad;
        };

        let params = &bytes[NEXT_HEADER_LEN..];
        if !params
            .chunks_exact(8)
            .all(|w| f64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]).is_finite())
        {
            return bad;
        }
        let l = Layout::new(DOF);
        let mut net = Net {
            params,
            in_mean: [[0.0; DOF]; 3],
            in_std: [[0.0; DOF]; 3],
            out_mean: [0.0; DOF],
            out_std: [0.0; DOF],
        };
        for c in 0..3 {
            for i in 0..DOF {
                net.in_mean[c][i] = read_f64(params, l.in_mean + c * DOF + i);
                net.in_std[c][i] = read_f64(params, l.in_std + c * DOF + i);
            }
        }
        net.out_mean = net.vector(l.out_mean);
        net.out_std = net.vector(l.out_std);
        if net.in_std.iter().flatten().any(|s| *s <= 1e-12) || net.out_std.iter().any(|s| *s <= 0.0) {
            return bad;
        }
        Ok(LstmObserver {
            net,
            mode,
            trained_gravity,
            trained_payload,
            ring: [[[0.0; DOF]; 3]; NEXT_WINDOW],
            newest: 0,
            pre: ZERO_STATE,
            pre_done: 0,
            phase: 0,
            started: false,
            warming: false,
            output: None,
        })
    }

    /// What the network predicts.
    pub fn mode(&self) -> ObserverMode {
        self.mode
    }

    /// Gravity (m/s², base frame) the model was trained under.
    pub fn trained_gravity(&self) -> &[f64; 3] {
        &self.trained_gravity
    }

    /// Payload the arm held while the training data was collected.
    pub fn trained_payload(&self) -> &Payload {
        &self.trained_payload
    }

    /// Latest network output (normalisation undone), `None` until the first
    /// window after a reset completes.
    pub fn last_output(&self) -> Option<[f64; DOF]> {
        if self.warming {
            None
        } else {
            self.output
        }
    }

    /// Control ticks per network sample at `control_freq_hz`:
    /// `round(control_freq_hz / 100)`, at least 1.
    pub const fn stride_ticks(control_freq_hz: u32) -> usize {
        let s = (control_freq_hz as usize + NEXT_SAMPLE_RATE_HZ as usize / 2)
            / NEXT_SAMPLE_RATE_HZ as usize;
        if s == 0 {
            1
        } else {
            s
        }
    }

    /// Forget the window; the next [`step`](Self::step) starts a new warm-up.
    pub fn reset(&mut self) {
        self.pre = ZERO_STATE;
        self.pre_done = 0;
        self.phase = 0;
        self.started = false;
        self.warming = false;
        self.output = None;
    }

    /// Advance one control tick with the raw input `[q, qd, q_cmd − q]`;
    /// every `stride` ticks (see [`stride_ticks`](Self::stride_ticks)) the
    /// input is sampled and a new window evaluated, and the output is held
    /// in between.
    ///
    /// `None` while warming up after a reset (at most `stride − 1` ticks),
    /// or if the input is not finite (the observer is not advanced).
    pub fn step(&mut self, u: &[[f64; DOF]; 3], stride: usize) -> Option<[f64; DOF]> {
        if !u.iter().flatten().all(|v| v.is_finite()) {
            return None;
        }
        let stride = stride.max(1);
        if self.phase >= stride {
            self.phase = 0;
        }
        if self.phase == 0 {
            let x = self.net.normalise(u);
            if !self.started {
                // Left-pad the first window with the first sample.
                self.ring = [x; NEXT_WINDOW];
                self.newest = 0;
                self.pre = ZERO_STATE;
                self.pre_done = 0;
                self.started = true;
                self.warming = true;
            } else {
                self.advance_prefix(PREFIX - self.pre_done);
                self.newest = (self.newest + 1) % NEXT_WINDOW;
                self.ring[self.newest] = x;
                self.net.frame(&mut self.pre, &x);
                self.output = Some(self.net.head(&self.pre.h[NEXT_LAYERS - 1]));
                self.pre = ZERO_STATE;
                self.pre_done = 0;
            }
        }

        let ticks_left = stride - self.phase;
        if self.warming {
            // The prefix, then the final frame (the first sample again).
            let work = (PREFIX - self.pre_done + 1).div_ceil(ticks_left);
            let prefix = work.min(PREFIX - self.pre_done);
            self.advance_prefix(prefix);
            if work > prefix {
                let mut s = self.pre;
                self.net.frame(&mut s, &self.ring[self.newest]);
                self.output = Some(self.net.head(&s.h[NEXT_LAYERS - 1]));
                // The second window's prefix is the same 49 padded samples.
                self.warming = false;
            }
        } else {
            self.advance_prefix((PREFIX - self.pre_done).div_ceil(ticks_left));
        }

        self.phase += 1;
        if self.phase == stride {
            self.phase = 0;
        }
        self.last_output()
    }

    /// Run `n` more frames of the next window's prefix: the 49 newest samples,
    /// oldest first.
    fn advance_prefix(&mut self, n: usize) {
        for _ in 0..n {
            let age = PREFIX - 1 - self.pre_done;
            let at = (self.newest + NEXT_WINDOW - age) % NEXT_WINDOW;
            self.net.frame(&mut self.pre, &self.ring[at]);
            self.pre_done += 1;
        }
    }
}

/// Builders and a naive reference for tests (shared with `.dexnext` tooling
/// in `codegen/next`, which reproduces the same parameters bit for bit).
#[cfg(test)]
pub mod test_support {
    extern crate std;

    use super::*;
    use std::boxed::Box;
    use std::vec;
    use std::vec::Vec;

    /// The deterministic generator shared with `codegen/next/dexnext.py`.
    pub struct Lcg(u64);

    impl Lcg {
        pub fn new(seed: u64) -> Lcg {
            Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }

        /// Uniform in `[-1, 1)`.
        pub fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// A synthetic six-joint observer.
    #[derive(Clone, Copy, Debug)]
    pub struct Spec {
        pub mode: ObserverMode,
        pub gravity: [f64; 3],
        pub payload: Payload,
    }

    pub fn spec() -> Spec {
        Spec {
            mode: ObserverMode::FullDynamics,
            gravity: [0.0, 0.0, -crate::math::arm::STANDARD_GRAVITY],
            payload: Payload::NONE,
        }
    }

    /// Parameters for `dof` joints drawn from `seed`, one draw each in file
    /// order: weights and biases `0.1·u`, `in_mean = u`,
    /// `in_std = 0.5 + 0.4·(u + 1)`, `out_mean = 10·u`, `out_std = 1 + 2·(u + 1)`.
    pub fn params(dof: usize, seed: u64) -> Vec<f64> {
        let l = Layout::new(dof);
        let mut rng = Lcg::new(seed);
        (0..l.end)
            .map(|i| {
                let u = rng.unit();
                if i < l.in_std {
                    u
                } else if i < l.w_ih0 {
                    0.5 + 0.4 * (u + 1.0)
                } else if i < l.out_mean {
                    0.1 * u
                } else if i < l.out_std {
                    10.0 * u
                } else {
                    1.0 + 2.0 * (u + 1.0)
                }
            })
            .collect()
    }

    impl Spec {
        pub fn encode_params(&self, dof: usize, params: &[f64]) -> Vec<u8> {
            let mut out = Vec::with_capacity(next_file_len(dof));
            out.extend_from_slice(&NEXT_MAGIC);
            out.push(dof as u8);
            out.push(self.mode as u8);
            out.extend_from_slice(&[0; 6]);
            let p = &self.payload;
            let header = self
                .gravity
                .iter()
                .chain(core::iter::once(&p.mass))
                .chain(p.com.iter())
                .chain(p.inertia.iter().flatten());
            for v in header.chain(params) {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }

        pub fn encode(&self, seed: u64) -> Vec<u8> {
            self.encode_params(6, &params(6, seed))
        }

        pub fn leak(&self, seed: u64) -> &'static [u8] {
            Box::leak(self.encode(seed).into_boxed_slice())
        }
    }

    /// Straightforward evaluation of one window from decoded parameters.
    pub struct Reference {
        dof: usize,
        p: Vec<f64>,
    }

    impl Reference {
        pub fn new(dof: usize, p: Vec<f64>) -> Reference {
            Reference { dof, p }
        }

        fn cell(p: &[f64], h: &mut [f64], c: &mut [f64], x: &[f64], w_ih: usize, w_hh: usize, b: usize) {
            let mut g = vec![0.0; G];
            for r in 0..G {
                let mut acc = p[b + r];
                for (k, xk) in x.iter().enumerate() {
                    acc += p[w_ih + k * G + r] * xk;
                }
                for (k, hk) in h.iter().enumerate() {
                    acc += p[w_hh + k * G + r] * hk;
                }
                g[r] = acc;
            }
            let sig = |v: f64| 1.0 / (1.0 + libm::exp(-v));
            for j in 0..H {
                c[j] = sig(g[H + j]) * c[j] + sig(g[j]) * libm::tanh(g[2 * H + j]);
                h[j] = sig(g[3 * H + j]) * libm::tanh(c[j]);
            }
        }

        /// Output for `window` (raw `[q, qd, q_cmd − q]` rows, oldest first).
        pub fn window(&self, window: &[Vec<f64>]) -> Vec<f64> {
            let (d, p) = (self.dof, &self.p);
            let l = Layout::new(d);
            let (mut h0, mut c0, mut h1, mut c1) = (vec![0.0; H], vec![0.0; H], vec![0.0; H], vec![0.0; H]);
            for u in window {
                let x: Vec<f64> = (0..3 * d).map(|i| (u[i] - p[l.in_mean + i]) / p[l.in_std + i]).collect();
                Self::cell(p, &mut h0, &mut c0, &x, l.w_ih0, l.w_hh0, l.b0);
                let below = h0.clone();
                Self::cell(p, &mut h1, &mut c1, &below, l.w_ih1, l.w_hh1, l.b1);
            }
            let z: Vec<f64> = (0..M)
                .map(|r| {
                    let mut acc = p[l.b_mlp + r];
                    for (k, hk) in h1.iter().enumerate() {
                        acc += p[l.w_mlp + k * M + r] * hk;
                    }
                    acc.max(0.0)
                })
                .collect();
            (0..d)
                .map(|r| {
                    let mut acc = p[l.b_out + r];
                    for (k, zk) in z.iter().enumerate() {
                        acc += p[l.w_out + k * d + r] * zk;
                    }
                    acc * p[l.out_std + r] + p[l.out_mean + r]
                })
                .collect()
        }

        /// Outputs of every window over `samples`, left-padded with the first.
        pub fn windows(&self, samples: &[Vec<f64>]) -> Vec<Vec<f64>> {
            (0..samples.len())
                .map(|n| {
                    let w: Vec<Vec<f64>> = (0..NEXT_WINDOW)
                        .map(|j| samples[(n + j).saturating_sub(PREFIX)].clone())
                        .collect();
                    self.window(&w)
                })
                .collect()
        }
    }

    /// Deterministic raw inputs `[q, qd, q_cmd − q]` (`3·dof` each) shared
    /// with `codegen/next/dexnext.py::parity_inputs`.
    pub fn parity_inputs(dof: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                (0..3 * dof)
                    .map(|i| {
                        let scale = [1.5, 0.8, 0.05][i / dof];
                        scale * rng.unit()
                    })
                    .collect()
            })
            .collect()
    }

    pub fn rows<const DOF: usize>(u: &[f64]) -> [[f64; DOF]; 3] {
        let mut x = [[0.0; DOF]; 3];
        for c in 0..3 {
            x[c].copy_from_slice(&u[c * DOF..(c + 1) * DOF]);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::test_support::*;
    use super::*;
    use std::vec::Vec;

    #[test]
    fn the_layout_matches_the_paper_architecture() {
        // 2×128 LSTM over 18 inputs, 128→256→6 head, plus normalisation.
        let lstm = (18 * 512 + 128 * 512 + 512) + (2 * 128 * 512 + 512);
        let head = 128 * 256 + 256 + 256 * 6 + 6;
        assert_eq!(next_param_count(6), 2 * 18 + lstm + head + 2 * 6);
        assert_eq!(spec().encode(1).len(), next_file_len(6));
        assert_eq!(LstmObserver::<6>::stride_ticks(1000), 10);
        assert_eq!(LstmObserver::<6>::stride_ticks(500), 5);
        assert_eq!(LstmObserver::<6>::stride_ticks(100), 1);
        assert_eq!(LstmObserver::<6>::stride_ticks(30), 1);
    }

    /// Run `samples` (one per `stride` ticks) and return the output of every tick.
    fn run(obs: &mut LstmObserver<6>, samples: &[Vec<f64>], stride: usize) -> Vec<Option<[f64; 6]>> {
        let mut out = Vec::new();
        for u in samples {
            for _ in 0..stride {
                out.push(obs.step(&rows::<6>(u), stride));
            }
        }
        out
    }

    #[test]
    fn the_spread_window_matches_the_reference_at_every_stride() {
        let seed = 7;
        let bytes = spec().leak(seed);
        let reference = Reference::new(6, params(6, seed));
        let samples = parity_inputs(6, 60, 3);
        let expected = reference.windows(&samples);
        let mut exact: Option<Vec<[f64; 6]>> = None;
        for stride in [1, 3, 10] {
            let mut obs = LstmObserver::<6>::from_bytes(bytes).unwrap();
            let ticks = run(&mut obs, &samples, stride);
            let mut windows = Vec::new();
            for (t, y) in ticks.iter().enumerate() {
                let n = t / stride;
                if n == 0 && t + 1 < stride {
                    assert_eq!(*y, None, "stride {stride} tick {t} is warming up");
                    continue;
                }
                let y = y.unwrap();
                for i in 0..6 {
                    assert!((y[i] - expected[n][i]).abs() < 1e-10, "stride {stride} tick {t}");
                }
                if t % stride == 0 || n == 0 {
                    windows.push(y);
                }
            }
            // Spreading the work over the stride changes no bit.
            match &exact {
                None => exact = Some(windows),
                Some(w) => assert_eq!(w, &windows, "stride {stride}"),
            }
        }
    }

    #[test]
    fn reset_restarts_the_warm_up() {
        let bytes = spec().leak(4);
        let samples = parity_inputs(6, 8, 5);
        let mut obs = LstmObserver::<6>::from_bytes(bytes).unwrap();
        let first = run(&mut obs, &samples, 5);
        assert!(obs.last_output().is_some());
        obs.reset();
        assert_eq!(obs.last_output(), None);
        // Mid-period reset too: the phase restarts.
        obs.step(&rows::<6>(&samples[0]), 5);
        obs.reset();
        assert_eq!(run(&mut obs, &samples, 5), first);

        let mut bad = rows::<6>(&samples[0]);
        bad[1][3] = f64::NAN;
        let before = obs.clone();
        assert_eq!(obs.step(&bad, 5), None);
        assert_eq!(obs.phase, before.phase);
        assert_eq!(obs.newest, before.newest);
    }

    #[test]
    fn malformed_files_are_rejected() {
        let good = spec().encode(2);
        assert!(LstmObserver::<6>::from_bytes(std::boxed::Box::leak(good.clone().into_boxed_slice())).is_ok());
        let reject = |bytes: Vec<u8>| {
            let bytes: &'static [u8] = std::boxed::Box::leak(bytes.into_boxed_slice());
            assert_eq!(LstmObserver::<6>::from_bytes(bytes).unwrap_err(), ModelError::InvalidObserver);
        };
        let patch = |at: usize, v: &[u8]| {
            let mut b = good.clone();
            b[at..at + v.len()].copy_from_slice(v);
            b
        };
        let l = Layout::new(6);
        let param = |i: usize| NEXT_HEADER_LEN + 8 * i;

        reject(patch(7, &[1])); // version 1
        reject(patch(0, b"X"));
        reject(patch(8, &[7])); // dof
        reject(patch(9, &[2])); // mode
        reject(patch(12, &[1])); // reserved
        reject(good[..good.len() - 8].to_vec());
        reject([good.as_slice(), &[0; 8]].concat());
        reject(patch(16, &f64::NAN.to_le_bytes())); // gravity
        reject(patch(40, &(-1.0f64).to_le_bytes())); // payload mass
        reject(patch(param(l.w_hh1 + 9), &f64::INFINITY.to_le_bytes()));
        reject(patch(param(l.in_std + 4), &1e-13f64.to_le_bytes()));
        reject(patch(param(l.out_std + 5), &0.0f64.to_le_bytes()));
        assert_eq!(
            LstmObserver::<7>::from_bytes(std::boxed::Box::leak(good.into_boxed_slice())).unwrap_err(),
            ModelError::InvalidObserver
        );
    }

    #[test]
    fn the_header_round_trips() {
        let mut s = spec();
        s.mode = ObserverMode::FrictionResidual;
        s.gravity = [0.1, -0.2, -9.7];
        s.payload = Payload::new(1.5, [0.0, 0.01, 0.08]).with_inertia([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03]]);
        let obs = LstmObserver::<6>::from_bytes(s.leak(3)).unwrap();
        assert_eq!(obs.mode(), ObserverMode::FrictionResidual);
        assert_eq!(obs.trained_gravity(), &s.gravity);
        assert_eq!(obs.trained_payload(), &s.payload);
        assert_eq!(obs.last_output(), None);
    }

    /// PyTorch's float64 outputs for the weights `params(6, PARITY_SEED)` over
    /// `parity_inputs(6, PARITY_SAMPLES, PARITY_INPUT_SEED)`, written by
    /// `codegen/next/make_fixtures.py`.
    const PARITY_SEED: u64 = 2026;
    const PARITY_INPUT_SEED: u64 = 59;
    const PARITY_SAMPLES: usize = 100;
    static PARITY_EXPECTED: &[u8] = include_bytes!("../../tests/fixtures/next/parity.expected");

    #[test]
    fn pytorch_parity() {
        assert_eq!(PARITY_EXPECTED.len(), PARITY_SAMPLES * 6 * 8);
        let bytes = spec().leak(PARITY_SEED);
        let samples = parity_inputs(6, PARITY_SAMPLES, PARITY_INPUT_SEED);
        let mut obs = LstmObserver::<6>::from_bytes(bytes).unwrap();
        // A 100 Hz model in a 1 kHz loop.
        let ticks = run(&mut obs, &samples, 10);
        for n in 0..PARITY_SAMPLES {
            let y = ticks[if n == 0 { 9 } else { 10 * n }].unwrap();
            for i in 0..6 {
                let torch = read_f64(PARITY_EXPECTED, n * 6 + i);
                assert!((y[i] - torch).abs() < 1e-10, "window {n} joint {i}: {} vs {torch}", y[i]);
            }
        }
    }
}
