//! Capability manifest: the static description of an arm, read once at boot.
//!
//! A manifest combines what the driver's model knows (name, DOF, position
//! limits, model capabilities) with what only the OEM knows (rates, modes,
//! latencies, sensing, joint ratings). The OEM part is a data file:
//!
//! ```
//! # use dexhal::math::contract::{control_mode, ArmManifest};
//! # use dexhal::math::model::{ArmModel, Pose};
//! # use dexhal::math::tolerances::TolerancePolicy;
//! # struct Demo;
//! # impl ArmModel<2, 1> for Demo {
//! #     fn flange_fk(_q: &[f64; 2]) -> Pose { Pose::identity() }
//! #     fn flange_jacobian(_q: &[f64; 2]) -> [[f64; 2]; 6] { [[0.0; 2]; 6] }
//! #     fn flange_solve(_t: &Pose, _m: Option<usize>, _s: Option<&[f64; 2]>, _l: bool, _r: bool,
//! #         _p: TolerancePolicy, _i: usize) -> [Option<[f64; 2]>; 1] { [None] }
//! # }
//! // In a driver: include_str!("../manifest.yaml")
//! static MANIFEST: &str = "
//! driver: demo
//! version: \"0.2\"
//! control_freq_hz: 500
//! control_modes: [joint_pos, joint_vel]
//! latency:
//!   state_staleness_cycles: 1
//!   command_dispatch_cycles: 2
//! limits:
//!   velocity: [3.1, 3.1]
//!   acceleration: [15.0, 15.0]
//!   effort: [150.0, 28.0]
//! ";
//! let manifest = ArmManifest::from_text::<Demo, 2, 1>(MANIFEST).unwrap();
//! assert!(manifest.supports(control_mode::JOINT_POS));
//! assert_eq!(manifest.effort_limits[1], 28.0);
//! ```
//!
//! # File format
//!
//! A strict subset of YAML: `key: value` lines, one level of indented
//! sections, `[a, b]` lists, `#` comments. Unknown or repeated keys are
//! errors. `examples/manifest.yaml` documents every key.
//!
//! | Key | Required | Meaning |
//! | :-- | :-- | :-- |
//! | `driver` | yes | Arm name; must equal [`ArmModel::NAME`] when the model sets one |
//! | `version` | yes | Contract version `"major.minor"` the driver was written against |
//! | `control_freq_hz` | yes | Command rate |
//! | `control_modes` | yes | List of `joint_pos`, `joint_vel`, `joint_torque`, `cartesian_pose`, `cartesian_twist`, `cartesian_wrench` |
//! | `realtime` | no | Needs a real-time thread (default `false`) |
//! | `latency.state_staleness_cycles`, `latency.command_dispatch_cycles` | yes | Delays in control cycles |
//! | `limits.velocity`, `limits.acceleration`, `limits.effort` | yes | One positive finite value per joint (rad/s, rad/s², N·m) |
//! | `limits.jerk` | no | One positive finite value per joint (rad/s³); `NAN` when omitted so the kernel applies its ramp-time fallback |
//! | `limits.position_min`, `limits.position_max` | no | Override [`ArmModel::JOINT_LIMITS`] (rad) |
//! | `has_static_inertials` | no | Static link mass / inertia model is trustworthy for host impedance (default [`ArmModel::HAS_DYNAMICS`]) |
//! | `sensing.torque`, `.wrench`, `.gripper` | no | Hardware sensing (default `false`) |
//! | `native.cartesian_impedance`, `.cartesian_admittance`, `.joint_impedance`, `.joint_admittance`, `.contact_detection`, `.reflexes`, `.payload`, `.tcp_offset` | no | Features the controller executes itself (default `false`) |
//! | `aux` | no | List of `- name: … / type: f64 / count: …` items following [`CoreState`] in the state buffer |

use super::{control_mode, ContractVersion, CoreCommand, CoreState, CONTRACT_VERSION, MAX_DOF};
use crate::math::chain::ModelError;
use crate::math::model::ArmModel;

/// Capacity of [`AuxSchema`].
pub const MAX_AUX_FIELDS: usize = 16;

/// One field of the driver-specific tail following [`CoreState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuxField {
    /// e.g. `"motor_temp"`.
    pub label: &'static str,
    /// Number of `f64` slots (1 for scalars).
    pub size: u32,
}

/// Layout of the state tail: up to [`MAX_AUX_FIELDS`] fields, stored inline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuxSchema {
    fields: [AuxField; MAX_AUX_FIELDS],
    len: usize,
}

impl AuxSchema {
    /// No aux tail.
    pub const EMPTY: AuxSchema = AuxSchema {
        fields: [AuxField { label: "", size: 0 }; MAX_AUX_FIELDS],
        len: 0,
    };

    /// Copy `fields`; `None` if there are more than [`MAX_AUX_FIELDS`].
    pub fn from_slice(fields: &[AuxField]) -> Option<AuxSchema> {
        let mut out = AuxSchema::EMPTY;
        for f in fields {
            out.push(*f)?;
        }
        Some(out)
    }

    fn push(&mut self, field: AuxField) -> Option<()> {
        *self.fields.get_mut(self.len)? = field;
        self.len += 1;
        Some(())
    }

    pub fn fields(&self) -> &[AuxField] {
        &self.fields[..self.len]
    }

    /// Total number of `f64` slots.
    pub fn slots(&self) -> usize {
        self.fields().iter().map(|f| f.size as usize).sum()
    }
}

/// What the host-side model of an arm can compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// Number of actuated joints.
    pub dof: u8,
    /// Capacity of the driver's IK solution set.
    pub max_ik_solutions: u16,
    /// `M`, `C`, `G` and inverse dynamics are available ([`ArmModel::HAS_DYNAMICS`]).
    pub dynamics: bool,
}

/// Per-joint ratings no model can default safely. All values must be finite
/// and strictly positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointRatings<const DOF: usize> {
    /// Maximum joint speed (rad/s).
    pub velocity: [f64; DOF],
    /// Maximum joint acceleration (rad/s²).
    pub acceleration: [f64; DOF],
    /// Maximum joint torque (N·m).
    pub effort: [f64; DOF],
}

impl<const DOF: usize> JointRatings<DOF> {
    fn validate(&self) -> Result<(), ModelError> {
        for v in self
            .velocity
            .iter()
            .chain(&self.acceleration)
            .chain(&self.effort)
        {
            if !v.is_finite() {
                return Err(ModelError::NonFinite);
            }
            if *v <= 0.0 {
                return Err(ModelError::NonPositive);
            }
        }
        Ok(())
    }
}

/// Static description of an arm, read once by the kernel at boot.
///
/// Arrays are [`MAX_DOF`]-padded; entries beyond `active_dof_count` are zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmManifest {
    pub name: &'static str,
    pub version: ContractVersion,
    /// Rate (Hz) at which the hardware expects commands.
    pub control_freq_hz: u32,
    /// OR of [`control_mode`] flags executed natively.
    pub control_modes: u32,
    /// The driver needs a real-time (SCHED_FIFO) thread.
    pub is_realtime_required: bool,
    pub active_dof_count: u8,
    /// Layout of the state tail following [`CoreState`].
    pub aux: AuxSchema,
    /// Cycles between the hardware sampling its state and `read_state` returning.
    pub state_staleness_cycles: u32,
    /// Cycles between `write_command` returning and the hardware acting.
    pub command_dispatch_cycles: u32,
    pub has_torque_sensing: bool,
    pub has_wrench_sensing: bool,
    pub has_integrated_gripper: bool,
    pub native_cartesian_impedance: bool,
    pub native_cartesian_admittance: bool,
    pub native_joint_impedance: bool,
    pub native_joint_admittance: bool,
    pub native_contact_detection: bool,
    pub native_reflexes: bool,
    /// The controller accepts payload changes itself.
    pub native_payload: bool,
    /// The controller accepts TCP changes itself.
    pub native_tcp_offset: bool,
    /// Position limits (rad); `±INFINITY` for a continuous joint.
    pub joint_limits_min: [f64; MAX_DOF],
    pub joint_limits_max: [f64; MAX_DOF],
    /// Speed limits (rad/s).
    pub velocity_limits: [f64; MAX_DOF],
    /// Acceleration limits (rad/s²).
    pub acceleration_limits: [f64; MAX_DOF],
    /// Jerk limits (rad/s³); `NAN` when omitted by the manifest.
    pub jerk_limits: [f64; MAX_DOF],
    /// Torque limits (N·m).
    pub effort_limits: [f64; MAX_DOF],
    /// Whether static link inertials are available and accurate enough for
    /// host-side inverse dynamics (`HostImpedance` / torque-residual
    /// admittance). Defaults to [`ArmModel::HAS_DYNAMICS`].
    pub has_static_inertials: bool,
    pub model: ModelCapabilities,
}

fn padded<const DOF: usize>(v: &[f64; DOF]) -> [f64; MAX_DOF] {
    let mut out = [0.0; MAX_DOF];
    out[..DOF].copy_from_slice(v);
    out
}

impl ArmManifest {
    /// Manifest of a bare model: name, DOF, position limits and model
    /// capabilities from `A`, the mandatory `ratings`, and nothing else — no
    /// native features, no sensing, no latency. Drivers then set the public
    /// fields they know, or use [`from_text`](Self::from_text).
    pub fn for_model<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize>(
        control_freq_hz: u32,
        control_modes: u32,
        ratings: &JointRatings<DOF>,
    ) -> Result<ArmManifest, ModelError> {
        const { assert!(DOF <= MAX_DOF) };
        ratings.validate()?;
        if control_freq_hz == 0 {
            return Err(ModelError::NonPositive);
        }
        let mut lo = [0.0; MAX_DOF];
        let mut hi = [0.0; MAX_DOF];
        for i in 0..DOF {
            (lo[i], hi[i]) = A::JOINT_LIMITS[i].unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        }
        Ok(ArmManifest {
            name: A::NAME,
            version: CONTRACT_VERSION,
            control_freq_hz,
            control_modes,
            is_realtime_required: false,
            active_dof_count: DOF as u8,
            aux: AuxSchema::EMPTY,
            state_staleness_cycles: 0,
            command_dispatch_cycles: 0,
            has_torque_sensing: false,
            has_wrench_sensing: false,
            has_integrated_gripper: false,
            native_cartesian_impedance: false,
            native_cartesian_admittance: false,
            native_joint_impedance: false,
            native_joint_admittance: false,
            native_contact_detection: false,
            native_reflexes: false,
            native_payload: false,
            native_tcp_offset: false,
            joint_limits_min: lo,
            joint_limits_max: hi,
            velocity_limits: padded(&ratings.velocity),
            acceleration_limits: padded(&ratings.acceleration),
            jerk_limits: [f64::NAN; MAX_DOF],
            effort_limits: padded(&ratings.effort),
            has_static_inertials: A::HAS_DYNAMICS,
            model: ModelCapabilities {
                dof: DOF as u8,
                max_ik_solutions: MAX as u16,
                dynamics: A::HAS_DYNAMICS,
            },
        })
    }

    /// Whether every flag in `modes` is executed natively.
    pub fn supports(&self, modes: u32) -> bool {
        self.control_modes & modes == modes
    }

    /// Length in `f64` of the state buffer: [`CoreState`] plus the aux tail.
    pub fn state_len(&self) -> usize {
        CoreState::LEN + self.aux.slots()
    }

    /// Length in `f64` of the command buffer.
    pub fn command_len(&self) -> usize {
        CoreCommand::LEN
    }
}

// ---------------------------------------------------------------------------
// Data file
// ---------------------------------------------------------------------------

/// Why a manifest file was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestErrorKind {
    /// The line is not `key: value`, a section header or a list item, or uses
    /// tabs / unsupported nesting.
    Syntax,
    /// The key is not part of the schema (at this nesting level).
    UnknownKey,
    /// The key appears twice.
    DuplicateKey,
    /// A required key is absent (reported at line 0).
    MissingKey(&'static str),
    /// The value has the wrong type or is not a recognised name.
    InvalidValue,
    /// A per-joint list does not have exactly `DOF` entries.
    WrongLength,
    /// `driver` differs from [`ArmModel::NAME`].
    ModelMismatch,
    /// `version` is not compatible with [`CONTRACT_VERSION`].
    VersionMismatch,
    /// More than [`MAX_AUX_FIELDS`] aux fields, or an incomplete aux item.
    InvalidAux,
    /// A value was rejected by the model (e.g. a non-positive rating).
    Model(ModelError),
}

/// A manifest file error with its 1-based line number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManifestError {
    pub line: u32,
    pub kind: ManifestErrorKind,
}

type Parsed<T> = Result<T, ManifestErrorKind>;

/// Remove a trailing comment and surrounding whitespace.
fn strip(line: &str) -> &str {
    let mut end = line.len();
    let mut prev_space = true;
    for (i, c) in line.char_indices() {
        if c == '#' && prev_space {
            end = i;
            break;
        }
        prev_space = c == ' ';
    }
    line[..end].trim_end()
}

fn split_key(s: &str) -> Parsed<(&str, &str)> {
    let (key, value) = s.split_once(':').ok_or(ManifestErrorKind::Syntax)?;
    let key = key.trim();
    if key.is_empty() || key.contains(' ') {
        return Err(ManifestErrorKind::Syntax);
    }
    Ok((key, value.trim()))
}

fn unquote(s: &str) -> &str {
    s.strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or(s)
}

fn parse_bool(s: &str) -> Parsed<bool> {
    match s {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ManifestErrorKind::InvalidValue),
    }
}

fn parse_u32(s: &str) -> Parsed<u32> {
    s.parse().map_err(|_| ManifestErrorKind::InvalidValue)
}

fn list_items(s: &str) -> Parsed<impl Iterator<Item = &str>> {
    let inner = s
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or(ManifestErrorKind::InvalidValue)?;
    Ok(inner.split(',').map(str::trim).filter(|v| !v.is_empty()))
}

fn parse_joint_list<const DOF: usize>(s: &str) -> Parsed<[f64; DOF]> {
    let mut out = [0.0; DOF];
    let mut n = 0;
    for item in list_items(s)? {
        let slot = out.get_mut(n).ok_or(ManifestErrorKind::WrongLength)?;
        *slot = match item {
            ".inf" | "+.inf" => f64::INFINITY,
            "-.inf" => f64::NEG_INFINITY,
            _ => item.parse().map_err(|_| ManifestErrorKind::InvalidValue)?,
        };
        n += 1;
    }
    if n != DOF {
        return Err(ManifestErrorKind::WrongLength);
    }
    Ok(out)
}

fn parse_modes(s: &str) -> Parsed<u32> {
    let mut modes = 0;
    for item in list_items(s)? {
        modes |= match item {
            "joint_pos" => control_mode::JOINT_POS,
            "joint_vel" => control_mode::JOINT_VEL,
            "joint_torque" => control_mode::JOINT_TORQUE,
            "cartesian_pose" => control_mode::CARTESIAN_POSE,
            "cartesian_twist" => control_mode::CARTESIAN_TWIST,
            "cartesian_wrench" => control_mode::CARTESIAN_WRENCH,
            _ => return Err(ManifestErrorKind::InvalidValue),
        };
    }
    Ok(modes)
}

fn parse_version(s: &str) -> Parsed<()> {
    let (major, minor) = unquote(s)
        .split_once('.')
        .ok_or(ManifestErrorKind::InvalidValue)?;
    let compatible =
        parse_u32(major)? == CONTRACT_VERSION.major && parse_u32(minor)? == CONTRACT_VERSION.minor;
    compatible
        .then_some(())
        .ok_or(ManifestErrorKind::VersionMismatch)
}

/// Record `key` as seen; `bit` indexes [`REQUIRED`] / the optional keys.
fn mark(seen: &mut u64, bit: u32) -> Parsed<()> {
    if *seen & (1 << bit) != 0 {
        return Err(ManifestErrorKind::DuplicateKey);
    }
    *seen |= 1 << bit;
    Ok(())
}

/// Required keys, in `seen` bit order.
const REQUIRED: [&str; 9] = [
    "driver",
    "version",
    "control_freq_hz",
    "control_modes",
    "latency.state_staleness_cycles",
    "latency.command_dispatch_cycles",
    "limits.velocity",
    "limits.acceleration",
    "limits.effort",
];

struct Parser<const DOF: usize> {
    manifest: ArmManifest,
    ratings: JointRatings<DOF>,
    seen: u64,
    /// An `aux` item being assembled: label, size.
    aux_item: Option<(Option<&'static str>, Option<u32>)>,
}

impl<const DOF: usize> Parser<DOF> {
    fn top_level(&mut self, key: &str, value: &'static str, model_name: &str) -> Parsed<()> {
        let m = &mut self.manifest;
        match key {
            "driver" => {
                mark(&mut self.seen, 0)?;
                let name = unquote(value);
                if name.is_empty() {
                    return Err(ManifestErrorKind::InvalidValue);
                }
                if !model_name.is_empty() && name != model_name {
                    return Err(ManifestErrorKind::ModelMismatch);
                }
                m.name = name;
            }
            "version" => {
                mark(&mut self.seen, 1)?;
                parse_version(value)?;
            }
            "control_freq_hz" => {
                mark(&mut self.seen, 2)?;
                m.control_freq_hz = parse_u32(value)?;
                if m.control_freq_hz == 0 {
                    return Err(ManifestErrorKind::Model(ModelError::NonPositive));
                }
            }
            "control_modes" => {
                mark(&mut self.seen, 3)?;
                m.control_modes = parse_modes(value)?;
            }
            "realtime" => {
                mark(&mut self.seen, 9)?;
                m.is_realtime_required = parse_bool(value)?;
            }
            "has_static_inertials" => {
                mark(&mut self.seen, 24)?;
                m.has_static_inertials = parse_bool(value)?;
            }
            _ => return Err(ManifestErrorKind::UnknownKey),
        }
        Ok(())
    }

    fn in_section(&mut self, section: &str, key: &str, value: &'static str) -> Parsed<()> {
        let m = &mut self.manifest;
        let (bit, flag): (u32, &mut bool) = match (section, key) {
            ("latency", "state_staleness_cycles") => {
                mark(&mut self.seen, 4)?;
                m.state_staleness_cycles = parse_u32(value)?;
                return Ok(());
            }
            ("latency", "command_dispatch_cycles") => {
                mark(&mut self.seen, 5)?;
                m.command_dispatch_cycles = parse_u32(value)?;
                return Ok(());
            }
            ("limits", "velocity") => {
                mark(&mut self.seen, 6)?;
                self.ratings.velocity = parse_joint_list(value)?;
                return Ok(());
            }
            ("limits", "acceleration") => {
                mark(&mut self.seen, 7)?;
                self.ratings.acceleration = parse_joint_list(value)?;
                return Ok(());
            }
            ("limits", "effort") => {
                mark(&mut self.seen, 8)?;
                self.ratings.effort = parse_joint_list(value)?;
                return Ok(());
            }
            ("limits", "jerk") => {
                mark(&mut self.seen, 23)?;
                let jerk = parse_joint_list::<DOF>(value)?;
                for v in &jerk {
                    if !v.is_finite() {
                        return Err(ManifestErrorKind::Model(ModelError::NonFinite));
                    }
                    if *v <= 0.0 {
                        return Err(ManifestErrorKind::Model(ModelError::NonPositive));
                    }
                }
                m.jerk_limits = padded(&jerk);
                return Ok(());
            }
            ("limits", "position_min") => {
                mark(&mut self.seen, 10)?;
                m.joint_limits_min = padded(&parse_joint_list::<DOF>(value)?);
                return Ok(());
            }
            ("limits", "position_max") => {
                mark(&mut self.seen, 11)?;
                m.joint_limits_max = padded(&parse_joint_list::<DOF>(value)?);
                return Ok(());
            }
            ("sensing", "torque") => (12, &mut m.has_torque_sensing),
            ("sensing", "wrench") => (13, &mut m.has_wrench_sensing),
            ("sensing", "gripper") => (14, &mut m.has_integrated_gripper),
            ("native", "cartesian_impedance") => (15, &mut m.native_cartesian_impedance),
            ("native", "cartesian_admittance") => (16, &mut m.native_cartesian_admittance),
            ("native", "joint_impedance") => (17, &mut m.native_joint_impedance),
            ("native", "joint_admittance") => (18, &mut m.native_joint_admittance),
            ("native", "contact_detection") => (19, &mut m.native_contact_detection),
            ("native", "reflexes") => (20, &mut m.native_reflexes),
            ("native", "payload") => (21, &mut m.native_payload),
            ("native", "tcp_offset") => (22, &mut m.native_tcp_offset),
            _ => return Err(ManifestErrorKind::UnknownKey),
        };
        mark(&mut self.seen, bit)?;
        *flag = parse_bool(value)?;
        Ok(())
    }

    fn aux_line(&mut self, body: &'static str) -> Parsed<()> {
        let body = match body.strip_prefix('-') {
            Some(rest) => {
                self.finish_aux_item()?;
                self.aux_item = Some((None, None));
                rest.trim_start()
            }
            None => body,
        };
        let item = self.aux_item.as_mut().ok_or(ManifestErrorKind::Syntax)?;
        if body.is_empty() {
            return Ok(());
        }
        let (key, value) = split_key(body)?;
        match key {
            "name" if item.0.is_none() => item.0 = Some(unquote(value)),
            "count" if item.1.is_none() => item.1 = Some(parse_u32(value)?),
            "type" if value == "f64" => {}
            "name" | "count" => return Err(ManifestErrorKind::DuplicateKey),
            "type" => return Err(ManifestErrorKind::InvalidValue),
            _ => return Err(ManifestErrorKind::UnknownKey),
        }
        Ok(())
    }

    fn finish_aux_item(&mut self) -> Parsed<()> {
        match self.aux_item.take() {
            None => Ok(()),
            Some((Some(label), Some(size))) if !label.is_empty() && size > 0 => self
                .manifest
                .aux
                .push(AuxField { label, size })
                .ok_or(ManifestErrorKind::InvalidAux),
            Some(_) => Err(ManifestErrorKind::InvalidAux),
        }
    }
}

impl ArmManifest {
    /// Build the manifest of model `A` from a manifest data file (see the
    /// [module docs](self) for the format).
    ///
    /// Allocation-free; meant to run once at start-up on text embedded with
    /// `include_str!`.
    pub fn from_text<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize>(
        text: &'static str,
    ) -> Result<ArmManifest, ManifestError> {
        let placeholder = JointRatings {
            velocity: [1.0; DOF],
            acceleration: [1.0; DOF],
            effort: [1.0; DOF],
        };
        let at = |line: u32| move |kind| ManifestError { line, kind };
        let mut parser = Parser {
            manifest: ArmManifest::for_model::<A, DOF, MAX>(1, 0, &placeholder)
                .map_err(ManifestErrorKind::Model)
                .map_err(at(0))?,
            ratings: placeholder,
            seen: 0,
            aux_item: None,
        };
        let mut section: Option<&str> = None;
        let mut sections_seen = 0u64;
        let mut last_line = 0;
        for (index, raw) in text.lines().enumerate() {
            let line = index as u32 + 1;
            last_line = line;
            let content = strip(raw);
            if content.is_empty() {
                continue;
            }
            let body = content.trim_start_matches(' ');
            if body.starts_with('\t') {
                return Err(at(line)(ManifestErrorKind::Syntax));
            }
            let indented = body.len() != content.len();
            let result = if !indented && !body.starts_with('-') {
                parser.finish_aux_item().and_then(|()| {
                    let (key, value) = split_key(body)?;
                    if !value.is_empty() {
                        section = None;
                        return parser.top_level(key, value, A::NAME);
                    }
                    let bit = match key {
                        "latency" => 0,
                        "limits" => 1,
                        "sensing" => 2,
                        "native" => 3,
                        "aux" => 4,
                        _ => return Err(ManifestErrorKind::UnknownKey),
                    };
                    section = Some(key);
                    mark(&mut sections_seen, bit)
                })
            } else {
                match section {
                    Some("aux") => parser.aux_line(body),
                    Some(name) if indented => {
                        split_key(body).and_then(|(key, value)| parser.in_section(name, key, value))
                    }
                    _ => Err(ManifestErrorKind::Syntax),
                }
            };
            result.map_err(at(line))?;
        }
        parser.finish_aux_item().map_err(at(last_line))?;
        for (bit, key) in REQUIRED.iter().enumerate() {
            if parser.seen & (1 << bit) == 0 {
                return Err(at(0)(ManifestErrorKind::MissingKey(key)));
            }
        }
        let checked = ArmManifest::for_model::<A, DOF, MAX>(1, 0, &parser.ratings)
            .map_err(|e| at(0)(ManifestErrorKind::Model(e)))?;
        let mut manifest = parser.manifest;
        manifest.velocity_limits = checked.velocity_limits;
        manifest.acceleration_limits = checked.acceleration_limits;
        manifest.effort_limits = checked.effort_limits;
        for i in 0..DOF {
            let (lo, hi) = (manifest.joint_limits_min[i], manifest.joint_limits_max[i]);
            if lo.is_nan() || hi.is_nan() || lo > hi {
                return Err(at(0)(ManifestErrorKind::InvalidValue));
            }
        }
        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::testing::SixR;
    use core::f64::consts::PI;

    const EXAMPLE: &str = include_str!("../../../examples/manifest.yaml");

    fn ratings() -> JointRatings<6> {
        JointRatings {
            velocity: [3.0; 6],
            acceleration: [15.0; 6],
            effort: [150.0, 150.0, 150.0, 28.0, 28.0, 28.0],
        }
    }

    fn kind(text: &'static str) -> (u32, ManifestErrorKind) {
        let e = ArmManifest::from_text::<SixR, 6, 1>(text).unwrap_err();
        (e.line, e.kind)
    }

    #[test]
    fn for_model_reflects_the_model_and_requires_ratings() {
        let modes = control_mode::JOINT_POS | control_mode::JOINT_VEL;
        let mut manifest = ArmManifest::for_model::<SixR, 6, 1>(500, modes, &ratings()).unwrap();
        manifest.aux = AuxSchema::from_slice(&[
            AuxField {
                label: "motor_temp",
                size: 6,
            },
            AuxField {
                label: "bus_voltage",
                size: 1,
            },
        ])
        .unwrap();
        assert_eq!((manifest.name, manifest.active_dof_count), ("six-r", 6));
        assert_eq!(
            (manifest.joint_limits_min[0], manifest.joint_limits_max[0]),
            (-PI, PI)
        );
        assert_eq!(manifest.joint_limits_max[5], f64::INFINITY);
        // Nothing is unbounded by default, and padding is zero.
        assert_eq!(manifest.velocity_limits[..6], [3.0; 6]);
        assert_eq!(manifest.acceleration_limits[5], 15.0);
        assert!(manifest.jerk_limits.iter().all(|v| v.is_nan()));
        assert_eq!(manifest.effort_limits[3], 28.0);
        assert!(manifest.has_static_inertials);
        for limits in [
            manifest.joint_limits_min,
            manifest.joint_limits_max,
            manifest.velocity_limits,
            manifest.acceleration_limits,
            manifest.effort_limits,
        ] {
            assert_eq!(limits[6..], [0.0, 0.0]);
        }
        assert!(manifest.model.dynamics);
        assert!(manifest.supports(control_mode::JOINT_POS));
        assert!(!manifest.supports(control_mode::JOINT_POS | control_mode::JOINT_TORQUE));
        assert_eq!(manifest.state_len(), CoreState::LEN + 7);
        assert_eq!(manifest.command_len(), CoreCommand::LEN);

        let mut bad = ratings();
        bad.effort[2] = f64::INFINITY;
        assert_eq!(
            ArmManifest::for_model::<SixR, 6, 1>(500, modes, &bad),
            Err(ModelError::NonFinite)
        );
        bad.effort[2] = 0.0;
        assert_eq!(
            ArmManifest::for_model::<SixR, 6, 1>(500, modes, &bad),
            Err(ModelError::NonPositive)
        );
        assert_eq!(
            ArmManifest::for_model::<SixR, 6, 1>(0, modes, &ratings()),
            Err(ModelError::NonPositive)
        );
        assert!(AuxSchema::from_slice(
            &[AuxField {
                label: "x",
                size: 1
            }; 17]
        )
        .is_none());
    }

    #[test]
    fn example_file_loads() {
        let m = ArmManifest::from_text::<SixR, 6, 1>(EXAMPLE).unwrap();
        assert_eq!((m.name, m.version), ("six-r", CONTRACT_VERSION));
        assert_eq!((m.control_freq_hz, m.is_realtime_required), (500, true));
        assert_eq!(
            m.control_modes,
            control_mode::JOINT_POS | control_mode::JOINT_VEL | control_mode::CARTESIAN_POSE
        );
        assert_eq!(
            (m.state_staleness_cycles, m.command_dispatch_cycles),
            (1, 2)
        );
        assert!(m.has_torque_sensing && !m.has_wrench_sensing && !m.has_integrated_gripper);
        assert!(m.native_cartesian_admittance && m.native_payload && m.native_tcp_offset);
        assert!(!m.native_cartesian_impedance && !m.native_reflexes);
        assert_eq!(m.velocity_limits[..6], [3.15, 3.15, 3.15, 3.2, 3.2, 3.2]);
        assert_eq!(m.acceleration_limits[..6], [15.0; 6]);
        assert_eq!(m.jerk_limits[..6], [300.0; 6]);
        assert_eq!(m.jerk_limits[6..], [0.0, 0.0]);
        assert_eq!(
            m.effort_limits[..6],
            [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]
        );
        assert!(m.has_static_inertials);
        // Position limits: overridden by the file, including a continuous joint.
        assert_eq!(m.joint_limits_min[..2], [-3.0, -6.2]);
        assert_eq!(
            (m.joint_limits_min[5], m.joint_limits_max[5]),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
        assert_eq!(
            m.aux.fields(),
            [
                AuxField {
                    label: "motor_temp",
                    size: 6
                },
                AuxField {
                    label: "bus_voltage",
                    size: 1
                }
            ]
        );
        assert_eq!(m.state_len(), CoreState::LEN + 7);
        assert_eq!(m.model.dof, 6);
    }

    const MINIMAL: &str = "driver: six-r
version: \"0.2\"
control_freq_hz: 250
control_modes: [joint_pos]
latency:
  state_staleness_cycles: 0
  command_dispatch_cycles: 0
limits:
  velocity: [1, 1, 1, 1, 1, 1]
  acceleration: [2, 2, 2, 2, 2, 2]
  effort: [3, 3, 3, 3, 3, 3]
";

    #[test]
    fn minimal_file_takes_position_limits_from_the_model() {
        let m = ArmManifest::from_text::<SixR, 6, 1>(MINIMAL).unwrap();
        let direct = ArmManifest::for_model::<SixR, 6, 1>(
            250,
            control_mode::JOINT_POS,
            &JointRatings {
                velocity: [1.0; 6],
                acceleration: [2.0; 6],
                effort: [3.0; 6],
            },
        )
        .unwrap();
        assert_eq!(m.velocity_limits, direct.velocity_limits);
        assert_eq!(m.acceleration_limits, direct.acceleration_limits);
        assert_eq!(m.effort_limits, direct.effort_limits);
        assert_eq!(m.joint_limits_min, direct.joint_limits_min);
        assert_eq!(m.joint_limits_max, direct.joint_limits_max);
        assert!(m.jerk_limits.iter().all(|v| v.is_nan()));
        assert_eq!(m.has_static_inertials, direct.has_static_inertials);
    }

    #[test]
    fn bad_files_are_rejected_with_a_line_number() {
        use ManifestErrorKind::*;
        assert_eq!(kind(""), (0, MissingKey("driver")));
        assert_eq!(
            kind("driver: six-r\nversion: \"0.2\"\ncontrol_freq_hz: 250\ncontrol_modes: [joint_pos]\nlatency:\n  state_staleness_cycles: 0\n  command_dispatch_cycles: 0\nlimits:\n  velocity: [1, 1, 1, 1, 1, 1]\n  effort: [3, 3, 3, 3, 3, 3]\n"),
            (0, MissingKey("limits.acceleration"))
        );
        assert_eq!(kind("driver: other-arm\n"), (1, ModelMismatch));
        assert_eq!(
            kind("driver: six-r\nversion: \"9.0\"\n"),
            (2, VersionMismatch)
        );
        assert_eq!(kind("driver: six-r\ndriver: six-r\n"), (2, DuplicateKey));
        assert_eq!(kind("# comment\n\nspeed: 3\n"), (3, UnknownKey));
        assert_eq!(kind("limits:\n  snap: [1]\n"), (2, UnknownKey));
        assert_eq!(kind("limits:\n  jerk: [1, 2, 3]\n"), (2, WrongLength));
        assert_eq!(
            kind("limits:\n  jerk: [1, 1, 1, 1, 1, 0]\n"),
            (2, Model(ModelError::NonPositive))
        );
        assert_eq!(kind("limits:\n  velocity: [1, 2, 3]\n"), (2, WrongLength));
        assert_eq!(
            kind("limits:\n  velocity: [1, 2, 3, 4, 5, x]\n"),
            (2, InvalidValue)
        );
        assert_eq!(
            kind("control_modes: [joint_pos, levitate]\n"),
            (1, InvalidValue)
        );
        assert_eq!(kind("control_freq_hz: fast\n"), (1, InvalidValue));
        assert_eq!(kind("realtime: yes\n"), (1, InvalidValue));
        assert_eq!(kind("  velocity: [1]\n"), (1, Syntax));
        assert_eq!(kind("just some words\n"), (1, Syntax));
        assert_eq!(kind("\tdriver: six-r\n"), (1, Syntax));
        assert_eq!(kind("limits:\nlimits:\n"), (2, DuplicateKey));
        assert_eq!(
            kind("aux:\n  - name: a\n  - name: b\n    count: 1\n"),
            (3, InvalidAux)
        );
        assert_eq!(kind("aux:\n  - name: a\n    type: u8\n"), (3, InvalidValue));
        assert_eq!(kind("aux:\n  - name: a\n    count: 0\n"), (3, InvalidAux));

        // Ratings must be positive and finite, position limits ordered.
        let zero_velocity: &'static str = "driver: six-r
version: \"0.2\"
control_freq_hz: 250
control_modes: [joint_pos]
latency:
  state_staleness_cycles: 0
  command_dispatch_cycles: 0
limits:
  velocity: [1, 1, 1, 1, 1, 0]
  acceleration: [2, 2, 2, 2, 2, 2]
  effort: [3, 3, 3, 3, 3, 3]
  position_min: [1, 0, 0, 0, 0, 0]
  position_max: [0, 1, 1, 1, 1, 1]
";
        assert_eq!(kind(zero_velocity), (0, Model(ModelError::NonPositive)));
    }
}
