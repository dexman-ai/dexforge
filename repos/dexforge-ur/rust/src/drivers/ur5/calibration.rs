//! Per-robot kinematic calibration from the UR controller's
//! `calibration.conf` (`/root/.urcontrol/calibration.conf`).
//!
//! The `[mounting]` section holds the factory DH deltas, one value per joint:
//!
//! ```text
//! [mounting]
//! delta_theta = [ 1.2e-05, 0.3714, -0.7615, 0.3901, 1.2e-06, 3.1e-07]
//! delta_a = [ 2.8e-05, 0.000187, 0.00131, 2.4e-05, 6.7e-05, 0]
//! delta_d = [ 0.000102, -71.93, 147.88, -75.95, -2.7e-05, 0.000118]
//! delta_alpha = [ -0.000159, 0.000118, 0.00175, -0.00211, 0.000384, 0]
//! ```
//!
//! They are added to [`NOMINAL_DH`] (standard DH). On the nearly parallel
//! axes 2–4 the `d` deltas can be tens of metres and cancel; dexhal's chain
//! keeps that precise. The DH chain uses the same base and flange frames as the
//! URDF `base_link` → `ee_link` chain of this driver (which is UR's own description,
//! see `tools/import_ur_description.py`), so no frame adapters are needed.
//!
//! ```ignore
//! let calibration = ur5::calibration::calibration_data(&text)?;
//! driver.set_calibration(&calibration)?;
//! ```

use core::f64::consts::FRAC_PI_2;
use core::fmt;

use dexhal::math::{
    apply_dh_deltas, CalibrationData, DhConvention, DhRow, KinematicChain, ModelError,
};

use super::arm::DOF;

/// UR's nominal UR5e standard DH table (`a`, `alpha`, `d`, `theta`).
///
/// The controller's `calibration.conf` deltas are defined against exactly these values.
/// They are the datasheet dimensions, the same ones as in the driver's own chain
/// (`ur5.urdf`, imported from UR's `ur_description`), so zero deltas reproduce that chain.
pub const NOMINAL_DH: [DhRow; DOF] = [
    DhRow::new(0.0, FRAC_PI_2, 0.1625, 0.0),
    DhRow::new(-0.425, 0.0, 0.0, 0.0),
    DhRow::new(-0.3922, 0.0, 0.0, 0.0),
    DhRow::new(0.0, FRAC_PI_2, 0.1333, 0.0),
    DhRow::new(0.0, -FRAC_PI_2, 0.0997, 0.0),
    DhRow::new(0.0, 0.0, 0.0996, 0.0),
];

/// The four `[mounting]` keys, in [`DhRow`] field order.
const KEYS: [&str; 4] = ["delta_a", "delta_alpha", "delta_d", "delta_theta"];

/// Why a `calibration.conf` could not be used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationFileError {
    /// No `[mounting]` section.
    MissingSection,
    /// A `delta_*` key is absent from `[mounting]`.
    MissingKey(&'static str),
    /// A `delta_*` key appears twice.
    DuplicateKey(&'static str),
    /// A `delta_*` value is not a list of finite numbers.
    BadValue(&'static str),
    /// A `delta_*` list does not have one value per joint.
    WrongLength(&'static str),
    /// The calibrated chain is not a valid model.
    Model(ModelError),
}

impl fmt::Display for CalibrationFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSection => write!(f, "calibration.conf has no [mounting] section"),
            Self::MissingKey(k) => write!(f, "calibration.conf: [mounting] lacks {k}"),
            Self::DuplicateKey(k) => write!(f, "calibration.conf: {k} appears twice"),
            Self::BadValue(k) => write!(f, "calibration.conf: {k} is not a list of numbers"),
            Self::WrongLength(k) => write!(f, "calibration.conf: {k} needs {DOF} values"),
            Self::Model(e) => write!(f, "calibration.conf gives an invalid chain: {e:?}"),
        }
    }
}

/// Parse `[ v1, v2, … ]` (brackets optional, comma and/or space separated).
fn parse_list(key: &'static str, value: &str) -> Result<[f64; DOF], CalibrationFileError> {
    let value = value.trim();
    let value = value.strip_prefix('[').unwrap_or(value);
    let value = value.strip_suffix(']').unwrap_or(value);
    let mut out = [0.0; DOF];
    let mut n = 0;
    for token in value
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
    {
        let v: f64 = token
            .parse()
            .map_err(|_| CalibrationFileError::BadValue(key))?;
        if !v.is_finite() {
            return Err(CalibrationFileError::BadValue(key));
        }
        if n == DOF {
            return Err(CalibrationFileError::WrongLength(key));
        }
        out[n] = v;
        n += 1;
    }
    if n != DOF {
        return Err(CalibrationFileError::WrongLength(key));
    }
    Ok(out)
}

/// The per-joint DH deltas of a `calibration.conf`. Other sections and keys
/// are ignored.
pub fn parse_deltas(text: &str) -> Result<[DhRow; DOF], CalibrationFileError> {
    let mut found = [None::<[f64; DOF]>; 4];
    let mut in_mounting = false;
    let mut seen_mounting = false;
    for line in text.lines() {
        let line = line.split(['#', ';']).next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(section) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            // A list value never stands alone on a line, so this is a header.
            if !section.contains(',') {
                in_mounting = section.trim() == "mounting";
                seen_mounting |= in_mounting;
                continue;
            }
        }
        if !in_mounting {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let Some(slot) = KEYS.iter().position(|k| *k == key.trim()) else {
            continue;
        };
        let key = KEYS[slot];
        if found[slot].is_some() {
            return Err(CalibrationFileError::DuplicateKey(key));
        }
        found[slot] = Some(parse_list(key, value)?);
    }
    if !seen_mounting {
        return Err(CalibrationFileError::MissingSection);
    }
    let mut lists = [[0.0; DOF]; 4];
    for (slot, list) in lists.iter_mut().enumerate() {
        *list = found[slot].ok_or(CalibrationFileError::MissingKey(KEYS[slot]))?;
    }
    let [a, alpha, d, theta] = lists;
    Ok(core::array::from_fn(|i| {
        DhRow::new(a[i], alpha[i], d[i], theta[i])
    }))
}

/// The robot's calibrated chain (`base_link` → `ee_link`).
pub fn calibrated_chain(text: &str) -> Result<KinematicChain<DOF>, CalibrationFileError> {
    let rows = apply_dh_deltas(&NOMINAL_DH, &parse_deltas(text)?);
    KinematicChain::from_dh(DhConvention::Standard, &rows).map_err(CalibrationFileError::Model)
}

/// [`calibrated_chain`] for `ArmDriver::set_calibration`.
pub fn calibration_data(text: &str) -> Result<CalibrationData, CalibrationFileError> {
    Ok(CalibrationData::from_chain(&calibrated_chain(text)?))
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::format;
    use std::string::String;

    use super::*;
    use crate::drivers::ur5::arm::{nominal_chain, Ur5};
    use crate::drivers::ur5::driver::Ur5Driver;
    use dexhal::math::{ArmDriver, ArmModel, Pose, MAX_DOF};

    /// Shape of a real controller file: other sections, comments, and large
    /// cancelling `d` deltas on the parallel axes.
    const LARGE: &str = "\
# Calibration file written by the controller
[firmware]
version = 3.15.7

[mounting]
delta_theta = [ -2.5059e-07, 0.371428, -0.761487, 0.390058, 1.2479e-06, 3.06727e-07]
delta_a = [ 2.81606e-05, 0.000187398, 0.0013095, 2.44787e-05, 6.71034e-05, 0]
delta_d = [ 0.000101845, -71.9298, 147.879, -75.9459, -2.65568e-05, 0.000118419]
delta_alpha = [ -0.000158658, 0.00011802, 0.00174624, -0.0021117, 0.000384082, 0]
joint_checksum = [ -1, -1, -1, -1, -1, -1]

[other]
delta_d = [ 1, 1, 1, 1, 1, 1]
";

    fn small() -> String {
        String::from(
            "[mounting]\n\
             delta_theta = [1e-4, -2e-4, 3e-4, -1e-4, 2e-4, 0]\n\
             delta_a = [0, 4e-4, -3e-4, 0, 0, 0]\n\
             delta_d = [2e-4, 0, 0, -1e-4, 3e-4, 1e-4]\n\
             delta_alpha = [1e-4, 0, -2e-4, 1e-4, 0, 0]\n",
        )
    }

    fn zero() -> String {
        let z = "[0, 0, 0, 0, 0, 0]";
        format!(
            "[mounting]\ndelta_theta = {z}\ndelta_a = {z}\ndelta_d = {z}\ndelta_alpha = {z}\n"
        )
    }

    /// Independent standard-DH forward kinematics.
    fn dh_fk(rows: &[DhRow; DOF], q: &[f64; DOF]) -> Pose {
        let mut t = Pose::identity();
        for (row, qi) in rows.iter().zip(q) {
            let (st, ct) = libm::sincos(qi + row.theta);
            let (sa, ca) = libm::sincos(row.alpha);
            t *= Pose::new(
                ct, -st * ca, st * sa, row.a * ct, //
                st, ct * ca, -ct * sa, row.a * st, //
                0.0, sa, ca, row.d, //
                0.0, 0.0, 0.0, 1.0,
            );
        }
        t
    }

    fn sample(k: usize) -> [f64; DOF] {
        core::array::from_fn(|i| ((k * 7 + i * 13) % 17) as f64 / 17.0 * 5.0 - 2.5)
    }

    fn padded(q: &[f64; DOF]) -> [f64; MAX_DOF] {
        let mut p = [0.0; MAX_DOF];
        p[..DOF].copy_from_slice(q);
        p
    }

    #[test]
    fn parses_the_mounting_section_only() {
        let rows = parse_deltas(LARGE).unwrap();
        assert_eq!(rows[1].d, -71.9298);
        assert_eq!(rows[2].theta, -0.761487);
        assert_eq!(rows[2].a, 0.0013095);
        assert_eq!(rows[3].alpha, -0.0021117);
        assert_eq!(rows[5].d, 0.000118419);
    }

    #[test]
    fn zero_deltas_reproduce_the_driver_chain() {
        let chain = calibrated_chain(&zero()).unwrap();
        let nominal = nominal_chain();
        for k in 0..16 {
            let q = sample(k);
            assert!((chain.fk(&q) - dh_fk(&NOMINAL_DH, &q)).norm() < 1e-12);
            assert!((chain.fk(&q) - nominal.fk(&q)).norm() < 1e-9);
            assert!((chain.fk(&q) - Ur5::flange_fk(&q)).norm() < 1e-9);
        }
    }

    #[test]
    fn large_cancelling_deltas_stay_precise() {
        let rows = apply_dh_deltas(&NOMINAL_DH, &parse_deltas(LARGE).unwrap());
        let chain = calibrated_chain(LARGE).unwrap();
        for k in 0..16 {
            let q = sample(k);
            let reference = dh_fk(&rows, &q);
            assert!((chain.fk(&q) - reference).norm() < 1e-9, "at {q:?}");
        }
    }

    #[test]
    fn driver_uses_the_file() {
        let mut driver = Ur5Driver::new().unwrap();
        let text = small();
        driver
            .set_calibration(&calibration_data(&text).unwrap())
            .unwrap();
        assert!(driver.arm().is_calibrated());
        let rows = apply_dh_deltas(&NOMINAL_DH, &parse_deltas(&text).unwrap());
        for k in 0..8 {
            let q = sample(k);
            let fk = driver.arm().forward_kinematics(&padded(&q)).unwrap();
            let reference = dh_fk(&rows, &q);
            for (i, v) in fk[..3].iter().enumerate() {
                assert!((v - reference[(i, 3)]).abs() < 1e-9);
            }
            // A sub-millimetre calibration moves the flange, but not by much.
            let nominal = Ur5::flange_fk(&q);
            let shift = (reference.fixed_view::<3, 1>(0, 3) - nominal.fixed_view::<3, 1>(0, 3)).norm();
            assert!(shift > 1e-6 && shift < 5e-3, "shift {shift}");
            // IK on the calibrated model closes on the calibrated target.
            let sol = driver.arm().inverse_kinematics(&fk, &padded(&q)).unwrap();
            let back = driver.arm().forward_kinematics(&sol).unwrap();
            assert!(back[..3].iter().zip(&fk[..3]).all(|(a, b)| (a - b).abs() < 1e-6));
        }
    }

    #[test]
    fn rejects_malformed_files() {
        use CalibrationFileError::*;
        assert_eq!(parse_deltas("delta_a = [0,0,0,0,0,0]"), Err(MissingSection));
        let without = small().replace("delta_alpha", "other");
        assert_eq!(parse_deltas(&without), Err(MissingKey("delta_alpha")));
        let short = small().replace("[0, 4e-4, -3e-4, 0, 0, 0]", "[0, 4e-4]");
        assert_eq!(parse_deltas(&short), Err(WrongLength("delta_a")));
        let long = small().replace("[0, 4e-4, -3e-4, 0, 0, 0]", "[0, 4e-4, -3e-4, 0, 0, 0, 1]");
        assert_eq!(parse_deltas(&long), Err(WrongLength("delta_a")));
        let bad = small().replace("4e-4", "x");
        assert_eq!(parse_deltas(&bad), Err(BadValue("delta_a")));
        let nan = small().replace("4e-4", "NaN");
        assert_eq!(parse_deltas(&nan), Err(BadValue("delta_a")));
        let twice = format!("{}delta_d = [0, 0, 0, 0, 0, 0]\n", small());
        assert_eq!(parse_deltas(&twice), Err(DuplicateKey("delta_d")));
    }
}
