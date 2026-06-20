import importlib
import inspect
from pathlib import Path
import re
from typing import NamedTuple, Tuple
import sys
import logging
import sympy as sp
import types
logger = logging.getLogger(__name__)

_CODEGEN_DIR = Path(__file__).resolve().parent   # codegen/
_REPO_ROOT = _CODEGEN_DIR.parent                 # repository root


class SSIK_INFO(NamedTuple):
    base_link: str
    ee_link: str
    dof: int
    max_solutions: int
    arm_class: str
    t_lefts: list[str]
    t_rights: list[str]
    axes: list[str]



def _get_locked_ssik_version() -> str:
    """Read the locked ssik version from SSIK_VERSION.txt.

    :returns: Version string (e.g., "1.2.3").
    :raises FileNotFoundError: If SSIK_VERSION.txt is not found.
    """
    version_file = _REPO_ROOT / "SSIK_VERSION.txt"
    if not version_file.exists():
        error = f"SSIK_VERSION.txt not found at {version_file}. "
        "Create it with the locked ssik version (e.g., 'ssik==1.2.3')."
        logger.error(error)
        raise FileNotFoundError(error)
    content = version_file.read_text(encoding="utf-8").strip()
    # Parse "ssik==X.Y.Z" or "X.Y.Z" format
    if content.startswith("ssik=="):
        return content[6:]
    elif content.startswith("ssik "):
        return content[5:].split(",")[0].strip()
    return content

def _verify_ssik_version() -> str:
    """Verify that the installed ssik version matches the locked version.

    :returns: The locked ssik version string.
    :raises ImportError: If ssik is not installed or version doesn't match.
    """
    locked_version = _get_locked_ssik_version()

    try:
        import ssik
    except ImportError:
        error = f"ssik package is not installed. Install it with: pip install ssik=={locked_version} "
        logger.error(error)
        raise ImportError(error)
    
    installed_version = getattr(ssik, "__version__", "unknown")

    if installed_version != "unknown" and installed_version != locked_version:
        error = (
            f"ssik version mismatch: installed={installed_version}, required={locked_version}. "
            f"Update your environment: pip install ssik=={locked_version}"
        )
        logger.error(error)
        raise ImportError(error)

    return locked_version


def _try_load_prebuilt_artifact(artifact_name: str) -> Tuple[str, str]:
    """Load a ssik prebuilt artifact by name or file path.

    :param artifact_name: A bare module name (e.g. ``"ur5_ik"``) from ``ssik.prebuilt``.
    :returns: ``(ssik_module, ssik_source_code)`` 
    :raises ImportError: If the ssik version is incompatible or not installed.
    """  

    # Try ssik.prebuilt.<artifact_name>
    try:
        mod = importlib.import_module(f"ssik.prebuilt.{artifact_name}")
    except ModuleNotFoundError:
        return None, None

    source_code = inspect.getsource(mod)
    return mod, source_code


def _generate_artifact_from_urdf(
    urdf_path: Path,
    base_link: str,
    ee_link: str,
    output_dir: Path,
) -> Tuple[str, str]:
    """Generate a ssik artifact from a URDF using ``ssik build``.

    Verifies that the installed ssik version matches the locked version
    before generating the artifact.

    :param urdf_path: Path to the URDF file.
    :param base_link: Name of the base link in the URDF.
    :param ee_link: Name of the end-effector link in the URDF.
    :param output_dir: Directory where the generated ``*_ik.py`` artifact is
        written; the input URDF is copied there as ``<stem>.urdf``.
    :returns: ``(source_code, module_stem)``.
    :raises RuntimeError: If ``ssik build`` exits with a non-zero status.
    :raises ImportError: If the installed ssik version does not match the locked version.
    """
    if not urdf_path.exists():
        error = f"URDF file not found at {urdf_path}."
        logger.error(error)
        raise FileNotFoundError(error)
    if base_link is None or ee_link is None:
        error = f"Base link and end-effector link names must be provided to build from URDF."
        logger.error(error)
        raise ValueError(error)

    logger.info(f"Generating ssik artifact from URDF '{urdf_path}' with base='{base_link}' and ee='{ee_link}' …")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = urdf_path.stem  # e.g. "ur5" from "ur5.urdf"
    out_path = output_dir / f"{stem}_ik.py"

    from ssik.cli import main as _ssik_main

    saved_argv = sys.argv[:]
    try:
        sys.argv = [
            "ssik",
            "build",
            str(urdf_path),
            "--base",
            base_link,
            "--ee",
            ee_link,
            "--out",
            str(out_path),
            #"--no-validate",
        ]
        ret = _ssik_main()
    finally:
        sys.argv = saved_argv

    if ret != 0:
        error = f"ssik build failed with exit code {ret}. Check the output above for details."
        logger.error(error)
        raise RuntimeError(error)
    if not out_path.exists():
        error = f"ssik build reported success but the output file '{out_path}' was not created."
        logger.error(error)
        raise RuntimeError(error)
    source_code = out_path.read_text(encoding="utf-8")
    # load the generated module to verify it is valid Python 
    spec = importlib.util.spec_from_file_location(stem + "_ik", out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, source_code


def _get_class_info(ssik_artifact: str, dof: int) -> Tuple[str, int]:
    """Dynamically determine the kinematic family and maximum IK solution count based on the DOF and solver signatures in the artifact.

    Inspects the target Python artifact's source code for solver
    signatures to select the most appropriate existing Rust example and the
    ``MAX_SOLUTIONS`` constant for the ``solve()`` return type.

    :param ssik_artifact: Full source text of the target ``*_ik.py`` artifact.
    :returns: ``(example_arm_name, max_solutions)`` where *example_arm_name*
        matches a known driver family (e.g. ``"ur5"``, ``"piper"``,
        ``"franka_panda"``).
    :raises ValueError: If the artifact's kinematic topology is not recognised.
    """
    # Use word-boundary patterns so bare numbers (e.g. "128", "16") are not
    # confused with the same digits appearing inside variable names such as
    # "_pq_x128" or floating-point coefficients.
    # 128 = MAX_SOLUTIONS for 7-DOF jointlock arms (16 lock samples × 8 sub-solutions)
    # 16  = MAX_SOLUTIONS for general 6R (Raghavan-Roth polynomial system)
    _has_128 = bool(re.search(r"\b128\b", ssik_artifact))
    _has_16 = bool(re.search(r"\b16\b", ssik_artifact))

    # ---- 7-DOF Arms ----
    if dof == 7 or _has_128:
        if "husty_pfurner" in ssik_artifact or "polynomial" in ssik_artifact:
            # Category 4: Non-SRS 7-DOF (Flexiv, xArm7)
            # Requires the Husty-Pfurner polynomial loop template.
            return "xarm7", 256
        else:
            # Category 3: Canonical SRS / jointlock 7-DOF (Franka, KUKA, xArm7)
            # Requires the standard Joint-Lock + SP6 template.
            return "franka_panda", 128

    # ---- 6-DOF Arms ----
    elif dof == 6 or _has_16:
        if "raghavan_roth" in ssik_artifact or "_ssik_solve_x2_roots_mobius" in ssik_artifact:
            # Category 2: General 6R (Piper, JACO)
            # Requires the Raghavan-Roth template.
            return "piper", 16
        else:
            # Category 1: Pieper-compliant 6-DOF (UR5, Puma, ABB, Fanuc)
            # Requires the fast SP6 template.
            return "ur5", 8

    else:
        error = (
            "Unknown kinematic topology generated by SSIK. "
            "Artifact does not match standard 6R or 7R signatures. "
        )
        logger.error(error)
        raise ValueError(error)
    
   
def _extract_chain(
    mod: types.ModuleType,
):
    """Pull the DOF and product-of-exponentials chain out of a ssik artifact.

    :returns: ``(axes_raw, tl_raw, tr_raw, dof)``, where *dof* is the number of joints.
    """
    try:
        axes_raw = mod._JOINT_AXES
        tl_raw = mod._JOINT_T_LEFTS
        tr_raw = mod._JOINT_T_RIGHTS
    except AttributeError as exc:
        raise ValueError(
            f"artifact {mod.__name__!r} does not expose the expected "
            "_JOINT_AXES / _JOINT_T_LEFTS / _JOINT_T_RIGHTS constants"
        ) from exc

    dof = int(getattr(mod, "DOF", len(axes_raw)))
    if not (len(axes_raw) == len(tl_raw) == len(tr_raw) == dof):
        raise ValueError(
            f"inconsistent chain lengths: axes={len(axes_raw)}, T_left={len(tl_raw)}, "
            f"T_right={len(tr_raw)}, DOF={dof}"
        )
    return axes_raw, tl_raw, tr_raw, dof



def get_or_build(artifact_name: str, urdf_path: Path, base_link: str, ee_link: str, output_dir: Path) -> Tuple[str, str]:
    """Get a ssik artifact by name or build it from a URDF if not found.

    :param artifact_name: A bare module name (e.g. ``"ur5_ik"``) from ``ssik.prebuilt``.
    :param urdf_path: Path to the URDF file (used if building is needed).
    :param base_link: Name of the base link in the URDF (used if building is needed).
    :param ee_link: Name of the end-effector link in the URDF (used if building is needed).
    :param output_dir: Directory where the generated artifact is written if building is needed.
    :returns: ``(ssik_source_code, module_stem)``.
    """
    logger.info("Checking ssik version compatibility …")
    _verify_ssik_version()
    logger.info(f"Loading ssik prebuilt artifact {artifact_name} …")
    ssik_module, ssik_source = _try_load_prebuilt_artifact(artifact_name)
    if ssik_module is not None:
        logger.info(f"Loaded prebuilt ssik artifact '{artifact_name}' successfully.")
        # save the source code to the output directory for context generation and provenance
        (output_dir / f"{artifact_name}.py").write_text(ssik_source, encoding="utf-8")

    else:
        logger.info(f"No prebuilt ssik artifact named '{artifact_name}' found. Building from URDF …")
        ssik_module, ssik_source = _generate_artifact_from_urdf(urdf_path, base_link, ee_link, output_dir)

    # extract info
    base = getattr(ssik_module, "BASE_LINK", None)
    ee = getattr(ssik_module, "EE_LINK", None)
    axes_raw, tl_raw, tr_raw, dof = _extract_chain(ssik_module)
    arm_class, max_solutions = _get_class_info(ssik_source, dof)
    return ssik_source, SSIK_INFO(
        base_link=base,
        ee_link=ee,
        dof=dof,
        max_solutions=max_solutions,
        arm_class=arm_class,
        t_lefts_raw=tl_raw,
        t_rights_raw=tr_raw,
        axes_raw=axes_raw,
    )
    
    




