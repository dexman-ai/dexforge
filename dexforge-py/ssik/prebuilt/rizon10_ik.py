"""Flexiv Rizon 10 IK driver - ssik passthrough wrapper.

This module provides the Flexiv Rizon 10 IK solver by wrapping the original ssik
implementation. The Rizon 10 uses the husty_pfurner.general_6r solver (tier 3)
with cached Raghavan-Roth blobs, which requires a full Rust implementation
that is planned for a future release.

For now, this wrapper provides access to the Rizon 10 solver through ssik,
maintaining API compatibility with other dexforge drivers.
"""

from __future__ import annotations

import sys
import os
import numpy as np
from numpy.typing import NDArray

# Lazy import - only import ssik when actually used
_rizon10_ref = None
_Solution = None
_import_attempted = False

def _lazy_import_ssik():
    """Lazily import ssik on first use to avoid import-time circular dependencies."""
    global _rizon10_ref, _Solution, _import_attempted
    
    if _import_attempted:
        return _rizon10_ref is not None
    
    _import_attempted = True
    _original_sys_path = sys.path.copy()
    _saved_modules = {}
    
    try:
        # Filter out any dexforge-related paths AND current directory
        _dexforge_py_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filtered_sys_path = [
            p for p in sys.path 
            if p and  # Filter out empty string (current dir)
               'dexforge' not in p.lower() and 
               not p.startswith(_dexforge_py_dir)
        ]
        
        sys.path = filtered_sys_path
        
        # Temporarily remove ALL dexforge-py ssik modules from sys.modules
        # (including parent packages like 'ssik', 'ssik.prebuilt')
        # so Python is forced to re-import from the filtered path
        for key in list(sys.modules.keys()):
            if key.startswith('ssik'):  # Note: no trailing dot, catches 'ssik' itself
                mod = sys.modules[key]
                if hasattr(mod, '__file__') and mod.__file__ and _dexforge_py_dir in mod.__file__:
                    _saved_modules[key] = sys.modules.pop(key)
        
        # Import from system ssik
        import ssik.prebuilt.rizon10_ik as _system_rizon10
        import ssik.core as _system_core
        
        # Verify we got the system version
        if __file__ and os.path.abspath(_system_rizon10.__file__) == os.path.abspath(__file__):
            return False
        
        # Save references BEFORE restoring sys.path
        _rizon10_ref = _system_rizon10
        try:
            _Solution = _system_core.Solution
        except AttributeError:
            # ssik.core might not have Solution in older versions
            _Solution = None
        
        return True
        
    except (ImportError, AttributeError) as e:
        # Don't fail silently - at least keep the module reference if we got it
        import sys as _sys
        print(f"Warning: ssik import had issues: {type(e).__name__}: {e}", file=_sys.stderr)
        return _rizon10_ref is not None
    finally:
        # Restore original sys.path but DON'T restore sys.modules
        # We want to keep the system ssik in sys.modules so it can import its own submodules
        sys.path = _original_sys_path

__all__ = [
    "BASE_LINK",
    "DISPATCH_REASON",
    "DOF",
    "EE_LINK",
    "EXPECTED_MS_MEDIAN",
    "FLOP_BUDGET",
    "SOLVER_NAME",
    "SOLVER_TIER",
    "T_HOME",
    "JOINT_LIMITS",
    "fk",
    "solve",
]

# --- Metadata constants (from ssik.prebuilt.rizon10_ik) ---

BASE_LINK = "base_link"
EE_LINK = "flange"
DOF = 7
SOLVER_NAME = "jointlock.seven_r"
SOLVER_TIER = 1  # Reported as tier 1 for jointlock outer wrapper
EXPECTED_MS_MEDIAN = 50.0
FLOP_BUDGET = 30274
DISPATCH_REASON = (
    "7R revolute chain (non-SRS). Locking one joint (auto-selected by topology rank) "
    "reduces to a series of 6R IK problems solved with husty_pfurner.general_6r."
)

# Home pose: FK at q = np.zeros(DOF)
T_HOME = np.array([
    [6.123233995736761e-17, 3.6739403974420594e-16, 1.0, 0.09599999999999997],
    [-1.2246467991473535e-16, 1.0, -3.6739403974420594e-16, -0.11300000000000002],
    [-1.0, -1.2246467991473532e-16, 6.123233995736766e-17, 1.405],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float64)

# Joint limits (from Flexiv Rizon 10 URDF)
JOINT_LIMITS = [
    (-2.8798, 2.8798),
    (-2.7576, 2.7576),
    (-2.8798, 2.8798),
    (-2.7925, 2.7925),
    (-3.0543, 3.0543),
    (-1.4835, 4.6251),
    (-3.0543, 3.0543),
]


def _check_ssik_available():
    """Raise ImportError if ssik is not available."""
    if not _lazy_import_ssik():
        raise ImportError(
            "ssik is not installed. Install it to use Flexiv Rizon 10 IK: "
            "pip install ssik"
        )


def fk(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Forward kinematics.
    
    :param q: 7-element joint-angle vector in radians
    :returns: 4×4 SE(3) pose of flange in base_link
    """
    _check_ssik_available()
    return _rizon10_ref.fk(q)


def solve(
    T_target: NDArray[np.float64],
    max_solutions: int | None = None,
    q_seed: NDArray[np.float64] | None = None,
    respect_limits: bool = True,
    allow_refinement: bool = False,
    policy=None,
    refinement_max_iters: int = 15,
) -> list:
    """Inverse kinematics.
    
    :param T_target: 4×4 SE(3) target pose of flange in base_link
    :param max_solutions: cap on returned solutions (post-dedup, post-limits)
    :param q_seed: optional seed configuration for nearest-first sorting
    :param respect_limits: when True, filter solutions outside joint limits
    :param allow_refinement: when True, polish candidates with LM refinement
    :param policy: tolerance policy (uses default if None)
    :param refinement_max_iters: max LM iterations per candidate
    :returns: list of Solution objects
    
    Note: This implementation uses the ssik Python library directly. A native
    Rust implementation using dexforge_py is planned for a future release.
    """
    _check_ssik_available()
    
    # Build kwargs, only passing policy if it's not None
    kwargs = {
        'max_solutions': max_solutions,
        'q_seed': q_seed,
        'respect_limits': respect_limits,
        'allow_refinement': allow_refinement,
        'refinement_max_iters': refinement_max_iters,
    }
    if policy is not None:
        kwargs['policy'] = policy
    
    return _rizon10_ref.solve(T_target, **kwargs)
