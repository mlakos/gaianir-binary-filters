"""
Configuration and constants for the filter optimization pipeline.

This module contains all constants, paths, and configuration flags.
No local imports to prevent circular dependencies.
"""

from __future__ import annotations

import os
from multiprocessing import cpu_count, get_start_method, set_start_method
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# =============================================================================
# Force fork start method (must happen before any Pool creation)
# =============================================================================
try:
    if get_start_method(allow_none=True) != "fork":
        set_start_method("fork", force=False)
except RuntimeError:
    pass  # Already set

# =============================================================================
# Physical Constants
# =============================================================================
KPCTOM = 3.085677581e19           # meters per kiloparsec
RSUN_M = 6.957e8                  # solar radius in meters
TEN_PC_M = KPCTOM / 100.0         # 10 parsecs in meters
RSUN_OVER_10PC_SQ = (RSUN_M / TEN_PC_M) ** 2

# =============================================================================
# CK04 Model Bounds
# =============================================================================
CK04_TEFF_MIN, CK04_TEFF_MAX = 3500.0, 6700.0
CK04_LOGG_MIN, CK04_LOGG_MAX = 0.0, 5.0
CK04_FEH_MIN, CK04_FEH_MAX = -2.5, 0.5

# =============================================================================
# Photometric Error Model
# =============================================================================
GAIA_G_FWHM_AA = 4435.0           # Gaia G filter FWHM in Angstrom
SYSTEMATIC_ERROR_MMAG = 1.0       # Systematic error floor in mmag (realistic calibration uncertainty)
SYSTEMATIC_ERROR_MAG = SYSTEMATIC_ERROR_MMAG / 1000.0
DEFAULT_APPARENT_MAG = 15.0       # Reference magnitude for error scaling

# =============================================================================
# Detection Model (Sigmoid)
# =============================================================================
SIGMOID_C = 5.0                   # Steepness
SIGMOID_S0 = 2                  # Detection threshold (sigma)

# =============================================================================
# Filter Configuration
# =============================================================================
FILTER_STARTING_EDGES = [8000.0, 11500.0, 14000.0, 18000.0]
N_FILTERS = len(FILTER_STARTING_EDGES)
N_NIR_COLORS = N_FILTERS - 1

# Wavelength limits for NIR filters (hard constraints)
MIN_WAVELENGTH = 8000.0           # Minimum wavelength for any filter (Angstrom)
MAX_WAVELENGTH = 25000.0          # Maximum wavelength for any filter (Angstrom)
MIN_FILTER_WIDTH = 500.0          # Minimum filter width (Angstrom)

# Edge optimization (uses latent parametrization with [0,1] variables for constraint satisfaction)
OPTIMIZE_FILTER_EDGES = False     # If True, optimize both edges and widths (8 latent params u,v)
EDGE_MIN = MIN_WAVELENGTH         # Minimum filter edge (Angstrom)
EDGE_MAX = 21500.0                # Maximum filter edge (Angstrom)
P1_MAX = 18000.0                  # Maximum edge for first filter in latent parametrization
EDGE_RANGE = 1750.0               # Maximum deviation from default edge values (Angstrom)

# Bounds for width-only optimization (4 parameters)
WIDTH_BOUNDS = [(MIN_FILTER_WIDTH, MAX_WAVELENGTH - edge) for edge in FILTER_STARTING_EDGES]

# Bounds for edge+width optimization (8 parameters: 4 edges + 4 widths)
# Note: The constraint edge + width <= MAX_WAVELENGTH is enforced at runtime
# by constrain_filter_params(), since scipy bounds are independent per parameter
EDGE_BOUNDS = [(EDGE_MIN, EDGE_MAX) for _ in range(N_FILTERS)]
EDGE_WIDTH_BOUNDS = EDGE_BOUNDS + [(MIN_FILTER_WIDTH, MAX_WAVELENGTH - EDGE_MIN) for _ in range(N_FILTERS)]

# Gaia optical filters
GAIA_FILTER_NAMES = ['GaiaDR2v2_BP', 'GaiaDR2v2_G', 'GaiaDR2v2_RP']
N_GAIA_COLORS = 2  # BP-G and G-RP

# =============================================================================
# Optimization Parameters
# =============================================================================
K_NEIGHBORS = 100
N_RANDOM_STARTS = 50
N_OPTIMIZER_CALLS = 300  # Total optimizer calls (was N_RANDOM_STARTS + 250)
ACQ_FUNC = "EI"
XI = 0.01
KAPPA = 1.96
REFINE_OPTIMUM = False
REFINE_MAXITER = 8

# =============================================================================
# Multiprocessing
# =============================================================================
N_WORKERS = max(cpu_count(), 1)

# =============================================================================
# Paths
# =============================================================================
PYSYN_CDBS = '/shared/storage/mlakaro/PYSYNPHOT'
os.environ.setdefault('PYSYN_CDBS', PYSYN_CDBS)

PATH_TO_BINARY_TABLE = '/shared/scratch/mlakaro/galaxia_full_136k.fits'
GRID_H5_PATH = '/shared/storage/mlakaro/nir_filter_optimization/gridsearch_seds/grid.h5'

# =============================================================================
# Test Mode
# =============================================================================
TEST_MODE = False
TEST_BINARY_LIMIT = 100
TEST_OPTIMIZER_CALLS = 3

# =============================================================================
# Binary Table Configuration (mutable, set via CLI)
# =============================================================================
BINARY_TABLE_PATH = PATH_TO_BINARY_TABLE  # Can be overridden via CLI
BINARY_TABLE_LIMIT = -1  # Number of rows to process, -1 for all rows

# =============================================================================
# Runtime Flags (mutable, set via CLI)
# =============================================================================
USE_GAIA_FILTERS = False
USE_LBFGS_REFINEMENT = True


def ck04_bounds_mask(teff, logg, feh):
    """Return a boolean mask for CK04-valid parameters.

    Accepts either scalars or arrays.
    """
    import numpy as np

    teff_arr = np.asarray(teff, dtype=np.float64)
    logg_arr = np.asarray(logg, dtype=np.float64)
    feh_arr = np.asarray(feh, dtype=np.float64)

    mask = (
        np.isfinite(teff_arr)
        & np.isfinite(logg_arr)
        & np.isfinite(feh_arr)
        & (teff_arr >= CK04_TEFF_MIN)
        & (teff_arr <= CK04_TEFF_MAX)
        & (logg_arr >= CK04_LOGG_MIN)
        & (logg_arr <= CK04_LOGG_MAX)
        & (feh_arr >= CK04_FEH_MIN)
        & (feh_arr <= CK04_FEH_MAX)
    )

    # CK04 hot-star validity rule.
    mask &= ~((teff_arr > 6000.0) & (logg_arr < 0.5))
    return mask


def is_within_ck04_bounds(teff: float, logg: float, feh: float) -> bool:
    """Return True when a single (teff, logg, feh) triplet is CK04-valid."""
    return bool(ck04_bounds_mask(teff, logg, feh))


def get_n_initial_points() -> int:
    """Get number of initial points for Bayesian optimization (equals N_RANDOM_STARTS)."""
    return N_RANDOM_STARTS


def get_n_colors() -> int:
    """Get total number of colors based on Gaia filter configuration."""
    return N_NIR_COLORS + (N_GAIA_COLORS if USE_GAIA_FILTERS else 0)


def set_gaia_filters(enabled: bool) -> None:
    """Set whether Gaia filters are used (called from CLI parsing)."""
    global USE_GAIA_FILTERS
    USE_GAIA_FILTERS = enabled


def set_lbfgs_refinement(enabled: bool) -> None:
    """Set whether L-BFGS-B refinement is used (called from CLI parsing)."""
    global USE_LBFGS_REFINEMENT
    USE_LBFGS_REFINEMENT = enabled


def set_n_random_starts(n: int) -> None:
    """Set number of random starts for optimization (called from CLI parsing)."""
    global N_RANDOM_STARTS
    N_RANDOM_STARTS = max(1, n)


def set_n_optimizer_calls(n: int) -> None:
    """Set total number of optimizer calls (called from CLI parsing)."""
    global N_OPTIMIZER_CALLS
    N_OPTIMIZER_CALLS = max(1, n)


def get_optimization_bounds() -> list:
    """Get optimization bounds based on optimization mode."""
    from skopt.space import Real
    if OPTIMIZE_FILTER_EDGES:
        # 8 latent variables in [0, 1]: u0..u3 for edges, v0..v3 for widths
        # Guarantees ordering and all constraints by construction
        return [Real(0.0, 1.0, name=f'u{i}') for i in range(N_FILTERS)] + \
               [Real(0.0, 1.0, name=f'v{i}') for i in range(N_FILTERS)]
    else:
        return WIDTH_BOUNDS


def get_n_params() -> int:
    """Get number of optimization parameters."""
    # Edge mode uses 8 latent params (4 u for edges, 4 v for widths)
    return 2 * N_FILTERS if OPTIMIZE_FILTER_EDGES else N_FILTERS


def set_optimize_edges(enabled: bool) -> None:
    """Set whether filter edges are optimized (called from CLI parsing)."""
    global OPTIMIZE_FILTER_EDGES
    OPTIMIZE_FILTER_EDGES = enabled


def set_binary_table_path(path: str) -> None:
    """Set the path to the binary table (called from CLI parsing)."""
    global BINARY_TABLE_PATH
    BINARY_TABLE_PATH = path


def set_binary_table_limit(limit: int) -> None:
    """Set the number of rows to process from binary table (called from CLI parsing)."""
    global BINARY_TABLE_LIMIT
    BINARY_TABLE_LIMIT = limit


def decode_latent_to_filters(latent_params: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """
    Decode latent [0,1] parameters to valid filter edges and widths.
    
    Each edge is independently controlled within EDGE_RANGE of its default value,
    with ordering constraints enforced to maintain p1 <= p2 <= p3 <= p4.
    
    This parametrization guarantees all constraints are satisfied by construction:
    - FILTER_STARTING_EDGES[i] <= p_i <= FILTER_STARTING_EDGES[i] + EDGE_RANGE
    - p1 <= p2 <= p3 <= p4 (non-strict ordering, allows overlaps)
    - w_i >= MIN_FILTER_WIDTH
    - p_i + w_i <= MAX_WAVELENGTH
    
    Parameters
    ----------
    latent_params : np.ndarray
        8 values in [0, 1]: [u0, u1, u2, u3, v0, v1, v2, v3]
        u values control edges independently, v values control widths
    
    Returns
    -------
    tuple
        (edges, widths) - both as np.ndarray of length N_FILTERS
    """
    import numpy as np
    
    U = MAX_WAVELENGTH
    WMIN = MIN_FILTER_WIDTH
    
    u = latent_params[:N_FILTERS]
    v = latent_params[N_FILTERS:]
    
    # Clamp to [0, 1]
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # Step 1: Decode each edge independently within its allowed range
    default_edges = np.array(FILTER_STARTING_EDGES, dtype=np.float64)
    unconstrained_edges = np.array([
        default_edges[i] + u[i] * EDGE_RANGE
        for i in range(N_FILTERS)
    ], dtype=np.float64)
    
    # Step 2: Enforce ordering constraint: p1 <= p2 <= p3 <= p4
    edges = np.zeros(N_FILTERS, dtype=np.float64)
    edges[0] = unconstrained_edges[0]
    for i in range(1, N_FILTERS):
        # Ensure each edge is at least as large as the previous one
        edges[i] = max(unconstrained_edges[i], edges[i-1])
    
    # Step 3: Compute widths
    widths = np.empty(N_FILTERS, dtype=np.float64)
    for i, (pi, vi) in enumerate(zip(edges, v)):
        wmax = U - pi
        # Keep widths continuous to avoid optimizer quantization artifacts.
        widths[i] = WMIN + vi * (wmax - WMIN)
    
    return edges, widths


def constrain_filter_params(params: "np.ndarray") -> "np.ndarray":
    """
    Enforce hard wavelength constraints on filter parameters.
    
    If OPTIMIZE_FILTER_EDGES is True, decodes latent [0,1] parameters.
    Otherwise, uses fixed edges with width-only optimization.
    
    Ensures:
    - All filter edges are within [MIN_WAVELENGTH, P1_MAX] (for p1) or ordered (for p2-p4)
    - All filter widths are at least MIN_FILTER_WIDTH
    - All filter long-wavelength ends (edge + width) are <= MAX_WAVELENGTH
    - Non-strict ordering: p1 <= p2 <= p3 <= p4 (allows overlapping filters)
    
    Parameters
    ----------
    params : np.ndarray
        If OPTIMIZE_FILTER_EDGES: [u0, u1, u2, u3, v0, v1, v2, v3] in [0,1]
        Otherwise: [w1, w2, w3, w4]
    
    Returns
    -------
    np.ndarray
        Constrained parameters: [e1, e2, e3, e4, w1, w2, w3, w4]
        (edges + widths concatenated, regardless of input mode)
    """
    import numpy as np
    params = np.asarray(params, dtype=np.float64).copy()
    
    if OPTIMIZE_FILTER_EDGES:
        # Decode latent parameters to edges and widths
        edges, widths = decode_latent_to_filters(params)
        return np.concatenate([edges, widths])
    else:

        # Widths only, use fixed starting edges
        edges = np.array(FILTER_STARTING_EDGES, dtype=np.float64)
        widths = params
        
        # Constrain widths: ensure edge + width <= MAX_WAVELENGTH
        # and width >= MIN_FILTER_WIDTH
        max_widths = MAX_WAVELENGTH - edges
        widths = np.clip(widths, MIN_FILTER_WIDTH, max_widths)
        
        return np.concatenate([edges, widths])