"""
Multiprocessing worker functions for binary star processing.

Handles worker initialization and binary SED → detection probability pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import BallTree

import config
from config import (
    K_NEIGHBORS,
    N_FILTERS,
    DEFAULT_APPARENT_MAG,
    get_n_colors,
)
from spectrum import get_binary_seds
from photometry import (
    compute_magnitudes_from_sed,
    compute_gaia_magnitudes_from_sed,
    compute_colors_from_mags,
)
from errors import compute_rmd_error, sigmoid_detection
from matching import (
    find_best_grid_match,
    refine_match_lbfgsb,
    compute_neighbor_rmd_error,
    build_ball_tree,
)

# =============================================================================
# Worker Global State
# =============================================================================

_WORKER_FILTER_MAP: Optional[Dict[str, any]] = None
_WORKER_WIDTHS: Optional[np.ndarray] = None
_WORKER_GRID_COLORS: Optional[np.ndarray] = None
_WORKER_GRID_TEFF: Optional[np.ndarray] = None
_WORKER_GRID_LOGG: Optional[np.ndarray] = None
_WORKER_GRID_FEH: Optional[np.ndarray] = None
_WORKER_BALL_TREE: Optional[BallTree] = None
_WORKER_GRID_SPACING: Optional[Tuple[float, float, float]] = None
_WORKER_VALID_INDICES: Optional[np.ndarray] = None
_WORKER_PARAM_INDEX: Optional[Dict] = None
_WORKER_BINARY_SEDS: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None


def _sanitize_positive(value: float, default: float) -> float:
    """Return a positive finite value, falling back to default otherwise."""
    if not np.isfinite(value) or value <= 0.0:
        return default
    return float(value)


def _sanitize_radius_rsun(radius: float) -> float:
    """Sanitize stellar radius in solar radii."""
    return _sanitize_positive(radius, 1.0)


def _sanitize_distance_pc(distance_pc: float) -> float:
    """Sanitize heliocentric distance in parsecs."""
    return _sanitize_positive(distance_pc, 10.0)


def worker_init(
    filter_map: Dict[str, any],
    widths: np.ndarray,
    grid_colors: np.ndarray,
    grid_teff: np.ndarray,
    grid_logg: np.ndarray,
    grid_feh: np.ndarray,
    grid_spacing: Tuple[float, float, float],
):
    """
    Initialize worker process with shared data.
    
    Called once per worker at Pool creation via initializer.
    Sets up global state for efficient binary processing.
    """
    global _WORKER_FILTER_MAP, _WORKER_WIDTHS
    global _WORKER_GRID_COLORS, _WORKER_GRID_TEFF, _WORKER_GRID_LOGG, _WORKER_GRID_FEH
    global _WORKER_BALL_TREE, _WORKER_GRID_SPACING, _WORKER_VALID_INDICES, _WORKER_PARAM_INDEX
    global _WORKER_BINARY_SEDS
    
    # Note: Debug logging removed to reduce log spam (128 workers × many iterations)
    # Verification can be done via: grid_colors.shape vs get_n_colors()
    
    _WORKER_FILTER_MAP = filter_map
    _WORKER_WIDTHS = np.asarray(widths, dtype=np.float64)
    if _WORKER_WIDTHS.shape[0] != N_FILTERS:
        raise ValueError(f"Expected {N_FILTERS} filter widths, got shape {_WORKER_WIDTHS.shape}")
    if not np.all(np.isfinite(_WORKER_WIDTHS)) or np.any(_WORKER_WIDTHS <= 0.0):
        raise ValueError("Filter widths must be finite and > 0")
    _WORKER_GRID_COLORS = grid_colors
    _WORKER_GRID_TEFF = grid_teff
    _WORKER_GRID_LOGG = grid_logg
    _WORKER_GRID_FEH = grid_feh
    _WORKER_GRID_SPACING = grid_spacing
    
    # Inherit precomputed binary SEDs (copy-on-write with fork)
    _WORKER_BINARY_SEDS = get_binary_seds()
    
    # Build BallTree for this worker
    _WORKER_BALL_TREE, _WORKER_VALID_INDICES = build_ball_tree(grid_colors)
    
    # Build parameter index for O(1) neighbor lookups
    teff_step, logg_step, feh_step = grid_spacing
    teff_min, logg_min, feh_min = grid_teff.min(), grid_logg.min(), grid_feh.min()
    
    teff_idx = np.round((grid_teff - teff_min) / teff_step).astype(np.int32)
    logg_idx = np.round((grid_logg - logg_min) / logg_step).astype(np.int32)
    feh_idx = np.round((grid_feh - feh_min) / feh_step).astype(np.int32)
    
    _WORKER_PARAM_INDEX = {
        '_meta': {
            'teff_min': teff_min, 'logg_min': logg_min, 'feh_min': feh_min,
            'teff_step': teff_step, 'logg_step': logg_step, 'feh_step': feh_step
        }
    }
    
    for i in range(len(grid_teff)):
        key = (int(teff_idx[i]), int(logg_idx[i]), int(feh_idx[i]))
        if key not in _WORKER_PARAM_INDEX:
            _WORKER_PARAM_INDEX[key] = i


def process_binary(args) -> Dict:
    """
    Process a single binary star.
    
    Pipeline: precomputed SED → photometry → BallTree match → 
              optional L-BFGS-B refinement → error computation → detection
    
    Parameters
    ----------
    args : tuple
        (idx, teff1, logg1, feh1, r1, teff2, logg2, feh2, r2, ref_mag, dist_pc)
    
    Returns
    -------
    dict
        Results including rmd, significance, detection_prob, etc.
    """
    (idx, teff1, logg1, feh1, r1, teff2, logg2, feh2, r2, ref_mag, dist_pc) = args
    
    result = {
        'idx': idx,
        'rmd': np.nan,
        'rmd_error': np.nan,
        'significance': np.nan,
        'detection_prob': 0.0,
        'best_teff': np.nan,
        'best_logg': np.nan,
        'best_feh': np.nan,
        'neighbor_error': 0.0,
    }
    
    # Localize frequently accessed globals
    binary_seds = _WORKER_BINARY_SEDS
    filter_map = _WORKER_FILTER_MAP
    widths = _WORKER_WIDTHS
    grid_colors = _WORKER_GRID_COLORS
    grid_teff = _WORKER_GRID_TEFF
    grid_logg = _WORKER_GRID_LOGG
    grid_feh = _WORKER_GRID_FEH
    grid_spacing = _WORKER_GRID_SPACING
    valid_indices = _WORKER_VALID_INDICES
    ball_tree = _WORKER_BALL_TREE
    param_index = _WORKER_PARAM_INDEX
    
    # Guard against missing binary SED cache
    if binary_seds is None or idx >= len(binary_seds):
        return result
    
    # Get precomputed binary SED (at 10pc) and scale to actual distance
    wave_AA, flux_binary_10pc = binary_seds[idx]
    dist_pc = _sanitize_distance_pc(float(dist_pc))
    flux_binary = flux_binary_10pc * (10.0 / dist_pc) ** 2
    binary_nir_mags = compute_magnitudes_from_sed(wave_AA, flux_binary, filter_map)
    binary_gaia_mags = compute_gaia_magnitudes_from_sed(wave_AA, flux_binary) if config.USE_GAIA_FILTERS else None
    binary_colors = compute_colors_from_mags(binary_nir_mags, binary_gaia_mags)
    
    if not np.all(np.isfinite(binary_colors)):
        return result
    
    # Dimension safety check
    if binary_colors.shape[0] != grid_colors.shape[1]:
        import logging
        logger = logging.getLogger("filter_optimization")
        logger.error(f"DIMENSION MISMATCH: binary_colors shape {binary_colors.shape} vs grid_colors shape {grid_colors.shape}")
        logger.error(f"USE_GAIA_FILTERS = {config.USE_GAIA_FILTERS}, expected n_colors = {get_n_colors()}")
        return result
    
    # Find best grid match
    if len(valid_indices) == 0:
        return result
    
    valid_colors = grid_colors[valid_indices]
    best_tree_idx, best_rmd = find_best_grid_match(
        binary_colors, 
        valid_colors, 
        ball_tree,
        k=min(K_NEIGHBORS, len(valid_indices))
    )
    
    if best_tree_idx < 0:
        return result
    
    # Map back to original grid index
    best_orig_idx = int(valid_indices[best_tree_idx])
    
    initial_teff = grid_teff[best_orig_idx]
    initial_logg = grid_logg[best_orig_idx]
    initial_feh = grid_feh[best_orig_idx]
    
    teff_step, logg_step, feh_step = grid_spacing
    
    # Refine or use grid minimum
    if config.USE_LBFGS_REFINEMENT:
        best_params, refined_rmd, best_single_mags = refine_match_lbfgsb(
            binary_colors,
            initial_teff, initial_logg, initial_feh,
            filter_map,
            teff_step, logg_step, feh_step,
        )
        neighbor_error = 0.0
    else:
        best_params = np.array([initial_teff, initial_logg, initial_feh])
        refined_rmd = best_rmd
        best_single_mags = {f"f{i+1}": np.nan for i in range(N_FILTERS)}
        
        neighbor_error = compute_neighbor_rmd_error(
            binary_colors, initial_teff, initial_logg, initial_feh,
            grid_colors, param_index
        )
    
    # Get matched grid colors for error calculation
    best_single_colors = grid_colors[best_orig_idx]
    
    # Compute RMD error (sign-dependent)
    rmd_error = compute_rmd_error(
        binary_nir_mags, widths,
        binary_colors, best_single_colors,
        gaia_mags=binary_gaia_mags,
        reference_mag=ref_mag
    )
    
    # Add neighbor error when not using refinement
    if not config.USE_LBFGS_REFINEMENT:
        rmd_error = np.sqrt(rmd_error**2 + neighbor_error**2)
    
    # Significance and detection
    significance = refined_rmd / rmd_error if rmd_error > 0 else 0.0
    detection_prob = sigmoid_detection(significance)
    
    result.update({
        'rmd': refined_rmd,
        'rmd_error': rmd_error,
        'significance': significance,
        'detection_prob': detection_prob,
        'best_teff': best_params[0],
        'best_logg': best_params[1],
        'best_feh': best_params[2],
        'neighbor_error': neighbor_error if not config.USE_LBFGS_REFINEMENT else 0.0,
    })
    
    # Add magnitudes and colors
    for k, v in binary_nir_mags.items():
        result[f'binary_{k}'] = v
    
    n_colors = get_n_colors()
    for i in range(n_colors):
        result[f'binary_c{i+1}'] = binary_colors[i]
    
    for k, v in best_single_mags.items():
        result[f'single_{k}'] = v
    
    if config.USE_GAIA_FILTERS and binary_gaia_mags is not None:
        for k, v in binary_gaia_mags.items():
            result[f'binary_gaia_{k}'] = v
    
    return result


# =============================================================================
# Task Preparation
# =============================================================================

def binary_row_to_task(idx: int, row, reference_mag: float = DEFAULT_APPARENT_MAG) -> tuple:
    """
    Convert a binary table row to a task tuple for process_binary.
    
    Parameters
    ----------
    idx : int
        Row index
    row : astropy.table Row
        Binary star parameters
    reference_mag : float
        Reference apparent magnitude for error scaling
    
    Returns
    -------
    tuple
        (idx, teff1, logg1, feh1, r1, teff2, logg2, feh2, r2, ref_mag, dist_pc)
    """
    r1 = _sanitize_radius_rsun(float(row.get('R1', row.get('R_1', row.get('radius_1', 1.0)))))
    r2 = _sanitize_radius_rsun(float(row.get('R2', row.get('R_2', row.get('radius_2', 1.0)))))
    dist_pc = _sanitize_distance_pc(float(row.get('helio_dist', 10.0)))
    
    return (
        idx,
        float(row['teff_1']), float(row['logg_1']), float(row['feh_1']), r1,
        float(row['teff_2']), float(row['logg_2']), float(row['feh_2']), r2,
        reference_mag,
        dist_pc
    )
