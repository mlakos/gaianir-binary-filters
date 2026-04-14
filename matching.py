"""
Grid matching using BallTree and L-BFGS-B refinement.

Implements fast k-NN color matching and optional continuous refinement.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import BallTree

import config
from config import (
    K_NEIGHBORS,
    RSUN_OVER_10PC_SQ,
    CK04_TEFF_MIN, CK04_TEFF_MAX, CK04_LOGG_MAX,
    CK04_FEH_MIN, CK04_FEH_MAX,
    N_FILTERS,
    is_within_ck04_bounds,
)
from spectrum import ck04_spectrum_cached
from photometry import (
    compute_magnitudes_from_sed,
    compute_gaia_magnitudes_from_sed,
    compute_colors_from_mags,
    compute_rmd,
)

# =============================================================================
# BallTree Matching
# =============================================================================

def build_ball_tree(grid_colors: np.ndarray) -> Tuple[BallTree, np.ndarray]:
    """
    Build BallTree from valid grid colors.
    
    Parameters
    ----------
    grid_colors : np.ndarray
        Grid star colors of shape (N_stars, N_colors)
    
    Returns
    -------
    ball_tree : BallTree
        Tree built on valid colors (Manhattan metric)
    valid_indices : np.ndarray
        Indices mapping tree rows back to original grid
    """
    valid_mask = np.all(np.isfinite(grid_colors), axis=1)
    valid_colors = grid_colors[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    ball_tree = BallTree(valid_colors, metric='manhattan')
    return ball_tree, valid_indices


def find_best_grid_match(
    binary_colors: np.ndarray,
    grid_colors: np.ndarray,
    ball_tree: BallTree,
    k: int = K_NEIGHBORS,
) -> Tuple[int, float]:
    """
    Find best matching grid star using BallTree.
    
    Parameters
    ----------
    binary_colors : np.ndarray
        Target colors to match (shape: N_colors)
    grid_colors : np.ndarray
        Grid colors to search (already filtered to valid rows)
    ball_tree : BallTree
        Pre-built tree on grid_colors
    k : int
        Number of neighbors to check
    
    Returns
    -------
    best_idx : int
        Index of best match in grid_colors (-1 if failed)
    best_rmd : float
        RMD at best match (inf if failed)
    """
    if not np.all(np.isfinite(binary_colors)):
        return -1, np.inf
    
    # Query k nearest neighbors
    binary_colors_2d = binary_colors.reshape(1, -1)
    k_actual = min(k, len(grid_colors))
    _, indices = ball_tree.query(binary_colors_2d, k=k_actual)
    
    indices = indices[0]
    
    # Compute exact RMD for candidates
    best_rmd = np.inf
    best_idx = -1
    
    for idx in indices:
        if 0 <= idx < len(grid_colors):
            rmd = compute_rmd(binary_colors, grid_colors[idx])
            if rmd < best_rmd:
                best_rmd = rmd
                best_idx = int(idx)
    
    return best_idx, best_rmd


# =============================================================================
# L-BFGS-B Refinement
# =============================================================================

def refine_match_lbfgsb(
    binary_colors: np.ndarray,
    initial_teff: float,
    initial_logg: float,
    initial_feh: float,
    filter_map: Dict[str, any],
    teff_step: float,
    logg_step: float,
    feh_step: float,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Refine the grid match using L-BFGS-B within grid cell bounds.
    
    Parameters
    ----------
    binary_colors : np.ndarray
        Target colors to match
    initial_teff, initial_logg, initial_feh : float
        Starting point from grid search
    filter_map : Dict
        Filter objects for photometry
    teff_step, logg_step, feh_step : float
        Grid spacing (defines bounds as ±step from initial)
    
    Returns
    -------
    best_params : np.ndarray
        [teff, logg, feh] of refined match
    best_rmd : float
        RMD at refined position
    best_mags : Dict[str, float]
        Magnitudes of the best matching single star
    """
    # Define bounds within one grid cell
    bounds = [
        (max(CK04_TEFF_MIN, initial_teff - teff_step),
         min(CK04_TEFF_MAX, initial_teff + teff_step)),
        (max(0.0 if initial_teff <= 6000 else 0.5, initial_logg - logg_step),
         min(CK04_LOGG_MAX, initial_logg + logg_step)),
        (max(CK04_FEH_MIN, initial_feh - feh_step),
         min(CK04_FEH_MAX, initial_feh + feh_step)),
    ]
    
    def objective(params: np.ndarray) -> float:
        teff, logg, feh = params
        if not is_within_ck04_bounds(teff, logg, feh):
            return 1000.0
        
        try:
            wave_AA, flux = ck04_spectrum_cached(teff, logg, feh)
            # Scale to 1 R_sun at 10 pc
            flux = flux * RSUN_OVER_10PC_SQ
            
            nir_mags = compute_magnitudes_from_sed(wave_AA, flux, filter_map)
            gaia_mags = compute_gaia_magnitudes_from_sed(wave_AA, flux) if config.USE_GAIA_FILTERS else None
            colors = compute_colors_from_mags(nir_mags, gaia_mags)
            return compute_rmd(binary_colors, colors)
        except Exception:
            return 1000.0
    
    # Optimize
    x0 = np.array([initial_teff, initial_logg, initial_feh], dtype=np.float64)
    
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 32, "ftol": 1e-4, "gtol": 1e-4},
    )
    
    best_params = result.x
    best_rmd = result.fun
    
    # Compute final magnitudes at best params
    try:
        wave_AA, flux = ck04_spectrum_cached(best_params[0], best_params[1], best_params[2])
        flux = flux * RSUN_OVER_10PC_SQ
        best_mags = compute_magnitudes_from_sed(wave_AA, flux, filter_map)
    except Exception:
        best_mags = {f"f{i+1}": np.nan for i in range(N_FILTERS)}
    
    return best_params, best_rmd, best_mags


# =============================================================================
# Neighbor RMD Error
# =============================================================================

def compute_neighbor_rmd_error(
    binary_colors: np.ndarray,
    center_teff: float,
    center_logg: float,
    center_feh: float,
    grid_colors: np.ndarray,
    param_index: dict,
) -> float:
    """
    Compute additional RMD error from 26 neighboring grid stars.
    
    Uses precomputed parameter index for O(1) neighbor lookups.
    
    Parameters
    ----------
    binary_colors : np.ndarray
        Binary star colors
    center_teff, center_logg, center_feh : float
        Center grid point parameters
    grid_colors : np.ndarray
        Full grid color array
    param_index : dict
        Parameter index from worker init
    
    Returns
    -------
    float
        Half of mean RMD to neighbor grid stars (0 if no neighbors found)
    """
    meta = param_index['_meta']
    teff_step = meta['teff_step']
    logg_step = meta['logg_step']
    feh_step = meta['feh_step']
    teff_min = meta['teff_min']
    logg_min = meta['logg_min']
    feh_min = meta['feh_min']
    
    # Convert center to grid indices
    center_ti = int(round((center_teff - teff_min) / teff_step))
    center_li = int(round((center_logg - logg_min) / logg_step))
    center_fi = int(round((center_feh - feh_min) / feh_step))
    
    # Collect neighbor colors
    neighbor_colors_list = []
    
    for dt in (-1, 0, 1):
        for dl in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dt == 0 and dl == 0 and df == 0:
                    continue
                
                key = (center_ti + dt, center_li + dl, center_fi + df)
                grid_idx = param_index.get(key)
                
                if grid_idx is not None:
                    colors = grid_colors[grid_idx]
                    if np.all(np.isfinite(colors)):
                        neighbor_colors_list.append(colors)
    
    if len(neighbor_colors_list) == 0:
        return 0.0
    
    # Vectorized RMD computation
    neighbor_colors = np.array(neighbor_colors_list)
    diffs = np.abs(neighbor_colors - binary_colors)
    rmds = np.mean(diffs, axis=1)
    
    return float(np.mean(rmds)) / 2.0
