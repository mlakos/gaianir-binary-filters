"""
Photometric error model and detection rate computation.

Implements Gaia-based magnitude error scaling and sigmoid detection model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Sequence

import numpy as np
from pygaia.errors.photometric import magnitude_uncertainty

import config
from config import (
    GAIA_G_FWHM_AA,
    SYSTEMATIC_ERROR_MAG,
    DEFAULT_APPARENT_MAG,
    SIGMOID_C,
    SIGMOID_S0,
    MIN_FILTER_WIDTH,
    N_FILTERS,
    N_NIR_COLORS,
    get_n_colors,
)

# =============================================================================
# Cached Error Functions
# =============================================================================

@lru_cache(maxsize=4096)
def _g_error_mmag_cached(mag_clamped: float, release: str = "dr4") -> float:
    """Cached wrapper for Gaia G-band error computation."""
    try:
        return float(magnitude_uncertainty("g", np.array([mag_clamped]), release=release)[0])
    except Exception:
        # Fallback: simple approximation
        return float(0.3 * np.power(10.0, 0.04 * (mag_clamped - 12.0)))


@lru_cache(maxsize=4096)
def _gaia_band_error_cached(band: str, mag_clamped: float, release: str = "dr4") -> float:
    """
    Cached wrapper for Gaia band (bp, g, rp) error computation.
    
    Uses pygaia.errors.photometric.magnitude_uncertainty.
    """
    try:
        return float(magnitude_uncertainty(band, np.array([mag_clamped]), release=release)[0])
    except Exception:
        # Fallback: use G-band approximation
        return float(0.3 * np.power(10.0, 0.04 * (mag_clamped - 12.0)))


# =============================================================================
# Magnitude Error
# =============================================================================

def _sanitize_apparent_mag(mag: float, default: float = DEFAULT_APPARENT_MAG) -> float:
    """Sanitize apparent magnitude, replacing invalid values with default."""
    if not np.isfinite(mag):
        return default
    return float(mag)


def compute_magnitude_error(
    apparent_mag: float,
    filter_width_aa: float,
    gaia_release: str = "dr4",
) -> float:
    """
    Compute magnitude error scaled from Gaia G: σ ∝ √(FWHM_G / width).
    
    Physical basis: photon noise scales as 1/√N_photons, and N_photons ∝ bandwidth.
    Therefore σ_mag ∝ 1/√(bandwidth).
    
    Parameters
    ----------
    apparent_mag : float
        Apparent magnitude of the star
    filter_width_aa : float
        Filter width in Angstrom
    gaia_release : str
        Gaia data release for error model ("dr4" default)
    
    Returns
    -------
    float
        Total magnitude error (photon noise + systematic floor)
    """
    mag_clamped = np.clip(apparent_mag, 4.0, 21.0)
    g_error_mag = _g_error_mmag_cached(float(mag_clamped), gaia_release) / 1000.0
    # Use sqrt scaling: σ ∝ 1/√(bandwidth) since SNR ∝ √(N_photons) and N_photons ∝ bandwidth
    scaled_error = g_error_mag * np.sqrt(GAIA_G_FWHM_AA / filter_width_aa)
    return float(np.sqrt(scaled_error**2 + SYSTEMATIC_ERROR_MAG**2))


# =============================================================================
# RMD Error
# =============================================================================

def compute_rmd_error(
    apparent_mags: Dict[str, float],
    filter_widths: Sequence[float],
    binary_colors: np.ndarray,
    grid_colors: np.ndarray,
    gaia_mags: Optional[Dict[str, float]] = None,
    reference_mag: float = DEFAULT_APPARENT_MAG,
    gaia_release: str = "dr4",
) -> float:
    """
    Compute RMD uncertainty using sign-dependent error propagation.
    
    For independent magnitude errors Σ_f = diag(σ²_f1, σ²_f2, σ²_f3, σ²_f4),
    the RMD error depends on the signs of color residuals:
    
    σ_RMD² ≈ (1/N²)[σ²_f1 + (s₁-s₂)² σ²_f2 + (s₂-s₃)² σ²_f3 + σ²_f4]
    
    where s_i = sign(c_i^binary - c_i^grid) ∈ {-1, +1}.
    The term (s_i - s_{i+1})² is 0 if same sign, 4 if opposite signs.
    
    Parameters
    ----------
    apparent_mags : Dict[str, float]
        NIR filter apparent magnitudes
    filter_widths : Sequence[float]
        Filter widths in Angstrom (one per filter)
    binary_colors : np.ndarray
        Binary star colors (length N_colors)
    grid_colors : np.ndarray
        Matched grid star colors (length N_colors)
    gaia_mags : Dict[str, float], optional
        Gaia magnitudes if using Gaia colors
    reference_mag : float
        Fallback magnitude for invalid values
    gaia_release : str
        Gaia data release for error model
    
    Returns
    -------
    float
        RMD uncertainty (sign-dependent error propagation)
    """
    filter_widths_arr = np.asarray(filter_widths, dtype=np.float64)
    if filter_widths_arr.shape[0] < N_FILTERS:
        return np.nan
    filter_widths_arr = filter_widths_arr[:N_FILTERS]
    filter_widths_arr = np.where(
        np.isfinite(filter_widths_arr) & (filter_widths_arr > 0.0),
        filter_widths_arr,
        MIN_FILTER_WIDTH,
    )

    # Compute color residuals and their signs
    color_residuals = binary_colors - grid_colors
    
    # Compute NIR magnitude errors
    nir_mag_errors = []
    for i in range(N_FILTERS):
        mag = _sanitize_apparent_mag(apparent_mags.get(f"f{i+1}", reference_mag), reference_mag)
        sigma_mag = compute_magnitude_error(mag, filter_widths_arr[i])
        nir_mag_errors.append(sigma_mag)
    
    # Start with contributions from edge filters (always included)
    variance_sum = nir_mag_errors[0]**2 + nir_mag_errors[-1]**2
    
    # Determine starting index for NIR colors
    nir_color_start = 2 if (config.USE_GAIA_FILTERS and gaia_mags is not None) else 0
    
    # Add contributions from middle filters based on residual signs
    # For NIR: (s_i - s_{i+1})² is 0 if same sign, 4 if opposite
    for i in range(N_NIR_COLORS - 1):
        color_idx = nir_color_start + i
        color_idx_next = nir_color_start + i + 1
        
        # Get signs of adjacent color residuals
        s_i = np.sign(color_residuals[color_idx]) if np.isfinite(color_residuals[color_idx]) else 1.0
        s_next = np.sign(color_residuals[color_idx_next]) if np.isfinite(color_residuals[color_idx_next]) else 1.0
        
        # Handle zero residuals (treat as positive)
        if s_i == 0:
            s_i = 1.0
        if s_next == 0:
            s_next = 1.0
        
        # Middle filter i+1 contributes: (s_i - s_{i+1})² × σ²_{f_{i+1}}
        sign_term = (s_i - s_next)**2
        variance_sum += sign_term * nir_mag_errors[i + 1]**2
    
    # Add Gaia contribution if present (treat as uncorrelated for simplicity)
    if config.USE_GAIA_FILTERS and gaia_mags is not None:
        bp_mag = _sanitize_apparent_mag(gaia_mags.get('bp', reference_mag), reference_mag)
        g_mag = _sanitize_apparent_mag(gaia_mags.get('g', reference_mag), reference_mag)
        rp_mag = _sanitize_apparent_mag(gaia_mags.get('rp', reference_mag), reference_mag)
        
        # Individual magnitude errors: σ_tot² = σ_phot² + σ_sys²
        sigma_bp_phot = _gaia_band_error_cached("bp", float(np.clip(bp_mag, 4.0, 21.0)), gaia_release) / 1000.0
        sigma_g_phot = _gaia_band_error_cached("g", float(np.clip(g_mag, 4.0, 21.0)), gaia_release) / 1000.0
        sigma_rp_phot = _gaia_band_error_cached("rp", float(np.clip(rp_mag, 4.0, 21.0)), gaia_release) / 1000.0
        
        sigma_bp = float(np.sqrt(sigma_bp_phot**2 + SYSTEMATIC_ERROR_MAG**2))
        sigma_g = float(np.sqrt(sigma_g_phot**2 + SYSTEMATIC_ERROR_MAG**2))
        sigma_rp = float(np.sqrt(sigma_rp_phot**2 + SYSTEMATIC_ERROR_MAG**2))
        
        # Apply sign-dependent formula to Gaia colors as well
        # Gaia colors are first two: c0 = BP-G, c1 = G-RP
        s0 = np.sign(color_residuals[0]) if np.isfinite(color_residuals[0]) else 1.0
        s1 = np.sign(color_residuals[1]) if np.isfinite(color_residuals[1]) else 1.0
        if s0 == 0:
            s0 = 1.0
        if s1 == 0:
            s1 = 1.0
        
        # For BP-G color: edge term (BP always contributes)
        variance_sum += sigma_bp**2
        
        # G magnitude: middle term with sign dependence
        sign_term_g = (s0 - s1)**2  # 0 if same sign, 4 if opposite
        variance_sum += sign_term_g * sigma_g**2
        
        # For G-RP color: edge term (RP always contributes)
        variance_sum += sigma_rp**2
    
    # RMD error: σ_RMD = √(variance_sum) / N
    n_colors = get_n_colors()
    sigma_rmd = np.sqrt(variance_sum) / n_colors
    
    return float(sigma_rmd)


# =============================================================================
# Detection Model
# =============================================================================

def sigmoid_detection(significance: float, c: float = SIGMOID_C, s0: float = SIGMOID_S0) -> float:
    """
    Compute detection probability using sigmoid function.
    
    D = 1 / (1 + exp(-C * (s - s0)))
    
    Parameters
    ----------
    significance : float
        Detection significance (RMD / σ_RMD)
    c : float
        Sigmoid steepness (default from config)
    s0 : float
        Detection threshold in sigma (default from config)
    
    Returns
    -------
    float
        Detection probability in [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-c * (significance - s0)))


def compute_detection_rate(significances: np.ndarray) -> float:
    """
    Compute total detection rate as sum of sigmoid probabilities.
    
    Parameters
    ----------
    significances : np.ndarray
        Array of detection significances
    
    Returns
    -------
    float
        Sum of detection probabilities
    """
    return float(np.sum(sigmoid_detection(significances)))
