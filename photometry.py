"""
Photometry and color computation for synthetic spectra.

Handles Vega magnitude calculation for custom tophat and Gaia filters.
"""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import Dict, Optional, Sequence

import numpy as np
import pyphot

import config
from config import N_FILTERS, N_NIR_COLORS, get_n_colors

# =============================================================================
# Output Suppression
# =============================================================================

class suppress_output:
    """Context manager to suppress stdout and stderr from noisy libraries."""
    
    def __enter__(self):
        self._stdout = StringIO()
        self._stderr = StringIO()
        self._redirect_stdout = redirect_stdout(self._stdout)
        self._redirect_stderr = redirect_stderr(self._stderr)
        self._redirect_stdout.__enter__()
        self._redirect_stderr.__enter__()
        return self
    
    def __exit__(self, *args):
        self._redirect_stdout.__exit__(*args)
        self._redirect_stderr.__exit__(*args)


def _to_number(val):
    """Extract numeric value from pyphot quantity or return as-is."""
    if hasattr(val, 'magnitude'):
        return float(val.magnitude)
    return float(val)


# =============================================================================
# Gaia Filter Cache
# =============================================================================

_GAIA_FILTERS: Optional[Dict[str, any]] = None


def get_gaia_filters() -> Dict[str, any]:
    """Load and cache Gaia BP, G, RP filters from pyphot."""
    global _GAIA_FILTERS
    if _GAIA_FILTERS is None:
        lib = pyphot.get_library()
        _GAIA_FILTERS = {
            'bp': lib['GaiaDR2v2_BP'],
            'g': lib['GaiaDR2v2_G'],
            'rp': lib['GaiaDR2v2_RP'],
        }
        import logging
        logger = logging.getLogger("filter_optimization")
        logger.info("Loaded Gaia DR2v2 filters: BP, G, RP")
    return _GAIA_FILTERS


# =============================================================================
# Magnitude Computation
# =============================================================================

def compute_magnitudes_from_sed(
    wave_AA: np.ndarray,
    flux_FLAM: np.ndarray,
    filter_map: Dict[str, any],
) -> Dict[str, float]:
    """
    Compute Vega magnitudes from an SED for all custom filters.
    
    Parameters
    ----------
    wave_AA : np.ndarray
        Wavelength array in Angstrom
    flux_FLAM : np.ndarray
        Flux array in erg/s/cm²/Å
    filter_map : Dict[str, FilterAdapter]
        Dictionary of filter name -> FilterAdapter objects
    
    Returns
    -------
    Dict[str, float]
        Filter name -> Vega magnitude
    """
    tiny = np.finfo(float).tiny
    mags = {}
    for fname, fobj in filter_map.items():
        with suppress_output():
            f_flux = fobj.get_flux(wave_AA, flux_FLAM)
        r = _to_number(f_flux / fobj.Vega_zero_flux)
        mags[fname] = -2.5 * np.log10(r if (np.isfinite(r) and r > 0.0) else tiny)
    return mags


def compute_gaia_magnitudes_from_sed(
    wave_AA: np.ndarray,
    flux_FLAM: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Gaia BP, G, RP magnitudes from an SED.
    
    Parameters
    ----------
    wave_AA : np.ndarray
        Wavelength array in Angstrom
    flux_FLAM : np.ndarray
        Flux array in erg/s/cm²/Å
    
    Returns
    -------
    Dict[str, float]
        Band name ('bp', 'g', 'rp') -> Vega magnitude
    """
    gaia_filters = get_gaia_filters()
    tiny = np.finfo(float).tiny
    mags = {}
    
    wave_with_unit = wave_AA * pyphot.unit['AA']
    flux_with_unit = flux_FLAM * pyphot.unit['flam']
    
    for fname, fobj in gaia_filters.items():
        try:
            f_flux = fobj.get_flux(wave_with_unit, flux_with_unit, axis=-1)
            ratio = f_flux / fobj.Vega_zero_flux
            r = float(ratio.magnitude) if hasattr(ratio, 'magnitude') else float(ratio)
            mags[fname] = -2.5 * np.log10(r if (np.isfinite(r) and r > 0.0) else tiny)
        except Exception:
            mags[fname] = np.nan
    
    return mags


# =============================================================================
# Color Computation
# =============================================================================

def compute_colors_from_mags(
    nir_mags: Dict[str, float],
    gaia_mags: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute color array from magnitudes.
    
    When USE_GAIA_FILTERS is True, returns [BP-G, G-RP, f1-f2, f2-f3, f3-f4].
    Otherwise returns [f1-f2, f2-f3, f3-f4].
    
    Parameters
    ----------
    nir_mags : Dict[str, float]
        NIR filter magnitudes (f1, f2, f3, f4, f5)
    gaia_mags : Dict[str, float], optional
        Gaia magnitudes ('bp', 'g', 'rp') if using Gaia colors
    
    Returns
    -------
    np.ndarray
        Color array of length get_n_colors()
    """
    colors = np.empty(get_n_colors(), dtype=np.float64)
    idx = 0
    
    if config.USE_GAIA_FILTERS and gaia_mags is not None:
        colors[idx] = gaia_mags['bp'] - gaia_mags['g']
        colors[idx + 1] = gaia_mags['g'] - gaia_mags['rp']
        idx += 2
    
    for i in range(N_NIR_COLORS):
        colors[idx] = nir_mags[f"f{i+1}"] - nir_mags[f"f{i+2}"]
        idx += 1
    
    return colors


def compute_rmd(colors1: np.ndarray, colors2: np.ndarray) -> float:
    """
    Compute RMD (Reduced Manhattan Distance) between two color arrays.
    
    RMD = mean(|c1 - c2|) = Manhattan_distance / n_colors
    
    This is the Manhattan distance normalized by the number of dimensions,
    making it comparable across different filter configurations.
    """
    return float(np.mean(np.abs(colors1 - colors2)))
