"""
Spectrum generation and caching for CK04 model atmospheres.

Handles single-star and binary SED computation with LRU caching.
"""

from __future__ import annotations

import time
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import numpy as np
from astropy.table import Table
from astropy import units as u
from stsynphot.catalog import grid_to_spec
from synphot import units as syu

from config import RSUN_M, TEN_PC_M

# =============================================================================
# Global Binary SED Cache (populated before fork, inherited via COW)
# =============================================================================
_BINARY_SEDS: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None


def _sanitize_positive(value: float, default: float) -> float:
    """Return a positive finite value, falling back to default otherwise."""
    if not np.isfinite(value) or value <= 0.0:
        return default
    return float(value)


def _sanitize_radius_rsun(radius: float) -> float:
    """Sanitize stellar radius in solar radii."""
    return _sanitize_positive(radius, 1.0)


def _sanitize_distance_pc(distance_pc: float) -> float:
    """Sanitize distance in parsecs."""
    return _sanitize_positive(distance_pc, 10.0)


def get_binary_seds() -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """Get the precomputed binary SEDs (for worker inheritance)."""
    return _BINARY_SEDS


def set_binary_seds(seds: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Set the binary SEDs cache (called in parent before fork)."""
    global _BINARY_SEDS
    _BINARY_SEDS = seds


# =============================================================================
# CK04 Spectrum Cache
# =============================================================================

@lru_cache(maxsize=8192)
def ck04_spectrum(teff: float, logg: float, feh: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build CK04 spectrum and return (wave_AA, flux_FLAM) with LRU caching.
    
    Parameters are rounded before caching for better hit rate.
    Returns surface flux (not scaled by radius).
    """
    sp = grid_to_spec('ck04models', t_eff=float(teff), metallicity=float(feh), log_g=float(logg))
    wave_q = sp.waveset
    flux_q = sp(wave_q, flux_unit=syu.FLAM)
    wave_AA = np.asarray(wave_q.to_value(u.AA), dtype=np.float64)
    flux_FLAM = np.asarray(flux_q.to_value(syu.FLAM), dtype=np.float64)
    return wave_AA, flux_FLAM


def ck04_spectrum_cached(teff: float, logg: float, feh: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get CK04 spectrum with parameter rounding for cache efficiency."""
    return ck04_spectrum(round(teff, 1), round(logg, 2), round(feh, 2))


# =============================================================================
# Binary SED Computation
# =============================================================================

def compute_binary_sed(
    teff1: float, logg1: float, feh1: float, r1: float,
    teff2: float, logg2: float, feh2: float, r2: float,
    distance_pc: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute combined SED for a binary star system.
    
    Parameters
    ----------
    teff1, logg1, feh1, r1 : float
        Primary star parameters (Teff in K, logg in dex, [Fe/H], radius in R_sun)
    teff2, logg2, feh2, r2 : float
        Secondary star parameters
    distance_pc : float
        Distance in parsecs (default 10 pc for absolute magnitudes)
    
    Returns
    -------
    wave_AA : np.ndarray
        Wavelength array in Angstrom
    flux_FLAM : np.ndarray
        Combined flux in erg/s/cm²/Å at the specified distance
    """
    r1 = _sanitize_radius_rsun(r1)
    r2 = _sanitize_radius_rsun(r2)
    distance_pc = _sanitize_distance_pc(distance_pc)

    # Primary spectrum (surface flux)
    wave_AA, flux1_surf = ck04_spectrum_cached(teff1, logg1, feh1)
    ang1 = (RSUN_M * r1 / TEN_PC_M) ** 2
    
    # Secondary spectrum (same wavelength grid)
    _, flux2_surf = ck04_spectrum_cached(teff2, logg2, feh2)
    ang2 = (RSUN_M * r2 / TEN_PC_M) ** 2
    
    # Combined flux at 10 pc
    flux_combined = flux1_surf * ang1 + flux2_surf * ang2
    
    # Scale to actual distance if needed
    if distance_pc != 10.0:
        flux_combined *= (10.0 / distance_pc) ** 2
    
    return wave_AA, flux_combined


def compute_single_sed(
    teff: float, logg: float, feh: float, radius: float = 1.0,
    distance_pc: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SED for a single star.
    
    Returns (wave_AA, flux_FLAM) at the specified distance.
    """
    radius = _sanitize_radius_rsun(radius)
    distance_pc = _sanitize_distance_pc(distance_pc)

    wave_AA, flux_surf = ck04_spectrum_cached(teff, logg, feh)
    ang = (RSUN_M * radius / TEN_PC_M) ** 2
    flux = flux_surf * ang
    
    if distance_pc != 10.0:
        flux *= (10.0 / distance_pc) ** 2
    
    return wave_AA, flux


# =============================================================================
# Precomputation for Multiprocessing
# =============================================================================

def _compute_single_binary_sed(args):
    """Worker function to compute one binary SED (for multiprocessing)."""
    idx, teff1, logg1, feh1, r1, teff2, logg2, feh2, r2 = args
    wave, flux = compute_binary_sed(teff1, logg1, feh1, r1, teff2, logg2, feh2, r2, distance_pc=10.0)
    return idx, wave, flux


def precompute_binary_seds(binary_table: Table) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Precompute SEDs for all binaries in the table (parallelized).
    
    Called once in parent process before forking. Workers inherit
    the cached SEDs via copy-on-write, avoiding repeated computation.
    
    Returns
    -------
    List of (wave_AA, flux_FLAM) tuples, one per binary
    """
    import logging
    logger = logging.getLogger("filter_optimization")
    
    n_binaries = len(binary_table)
    logger.info(f"Precomputing binary SEDs for {n_binaries} binaries in parallel...")
    t0 = time.time()
    
    # Prepare arguments for parallel processing
    tasks = []
    invalid_r1 = 0
    invalid_r2 = 0
    for i, row in enumerate(binary_table):
        r1_raw = float(row.get('R1', row.get('R_1', row.get('radius_1', 1.0))))
        r2_raw = float(row.get('R2', row.get('R_2', row.get('radius_2', 1.0))))
        if not np.isfinite(r1_raw) or r1_raw <= 0.0:
            invalid_r1 += 1
        if not np.isfinite(r2_raw) or r2_raw <= 0.0:
            invalid_r2 += 1

        r1 = _sanitize_radius_rsun(r1_raw)
        r2 = _sanitize_radius_rsun(r2_raw)
        tasks.append((
            i,
            float(row['teff_1']), float(row['logg_1']), float(row['feh_1']), r1,
            float(row['teff_2']), float(row['logg_2']), float(row['feh_2']), r2
        ))

    if invalid_r1 > 0 or invalid_r2 > 0:
        logger.warning(
            "Radius fallback to 1.0 R_sun applied for invalid values: "
            f"primary={invalid_r1}, secondary={invalid_r2}"
        )
    
    # Parallelize SED computation
    n_workers = min(cpu_count(), n_binaries)
    logger.info(f"  Using {n_workers} workers for SED precomputation...")
    
    results = []
    with Pool(processes=n_workers) as pool:
        chunksize = max(1, n_binaries // (n_workers * 4))
        for i, result in enumerate(pool.imap(_compute_single_binary_sed, tasks, chunksize=chunksize)):
            results.append(result)
            
            # Progress logging every 200 binaries
            if (i + 1) % 200 == 0 or (i + 1) == n_binaries:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_binaries - (i + 1)) / rate if rate > 0 else 0
                logger.info(f"  Progress: {i+1}/{n_binaries} binaries ({(i+1)/n_binaries*100:.1f}%) | "
                           f"Rate: {rate:.1f} binaries/s | ETA: {eta:.1f}s")
    
    # Sort by index and extract SEDs
    results.sort(key=lambda x: x[0])
    seds = [(wave, flux) for _, wave, flux in results]
    
    elapsed = time.time() - t0
    logger.info(f"Precomputed {len(seds)} binary SEDs in {elapsed:.1f}s "
               f"({elapsed/len(seds)*1000:.1f}ms per binary)")
    
    # Memory estimate
    if seds:
        bytes_per_sed = seds[0][0].nbytes + seds[0][1].nbytes
        total_mb = (bytes_per_sed * len(seds)) / (1024**2)
        logger.info(f"Binary SED cache size: ~{total_mb:.1f} MB")
    
    # Store globally for worker inheritance
    set_binary_seds(seds)
    return seds


def get_ck04_cache_info():
    """Return LRU cache statistics for debugging."""
    return ck04_spectrum.cache_info()


def clear_ck04_cache():
    """Clear the CK04 spectrum cache."""
    ck04_spectrum.cache_clear()
