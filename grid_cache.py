"""
Grid SED data cache and vectorized photometry computation.

Handles loading and caching of ~170k grid star SEDs with efficient
batch photometry calculation.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import floor, log10
from multiprocessing import cpu_count
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pyphot

import config
from config import (
    GRID_H5_PATH,
    N_NIR_COLORS,
    get_n_colors,
)
from photometry import get_gaia_filters, _to_number

# =============================================================================
# Grid Cache Class
# =============================================================================

class GridCache:
    """
    Cache for grid SED data and photometry.
    
    Loads grid SEDs once and computes photometry per filter configuration.
    Photometry caching disabled to prevent stale cache bugs.
    """
    
    def __init__(self, h5_path: str = GRID_H5_PATH):
        self.h5_path = h5_path
        self._load_grid_data()
        self._compute_grid_spacing()
        # Iteration counter for debugging
        self._photometry_call_count = 0
    
    def _load_grid_data(self):
        """Load grid SEDs into memory."""
        import logging
        logger = logging.getLogger("filter_optimization")
        logger.info(f"Loading grid SEDs from {self.h5_path}")
        t0 = time.time()
        
        with h5py.File(self.h5_path, 'r') as h5:
            self.wave_AA = h5['wave_AA'][:].astype(np.float64)
            self.flux_FLAM = h5['flux_FLAM'][:].astype(np.float64)
            self.teff = h5['teff'][:].astype(np.float64)
            self.logg = h5['logg'][:].astype(np.float64)
            self.feh = h5['feh'][:].astype(np.float64)
            self.radius = h5['radius_rsun'][:].astype(np.float64)
        
        self.n_stars = len(self.teff)
        
        logger.info(f"Loaded {self.n_stars} grid stars in {time.time()-t0:.1f}s")
    
    def _compute_grid_spacing(self):
        """Automatically determine grid spacing from unique parameter values."""
        import logging
        logger = logging.getLogger("filter_optimization")
        logger.info("Computing grid spacing from data...")
        
        teff_unique = np.unique(self.teff)
        logg_unique = np.unique(self.logg)
        feh_unique = np.unique(self.feh)
        
        def get_median_spacing(arr: np.ndarray) -> float:
            if len(arr) < 2:
                return 1.0
            diffs = np.diff(np.sort(arr))
            diffs = diffs[diffs > 1e-6]
            if len(diffs) == 0:
                return 1.0
            return float(np.median(diffs))
        
        def round_to_n_significant(x: float, n: int = 3) -> float:
            if x == 0:
                return 0.0
            return round(x, -int(floor(log10(abs(x)))) + (n - 1))
        
        self.teff_step = round_to_n_significant(get_median_spacing(teff_unique), 3)
        self.logg_step = round_to_n_significant(get_median_spacing(logg_unique), 3)
        self.feh_step = round_to_n_significant(get_median_spacing(feh_unique), 3)
        
        print(f"  Grid spacing: Teff={self.teff_step:.2f} K, "
              f"logg={self.logg_step:.3f} dex, "
              f"[Fe/H]={self.feh_step:.3f} dex")
        print(f"  Parameter ranges: Teff=[{teff_unique.min():.0f}, {teff_unique.max():.0f}], "
              f"logg=[{logg_unique.min():.1f}, {logg_unique.max():.1f}], "
              f"[Fe/H]=[{feh_unique.min():.1f}, {feh_unique.max():.1f}]")
    
    def get_grid_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (teff, logg, feh) arrays."""
        return self.teff, self.logg, self.feh
    
    def get_grid_spacing(self) -> Tuple[float, float, float]:
        """Return (teff_step, logg_step, feh_step)."""
        return self.teff_step, self.logg_step, self.feh_step
    
    def compute_grid_photometry(
        self,
        filter_map: Dict[str, any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitudes and colors for all grid stars.
        
        Uses vectorized get_flux_batch for efficiency.
        
        Parameters
        ----------
        filter_map : Dict[str, FilterAdapter]
            Dictionary of filter objects
        
        Returns
        -------
        mags : np.ndarray
            Shape (N_stars, N_filters)
        colors : np.ndarray
            Shape (N_stars, N_colors)
        """
        self._photometry_call_count += 1
        
        # Log filter configuration for debugging
        filter_info = ", ".join([
            f"{fname}=[{fobj.wavelength[0]:.0f}-{fobj.wavelength[-1]:.0f}]"
            for fname, fobj in sorted(filter_map.items())
        ])
        print(f"[INFO] Computing grid photometry (call #{self._photometry_call_count}): {filter_info}")
        t0 = time.time()
        
        filter_list = list(filter_map.values())
        n_filters = len(filter_list)
        
        mags = np.empty((self.n_stars, n_filters), dtype=np.float64)
        tiny = np.finfo(float).tiny
        
        # Process filters in parallel (numpy releases GIL)
        max_workers = min(n_filters, max(2, cpu_count() // 2))
        
        def compute_filter_mags(j_fobj_tuple):
            j, fobj = j_fobj_tuple
            t_filt = time.time()
            fluxes = fobj.get_flux_batch(self.wave_AA, self.flux_FLAM)
            zero_flux = _to_number(fobj.Vega_zero_flux)
            ratios = fluxes / zero_flux
            ratios = np.where((ratios > 0) & np.isfinite(ratios), ratios, tiny)
            filter_mags = -2.5 * np.log10(ratios)
            elapsed = time.time() - t_filt
            return j, filter_mags, elapsed
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(compute_filter_mags, (j, fobj)) 
                      for j, fobj in enumerate(filter_list)]
            
            for future in as_completed(futures):
                j, filter_mags, elapsed = future.result()
                mags[:, j] = filter_mags
                print(f"    Filter {j+1}/{n_filters} done in {elapsed:.1f}s")
        
        # Compute colors
        n_colors = get_n_colors()
        colors = np.empty((self.n_stars, n_colors), dtype=np.float64)
        
        color_idx = 0
        
        # Gaia colors if enabled
        if config.USE_GAIA_FILTERS:
            print("[INFO] Computing Gaia photometry for grid...")
            t_gaia = time.time()
            gaia_filters = get_gaia_filters()
            gaia_mags = np.empty((self.n_stars, 3), dtype=np.float64)
            
            wave_with_unit = self.wave_AA * pyphot.unit['AA']
            
            for j, (band, fobj) in enumerate(gaia_filters.items()):
                fluxes = fobj.get_flux(wave_with_unit, self.flux_FLAM * pyphot.unit['flam'], axis=-1)
                vega_flux = fobj.Vega_zero_flux
                ratios = fluxes / vega_flux
                if hasattr(ratios, 'magnitude'):
                    ratios = ratios.magnitude
                ratios = np.asarray(ratios, dtype=np.float64)
                ratios = np.where((ratios > 0) & np.isfinite(ratios), ratios, tiny)
                gaia_mags[:, j] = -2.5 * np.log10(ratios)
            
            colors[:, color_idx] = gaia_mags[:, 0] - gaia_mags[:, 1]  # BP - G
            color_idx += 1
            colors[:, color_idx] = gaia_mags[:, 1] - gaia_mags[:, 2]  # G - RP
            color_idx += 1
            
            print(f"    Gaia photometry done in {time.time()-t_gaia:.1f}s")
        
        # NIR colors
        for i in range(N_NIR_COLORS):
            colors[:, color_idx] = mags[:, i] - mags[:, i + 1]
            color_idx += 1
        
        print(f"[INFO] Grid photometry completed in {time.time()-t0:.1f}s")
        
        return mags, colors


# =============================================================================
# Global Grid Cache
# =============================================================================

_GRID_CACHE: Optional[GridCache] = None


def get_grid_cache() -> GridCache:
    """Get or initialize the global grid cache."""
    global _GRID_CACHE
    if _GRID_CACHE is None:
        _GRID_CACHE = GridCache()
    return _GRID_CACHE


def clear_grid_cache():
    """Clear the global grid cache."""
    global _GRID_CACHE
    _GRID_CACHE = None
