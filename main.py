#!/usr/bin/env python3
"""
Gaia NIR Filter Width Optimization for Binary Star Detection.

Main entry point for the filter optimization pipeline. Uses modular
components for spectrum generation, photometry, error modeling, and
grid matching.

Usage:
    python main.py [--gaia] [--no-refine]

Modules:
    config.py      - All constants and configuration
    errors.py      - Photometric error model and detection rate
    grid_cache.py  - Grid SED caching and batch photometry
    matching.py    - BallTree matching and L-BFGS-B refinement
    photometry.py  - Magnitude and color computation
    spectrum.py    - CK04 spectrum generation and binary SED computation    
    workers.py     - Multiprocessing worker functions
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence

import numpy as np

# =============================================================================
# Logging Setup
# =============================================================================

# Create run-specific log file with timestamp for crash safety
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Store logs in logs/ subdirectory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"optimization_log_{RUN_TIMESTAMP}.log")
ITERATION_LOG_FILE = os.path.join(LOGS_DIR, f"iterations_{RUN_TIMESTAMP}.csv")

# Create logger
logger = logging.getLogger("filter_optimization")
logger.setLevel(logging.DEBUG)

# Console handler (INFO level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# File handler (DEBUG level, immediate flush)
file_handler = logging.FileHandler(LOG_FILE, mode='a')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# Iteration counter for tracking objective function calls
ITERATION_COUNTER = 0


def init_iteration_log():
    """Initialize the iteration CSV log file with header."""
    if config.OPTIMIZE_FILTER_EDGES:
        edge_cols = ",".join(f"e{i+1}" for i in range(N_FILTERS))
        width_cols = ",".join(f"w{i+1}" for i in range(N_FILTERS))
        param_cols = f"{edge_cols},{width_cols}"
    else:
        param_cols = ",".join(f"w{i+1}" for i in range(N_FILTERS))
    
    header = (
        f"iteration,timestamp,{param_cols},"
        "detection_rate,detection_rate_valid,n_detected,valid_count,avg_significance,duration_s\n"
    )
    with open(ITERATION_LOG_FILE, 'w') as f:
        f.write(header)
        f.flush()
        os.fsync(f.fileno())


def log_iteration(iteration: int, params: np.ndarray, detection_rate: float,
                  detection_rate_valid: float, n_detected: int, valid_count: int, 
                  avg_significance: float, duration: float):
    """Log iteration data to CSV file with immediate flush for crash safety."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Decode latent params to actual edge/width values for logging
    if config.OPTIMIZE_FILTER_EDGES:
        from config import decode_latent_to_filters
        edges, widths = decode_latent_to_filters(params)
        # Log edges first, then widths (matching header order)
        decoded_params = np.concatenate([edges, widths])
        param_str = ",".join(f"{p:.1f}" for p in decoded_params)
    else:
        param_str = ",".join(f"{p:.1f}" for p in params)
    
    line = (
        f"{iteration},{timestamp},{param_str},"
        f"{detection_rate:.6f},{detection_rate_valid:.6f},{n_detected},{valid_count},"
        f"{avg_significance:.4f},{duration:.2f}\n"
    )
    
    with open(ITERATION_LOG_FILE, 'a') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


from astropy.table import Table
from skopt import forest_minimize
from scipy.optimize import minimize

# Local modules
import config
from config import (
    FILTER_STARTING_EDGES,
    N_FILTERS,
    N_NIR_COLORS,
    N_WORKERS,
    N_OPTIMIZER_CALLS,
    N_RANDOM_STARTS,
    get_n_initial_points,
    K_NEIGHBORS,
    WIDTH_BOUNDS,
    ACQ_FUNC,
    XI,
    KAPPA,
    SIGMOID_C,
    SIGMOID_S0,
    REFINE_OPTIMUM,
    REFINE_MAXITER,
    DEFAULT_APPARENT_MAG,
    GAIA_G_FWHM_AA,
    SYSTEMATIC_ERROR_MMAG,
    PATH_TO_BINARY_TABLE,
    TEST_MODE,
    TEST_BINARY_LIMIT,
    TEST_OPTIMIZER_CALLS,
    MIN_WAVELENGTH,
    MAX_WAVELENGTH,
    P1_MAX,
    get_n_colors,
    get_optimization_bounds,
    get_n_params,
    set_gaia_filters,
    set_lbfgs_refinement,
    set_optimize_edges,
    set_binary_table_path,
    set_binary_table_limit,
    set_n_random_starts,
    set_n_optimizer_calls,
    ck04_bounds_mask,
    constrain_filter_params,
    decode_latent_to_filters,
)
from spectrum import precompute_binary_seds
from grid_cache import get_grid_cache
from workers import worker_init, process_binary, binary_row_to_task
from custom_filter_lib_mod import make_tophat, FilterAdapter

# =============================================================================
# Binary Table Filtering
# =============================================================================

def filter_table_by_ck04_bounds(table: Table) -> Table:
    """Filter table to binaries where both components are within CK04 bounds."""
    teff_1 = np.asarray(table['teff_1'], dtype=np.float64)
    logg_1 = np.asarray(table['logg_1'], dtype=np.float64)
    feh_1 = np.asarray(table['feh_1'], dtype=np.float64)

    teff_2 = np.asarray(table['teff_2'], dtype=np.float64)
    logg_2 = np.asarray(table['logg_2'], dtype=np.float64)
    feh_2 = np.asarray(table['feh_2'], dtype=np.float64)

    mask = ck04_bounds_mask(teff_1, logg_1, feh_1) & ck04_bounds_mask(teff_2, logg_2, feh_2)
    return table[mask]


# =============================================================================
# Filter Building
# =============================================================================

def build_filter_map(params: Sequence[float]) -> Dict[str, FilterAdapter]:
    """
    Build filter dictionary from parameter specifications.
    
    Applies hard wavelength constraints to ensure all filters stay within
    [MIN_WAVELENGTH, MAX_WAVELENGTH] bounds.
    
    Parameters
    ----------
    params : Sequence[float]
        If OPTIMIZE_FILTER_EDGES: [u0, u1, u2, u3, v0, v1, v2, v3] in [0,1] (latent)
        Otherwise: [w1, w2, w3, w4] (widths only)
    
    Returns
    -------
    Dict[str, FilterAdapter]
        Filter name -> FilterAdapter objects
    """
    # Apply hard wavelength constraints before building filters
    # constrain_filter_params always returns [edges..., widths...]
    constrained = constrain_filter_params(np.asarray(params, dtype=np.float64))
    
    edges = constrained[:N_FILTERS]
    widths = constrained[N_FILTERS:]
    
    return {
        f"f{i+1}": make_tophat(
            edges[i],
            edges[i] + widths[i],
            name=f"f{i+1}"
        )
        for i in range(N_FILTERS)
    }


# =============================================================================
# Objective Function
# =============================================================================

# Global binary table (set in main)
BINARY_TABLE: Optional[Table] = None


def objective_function(params: Sequence[float]) -> float:
    """
    Objective function for filter optimization.
    
    Computes negative detection rate for the binary star sample.
    
    Parameters
    ----------
    params : Sequence[float]
        If OPTIMIZE_FILTER_EDGES: [e1, e2, e3, e4, w1, w2, w3, w4]
        Otherwise: [w1, w2, w3, w4]
    """
    global ITERATION_COUNTER
    ITERATION_COUNTER += 1
    iteration = ITERATION_COUNTER
    
    params = np.asarray(params, dtype=np.float64)
    
    # Format display based on mode
    if config.OPTIMIZE_FILTER_EDGES:
        # Decode latent variables for display
        edges, widths = decode_latent_to_filters(params)
        logger.info(f"[Iter {iteration}] Latent: u={np.round(params[:N_FILTERS], 3)}, v={np.round(params[N_FILTERS:], 3)}")
        logger.info(f"[Iter {iteration}] Decoded edges: {np.round(edges, 0)}, widths: {np.round(widths, 0)}")
    else:
        widths = params
        logger.info(f"[Iter {iteration}] Evaluating widths: {np.round(widths, 0)}")
    t0 = time.time()

    constrained_params = constrain_filter_params(params)
    widths = constrained_params[N_FILTERS:]
    if widths.shape[0] != N_FILTERS:
        logger.error(f"[Iter {iteration}] Invalid width vector shape: {widths.shape}")
        elapsed = time.time() - t0
        log_iteration(iteration, params, 0.0, 0.0, 0, 0, 0.0, elapsed)
        return 0.0
    
    # Build filters
    filter_map = build_filter_map(params)
    
    # Get grid cache
    grid_cache = get_grid_cache()
    
    # Compute grid photometry
    grid_mags, grid_colors = grid_cache.compute_grid_photometry(filter_map)
    
    # Get grid parameters
    grid_teff, grid_logg, grid_feh = grid_cache.get_grid_params()
    grid_spacing = grid_cache.get_grid_spacing()
    
    # Log grid/color consistency (once per iteration, not per worker)
    expected_n_colors = get_n_colors()
    logger.debug(f"[Iter {iteration}] grid_colors shape = {grid_colors.shape}, USE_GAIA_FILTERS = {config.USE_GAIA_FILTERS}, expected n_colors = {expected_n_colors}")
    if grid_colors.ndim != 2 or grid_colors.shape[1] != expected_n_colors:
        logger.error(
            f"[Iter {iteration}] Grid color dimensionality mismatch: "
            f"shape={grid_colors.shape}, expected second axis={expected_n_colors}"
        )
        elapsed = time.time() - t0
        log_iteration(iteration, params, 0.0, 0.0, 0, 0, 0.0, elapsed)
        return 0.0
    
    # Prepare tasks
    tasks = [binary_row_to_task(i, row, DEFAULT_APPARENT_MAG) for i, row in enumerate(BINARY_TABLE)]
    
    n_binaries = len(tasks)
    n_workers = min(N_WORKERS, n_binaries)
    
    logger.debug(f"[Iter {iteration}] Processing {n_binaries} binaries with {n_workers} workers")
    
    # Process binaries
    if n_workers == 1 or n_binaries < 100:
        worker_init(filter_map, widths, grid_colors, grid_teff, grid_logg, grid_feh, grid_spacing)
        results = [process_binary(task) for task in tasks]
    else:
        chunksize = max(5, n_binaries // (n_workers * 4))
        
        with Pool(
            processes=n_workers,
            initializer=worker_init,
            initargs=(filter_map, widths, grid_colors, grid_teff, grid_logg, grid_feh, grid_spacing)
        ) as pool:
            results = pool.map(process_binary, tasks, chunksize=chunksize)
    
    # Aggregate results
    detection_sum = 0.0
    valid_count = 0
    significances = []
    
    for r in results:
        if np.isfinite(r['detection_prob']):
            detection_sum += r['detection_prob']
            valid_count += 1
            if np.isfinite(r['significance']):
                significances.append(r['significance'])
    
    if valid_count == 0:
        logger.warning(f"[Iter {iteration}] No valid detections computed")
        elapsed = time.time() - t0
        log_iteration(iteration, params, 0.0, 0.0, 0, 0, 0.0, elapsed)
        return 0.0
    
    detection_rate = detection_sum / n_binaries
    detection_rate_valid = detection_sum / valid_count
    
    significances = np.array(significances)
    n_detected = int(np.sum(significances >= SIGMOID_S0))
    avg_significance = float(np.mean(significances)) if len(significances) > 0 else 0.0
    
    elapsed = time.time() - t0
    
    logger.info(f"[Iter {iteration}] Detection rate (all): {detection_rate*100:.2f}% | "
                f"(valid): {detection_rate_valid*100:.2f}% | "
                f"Detected: {n_detected}/{valid_count} | "
                f"Avg σ: {avg_significance:.2f} | "
                f"Time: {elapsed:.1f}s")
    
    log_iteration(iteration, params, detection_rate, detection_rate_valid, 
                  n_detected, valid_count, avg_significance, elapsed)
    
    return -detection_rate


# =============================================================================
# Optimization Helpers
# =============================================================================

def run_powell_refinement(best_x: np.ndarray) -> tuple:
    """Refine optimum using Powell's method (1 Å parameter tolerance)."""
    logger.info("=" * 60)
    logger.info("REFINING WITH POWELL MINIMIZATION")
    logger.info("=" * 60)
    
    result = minimize(
        objective_function,
        best_x,
        method="Powell",
        bounds=get_optimization_bounds(),
        options={
            "maxiter": REFINE_MAXITER,
            "xtol": 1.0,   # stop when parameter changes < 1 Å
            "disp": True,
        },
    )
    
    # Log Powell summary (iterations & function evaluations)
    nit = getattr(result, "nit", None)
    nfev = getattr(result, "nfev", None)
    logger.info(f"Powell finished: success={result.success}, nit={nit}, nfev={nfev}, message={result.message}")
    
    return np.asarray(result.x, dtype=np.float64), float(result.fun)


def compute_param_uncertainties(result) -> np.ndarray:
    """Estimate parameter uncertainties from optimization history."""
    uncertainties = []
    X_evaluated = np.asarray(result.x_iters, dtype=np.float64)
    y_evaluated = np.asarray(result.func_vals, dtype=np.float64)
    
    bounds = get_optimization_bounds()
    n_params = get_n_params()
    
    for i in range(n_params):
        threshold = result.fun + 0.05 * abs(result.fun)
        good_mask = y_evaluated <= threshold
        
        if good_mask.sum() > 1:
            uncertainties.append(float(np.std(X_evaluated[good_mask, i])))
        else:
            # Handle both tuple bounds and skopt Real dimensions
            b = bounds[i]
            if hasattr(b, 'low') and hasattr(b, 'high'):
                param_range = b.high - b.low
            else:
                param_range = b[1] - b[0]
            distances = np.abs(X_evaluated[:, i] - result.x[i])
            close_points = X_evaluated[distances < 0.1 * param_range, i]
            if close_points.size > 1:
                uncertainties.append(float(np.std(close_points)))
            else:
                uncertainties.append(float(param_range * 0.025))
    
    return np.asarray(uncertainties, dtype=np.float64)


def save_results(
    optimal_params: np.ndarray,
    uncertainties: np.ndarray,
    detection_rate: float,
    result,
    binary_results: Optional[List[Dict]] = None,
) -> str:
    """Save optimization results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Apply hard wavelength constraints to optimal parameters
    # constrain_filter_params always returns [edges..., widths...]
    constrained = constrain_filter_params(optimal_params)
    optimal_edges = constrained[:N_FILTERS]
    optimal_widths = constrained[N_FILTERS:]
    
    # For edge optimization (latent mode), uncertainties are in latent space (0-1)
    # We note this in the output file
    if config.OPTIMIZE_FILTER_EDGES:
        # Uncertainties are in latent space
        latent_uncertainties = uncertainties
    else:
        # Width-only mode
        latent_uncertainties = None
        width_uncertainties = uncertainties
    
    summary_file = f"optimal_filters_{timestamp}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("GAIA NIR FILTER WIDTH OPTIMIZATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Binary sample size: {len(BINARY_TABLE)}\n")
        f.write(f"Grid stars: {get_grid_cache().n_stars}\n")
        f.write(f"Optimizer calls: {len(result.x_iters)}\n")
        f.write(f"Gaia filters included: {config.USE_GAIA_FILTERS}\n")
        f.write(f"L-BFGS-B refinement: {config.USE_LBFGS_REFINEMENT}\n")
        f.write(f"Optimize filter edges: {config.OPTIMIZE_FILTER_EDGES}\n")
        if config.OPTIMIZE_FILTER_EDGES:
            f.write(f"Parametrization: latent [0,1] with guaranteed ordering\n")
        f.write(f"Total colors: {get_n_colors()}\n\n")
        
        f.write("OPTIMAL FILTER CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        for i in range(N_FILTERS):
            left = optimal_edges[i]
            width = optimal_widths[i]
            right = left + width
            f.write(f"  Filter {i+1}: {left:.0f} - {right:.0f} Å (width: {width:.0f})\n")
        
        f.write(f"\nDETECTION RATE: {detection_rate*100:.2f}%\n")
        f.write(f"Sigmoid parameters: C={SIGMOID_C}, s0={SIGMOID_S0}\n")
        
        f.write("\nRAW OPTIMIZATION DATA\n")
        f.write("-" * 40 + "\n")
        f.write(f"Optimal edges: {optimal_edges.tolist()}\n")
        f.write(f"Optimal widths: {optimal_widths.tolist()}\n")
        
        if config.OPTIMIZE_FILTER_EDGES:
            f.write(f"\nLatent parameters (raw optimizer output):\n")
            f.write(f"  u (edge control): {optimal_params[:N_FILTERS].tolist()}\n")
            f.write(f"  v (width control): {optimal_params[N_FILTERS:].tolist()}\n")
            f.write(f"  Latent uncertainties: {latent_uncertainties.tolist()}\n")
        else:
            f.write(f"Width uncertainties: {width_uncertainties.tolist()}\n")
        
        f.write(f"Best objective: {result.fun}\n")
    
    logger.info(f"Results saved to {summary_file}")
    
    # Save binary results table
    if binary_results is not None:
        table_file = f"binary_results_{timestamp}.fits"
        out_table = BINARY_TABLE.copy()
        
        for key in ['rmd', 'rmd_error', 'significance', 'detection_prob',
                    'best_teff', 'best_logg', 'best_feh', 'neighbor_error']:
            values = [r.get(key, np.nan) for r in binary_results]
            out_table[key] = values
        
        for i in range(N_FILTERS):
            fkey = f"f{i+1}"
            out_table[f'binary_{fkey}'] = [r.get(f'binary_{fkey}', np.nan) for r in binary_results]
            out_table[f'single_{fkey}'] = [r.get(f'single_{fkey}', np.nan) for r in binary_results]
        
        n_colors = get_n_colors()
        for i in range(n_colors):
            out_table[f'binary_c{i+1}'] = [r.get(f'binary_c{i+1}', np.nan) for r in binary_results]
        
        if config.USE_GAIA_FILTERS:
            for band in ['bp', 'g', 'rp']:
                out_table[f'binary_gaia_{band}'] = [r.get(f'binary_gaia_{band}', np.nan) for r in binary_results]
        
        out_table.write(table_file, overwrite=True)
        logger.info(f"Binary results saved to {table_file}")
    
    return summary_file


# =============================================================================
# Main
# =============================================================================

def main():
    """Main optimization loop."""
    global BINARY_TABLE
    
    # Initialize iteration log file
    init_iteration_log()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Optimize NIR filter widths for binary star detection"
    )
    parser.add_argument(
        "--gaia", 
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help="Include Gaia BP, G, RP filters (default: False)"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        default=False,
        help="Disable L-BFGS-B refinement (default: False)"
    )
    parser.add_argument(
        "--optimize-edges",
        action="store_true",
        default=False,
        help="Optimize filter edges in addition to widths (8 latent params, guarantees ordering)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=PATH_TO_BINARY_TABLE,
        help=f"Path to binary table FITS file (default: {PATH_TO_BINARY_TABLE})"
    )
    parser.add_argument(
        "--nrows", "-n",
        type=int,
        default=-1,
        help="Number of rows to process from binary table, -1 for all rows (default: -1)"
    )
    parser.add_argument(
        "--n-random-starts",
        type=int,
        default=N_RANDOM_STARTS,
        help=f"Number of random starts for Bayesian optimization (default: {N_RANDOM_STARTS})"
    )
    parser.add_argument(
        "--n-optimizer-calls",
        type=int,
        default=N_OPTIMIZER_CALLS,
        help=f"Total number of optimizer calls (default: {N_OPTIMIZER_CALLS})"
    )
    args = parser.parse_args()
    
    set_gaia_filters(args.gaia)
    set_lbfgs_refinement(not args.no_refine)
    set_optimize_edges(args.optimize_edges)
    set_binary_table_path(args.input)
    set_binary_table_limit(args.nrows)
    set_n_random_starts(args.n_random_starts)
    set_n_optimizer_calls(args.n_optimizer_calls)
    
    logger.info("=" * 60)
    logger.info("GAIA NIR FILTER WIDTH OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Iteration CSV: {ITERATION_LOG_FILE}")
    logger.info("")
    logger.info("Input Data:")
    logger.info(f"  Binary table: {config.BINARY_TABLE_PATH}")
    logger.info(f"  Row limit: {config.BINARY_TABLE_LIMIT if config.BINARY_TABLE_LIMIT > 0 else 'all rows'}")
    logger.info("")
    logger.info("Configuration Parameters:")
    logger.info(f"  Workers: {N_WORKERS}")
    logger.info(f"  Use Gaia filters: {config.USE_GAIA_FILTERS}")
    logger.info(f"  Use L-BFGS-B refinement: {config.USE_LBFGS_REFINEMENT}")
    logger.info(f"  Optimize filter edges: {config.OPTIMIZE_FILTER_EDGES}")
    logger.info(f"  Optimization parameters: {get_n_params()} ({'latent u+v [0,1]' if config.OPTIMIZE_FILTER_EDGES else 'widths only'})")
    logger.info(f"  Total colors: {get_n_colors()} ({'NIR+Gaia' if config.USE_GAIA_FILTERS else 'NIR only'})")
    logger.info(f"  Test mode: {TEST_MODE}")
    logger.info("")
    logger.info("Error Model:")
    logger.info(f"  Scaling: sqrt (physically correct: σ ∝ 1/√width)")
    logger.info(f"  Gaia G FWHM: {GAIA_G_FWHM_AA:.1f} Å")
    logger.info(f"  Systematic error floor: {SYSTEMATIC_ERROR_MMAG:.1f} mmag")
    logger.info(f"  Reference magnitude: {DEFAULT_APPARENT_MAG:.1f} mag")
    logger.info("")
    logger.info("Detection Model (Sigmoid):")
    logger.info(f"  Steepness (C): {SIGMOID_C}")
    logger.info(f"  Detection threshold (s0): {SIGMOID_S0} σ")
    logger.info("")
    logger.info("Grid Matching:")
    logger.info(f"  BallTree k-neighbors: {K_NEIGHBORS}")
    if TEST_MODE:
        logger.info("")
        logger.info("*** TEST MODE ENABLED ***")
    
    # Load binary table
    logger.info(f"Loading binary table from {config.BINARY_TABLE_PATH}")
    try:
        if config.BINARY_TABLE_LIMIT > 0:
            BINARY_TABLE = Table.read(config.BINARY_TABLE_PATH)[:config.BINARY_TABLE_LIMIT]
            logger.info(f"  Limited to first {config.BINARY_TABLE_LIMIT} rows")
        else:
            BINARY_TABLE = Table.read(config.BINARY_TABLE_PATH)
    except Exception as exc:
        logger.exception(f"Failed to read binary table at {config.BINARY_TABLE_PATH}: {exc}")
        raise
    logger.info(f"  Loaded {len(BINARY_TABLE)} binaries")
    
    BINARY_TABLE = filter_table_by_ck04_bounds(BINARY_TABLE)
    logger.info(f"  After CK04 filtering: {len(BINARY_TABLE)} binaries")
    
    if TEST_MODE and len(BINARY_TABLE) > TEST_BINARY_LIMIT:
        BINARY_TABLE = BINARY_TABLE[:TEST_BINARY_LIMIT]
        logger.info(f"  TEST MODE: Limited to {len(BINARY_TABLE)} binaries")
    
    # Precompute binary SEDs (inherited via fork COW)
    logger.info(f"Starting binary SED precomputation for {len(BINARY_TABLE)} binaries...")
    precompute_binary_seds(BINARY_TABLE)
    logger.info("Binary SED precomputation complete")
    
    # Initialize grid cache
    logger.info("Initializing grid cache...")
    get_grid_cache()
    logger.info("Grid cache initialized")
    
    # Optimization parameters
    n_calls = TEST_OPTIMIZER_CALLS if TEST_MODE else config.N_OPTIMIZER_CALLS
    n_random = min(config.N_RANDOM_STARTS, n_calls // 2) if TEST_MODE else config.N_RANDOM_STARTS
    n_initial = min(get_n_initial_points(), n_calls // 2) if TEST_MODE else get_n_initial_points()
    
    logger.info("")
    logger.info("Optimization Configuration:")
    logger.info(f"  Wavelength limits: [{MIN_WAVELENGTH:.0f}, {MAX_WAVELENGTH:.0f}] Å")
    if config.OPTIMIZE_FILTER_EDGES:
        logger.info(f"  Parametrization: latent [0,1] variables (constraints satisfied by construction)")
        logger.info(f"  Edge ranges: {[(int(e), int(e+config.EDGE_RANGE)) for e in FILTER_STARTING_EDGES]}")
        logger.info(f"  Non-strict edge ordering: p1 <= p2 <= p3 <= p4 (overlaps allowed)")
        logger.info(f"  Width constraints: w_i >= {config.MIN_FILTER_WIDTH:.0f} Å, p_i + w_i <= {MAX_WAVELENGTH:.0f} Å")
    else:
        logger.info(f"  Filter edges (fixed): {FILTER_STARTING_EDGES}")
        logger.info(f"  Width bounds: {get_optimization_bounds()}")
    logger.info(f"  Optimizer: forest_minimize (base_estimator=ET)")
    logger.info(f"  Acquisition function: {ACQ_FUNC}")
    logger.info(f"  Total calls: {n_calls}")
    logger.info(f"  Random starts: {n_random}")
    logger.info(f"  Initial points: {n_initial}")
    logger.info(f"  Refine optimum with Powell: {REFINE_OPTIMUM}")
    if REFINE_OPTIMUM:
        logger.info(f"  Powell tolerance: 1.0 Å")
        logger.info(f"  Powell max iterations: {REFINE_MAXITER}")
    
    # Run optimization
    logger.info("Starting forest_minimize optimization...")
    logger.info("About to call first objective evaluation (this may take several minutes)...")
    t_opt_start = time.time()
    
    result = forest_minimize(
        objective_function,
        get_optimization_bounds(),
        n_calls=n_calls,
        n_random_starts=n_random,
        n_initial_points=n_initial,
        random_state=42,
        verbose=True,
        n_jobs=1,
        base_estimator="ET",
        acq_func=ACQ_FUNC,
        xi=XI,
        kappa=KAPPA,
    )
    
    best_x = np.asarray(result.x, dtype=np.float64)
    best_fun = float(result.fun)
    
    logger.info(f"Forest minimize completed in {(time.time() - t_opt_start)/60:.1f} minutes")
    
    # Powell refinement
    if REFINE_OPTIMUM:
        refined_x, refined_fun = run_powell_refinement(best_x)
        if refined_fun < best_fun:
            logger.info(f"Powell improved: {best_fun:.6f} → {refined_fun:.6f}")
            best_x, best_fun = refined_x, refined_fun
            result.x, result.fun = refined_x.tolist(), refined_fun
        else:
            logger.info("Powell did not improve objective")
    
    # Final evaluation
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    
    constrained_best = constrain_filter_params(best_x)
    best_widths = constrained_best[N_FILTERS:]

    filter_map = build_filter_map(best_x)
    grid_cache = get_grid_cache()
    grid_mags, grid_colors = grid_cache.compute_grid_photometry(filter_map)
    grid_teff, grid_logg, grid_feh = grid_cache.get_grid_params()
    grid_spacing = grid_cache.get_grid_spacing()
    
    tasks = [binary_row_to_task(i, row, DEFAULT_APPARENT_MAG) for i, row in enumerate(BINARY_TABLE)]
    
    worker_init(filter_map, best_widths, grid_colors, grid_teff, grid_logg, grid_feh, grid_spacing)
    
    n_workers = min(N_WORKERS, len(tasks))
    if n_workers > 1:
        with Pool(
            processes=n_workers,
            initializer=worker_init,
            initargs=(filter_map, best_widths, grid_colors, grid_teff, grid_logg, grid_feh, grid_spacing)
        ) as pool:
            binary_results = pool.map(process_binary, tasks, chunksize=max(5, len(tasks) // (n_workers * 4)))
    else:
        binary_results = [process_binary(t) for t in tasks]
    
    # Results
    detection_rate = -best_fun
    uncertainties = compute_param_uncertainties(result)
    
    # Apply hard wavelength constraints to final parameters
    # constrain_filter_params returns [edges..., widths...] for all modes
    constrained_params = constrained_best
    
    # Parse final parameters for display
    final_edges = constrained_params[:N_FILTERS]
    final_widths = constrained_params[N_FILTERS:]
    
    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Detection rate: {detection_rate*100:.2f}%")
    logger.info(f"Wavelength limits enforced: [{MIN_WAVELENGTH:.0f}, {MAX_WAVELENGTH:.0f}] Å")
    if config.OPTIMIZE_FILTER_EDGES:
        logger.info(f"Latent parameters: u={np.round(best_x[:N_FILTERS], 4)}, v={np.round(best_x[N_FILTERS:], 4)}")
    logger.info("Optimal filter configuration:")
    for i in range(N_FILTERS):
        left = final_edges[i]
        width = final_widths[i]
        right = left + width
        logger.info(f"  Filter {i+1}: {left:.0f} - {right:.0f} Å (width: {width:.0f})")
    
    save_results(best_x, uncertainties, detection_rate, result, binary_results)
    
    logger.info(f"Total time: {(time.time() - t_opt_start)/60:.1f} minutes")
    logger.info(f"Iteration log saved to: {ITERATION_LOG_FILE}")


if __name__ == "__main__":
    main()
