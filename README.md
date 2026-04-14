# gaianir-binary-filters
Code for determining the optimal photometric filter set for the GaiaNIR mission for detecting multiple stars.

# The objective
The core of the script is the objective function, which we aim to minimize. This objective function describes how well a given filter set separates binary stars from single stars in colour space using the Reduced Manhattan metric and returns the detection rate, a fraction of detected binary stars versus the total number of binary stars. We are minimizing the negative value of the objective function to obtain a filter set that yield the highest detection rate on a sample of binary stars.

We use the ```stsynphot``` library (Lim et al. 2016) to compute the synthetic spectra using the Castelli and Kurucz 2004 atlas (henceforth CK04, https://doi.org/10.48550/arXiv.astro-ph/0405087). This is quite computationally expensive, which is why we decided to use the `forest_minimize` optimizer from `scikit-optimize` (https://scikit-optimize.github.io/stable/modules/generated/skopt.forest_minimize.html), which requires only a moderate number of objective function evaluations (on the order of a few 100 iterations) to reach a minimum within the specified tolerance.

A filter set is defined as a set of 4 top-hat filters, which we parametrise with the starting wavelengths and widths. We impose a hard limit that the possible wavelengths of filter are within the range [8000 Å - 25000 Å]. These are the inputs to our objective function.

# Identifying binary stars in colour space

In colour space, a binary star's SED is theoretically different from any single star's SED, meaning we could use photometry to probe the SED shape and determine whether a star is a single star. To determine the separation in the colour space between two SEDs, we use the Reduced Manhattan distance, defined as

    RMD(c_b, c_s) = (1 / N) * sum_{i=1}^N |c_{b,i} - c_{s,i}|,

where $\vec{c_{b}}$ is the colour vector of the binary star and $\vec{c_{s}}$ of a single star. $N$ is the number of colours.

As the $\text{RMD}$ separation in colour space is typically on the order of a few mmag, it is imperative that the uncertainty of RMD is minimized. For estimating the photometric uncertainty we use ```pygaia``` to model errors as if we used the Gaia G band and then scale this error by $\sqrt{w_G/w_{NIR}}$, where $w_G$ is the FWHM of the G band filter and $w_{NIR}$ the width of the top-hat NIR filter. This serves us to effectively avoid impossibly narrow filters. The overall error is then

    sigma_{m_i} = sqrt(sigma_{G,phot}^2 * (w_G / w_NIR) + sigma_sys^2),

where $\sigma_\text{G,phot}$ is the G band error obtained from ```pygaia```, and $\sigma_\mathrm{sys}$ the systematic error, equal to the systematic error in G band (1 mmag).

A simple way to determine detected stars is to set a threshold on the significance of the RMD separation $s = \text{RMD}/\sigma_\mathrm{RMD}$. A binary labeling of stars as detected/nondetected could pose a problem to the optimizer as it results in a discrete cost function landscape. To determine whether a binary star has been detected or not we use the following sigmoid function

    1 / (1 + exp(-G * (s - s_0))),

with steepness $G=5$ and halfway point $s_0=2$. This is set such that a detection with 1 sigma significance contributes a small amount to detection rate, while a 3 sigma significance represents a nearly complete detection. This is summed over the entire sample to obtain the detection rate $D$.

    D = sum_{m=0}^M 1 / (1 + exp(-G * (s - s_0)))





# File description
```main.py``` - the main script, which controls the various stages of optimization  

```config.py``` - a library script which serves as the input interface for setting parameter, input/output and logging  

```errors.py``` - computing the magnitude and RMD errors, and the detection rate  

```grid_cache.py``` - handling of the single star SED grid  

```matching.py``` - gridsearch on single star SED grid + refinement of the minimum with L-BFGS-B  

```photometry.py``` - computation of magnitudes and colours  

```spectrum.py``` - computation of SEDs from stellar parameters  

```workers.py``` - handles workers; initialization, per worker and global variables and logic