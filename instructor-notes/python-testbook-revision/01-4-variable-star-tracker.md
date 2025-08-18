# Variable Star Exercise Opportunities Tracker

## Overview
This document tracks opportunities to integrate variable star light curve analysis exercises throughout the ASTR 596 Python textbook, building from simple brightness calculations to complete photometric pipelines.

---

## Chapter 01: Python Environment

### Key Concepts for Variable Stars
- File I/O basics (reading observation files)
- IPython for interactive data exploration
- Environment management for astronomy packages

### Current Exercises
- IPython proficiency tasks
- Notebook to script conversion
- Environment diagnostic functions

### Exercise Opportunities

**Quick Practice (5-10 lines):**
```python
# Read a simple light curve file and print basic info
# Load 'cepheid_simple.txt' with 20 time,magnitude pairs
# Print number of observations and magnitude range
```

**Synthesis (15-30 lines):**
```python
# Convert light curve notebook cells to proper script
# Take messy exploratory analysis and create clean module
# Practice good file organization for time-series data
```

**Challenge (Optional):**
```python
# Create environment diagnostic for photometry packages
# Check if photutils, astropy.timeseries are available
# Report versions and compatibility
```

### Data Requirements
- Format: Simple 2-column text file (time, magnitude)
- Size: 20-50 points for quick file reading
- Type: Clean Cepheid data (no errors yet)
- Example: Classical Cepheid with 5-day period

### Dependencies
- None (first chapter)
- Prepares for: File reading patterns used throughout

---

## Chapter 02: Python as Calculator

### Key Concepts for Variable Stars
- Magnitude system and logarithmic scales
- Floating-point precision in photometry
- Converting between magnitude and flux
- Handling very small/large numbers (flux units)

### Current Exercises
- Numerical precision exploration
- Safe numerical functions
- Format scientific output

### Exercise Opportunities

**Quick Practice (5-10 lines):**
```python
# Convert magnitude to flux and back
# Given mag = 12.5, calculate flux in ergs/cm²/s
# Verify mag_to_flux(flux_to_mag(m)) ≈ m within precision
```

**Synthesis (15-30 lines):**
```python
# Calculate magnitude differences and ratios
# Given two stars' magnitudes, find brightness ratio
# Handle edge cases (very faint stars, saturation)
# Work in log space to avoid overflow
```

**Challenge (Optional):**
```python
# Implement robust magnitude averaging
# Account for asymmetric magnitude scale
# Convert to flux, average, convert back
# Compare with naive magnitude average
```

### Data Requirements
- Format: Individual magnitude values or pairs
- Size: Single measurements or small arrays
- Type: Various magnitude ranges (bright to faint)
- Example: -1.5 (Sirius) to 20 (faint variable)

### Dependencies
- None (basic calculations)
- Prepares for: Magnitude statistics in later chapters

---

## Chapter 03: Control Flow

### Key Concepts for Variable Stars
- Iterating through observations
- Finding maxima/minima in light curves
- Convergence in period-finding algorithms
- Quality filtering of observations

### Current Exercises
- Convergence checking
- Adaptive algorithms
- Logic debugging

### Exercise Opportunities

**Quick Practice (5-10 lines):**
```python
# Find brightest and faintest observations
# Loop through magnitudes, track extrema
# Remember: smaller magnitude = brighter!
```

**Synthesis (15-30 lines):**
```python
# Filter bad observations using quality flags
# if magnitude_error > threshold: skip
# if time_gap > max_gap: mark discontinuity
# Count good vs rejected observations
```

**Challenge (Optional):**
```python
# Implement phase dispersion minimization
# Try different periods, check dispersion convergence
# Early version of period-finding algorithm
```

### Data Requirements
- Format: Time, magnitude, error, quality_flag
- Size: 100-200 points (enough for patterns)
- Type: RR Lyrae with some bad points
- Example: Include gaps, outliers, high-error points

### Dependencies
- Builds on: Ch02 magnitude calculations
- Prepares for: Array-based filtering in NumPy

---

## Chapter 04: Data Structures

### Key Concepts for Variable Stars
- Lists for time-series data
- Dictionaries for star catalogs
- Sets for unique observation nights
- Tuples for immutable (time, mag, error) triplets

### Current Exercises
- Performance profiling of structures
- Deep copy debugging
- Cache implementation

### Exercise Opportunities

**Quick Practice (5-10 lines):**
```python
# Store light curve in dictionary
# {'times': [...], 'mags': [...], 'errors': [...]}
# Access and print specific observation
```

**Synthesis (15-30 lines):**
```python
# Create star catalog with multiple variables
# Dict of dicts: {star_id: {'period': p, 'type': t, 'data': d}}
# Find all RR Lyrae with period < 1 day
# Use sets to find overlapping observation nights
```

**Challenge (Optional):**
```python
# Implement efficient observation cache
# LRU cache for phase-folded light curves
# Avoid recomputing for tried periods
```

### Data Requirements
- Format: Multiple light curves, catalog format
- Size: 5-10 stars, 50-100 points each
- Type: Mixed (Cepheids, RR Lyrae, Miras)
- Example: Small variable star survey subset

### Dependencies
- Builds on: Ch03 filtering logic
- Prepares for: Structured arrays in NumPy

---

## Chapter 05: Functions & Modules

### Key Concepts for Variable Stars
- Reusable photometry functions
- Period-finding algorithms
- Light curve feature extraction
- Modular analysis pipeline

### Current Exercises
- Function performance comparison
- Scope debugging
- Module organization

### Exercise Opportunities

**Quick Practice (5-10 lines):**
```python
def phase_fold(times, magnitudes, period):
    """Fold light curve at given period."""
    phases = (times % period) / period
    return phases, magnitudes
```

**Synthesis (15-30 lines):**
```python
# Create photometry module with:
# - read_light_curve(filename)
# - calculate_amplitude(mags)
# - estimate_period_fft(times, mags)
# - phase_fold(times, mags, period)
```

**Challenge (Optional):**
```python
# Implement Lomb-Scargle periodogram
# Handle uneven sampling properly
# Add significance testing
# Memoize for multiple frequency trials
```

### Data Requirements
- Format: Standard time-series format
- Size: 200-500 points (good for FFT)
- Type: Multi-periodic variables (challenging)
- Example: Blazhko RR Lyrae, eclipsing Cepheid

### Dependencies
- Builds on: All previous chapters
- Prepares for: Object-oriented star classes

---

## Chapter 06: Object-Oriented Programming

### Key Concepts for Variable Stars
- VariableStar class with methods
- Inheritance for star types
- Properties for computed features
- Composition for multi-band data

### Current Exercises
- Class implementation
- Inheritance hierarchy
- Performance comparison

### Exercise Opportunities

**Quick Practice (5-10 lines):**
```python
class VariableStar:
    def __init__(self, name, times, mags):
        self.name = name
        self.times = times
        self.mags = mags
    
    @property
    def amplitude(self):
        return max(self.mags) - min(self.mags)
```

**Synthesis (15-30 lines):**
```python
# Create class hierarchy:
# VariableStar (base)
# ├── Cepheid (regular pulsator)
# ├── RRLyrae (short period)
# └── Mira (long period)
# Each with specific period ranges and methods
```

**Challenge (Optional):**
```python
# Multi-band photometry system
# Composition: Star has multiple LightCurves
# Each LightCurve has a filter
# Color indices from band differences
```

### Data Requirements
- Format: Object-serializable format (JSON, pickle)
- Size: Full dataset for multiple stars
- Type: Multi-band observations (B, V, R, I)
- Example: Well-studied Cepheids with colors

### Dependencies
- Builds on: Ch05 functions become methods
- Prepares for: NumPy array methods

---

## Chapter 07: NumPy

### Key Concepts for Variable Stars
- Vectorized magnitude operations
- FFT for period finding
- Array slicing for phase bins
- Statistical functions for light curves

### Current Exercises
- Moving average implementation
- Star catalog operations
- Memory-efficient processing

### Exercise Opportunities

**Quick Practice (5-10 lines):**
```python
# Vectorized sigma clipping
times, mags = np.loadtxt('variable.dat', unpack=True)
median = np.median(mags)
mad = np.median(np.abs(mags - median))
good = np.abs(mags - median) < 3 * mad
```

**Synthesis (15-30 lines):**
```python
# Fast period finding with FFT
# Remove mean, compute power spectrum
# Find peak frequency, refine with finer grid
# Phase fold and bin for template
```

**Challenge (Optional):**
```python
# Implement string-length method
# Minimize sum of magnitude differences
# Between consecutive phase points
# Vectorized for speed on period grid
```

### Data Requirements
- Format: NumPy-readable (CSV, NPY, FITS)
- Size: 1000+ points for good FFT
- Type: Well-sampled Cepheid
- Example: OGLE or Kepler light curve

### Dependencies
- Builds on: All previous, now vectorized
- Prepares for: Matplotlib visualization

---

## Chapter 08: Matplotlib

### Key Concepts for Variable Stars
- Light curve visualization
- Phase diagrams
- Period-amplitude plots
- Multi-panel comparisons

### Current Exercises
- (Need to see chapter content)

### Exercise Opportunities

**Quick Practice (5-10 lines):**
```python
# Basic light curve plot
fig, ax = plt.subplots()
ax.scatter(times, mags, s=10, alpha=0.6)
ax.invert_yaxis()  # Astronomical convention!
ax.set_xlabel('Time (days)')
ax.set_ylabel('Magnitude')
```

**Synthesis (15-30 lines):**
```python
# Multi-panel diagnostic plot:
# 1. Raw light curve
# 2. Phase-folded curve
# 3. Periodogram
# 4. O-C (observed - calculated) residuals
```

**Challenge (Optional):**
```python
# Interactive period explorer
# Slider for period adjustment
# Live phase-folding update
# Residual display
# Save best period
```

### Data Requirements
- Format: Same as NumPy chapter
- Size: Full datasets for visual impact
- Type: Various for comparison
- Example: Gallery of different variable types

### Dependencies
- Builds on: Ch07 NumPy analysis
- Prepares for: Publication-quality figures

---

## Chapter 09: Robust Computing

### Key Concepts for Variable Stars
- Handling missing data
- Error propagation in photometry
- Validation of period solutions
- Testing photometry pipelines

### Current Exercises
- Error handling
- Input validation
- Testing strategies

### Exercise Opportunities

**Quick Practice (5-10 lines):**
```python
def safe_magnitude_average(mags, errors=None):
    """Average magnitudes with proper error handling."""
    if len(mags) == 0:
        raise ValueError("No magnitudes to average")
    if errors is not None and len(errors) != len(mags):
        raise ValueError("Magnitude and error arrays must match")
    # Weighted average if errors provided
```

**Synthesis (15-30 lines):**
```python
# Robust period finding with validation:
# - Check data quality first
# - Try multiple algorithms
# - Validate period (within reasonable range)
# - Test against aliases (1 day, 1 year)
# - Return confidence metrics
```

**Challenge (Optional):**
```python
# Complete testing suite for photometry:
# - Test with known synthetic signals
# - Test edge cases (single point, gaps)
# - Test numerical stability
# - Performance benchmarks
```

### Data Requirements
- Format: Include corrupted/missing data
- Size: Various to test edge cases
- Type: Problematic cases (aliases, noise)
- Example: Ground-based with weather gaps

### Dependencies
- Builds on: All previous chapters
- Prepares for: Research-grade analysis

---

## Chapter 10: SciPy & Beyond

### Key Concepts for Variable Stars
- Optimization for template fitting
- Interpolation for phase curves
- Signal processing filters
- Statistical tests for variability

### Current Exercises
- (Need to see chapter content)

### Exercise Opportunities

**Quick Practice (5-10 lines):**
```python
# Fit harmonic series to light curve
from scipy.optimize import curve_fit
def fourier_series(t, a0, a1, b1, period):
    w = 2 * np.pi / period
    return a0 + a1*np.cos(w*t) + b1*np.sin(w*t)
```

**Synthesis (15-30 lines):**
```python
# Complete variable star pipeline:
# 1. Read multiple light curves
# 2. Find periods with Lomb-Scargle
# 3. Fit templates with optimization
# 4. Classify based on features
# 5. Generate report
```

**Challenge (Optional):**
```python
# Machine learning classification:
# - Extract features (period, amplitude, shape)
# - Train classifier on known types
# - Predict unknown variables
# - Validate with cross-validation
```

### Data Requirements
- Format: Research-grade (FITS tables)
- Size: Large survey subset
- Type: Complete variety for classification
- Example: ASAS-SN, ZTF, or Gaia variables

### Dependencies
- Builds on: Complete foundation
- Prepares for: Research projects

---

## Overall Progression Summary

### Skill Development Arc

1. **Foundation (Ch01-02):** Read files, understand magnitudes
2. **Logic (Ch03-04):** Filter data, organize observations
3. **Functions (Ch05):** Build reusable analysis tools
4. **Objects (Ch06):** Model stars as objects with behavior
5. **Arrays (Ch07):** Vectorize for performance
6. **Visualization (Ch08):** Create diagnostic plots
7. **Robustness (Ch09):** Handle real-world data issues
8. **Advanced (Ch10):** Research-grade analysis

### Data Complexity Progression

1. **Simple:** 20 points, clean, single star (Ch01-02)
2. **Moderate:** 100-200 points, some errors (Ch03-04)
3. **Realistic:** 500+ points, gaps, multiple stars (Ch05-06)
4. **Research:** 1000+ points, multi-band, survey data (Ch07-10)

### Analysis Sophistication

1. **Basic:** Read and display (Ch01-02)
2. **Filtering:** Clean and select good data (Ch03-04)
3. **Period Finding:** Simple FFT/folding (Ch05-06)
4. **Advanced Periods:** Lomb-Scargle, aliases (Ch07-08)
5. **Complete Pipeline:** Classification and reporting (Ch09-10)

### Key Learning Outcomes

By the end, students will have built:
- **Modular photometry library** (functions for reuse)
- **Variable star classes** (OOP for different types)
- **Period-finding suite** (multiple algorithms)
- **Visualization tools** (publication-quality plots)
- **Complete pipeline** (from raw data to classification)

### Recommended Test Dataset

Create a standard dataset with:
- 5 Cepheids (different periods)
- 5 RR Lyrae (including Blazhko)
- 2 Miras (long period)
- 2 Eclipsing binaries (for contrast)
- 1 Irregular variable

Each with:
- 200-1000 observations
- Realistic gaps and errors
- Multiple bands when appropriate
- Known periods for validation

This progression ensures students build real research capabilities while learning Python, with each chapter's exercises contributing to a complete photometric analysis toolkit they can use in their research.