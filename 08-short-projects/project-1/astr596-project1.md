# Project 1: Stellar Populations and Object-Oriented Design

**Course:** ASTR 596: Modeling the Universe  
**Instructor:** Dr. Anna Rosen  
**Duration:** 1.5 weeks  
**Due:** Monday, September 8, 2025

**Learning Objectives:** Object-oriented programming, vectorization, modular code design, **matplotlib mastery**

**AI Policy:** Phase 1 - Struggle first, AI only after 30 minutes of independent work  

**Textbook Reference:** This assignment covers the online course Scientific Computing with Python "textbook", The Python Fundamentals Module, Chapters 1-6

**Note:** This project does not require extensions. Instead, you're expected to become proficient with matplotlib - study the documentation, explore examples, and create publication-quality visualizations. Your plotting module is an investment for the entire course!

**Note on Implementation Order:** Parts are labeled A-F for reference, but complete them in whatever order makes sense to you. Consider starting with `constants.py`, then validating `zams.py` before building the rest.

## Background

Stars on the Zero-Age Main Sequence (ZAMS) represent stars at the moment they begin stable hydrogen fusion in their cores - essentially their "birth" as true stars. The ZAMS is a theoretical starting point where stars have just achieved hydrostatic equilibrium between gravity and radiation pressure, with a homogeneous composition throughout.

**Important:** Stars don't stay at their ZAMS properties! They evolve continuously on the main sequence, becoming gradually larger, hotter, and more luminous as they convert hydrogen to helium in their cores. The Sun, for example, was about 30% less luminous when it formed 4.6 billion years ago (L☉,ZAMS ≈ 0.7 L☉) compared to today. This is why your ZAMS calculations won't match present-day solar values - that's expected and correct!

**How the Tout formulae were derived:** The authors ran detailed stellar evolution models for hundreds of combinations of mass and metallicity, then used nonlinear least-squares regression to fit polynomial expressions to the model outputs. The coefficients (α, β, γ, etc.) are the fitted parameters that minimize the squared differences between the formula and the models. We'll explore fitting methods like this in detail in Project 4.

## Quick Start: Validation First!

**Before diving into implementation, validate your setup:**

1. Download Tout et al. (1996): <https://articles.adsabs.harvard.edu/pdf/1996MNRAS.281..257T>
2. Create `constants.py` with CGS units
3. Implement the luminosity function in `zams.py`
4. Test immediately: `import zams; print(zams.luminosity(1.0))`
5. Should get L ≈ 0.698 L☉ for 1 M☉

Starting with validation ensures you're on the right track before building everything else.

## TODOs Summary

| Module | Class? | Required Implementation |
|--------|--------|------------------------|
| **constants.py** | ❌ | Define physical constants in CGS units |
| **zams.py** | ❌ | • Validate Z range (0.0001-0.03)<br>• Check Z=0.02 if solar_Z_only<br>• Complete coefficients dictionary<br>• Implement Eq. 1 for luminosity<br>• Implement Eq. 2 for radius |
| **star.py** | ✓ | • Mass validation<br>• Calculate properties<br>• Methods for T_eff, t_MS, t_KH, λ_peak<br>• f-string representation with units |
| **stellar_population.py** | ✓ | • Handle numpy arrays only<br>• Vectorized calculations<br>• Performance comparison methods |
| **astro_plot.py** | ❌ | • Style setup<br>• HR diagram with inverted T-axis<br>• Multi-panel support<br>• Return figure AND axis objects |

## Part A: Constants Module

Create `constants.py` with physical constants in CGS units. No starter code provided - this is straightforward. Include at minimum: LSUN, RSUN, MSUN, TSUN, G, SIGMA_SB, WIEN_B, CSOL, YEAR, GYR, MYR.

## Part B: ZAMS Functions Module

**Question to consider:** Why use a module with functions instead of a class for ZAMS calculations?

Create `zams.py` implementing Tout et al. (1996) relations. Starter code provided separately.

**Key points:**
- `isinstance(M, np.ndarray)` checks if M is a numpy array (numpy arrays are objects/instances of the ndarray class!)
- `np.all()` returns True only if ALL array elements satisfy the condition
- Look at equations (3)-(4) to understand how Table 1 is organized
- For Z=0.02, certain terms simplify (hint: what is log₁₀(Z/0.02) when Z=0.02? What is log₁₀(1)?)

## Part C: Star Class

Create `star.py` for individual star objects. Starter code provided separately.

**This is the ONLY project with starter code. Study it carefully for future projects!**

Key requirements:
- Always use f-strings with units for output (e.g., `f"L = {self.luminosity:.3f} L_sun"`)
- Implement all physics formulas (we're providing the equations this time - not always!)
- For error checking, use `assert` for debugging/development, `raise ValueError` for user-facing errors

## Part D: StellarPopulation Class

Create `stellar_population.py` to handle collections of stars.

**Design requirement:** Your `__init__` must work two ways:
1. Pass in a numpy array of masses directly: `StellarPopulation(masses=my_array)`
2. Pass N and sampling method: `StellarPopulation(n_stars=1000, sampling='logspace')`

Required sampling methods (_look at NumPy Docs_):
- `'logspace'`: Logarithmically spaced from 0.1 to 100 M☉
- `'linspace'`: Linearly spaced from 0.1 to 100 M_☉  
- `'uniform'`: Random uniform between 0.1 and 100 M_☉
- `'pareto'`: Pareto distribution (approximates Salpeter IMF $ξ(m) ∝ m^(-α) where α = 2.35 \to a = 1.35$)

**Important:** This class works with numpy arrays ONLY, not lists. Vectorization requires arrays!

## Part E: Performance Testing

Write timing tests in your analysis script or notebook to compare:
1. Building lists with append vs pre-allocated arrays
2. Looping vs vectorized calculations  
3. Time and report speedup factors

Do NOT put timing code in the StellarPopulation class itself - test it from outside!

## Part F: Plotting Module

Create `astro_plot.py` with reusable plotting functions. Starter code provided separately.

**Essential Documentation:**
- [Matplotlib User Guide](https://matplotlib.org/stable/users/index.html)
- [Pyplot Tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)  
- [Gallery of Examples](https://matplotlib.org/stable/gallery/index.html)

Browse MANY examples to learn different techniques!

**Design notes:**

- `setup_plot()` returns ONE figure with ONE axis
- Consider: couldn't `plot_hr_diagram()` just call `setup_plot()` internally?
- Why accept `ax` parameter? Enables multi-panel plots!
- If ax provided: plot on it and return it. If not: create new single-panel figure.
- **Always return (fig, ax)** for consistency!

## Required Plots

| Plot Type | Purpose | Key Requirements |
|-----------|---------|------------------|
| **HR Diagram** | Show main sequence | Inverted T-axis, log scales, color by mass |
| **Mass-Luminosity** | Validate Tout fits | Log-log scale, overplot L ∝ M³ |
| **Performance** | Show speedup | Multi-panel: times and speedup |
| **Physical Properties** | Mass dependencies | Multi-panel: timescales, Wien's peak |
| **Mass Distributions** | Compare sampling | 2x2 grid of histograms |

## Validation Requirements

**ZAMS values for 1 $M_☉$:**

- $L_{☉,ZAMS} = 0.698 ± 0.01 L_☉$
- $R☉,ZAMS = 0.888 ± 0.01 R_☉$
- $T☉,eff,ZAMS = 5602 ± 50 K$
- $t_MS ≈ 14 Gyr$
- $t_KH ≈ 50 Myr$

**Performance:**

- Vectorization should show 5-10x speedup (I saw 8.5x on my MacBook M2 Max Pro)
- No speedup = bug in your implementation!

## Analysis Requirements

Create `project1_analysis.ipynb` OR `project1_analysis.py` containing:

1. **Validation**: Test solar values match Tout et al.
2. **Population Analysis**: All required plots
3. **Performance Study**: Timing comparisons with clear output
4. **Physical Insights**: Discussion of results

If using `.py`, print all output clearly and include screenshots in report appendix.

## Deliverables

**Code Files:**

- `constants.py`
- `zams.py`  
- `star.py`
- `stellar_population.py`
- `astro_plot.py`
- `project1_analysis.ipynb` OR `project1_analysis.py`

**Report (3-4 pages total):**

- Research memo (2-3 pages text): Methods, results, insights
- Appendix A: Your 2-4 "worst" plots with captions explaining failures
- Appendix B (if using .py): Screenshots of output

## Grading Rubric

### Core Functionality (40%)

- Values match Tout et al. predictions
- Vectorization demonstrates measurable speedup
- All required analysis completed

### Code Quality & Design (30%)

- Clean implementation using numpy efficiently
- Professional documentation with complete docstrings
- Modular design, no code duplication
- Versatile functions handling general cases

### Visualization & Scientific Communication (30%)

- Plot quality with appropriate scaling and labels
- Evidence of experimentation (saved versions)
- Clear research memo with physical insights
- Reusable plotting module

**Development tips:**

- Start simple and iterate! Get a basic working version, verify it's correct, then refine.
- Writing good code is like writing an essay - you need multiple drafts before the final version.
- Test continuously: implement → validate → improve → repeat

## Stellar Astrophysics Equations Reference

| Equation | Formula    | Description | Notes |
|----------|------------|-------------|-------|
| **Effective <br> Temperature** | $\frac{T_{\rm{eff}}}{T_☉} = \left(\frac{L}{L_☉}\right)^{0.25} × \left(\frac{R}{R_☉} \right)^{-0.5}$ | Stefan-<br>Boltzmann <br> law | $T_☉ = 5777 K$ <br> (present-day);<br> use ZAMS $L$ and $R$ |
| **Main Sequence <br> Lifetime** | $t_{\rm MS} = 10 \text{ Gyr} × \frac{M}{M_☉} × \left(\frac{L}{L_☉}\right)$ | Fuel consumption <br> rate | Normalized to Sun's <br> present-day values; <br> use $L = L_{\rm ZAMS}$ |
| **Kelvin-Helmholtz <br> Timescale** | $t_{\rm KH} = \frac{GM²}{RL}$ | Gravitational <br> contraction <br> timescale | Convert to CGS; <br> $G = 6.674×10^{-8}~\text{cm³/g/s²}$ |
| **Wien's Peak Wavelength** | $λ_{\rm max} = b/T$ | Wien's <br> displacement <br> law | $b = 0.2898~\text{cm K}$; <br> convert result to nm |

**Variable Definitions:**

- $M$: Stellar mass (solar masses for input, convert to grams for KH time)
- $L$: ZAMS luminosity from your zams.py module (solar units)
- $R$: ZAMS radius from your zams.py module (solar units)
- $T_{\rm eff}$: Effective temperature (K)
- $T_☉$: Present-day solar temperature = 5777 K
- $G$: Gravitational constant (CGS)
- $b$: Wien's constant (CGS)

