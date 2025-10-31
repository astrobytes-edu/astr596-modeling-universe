# HMC Pedagogical Figures - Generation Summary

Generated: 2025-10-31

## Script
`generate_hmc_figures.py` - Comprehensive Python script to create all HMC module figures

## Generated Figures

### Figure 1: Metropolis-Hastings vs HMC (Four-panel comparison)
**File**: `04-mod5-part4-fig1-mh-vs-hmc.png`

Four panels showing:
1. **True 2D Posterior**: Strongly correlated Gaussian (ρ=0.95) with contours showing the ridge structure
2. **M-H Trace (10,000 samples)**: Random walk bouncing inefficiently, colored by time (red=burn-in, blue=post-burnin)
3. **HMC Trace (1,000 samples)**: Smooth trajectories following the ridge, showing first 10 HMC trajectories in rainbow colors
4. **Marginal Distributions**: Histogram comparison showing HMC achieves better coverage with 10× fewer samples

**Key Results**:

- M-H acceptance: 44.8%
- HMC acceptance: 97.5%
- Demonstrates why HMC is dramatically more efficient for correlated posteriors

**Location in document**: Section 3.4 (line 674)

---

### Figure 2: HMC Parameter Tuning (Grid search)
**File**: `04-mod5-part4-fig2-parameter-grid.png`

Two-panel grid search over step size (ε) and trajectory length (L):

1. **ESS Heatmap**: Effective Sample Size colored (green=good), with optimal region highlighted (dashed blue contour)
2. **Acceptance Rate + ESS**: Acceptance rate as color, ESS as contour lines, target acceptance (65-80%) marked in green

**Shows**:

- Trade-off between step size and acceptance rate
- Sweet spot for (ε, L) parameters
- How ESS varies across parameter space
- Practical tuning guidance

**Parameters tested**:
- ε: 0.01 to 0.5 (15 values)
- L: 5 to 100 (20 values)
- Total: 300 HMC runs with 2,000 samples each

**Location in document**: Section 3.6 (line 842)

---

### Figure 3: NUTS U-turn Criterion
**File**: `04-mod5-part4-fig3-uturn-criterion.png`

Three-panel progression showing:
1. **Before U-turn**: Displacement and momentum aligned (angle < 90°)
2. **At U-turn Detection**: Angle exceeds 90°, dot product becomes negative
3. **After U-turn**: Trajectory has doubled back (wasteful to continue)

Each panel shows:
- Trajectory path (colored by time)
- Start point (green circle)
- Current point (red square)
- Displacement vector (blue arrow: θ₀ → θₜ)
- Momentum vector (orange arrow: pₜ)
- U-turn criterion: (θₜ - θ₀) · pₜ < 0
- Angle calculation

**Demonstrates**: The geometric intuition behind NUTS—stop when trajectory curves back toward start

**Location in document**: Section 4.2 (line 951)

---

### Bonus Figure: Energy Conservation Diagnostic
**File**: `04-mod5-part4-bonus-energy-conservation.png`

Four-panel diagnostic showing Hamiltonian conservation for different step sizes:
- ε = 0.05 (excellent: small ΔH)
- ε = 0.15 (good: moderate ΔH)
- ε = 0.30 (acceptable: larger ΔH)
- ε = 0.60 (poor: very large ΔH)

Each panel shows:
- Histogram of ΔH across 500 trajectories
- Mean, standard deviation, and max |ΔH|
- Quality assessment (✓ Excellent, ⚠ Good, ✗ Poor)

**Teaches**: How to diagnose HMC performance via energy conservation

---

## Usage

### Generate all figures:
```bash
conda activate astro
cd figures/
python generate_hmc_figures.py
```

### Regenerate specific figure:
Edit `generate_hmc_figures.py` and comment out unwanted figures in the `if __name__ == "__main__":` block.

### Customize:
- Modify correlation strength: change `rho` in each function
- Adjust grid resolution: modify `epsilons` and `Ls` arrays
- Change color schemes: modify `cmap` parameters
- Adjust figure size: modify `figsize` tuples

---

## Implementation Details

### Algorithms Implemented:
1. **Metropolis-Hastings**: Random walk with Gaussian proposals
2. **HMC**: Full implementation with leapfrog integrator
3. **ESS Estimation**: Via integrated autocorrelation time
4. **Hamiltonian**: U(θ) = -log p(θ|D), K(p) = ½p^T M^(-1) p

### Target Distribution:
- 2D correlated Gaussian with correlation ρ=0.95
- Strong correlation creates "ridge" structure (challenging for random walk)
- Analytical gradients available

### Performance:
- Figure 1: ~3 seconds (11,000 MCMC samples total)
- Figure 2: ~90 seconds (600,000 MCMC samples, 300 parameter combinations)
- Figure 3: ~2 seconds (trajectory simulation)
- Bonus: ~5 seconds (2,000 MCMC samples × 4 step sizes)

Total runtime: ~2 minutes

---

## Pedagogical Goals

These figures teach students:

1. **Why HMC is revolutionary**: Visual comparison showing 10× efficiency gain
2. **How to tune HMC**: Concrete guidance on choosing (ε, L)
3. **How NUTS works**: Geometric intuition for U-turn detection
4. **How to diagnose problems**: Energy conservation as quality metric

Students can:
- See the difference between random walk and gradient-guided sampling
- Understand parameter trade-offs (acceptance vs exploration)
- Visualize phase space dynamics in parameter space
- Learn diagnostic tools for their own research

---

## Connection to Course Modules

- **Module 3 (Dynamics)**: Same leapfrog integrator from N-body Project 2
- **Module 5 Part 3**: Builds on M-H from previous lecture
- **Module 6 (ML)**: Gradients everywhere—HMC uses autodiff just like neural networks
- **Project 4**: Students implement HMC for cosmology inference

These aren't isolated figures—they're part of a unified story about gradient-based methods in computational science.

---

## References

- Neal (2012): "MCMC Using Hamiltonian Dynamics"
- Hoffman & Gelman (2014): "The No-U-Turn Sampler"
- Betancourt (2017): "A Conceptual Introduction to HMC"

---

## Files in this directory

```
04-mod5-part4-HMC.md                          # Main lecture document
generate_hmc_figures.py                       # This script
04-mod5-part4-fig1-mh-vs-hmc.png             # Figure 1
04-mod5-part4-fig2-parameter-grid.png         # Figure 2
04-mod5-part4-fig3-uturn-criterion.png        # Figure 3
04-mod5-part4-bonus-energy-conservation.png   # Bonus diagnostic
README_HMC_FIGURES.md                         # This file
```

---

**Note**: All figures use high-resolution (300 DPI) for publication quality. For web/slides, you may want to create lower-resolution versions.
