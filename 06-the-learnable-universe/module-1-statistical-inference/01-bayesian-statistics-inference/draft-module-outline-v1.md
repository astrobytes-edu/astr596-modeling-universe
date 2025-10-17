## **Statistical Inference Module: From Photons to Cosmological Parameters**

*The Big Picture: Every measurement in astronomy is an inference problem. We collect photons, we have models of how the Universe works, and we want to extract the Universe's properties. This module shows you how.*

---

### **Week 1: The Architecture of Astronomical Inference**

---

### **Day 1-2: The Fundamental Problem - What Are We Actually Measuring?**

#### **Opening Mind-Shift: Parameters Are Physical Reality**

Start with conceptual mapping:
- **Parameters (θ)** = The actual properties of the Universe
  - Mass of a star, Distance to a galaxy, Dark energy density (Ωₘ)
  - These exist independent of our measurements
  
- **Data (D)** = What we actually observe
  - Photon counts, Spectral lines, Brightness measurements
  - Always incomplete, always noisy, always indirect

- **Model/Likelihood P(D|θ)** = The physics connecting parameters to observations
  - How a star of mass M produces photons
  - How distance affects apparent brightness
  - This IS your astrophysics knowledge encoded mathematically

- **Prior P(θ)** = Everything we knew before this observation
  - The IMF tells us most stars are low-mass
  - Distances must be positive
  - Previous experiments constrain H₀ to ~70 km/s/Mpc

#### **The Supernova Story: One Measurement, Many Uncertainties**

Take ONE supernova from the dataset:
```
SN2011fe: z = 0.081, m = 16.89 ± 0.07
```

What do we actually know?
1. **Observed brightness** has Poisson noise (photon statistics)
2. **Intrinsic brightness** varies (~0.15 mag scatter even after standardization)
3. **Dust** could dim it (uncertain amount)
4. **Redshift** has measurement error
5. **Cosmology** affects the distance-redshift relation

**The revelation**: That simple magnitude measurement depends on stellar physics, dust physics, AND cosmology. We need to untangle all of this!

---

### **Day 3: Why Everything Depends on Everything - The Need for Linear Algebra**

#### **The Correlation Problem in Astronomy**

**Nothing is independent in astronomy!** Examples:
- Dust affects both color AND brightness (A_V and E(B-V) are linked)
- Metallicity affects stellar evolution AND supernova brightness
- Calibration errors affect ALL measurements from an instrument

#### **From One to Many: Building the Covariance Matrix**

Start with 2 supernovae:
```python
# If they share systematic errors (same instrument, same night)
# Their measurements are CORRELATED

# Covariance matrix for 2 SNe:
C = [[σ₁², σ₁₂],
     [σ₁₂, σ₂²]]

# Off-diagonal σ₁₂ ≠ 0 means correlated errors!
```

Scale to 31 supernovae (JLA dataset):
- 31×31 = 961 numbers!
- Diagonal: Individual measurement uncertainties
- Off-diagonal: Correlations from systematics

**Why linear algebra?**
- C⁻¹ appears in χ² = (data - model)ᵀ C⁻¹ (data - model)
- Eigenvalues of C tell you the independent modes of variation
- Determinant |C| tells you the "volume" of uncertainty

**The key insight**: Linear algebra isn't abstract math - it's the language of correlated measurements. When astronomers say "we account for systematic uncertainties," they mean "we use the full covariance matrix."

---

### **Day 4-5: Bayesian Thinking as Astronomical Reasoning**

#### **Building Intuition: The Distance Ladder as Bayesian Inference**

The cosmic distance ladder IS Bayesian inference in action:

**Step 1: Parallax (nearby stars)**
- Prior: Stars in cluster have same distance
- Likelihood: Parallax measurements
- Posterior: Distance to cluster

**Step 2: Cepheids (nearby galaxies)**  
- Prior: Distance from parallax calibration
- Likelihood: Period-Luminosity relation
- Posterior: Distance to LMC

**Step 3: SNe Ia (cosmological distances)**
- Prior: Distances from Cepheids
- Likelihood: Brightness-decline relation
- Posterior: Hubble constant

Each rung uses the posterior of the previous as its prior!

#### **Bayes Theorem: The Engine of Learning**

$$\underbrace{P(\text{Universe}|\text{photons})}_{\text{What we want}} = \frac{\overbrace{P(\text{photons}|\text{Universe})}^{\text{Physics model}} \times \overbrace{P(\text{Universe})}^{\text{Prior knowledge}}}{\underbrace{P(\text{photons})}_{\text{Normalization}}}$$

For our SNe problem:
$$P(\Omega_m, h | \text{31 SNe}) \propto P(\text{31 SNe} | \Omega_m, h) \times P(\Omega_m, h)$$

#### **The Brutal Reality: Why We Need MCMC**

Try grid search for 2 parameters:
- 100 × 100 grid = 10,000 evaluations (doable)

But real problems:
- Each SN has ~5 parameters (distance, stretch, color, ...)
- 700 SNe × 5 parameters = 3500 dimensions
- 100³⁵⁰⁰ grid points = more than atoms in universe!

**The solution**: Don't evaluate everywhere - explore intelligently with MCMC

---

### **Week 2: MCMC - Exploring the Space of Possible Universes**

---

### **Day 6-7: The Random Walk Through Parameter Space**

#### **Core Concept: Detailed Balance Creates the Right Distribution**

The Metropolis algorithm genius - you only need ratios:

```python
# Never need P(photons) denominator!
acceptance_ratio = P(new_universe|data) / P(current_universe|data)
                 = [P(data|new) × P(new)] / [P(data|current) × P(current)]

if acceptance_ratio > random():
    accept new_universe
```

This naturally spends more time in high-probability regions!

#### **Building Physical Intuition: MCMC as Exploration**

Think of MCMC chains as explorers mapping territory:

- Each chain starts somewhere random in (Ωₘ, h) space
- Takes steps, accepting/rejecting based on posterior probability  
- Gradually maps out the "landscape" of allowed universes
- Multiple chains = multiple explorers comparing notes

**Convergence** = All explorers agree on the map

#### **Live Implementation: From Grid to MCMC**

Start with visual comparison on simple 2D problem:

1. Grid search: Systematic but expensive
2. Random sampling: Fast but misses structure  
3. MCMC: Finds and maps high-probability regions

Key insights:
- Step size matters (too small = stuck, too big = reject everything)
- Burn-in = forgetting bad starting point
- Thinning = reducing correlation between samples

---

### **Day 8: Cosmological Models and Distance Calculations**

#### **The Physics in Our Likelihood**

The distance modulus:
$$\mu = 5\log_{10}(d_L) + 25$$

Where luminosity distance depends on cosmology:
$$d_L(z; \Omega_m, h) = \frac{c(1+z)}{H_0} \int_0^z \frac{dz'}{\sqrt{\Omega_m(1+z')^3 + (1-\Omega_m)}}$$

**This integral IS general relativity applied to the expanding universe!**

#### **Building the Full Likelihood with Covariance**

$$\ln P(\text{data}|\Omega_m, h) = -\frac{1}{2}[\vec{\mu}_{obs} - \vec{\mu}_{model}]^T C^{-1} [\vec{\mu}_{obs} - \vec{\mu}_{model}]$$

Breaking this down:

- $\vec{\mu}_{obs}$: The 31 observed distance moduli
- $\vec{\mu}_{model}(\Omega_m, h)$: Your prediction for each redshift
- $C^{-1}$: Inverse covariance (accounts for all correlations)

**Critical implementation detail**: Always use log-likelihood to avoid underflow!

---

### **Day 9: Convergence and Truth-Finding**

#### **How Do We Know We've Found Truth?**

**Visual diagnostics** - Pattern recognition:
- Trace plots: Look for "fuzzy caterpillars" (good mixing)
- Autocorrelation: How many steps before independence?
- Corner plots: The standard visualization in astronomy

**Quantitative metrics**:
- Gelman-Rubin R̂: Do different chains agree? (Need R̂ < 1.01)
- Effective sample size: How many independent samples?
- Posterior predictive: Do our parameters explain the data?

#### **The Degenerate Banana: Why Cosmology is Hard**

Show the real (Ωₘ, h) posterior - it's not circular, it's a diagonal banana!

- These parameters are degenerate (trade off against each other)
- This is WHY we need MCMC - the peak doesn't tell the whole story
- The shape of the banana contains physics (how universe expansion works)

---

### **Day 10: Your Analysis and the Universe's Composition**

#### **The Full Pipeline**

```python
# Your actual analysis structure
class CosmologyInference:
    def __init__(self):
        self.data = self.load_observations()      # 31 SNe
        self.cov = self.load_systematics()        # 31×31 matrix
        self.cov_inv = np.linalg.inv(self.cov)    # Linear algebra!
    
    def likelihood(self, theta):
        omega_m, h = theta
        model = self.cosmological_distance(omega_m, h)
        residual = self.data - model
        chi2 = residual.T @ self.cov_inv @ residual  # Linear algebra!
        return -0.5 * chi2  # Log likelihood
    
    def prior(self, theta):
        omega_m, h = theta
        if 0 < omega_m < 1 and 0.5 < h < 0.9:
            return 0  # Log of uniform prior
        return -np.inf  # Outside bounds
    
    def posterior(self, theta):
        return self.likelihood(theta) + self.prior(theta)
```

#### **Interpreting Your Results**

Your corner plot shows:

- **Marginalized constraints**: Ωₘ = 0.3 ± 0.02
- **The banana shape**: Degeneracy between parameters
- **Comparison to Planck**: You're measuring the same universe!

**The profound conclusion**: "Dark energy makes up ~70% of the Universe"

- This comes from YOUR analysis
- Using the same methods that won the Nobel Prize
- Uncertainty included!

---

### **The Module's Big Picture Lessons**

1. **Parameters = Reality**: We're not fitting abstract numbers, we're measuring the Universe
2. **Everything Affects Everything**: Correlations are everywhere, hence covariance matrices
3. **Priors = Accumulated Knowledge**: Not arbitrary, but centuries of astronomy
4. **Likelihood = Physics**: Your model of how the Universe works
5. **MCMC = Exploration**: Mapping the landscape of possible universes
6. **Convergence = Confidence**: We've explored enough to trust our answer

### **What Makes This Different from Statistics Class**

- **Start with astronomy, find the statistics**: Not vice versa
- **Physical motivation for every concept**: Why do we need covariance? Because systematics exist!
- **Real data, real discovery**: You're literally measuring dark energy
- **Visualization-first**: See the random walk, see the banana, see the convergence
- **Connection to papers**: Every cosmology paper uses exactly these methods

This is how we've learned that the Universe is accelerating, that dark energy exists, and that we live in a geometrically flat cosmos. These aren't abstract statistical methods - they're the tools that revealed the Universe's composition!