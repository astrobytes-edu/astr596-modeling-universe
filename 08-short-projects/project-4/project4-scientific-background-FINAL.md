# Scientific Background: Measuring the Universe with Supernovae

## ASTR 596: Modeling the Universe

> *"Equipped with his five senses, man explores the universe around him and calls the adventure Science."*  
> — Edwin Hubble

---

## The 1998 Revolution

In 1998, two independent teams studying distant Type Ia supernovae made a discovery so unexpected that it overturned our understanding of the cosmos: **the expansion of the universe is accelerating**.

This was shocking. Gravity should be *slowing down* the expansion. Imagine throwing a ball upward — gravity pulls it back. The universe should behave the same way. But observations showed the opposite: the expansion is *speeding up*, as if some mysterious "dark energy" is pushing space apart.

The 2011 Nobel Prize in Physics recognized this discovery. **In Project 4, you'll analyze the same data using the same methods.**

---

## Why Type Ia Supernovae Are Special

### The Physics: A Cosmic Bomb

Type Ia supernovae occur when a white dwarf (the dense core remnant of a Sun-like star) in a binary system accretes matter from its companion. When the white dwarf reaches the **Chandrasekhar limit** ($M_\text{Ch} \approx 1.4\, M_\odot$), electron degeneracy pressure can no longer support it, and the entire star undergoes runaway thermonuclear fusion in seconds.

**Why this matters**: Because they all explode at roughly the same mass, Type Ia supernovae have remarkably similar intrinsic luminosities. With careful calibration (accounting for light curve shapes and colors), they become **standardizable candles** — objects whose true brightness we know.

### Standard Candles: Nature's Gift to Cosmologists

If you know an object's intrinsic luminosity $L$ and measure its observed flux $f$, you can determine its distance:

$$f = \frac{L}{4\pi D_L^2}$$

where $D_L$ is the **luminosity distance**. In astronomy, we work with magnitudes instead of flux. The **distance modulus** is:

$$\mu = m - M = 5\log_{10}\left(\frac{D_L}{10\,\text{pc}}\right)$$

where:

- $m$ is the apparent magnitude (what we measure)
- $M$ is the absolute magnitude (the apparent magnitude the object would have if placed at a standard distance of 10 pc)
- $D_L$ is the luminosity distance in parsecs

For our purposes, we'll use a more convenient form with $D_L$ in Mpc and factoring out the Hubble constant:

$$\mu = 25 - 5\log_{10}(h) + 5\log_{10}\left(\frac{D_L^*}{\text{Mpc}}\right)$$

where $D_L^* \equiv D_L(h=1)$ is the luminosity distance computed with $h=1$ (i.e., with $H_0 = 100$ km/s/Mpc), and $H_0 = 100h\,\text{km/s/Mpc}$ is the Hubble constant.

**Understanding the $h$-factorization**: The luminosity distance naturally contains $H_0 = 100h$ km/s/Mpc in the denominator. By defining $D_L^* \equiv D_L(h=1)$ as the distance computed with $h=1$, we can write the actual luminosity distance as $D_L = D_L^*/h$. This separates the expansion rate normalization ($h$) from the shape of the distance-redshift relation (which depends on $\Omega_m$ and $\Omega_\Lambda$). When computing $\mu$, we then have:

$$\mu = 5\log_{10}\left(\frac{D_L}{10\,\text{pc}}\right) = 5\log_{10}\left(\frac{D_L^*}{h \cdot 10\,\text{pc}}\right) = 25 - 5\log_{10}(h) + 5\log_{10}\left(\frac{D_L^*}{\text{Mpc}}\right)$$

This form is computationally convenient because $D_L^*(z; \Omega_m)$ needs to be computed only once for given cosmological parameters, and the $h$ dependence enters analytically.

---

## What We're Measuring: The Contents of the Universe

The distance-redshift $(D_L-z)$ relationship depends on **what the universe is made of**. Three parameters govern cosmic expansion:

### $\Omega_m$: Matter Density Parameter

$$\Omega_m \equiv \frac{\rho_m}{\rho_\text{crit}}$$

This is the ratio of the current matter density (dark matter + baryonic matter) to the **critical density** — the density needed for a flat universe. Current measurements give $\Omega_m \approx 0.3$.

**Physical interpretation**: About 30% of the universe's energy budget is matter. The remaining ~70% is dark energy.

:::{note} **What About Radiation?**
The universe also contains radiation (photons, neutrinos) with energy density $\Omega_r$. However, radiation density scales as $(1+z)^4$ (energy density dilutes *and* photons redshift), while matter scales as $(1+z)^3$ and dark energy is constant. At $z=0$ (today), $\Omega_r \sim 10^{-4}$ is negligible compared to matter and dark energy. For our analysis of $z < 1.3$ supernovae, we can safely ignore radiation's contribution to the expansion rate.
:::

### $\Omega_\Lambda$: Dark Energy Density Parameter

$$\Omega_\Lambda \equiv \frac{\Lambda}{3H_0^2}$$

This parameterizes the "cosmological constant" $\Lambda$ — Einstein's biggest "blunder" that turned out to be real. Dark energy has negative pressure and causes accelerated expansion. Measurements give $\Omega_\Lambda \approx 0.7$.

**Physical interpretation**: Dark energy dominates the universe today. We don't know what it is — this is one of the biggest mysteries in physics.

### $h$: Normalized Hubble Constant

$$H_0 = 100h\,\text{km/s/Mpc}$$

The Hubble constant sets the current expansion rate of the universe. The parameter $h$ is dimensionless and roughly $h \approx 0.7$, meaning $H_0 \approx 70\,\text{km/s/Mpc}$.

**Physical interpretation**: For every megaparsec of distance, recession velocity increases by about 70 km/s. This tells us the age and size of the observable universe.

### The Flatness Constraint

For a "flat" universe (zero spatial curvature), we have:

$$\Omega_m + \Omega_\Lambda = 1$$

This is consistent with cosmic microwave background observations. The Planck 2018 results, when combined with baryon acoustic oscillations (BAO) measurements, find spatial curvature consistent with zero: $\Omega_K = 0.001 \pm 0.002$ (Planck Collaboration 2018), where $\Omega_K \equiv 1 - \Omega_m - \Omega_\Lambda$ is the curvature parameter. **Note**: The CMB alone provides weaker constraints; the tight bound requires combining Planck with complementary distance measurements like BAO.

**Why this matters for parameter estimation**: The flatness constraint provides a critical simplification. Instead of having three independent parameters $(\Omega_m, \Omega_\Lambda, h)$, we can work with just two parameters $(\Omega_m, h)$ by expressing dark energy in terms of matter density:

$$\Omega_\Lambda = 1 - \Omega_m$$

This **reduces the dimensionality** of our inference problem. We're trading physical generality (allowing for spatial curvature) for statistical precision (tighter constraints on the remaining parameters). When $\Omega_m + \Omega_\Lambda \neq 1$, we'd need a third parameter, which enlarges uncertainty and introduces additional degeneracies.

**Why is this justified?** The flatness constraint comes from independent observations (CMB acoustic peaks, baryon acoustic oscillations), not from the supernova data itself. We're combining multiple lines of evidence — this is *more* powerful than analyzing supernovae in isolation. The CMB tells us the universe is flat; supernovae tell us *what* fills that flat universe (matter vs. dark energy).

**Physical interpretation**: The flat universe assumption is well-motivated by inflationary cosmology, which predicts $\Omega_{\text{total}} = 1$ to extraordinary precision. Our analysis implicitly assumes this paradigm.

---

## The Distance-Redshift Relation

### Redshift: A Cosmic Speedometer

When we observe distant supernovae, their light is **redshifted** — stretched to longer wavelengths—due to cosmic expansion. The redshift $z$ is defined as:

$$z = \frac{\lambda_\text{obs} - \lambda_\text{emit}}{\lambda_\text{emit}} = \frac{a_\text{now}}{a_\text{then}} - 1$$

where $a(t)$ is the cosmic scale factor (normalized so $a_\text{now} = 1$).

**Physical interpretation**: $z = 0.5$ means the universe was 2/3 its current size when the light was emitted. $z = 1$ means it was half the current size.

### The Friedmann Equation

General relativity tells us how the scale factor $a(t)$ evolves:

$$\left(\frac{\dot{a}}{a}\right)^2 = H_0^2\left[\Omega_m a^{-3} + \Omega_\Lambda + (1-\Omega_m-\Omega_\Lambda)a^{-2}\right]$$

This is the **Friedmann equation**. Each term corresponds to a component with different evolution because each has a different **equation of state**.

**Equation of State**: The relationship between pressure $p$ and energy density $\rho$ for a cosmic fluid:

$$w \equiv \frac{p}{\rho c^2}$$

where $w$ is the equation of state parameter. This determines how energy density scales with the scale factor:

$$\rho \propto a^{-3(1+w)}$$

**The three components**:

| Component | $w$ | Scaling $\rho(a)$ | Physical Reason |
|-----------|-----|-------------------|-----------------|
| Matter (CDM) | $0$ | $a^{-3}$ | Volume dilution only |
| Radiation | $+1/3$ | $a^{-4}$ | Volume dilution + redshift |
| Dark Energy ($\Lambda$) | $-1$ | $a^{0}$ (constant) | Vacuum energy density doesn't dilute |

**Why matter dilutes as $a^{-3}$**: As the universe expands, the number density of particles decreases as $n \propto a^{-3}$ (inverse volume). Since matter has negligible pressure ($w=0$), the energy density is just $\rho_m = nm c^2 \propto a^{-3}$.

**Why dark energy stays constant**: An equation of state $w=-1$ means negative pressure $p = -\rho c^2$. This is the defining property of a cosmological constant — the energy density of empty space itself, which doesn't change as space expands.

The Friedmann equation above can be rewritten to show each component's contribution explicitly:

$$\left(\frac{\dot{a}}{a}\right)^2 = H_0^2\left[\Omega_m a^{-3} + \Omega_\Lambda + \Omega_K a^{-2}\right]$$

where $\Omega_K = 1 - \Omega_m - \Omega_\Lambda$ is the curvature density parameter. For a flat universe, $\Omega_K = 0$.

:::{note} **What about spatial curvature?**
The $\Omega_K a^{-2}$ term in the Friedmann equation doesn't correspond to an energy component with an equation of state. It arises from the geometric contribution to the Friedmann equation in general relativity. When space has positive curvature ($\Omega_K < 0$, closed universe, $k=+1$) or negative curvature ($\Omega_K > 0$, open universe, $k=-1$), this term captures how geometry affects expansion. For a flat universe, $\Omega_K = 0$ and this term vanishes.
:::

### Luminosity Distance Formula

**For the general case** (including spatial curvature), the luminosity distance requires the transverse comoving distance with the $S_K$ mapping:

$$D_L(z) = \frac{c(1+z)}{H_0\sqrt{|\Omega_K|}} S_K\left(\sqrt{|\Omega_K|}\int_0^z \frac{dz'}{E(z')}\right)$$

where:
$$E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda + \Omega_K(1+z)^2}$$

and the function $S_K(x)$ depends on spatial curvature:

$$S_K(x) = \begin{cases}
\sinh(x) & \text{if } \Omega_K > 0 \text{ (open universe)} \\
x & \text{if } \Omega_K = 0 \text{ (flat universe)} \\
\sin(x) & \text{if } \Omega_K < 0 \text{ (closed universe)}
\end{cases}$$

**For the flat case** ($\Omega_K = 0$, which we use in this project), this simplifies significantly:

$$D_L(z) = \frac{c(1+z)}{H_0}\int_0^z \frac{dz'}{E(z')}$$

where:
$$E(z) = \sqrt{\Omega_m(1+z)^3 + (1-\Omega_m)}$$

In the flat case, $c = 299{,}792.458\,\text{km/s}$ is the speed of light.

**Understanding the formula**: The integral computes the **comoving distance** (distance in coordinates that expand with the universe). For flat space, this comoving distance multiplied by $(1+z)$ gives the luminosity distance directly. The factor $c/H_0$ converts from dimensionless integration variable to physical distance, while the integrand $1/E(z')$ reflects how the expansion rate changes with redshift.

**Key insight**: The shape of $D_L(z)$ depends on $(\Omega_m, \Omega_\Lambda, h)$. By measuring $D_L$ at many redshifts, we constrain these parameters.

### The Flat Universe Simplification

For a flat universe ($\Omega_m + \Omega_\Lambda = 1$), this becomes:

$$D_L(z) = \frac{c(1+z)}{H_0}\int_0^z \frac{dz'}{\sqrt{\Omega_m(1+z')^3 + (1-\Omega_m)}}$$

Alternatively, this can be written in terms of the dimensionless Hubble parameter $E(z)$:

$$D_L(z) = \frac{c(1+z)}{H_0}\int_0^z \frac{dz'}{E(z')}$$

where:

$$E(z) = \sqrt{\Omega_m(1+z)^3 + (1-\Omega_m)}$$

for the flat case. This $E(z)$ parameterization is convenient because it clearly shows how different components contribute to the expansion rate at different epochs.

**Computational note**: You should implement **both methods** and verify they agree — this is an excellent validation strategy for your forward model.

**Method 1: Numerical Integration** using `scipy.integrate.quad()`:

- Define the integrand $1/E(z') = 1/\sqrt{\Omega_m(1+z')^3 + (1-\Omega_m)}$ for the flat case
- Integrate from 0 to $z$
- Multiply by $(c/H_0)(1+z)$ to get $D_L$ (flat case)
- More general: For non-flat universes, integrate $1/E(z')$ with the full $E(z')$ including curvature, then wrap the integral result with the $S_K$ mapping as shown in the general formula above

**Method 2: Pen (1999) Fitting Formula**:
- Use the analytical approximation below
- Much faster (no numerical integration)
- Accurate to 0.4% for $0.2 \leq \Omega_m \leq 1.0$

**Why implement both?** If your two implementations agree to within ~0.4%, you can be confident your forward model is correct. This is how professional astronomers validate their code — independent implementations should give (roughly) the same answer. Additionally, comparing execution time teaches you about computational trade-offs: numerical integration is flexible but slow, while fitting formulas are fast but limited in scope.

**The Pen (1999) fitting formula**:

$$D_L(z) = \frac{c}{H_0}(1+z)\left[\eta(1, \Omega_m) - \eta\left(\frac{1}{1+z}, \Omega_m\right)\right]$$

where:

$$\eta(a, \Omega_m) = 2\sqrt{s^3 + 1}\left[\frac{1}{a^4} - 0.1540\frac{s}{a^3} + 0.4304\frac{s^2}{a^2} + 0.19097\frac{s^3}{a} + 0.066941s^4\right]^{-1/8}$$

and $s^3 \equiv (1-\Omega_m)/\Omega_m$.

---

## Worked Example: Computing Luminosity Distance

Let's compute $D_L$ and $\mu$ for a supernova at redshift $z = 0.5$ in a flat universe with $\Omega_m = 0.3$ and $h = 0.7$.

**Given**:
- $z = 0.5$
- $\Omega_m = 0.3$ → $\Omega_\Lambda = 1 - 0.3 = 0.7$
- $h = 0.7$ → $H_0 = 70\,\text{km/s/Mpc}$

**Step 1: Compute the integral**

We need to evaluate:

$$I = \int_0^{0.5} \frac{dz'}{\sqrt{0.3(1+z')^3 + 0.7}}$$

Evaluating numerically (or using Pen's formula) with high precision:
$$I \approx 0.440984$$

**Step 2: Compute luminosity distance**

$$D_L = \frac{c(1+z)}{H_0} I = \frac{299{,}792.458\,\text{km/s} \times 1.5}{70\,\text{km/s/Mpc}} \times 0.440984$$

$$D_L \approx 2{,}833\,\text{Mpc}$$

**Step 3: Compute $D_L^*$ (with $h$ factored out)**

Since $D_L^*$ is defined as the luminosity distance computed with $h=1$ (i.e., $H_0 = 100$ km/s/Mpc), we have:

$$D_L^* = D_L \times h = 2{,}833 \times 0.7 \approx 1{,}983\,\text{Mpc}$$

Alternatively, compute directly with $H_0 = 100$ km/s/Mpc:

$$D_L^* = \frac{299{,}792.458\,\text{km/s} \times 1.5}{100\,\text{km/s/Mpc}} \times 0.440984 \approx 1{,}983\,\text{Mpc}$$

**Step 4: Compute distance modulus**

$$\mu = 25 - 5\log_{10}(h) + 5\log_{10}\left(\frac{D_L^*}{\text{Mpc}}\right)$$

$$\mu = 25 - 5\log_{10}(0.7) + 5\log_{10}(1983)$$

$$\mu = 25 - 5(-0.1549) + 5(3.2974) = 25 + 0.7745 + 16.487$$

$$\mu \approx 42.26\,\text{mag}$$

**Verification**: You can check this against [Ned Wright's Cosmology Calculator](http://www.astro.ucla.edu/~wright/CosmoCalc.html) with these parameters. Your implementation should reproduce this result to within $\sim 0.1\%$ accuracy.

:::{tip} **Test Your Code**
Use this as a unit test! Your `luminosity_distance()` function should return $D_L^* \approx 1{,}983$ Mpc (when computed with $h=1$), or $D_L \approx 2{,}833$ Mpc (when computed with $h=0.7$), and your `distance_modulus()` function should return $\mu \approx 42.26$ mag for these input parameters (±0.02 mag tolerance). Small numerical differences reflect integration tolerances.
:::

---

## How Different Cosmologies Look

The key observational signature is how $\mu(z)$ differs between cosmological models:

**Matter-dominated universe** ($\Omega_m = 1, \Omega_\Lambda = 0$):
- Expansion decelerates rapidly
- Distant SNe appear *brighter* than expected (closer than they should be)

**Accelerating universe** ($\Omega_m = 0.3, \Omega_\Lambda = 0.7$):
- Expansion accelerates at late times
- Distant SNe appear *fainter* than expected (farther than they should be)

**Empty universe** ($\Omega_m = 0, \Omega_\Lambda = 0$):
- Linear expansion (Hubble flow)
- Reference case

The 1998 teams found that high-redshift supernovae were systematically fainter than predicted by matter-dominated models—direct evidence for cosmic acceleration.

---

## The Data: JLA Sample

The **Joint Light-curve Analysis (JLA)** sample combines data from:
- Supernova Legacy Survey (SNLS)
- SDSS-II Supernova Survey
- Nearby supernova samples
- HST observations

**What you'll work with**:
- **n = 31 redshift bins** covering $0 < z < 1.3$
- Each bin contains averaged measurements from multiple supernovae
- Distance modulus $\mu_i$ and redshift $z_i$ for each bin
- **31×31 covariance matrix** $\mathbf{C}$ accounting for statistical and systematic uncertainties

**Why a covariance matrix?** Uncertainties are **correlated** between bins due to:
- Common systematic uncertainties (calibration, extinction corrections)
- Light curve fitting procedures
- Shared observational campaigns

Ignoring these correlations gives wrong error bars. The full covariance matrix is essential.

**Critical importance**: If you treat errors as independent (using only a diagonal covariance matrix), you'll systematically **underestimate your uncertainties**. Common systematics like photometric calibration errors shift all data points together in the same direction—a calibration error of 2% affects *every* supernova in the sample. When you ignore correlations, the data appears artificially constraining because you're counting the same systematic uncertainty 31 separate times instead of once. This is a dangerous mistake in scientific inference that leads to falsely confident conclusions.

---

## The Inference Problem: Forward vs. Inverse

### The Forward Problem (Easy)

Given cosmological parameters $(\Omega_m, h)$:
1. Compute $D_L(z_i; \Omega_m, h)$ for each supernova redshift
2. Predict $\mu_i^\text{theory} = 25 - 5\log_{10}(h) + 5\log_{10}(D_L^*/\text{Mpc})$

This is a **deterministic calculation** — just plug in numbers.

### The Inverse Problem (Hard)

Given observed $\mu_i^\text{obs}$ and covariance $\mathbf{C}$:

- What values of $(\Omega_m, h)$ are most consistent with the data?
- What's the **uncertainty** on these parameters?
- Are they **correlated**? (If I increase $\Omega_m$, must I also change $h$?)

This is an **inference problem**. You need:
1. A **likelihood function** $\mathcal{L}(\Omega_m, h | \text{data})$
2. **Prior probabilities** $p(\Omega_m, h)$
3. A way to **sample the posterior** $p(\Omega_m, h | \text{data})$

That's where MCMC comes in.

---

## The Likelihood Function

For Gaussian errors with covariance $\mathbf{C}$, the log-likelihood is:

$$\ln\mathcal{L}(\theta) = -\frac{1}{2}\sum_{i,j=1}^n r_i\,[\mathbf{C}^{-1}]_{ij}\,r_j - \frac{1}{2}\ln|\mathbf{C}| - \frac{n}{2}\ln(2\pi)$$

which can be written more compactly in matrix notation as:

$$\ln\mathcal{L}(\theta) = -\frac{1}{2}\mathbf{r}^\top \mathbf{C}^{-1} \mathbf{r} - \frac{1}{2}\ln|\mathbf{C}| - \frac{n}{2}\ln(2\pi)$$

where the **residual vector** is:

$$r_i = \mu_i^\text{obs} - \mu_i^\text{theory}(z_i; \theta)$$

and $\theta = (\Omega_m, h)$ for the flat case.

**Understanding each term**:
- The first term $-\frac{1}{2}\mathbf{r}^\top \mathbf{C}^{-1} \mathbf{r}$ is the **chi-squared statistic**, measuring how well the model fits the data while properly accounting for correlated errors
- The second term $-\frac{1}{2}\ln|\mathbf{C}|$ is a **normalization factor** ensuring the Gaussian integrates to 1. Since $\mathbf{C}$ is the *data* covariance (fixed, independent of model parameters $\theta$), this term is just a constant in our inference problem.
- The third term $-\frac{n}{2}\ln(2\pi)$ is another **normalization constant** for the Gaussian probability density

**Practical note**: The constant terms $\ln|\mathbf{C}|$ and $\ln(2\pi)$ don't affect MCMC sampling (they cancel in acceptance ratios since $\mathbf{C}$ is independent of $\theta$), so you can omit them for computational efficiency:

$$\ln\mathcal{L}(\theta) = -\frac{1}{2}\mathbf{r}^\top \mathbf{C}^{-1} \mathbf{r}$$

However, including the full expression is good practice for understanding the probabilistic framework and becomes essential if comparing models with different numbers of parameters (e.g., via Bayesian evidence).

---

## Why This Is Hard: Degeneracies and Tensions

### Parameter Degeneracies

$\Omega_m$ and $h$ are **correlated** (degenerate). Since $D_L \propto 1/h$, you can increase $\Omega_m$ (more matter → slower expansion → objects appear closer/brighter) and simultaneously decrease $h$ (larger distances → objects appear farther/fainter) to keep $\mu$ roughly constant. This creates a **negative correlation**: $\Omega_m \uparrow$ ↔ $h \downarrow$.

This degeneracy produces a "banana-shaped" posterior distribution in the $(\Omega_m, h)$ plane. MCMC efficiently explores this correlated structure.

### The Hubble Tension

Different methods give different values for $H_0$:

- **Early universe** (CMB + Planck): $h = 0.674 \pm 0.005$
- **Late universe** (SNe + SH0ES): $h = 0.730 \pm 0.010$

This $\sim5\sigma$ discrepancy is called the **Hubble Tension**—an active crisis in cosmology.

**Why this is a big deal**: A 5σ discrepancy is statistically overwhelming — if both measurements are correct and systematic errors properly accounted for, there's only about a 1-in-3.5-million chance this arose from random statistical fluctuations. When two precision measurements disagree at this level, something fundamental is wrong. Possible explanations:

1. **Unknown systematics**: One (or both) measurements has unaccounted systematic errors
2. **New physics in the early universe**: Extra radiation, early dark energy, or modifications to expansion history before recombination
3. **Breakdown of ΛCDM**: Our standard cosmological model may be incomplete
4. **Local inhomogeneities**: We may live in an underdense region affecting local measurements

Your measurement will fall somewhere between these values (around $h \approx 0.70$), illustrating the tension firsthand. This isn't a textbook exercise — you're exploring an open question at the frontier of cosmology. The resolution of the Hubble Tension may require new physics beyond the Standard Model of cosmology, but that is beyond the scope of this course.

---

## Connection to Module 5: This Is a Forward Model

Recall from **Module 5 Part 1** the fundamental structure of computational science:

1. **Physical law** → Mathematical model (Friedmann equations)
2. **Forward model** → Predict observables ($\mu$ from $\Omega_m, h$)
3. **Inverse problem** → Infer parameters from noisy measurements
4. **Sampling** → Use MCMC to explore parameter space

This project brings together everything:
- **Module 1**: Statistical thinking, sampling, the CLT
- **Module 2**: You'll see echoes of thermal equilibrium in MCMC convergence
- **Module 5**: Bayesian inference, Metropolis-Hastings, diagnostics
- **Project 2**: Leapfrog integration repurposed for HMC

You're not just analyzing data — you're measuring the **composition of the universe** using Nobel Prize-winning methods you built from scratch. This is the project where you are legitimately **modeling the universe**.

---

## Further Reading

**Original Papers**:
- Riess et al. (1998): "Observational Evidence from Supernovae for an Accelerating Universe..." ([AJ 116, 1009](https://ui.adsabs.harvard.edu/abs/1998AJ....116.1009R))
- Perlmutter et al. (1999): "Measurements of Omega and Lambda from 42 High-Redshift Supernovae" ([ApJ 517, 565](https://ui.adsabs.harvard.edu/abs/1999ApJ...517..565P))

**Data Release**:
- Betoule et al. (2014): "Improved cosmological constraints from a joint analysis..." (JLA sample) ([A&A 568, A22](https://ui.adsabs.harvard.edu/abs/2014A%26A...568A..22B))

**Cosmology Background**:
- Hogg (1999): "Distance measures in cosmology" ([arXiv:astro-ph/9905116](https://arxiv.org/abs/astro-ph/9905116))
- Pen (1999): "Brief Note: Analytical Fit to the Luminosity Distance for Flat Cosmologies with a Cosmological Constant" ([arXiv:astro-ph/9904172](https://arxiv.org/abs/astro-ph/9904172))

**Verification Tools**:
- [Ned Wright's Cosmology Calculator](http://www.astro.ucla.edu/~wright/CosmoCalc.html)

---

*Now proceed to the **Project 4 Description** to learn what you'll implement.*

---

:::{important} **Frontier Research: Why We Need Surveys Like DESI**

**Your project uses 31 supernovae.** Professional cosmology experiments like DESI use **15 million galaxies**. Why the massive scale?

**The Dark Energy Problem**: Recent DESI results (April 2024) hint that dark energy may not be a true cosmological constant—its equation of state $w$ may **evolve over time** rather than staying fixed at $w = -1$. When DESI combined their baryon acoustic oscillation (BAO) measurements with CMB, supernova, and lensing data, they found 2.5-3.9σ preference for evolving dark energy.

**Why your 31 SNe can't measure this**:

- **Degeneracies**: With limited redshift range ($z < 1.3$) and sparse sampling, you can't distinguish $w = -1$ from $w(z) = w_0 + w_a(1-a)$
- **Statistical power**: Need thousands of SNe across wide redshift ranges to break $(\Omega_m, w_0, w_a, h)$ correlations
- **Systematics**: Must combine multiple independent probes (BAO, CMB, lensing, SNe) to check consistency

**What makes DESI different**:

- **15 million spectra**: Measures 3D positions + velocities of galaxies spanning 11 billion years
- **Baryon acoustic oscillations**: "Standard ruler" imprinted by sound waves in early universe, provides distance scale independent of SNe
- **Complementary systematics**: BAO systematics (galaxy bias, peculiar velocities) differ from SNe systematics (dust extinction, evolution)
- **Higher redshift reach**: Probes $z = 0$ to $z > 3$, tracing dark energy evolution across cosmic history

**The broader lesson**: Your MCMC machinery is **exactly** what DESI uses — same Bayesian inference, same convergence diagnostics, same posterior analysis. The difference is:

- **Scale**: Millions of data points instead of 31
- **Dimensionality**: 10+ parameters instead of 2
- **Complementarity**: Multiple datasets that constrain different parameter combinations

After you master the fundamentals with SNe, you'll understand why frontier experiments need:

1. **Large surveys** → Break degeneracies with statistical power
2. **Multiple probes** → Independent systematics, complementary constraints  
3. **Wide redshift coverage** → Trace evolution of cosmic expansion
4. **Sophisticated inference** → The same MCMC/HMC methods you're building

Your 31 SNe teach you the method. DESI's 15 million galaxies push the boundaries of what's measurable. Both use the same statistical framework you're implementing from scratch.

**If dark energy truly evolves, we're missing fundamental physics.** Surveys like DESI, combined with next-generation experiments (Vera Rubin Observatory, Euclid, Nancy Grace Roman Space Telescope), will determine whether we need new physics beyond the Standard Model of cosmology. You're learning the tools to participate in that frontier.

**Further Reading**:

- DESI Collaboration (2024): [Year 1 cosmology results](https://data.desi.lbl.gov/doc/papers/)
- [DESI discovers hints of evolving dark energy](https://newscenter.lbl.gov/2024/04/04/desi-first-year-data-shows-universe-expansion/) (Berkeley Lab press release)
:::

---
