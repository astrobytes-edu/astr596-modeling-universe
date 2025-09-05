---
title: "Module 5: Radiative Transfer and Photon Statistics"
subtitle: "Radiative Processes in Astrophysics | ASTR 596"
exports:
  - format: pdf
---


## Opacity: The Photon Traffic Controller

:::{margin}
**Opacity (κ)**
A measure of how resistant material is to the passage of radiation, with units cm²/g. Physically, κ represents the cross-section for photon interaction per unit mass of material. High opacity means photons are frequently absorbed or scattered (opaque material), while low opacity means they pass through easily (transparent material). In stellar interiors, opacity controls how steep the temperature gradient must be to transport energy outward.
:::

:::{margin}
**Mean Free Path**
The average distance a particle (here, a photon) travels before interacting with matter. For photons in stellar material, $\ell_\text{photon} = (κρ)^{-1}$. 

In the Sun's core, photons have mean free paths of only ~1 cm despite the core being 700,000 km across! This is why energy takes ~100,000 years to random walk from core to surface, even though light travels at $3×10^{10}$ cm/s.

**Kramers' Opacity**
An approximation for opacity in partially ionized gas: $κ_\text{kr} ∝ ρT^{-3.5}$. Named after Dutch physicist Hendrik Kramers (1916), this combines bound-free and free-free absorption. The strong negative temperature dependence means opacity drops rapidly as temperature rises, making hot stars more transparent to radiation. Valid for temperatures ~$10^4 - 10^6$ K where hydrogen and helium are partially ionized.

**Optical Depth (τ)**
A dimensionless measure of how opaque a medium is along a given path. τ = ∫κρ ds counts how many photon mean free paths fit into the distance traveled. When τ < 1, the medium is "optically thin" (transparent) and photons escape freely. When τ > 1, it's "optically thick" (opaque) and photons undergo many interactions before escaping. The photosphere of a star is defined as the surface where $τ ≈ 2/3$, where photons can finally escape to space.

**Thomson Scattering**
Elastic scattering of photons by free electrons, independent of photon frequency (in the non-relativistic limit). Named after J.J. Thomson who derived it classically in 1906. The cross-section σ_T = 6.65 × 10^(-25) cm² is a fundamental constant. This process doesn't change photon energy, just direction, making it crucial for radiation pressure in hot stars where it dominates the opacity.
:::

Now we need to address a crucial component that appears in the radiative transport equation: **opacity**. While we've been focusing on how particles create pressure and structure through their motion, energy transport through stars depends on how photons interact with matter. This brings us to opacity $κ$ — a quantity that will become central to your understanding of radiative transfer and essential for Project 3 (MCRT).

The opacity $κ$ appearing in the radiative transport equation measures how opaque material is to radiation—essentially, how hard it is for photons to travel through matter. **Think of it as the "resistance" photons face as they try to escape from the stellar interior.**

**Physical meaning of opacity**:
$$\kappa = \frac{1}{\rho \, \ell_{\text{photon}}}$$

where $\ell_{\text{photon}}$ is the photon **mean free path** — the average distance a photon travels before interacting with matter. The opacity $κ$ has units of cm²/g (cross-section per unit mass).

**What determines opacity?** Several processes impede photon flow:

1. **Bound-free absorption** (photoionization):
   - Photon ionizes atom, freeing an electron
   - Strong dependence on ionization state and photon energy
   - Must exceed ionization threshold energy
   - Important at moderate temperatures $(10^4 - 10^5 K)$ 
   <br>

2. **Free-free absorption** (inverse Bremsstrahlung):
   - Free electron absorbs photon while passing near an ion
   - Important in hot, ionized regions
   - Together with bound-free, gives Kramers' opacity: $κ_\text{Kramers} ∝ ρT^(-3.5)$
   <br>

3. **Bound-bound absorption** (line absorption):
   - Photon excites electron between discrete energy levels
   - Creates absorption lines in spectra
   - Millions of lines in cool stellar atmospheres
   - Dominant in stellar atmospheres where lines haven't saturated
   <br>

4. **Electron scattering** (Thomson scattering):

   - Photon deflected by free electron, no energy exchange
   - Nearly independent of temperature and frequency
   - $κ_\text{es} ≈ 0.2(1 + X)$ cm²/g for fully ionized gas ($X$ = hydrogen mass fraction)
   - Dominates in very hot stars (T > 10^6 K) where everything is ionized
   <br>

**How opacities combine**:

In real stars, multiple processes can absorb or scatter photons, and the total opacity comes from all of them working together. For practical calculations, we often combine them into a single effective opacity:

$$\kappa_{\text{total}} \approx \kappa_{\text{bf}} + \kappa_{\text{ff}} + \kappa_{\text{es}}$$

This is an approximation that works reasonably well for stellar interiors where we care about the overall resistance to photon flow. The dominant contribution changes with temperature:

- **Cool regions** (T < 10^4 K): Atomic and molecular absorption dominates
- **Warm regions** (10^4 - 10^6 K): Bound-free and free-free processes dominate, giving approximately $\kappa \propto \rho T^{-3.5}$
- **Hot regions** (T > 10^6 K): Electron scattering dominates with $\kappa_{\text{es}} \approx 0.2(1+X)$ cm²/g

In stellar atmospheres, millions of spectral lines complicate things enormously, but for stellar interiors and your MCRT project, you'll use simplified approaches. The key insight is that opacity determines how hard it is for photons to escape — high opacity means steep temperature gradients and possibly convection, while low opacity means efficient radiative transport.

**Application to Your MCRT Project**

In your Monte Carlo radiative transfer code, you'll implement these opacity concepts directly. You'll likely use either "gray" opacity (frequency-independent) or a simple power-law like Kramers' opacity. The fundamental quantity you'll work with is the **optical depth**:

$$d\tau = \kappa \rho \, dr$$

This dimensionless quantity counts how many photon mean free paths fit into a given distance. When $\tau < 1$, the medium is optically thin (transparent) and photons stream through freely. When $\tau > 1$, it's optically thick (opaque) and photons undergo many scatterings before escaping.

Your Monte Carlo code will use opacity to determine where photons interact with matter. The probability that a photon travels distance $s$ without interaction follows an exponential distribution:

$$P(s) = e^{-\tau(s)} = \exp\left(-\int_0^s \kappa \rho \, ds\right)$$

You'll sample from this distribution to determine interaction points — this is the heart of MCRT. Each photon starts at some location, you randomly sample a distance from this exponential distribution, the photon travels that far, then interacts (absorbs, scatters, or re-emits depending on your physics). This process repeats until the photon either escapes the system or is fully absorbed.

The beauty of Monte Carlo is that complex radiation transport emerges naturally from these simple probabilistic rules. By tracking many photons, you'll build up the radiation field, temperature structure, and observables like spectra and images — all from the fundamental physics of how opacity controls photon propagation.

