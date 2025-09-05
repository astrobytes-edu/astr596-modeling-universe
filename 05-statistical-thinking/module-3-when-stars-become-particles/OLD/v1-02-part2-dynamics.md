---
title: "Part 2: Stellar Dynamics as Collisionless Statistics"
subtitle: "From Stars to Galaxies | Statistical Thinking Module 3 | ASTR 596"
---

## Navigation

[← Part 1: Phase Space](./01-phase-space.md) | [Module 3 Home](./00-overview.md) | [Part 3: The Virial Theorem →](./03-virial-theorem.md)

---

## Learning Objectives

By the end of Part 2, you will be able to:

- [ ] **Write** the collisionless Boltzmann equation for stellar systems
- [ ] **Derive** the Jeans equations as moments of the distribution function
- [ ] **Explain** why stellar systems don't thermalize like gases
- [ ] **Calculate** velocity dispersions and understand their role as "temperature"
- [ ] **Apply** these equations to real stellar systems from clusters to galaxies

---

## Part 2: Application - Star Clusters (Stars as Particles)

Having established that stars can be treated as particles in phase space, let's now apply our statistical mechanics machinery to derive the fundamental equations of stellar dynamics.

### 2.1 The Collisionless Boltzmann Equation (Again)

**Priority: 🟡 Standard Path**

For star clusters, gravitational "collisions" (close encounters between stars) are extremely rare. The mean free path exceeds the cluster size, so we set the collision term to zero:

$\boxed{\frac{\partial f}{\partial t} + \vec{v} \cdot \nabla_r f - \nabla\Phi \cdot \nabla_v f = 0}$

This is the **collisionless Boltzmann equation** or **Vlasov equation**. Note that gravity enters through the **gravitational potential** $(\Phi)$ gradient $\nabla\Phi$ rather than individual forces.

*Why can we ignore collisions?* Let's do an order-of-magnitude estimate to see when two stars significantly deflect each other.

**The Setup**: Two stars strongly interact when they pass close enough that their gravitational interaction significantly changes their velocities. The characteristic distance for a 90° deflection is:

$b_{90} = \frac{2GM_*}{v_{rel}^2}$

**Order-of-Magnitude Estimate** (typical globular cluster parameters):
- Average stellar mass: $M_* \sim M_{\odot} = 2 \times 10^{33}$ g
- Cluster velocity dispersion: $\sigma \sim 10$ km/s  
- Relative velocity between stars: $v_{rel} \sim \sqrt{2}\sigma \sim 14$ km/s

This gives:
$b_{90} \sim \frac{2 \times 6.67 \times 10^{-8} \times 2 \times 10^{33}}{(1.4 \times 10^{6})^2} \sim 10^{14} \text{ cm} \sim 7 \text{ AU}$

With stellar density $n_* \sim 10$ stars/pc³ (in the cluster core), the collision rate is extremely low. The relaxation time (time for gravitational encounters to redistribute energy) is:

$t_{\text{relax}} = \frac{0.1 N}{\ln N} t_{\text{cross}}$

where $t_\text{cross} = R_\text{cl}/\sigma$ is the crossing time.

For concrete examples:
- Open cluster ($N \sim 10^3$): $t_\text{relax} \sim 100$ Myr (will evaporate)
- Globular cluster ($N \sim 10^6$): $t_\text{relax} \sim 10$ Gyr (quasi-stable)
- Galaxy ($N \sim 10^{11}$): $t_\text{relax} \sim 10^{18}$ yr (truly collisionless)

This exceeds the age of many clusters, so they never reach "thermodynamic" equilibrium!

### 2.2 Velocity Dispersion and Kinetic Energy

**Priority: 🔴 Essential**

Just as temperature measures the kinetic energy per particle ($\sim k_B T$) in a gas, **velocity dispersion** $(\sigma)$ measures the kinetic energy per star in a cluster:

:::{margin}
**Velocity Dispersion ($\sigma$)**  
The RMS spread of stellar velocities around the mean. Typical values: ~1 km/s for open clusters, ~10 km/s for globular clusters, ~200 km/s for galaxy bulges. Directly observable from Doppler broadening of spectral lines.
:::

$\boxed{\sigma^2 = \langle v^2 \rangle - \langle v \rangle^2 = \text{Var}(v)}$

The key insight is that velocity dispersion directly gives us the **kinetic energy per unit mass**:

$\boxed{\text{Kinetic energy per star} = \frac{1}{2}M_\star \sigma^2}$

For a cluster with $N$ stars, the total kinetic energy is:
$K = \frac{1}{2}N M_\star \sigma^2_{3D} = \frac{3}{2}N M_\star \sigma^2_{1D}$

But here's the crucial difference:
- **Gas**: Particles collide → energy redistributes → Maxwell-Boltzmann distribution → temperature has meaning
- **Star cluster**: Stars don't collide → no thermalization → velocity distribution set by gravity → no temperature

:::{admonition} 💻 Connection to Project 2: Measuring Velocity Dispersion
:class: note

In your N-body simulations, you'll calculate velocity dispersion as:
$\sigma^2 = \frac{1}{N}\sum_{i=1}^N |\vec{v}_i - \vec{v}_{cm,0}|^2$

**Important**: Use $\vec{v}_{cm,0}$, the INITIAL center of mass velocity, not the current COM velocity. Recalculating COM at each timestep can hide numerical drift!

Watch how $\sigma$ evolves: it should remain roughly constant for a virialized cluster but increase during collapse (virial theorem: as R decreases, σ increases).
:::

**What really matters: The Virial Theorem**

For any self-gravitating system in equilibrium:
$\boxed{2K + U = 0}$

This gives us:
$\sigma^2_{3D} = \frac{|U|}{NM_\star} = \frac{3GM_{\text{total}}}{2R}$

Velocity dispersion is completely determined by the gravitational potential - not by any "temperature" or thermal process.

:::{admonition} 📊 Statistical Insight: Energy is Universal, Temperature is Not
:class: important

Energy is always well-defined - every moving object has kinetic energy. But temperature only makes sense when:

1. Particles exchange energy through collisions
2. The system reaches thermodynamic equilibrium
3. Velocities follow a Maxwell-Boltzmann distribution

Star clusters violate all three conditions! The velocity dispersion $\sigma$ measures kinetic energy, not temperature. This distinction matters because:

- **Energy conservation always applies**: Total E = K + U is conserved
- **Virial equilibrium replaces thermal equilibrium**: Systems settle into 2K + U = 0
- **Jeans equations replace fluid equations**: We use moments of the collisionless Boltzmann

The mathematics looks similar (both involve velocity moments), but the physics is fundamentally different.
:::

### 2.3 The Jeans Equations: Stellar Fluid Dynamics

**Priority: 🔴 Essential**

Taking moments of the collisionless Boltzmann equation gives the **Jeans equations** — the stellar dynamics equivalent of fluid equations.

#### Zeroth Moment: Continuity

Multiply by 1 and integrate over velocity space:

$\boxed{\frac{\partial \nu}{\partial t} + \nabla \cdot (\nu \vec{u}) = 0}$

where $\nu(\mathbf{r},t)$ is the stellar number density and $\vec{u} = \langle \vec{v} \rangle$ is the mean stellar velocity. This is mass conservation for a "fluid" of stars.

#### First Moment: Momentum (Jeans Equation)

Multiply by $\mathbf{v}$ and integrate. For a spherically symmetric system in steady state:

$\boxed{\frac{1}{\nu}\frac{d(\nu \sigma_r^2)}{dr} + \frac{2\beta\sigma_r^2}{r} = -\frac{d\Phi}{dr} = -\frac{GM_r}{r^2}}$

This is the stellar dynamics equivalent of hydrostatic equilibrium! The left side represents "pressure" support from stellar random motions, while the right side is gravity.

:::{margin}
**Anisotropy Parameter $(\beta)$**  
Measures orbital shape distribution in stellar systems. Defined as $\beta = 1 - \sigma_\theta^2/\sigma_r^2$. Values: $\beta = 0$ for isotropic orbits, $\beta > 0$ for radially-biased orbits, $\beta < 0$ for tangentially-biased orbits.
:::

The **anisotropy parameter** $\beta$ captures something gas doesn't have — stars can have different velocity dispersions in different directions:

- $\beta = 0$: isotropic orbits ($\sigma_r = \sigma_\theta = \sigma_\phi$)
- $\beta \to 1$: radial orbits dominate (like comets)
- $\beta < 0$: tangential orbits dominate (like planets)

### 2.4 The Beautiful Parallel: From Atoms to Stars

**Priority: 🔴 Essential**

The profound insight is that the same mathematical framework describes both stellar interiors (atoms as particles) and star clusters (stars as particles):

| Quantity | Stellar Interior (atoms) | Star Cluster (stars) |
|----------|-------------------------|---------------------|
| "Particles" | Atoms with mass $m$ | Stars with mass $M_\star$ |
| Number density | $n(\mathbf{r})$ atoms/cm³ | $\nu(\mathbf{r})$ stars/pc³ |
| Velocity spread | Temperature $T$ | Dispersion $\sigma^2$ |
| "Pressure" | $P = nkT$ (thermal) | $\Pi_{ij} = \nu\sigma_{ij}^2$ (kinetic) |
| Equilibrium | $\frac{dP}{dr} = -\rho g$ | Jeans equation |
| Energy exchange | Collisions (nanoseconds) | Relaxation ($10^{10}$ yr) |

Both equations come from the same source — the first moment of the Boltzmann equation! The only differences are:

1. **Collision timescale**: Atoms collide constantly → isotropic pressure. Stars rarely "collide" → anisotropic "pressure"
2. **Thermalization**: Atoms reach Maxwell-Boltzmann quickly. Stars never thermalize
3. **Extra term**: The $2\beta\sigma_r^2/r$ term appears because stellar systems can have anisotropic orbits

:::{admonition} 🎯 The Profound Unity: One Framework, All Scales
:class: important

We've now derived fundamental equations at vastly different scales using the SAME mathematical machinery:

**The Moment Method Applied to Different "Particles":**

| System | "Particle" | Particle Mass | 0th Moment | 1st Moment |
|--------|-----------|---------------|------------|------------|
| Stellar interior | Atoms | $\sim 2 \times 10^{-24}$ g | Mass continuity | Hydrostatic equilibrium |
| Star cluster | Stars | $\sim 2 \times 10^{33}$ g | Stellar continuity | Jeans equation |
| Galaxy | Star clusters | $\sim 10^{38}$ g | Cluster continuity | Cluster Jeans equation |
| Universe | Galaxies | $\sim 10^{44}$ g | Galaxy continuity | Cosmic Jeans equation |

That's a range of 68 orders of magnitude in mass! Yet the mathematics is identical:

1. Write down the Boltzmann equation for your "particles"
2. Take the 0th moment → continuity equation
3. Take the 1st moment → force balance equation
4. Take the 2nd moment → energy equation (if needed)

**Why this works**: Statistical mechanics doesn't care what you call a "particle" — only that you have many of them. Whether it's $10^{57}$ atoms in a star or $10^{11}$ stars in a galaxy, the statistical framework applies.
:::

## Part 2 Synthesis: Collisionless Statistics Creates Structure

You've discovered that stellar dynamics is just statistical mechanics without thermalization. The key differences from gases:

1. **No collisions = No thermalization**: Stars maintain distinct orbits for billions of years. This creates rich structures (spiral arms, bars, streams) impossible in gases.

2. **Anisotropy matters**: Without collisions to isotropize velocities, radial and tangential dispersions differ. The β parameter becomes essential.

3. **Phase space structure persists**: Stellar streams maintain coherent phase space structure for gigayears. This "memory" lets us reconstruct galactic history.

4. **Same math, different physics**: The Jeans equations are mathematically identical to fluid equations but describe fundamentally different physics — orbits rather than pressure.

The profound realization: **The universe recycles the same statistical framework at every scale**. Master it once with atoms, apply it to stars, extend it to galaxies. The labels change but the mathematics is eternal.

:::{admonition} 🌉 Bridge to Part 3
:class: note

**Where we've been**: You've seen how the collisionless Boltzmann equation leads to the Jeans equations — the stellar dynamics analog of fluid dynamics. The absence of collisions creates fundamental differences: no thermalization, persistent anisotropy, and rich phase space structure.

**Where we're going**: Part 3 will introduce the virial theorem, the universal diagnostic that revealed dark matter. You'll learn how this simple relationship (2K + U = 0) applies to every gravitating system from molecular clouds to galaxy clusters, and how it becomes your primary tool for N-body simulations.

**The key insight to carry forward**: Stellar systems are statistical but not thermal. This distinction creates the diverse structures we observe in the universe.
:::

---

## Navigation

[← Part 1: Phase Space](./01-phase-space.md) | [Module 3 Home](./00-overview.md) | [Part 3: The Virial Theorem →](./03-virial-theorem.md) reaches thermodynamic equilibrium
3. Velocities follow a Maxwell-Boltzmann distribution

Star clusters violate all three conditions! The velocity dispersion $\sigma$ measures kinetic energy, not temperature. This distinction matters because:

- **Energy conservation always applies**: Total E = K + U is conserved
- **Virial equilibrium replaces thermal equilibrium**: Systems settle into 2K + U = 0
- **Jeans equations replace fluid equations**: We use moments of the collisionless Boltzmann

The mathematics looks similar (both involve velocity moments), but the physics is fundamentally different.
:::

### 5.3 The Jeans Equations: Stellar Fluid Dynamics

**Priority: 🔴 Essential**

Taking moments of the collisionless Boltzmann equation gives the **Jeans equations** — the stellar dynamics equivalent of fluid equations.

#### Zeroth Moment: Continuity

Multiply by 1 and integrate over velocity space:

$$\boxed{\frac{\partial \nu}{\partial t} + \nabla \cdot (\nu \vec{u}) = 0}$$

where $\nu(\mathbf{r},t)$ is the stellar number density and $\vec{u} = \langle \vec{v} \rangle$ is the mean stellar velocity. This is mass conservation for a "fluid" of stars.

#### First Moment: Momentum (Jeans Equation)

Multiply by $\mathbf{v}$ and integrate. For a spherically symmetric system in steady state:

$$\boxed{\frac{1}{\nu}\frac{d(\nu \sigma_r^2)}{dr} + \frac{2\beta\sigma_r^2}{r} = -\frac{d\Phi}{dr} = -\frac{GM_r}{r^2}}$$

This is the stellar dynamics equivalent of hydrostatic equilibrium! The left side represents "pressure" support from stellar random motions, while the right side is gravity.

:::{margin}
**Anisotropy Parameter $(\beta)$**  
Measures orbital shape distribution in stellar systems. Defined as $\beta = 1 - \sigma_\theta^2/\sigma_r^2$. Values: $\beta = 0$ for isotropic orbits, $\beta > 0$ for radially-biased orbits, $\beta < 0$ for tangentially-biased orbits.
:::

The **anisotropy parameter** $\beta$ captures something gas doesn't have — stars can have different velocity dispersions in different directions:

- $\beta = 0$: isotropic orbits ($\sigma_r = \sigma_\theta = \sigma_\phi$)
- $\beta \to 1$: radial orbits dominate (like comets)
- $\beta < 0$: tangential orbits dominate (like planets)

### 5.4 The Beautiful Parallel: From Atoms to Stars

**Priority: 🔴 Essential**

The profound insight is that the same mathematical framework describes both stellar interiors (atoms as particles) and star clusters (stars as particles):

| Quantity | Stellar Interior (atoms) | Star Cluster (stars) |
|----------|-------------------------|---------------------|
| "Particles" | Atoms with mass $m$ | Stars with mass $M_\star$ |
| Number density | $n(\mathbf{r})$ atoms/cm³ | $\nu(\mathbf{r})$ stars/pc³ |
| Velocity spread | Temperature $T$ | Dispersion $\sigma^2$ |
| "Pressure" | $P = nkT$ (thermal) | $\Pi_{ij} = \nu\sigma_{ij}^2$ (kinetic) |
| Equilibrium | $\frac{dP}{dr} = -\rho g$ | Jeans equation |
| Energy exchange | Collisions (nanoseconds) | Relaxation ($10^{10}$ yr) |

Both equations come from the same source — the first moment of the Boltzmann equation! The only differences are:

1. **Collision timescale**: Atoms collide constantly → isotropic pressure. Stars rarely "collide" → anisotropic "pressure"
2. **Thermalization**: Atoms reach Maxwell-Boltzmann quickly. Stars never thermalize
3. **Extra term**: The $2\beta\sigma_r^2/r$ term appears because stellar systems can have anisotropic orbits

:::{admonition} 🎯 The Profound Unity: One Framework, All Scales
:class: important

We've now derived fundamental equations at vastly different scales using the SAME mathematical machinery:

**The Moment Method Applied to Different "Particles":**

| System | "Particle" | Particle Mass | 0th Moment | 1st Moment |
|--------|-----------|---------------|------------|------------|
| Stellar interior | Atoms | $\sim 2 \times 10^{-24}$ g | Mass continuity | Hydrostatic equilibrium |
| Star cluster | Stars | $\sim 2 \times 10^{33}$ g | Stellar continuity | Jeans equation |
| Galaxy | Star clusters | $\sim 10^{38}$ g | Cluster continuity | Cluster Jeans equation |
| Universe | Galaxies | $\sim 10^{44}$ g | Galaxy continuity | Cosmic Jeans equation |

That's a range of 68 orders of magnitude in mass! Yet the mathematics is identical:

1. Write down the Boltzmann equation for your "particles"
2. Take the 0th moment → continuity equation
3. Take the 1st moment → force balance equation
4. Take the 2nd moment → energy equation (if needed)

**Why this works**: Statistical mechanics doesn't care what you call a "particle" — only that you have many of them. Whether it's $10^{57}$ atoms in a star or $10^{11}$ stars in a galaxy, the statistical framework applies.
:::

## Part 2 Synthesis: Collisionless Statistics Creates Structure

You've discovered that stellar dynamics is just statistical mechanics without thermalization. The key differences from gases:

1. **No collisions = No thermalization**: Stars maintain distinct orbits for billions of years. This creates rich structures (spiral arms, bars, streams) impossible in gases.

2. **Anisotropy matters**: Without collisions to isotropize velocities, radial and tangential dispersions differ. The β parameter becomes essential.

3. **Phase space structure persists**: Stellar streams maintain coherent phase space structure for gigayears. This "memory" lets us reconstruct galactic history.

4. **Same math, different physics**: The Jeans equations are mathematically identical to fluid equations but describe fundamentally different physics — orbits rather than pressure.

The profound realization: **The universe recycles the same statistical framework at every scale**. Master it once with atoms, apply it to stars, extend it to galaxies. The labels change but the mathematics is eternal.

Ready to see the universal diagnostic that revealed dark matter? The virial theorem awaits in Part 3.

---

## Navigation

[← Part 1: Phase Space](./01-phase-space.md) | [Module 3 Home](./00-overview.md) | [Part 3: The Virial Theorem →](./03-virial-theorem.md)