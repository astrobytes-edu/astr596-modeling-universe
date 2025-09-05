## Part 2: From Distributions to Physics - Moments

### Bridge: Why We Need More Than Distributions

**Priority: 🔴 Essential**

Part 1 showed you that macroscopic properties emerge from statistical distributions. Temperature characterizes velocity distributions, pressure emerges from momentum transfer statistics, and marginalization extracts the information we need. But here's the challenge: we have a distribution function $f(\vec{r}, \vec{v}, t)$ describing $10^{57}$ particles in 6D phase space. How do we extract the physics—the actual equations that govern stellar structure, fluid dynamics, and energy transport?

The answer is **moments**—weighted averages that transform probability distributions into conservation laws. This isn't just a mathematical trick. It's how nature compresses infinite-dimensional information into the handful of quantities we can measure: density, velocity, pressure, temperature. Every conservation law in physics is really a statement about how moments of distributions evolve in time.

### 2.1 The Boltzmann Equation: Evolution of Distributions

**Priority: 🔴 Essential**

Before we can take moments, we need the master equation that governs how distributions evolve. The **Boltzmann equation** is Newton's second law for probability distributions—it tells us how the distribution function $f(\vec{r}, \vec{v}, t)$ changes in time due to particle motion, forces, and collisions.

:::{margin}
**Distribution Function**
$f(\vec{r}, \vec{v}, t)$ gives the number density of particles at position $\vec{r}$ with velocity $\vec{v}$ at time $t$. Integrating over all velocities gives spatial density; integrating over all positions gives velocity distribution.
:::

The distribution function contains all statistical information about our system:
$$f(\vec{r}, \vec{v}, t) \, d^3r \, d^3v = \text{number of particles in phase space element } d^3r \, d^3v$$

The Boltzmann equation describes how this distribution evolves:

$$\boxed{\frac{\partial f}{\partial t} + \vec{v} \cdot \nabla_r f + \frac{\vec{F}}{m} \cdot \nabla_v f = \left(\frac{\partial f}{\partial t}\right)_{\text{coll}}}$$

Let's understand each term physically:

- **$\frac{\partial f}{\partial t}$**: Local time change - how the distribution changes at a fixed point in phase space
- **$\vec{v} \cdot \nabla_r f$**: Streaming/advection - particles moving through space change the local density
- **$\frac{\vec{F}}{m} \cdot \nabla_v f$**: Force term - accelerations change particle velocities, reshaping the velocity distribution
- **$\left(\frac{\partial f}{\partial t}\right)_{\text{coll}}$**: Collision term - particle interactions scramble velocities, driving toward equilibrium

:::{admonition} 🔗 Connection to Your Projects
:class: note

**Project 2 (N-body)**: When collision term = 0, you get the collisionless Boltzmann equation that governs star clusters. Stars don't collide, so their distribution evolves only through gravity.

**Project 3 (MCRT)**: Photons follow a similar equation with scattering as the "collision" term. The radiative transfer equation is essentially the Boltzmann equation for photons.

**Project 4 (MCMC)**: Your Markov chains follow a master equation analogous to Boltzmann, with the transition kernel playing the role of collisions, driving toward the equilibrium (posterior) distribution.
:::

### 2.2 Taking Moments: From Microscopic to Macroscopic

**Priority: 🔴 Essential**

Now comes the magic. We transform the unsolvable Boltzmann equation (tracking $10^{57}$ particles) into a few macroscopic equations by taking **moments**—multiplying by powers of velocity and integrating.

:::{margin}
**Moment**
The $n$-th moment of a distribution is $\langle v^n \rangle = \int v^n f(\vec{v}) d^3v$. Each moment extracts different physical information.
:::

The procedure is systematic:
1. Multiply the Boltzmann equation by $v^n$ (or $m v^n$ for dimensional consistency)
2. Integrate over all velocities
3. Get an evolution equation for the $n$-th moment

This transforms:
- **Microscopic**: Evolution of $f(\vec{r}, \vec{v}, t)$ - impossibly complex
- **Macroscopic**: Evolution of moments like $\rho(\vec{r}, t)$, $\vec{u}(\vec{r}, t)$, $P(\vec{r}, t)$ - tractable!

Let's see this transformation explicitly.

### 2.3 The Zeroth Moment: Mass Conservation

**Priority: 🔴 Essential**

Multiply the Boltzmann equation by mass $m$ and integrate over all velocities:

$$m \int \left[\frac{\partial f}{\partial t} + \vec{v} \cdot \nabla_r f + \frac{\vec{F}}{m} \cdot \nabla_v f\right] d^3v = 0$$

(The collision integral vanishes because collisions conserve particle number.)

**Term by term:**

**First term:**
$$m \int \frac{\partial f}{\partial t} d^3v = \frac{\partial}{\partial t} \left(m \int f d^3v\right) = \frac{\partial \rho}{\partial t}$$

where $\rho = mn$ is the mass density.

**Second term:**
$$m \int \vec{v} \cdot \nabla_r f d^3v = \nabla_r \cdot \left(m \int \vec{v} f d^3v\right) = \nabla \cdot (\rho \vec{u})$$

where $\vec{u} = \langle \vec{v} \rangle$ is the mean velocity (bulk flow).

**Third term:** Vanishes! After integration by parts, we get boundary terms at $v = \pm\infty$ where $f \to 0$.

**Result - The Continuity Equation:**
$$\boxed{\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \vec{u}) = 0}$$

This is **mass conservation**! Density changes only due to flow divergence. The zeroth moment transformed the complex Boltzmann equation into a simple conservation law.

### 2.4 The First Moment: Momentum Conservation

**Priority: 🔴 Essential**

Now multiply by $m\vec{v}$ (momentum) and integrate:

$$m \int \vec{v} \left[\frac{\partial f}{\partial t} + \vec{v} \cdot \nabla_r f + \frac{\vec{F}}{m} \cdot \nabla_v f\right] d^3v = 0$$

After working through the algebra (integrating by parts, using the definition of pressure):

**First term:** $\frac{\partial (\rho \vec{u})}{\partial t}$ - rate of change of momentum density

**Second term:** $\nabla \cdot (\rho \vec{u} \vec{u} + \mathbf{P})$ where the pressure tensor emerges:
$$P_{ij} = m \int (v_i - u_i)(v_j - u_j) f d^3v = \rho \langle (v_i - u_i)(v_j - u_j) \rangle$$

**Third term:** $\rho \vec{F}$ - force density

**Result - The Euler Equation:**
$$\boxed{\rho \frac{D\vec{u}}{Dt} = -\nabla \cdot \mathbf{P} + \rho \vec{F}}$$

where $\frac{D}{Dt} = \frac{\partial}{\partial t} + \vec{u} \cdot \nabla$ is the material derivative.

For isotropic pressure (same in all directions): $P_{ij} = P\delta_{ij}$, giving:
$$\boxed{\rho \frac{D\vec{u}}{Dt} = -\nabla P + \rho \vec{F}}$$

This is Newton's second law for fluids! Force = mass × acceleration, but now for a continuous medium.

:::{important} 💡 The Pressure-Variance Connection
The pressure tensor that emerged naturally from taking the first moment is:
$$P_{ij} = \rho \langle (v_i - u_i)(v_j - u_j) \rangle$$

This is the **covariance matrix** of the velocity distribution! 
- Diagonal terms: variance in each direction (related to temperature)
- Off-diagonal terms: correlations between velocity components (shear stresses)

For ideal gas with Maxwell-Boltzmann velocities: $P = nkT$ emerges naturally because the variance of the velocity distribution is $\langle (v - u)^2 \rangle = 3kT/m$.

**This isn't coincidence**: pressure IS the momentum flux, which IS the velocity variance times mass density!
:::

### 2.5 The Second Moment: Energy Conservation

**Priority: 🟡 Standard Path**

Multiply by $\frac{1}{2}m|\vec{v}|^2$ (kinetic energy) and integrate:

After substantial algebra, we get the energy equation:
$$\boxed{\frac{\partial E}{\partial t} + \nabla \cdot \left[(E + P)\vec{u} + \vec{q}\right] = \rho \vec{F} \cdot \vec{u}}$$

where:
- $E = \frac{1}{2}\rho u^2 + \rho e$ is total energy density (kinetic + internal)
- $\vec{q}$ is the heat flux (from the third moment)
- The pressure $P$ appears in the energy flux naturally!

### 2.6 The Pattern: Moment Hierarchy

**Priority: 🔴 Essential**

Taking moments creates a hierarchy of equations:

| Moment | Weight | Conservation Law | Physical Quantity |
|--------|--------|-----------------|------------------|
| 0th | $m$ | Continuity | Mass density $\rho$ |
| 1st | $m\vec{v}$ | Momentum | Flow velocity $\vec{u}$, Pressure $P$ |
| 2nd | $\frac{1}{2}m v^2$ | Energy | Temperature $T$, Heat flux $\vec{q}$ |
| 3rd | Higher order | Heat flux | Transport coefficients |

**The closure problem**: Each equation involves the next higher moment! 
- Continuity involves velocity (1st moment)
- Momentum involves pressure (2nd moment) 
- Energy involves heat flux (3rd moment)

We need a **closure relation** to truncate this hierarchy. In LTE (Local Thermodynamic Equilibrium):
- Equation of state: $P = P(\rho, T)$ - relates pressure to lower moments
- Fourier's law: $\vec{q} = -\kappa \nabla T$ - relates heat flux to temperature gradient

### 2.7 Connection to Stellar Structure

**Priority: 🔴 Essential**

For a spherically symmetric star in hydrostatic equilibrium, our moment equations simplify dramatically:

- **Time-independent**: $\partial/\partial t = 0$
- **Spherical symmetry**: Only radial dependence
- **No bulk flow**: $\vec{u} = 0$ (hydrostatic)

The moment equations become:

**From continuity (0th moment):**
$$\frac{d}{dr}(4\pi r^2 \rho) = 0 \quad \Rightarrow \quad \boxed{\frac{dM_r}{dr} = 4\pi r^2 \rho}$$

**From momentum (1st moment):**
$$\frac{dP}{dr} = -\rho g = -\frac{GM_r \rho}{r^2} \quad \Rightarrow \quad \boxed{\frac{dP}{dr} = -\frac{GM_r \rho}{r^2}}$$

**From energy (2nd moment) + radiative transfer:**
$$\boxed{\frac{dL_r}{dr} = 4\pi r^2 \rho \epsilon}$$
$$\boxed{\frac{dT}{dr} = -\frac{3\kappa \rho L_r}{16\pi ac r^2 T^3}}$$

These are the four stellar structure equations! They emerged from taking moments of the Boltzmann equation for the particles in the star.

:::{admonition} 🎯 The Profound Realization
:class: important

We started with:
- The Boltzmann equation describing $10^{57}$ particles in 6D phase space
- An impossibly complex problem

Through taking moments, we got:
- Four ordinary differential equations in radius
- A completely tractable problem

This is the power of statistical mechanics: **moments transform distributions into physics**. The same procedure works whether your "particles" are atoms (giving stellar structure), photons (giving radiative transfer), or stars (giving galactic dynamics).
:::

### Part 2 Synthesis: The Universal Recipe

:::{admonition} 🎯 What We Just Learned
:class: important

**Taking moments of the Boltzmann equation transforms statistical distributions into macroscopic physics:**

1. **The Boltzmann equation** governs how probability distributions evolve in phase space
2. **Taking moments** (weighted velocity averages) extracts macroscopic quantities
3. **Each moment gives a conservation law**:
   - 0th moment → Mass conservation
   - 1st moment → Momentum conservation (Newton's 2nd law)
   - 2nd moment → Energy conservation
4. **Pressure emerges as velocity variance**: $P = \rho \text{Var}(v)$
5. **The same procedure works at all scales**: atoms → stellar structure, stars → galactic dynamics

This is why we can model stars despite their enormous particle numbers. Statistical mechanics provides the compression algorithm: from $10^{57}$ particle trajectories to 4 differential equations. The information we "lose" (individual particle paths) doesn't affect the macroscopic physics we care about.

**For your projects**: This moment-taking procedure appears everywhere. In Project 2, you're essentially sampling from the distribution function. In Project 3, you'll take moments of the photon distribution. In Project 4, MCMC explores parameter distributions whose moments give you posterior statistics.
:::

---

## Part 3: Random Sampling - From Distributions to Simulations

### 3.1 Why Random Sampling Matters in Astrophysics

**Priority: 🔴 Essential**

So far, we've learned that astrophysical systems are governed by probability distributions—Maxwell-Boltzmann for velocities, Planck for photon energies, and various power laws for stellar masses. But how do we actually generate samples from these distributions for our simulations?

This is where **random sampling** becomes essential. In Project 2, you'll need to:
- Sample stellar masses from the Kroupa Initial Mass Function (IMF)
- Sample positions from Plummer or King profiles 
- Sample velocities from Maxwell-Boltzmann distributions

The challenge: your computer only knows how to generate uniform random numbers between 0 and 1. How do we transform these into samples from complex astrophysical distributions?

### 3.2 The Cumulative Distribution Function (CDF)

**Priority: 🔴 Essential**

The key to random sampling is the **Cumulative Distribution Function (CDF)**:

$$\boxed{F(x) = P(X \leq x) = \int_{-\infty}^{x} f(x') dx'}$$

where $f(x)$ is the probability density function (PDF).

:::{margin}
**CDF Properties**
- Always increases from 0 to 1
- $F(-\infty) = 0$, $F(\infty) = 1$
- Slope = PDF: $dF/dx = f(x)$
- Jumps at discrete values
:::

The CDF answers: "What's the probability that a random sample is less than or equal to $x$?"

**Key insight**: The CDF always maps to the range [0, 1], exactly the range of uniform random numbers!

**Example: Exponential Distribution**
PDF: $f(x) = \lambda e^{-\lambda x}$ for $x \geq 0$

CDF: $F(x) = \int_0^x \lambda e^{-\lambda x'} dx' = 1 - e^{-\lambda x}$

The CDF smoothly increases from 0 to 1 as $x$ goes from 0 to $\infty$.

### 3.3 Inverse Transform Sampling

**Priority: 🔴 Essential**

The **inverse transform method** is the fundamental technique for random sampling:

:::{admonition} 🎯 The Inverse Transform Algorithm
:class: important

To sample from distribution with CDF $F(x)$:
1. Generate uniform random number $u \sim U(0,1)$
2. Solve $F(x) = u$ for $x$
3. The solution $x = F^{-1}(u)$ follows the desired distribution

**Why this works**: If $U \sim \text{Uniform}(0,1)$, then $F^{-1}(U)$ has CDF equal to $F$.
:::

**Example: Exponential Distribution**
1. Generate $u \sim U(0,1)$
2. Solve: $1 - e^{-\lambda x} = u$
3. Get: $x = -\frac{1}{\lambda}\ln(1-u)$

Since $(1-u)$ is also uniform on [0,1], we can simplify to:
$$\boxed{x = -\frac{1}{\lambda}\ln(u)}$$

This is how you'll sample photon path lengths in Project 3!

### 3.4 Sampling from Power Laws: The Kroupa IMF

**Priority: 🔴 Essential**

The **Initial Mass Function (IMF)** describes the distribution of stellar masses at birth. The Kroupa IMF is a broken power law:

$$\xi(m) \propto \begin{cases}
m^{-0.3} & 0.01 < m/M_\odot < 0.08 \\
m^{-1.3} & 0.08 < m/M_\odot < 0.5 \\
m^{-2.3} & 0.5 < m/M_\odot < 150
\end{cases}$$

To sample from a single power law segment $f(m) \propto m^{-\alpha}$ between $[m_{\min}, m_{\max}]$:

**Step 1: Normalize the PDF**
$$f(m) = \frac{(1-\alpha)m^{-\alpha}}{m_{\max}^{1-\alpha} - m_{\min}^{1-\alpha}} \quad \text{for } \alpha \neq 1$$

**Step 2: Compute the CDF**
$$F(m) = \frac{m^{1-\alpha} - m_{\min}^{1-\alpha}}{m_{\max}^{1-\alpha} - m_{\min}^{1-\alpha}}$$

**Step 3: Invert to get sampling formula**
$$\boxed{m = \left[m_{\min}^{1-\alpha} + u(m_{\max}^{1-\alpha} - m_{\min}^{1-\alpha})\right]^{1/(1-\alpha)}}$$

For the Kroupa IMF with multiple segments, you first choose which segment based on the integrated probability in each range, then sample within that segment.

:::{admonition} 💻 Implementation for Project 2
:class: note

```python
def sample_kroupa_imf(n_stars):
    """Sample stellar masses from Kroupa IMF"""
    masses = []
    
    # Segment boundaries and slopes
    m_bounds = [0.01, 0.08, 0.5, 150]  # M_sun
    alphas = [0.3, 1.3, 2.3]
    
    # Calculate probability of each segment (normalization)
    # ... (compute relative probabilities)
    
    for i in range(n_stars):
        u = np.random.random()
        
        # Choose segment based on cumulative probability
        # Then sample within that segment using formula above
        # ... (implementation details)
        
        masses.append(m)
    
    return np.array(masses)
```

This gives you a realistic stellar population with many low-mass stars and few high-mass stars, matching observations!
:::

### 3.5 Rejection Sampling: When Inversion Fails

**Priority: 🟡 Standard Path**

Sometimes the CDF can't be inverted analytically. **Rejection sampling** provides an alternative:

:::{admonition} 🎯 The Rejection Algorithm
:class: important

To sample from PDF $f(x)$ with support on $[a,b]$:
1. Find constant $M$ such that $f(x) \leq M$ for all $x \in [a,b]$
2. Generate $x \sim U(a,b)$ and $y \sim U(0,M)$
3. If $y \leq f(x)$: accept $x$
4. Otherwise: reject and repeat

**Efficiency**: Acceptance rate = (area under $f$) / (area of box) = $1/M(b-a)$
:::

**Visual intuition**: You're throwing darts uniformly in a box. Keep only the darts that land under the curve $f(x)$.

### 3.6 Sampling Spatial Distributions: Plummer Profile

**Priority: 🔴 Essential**

For star clusters, you need to sample positions from density profiles. The **Plummer profile** is popular because it's analytically tractable:

$$\rho(r) = \frac{3M}{4\pi a^3}\left(1 + \frac{r^2}{a^2}\right)^{-5/2}$$

where $a$ is the scale radius and $M$ is total mass.

**The trick**: Sample from the mass profile (cumulative mass), not density directly!

**Step 1: Mass enclosed within radius $r$**
$$M(r) = M \frac{r^3}{(r^2 + a^2)^{3/2}}$$

**Step 2: Cumulative distribution**
$$F(r) = \frac{M(r)}{M} = \frac{r^3}{(r^2 + a^2)^{3/2}}$$

**Step 3: Invert** (this one's tricky!)
Let $u = F(r)$, then solving gives:
$$\boxed{r = \frac{a u^{1/3}}{\sqrt{1 - u^{2/3}}}}$$

**Step 4: Generate 3D positions**
1. Sample radius $r$ using formula above
2. Generate random point on sphere: 
   - $\theta = \arccos(1 - 2u_1)$ (polar angle)
   - $\phi = 2\pi u_2$ (azimuthal angle)
3. Convert to Cartesian: $(x,y,z) = r(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$

### 3.7 Sampling Velocities: Maxwell-Boltzmann in 3D

**Priority: 🔴 Essential**

For your star cluster, you need velocity distributions. From Part 1, we know velocities follow Maxwell-Boltzmann. Here's how to sample them:

**Method 1: Box-Muller for Gaussian components**

Each velocity component is Gaussian with variance $\sigma^2 = kT/m$:

```python
def sample_maxwell_boltzmann_velocities(n, sigma):
    """Sample 3D velocities from Maxwell-Boltzmann"""
    # Box-Muller transform for Gaussian
    u1, u2 = np.random.random((2, n))
    
    # Generate standard normal
    z1 = np.sqrt(-2*np.log(u1)) * np.cos(2*np.pi*u2)
    z2 = np.sqrt(-2*np.log(u1)) * np.sin(2*np.pi*u2)
    
    # Need one more for 3D
    u3, u4 = np.random.random((2, n))
    z3 = np.sqrt(-2*np.log(u3)) * np.cos(2*np.pi*u4)
    
    # Scale by velocity dispersion
    vx = sigma * z1
    vy = sigma * z2  
    vz = sigma * z3
    
    return vx, vy, vz
```

**Method 2: Direct sampling of speed + random direction**

Since the speed distribution $f(v) \propto v^2 e^{-v^2/2\sigma^2}$ is harder to invert, use rejection sampling or the Gaussian component method above.

### 3.8 Putting It Together: Complete Initial Conditions

**Priority: 🔴 Essential**

For Project 2, you'll combine all these sampling techniques:

```python
def create_star_cluster(n_stars, cluster_mass, scale_radius):
    """Generate a realistic star cluster"""
    
    # 1. Sample masses from Kroupa IMF
    masses = sample_kroupa_imf(n_stars)
    
    # 2. Sample positions from Plummer profile
    positions = sample_plummer_positions(n_stars, scale_radius)
    
    # 3. Sample velocities from Maxwell-Boltzmann
    # Velocity dispersion from virial theorem
    sigma = np.sqrt(G * cluster_mass / (6 * scale_radius))
    velocities = sample_maxwell_boltzmann(n_stars, sigma)
    
    # 4. Remove net momentum (move to CM frame)
    velocities -= np.mean(velocities, axis=0)
    
    return masses, positions, velocities
```

This creates a self-consistent star cluster in approximate virial equilibrium!

### Part 3 Synthesis: Random Sampling Toolkit

:::{admonition} 🎯 What We Just Learned
:class: important

**Random sampling transforms probability theory into computational practice:**

1. **CDFs are the key**: They map any distribution to [0,1], the range of uniform random numbers
2. **Inverse transform is fundamental**: $x = F^{-1}(u)$ samples from any invertible CDF
3. **Rejection sampling handles complex PDFs**: Accept/reject points under the curve
4. **Astrophysical distributions have structure**:
   - Power laws (IMF) → Use piecewise inverse transform
   - Density profiles (Plummer) → Sample from mass profile, not density
   - Maxwell-Boltzmann → Use Box-Muller for Gaussian components

**For Project 2**: You now have all the tools to create realistic initial conditions:
- Stellar masses following observed IMF
- Spatial distribution matching cluster profiles
- Velocities in thermal equilibrium

**For Project 3**: These same techniques will help you sample:
- Photon emission directions (isotropic on sphere)
- Path lengths (exponential distribution)
- Scattering angles (phase functions)

**The key insight**: Random sampling is how we bridge from theoretical distributions (Part 1) to practical simulations. It's the computational realization of statistical mechanics!
:::

---

## Module 2a Complete Summary

You've now mastered three fundamental aspects of computational astrophysics:

**Part 1: Statistical Foundations**
- Temperature, pressure, and other macroscopic properties emerge from distributions
- Maximum entropy gives the least-biased distributions
- Large numbers create order from chaos

**Part 2: Moments Transform Distributions into Physics**
- The Boltzmann equation governs distribution evolution
- Taking moments yields conservation laws
- The same procedure works from atoms to galaxies

**Part 3: Random Sampling Brings Theory to Life**
- CDFs and inverse transforms sample any distribution
- Rejection sampling handles complex cases
- You can now create realistic astrophysical initial conditions

These three parts form a complete framework: distributions describe nature (Part 1), moments extract the physics (Part 2), and sampling lets us simulate (Part 3). With these tools, you're ready to tackle any computational astrophysics problem!

**Next Steps**: Module 2b will show you how these principles scale up to stellar modeling, where $10^{57}$ particles become just four differential equations through the magic of statistical mechanics.