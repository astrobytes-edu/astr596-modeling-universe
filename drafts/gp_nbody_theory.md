# Comprehensive Lecture Notes: From Stars to Gaussian Processes

## Preface: The Journey Ahead

These lecture notes trace a complete path through computational astrophysics, from the fundamental physics of stars to the sophisticated mathematics of machine learning. We'll build each concept from first principles, explaining not just the equations but the physical and mathematical intuition behind them. Every symbol will be defined, every assumption made explicit, and every connection between topics illuminated.

Think of this journey as climbing a mountain. Each base camp (project) provides the tools and understanding needed for the next ascent. By the summit, you'll look back and see how every step was necessary, how every concept connects, and how the same mathematical patterns echo throughout the entire landscape of computational science.

---

# Chapter 1: The Foundation - Object-Oriented Stellar Physics

## 1.1 The Physics of Stars: From Gas to Light

### The Fundamental Balance

A star exists because of a delicate balance between two competing forces. Gravity, described by Newton's law, creates an inward force trying to collapse the star:

$$F_{\text{gravity}} = -\frac{GMm}{r^2}$$

where:
- $G = 6.674 \times 10^{-11}$ m³ kg⁻¹ s⁻² is the gravitational constant
- $M$ is the mass contained within radius $r$
- $m$ is a small test mass at radius $r$
- The negative sign indicates the force points inward

Meanwhile, pressure from hot gas and radiation creates an outward force. The pressure gradient (change in pressure with radius) generates a force per unit volume:

$$\frac{dP}{dr} = -\frac{GM\rho}{r^2}$$

where:
- $P$ is pressure (force per unit area, measured in Pascals)
- $\rho$ is density (mass per unit volume, kg/m³)
- This equation tells us how pressure must change with radius to balance gravity

This is the equation of hydrostatic equilibrium—"hydrostatic" because the fluid (stellar gas) is static (not accelerating), and "equilibrium" because forces balance.

### The Origin of Pressure

But what creates this pressure? In stars, pressure comes from two sources:

**Gas Pressure**: Hot gas particles bounce around randomly, creating pressure through their kinetic energy. From the kinetic theory of gases:

$$P_{\text{gas}} = nkT = \frac{\rho kT}{\mu m_H}$$

where:
- $n$ is the number density of particles (particles per unit volume)
- $k = 1.381 \times 10^{-23}$ J/K is Boltzmann's constant
- $T$ is temperature in Kelvin
- $\mu$ is the mean molecular weight (average mass of particles in units of hydrogen mass)
- $m_H = 1.673 \times 10^{-27}$ kg is the mass of a hydrogen atom

**Radiation Pressure**: Photons carry momentum and exert pressure when absorbed or scattered:

$$P_{\text{rad}} = \frac{1}{3}aT^4$$

where:
- $a = 7.566 \times 10^{-16}$ J m⁻³ K⁻⁴ is the radiation constant
- The factor of 1/3 comes from averaging over all directions in 3D space
- The $T^4$ dependence comes from the Stefan-Boltzmann law

For the Sun, gas pressure dominates, but for massive stars, radiation pressure becomes increasingly important.

### Energy Generation and Transport

Stars shine because nuclear fusion in their cores converts mass to energy via Einstein's relation $E = mc^2$. For stars like the Sun, the primary reaction is the proton-proton chain, which effectively converts four hydrogen nuclei into one helium nucleus:

$$4^1\text{H} \rightarrow ^4\text{He} + 2e^+ + 2\nu_e + 26.73 \text{ MeV}$$

The energy generation rate depends sensitively on temperature. For the proton-proton chain:

$$\epsilon_{pp} \propto \rho T^4$$

For more massive stars burning via the CNO cycle:

$$\epsilon_{CNO} \propto \rho T^{16}$$

This extreme temperature dependence means energy generation is highly concentrated in the stellar core.

### The Mass-Luminosity Relation

Now we can derive why luminosity scales with mass. Consider a star in equilibrium. The virial theorem tells us that the average kinetic energy relates to gravitational potential energy:

$$\langle T \rangle \sim \frac{GM}{R}$$

where $\langle T \rangle$ is the average temperature, $M$ is stellar mass, and $R$ is stellar radius.

From hydrostatic equilibrium, the central pressure must support the weight of the star:

$$P_c \sim \frac{GM\rho}{R} \sim \frac{GM^2}{R^4}$$

Using the ideal gas law $P \sim \rho T$:

$$\rho_c T_c \sim \frac{M^2}{R^4}$$

Since $\rho \sim M/R^3$:

$$T_c \sim \frac{M}{R}$$

The luminosity depends on the energy generation rate integrated over the star's volume:

$$L \sim \epsilon \cdot M \sim \rho T^{\nu} \cdot M$$

where $\nu \approx 4$ for the pp-chain. Substituting our scaling relations:

$$L \sim \frac{M}{R^3} \cdot \left(\frac{M}{R}\right)^4 \cdot M \sim \frac{M^6}{R^7}$$

But we also need to consider how energy escapes. For radiative transport, luminosity scales as:

$$L \sim \frac{R^2 T^4}{R} \sim RT^4 \sim R \cdot \left(\frac{M}{R}\right)^4 \sim \frac{M^4}{R^3}$$

Combining energy generation and transport constraints yields:

$$L \propto M^{\alpha}$$

where $\alpha \approx 3.5$ for solar-mass stars. This relation shifts for very low-mass stars (where convection dominates) and very high-mass stars (where radiation pressure dominates).

## 1.2 Computational Representation: Objects as Physical Entities

### Why Object-Oriented Programming?

When we translate stellar physics into code, we face a fundamental design choice. We could represent stellar properties as separate arrays:

```
masses[i] = 1.0  # Solar masses
luminosities[i] = 1.0  # Solar luminosities  
temperatures[i] = 5778  # Kelvin
ages[i] = 0.0  # Gigayears
```

But this obscures the physical relationships. A star's luminosity isn't independent of its mass—they're connected by physical laws. Object-oriented programming lets us encode these relationships directly:

```
class Star:
    def __init__(self, mass):
        self.mass = mass  # Fundamental property
        self.age = 0.0    # Will evolve over time
        
    def luminosity(self):
        # Luminosity derived from mass
        return self.mass ** 3.5
        
    def temperature(self):
        # Temperature from Stefan-Boltzmann law
        # L = 4πR²σT⁴
        radius = self.radius()
        return (self.luminosity() * L_sun / 
                (4 * pi * radius**2 * sigma))**(1/4)
```

The object encapsulates both state (mass, age) and behavior (evolution, radiation). This mirrors physical reality: a star is an entity with properties that determine its behavior.

### The Initial Mass Function: Statistics of Star Formation

When stars form from a collapsing molecular cloud, fragmentation and competitive accretion produce a characteristic distribution of masses. The Initial Mass Function (IMF) describes the number of stars born per unit mass interval:

$$\xi(M) = \frac{dN}{dM}$$

This has units of "number per unit mass" and tells us: in a mass interval $[M, M+dM]$, we expect to find $\xi(M)dM$ stars.

The Salpeter IMF (1955) proposed a simple power law:

$$\xi(M) = \xi_0 M^{-2.35}$$

where $\xi_0$ is a normalization constant. But observations revealed this breaks down at low masses. The Kroupa IMF (2001) provides a better fit:

$$\xi(M) = \begin{cases} 
\xi_0 M^{-0.3} & 0.01 < M < 0.08 M_{\odot} \\
\xi_1 M^{-1.3} & 0.08 < M < 0.5 M_{\odot} \\
\xi_2 M^{-2.3} & 0.5 < M < 100 M_{\odot}
\end{cases}$$

The constants $\xi_0$, $\xi_1$, $\xi_2$ are chosen to ensure continuity at the break points. The physical interpretation of these breaks is profound:

- Below 0.08 $M_{\odot}$: Objects cannot sustain hydrogen fusion (brown dwarfs)
- The 0.5 $M_{\odot}$ break: Transition between different formation mechanisms
- Above 100 $M_{\odot}$: Radiation pressure prevents further accretion

### Stellar Populations and Vectorization

A globular cluster contains perhaps 10⁶ stars, all formed from the same gas cloud. They share the same age and initial composition but have different masses drawn from the IMF. When we model such a population, we face a computational challenge: how to efficiently evolve millions of stars?

The naive approach uses loops:
```
for star in population:
    star.luminosity = star.mass ** 3.5
    star.temperature = calculate_temperature(star)
    star.radius = calculate_radius(star)
```

But modern processors can perform the same operation on multiple data points simultaneously (SIMD - Single Instruction, Multiple Data). NumPy exploits this through vectorization:

```
luminosities = masses ** 3.5  # All stars at once!
```

This isn't just faster—it represents a conceptual shift from thinking about individual stars to thinking about distributions. The luminosity function—the number of stars per unit luminosity interval—becomes:

$$\Phi(L) = \xi(M(L)) \left|\frac{dM}{dL}\right|$$

The Jacobian factor $|dM/dL|$ accounts for the transformation of variables. Since $L \propto M^{3.5}$:

$$\frac{dL}{dM} = 3.5M^{2.5} \implies \frac{dM}{dL} = \frac{1}{3.5}M^{-2.5} = \frac{1}{3.5}L^{-5/7}$$

This mathematical framework—transforming probability distributions under variable changes—will prove crucial when we encounter MCMC and change of variables in Bayesian inference.

---

# Chapter 2: The Dance of Gravity - N-body Dynamics

## 2.1 The Fundamental N-body Problem

### Newton's Law in Vector Form

For two bodies with masses $m_i$ and $m_j$ at positions $\vec{r}_i$ and $\vec{r}_j$, the gravitational force on body $i$ due to body $j$ is:

$$\vec{F}_{ij} = -G\frac{m_i m_j}{|\vec{r}_i - \vec{r}_j|^3}(\vec{r}_i - \vec{r}_j)$$

Let's unpack every component:
- $\vec{F}_{ij}$ is a vector force (has magnitude and direction)
- $G = 6.674 \times 10^{-11}$ m³ kg⁻¹ s⁻² is the universal gravitational constant
- $|\vec{r}_i - \vec{r}_j|$ is the Euclidean distance between the bodies
- The factor $|\vec{r}_i - \vec{r}_j|^3$ in the denominator comes from: distance squared for the inverse square law, plus one more factor to normalize the direction vector
- $(\vec{r}_i - \vec{r}_j)$ is the vector pointing from $j$ to $i$
- The negative sign ensures attraction (force points opposite to separation vector)

### The Complete N-body System

For N bodies, the total force on body $i$ is the vector sum of forces from all other bodies:

$$\vec{F}_i = \sum_{j \neq i} \vec{F}_{ij} = -Gm_i \sum_{j \neq i} \frac{m_j}{|\vec{r}_i - \vec{r}_j|^3}(\vec{r}_i - \vec{r}_j)$$

Newton's second law $\vec{F} = m\vec{a}$ gives us the acceleration:

$$\vec{a}_i = \frac{d^2\vec{r}_i}{dt^2} = -G \sum_{j \neq i} \frac{m_j}{|\vec{r}_i - \vec{r}_j|^3}(\vec{r}_i - \vec{r}_j)$$

This is a system of $3N$ coupled second-order differential equations (3 spatial dimensions × N bodies). We can rewrite it as $6N$ first-order equations by introducing velocities:

$$\frac{d\vec{r}_i}{dt} = \vec{v}_i$$
$$\frac{d\vec{v}_i}{dt} = -G \sum_{j \neq i} \frac{m_j}{|\vec{r}_i - \vec{r}_j|^3}(\vec{r}_i - \vec{r}_j)$$

### The Impossibility of Analytical Solutions

For N = 2 (like Earth orbiting the Sun), analytical solutions exist—the conic sections (circles, ellipses, parabolas, hyperbolas). But for N ≥ 3, no general analytical solution exists. This was proven by Poincaré, who showed the three-body problem exhibits chaos: solutions depend sensitively on initial conditions.

The Lyapunov time quantifies this chaos. Two trajectories initially separated by $\delta_0$ diverge exponentially:

$$\delta(t) \approx \delta_0 e^{\lambda t}$$

where $\lambda$ is the Lyapunov exponent. For the solar system, the Lyapunov time $\tau_L = 1/\lambda \approx 5$ million years. This means we cannot predict planetary positions beyond ~100 million years, even with perfect knowledge of current positions.

## 2.2 Numerical Integration: The Art of Discretization

### Euler's Method: Simple but Flawed

To solve the N-body equations numerically, we discretize time into small steps $\Delta t$. The simplest approach approximates derivatives with finite differences:

$$\frac{d\vec{r}}{dt} \approx \frac{\vec{r}(t + \Delta t) - \vec{r}(t)}{\Delta t}$$

Rearranging gives Euler's method:

$$\vec{r}(t + \Delta t) = \vec{r}(t) + \vec{v}(t)\Delta t$$
$$\vec{v}(t + \Delta t) = \vec{v}(t) + \vec{a}(t)\Delta t$$

This seems reasonable, but let's examine what happens to energy. The total energy of an N-body system is:

$$E = T + V = \sum_{i} \frac{1}{2}m_i v_i^2 - \sum_{i<j} \frac{Gm_i m_j}{r_{ij}}$$

where:
- $T$ is kinetic energy (energy of motion)
- $V$ is potential energy (energy of configuration)
- The potential energy is negative because we define zero potential at infinite separation

For an isolated system, energy should be conserved: $dE/dt = 0$. But with Euler's method, energy drifts systematically. To understand why, consider the Taylor expansion:

$$\vec{r}(t + \Delta t) = \vec{r}(t) + \vec{v}(t)\Delta t + \frac{1}{2}\vec{a}(t)\Delta t^2 + O(\Delta t^3)$$

Euler's method truncates at first order, introducing an error of $O(\Delta t^2)$ per step. Over time $T$, we take $N_{\text{steps}} = T/\Delta t$ steps, accumulating error:

$$\text{Total Error} \sim N_{\text{steps}} \times \Delta t^2 \sim \frac{T}{\Delta t} \times \Delta t^2 = T\Delta t$$

The error grows linearly with time! Worse, this error is systematic—always in the same direction—causing energy to grow exponentially.

### The Runge-Kutta Solution: Higher Order Accuracy

The fourth-order Runge-Kutta method (RK4) achieves higher accuracy by evaluating derivatives at multiple points:

$$k_1 = \vec{v}(t)$$
$$l_1 = \vec{a}(\vec{r}(t))$$

$$k_2 = \vec{v}(t) + \frac{\Delta t}{2}l_1$$
$$l_2 = \vec{a}(\vec{r}(t) + \frac{\Delta t}{2}k_1)$$

$$k_3 = \vec{v}(t) + \frac{\Delta t}{2}l_2$$
$$l_3 = \vec{a}(\vec{r}(t) + \frac{\Delta t}{2}k_2)$$

$$k_4 = \vec{v}(t) + \Delta t \cdot l_3$$
$$l_4 = \vec{a}(\vec{r}(t) + \Delta t \cdot k_3)$$

$$\vec{r}(t + \Delta t) = \vec{r}(t) + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
$$\vec{v}(t + \Delta t) = \vec{v}(t) + \frac{\Delta t}{6}(l_1 + 2l_2 + 2l_3 + l_4)$$

The weights (1, 2, 2, 1)/6 come from Simpson's rule for numerical integration. RK4 has error $O(\Delta t^5)$ per step, giving total error $O(\Delta t^4)$—much better than Euler! But it requires four force calculations per timestep, making it computationally expensive.

### Symplectic Integration: Preserving Geometric Structure

The breakthrough comes from recognizing that the N-body system is Hamiltonian. The Hamiltonian (total energy) is:

$$H(\vec{r}_1, ..., \vec{r}_N, \vec{p}_1, ..., \vec{p}_N) = \sum_{i} \frac{p_i^2}{2m_i} - \sum_{i<j} \frac{Gm_i m_j}{|\vec{r}_i - \vec{r}_j|}$$

where $\vec{p}_i = m_i\vec{v}_i$ is momentum. Hamilton's equations are:

$$\frac{d\vec{r}_i}{dt} = \frac{\partial H}{\partial \vec{p}_i} = \frac{\vec{p}_i}{m_i}$$

$$\frac{d\vec{p}_i}{dt} = -\frac{\partial H}{\partial \vec{r}_i} = \vec{F}_i$$

The phase space flow preserves the symplectic 2-form:

$$\omega = \sum_i d\vec{p}_i \wedge d\vec{r}_i$$

This is a deep geometric property: phase space volume is preserved (Liouville's theorem), and more subtly, the symplectic structure is preserved.

The Leapfrog integrator preserves these properties:

$$\vec{v}_{i}^{1/2} = \vec{v}_i^0 + \frac{\Delta t}{2}\vec{a}_i^0$$
$$\vec{r}_{i}^1 = \vec{r}_i^0 + \Delta t \cdot \vec{v}_i^{1/2}$$
$$\vec{v}_{i}^1 = \vec{v}_i^{1/2} + \frac{\Delta t}{2}\vec{a}_i^1$$

Why "Leapfrog"? Positions and velocities "leapfrog" over each other—velocities are defined at half-integer times, positions at integer times.

The Leapfrog integrator is:
- **Symplectic**: Preserves phase space structure
- **Time-reversible**: Running backwards recovers initial conditions exactly
- **Energy-conserving**: Not exactly, but bounded—energy oscillates rather than drifts

The modified Hamiltonian $\tilde{H} = H + O(\Delta t^2)$ is conserved exactly. This means energy errors remain bounded forever, enabling billion-year simulations.

## 2.3 Monte Carlo Initial Conditions

### The Plummer Model

Real star clusters don't start with stars arranged in grids. The Plummer model provides a physically motivated density distribution:

$$\rho(r) = \frac{3M}{4\pi a^3}\left(1 + \frac{r^2}{a^2}\right)^{-5/2}$$

where:
- $M$ is total cluster mass
- $a$ is the Plummer radius (scale length)
- $\rho(r)$ is density at radius $r$ from cluster center

This profile has several nice properties:
- Finite central density: $\rho(0) = 3M/(4\pi a^3)$
- Total mass converges: $\int_0^{\infty} 4\pi r^2 \rho(r) dr = M$
- Analytical potential: $\Phi(r) = -GM/\sqrt{r^2 + a^2}$

### Sampling from Arbitrary Distributions

To generate stars following the Plummer distribution, we use the inverse transform method. First, compute the cumulative mass function:

$$M(r) = \int_0^r 4\pi r'^2 \rho(r') dr' = M \frac{r^3}{(r^2 + a^2)^{3/2}}$$

Setting $u = M(r)/M$ and solving for $r$:

$$u = \frac{r^3}{(r^2 + a^2)^{3/2}}$$

This is a cubic equation in $r^2$. After algebra:

$$r = a\frac{u^{2/3}}{\sqrt{1 - u^{2/3}}}$$

The algorithm:
1. Generate uniform random $u \in [0,1]$
2. Transform to radius: $r = a u^{2/3}/\sqrt{1 - u^{2/3}}$
3. Generate random direction: uniform on sphere
4. Convert to Cartesian coordinates

For the Kroupa IMF with piecewise power laws:

$$M^{-1}(u) = \begin{cases}
M_{\min}\left(1 + u(C_1 - 1)\right)^{1/(1-\alpha_1)} & 0 < u < u_1 \\
M_1\left(1 + \frac{u - u_1}{u_2 - u_1}(C_2 - 1)\right)^{1/(1-\alpha_2)} & u_1 < u < u_2 \\
M_2\left(1 + \frac{u - u_2}{1 - u_2}(C_3 - 1)\right)^{1/(1-\alpha_3)} & u_2 < u < 1
\end{cases}$$

where the break points $u_1$, $u_2$ ensure continuity and proper normalization.

---

# Chapter 3: Light Through Dust - Monte Carlo Radiative Transfer

## 3.1 The Physics of Radiation Transport

### The Equation of Radiative Transfer

When light travels through a medium, its intensity $I_{\nu}$ (energy per unit area, time, frequency, and solid angle) changes according to:

$$\frac{dI_{\nu}}{ds} = -\kappa_{\nu}\rho I_{\nu} + j_{\nu}$$

where:
- $s$ is distance along the ray
- $\kappa_{\nu}$ is the opacity (cross-section per unit mass) at frequency $\nu$
- $\rho$ is the mass density of the medium
- $j_{\nu}$ is the emission coefficient (energy emitted per unit volume, time, frequency, and solid angle)

The first term represents absorption and scattering (removal of photons), the second represents emission (addition of photons).

### Optical Depth: The Natural Scale

Rather than distance, it's natural to measure path length in terms of optical depth:

$$d\tau_{\nu} = \kappa_{\nu}\rho ds$$

Integrating along a path:

$$\tau_{\nu} = \int_0^s \kappa_{\nu}\rho ds'$$

Optical depth is dimensionless and represents the "number of mean free paths." When $\tau = 1$, the probability of interaction is $1 - e^{-1} \approx 0.63$.

The formal solution to the radiative transfer equation is:

$$I_{\nu}(s) = I_{\nu}(0)e^{-\tau_{\nu}(s)} + \int_0^s j_{\nu}(s')e^{-(\tau_{\nu}(s) - \tau_{\nu}(s'))} ds'$$

The first term is the initial intensity attenuated by absorption. The second term is emission along the path, each point attenuated by absorption between that point and the observer.

### Scattering: Changing Direction

When a photon scatters off a dust grain, it changes direction according to the phase function $p(\cos\theta)$, where $\theta$ is the scattering angle. For spherical particles with radius $a$ comparable to wavelength $\lambda$, Mie theory provides exact solutions. But these are computationally expensive, so we often use the Henyey-Greenstein approximation:

$$p(\cos\theta) = \frac{1 - g^2}{4\pi(1 + g^2 - 2g\cos\theta)^{3/2}}$$

where $g = \langle\cos\theta\rangle$ is the asymmetry parameter:
- $g = 0$: isotropic scattering (equal probability in all directions)
- $g > 0$: forward scattering (prefers small angles)
- $g < 0$: backward scattering (rare for dust)

Interstellar dust typically has $g \approx 0.6$, meaning forward scattering dominates.

## 3.2 Monte Carlo Radiative Transfer

### The Photon Packet Approach

Instead of solving the integro-differential radiative transfer equation directly, we simulate individual photon packets. Each packet represents many photons with total energy:

$$E_{\text{packet}} = \frac{L\Delta t}{N_{\text{packets}}}$$

where $L$ is source luminosity, $\Delta t$ is time interval, and $N_{\text{packets}}$ is the number of packets we simulate.

### The Random Walk Algorithm

For each photon packet:

1. **Emission**: Start at source position with initial direction (isotropic for stars)

2. **Path length sampling**: Distance to next interaction drawn from exponential distribution:
   $$P(s) = e^{-\tau} \implies s = -\ln(1 - \xi)/\kappa\rho$$
   where $\xi$ is a uniform random number in [0,1]

3. **Interaction type**: Probability of absorption vs scattering:
   $$P_{\text{abs}} = \frac{\kappa_{\text{abs}}}{\kappa_{\text{abs}} + \kappa_{\text{sca}}} = \frac{\kappa_{\text{abs}}}{\kappa_{\text{ext}}}$$

4. **Scattering**: If scattered, sample new direction from phase function using rejection method or inverse transform

5. **Absorption/Re-emission**: If absorbed, packet can be re-emitted (for thermal equilibrium) at longer wavelength according to dust temperature

6. **Escape**: Continue until packet escapes the computational domain

### The Mathematics of Convergence

Monte Carlo methods converge as $1/\sqrt{N}$ regardless of dimensionality. The measured intensity has uncertainty:

$$\sigma_I = \frac{\sigma}{\sqrt{N}}$$

where $\sigma$ is the standard deviation of individual packet contributions. This is the Central Limit Theorem in action: the mean of N random variables has standard error decreasing as $1/\sqrt{N}$.

For 1% accuracy, we need $N \sim 10^4$ packets. For 0.1% accuracy, $N \sim 10^6$. This scaling is independent of the problem's dimensionality—a huge advantage over grid-based methods.

## 3.3 Observable Quantities

### Extinction and Reddening

The observed magnitude of a star is:

$$m = -2.5\log_{10}(F) + \text{constant}$$

where $F$ is the observed flux. Dust extinction increases the magnitude (makes stars appear fainter):

$$m_{\text{observed}} = m_{\text{intrinsic}} + A$$

where the extinction $A$ relates to optical depth:

$$A = 2.5\log_{10}(e^{\tau}) = 1.086\tau$$

Different wavelengths experience different extinction. The extinction curve approximately follows:

$$A_{\lambda} \propto \lambda^{-\beta}$$

with $\beta \approx 1.7$ for typical interstellar dust. This means blue light is extinguished more than red light, causing "reddening."

The color excess quantifies this:

$$E(B-V) = (B - V)_{\text{observed}} - (B - V)_{\text{intrinsic}}$$

where $B$ and $V$ are magnitudes in blue and visual bands. The ratio of total to selective extinction:

$$R_V = \frac{A_V}{E(B-V)} \approx 3.1$$

for typical interstellar dust, though this varies with environment.

---

# Chapter 4: From Light to Knowledge - MCMC and Bayesian Inference

## 4.1 The Inverse Problem in Astronomy

### Forward vs Inverse Modeling

So far, we've done forward modeling: given physical parameters, predict observations. But astronomy usually works backwards: given observations, infer physical parameters. This is an inverse problem, and it's fundamentally harder because:

1. **Non-uniqueness**: Multiple parameter sets might produce identical observations
2. **Noise**: Observations have uncertainties
3. **Incompleteness**: We never observe everything

Consider measuring the expansion rate of the universe using Type Ia supernovae. We observe apparent magnitude $m$ and redshift $z$. The distance modulus is:

$$\mu = m - M = 5\log_{10}\left(\frac{D_L}{10\text{ pc}}\right)$$

where $M$ is absolute magnitude and $D_L$ is luminosity distance. In a flat universe with matter and dark energy:

$$D_L(z) = \frac{c(1+z)}{H_0}\int_0^z \frac{dz'}{\sqrt{\Omega_m(1+z')^3 + \Omega_{\Lambda}}}$$

where:
- $c = 3 \times 10^8$ m/s is the speed of light
- $H_0$ is the Hubble constant (current expansion rate)
- $\Omega_m$ is the matter density parameter
- $\Omega_{\Lambda}$ is the dark energy density parameter
- The factor $(1+z)$ accounts for cosmological redshift

Given observations $\{m_i, z_i\}$, how do we determine $\Omega_m$ and $H_0$?

## 4.2 The Bayesian Framework

### Bayes' Theorem: The Foundation

Bayesian inference treats parameters as random variables with probability distributions. Bayes' theorem relates what we want to know (posterior) to what we can calculate (likelihood times prior):

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

Let's understand each term:

**Posterior** $P(\theta|D)$: Probability of parameters $\theta$ given data $D$. This is what we want—our knowledge after seeing data.

**Likelihood** $P(D|\theta)$: Probability of observing data $D$ if parameters are $\theta$. For Gaussian errors:

$$P(D|\theta) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma_i^2}}\exp\left(-\frac{(y_i - f(x_i;\theta))^2}{2\sigma_i^2}\right)$$

where $f(x;\theta)$ is our model prediction.

**Prior** $P(\theta)$: Our knowledge before seeing data. This encodes physical constraints (e.g., $0 < \Omega_m < 1$) and previous measurements.

**Evidence** $P(D) = \int P(D|\theta)P(\theta)d\theta$: Normalization constant ensuring $\int P(\theta|D)d\theta = 1$.

### Connection to Frequentist Statistics

Maximum likelihood estimation finds parameters that maximize $P(D|\theta)$. This is equivalent to minimizing negative log-likelihood:

$$\chi^2 = -2\ln P(D|\theta) = \sum_{i=1}^N \frac{(y_i - f(x_i;\theta))^2}{\sigma_i^2} + \text{constant}$$

This is the familiar chi-squared statistic! With flat priors, the maximum posterior estimate equals the maximum likelihood estimate.

### Linear Regression as Bayesian Inference

Consider fitting a line $y = ax + b$ to data. The likelihood for Gaussian errors is:

$$P(D|a,b) \propto \exp\left(-\frac{1}{2}\sum_i \frac{(y_i - ax_i - b)^2}{\sigma_i^2}\right)$$

With flat priors, the posterior is proportional to the likelihood. The maximum occurs where derivatives vanish:

$$\frac{\partial \ln P}{\partial a} = 0 \implies \sum_i \frac{x_i(y_i - ax_i - b)}{\sigma_i^2} = 0$$

$$\frac{\partial \ln P}{\partial b} = 0 \implies \sum_i \frac{y_i - ax_i - b}{\sigma_i^2} = 0$$

These are the normal equations! Their solution:

$$a = \frac{\sum w_i x_i y_i - (\sum w_i x_i)(\sum w_i y_i)/\sum w_i}{\sum w_i x_i^2 - (\sum w_i x_i)^2/\sum w_i}$$

where $w_i = 1/\sigma_i^2$ are weights. The parameter covariance matrix:

$$\text{Cov}(a,b) = \begin{pmatrix}
\sum w_i x_i^2 & \sum w_i x_i \\
\sum w_i x_i & \sum w_i
\end{pmatrix}^{-1}$$

gives parameter uncertainties. Regression always had uncertainty—Bayesian framework makes it explicit!

## 4.3 Markov Chain Monte Carlo

### The Fundamental Problem

For complex models, we can't compute the posterior analytically. The evidence integral:

$$P(D) = \int P(D|\theta)P(\theta)d\theta$$

is intractable in high dimensions. MCMC solves this by generating samples from the posterior without computing the normalization.

### The Metropolis-Hastings Algorithm

The key insight: construct a Markov chain whose stationary distribution is the posterior. The algorithm:

1. **Initialize**: Start at some $\theta_0$

2. **Propose**: Generate candidate $\theta'$ from proposal distribution $q(\theta'|\theta)$

3. **Accept/Reject**: Accept with probability:
   $$\alpha = \min\left(1, \frac{P(\theta'|D)q(\theta|\theta')}{P(\theta|D)q(\theta'|\theta)}\right)$$

4. **Update**: If accepted, $\theta_{n+1} = \theta'$; otherwise $\theta_{n+1} = \theta_n$

5. **Repeat**: Continue until convergence

The genius is that we only need the ratio of posteriors:

$$\frac{P(\theta'|D)}{P(\theta|D)} = \frac{P(D|\theta')P(\theta')}{P(D|\theta)P(\theta)}$$

The evidence $P(D)$ cancels! We never need to compute the intractable normalization.

### Detailed Balance and Convergence

The algorithm satisfies detailed balance:

$$P(\theta)T(\theta \to \theta') = P(\theta')T(\theta' \to \theta)$$

where $T$ is the transition probability. This ensures the posterior is the stationary distribution.

Convergence is monitored via:
- **Trace plots**: Parameter values vs iteration
- **Autocorrelation**: Correlation between samples at different lags
- **Gelman-Rubin statistic**: Compares variance within and between chains

### Hamiltonian Monte Carlo: Physics Meets Statistics

HMC introduces auxiliary momentum variables $p$ and defines a Hamiltonian:

$$H(\theta, p) = -\ln P(\theta|D) + \frac{1}{2}p^TM^{-1}p$$

The first term is potential energy (negative log posterior), the second is kinetic energy. Hamilton's equations:

$$\frac{d\theta}{dt} = \frac{\partial H}{\partial p} = M^{-1}p$$

$$\frac{dp}{dt} = -\frac{\partial H}{\partial \theta} = \nabla \ln P(\theta|D)$$

We integrate these using... the Leapfrog algorithm from N-body dynamics!

$$p_{1/2} = p_0 + \frac{\epsilon}{2}\nabla \ln P(\theta_0|D)$$
$$\theta_1 = \theta_0 + \epsilon M^{-1}p_{1/2}$$
$$p_1 = p_{1/2} + \frac{\epsilon}{2}\nabla \ln P(\theta_1|D)$$

The same symplectic properties that conserved energy in stellar orbits now ensure efficient parameter space exploration! HMC can take large steps while maintaining high acceptance rates, dramatically improving efficiency over random-walk Metropolis.

---

# Chapter 5: From Parameters to Functions - Gaussian Processes

## 5.1 The Conceptual Foundation

### The Limitation of Parametric Models

Everything we've done so far involves learning a finite set of parameters. Linear regression learns slope and intercept. MCMC learns cosmological parameters. But what if we want to learn an entire function $f: \mathcal{X} \to \mathbb{R}$ without assuming a specific functional form?

Consider the problem of emulating our N-body simulations. The simulation is effectively a function:

$$f: \text{(mass, concentration, virial ratio)} \to \text{(relaxation time)}$$

We've evaluated this function at some points (by running simulations), and we want to predict it everywhere else. We need a probability distribution over functions.

### From Finite to Infinite Dimensions

Start with a finite-dimensional Gaussian distribution. For a vector $\mathbf{x} \in \mathbb{R}^n$:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)\right)$$

The mean vector $\mu_i = \mathbb{E}[x_i]$ specifies the average value of each component. The covariance matrix $\Sigma_{ij} = \text{Cov}(x_i, x_j) = \mathbb{E}[(x_i - \mu_i)(x_j - \mu_j)]$ specifies relationships between components.

Now consider a function $f(x)$. At any finite set of points $\{x_1, ..., x_n\}$, the function values $\mathbf{f} = [f(x_1), ..., f(x_n)]^T$ follow a multivariate Gaussian:

$$p(\mathbf{f}) = \mathcal{N}(\mathbf{m}, \mathbf{K})$$

where $m_i = m(x_i)$ and $K_{ij} = k(x_i, x_j)$. As we consider more points, we approach a distribution over the entire function.

### The Gaussian Process Definition

A Gaussian Process is a collection of random variables, any finite subset of which follows a multivariate Gaussian distribution. It's completely specified by:

- **Mean function**: $m(x) = \mathbb{E}[f(x)]$
- **Covariance function (kernel)**: $k(x, x') = \text{Cov}(f(x), f(x'))$

We write: $f(x) \sim \mathcal{GP}(m(x), k(x, x'))$

## 5.2 The Kernel: Encoding Prior Knowledge

### What Makes a Valid Kernel?

Not every function $k(x, x')$ can be a kernel. Valid kernels must produce positive semi-definite covariance matrices. Mathematically, for any set of points $\{x_i\}$ and any vector $\mathbf{a}$:

$$\sum_{i,j} a_i a_j k(x_i, x_j) \geq 0$$

This ensures the resulting distribution is a valid probability distribution (no negative probabilities).

### Common Kernels and Their Interpretations

**Radial Basis Function (RBF) / Squared Exponential**:

$$k(x, x') = \sigma_f^2 \exp\left(-\frac{(x - x')^2}{2\ell^2}\right)$$

- $\sigma_f^2$ is the signal variance: How much does the function vary overall?
- $\ell$ is the length scale: Over what distance does the function change significantly?
- Infinitely differentiable → very smooth functions
- As $|x - x'| \to 0$: $k \to \sigma_f^2$ (perfect correlation)
- As $|x - x'| \to \infty$: $k \to 0$ (no correlation)

**Matérn Kernel**:

$$k_{\nu}(x, x') = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}|x - x'|}{\ell}\right)^{\nu}K_{\nu}\left(\frac{\sqrt{2\nu}|x - x'|}{\ell}\right)$$

where $K_{\nu}$ is the modified Bessel function. Special cases:
- $\nu = 1/2$: Exponential kernel, non-differentiable (rough) functions
- $\nu = 3/2$: Once differentiable
- $\nu = 5/2$: Twice differentiable
- $\nu \to \infty$: Recovers RBF kernel

**Periodic Kernel**:

$$k(x, x') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi|x - x'|/p)}{\ell^2}\right)$$

- $p$ is the period
- $\ell$ controls bandwidth (how quickly correlation decays within period)
- Perfect for modeling seasonal effects or periodic phenomena

### Kernel Composition

Kernels can be combined to build complex priors:

**Addition**: $k_1 + k_2$ represents independent processes added together
**Multiplication**: $k_1 \times k_2$ represents interaction/modulation
**Example**: Quasi-periodic kernel for variable stars:
$$k(x, x') = k_{\text{periodic}}(x, x') \times k_{\text{RBF}}(x, x')$$

Models periodic signal with slowly varying amplitude.

## 5.3 Gaussian Process Regression

### The Setup

We have training data:
- Inputs: $\mathbf{X} = \{x_1, ..., x_n\}$
- Outputs: $\mathbf{y} = \{y_1, ..., y_n\}$
- Observation noise: $y_i = f(x_i) + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$

We want to predict $f_*$ at test point $x_*$.

### The Joint Distribution

The joint distribution of training outputs and test output is:

$$\begin{bmatrix} \mathbf{y} \\ f_* \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \mathbf{m} \\ m_* \end{bmatrix}, \begin{bmatrix} \mathbf{K} + \sigma_n^2\mathbf{I} & \mathbf{k}_* \\ \mathbf{k}_*^T & k_{**} \end{bmatrix}\right)$$

where:
- $\mathbf{K}$ is the $n \times n$ matrix with $K_{ij} = k(x_i, x_j)$
- $\mathbf{k}_* = [k(x_1, x_*), ..., k(x_n, x_*)]^T$
- $k_{**} = k(x_*, x_*)$
- $\sigma_n^2\mathbf{I}$ accounts for observation noise

### Conditioning: The Key Result

To predict at $x_*$, we condition on observed data. For multivariate Gaussians, conditioning gives another Gaussian:

$$p(f_*|\mathbf{y}, \mathbf{X}, x_*) = \mathcal{N}(\mu_*, \sigma_*^2)$$

with:

$$\mu_* = m_* + \mathbf{k}_*^T(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}(\mathbf{y} - \mathbf{m})$$

$$\sigma_*^2 = k_{**} - \mathbf{k}_*^T(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\mathbf{k}_*$$

### Understanding the Predictive Equations

**Mean Prediction**:
$$\mu_* = m_* + \sum_{i=1}^n \alpha_i k(x_i, x_*)$$

where $\boldsymbol{\alpha} = (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}(\mathbf{y} - \mathbf{m})$.

- Start with prior mean $m_*$
- Add weighted sum of kernel functions centered at training points
- Weights $\alpha_i$ depend on how well training points explain the data

**Variance Prediction**:
$$\sigma_*^2 = k_{**} - \mathbf{k}_*^T(\mathbf{K} + \sigma_n^2\mathbf{I})^{-1}\mathbf{k}_*$$

- Start with prior variance $k_{**}$
- Subtract information gained from training data
- More nearby training points → lower uncertainty

### The Marginal Likelihood

To learn hyperparameters $\theta$ (like length scales), maximize the marginal likelihood:

$$p(\mathbf{y}|\mathbf{X}, \theta) = \int p(\mathbf{y}|f, \mathbf{X})p(f|\theta)df$$

This integral is tractable for GPs:

$$\log p(\mathbf{y}|\mathbf{X}, \theta) = -\frac{1}{2}\mathbf{y}^T\mathbf{K}_y^{-1}\mathbf{y} - \frac{1}{2}\log|\mathbf{K}_y| - \frac{n}{2}\log(2\pi)$$

where $\mathbf{K}_y = \mathbf{K} + \sigma_n^2\mathbf{I}$.

The three terms have interpretations:
1. **Data fit**: $-\frac{1}{2}\mathbf{y}^T\mathbf{K}_y^{-1}\mathbf{y}$ measures how well the model explains data
2. **Complexity penalty**: $-\frac{1}{2}\log|\mathbf{K}_y|$ penalizes complex models
3. **Normalization**: $-\frac{n}{2}\log(2\pi)$ ensures proper probability

This automatically implements Occam's razor—balancing fit against complexity.

## 5.4 Application to N-body Emulation

### The Emulation Problem

Our N-body simulation is an expensive function:

$$f: \underbrace{[N, M, c, Q, S]}_{\text{initial conditions}} \to \underbrace{[T_{\text{relax}}, T_{\text{core}}, f_{\text{escape}}]}_{\text{outcomes}}$$

where:
- $N$ = number of stars
- $M$ = total mass
- $c$ = concentration parameter
- $Q$ = virial ratio
- $S$ = mass segregation

Each evaluation (simulation) takes ~30 minutes. We want predictions in milliseconds.

### Multi-dimensional Kernels

For vector inputs $\mathbf{x} = [x_1, ..., x_D]$, we use:

**Isotropic RBF**: 
$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{|\mathbf{x} - \mathbf{x}'|^2}{2\ell^2}\right)$$

All dimensions share the same length scale.

**Automatic Relevance Determination (ARD)**:
$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\sum_{d=1}^D \frac{(x_d - x'_d)^2}{2\ell_d^2}\right)$$

Each dimension has its own length scale. Large $\ell_d$ means dimension $d$ is less relevant.

### The Complete Emulation Pipeline

1. **Design of Experiments**: Choose training points using Latin Hypercube Sampling for good coverage

2. **Run Simulations**: Generate training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$

3. **Standardize**: Transform inputs and outputs to zero mean, unit variance

4. **Optimize Hyperparameters**: Maximize marginal likelihood to find optimal $[\sigma_f, \ell_1, ..., \ell_D, \sigma_n]$

5. **Validate**: Test on held-out simulations

6. **Deploy**: Replace expensive simulation with instant GP prediction

### Active Learning

The GP's uncertainty guides where to run next simulation:

**Maximum Uncertainty**:
$$\mathbf{x}_{\text{next}} = \arg\max_{\mathbf{x}} \sigma^2(\mathbf{x})$$

Explores regions we know least about.

**Expected Improvement**:
$$\text{EI}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f_{\text{best}}, 0)]$$

Balances exploration and exploitation for optimization.

## 5.5 Deep Connections

### Connection to Neural Networks

Consider a single-layer neural network:

$$f(x) = \sum_{i=1}^H w_i \phi(v_i^T x + b_i)$$

where $\phi$ is activation function, $w_i$ are output weights, $v_i$ are input weights, $b_i$ are biases.

If we place priors on weights and take $H \to \infty$, the Central Limit Theorem gives us a Gaussian Process! The kernel depends on the activation function:

- ReLU activation → Matérn-1/2 kernel
- Error function activation → RBF kernel
- Sigmoid activation → Arc-sine kernel

This reveals that neural networks are approximating GPs with learned basis functions.

### Connection to Reproducing Kernel Hilbert Spaces

Every kernel defines an RKHS—a space of functions with inner product:

$$\langle f, g \rangle_{\mathcal{H}} = \sum_{i,j} \alpha_i \beta_j k(x_i, x_j)$$

The GP finds the minimum-norm interpolant in this space:

$$f^* = \arg\min_{f \in \mathcal{H}} ||f||_{\mathcal{H}}^2 \text{ subject to } f(x_i) = y_i$$

This connects to regularization theory and provides theoretical guarantees.

### Connection to Quantum Field Theory

The GP can be viewed through path integral formalism:

$$p(f) \propto \exp\left(-\frac{1}{2}\int\int f(x)k^{-1}(x, x')f(x')dxdx'\right)$$

This is analogous to the Euclidean path integral in quantum field theory, where $k^{-1}$ plays the role of the propagator. The GP marginalizes over all possible functions, weighted by their probability under the prior.

---

# Epilogue: The Unity of Computational Science

## The Journey We've Taken

We began with stars—points of light that we encoded as objects with mass, luminosity, and temperature. We discovered that numerical methods matter profoundly when we evolved these stars under gravity, watching our solar systems explode with Euler integration but remain stable with Leapfrog.

We learned that randomness isn't our enemy but our tool, using Monte Carlo methods to sample from complex distributions and transport photons through dusty media. We discovered that all measurements have uncertainty and that Bayesian inference provides the mathematical framework for reasoning about that uncertainty.

We saw that regression and MCMC are two faces of the same coin—both are parameter estimation under uncertainty. The Leapfrog algorithm that kept our planets stable returned to help us explore parameter spaces efficiently.

Finally, we transcended parameters entirely, learning to place probability distributions over functions themselves. The Gaussian Process emerged not as an arbitrary construction but as the natural culmination of our journey from deterministic computation through statistical inference to function learning.

## The Mathematical Threads

Throughout this journey, certain mathematical structures appeared repeatedly:

**Linear Algebra** underlies everything. The equation $\mathbf{Ax} = \mathbf{b}$ appears as:
- Force calculations in N-body systems
- Radiative transfer through discretized media
- Normal equations in regression
- Covariance conditioning in Gaussian Processes

**Differential Equations** describe change:
- Stellar evolution
- Orbital dynamics
- Hamiltonian flow in phase space and parameter space

**Probability Theory** evolved from:
- Simple sampling (initial mass functions)
- Through transport (photon random walks)
- To inference (parameter posteriors)
- To function spaces (Gaussian Processes)

**Optimization** connects:
- Energy minimization (symplectic integration)
- Likelihood maximization (parameter inference)
- Marginal likelihood (hyperparameter learning)

## The Physical Insights

The mathematics isn't abstract—it encodes physical understanding:

- Conservation laws (energy, momentum) constrain our numerical methods
- Smoothness of physical laws enables function emulation
- Measurement uncertainty necessitates probabilistic inference
- The universe's statistical nature (quantum mechanics, thermodynamics) makes Monte Carlo natural

## The Computational Wisdom

We've learned that computation isn't about computers but about transformation:
- Transforming intractable integrals into random sampling
- Transforming infinite-dimensional problems into finite approximations
- Transforming forward models into inverse inference
- Transforming expensive simulations into instant predictions

## The Path Forward

This foundation prepares you for the frontiers of computational science:

**Neural Networks**: You understand they're approximating GPs with learned basis functions

**Automatic Differentiation**: You've seen gradients everywhere—force calculations, MCMC proposals, GP optimization

**High-Performance Computing**: You understand why vectorization matters and how algorithms affect scaling

**Uncertainty Quantification**: You know that every prediction should come with uncertainty estimates

**Scientific Machine Learning**: You can combine physical models with data-driven approaches

## The Final Message

The true power isn't in any individual technique but in their synthesis. Each method we've learned addresses limitations of the previous ones, creating a complete toolkit for computational discovery.

You now possess something profound: the ability to translate physical understanding into computational capability. You can simulate the universe, observe it through dust, measure its fundamental parameters, and build machines that learn its patterns.

This isn't just technical skill—it's a way of thinking that will serve you whether you study galaxies or genomes, climate or consciousness. The same mathematics that describes stellar evolution helps us understand neural networks. The same algorithm that keeps planets in stable orbits helps us explore probability distributions.

In the end, computational astrophysics teaches us that the universe is comprehensible—not because it's simple, but because it follows mathematical patterns that we can discover, encode, and compute. And in that comprehension lies both the beauty of science and the power to push beyond current frontiers into the unknown.