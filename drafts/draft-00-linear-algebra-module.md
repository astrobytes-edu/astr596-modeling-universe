---
title: "Module 0: Linear Algebra Foundations"
subtitle: "Mathematical Foundations for Computational Astrophysics | ASTR 596"
exports:
  - format: pdf
---

## Quick Navigation Guide

### 🔍 Choose Your Learning Path

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🏃 **Fast Track**
Just starting the course? Read only sections marked with 🔴

- [Vectors Essentials](#part-2-vectors)
- [Matrix Fundamentals](#part-3-matrices)
- [Numerical Reality Check](#part-7-numerical-reality)
- [Quick Reference](#essential-scipy-reference)
:::

:::{grid-item-card} 🚶 **Standard Path**
Preparing for projects? Read 🔴 and 🟡 sections

- Everything in Fast Track, plus:
- [Eigenvalues & Eigenvectors](#part-4-eigenvalues)
- [Positive Definite Matrices](#part-5-positive-definite)
- [Implementation Examples](#numerical-implementation)
:::

:::{grid-item-card} 🧗 **Complete Path**
Want deep understanding? Read all sections including 🟢

- Complete module with:
- Historical contexts
- Mathematical proofs
- Advanced topics
- All worked examples
:::
::::

### 🎯 Navigation by Project Needs

:::{important} Quick Jump to What You Need by Project
:class: dropdown

**For Project 1 (Stellar Populations)**:

- [Section 2: Vectors](#part-2-vectors) - State representation
- [Section 3.1-3.2: Basic Matrices](#matrices-are-transformations) - Vectorization
- [Section 7.1: Floating Point](#floating-point-truth) - Numerical precision

**For Project 2 (N-body Dynamics)**:  

- [Section 2.3-2.4: Cross Products](#cross-product) - Angular momentum
- [Section 4: Eigenvalues](#part-4-eigenvalues) - Stability analysis
- [Section 3.3: Matrix Multiplication](#matrix-multiplication) - Transformations

**For Project 3 (Monte Carlo Radiative Transfer)**:

- [Section 3.2: Transformation Matrices](#building-intuition-concrete) - Scattering
- [Section 2.1: Vector Spaces](#vector-spaces) - Photon directions
- [Section 7.2: Condition Numbers](#condition-numbers) - Error propagation

**For Project 4 (MCMC)**:

- [Section 5.3: Covariance Matrices](#covariance-matrices) - Proposal distributions
- [Section 4: Eigenvalues](#part-4-eigenvalues) - Convergence rates
- [Section 5.2: Positive Definiteness](#positive-definiteness) - Valid covariances

**For Project 5 (Gaussian Processes)**:

- [Section 5: Positive Definite Matrices](#part-5-positive-definite) - Kernel matrices
- [Section 5.5: Cholesky Decomposition](#cholesky-decomposition) - GP implementation
- [Section 6.2: Schur Complement](#block-matrices) - GP updates

**For Final Project (Neural Networks)**:

- [Section 3: Matrix Operations](#part-3-matrices) - Layer transformations
- [Section 6.3: Jacobian Matrix](#jacobian-matrix) - Backpropagation
- [Section 4: Eigenvalues](#part-4-eigenvalues) - Optimization landscape
:::

### 📚 Quick Topic Index

:::{important} Quick Jump to Important Topics
:class: dropdown

| Topic | Section | Priority | First Used |
|-------|---------|----------|------------|
| Vectors & Dot Products | [2.1-2.3](#part-2-vectors) | 🔴 Essential | Project 1 |
| Matrix Multiplication | [3.1-3.3](#part-3-matrices) | 🔴 Essential | Project 1 |
| Eigenvalues/Eigenvectors | [4.1-4.3](#part-4-eigenvalues) | 🟡 Important | Project 2 |
| Covariance Matrices | [5.3](#covariance-matrices) | 🟡 Important | Project 4 |
| Positive Definiteness | [5.1-5.2](#part-5-positive-definite) | 🟡 Important | Project 5 |
| Cholesky Decomposition | [5.5](#cholesky-decomposition) | 🟡 Important | Project 5 |
| SVD | [6.1](#svd-swiss-army) | 🟢 Enrichment | Advanced |
| Matrix Exponentials | [6.4](#matrix-exponentials) | 🟢 Enrichment | Advanced |

:::

---

## Learning Objectives

:::{hint} 📅 When to Read This Module
:class: dropdown

- [ ] **Initial Reading**: Before starting Project 1, read sections marked 🔴 (2-3 hours)
- [ ] **Deep Dive**: Return to 🟡 sections as you start each new project
- [ ] **Reference**: Use throughout the course as questions arise
- [ ] **Mastery**: Read 🟢 sections when you want deeper understanding

**Tip:** Reread important sections to solidify understanding — linear algebra intuition builds through repetition and practical application. Concepts that seem abstract now will click after you've coded them, debugged them, and seen them across different projects.
:::

By the end of this module, you will be able to:

- [ ] **Translate** physical problems into vector and matrix representations 🔴
- [ ] **Apply** conservation laws to identify invariant mathematical structures in dynamical systems 🟡
- [ ] **Calculate** eigenvalues and eigenvectors for 2×2 and 3×3 matrices by hand 🟡
- [ ] **Determine** whether a matrix is positive definite using three different methods 🟡
- [ ] **Choose** the appropriate matrix decomposition for solving different computational problems 🟢
- [ ] **Connect** linear algebra concepts to specific applications in all six course projects 🔴

---

## Prerequisites Review

:::{admonition} 📚 Mathematical Prerequisites Check
:class: note, dropdown

**Priority: 🔴 Essential** - Review this before starting

Before diving into the module, let's review essential mathematical concepts. If any of these are unfamiliar, spend extra time on the review sections provided.

**You should be comfortable with:**

- [ ] Basic matrix arithmetic (addition, multiplication)
- [ ] Solving quadratic equations
- [ ] Summation notation $(\sum)$
- [ ] Basic trigonometry (sin, cos, radians)
- [ ] Complex numbers (for eigenvalues)

**Quick Review - Matrix Multiplication by Hand:**
To multiply two matrices, use the row-column rule:
$$C_{ij} = \sum_k A_{ik}B_{kj}$$

*Example:* Multiply
$$A = \begin{pmatrix} 2 & 1 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$$

For element $C_{11}$: Take row 1 of A, column 1 of B:
$$C_{11} = (2)(5) + (1)(7) = 10 + 7 = 17$$

For element $C_{12}$: Take row 1 of A, column 2 of B:
$$C_{12} = (2)(6) + (1)(8) = 12 + 8 = 20$$

Continue for all elements:
$$C = \begin{pmatrix} 17 & 20 \\ 43 & 50 \end{pmatrix}$$

**Complex Numbers Review:**

A complex number $z = a + bi$ where $i = \sqrt{-1}$.

- Magnitude: $|z| = \sqrt{a^2 + b^2}$
- Complex conjugate: $\bar{z} = a - bi$
- For eigenvalues, complex pairs indicate oscillatory behavior
:::

---

## Module Overview

:::{admonition} 🎯 Core Message
:class: important

**Priority: 🔴 Essential**:

**Linear algebra** is the mathematical foundation that makes computational astrophysics possible. Without it, simulating even three gravitationally interacting stars would be computationally intractable. With it, we can simulate millions.
:::

:::{note} 🔭 Where We're Going

This module takes you on a journey from the simplest mathematical objects (scalars and vectors) to the most sophisticated (matrix decompositions). Along the way, you'll discover that the same mathematical structures appear everywhere in physics - from describing stellar positions to encoding the stability of the universe itself. By the end, you'll see linear algebra not as abstract mathematics, but as the natural language for describing physical reality.

:::

```{margin}
**linear algebra**
The branch of mathematics concerning vector spaces and linear transformations between them
```

**Linear algebra** is the mathematical foundation that makes computational astrophysics possible. Without it, simulating even three gravitationally interacting stars would be computationally intractable. With it, we can simulate millions. But linear algebra is more than just a computational tool – it reveals deep structures in physical systems that would otherwise remain hidden.

Consider Omega Centauri, the most massive **globular cluster** orbiting our galaxy. It contains approximately 10 million stars, all gravitationally bound, orbiting their common center of mass for the past 12 billion years. To describe this system's state at any moment requires 60 million numbers (3 position and 3 velocity components for each star). Yet this apparent complexity hides elegant simplicity: the cluster's overall shape is described by just three numbers (the **eigenvalues** of its moment of inertia tensor), its stability is determined by eigenvalues of the gravitational potential, and its statistical properties are encoded in **covariance matrices**.

```{figure} [](./images/OmegaCen.jpg)
:label: omegaCen
:alt: Omega Centauri Globular Cluster
:align: center

Zoom-in HST image of the Omega Centauri Globular Cluster located at a distance of $\sim$4.8 kpc.<br> *Credit: NASA* 
```

```{margin}
**globular cluster**
A spherical collection of 10⁴ to 10⁶ stars bound by gravity, orbiting as satellites of galaxies
```

This module builds your understanding from first principles, following the same progression your projects will take. We start with **scalars and vectors** describing individual stellar properties and positions (Project 1: Stellar Populations), build to **matrices** transforming entire systems (Project 2: N-body Dynamics), explore how these structures handle uncertainty (Projects 3-4: Monte Carlo and MCMC), and preview how they extend to high-dimensional learning (Projects 5-6: Gaussian Processes and Neural Networks).

:::{admonition} 🎯 Why This Module Matters
:class: important

Every computational method you'll implement in this course ultimately reduces to linear algebra:

- **Project 1**: Your `StellarPopulation` class uses matrix operations for vectorized calculations
- **Project 2**: N-body integration operates in 6N-dimensional phase space
- **Project 3**: Monte Carlo photon transport uses transformation matrices for scattering
- **Project 4**: MCMC efficiency depends on proposal covariance matrices
- **Project 5**: Gaussian Processes require positive definite kernel matrices
- **Final Project**: Neural networks are compositions of matrix transformations

Master these foundations, and you master computational astrophysics.
:::

---

## Part 1: The Opening - A Globular Cluster as a Mathematical Universe

**Priority: 🟡 Important** - Provides physical context for abstract concepts

:::{hint} 📅 When You'll Use This
:class: dropdown

- **First appears in**: Project 2 (N-body dynamics)
- **Critical for**: Understanding phase space throughout the course
- **Returns in**: Every project dealing with multi-particle systems
:::

### 1.1 The Physical System That Motivates Everything

Let's continue with Omega Centauri, the most massive **globular cluster** orbiting our galaxy. It contains approximately 10 million stars, all gravitationally bound, orbiting their common center of mass for the past 12 billion years. At its core, stellar densities reach thousands of stars per cubic parsec — if Earth orbited a star there, our night sky would blaze with thousands of stars brighter than Venus.

```{margin}
**phase space**
The space of all possible states of a system; for N particles in 3D, has 6N dimensions (3 position + 3 velocity per particle)
```

To simulate this cluster, we need to track for each star:

- Position: $\vec{r}_i = (x_i, y_i, z_i)$ measured from the cluster center
- Velocity: $\vec{v}_i = (v_{x,i}, v_{y,i}, v_{z,i})$ relative to the cluster's motion

That's 6 million numbers evolving according to Newton's laws. The gravitational force on star $i$ is:

$$
\vec{F}_i = -G m_i \sum_{j \neq i} \frac{m_j (\vec{r}_i - \vec{r}_j)}{|\vec{r}_i - \vec{r}_j|^3}
$$

Each star feels forces from all others. For $N = 10⁷$ stars (Omega Centauri's estimated count), the number of unique pairwise interactions is $N(N-1)/2 ≈ 5×10¹³$. That's approximately 50 trillion unique force pair calculations per timestep! Without linear algebra, this would be impossible. With it, we can organize these calculations efficiently, identify conserved quantities, and understand the cluster's long-term evolution.

::::{admonition} 🤔 Check Your Understanding
:class: hint

A small globular cluster has 100,000 stars. How many unique force pair calculations are needed per timestep if we compute all pairwise interactions?

:::{admonition} Solution
:class: tip, dropdown

Each star feels forces from all others, so we need to count unique pairs. The binomial coefficient $\binom{n}{2}$ (read "n choose 2") counts the number of ways to choose 2 items from n total items:

$$
\binom{100,000}{2} = \frac{100,000!}{2!(100,000-2)!} = \frac{100,000 \times 99,999}{2 \times 1} \approx 5 \times 10^9
$$

The factorial notation cancels nicely: the (100,000-2)! in the denominator cancels all but the first two terms of 100,000! in the numerator, leaving us with the simple formula N(N-1)/2.

That's 5 billion unique force pair calculations per timestep! Note that while each star experiences 99,999 forces, we can use Newton's third law (action-reaction) to compute each pair only once. This is why we need efficient matrix operations and clever algorithms like tree codes.
:::
::::

### 1.2 Conservation Laws: The Hidden Structure

**Priority: 🟡 Important** - Fundamental for understanding physical constraints

```{margin}
**conservation law**
A physical quantity that remains constant over time due to an underlying symmetry
```

The cluster's apparent chaos hides profound order. Despite the complex individual stellar orbits, certain quantities never change:

Here are the updated equations with terms properly introduced at first appearance:

**Total Energy**:
$$E = \sum_i \frac{1}{2} m_i |\vec{v}_i|^2 - G \sum_{i<j} \frac{m_i m_j}{|\vec{r}_i - \vec{r}_j|}$$

where $m_i$ is the mass of star $i$, $\vec{v}_i$ is its velocity vector, $G$ is the gravitational constant, and $|\vec{r}_i - \vec{r}_j|$ is the distance between stars $i$ and $j$. The first term is the total kinetic energy and the second term is the total gravitational potential energy (summed over all unique pairs).

**Total Momentum**:
$$\vec{P} = \sum_i m_i \vec{v}_i$$

This vector sum represents the net momentum of the entire cluster.

**Total Angular Momentum**:
$$\vec{L} = \sum_i m_i (\vec{r}_i \times \vec{v}_i)$$

where $\vec{r}_i$ is the position vector of star $i$ from the cluster center and $\times$ denotes the cross product. This measures the total rotational momentum of the system.

These **conservation laws** aren't accidents — they arise from fundamental symmetries of space and time. Emmy Noether proved that every continuous symmetry implies a conservation law:

| Symmetry | Conservation Law | Mathematical Structure |
|----------|-----------------|----------------------|
| Time translation invariance | Energy | Quadratic forms |
| Space translation invariance | Momentum | Vector addition |
| Rotational invariance | Angular momentum | Cross products, orthogonal matrices |

These symmetries manifest mathematically as properties of vectors and matrices. Translation invariance means physics doesn't change when we add the same vector to all positions. Rotational invariance means physics is preserved under **orthogonal transformations**.

:::{note} 💡 Deeper Insight: Noether's Theorem in Action
**Priority: 🟢 Enrichment**

Noether's theorem tells us that symmetries and conservation laws are two sides of the same coin. In your Project 2, you'll discover that symplectic integrators (which preserve **phase space** volume) automatically conserve energy over long timescales. This isn't a coincidence – it's Noether's theorem at work! The symplectic structure (preserved determinant) encodes time-translation symmetry, which guarantees energy conservation.
:::

:::{tip} 🌟 The More You Know: Emmy Noether's Revolutionary Theorem
:class: dropdown

**Priority: 🟢 Enrichment**

Emmy Noether (1882-1935) fundamentally changed physics with her 1915 theorem connecting symmetries to conservation laws. Despite facing discrimination that prevented her from holding a paid position for years, she persisted in her work at the University of Göttingen.

When Einstein's general relativity seemed to violate energy conservation, it was Noether who resolved the paradox. She proved that in general relativity, energy conservation becomes local rather than global—a subtle but profound insight. Einstein wrote to Hilbert: "Yesterday I received from Miss Noether a very interesting paper on invariant forms. I'm impressed that such things can be understood in such a general way" (Letter from Einstein to Hilbert, May 24, 1918, as documented in *The Collected Papers of Albert Einstein*, Volume 8).

Her theorem now underlies all of modern physics. Every conservation law you use—energy, momentum, angular momentum, electric charge—exists because of an underlying symmetry. When you implement conservation checks in your N-body code, you're using Noether's profound insight that geometry determines physics.

*Source: Osen, L. M. (1974). Women in Mathematics. MIT Press. pp. 141–152.*
:::

### 1.3 The Phase Space Perspective

**Priority: 🟡 Important** - Conceptual foundation for Projects 2-6

```{margin}
**parameter**
Most fundamentally: a number that controls the behavior of a mathematical function or system. In physics: quantities like mass, temperature, or coupling constants that define system properties but are not the variables being solved for
```

Here's the profound insight that changes everything: the million-star cluster isn't really moving through 3D space. It's tracing a **trajectory** through a 6-million-dimensional **phase space** where each axis represents one position or velocity component. The system's entire state is a single point in this vast space, and its time evolution is a trajectory through it.

```{margin}
**trajectory**
The path traced by a system's state through phase space as it evolves in time
```

This perspective reveals hidden simplicities:

- **Conservation laws constrain the trajectory to a lower-dimensional surface** - Just as a ball rolling in a bowl is confined to the bowl's 2D surface despite existing in 3D space, the cluster's evolution is restricted to a much smaller subspace where energy, momentum, and angular momentum remain constant. The 6-million dimensions collapse to far fewer effective dimensions.

- **Near equilibrium, motion decomposes into independent oscillation modes (eigenvectors)** - Like a drum that vibrates in distinct patterns (fundamental, overtones), the cluster has natural oscillation modes. Small disturbances trigger these modes independently, each oscillating at its own characteristic frequency - these are the eigenvectors and eigenvalues of the system.

- **Statistical properties emerge from the geometry of this high-dimensional space** - Velocity dispersion, dynamical stability, and evolution rates aren't separate properties but geometric features of how the system explores phase space. The volume it occupies relates to entropy, the shape of its trajectory relates to energy distribution, and the curvature of energy surfaces determines bound vs. unbound orbits. (See [Module 1a, Section 4](./01a-stat-mech-module.md) for the deep connection between stellar dynamics and statistical mechanics.)

```{margin}
**parameter space**
The space of all possible parameter values; each point represents a different configuration or model of the system
```

:::{admonition} 🚀 Real-World Application
:class: tip

The Gaia space telescope measures positions and velocities for over 1 billion stars in our galaxy. That's a 6-billion-dimensional phase space! Linear algebra makes it possible to:

- Identify moving groups and stellar streams (eigenanalysis)
- Reconstruct the galaxy's gravitational potential (matrix inversion)
- Find hidden structures like dissolved star clusters (principal components)

Without linear algebra, Gaia's data would be incomprehensible noise.
:::

:::{admonition} 📌 Key Distinction: Phase Space vs. Parameter Space
:class: important

**Phase space** contains dynamical variables (positions, velocities) that evolve with time according to equations of motion. A trajectory through phase space represents the system's time evolution.

**Parameter space** contains fixed quantities that define the system but don't evolve dynamically. When you do MCMC in Project 4, you'll explore parameter space to find optimal values of cosmological parameters (Ωm, h). When you train neural networks, you'll search weight space (a type of parameter space) for optimal network configurations.

The key difference:

- **Phase space:** Where systems evolve (dynamics)
- **Parameter space:** What defines systems (characteristics)
:::

:::{important} 📌 Key Takeaway

Phase space transforms complexity into geometry. Instead of tracking millions of individual trajectories, we can understand the system through the geometry of its phase space—its conserved surfaces, its stable manifolds, its eigenstructure. This geometric view is what makes the seemingly impossible (simulating millions of stars) actually tractable.

Remember the distinction: **phase space** (where systems evolve dynamically) vs. **parameter space** (the values that define the system). In Project 2, you'll watch star clusters evolve through phase space. In Project 4, you'll explore parameter space to find the Universe's composition. Both use the same mathematical machinery of linear algebra, but for fundamentally different purposes—one tracks dynamics, the other searches for optimal models.
:::

---

## Part 2: Vectors - The Atoms of Physical Description

**Priority: 🔴 Essential** - Foundation for everything that follows

:::{admonition} 📅 When You'll Use This
:class: dropdown, hint

- **First appears in**: Project 1 (representing stellar properties)
- **Used throughout**: Every single project
- **Most critical for**: Understanding state representation and transformations
:::

:::{admonition} 🔭 Where We're Going
:class: note

We begin with scalars — quantities with only magnitude — then advance to vectors that encode both magnitude and direction. We'll explore vectors' three complementary interpretations (physical, geometric, algebraic) and the operations that let us combine and manipulate them. By the end of this section, you'll see vectors not as lists of numbers but as the natural way to encode any quantity requiring directional information.
:::

### 2.1 From Scalars to Vectors: Building Deep Understanding {#part-2-vectors}

```{margin}
**scalar**
A quantity with magnitude only, represented by a single number
```

```{margin}
**vector**
A mathematical object with both magnitude and direction that transforms according to specific rules
```

The simplest quantities in physics are **scalars** — single numbers that have magnitude but no direction. Mass ($2×10^{33}$ g), temperature (5800 K), luminosity ($3.8×10^{33}$ erg/s), and density (1.4 g/cm³) are all scalars. They tell us "how much" but not "which way." For many physical quantities, however, magnitude alone is insufficient — we need direction too. This is where vectors enter.

A **vector** is simultaneously three complementary things, and masterful computational scientists fluidly shift between these perspectives:

- **Physical Perspective**: A vector represents any quantity with both magnitude and direction. The velocity of Earth orbiting the Sun is a vector—it has a speed (30 km/s) and a direction (tangent to the orbit). Forces, electric fields, angular momenta—all are vectors because they have this magnitude-direction character that scalars cannot express.

- **Geometric Perspective**: A vector is an arrow in space. Crucially, this arrow is **free**—it doesn't have a fixed starting point. The displacement "3 km north" is the same vector whether you start from your house or from campus. This freedom is what allows us to add forces acting at different points on a rigid body.

- **Algebraic Perspective**: A vector is an ordered list of numbers—its **components** in some coordinate system. Earth's velocity might be written as:

$$\vec{v} = \begin{pmatrix} -15.2 \\ 25.8 \\ 0.0 \end{pmatrix} \text{ km/s (in ecliptic coordinates)}$$

But here's the crucial insight: these numbers are not the vector itself — they're just one representation. The vector exists independently of any coordinate system.

:::{warning} ⚠️ Common Misconception Alert

Students often think a vector IS its components. This is wrong! A vector is a geometric object that exists independently of coordinates. When you rotate your coordinate system, the components change but the vector itself doesn't. Think of it like describing a person's location: "3 blocks north, 2 blocks east" versus "3.6 blocks northeast" – different descriptions, same displacement.
:::

:::{note} 💡 Building Intuition: Vectors as Instructions

Think of a vector as an instruction for movement. The vector $\vec{v} = (3, 4, 0)$ says: "Go 3 units east, 4 units north, stay at the same height." No matter where you start, following this instruction produces the same displacement. This is why we can slide vectors around freely—they're instructions, not fixed objects.

This becomes powerful in physics: a force vector tells you which way to accelerate and how strongly. A velocity vector tells you which way you're moving and how fast. The vector nature captures both pieces of information in one mathematical object.
:::

:::{admonition} 🧠 Build Your Intuition: Vector Components
:class: note

Without calculating, predict what happens to vector components when you:

1. Rotate the coordinate system 90° clockwise: The x-component becomes the ___ component
2. Double all coordinate axes scales: Components are ___
3. Flip the x-axis direction: The x-component ___

Answers: (1) becomes the negative y-component, (2) halved, (3) changes sign

This shows components depend on your coordinate choice, but the vector itself doesn't change!
:::

### 2.2 Vector Spaces: The Mathematical Framework {#vector-spaces}

**Priority: 🟡 Important** - Theoretical foundation

```{margin}
**vector space**
A set equipped with addition and scalar multiplication operations satisfying eight specific axioms
```

A **vector space** is a set equipped with two operations (vector addition and scalar multiplication) that satisfy eight axioms. These axioms aren't arbitrary mathematical rules — each captures an essential physical property:

| Axiom | Mathematical Statement | Physical Meaning |
|-------|------------------------|------------------|
| **Closure under addition** | $\vec{u} + \vec{v} \in V$ | Adding velocities gives another velocity |
| **Commutativity** | $\vec{u} + \vec{v} = \vec{v} + \vec{u}$ | Order of displacements doesn't matter |
| **Associativity** | $(\vec{u} + \vec{v}) + \vec{w} = \vec{u} + (\vec{v} + \vec{w})$ | Grouping of forces doesn't affect total |
| **Zero vector** | $\exists \vec{0}: \vec{v} + \vec{0} = \vec{v}$ | State of no motion or displacement |
| **Additive inverse** | $\forall \vec{v}, \exists -\vec{v}: \vec{v} + (-\vec{v}) = \vec{0}$ | Every motion has an opposite |
| **Scalar closure** | $c\vec{v} \in V$ | Scaling a force gives another force |
| **Distributivity** | $c(\vec{u} + \vec{v}) = c\vec{u} + c\vec{v}$ | Scaling preserves addition |
| **Identity** | $1 \cdot \vec{v} = \vec{v}$ | Multiplying by 1 changes nothing |

**Linear independence** means vectors cannot be written as combinations of each other - they represent truly independent directions in space.

### From Vectors to Matrices: The Natural Progression

```{margin}
**linear independence**
Vectors that cannot be written as linear combinations of each other
```

You've mastered vectors - quantities with magnitude and direction. But what happens when you need to transform many vectors simultaneously? This is where **matrices** emerge naturally. A matrix isn't just a grid of numbers - it's a machine that transforms entire vector spaces. When you multiply a matrix by a vector, you're asking: "Where does this vector go under this transformation?" This perspective transforms matrices from abstract number arrays into concrete geometric operations.

### 2.3 The Dot Product: Projection, Angle, and Energy

**Priority: 🔴 Essential** - Used constantly throughout all projects

```{margin}
**dot product**
Scalar operation on vectors:<br> $\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\theta$
```

The **dot product** is perhaps the most important operation in physics because it answers a fundamental question: "How much does one vector contribute in the direction of another?"

**Definition and Formula**:
$$\vec{a} \cdot \vec{b} = a_x b_x + a_y b_y + a_z b_z = |\vec{a}||\vec{b}|\cos\theta$$

But why are these two formulas equal? Let's derive this connection from first principles.

:::{admonition} 📝 Mathematical Derivation: Dot Product Formula
:class: note, dropdown

**Priority: 🟢 Enrichment**

**Starting from the Law of Cosines:**

Consider vectors $\vec{a}$ and $\vec{b}$ with angle $\theta$ between them. The vector from tip of $\vec{a}$ to tip of $\vec{b}$ is $\vec{b} - \vec{a}$.

By the law of cosines:
$$|\vec{b} - \vec{a}|^2 = |\vec{a}|^2 + |\vec{b}|^2 - 2|\vec{a}||\vec{b}|\cos\theta$$

Expanding the left side:
$$|\vec{b} - \vec{a}|^2 = (b_x - a_x)^2 + (b_y - a_y)^2 + (b_z - a_z)^2$$
$$= |\vec{a}|^2 + |\vec{b}|^2 - 2(a_x b_x + a_y b_y + a_z b_z)$$

Comparing both expressions:
$$a_x b_x + a_y b_y + a_z b_z = |\vec{a}||\vec{b}|\cos\theta$$

This proves the component formula equals the geometric formula!
:::

**Physical Applications of the Dot Product**:

<!-- Suggested figure: Visual showing force vector, displacement vector, and the component of force along displacement -->

**Work Done by a Force**:
$$W = \vec{F} \cdot \vec{d} = |\vec{F}||\vec{d}|\cos\theta$$

Only the component of force along the displacement does work. A force perpendicular to motion (like the normal force on a sliding block) does zero work.

**Power from Solar Panels**:
$$P = \vec{S} \cdot \hat{n} = |\vec{S}|\cos\theta$$

where $\vec{S}$ is the solar flux vector and $\hat{n}$ is the panel's normal. Maximum power when facing the sun ($\theta = 0$), zero when edge-on ($\theta = 90°$).

::::{admonition} 🤔 Check Your Understanding
:class: hint

Two stars have velocities $\vec{v}_1 = (200, 100, 50)$ km/s and $\vec{v}_2 = (150, -100, 100)$ km/s.

1. Calculate their relative velocity
2. Are they approaching or receding? (Hint: use the dot product)

:::{admonition} Solution
:class: tip, dropdown

1. Relative velocity: $\vec{v}_{\text{rel}} = \vec{v}_1 - \vec{v}_2 = (50, 200, -50)$ km/s

2. To determine if approaching or receding, we need the dot product of relative velocity with separation vector. If we assume star 2 is at origin and star 1 at position $\vec{r}$:
   - If $\vec{v}_{\text{rel}} \cdot \vec{r} > 0$: receding
   - If $\vec{v}_{\text{rel}} \cdot \vec{r} < 0$: approaching
   - If $\vec{v}_{\text{rel}} \cdot \vec{r} = 0$: constant separation

Without knowing positions, we can't determine if they're approaching or receding – we need both position and velocity information!
:::
::::

### 2.4 The Cross Product: Creating Perpendicularity {#cross-product}

**Priority: 🔴 Essential** - Critical for angular momentum in Project 2

```{margin}
**cross product**
Vector operation producing a perpendicular vector: $\vec{a} \times \vec{b}$
```

The **cross product** creates a vector perpendicular to two input vectors:

$$\vec{a} \times \vec{b} = \begin{pmatrix} a_y b_z - a_z b_y \\ a_z b_x - a_x b_z \\ a_x b_y - a_y b_x \end{pmatrix}$$

**Geometric Intuition**: To find the direction of $\vec{a} \times \vec{b}$, use the right-hand rule: curl your right hand's fingers from the first vector $\vec{a}$ toward the second vector $\vec{b}$ through the smaller angle. Your thumb points in the direction of the cross product. This isn't arbitrary - it ensures consistency with our choice of right-handed coordinate systems.

**Key Properties**:

- **Magnitude**: $|\vec{a} \times \vec{b}| = |\vec{a}||\vec{b}|\sin\theta$ (area of parallelogram)
- **Direction**: Right-hand rule
- **Anti-commutative**: $\vec{a} \times \vec{b} = -\vec{b} \times \vec{a}$

**Physical Meaning - Angular Momentum**:

For a star at position $\vec{r}$ with momentum $\vec{p} = m\vec{v}$:
$$\vec{L} = \vec{r} \times \vec{p}$$

This vector is perpendicular to the orbital plane. Its magnitude equals twice the rate of area swept out—this is Kepler's second law!

:::{warning} ⚠️ Common Pitfall: Cross Product Order
:class: warning

Students often forget that cross product order matters!
$\vec{a} \times \vec{b} = -\vec{b} \times \vec{a}$ (they point in opposite directions)

This is crucial in physics:

- Torque: $\vec{\tau} = \vec{r} \times \vec{F}$ (position THEN force)
- Angular momentum: $\vec{L} = \vec{r} \times \vec{p}$ (position THEN momentum)

Getting the order wrong flips the direction - your planet orbits backward!
:::

### 2.5 Orthogonality: When Vectors Don't Talk

**Priority: 🟡 Important** - Essential for understanding basis vectors

```{margin}
**orthogonal**
Perpendicular; vectors with zero dot product
```

Two vectors are **orthogonal** (perpendicular) when their dot product is zero: $\vec{a} \cdot \vec{b} = 0$. This isn't just a geometric curiosity - orthogonality is why we can decompose complex systems into independent components.

**Why Orthogonal Bases Are Special**:

When basis vectors are orthogonal, calculations become dramatically simpler:

- **Projections are independent**: The component along one axis doesn't affect others
- **Pythagoras works**: $|\vec{v}|^2 = v_x^2 + v_y^2 + v_z^2$
- **Rotations preserve dot products**: Orthogonal transformations preserve angles and lengths

**Physical Example**: In quantum mechanics, energy eigenstates are orthogonal. This means measuring one energy level doesn't affect the probability of finding the system in another level. The orthogonality of spherical harmonics is why we can decompose the cosmic microwave background into independent multipole moments! 

**Astrophysical Application**: When tracking stellar proper motions, observations often come in non-orthogonal coordinate systems (right ascension, declination, radial velocity). Orthogonalization creates independent velocity components where each represents truly independent motion - essential for understanding 3D stellar kinematics and discovering moving groups in the solar neighborhood.

:::{admonition} 💻 Computational Reality Check: Vectorization
:class: tip

Orthogonal decomposition isn't just mathematically elegant—it's computationally essential. Let's see why with a stellar clustering analysis:

```python
# Finding nearest neighbors for N stars (e.g., identifying binary pairs)

# Slow nested loops (DON'T DO THIS):
nearest_distances = np.zeros(N)
nearest_indices = np.zeros(N, dtype=int)
for i in range(N):
    min_dist = np.inf
    min_idx = -1
    for j in range(N):
        if i != j:
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_dist:
                min_dist = dist
                min_idx = j
    nearest_distances[i] = min_dist
    nearest_indices[i] = min_idx
# Time for N=1000: ~1.8 seconds

# Fast vectorized version (DO THIS):
# Compute all pairwise distances at once using broadcasting
pos_i = positions[:, np.newaxis, :]  # Shape: (N, 1, 3)
pos_j = positions[np.newaxis, :, :]  # Shape: (1, N, 3)
separations = pos_i - pos_j           # Broadcasting! Shape: (N, N, 3)
distances = np.linalg.norm(separations, axis=2)  # Shape: (N, N)

# Mask out self-distances
np.fill_diagonal(distances, np.inf)

# Find nearest neighbor for each star
nearest_distances = np.min(distances, axis=1)
nearest_indices = np.argmin(distances, axis=1)
# Time for N=1000: ~0.015 seconds (120x faster!)

# The speedup comes from:
# 1. NumPy operations run in optimized C code
# 2. CPU vector instructions (SIMD) process multiple values simultaneously
# 3. Better cache usage from processing contiguous memory blocks
# 4. Broadcasting eliminates Python loop overhead

# Key insight for Project 2: This SAME broadcasting pattern works for ANY 
# pairwise calculation. Think about how you could adapt this structure 
# for computing interactions between particles...
```

**Understanding the broadcasting magic**:

- `positions[:, np.newaxis, :]` adds a new axis, creating shape (N, 1, 3)
- When NumPy sees shapes (N, 1, 3) and (1, N, 3), it "broadcasts" to (N, N, 3)
- This creates all N² pairwise differences in one operation!
- No Python loops = no Python overhead = massive speedup

**Why this matters**: In Project 2, you'll need similar pairwise calculations. Master this broadcasting pattern and your code will run 100x faster than nested loops!
:::

### 2.6 Basis Vectors and Coordinate Systems

**Priority: 🟡 Important** - Needed when changing coordinate systems

```{margin}
**basis**
A set of linearly independent vectors that span the entire vector space
```

A **basis** is a set of linearly independent vectors that span the entire space. The familiar Cartesian basis:

$$\hat{x} = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \quad \hat{y} = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \quad \hat{z} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$

These are **orthonormal**: mutually perpendicular and unit length.

Any vector can be decomposed:
$$\vec{v} = v_x \hat{x} + v_y \hat{y} + v_z \hat{z}$$

The coefficients are found by projection (via the dot product):
$$v_x = \vec{v} \cdot \hat{x}, \quad v_y = \vec{v} \cdot \hat{y}, \quad v_z = \vec{v} \cdot \hat{z}$$

:::{admonition} 💡 Practical Insight: Choosing the Right Coordinates
:class: note

**Priority: 🟡 Important for Project 2**<br>
Different problems benefit from different coordinate systems:

| System | Best Coordinates | Why |
|--------|-----------------|-----|
| Binary star orbit | Centered on barycenter | Simplifies equations of motion |
| Spiral galaxy | Cylindrical (R, φ, z) | Matches galaxy symmetry |
| Globular cluster | Spherical (r, θ, φ) | Exploits spherical symmetry |
| Stellar stream | Along stream trajectory | Reveals stream structure |

In Project 2, you'll transform between coordinate systems to simplify your N-body calculations.
:::

:::{note} 📝 Worked Example: Gram-Schmidt Orthogonalization
:class: dropdown

**Priority: 🟢 Enrichment** - Helps understand QR decomposition

**Physical Motivation**: You're observing a galaxy and measure velocities in a skewed coordinate system. Gram-Schmidt creates an orthonormal basis where velocity components are independent - essential for understanding the true motion patterns!

The **Gram-Schmidt process** creates an orthonormal basis from any linearly independent set. This is essential for understanding QR decomposition.

**Given vectors**: $\vec{v}_1 = (1, 1, 0)$, $\vec{v}_2 = (1, 0, 1)$, $\vec{v}_3 = (0, 1, 1)$

**Step 1**: Normalize first vector
$$\vec{u}_1 = \frac{\vec{v}_1}{|\vec{v}_1|} = \frac{(1, 1, 0)}{\sqrt{2}} = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0\right)$$

**Step 2**: Remove $\vec{u}_1$ component from $\vec{v}_2$
$$\vec{w}_2 = \vec{v}_2 - (\vec{v}_2 \cdot \vec{u}_1)\vec{u}_1$$
$$= (1, 0, 1) - \frac{1}{\sqrt{2}} \cdot \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0\right)$$
$$= (1, 0, 1) - \left(\frac{1}{2}, \frac{1}{2}, 0\right) = \left(\frac{1}{2}, -\frac{1}{2}, 1\right)$$

Normalize: $\vec{u}_2 = \vec{w}_2/|\vec{w}_2| = \left(\frac{1}{\sqrt{6}}, -\frac{1}{\sqrt{6}}, \frac{2}{\sqrt{6}}\right)$

**Step 3**: Remove $\vec{u}_1$ and $\vec{u}_2$ components from $\vec{v}_3$
$$\vec{w}_3 = \vec{v}_3 - (\vec{v}_3 \cdot \vec{u}_1)\vec{u}_1 - (\vec{v}_3 \cdot \vec{u}_2)\vec{u}_2$$

After calculation: $\vec{u}_3 = \left(-\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}\right)$

The result is an orthonormal basis that spans the same space as the original vectors!
:::

::::{hint} ✅ Test Your Understanding: Vectors

Before moving on, can you answer these conceptual questions?

1. Why can we slide vectors around freely in space?
2. What does it mean physically when two vectors are orthogonal?
3. Why does the dot product give us work done by a force?
4. What information does the cross product magnitude tell us?

Think about these before checking the answers!

:::{admonition} Answers
:class: tip, dropdown

1. Vectors represent displacements/instructions, not fixed positions. The instruction "go 3 km north" is the same regardless of starting point.

2. The vectors represent independent directions - no component of one lies along the other. Physically: perpendicular forces, independent measurements.

3. Work is energy transferred along the direction of motion. The dot product extracts exactly the component of force along displacement.

4. The area of the parallelogram formed by the vectors. In physics: angular momentum magnitude, torque strength, rate of area swept in orbit.
:::
::::

### Progressive Problems: Vectors

::::{admonition} 📝 Practice Problems
:class: note

**Level 1 (Conceptual)**: <br> Verify that the vectors $\vec{a} = (1, 0, 0)$ and $\vec{b} = (0, 1, 0)$ are orthogonal.

**Level 2 (Computational)**: <br> Find a unit vector perpendicular to both $\vec{a} = (1, 2, 3)$ and $\vec{b} = (4, 5, 6)$.

**Level 3 (Theoretical)**: <br> Prove that for any vectors $\vec{a}$, $\vec{b}$, $\vec{c}$, the scalar triple product is cyclic:
  $$\vec{a} \cdot (\vec{b} \times \vec{c}) = \vec{b} \cdot (\vec{c} \times \vec{a}) = \vec{c} \cdot (\vec{a} \times \vec{b}).$$

:::{tip} Solutions
:class: dropdown

**Level 1**:<br> $\vec{a} \cdot \vec{b} = (1)(0) + (0)(1) + (0)(0) = 0$ ✓ They're orthogonal.

**Level 2**:

- $\vec{a} \times \vec{b} = (2·6-3·5, 3·4-1·6, 1·5-2·4) = (-3, 6, -3)$
- Magnitude: $|\vec{a} \times \vec{b}| = \sqrt{9+36+9} = \sqrt{54} = 3\sqrt{6}$
- Unit vector: $\hat{n} = \frac{1}{3\sqrt{6}}(-3, 6, -3) = \frac{1}{\sqrt{6}}(-1, 2, -1)$

**Level 3**:<br> The scalar triple product represents the volume of the parallelepiped formed by the three vectors. Since volume doesn't depend on which face you consider as the base, the product is cyclic. Algebraically, this follows from the determinant representation of the triple product.
:::
::::

:::{important} 📌 Key Takeaway

Vectors are not just arrows or lists of numbers - they're the natural language for any quantity with magnitude and direction. Master the dot product (projection/angle), cross product (perpendicularity/rotation), and basis decomposition, and you can describe any physical system from stellar velocities to electromagnetic fields.
:::

---

## Part 3: Matrices - Transformations That Preserve Structure

**Priority: 🔴 Essential** - Core to all computational methods

:::{admonition} 📅 When You'll Use This
:class: dropdown, hint

- **First appears in**: Project 1 (vectorization with matrices)
- **Critical for**: All projects, especially neural networks
- **Most important**: Understanding matrices as transformations, not just number grids
:::

:::{note} 🔭 Where We're Going

Matrices are where linear algebra becomes powerful. We'll discover that matrices aren't just rectangular arrays of numbers - they're transformations that reshape space while preserving its linear structure. You'll learn to read a matrix and immediately understand what it does geometrically, setting the foundation for everything from N-body dynamics to neural networks.
:::

:::{admonition} 📚 Prerequisites Check
:class: hint

Before proceeding, can you:

- [ ] Multiply two 2×2 matrices by hand?
- [ ] Explain why matrix multiplication order matters?
- [ ] Calculate the determinant of a 2×2 matrix?

If not, review the Prerequisites section before continuing.
:::

### 3.1 Matrices ARE Linear Transformations {#part-3-matrices}

```{margin}
**matrix**
A linear transformation represented as a rectangular array of numbers
```

A **matrix** is not just a grid of numbers — it's a rule for transforming vectors. When we multiply matrix $A$ by vector $\vec{v}$, we get a new vector $A\vec{v}$. The crucial property is **linearity**:

$$A(\alpha\vec{u} + \beta\vec{v}) = \alpha A\vec{u} + \beta A\vec{v}$$

:::{admonition} 💡 Building Intuition: Why Row-Times-Column?
:class: note

The strange-looking matrix multiplication rule (row times column) ensures that transformation composition works correctly. Here's why:

When you apply transformation $A$ to vector $\vec{v}$, each component of the output is a linear combination of input components. The first row of $A$ tells you how to compute the first output component:

$$\text{output}_1 = a_{11} v_1 + a_{12} v_2 + a_{13} v_3$$

This is exactly a dot product of row 1 with the vector! Each row defines how one output component depends on all input components. This structure ensures that composing transformations (multiplying matrices) gives the same result as applying them sequentially.
:::

### 3.2 Building Intuition with Concrete Examples {#building-intuition-concrete}

**Priority: 🔴 Essential** - Recognize these patterns in your code

Let's see actual matrices and understand what they do:

**The Identity Matrix** (the "do nothing" transformation):
$$I = \begin{pmatrix} 
\boxed{1} & 0 & 0 \\ 
0 & \boxed{1} & 0 \\ 
0 & 0 & \boxed{1} 
\end{pmatrix}$$

Notice the diagonal of ones! For any vector: $I\vec{v} = \vec{v}$. Visually, this leaves all vectors exactly where they are - no rotation, no scaling, no shearing.

**A Rotation Matrix** (30° around z-axis):
$$R_{30°} = \begin{pmatrix} 
\cos(30°) & -\sin(30°) & 0 \\
\sin(30°) & \cos(30°) & 0 \\
0 & 0 & 1 
\end{pmatrix} = \begin{pmatrix} 
\frac{\sqrt{3}}{2} & -\frac{1}{2} & 0 \\
\frac{1}{2} & \frac{\sqrt{3}}{2} & 0 \\
0 & 0 & 1 
\end{pmatrix}$$

Note: $\frac{\sqrt{3}}{2} \approx 0.866$ and $\frac{1}{2} = 0.5$. The $z$-component is unchanged (last row is [0, 0, 1]) because we rotate around $z$. Imagine looking down the $z$-axis: this matrix spins everything counterclockwise by 30° like a record player, preserving all lengths.

**A Diagonal Matrix** (scaling transformation):
$$D = \begin{pmatrix}
\boxed{2} & 0 & 0 \\
0 & \boxed{3} & 0 \\
0 & 0 & \boxed{1}
\end{pmatrix}$$

Doubles $x$, triples $y$, leaves $z$ unchanged. Only diagonal elements are non-zero! This stretches space non-uniformly - imagine stretching a rubber sheet more in one direction than another.

**A Symmetric Matrix** (moment of inertia):
$$I = \begin{pmatrix}
5 & \boxed{2} & 0 \\
\boxed{2} & 8 & \boxed{1} \\
0 & \boxed{1} & 6
\end{pmatrix}$$

Notice $I_{ij} = I_{ji}$ (boxed elements show symmetry). Symmetric matrices represent reciprocal relationships - if $x$ affects $y$, then $y$ affects $x$ equally.

::::{hint} 🤔 Check Your Understanding

What does this matrix do to vectors?
$$M = \begin{pmatrix} 
-1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 
\end{pmatrix}$$

:::{tip} Solution
:class: dropdown

This matrix flips the x-component while leaving y and z unchanged. It's a reflection through the yz-plane! If you apply it twice, you get back the original vector (since $(-1) \times (-1) = 1$).

Test with $\vec{v} = (3, 4, 5)$:
$$M\vec{v} = \begin{pmatrix} -3 \\ 4 \\ 5 \end{pmatrix}$$

This transformation appears in physics when dealing with parity operations or mirror symmetries.
:::
::::

### 3.3 Matrix Multiplication: Composition of Transformations {#matrix-multiplication}

**Priority: 🔴 Essential** - Fundamental operation

```{margin}
**matrix multiplication**
Operation combining two transformations into one: $(AB)\vec{v} = A(B\vec{v})$
```

Matrix multiplication encodes function composition. The element in row $i$, column $j$ of product $AB$ is:

$$(AB)_{ij} = \sum_k A_{ik}B_{kj}$$

This seemingly weird formula ensures that:
$$(AB)\vec{v} = A(B\vec{v})$$

**Why Order Matters**:

Consider rotating 90° around x-axis, then 90° around z-axis, versus the opposite order:

<!-- Suggested figure: Show a cube being rotated in both orders, ending in different orientations -->

Different orders give different results! This is why $AB \neq BA$ in general.

:::{warning} ⚠️ Common Pitfall: Matrix Multiplication Order
:class: warning

Students often try: $A \cdot B \cdot C = A \cdot C \cdot B$ (wrong!)

Matrix multiplication is NOT commutative. Order matters because each matrix is a transformation applied in sequence. Rotating then scaling gives different results than scaling then rotating!

Remember: When you see $ABC\vec{v}$, read it right-to-left: "First apply C, then B, then A to vector v."
:::

**Understanding Determinants - Three Levels**:

**Level 1 (Concrete):** Det = 0 means "information is lost"
- Example: Projecting 3D onto 2D loses depth information

**Level 2 (Geometric):** Det measures volume scaling
- Example: det=2 means volumes double under transformation

**Level 3 (Abstract):** Det is the product of eigenvalues
- This connects spectrum to geometry to invertibility!

:::{note} 📝 Worked Example: 3×3 Matrix Inversion by Hand
:class: dropdown

**Priority: 🟢 Enrichment** - Useful for understanding numerical methods

Let's invert a 3×3 matrix using the cofactor method. Given:
$$A = \begin{pmatrix} 
2 & 1 & 0 \\
1 & 2 & 1 \\
0 & 1 & 2 
\end{pmatrix}$$

**Step 1**: Calculate the determinant
$$\det(A) = 2\begin{vmatrix}2&1\\1&2\end{vmatrix} - 1\begin{vmatrix}1&1\\0&2\end{vmatrix} + 0$$
$$= 2(4-1) - 1(2-0) = 6 - 2 = 4$$

Since $\det(A) \neq 0$, the matrix is invertible.

**Step 2**: Calculate the matrix of minors
For element $(1,1)$: $M_{11} = \begin{vmatrix}2&1\\1&2\end{vmatrix} = 3$

For element $(1,2)$: $M_{12} = \begin{vmatrix}1&1\\0&2\end{vmatrix} = 2$

For element $(1,3)$: $M_{13} = \begin{vmatrix}1&2\\0&1\end{vmatrix} = 1$

Continue for all nine elements to get the complete matrix of minors.

**Step 3**: Apply cofactor signs (checkerboard pattern)
$$C = \begin{pmatrix} 
+3 & -2 & +1 \\
-2 & +4 & -2 \\
+1 & -2 & +3 
\end{pmatrix}$$

**Step 4**: Transpose and divide by determinant
$$A^{-1} = \frac{1}{4}C^T = \frac{1}{4}\begin{pmatrix} 
3 & -2 & 1 \\
-2 & 4 & -2 \\
1 & -2 & 3 
\end{pmatrix}$$

**Verify**: Multiply $AA^{-1}$ to confirm it equals $I$ ✓
:::

### 3.4 The Determinant: Volume, Orientation, and Information

**Priority: 🟡 Important** - Helps understand singularity and numerical stability

```{margin}
**determinant**
Scalar value measuring how a linear transformation scales volumes
```

```{margin}
**symplectic**
Transformation preserving phase space volume (determinant = 1)
```

The **determinant** tells us three crucial things:

| $\det(A)$ | Meaning | Physical Interpretation |
|-----------|---------|------------------------|
| $\|\det(A)\|$ | Volume scaling factor | How much the transformation stretches/shrinks space |
| $\det(A) > 0$ | Preserves orientation | Right-handed stays right-handed |
| $\det(A) < 0$ | Flips orientation | Right-handed becomes left-handed |
| $\det(A) = 0$ | Singular (non-invertible) | Information is lost, dimension collapses |

```{margin}
**rank**
The number of linearly independent rows (or columns) in a matrix; the dimension of the space the matrix actually maps to
```

When $\det(A) = 0$, the matrix is singular and has **rank** less than its dimension. Let's understand what this really means:

**Intuitive Understanding of Rank**:
- **Full rank** (rank = n for n×n matrix): All rows/columns point in independent directions. The matrix preserves dimensionality—3D space stays 3D.
- **Rank deficient** (rank < n): Some rows/columns are linear combinations of others. The matrix collapses space—maybe 3D becomes a 2D plane or even a 1D line.

**Physical Example**: Consider measuring stellar properties:
```python
# Columns: [mass, luminosity, log(luminosity)]
data = np.array([
    [1.0,  1.0,  0.0],
    [2.0,  8.0,  0.9],
    [0.5,  0.125, -0.9]
])
```
This matrix has rank 2, not 3! The third column (log luminosity) is completely determined by the second column. We're really only measuring 2 independent quantities, not 3. The covariance matrix of this data would be singular—we can't invert it because one dimension is redundant.

**Geometric Interpretation**:
- Rank 3 matrix: Maps 3D space to full 3D space
- Rank 2 matrix: Squashes 3D space onto a 2D plane
- Rank 1 matrix: Collapses 3D space onto a 1D line
- Rank 0 matrix: Maps everything to a single point (zero matrix)

Specifically, an $n \times n$ matrix with rank $r < n$ maps $n$-dimensional space onto an $r$-dimensional subspace, losing $(n-r)$ dimensions of information.

For a 2×2 matrix:
$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

**Geometric interpretation**: For 2×2 matrices, the determinant equals the (signed) area of the parallelogram formed by the column vectors. For 3×3 matrices, it's the (signed) volume of the parallelepiped.

For a 3×3 matrix, the determinant can be computed by cofactor expansion along any row or column. The pattern extends to larger matrices, though computation becomes expensive ($O(n!)$ operations).

:::{warning} ⚠️ Numerical Caution
Computing determinants directly is numerically unstable for large matrices. Use `np.linalg.slogdet()` for the log-determinant when you only need the sign and magnitude separately—this avoids overflow/underflow issues.
:::

:::{important} 🎯 Why This Matters for Your Projects

**In Project 2**: Leapfrog integrator is **symplectic** — preserves phase space volume (det = 1), conserving energy over millions of orbits.

**In Project 4-5**: Rank-deficient covariance matrices indicate perfectly correlated parameters—you're not learning independent information about all variables. This is why regularization (adding small values to diagonal) is needed to ensure full rank.

**In Neural Networks**: Rank of weight matrices determines information bottlenecks—low rank layers compress information, potentially losing important features.
:::

### 3.5 The Inverse: Undoing Transformations

**Priority: 🟡 Important** - Critical for solving systems

```{margin}
**matrix inverse**
The transformation that undoes another: $A^{-1}A = I$
```

The **inverse** matrix $A^{-1}$ satisfies:
$$A^{-1}A = AA^{-1} = I$$

For a 2×2 matrix:
$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix} \implies A^{-1} = \frac{1}{\det(A)}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

Notice you divide by the determinant — this is why singular matrices (det = 0) have no inverse!

### Progressive Problems: Matrices

::::{note} 📝 Practice Problems

**Level 1**: Show that rotation matrices preserve lengths (hint: check that $|R\vec{v}| = |\vec{v}|$).

**Level 2**: Prove that the determinant of a product equals the product of determinants: $\det(AB) = \det(A)\det(B)$.

**Level 3**: Find conditions on matrix $A$ such that $A^2 = I$ (such matrices are called involutions).

:::{tip} Solutions
:class: dropdown

**Level 1**: For rotation matrix $R$, we have $R^TR = I$ (orthogonal property). Then:
$|R\vec{v}|^2 = (R\vec{v})^T(R\vec{v}) = \vec{v}^TR^TR\vec{v} = \vec{v}^T\vec{v} = |\vec{v}|^2$ ✓

**Level 2**: This follows from the multiplicative property of determinants. Geometrically: if $A$ scales volumes by factor $\det(A)$ and $B$ by $\det(B)$, then $AB$ scales by $\det(A) \cdot \det(B)$.

**Level 3**: $A^2 = I$ means $A = A^{-1}$. This requires eigenvalues to be ±1 (since if $A\vec{v} = \lambda\vec{v}$, then $A^2\vec{v} = \lambda^2\vec{v} = \vec{v}$, so $\lambda^2 = 1$). Examples: reflections, 180° rotations.
:::
::::

:::{admonition} 📌 Key Takeaway
:class: important

Matrices are **transformations**, not just number grids. Every matrix tells a geometric story: rotations preserve lengths, scalings stretch space, projections collapse dimensions. When you multiply matrices, you're composing transformations. This geometric view transforms abstract calculations into visual understanding.
:::

---

## Part 4: Eigenvalues and Eigenvectors - Finding Invariant Structure

**Priority: 🟡 Important** - Critical for stability analysis and convergence

:::{hint} 📅 When You'll Use This
:class: dropdown

- **First appears in**: Project 2 (orbital stability)
- **Critical for**: Projects 4-5 (convergence, optimization)
- **Returns in**: Final Project (neural network training dynamics)

Skip on first reading if time-constrained, return when starting Project 2.
:::

:::{admonition} 🔭 Where We're Going
:class: note

**Eigenvalues** and **eigenvectors** reveal the hidden skeleton of a transformation. We'll discover special directions that don't rotate (only stretch or shrink), and see how these invariant directions determine everything from system stability to convergence rates. This is where linear algebra reveals the deep structure of physical systems.
:::

:::{admonition} 📚 Prerequisites Check
:class: hint

Before this section, ensure you can:
- [ ] Explain why matrix multiplication is not commutative
- [ ] Identify when a matrix is singular by its determinant
- [ ] Calculate determinants of 2×2 and 3×3 matrices

Review Section 3 if needed.
:::

### 4.1 The Eigenvalue Equation: Directions That Don't Rotate {#part-4-eigenvalues}

```{margin}
**eigenvector**
A vector that is only scaled (not rotated) by a transformation
```

```{margin}
**eigenvalue**
The scaling factor for an eigenvector
```

Some special vectors are only scaled by a transformation:

$$A\vec{v} = \lambda\vec{v}$$

These **eigenvectors** $\vec{v}$ and their **eigenvalues** $\lambda$ reveal fundamental structure.

**Geometric Intuition**: Imagine stretching a rubber sheet. Most directions get rotated as the sheet deforms, but along eigenvector directions, points only move closer to or farther from the origin without changing direction. These are the "natural axes" of the transformation.

**Physical Example - Spinning Objects**:

Consider a football. It has three principal axes:
- Long axis (through the points)
- Two short axes (perpendicular to long axis)
  - Spin it around the long axis → stable rotation
  - Spin it around a short axis → stable rotation
  - Spin it at any other angle → it wobbles!

The principal axes are eigenvectors of the moment of inertia tensor.

::::{note} 🧠 Build Your Intuition: Eigenvalues

Without calculating, predict the eigenvalues:
1. Identity matrix I → eigenvalues are all ___?
2. Matrix 2I (doubles all vectors) → eigenvalues are all ___?
3. Rotation matrix (90°) → eigenvalues are ___? (Hint: what real vectors don't change direction?)

::::{tip} Answers
:class: dropdown
1. All 1
2. All 2
3. Complex with |λ|=1. For a 90° rotation in 2D, eigenvalues are $e^{±i\pi/2} = ±i$. No real vectors maintain their direction under 90° rotation, hence complex eigenvalues that explicitly encode the rotation angle!
:::
::::

### 4.2 Finding Eigenvalues: The Characteristic Equation

```{margin}
**characteristic equation**
$\det(A - \lambda I) = 0$, whose roots are eigenvalues
```

```{margin}
**trace**
Sum of diagonal elements, equals sum of eigenvalues
```

To find eigenvalues:

1. Rearrange: $(A - \lambda I)\vec{v} = \vec{0}$
2. For non-trivial solutions: $\det(A - \lambda I) = 0$
3. This gives the **characteristic equation**

For an $n×n$ matrix, the characteristic polynomial has degree $n$, potentially yielding $n$ eigenvalues (counting multiplicity) in the complex numbers. The **trace** (sum of diagonal elements) equals the sum of eigenvalues.

:::{admonition} 📝 Worked Example: Complete Eigenvalue and Eigenvector Calculation
:class: note

**Priority: 🟡 Important** - You'll do this for stability analysis

Find eigenvalues AND eigenvectors of:
$$A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$$

**Step 1**: Set up characteristic equation:
$$\det(A - \lambda I) = \det\begin{pmatrix} 3-\lambda & 1 \\ 1 & 3-\lambda \end{pmatrix} = 0$$

**Step 2**: Calculate determinant:
$$(3-\lambda)^2 - 1 = 0$$
$$\lambda^2 - 6\lambda + 8 = 0$$

**Step 3**: Solve quadratic:
$$\lambda = \frac{6 \pm \sqrt{36-32}}{2} = \frac{6 \pm 2}{2}$$

Therefore: $\lambda_1 = 4$, $\lambda_2 = 2$

**Step 4**: Find eigenvector for $\lambda_1 = 4$:

Solve $(A - 4I)\vec{v} = \vec{0}$:
$$\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives: $-v_1 + v_2 = 0$, so $v_1 = v_2$

Eigenvector: $\vec{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ (or any scalar multiple)

**Step 5**: Find eigenvector for $\lambda_2 = 2$:

Solve $(A - 2I)\vec{v} = \vec{0}$:
$$\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives: $v_1 + v_2 = 0$, so $v_1 = -v_2$

Eigenvector: $\vec{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

**Verification**: Check $A\vec{v}_1 = 4\vec{v}_1$ ✓ and $A\vec{v}_2 = 2\vec{v}_2$ ✓

**Physical Meaning**: This matrix stretches by factor 4 along the diagonal direction $(1,1)$ and by factor 2 along the anti-diagonal $(1,-1)$.
:::

### 4.3 Physical Meaning Throughout Your Projects

**Physical Intuition: Why eigenvalues determine stability**

Consider a star cluster near equilibrium. Small perturbations evolve as:
$$\delta x(t) = e^{\lambda t} v$$

- If $\lambda < 0$: perturbation decays → stable (like a ball in a valley)
- If $\lambda > 0$: perturbation grows → unstable (like a ball on a hill)  
- If $\lambda$ is complex: oscillation with growth/decay (spiral behavior)

The largest eigenvalue determines the fate: even one positive eigenvalue means the system will eventually fly apart!

For instance, a 2D rotation matrix has eigenvalues $e^{±i\theta}$, where the imaginary unit explicitly encodes the rotation angle $\theta$. This shows how complex eigenvalues naturally describe oscillatory behavior in physical systems.

Eigenvalues appear in most projects in this course:

| Project | Where Eigenvalues Appear | What They Tell You |
|---------|-------------------------|-------------------|
| Project 2 | Linearized dynamics near equilibrium | Orbital stability (stable if all $\lambda < 0$) |
| Project 3 | Scattering matrix | Preferred scattering directions |
| Project 4 | MCMC transition matrix | Convergence rate: $\sim 1/|1-\lambda_2|$ where $\lambda_2$ is the second-largest eigenvalue by absolute value (with $|\lambda_2| < 1$) |
| Project 5 | GP kernel matrix | Effective degrees of freedom |
| Final Project | Neural network Hessian | Optimization landscape curvature |

:::{admonition} 💡 Deep Connection: Why Symmetric Matrices Are Special
:class: note

**Priority: 🟡 Important for Projects 4-5**

**The Spectral Theorem** guarantees that symmetric matrices have:
1. All real eigenvalues
2. Orthogonal eigenvectors

This isn't mathematical coincidence — it's physical necessity! Symmetric matrices represent quantities where direction doesn't matter for the relationship (like forces between particles). If eigenvalues could be complex, we'd have complex energies or distances, which is nonsensical. Mathematics enforces physical reasonableness!
:::

:::{tip} 🌟 The More You Know: Jacobi's Method Born from Astronomy
:class: dropdown

**Priority: 🟢 Enrichment**

Carl Gustav Jacob Jacobi (1804-1851) developed his famous eigenvalue algorithm while studying the rotation of celestial bodies. In 1846, he published "Über ein leichtes Verfahren die in der Theorie der Säcularstörungen vorkommenden Gleichungen numerisch aufzulösen" (*On an easy method to numerically solve equations occurring in the theory of secular perturbations*).

The problem arose from calculating planetary perturbations—small gravitational influences planets exert on each other. These perturbations accumulate over centuries (hence "secular") and are described by symmetric matrices whose eigenvalues determine the long-term stability of orbits.

Jacobi's insight was to diagonalize the matrix through a sequence of rotations, each zeroing out one off-diagonal element. Though computers didn't exist, his method was designed for hand calculation—each step is simple enough to do with pencil and paper. Today, variants of Jacobi's method run on every supercomputer simulating galaxy collisions or climate models.

The next time your code calls `numpy.linalg.eigh()`, remember: you're using an algorithm invented to predict whether the solar system is stable over millions of years!

*Source: Jacobi, C.G.J. (1846). Journal für die reine und angewandte Mathematik (Crelle's Journal), 30, pp. 51-94.*
:::

:::{note} 📚 Mathematical Deep Dive: Proof that Symmetric Matrices Have Real Eigenvalues
:class: dropdown

**Priority: 🟢 Enrichment**

This proof reveals why physics "prefers" symmetric matrices.

**Theorem**: Every real symmetric matrix has only real eigenvalues.

**Proof**:
Let $A$ be a real symmetric matrix and suppose $\lambda$ is an eigenvalue with eigenvector $\vec{v}$. We'll show $\lambda$ must be real.

Consider the complex conjugate of the eigenvalue equation:
$$A\vec{v} = \lambda\vec{v}$$

Taking complex conjugates:
$$\overline{A\vec{v}} = \overline{\lambda\vec{v}}$$

Since $A$ is real: $A\overline{\vec{v}} = \overline{\lambda}\overline{\vec{v}}$

Now compute $\vec{v}^* A \vec{v}$ in two ways:

Method 1: $\vec{v}^* A \vec{v} = \vec{v}^* (\lambda \vec{v}) = \lambda (\vec{v}^* \vec{v}) = \lambda |\vec{v}|^2$

Method 2: $\vec{v}^* A \vec{v} = (A^T \vec{v})^* \vec{v} = (A\vec{v})^* \vec{v}$ (since $A = A^T$)
         $= (\lambda\vec{v})^* \vec{v} = \overline{\lambda} \vec{v}^* \vec{v} = \overline{\lambda} |\vec{v}|^2$

Therefore: $\lambda |\vec{v}|^2 = \overline{\lambda} |\vec{v}|^2$

Since $|\vec{v}|^2 > 0$ (eigenvector is non-zero):
$$\lambda = \overline{\lambda}$$

This means $\lambda$ is real! ∎

**Physical Interpretation**: Symmetric matrices represent measurements where the order doesn't matter (distance from A to B equals distance from B to A). Complex eigenvalues would imply complex measurements, violating physical reality.
:::

### Progressive Problems: Eigenvalues

::::{note} 📝 Practice Problems

**Level 1**: Find the eigenvalues of $\begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$ by inspection.

**Level 2**: Prove that if $\lambda$ is an eigenvalue of $A$, then $\lambda^2$ is an eigenvalue of $A^2$.

**Level 3**: Show that positive definite matrices have all positive eigenvalues.

:::{tip} Solutions
:class: dropdown

**Level 1**: For diagonal matrices, eigenvalues are the diagonal elements: $\lambda_1 = 2$, $\lambda_2 = 3$. The eigenvectors are the standard basis vectors.

**Level 2**: If $A\vec{v} = \lambda\vec{v}$, then $A^2\vec{v} = A(A\vec{v}) = A(\lambda\vec{v}) = \lambda(A\vec{v}) = \lambda(\lambda\vec{v}) = \lambda^2\vec{v}$. Thus $\lambda^2$ is an eigenvalue of $A^2$ with the same eigenvector.

**Level 3**: For positive definite $A$, we have $\vec{v}^TA\vec{v} > 0$ for all $\vec{v} \neq 0$. For eigenvector $\vec{v}$ with eigenvalue $\lambda$: $\vec{v}^TA\vec{v} = \vec{v}^T(\lambda\vec{v}) = \lambda|\vec{v}|^2 > 0$. Since $|\vec{v}|^2 > 0$, we must have $\lambda > 0$.
:::
::::

:::{admonition} 📌 Key Takeaway
:class: important

Eigenvalues and eigenvectors are the DNA of a matrix. They reveal invariant directions (eigenvectors) and scaling factors (eigenvalues) that determine stability, convergence, and long-term behavior. When all eigenvalues are negative, systems are stable. When they're positive, systems grow. When they're complex, systems oscillate. Master this concept and you can predict the fate of any linear system.
:::

---

## Part 5: Positive Definite Matrices and Statistical Foundations

**Priority: 🟡 Important** - Essential for Projects 4-5

:::{admonition} 📅 When You'll Use This
:class: dropdown, hint

- **First appears in**: Project 4 (MCMC covariance matrices)
- **Critical for**: Project 5 (Gaussian Process kernels)
- **Why it matters**: Ensures physical validity of statistical methods

Can skip initially, but must understand before Project 4.
:::

:::{admonition} 🔭 Where We're Going
:class: note

**Positive definite matrices** are where linear algebra meets statistics and optimization. We'll see why these special matrices guarantee that energies are positive, distances make sense, and probability distributions are valid. This section bridges deterministic physics with statistical methods, preparing you for MCMC and Gaussian Processes.
:::

### 5.1 Quadratic Forms and Energy {#part-5-positive-definite}

```{margin}
**quadratic form**
Expression $Q(\vec{x}) = \vec{x}^T A \vec{x}$ where A is symmetric
```

A **quadratic form** is:
$$Q(\vec{x}) = \vec{x}^T A \vec{x} = \sum_{i,j} A_{ij} x_i x_j$$

These appear as energy expressions throughout physics:

| Type | Formula | Physical Meaning |
|------|---------|------------------|
| Kinetic Energy | $T = \frac{1}{2}\vec{v}^T M \vec{v}$ | M is mass matrix |
| Potential Energy | $V = \frac{1}{2}\vec{x}^T K \vec{x}$ | K is stiffness matrix |
| Statistical Distance | $d^2 = (\vec{x}-\vec{\mu})^T \Sigma^{-1} (\vec{x}-\vec{\mu})$ | Mahalanobis distance |

**Physical Example: The kinetic energy of a rotating rigid body**
$$T = \frac{1}{2}\vec{\omega}^T I \vec{\omega}$$
where $\vec{\omega}$ is angular velocity and $I$ is the moment of inertia tensor. This quadratic form is always positive (energy can't be negative), making $I$ positive definite. The eigenvectors of $I$ are the principal axes - spin around these and the object doesn't wobble!

### 5.2 Positive Definiteness: Ensuring Physical Reality {#positive-definiteness}

```{margin}
**positive definite**
Matrix where $\vec{x}^T A \vec{x} > 0$ for all $\vec{x} \neq \vec{0}$
```

```{margin}
**Cholesky decomposition**
Factorization of a positive definite matrix $A = LL^T$ where $L$ is lower triangular with positive diagonal entries
```

A symmetric matrix is **positive definite** if its quadratic form is always positive. This property ensures physical validity—energies are positive, distances are non-negative, and covariance matrices represent valid uncertainties.

**Four Equivalent Tests for Positive Definiteness**:

1. ✅ All eigenvalues > 0
2. ✅ All leading principal minors > 0
3. ✅ Has Cholesky decomposition $A = LL^T$ (requires strict positive definiteness)
4. ✅ Can write as $A = B^T B$ for some invertible matrix $B$

**Quick check**: Positive definite matrices always have positive diagonal elements. Proof: For standard basis vector $\vec{e}_i$, we have $\vec{e}_i^T A \vec{e}_i = A_{ii} > 0$.

**Important Distinction**:
- **Positive definite** (all eigenvalues > 0): Standard Cholesky works
- **Positive semi-definite** (eigenvalues ≥ 0): May have rank deficiency (rank < dimension), requiring modified Cholesky or eigendecomposition
- **Why it matters**: Zero eigenvalues mean the matrix maps some directions to zero—information is lost

```{margin}
**regularization**
Adding small positive values (often to diagonals) to improve numerical stability or prevent singularities
```

:::{warning} ⚠️ Common Bug in Project 5

When implementing Gaussian Processes, your kernel matrix might lose positive definiteness due to numerical errors. 

**Symptoms**:
- Cholesky decomposition fails with `numpy.linalg.LinAlgError`
- Negative variance predictions (physically impossible!)
- Eigenvalues that should be positive show as tiny negatives (e.g., -1e-15)

**Why this happens**: Floating-point arithmetic accumulates tiny errors. A mathematically positive definite matrix can become numerically indefinite.

**Fix**: Add small "jitter" to diagonal:
```python
K_stable = K + 1e-6 * np.eye(n)  # Add small positive value to diagonal
# np.eye(n) creates an n×n identity matrix (1s on diagonal, 0s elsewhere)
```

This is **regularization**, not a hack! It accounts for numerical precision limits and is standard practice in GP implementations.
:::

:::{note} 🔍 Regularization Throughout Your Projects

**Regularization**—adding small terms to prevent numerical issues—appears everywhere in computational astrophysics:

**Project 2 (N-body)**: Gravitational softening prevents $r \to 0$ singularities.

```python
F = GMm/(r^2 + epsilon^2)  # epsilon prevents division by zero
```

**Project 4 (MCMC)**: Ridge regression adds $\lambda ||w||^2$ to prevent overfitting.

```python
loss = data_term + lambda * np.sum(weights**2)  # L2 regularization
```

**Project 5 (GPs)**: Jitter term ensures positive definiteness.

```python
K = K + sigma_n^2 * I  # Noise term regularizes covariance
```

**Final Project (Neural Networks)**: Weight decay prevents overfitting.

```python
loss = cross_entropy + weight_decay * sum_of_squared_weights
```

Regularization isn't cheating — it's acknowledging that perfect mathematical conditions don't exist in finite-precision computation. The art is choosing regularization strength: too little fails to stabilize, too much distorts your physics.
:::

:::{admonition} 💡 Building Intuition: The Bowl Analogy
:class: note

A positive definite matrix creates a bowl-shaped quadratic form. Imagine the function $f(\vec{x}) = \vec{x}^T A \vec{x}$ as a landscape:

- **Positive definite**: Bowl opening upward (unique minimum at origin)
- **Negative definite**: Bowl opening downward (unique maximum at origin)
- **Indefinite**: Saddle shape (some directions go up, others down)
- **Positive semi-definite**: Bowl with flat bottom (minimum not unique)

This is why positive definite matrices guarantee unique solutions in optimization—there's only one bottom of the bowl!
:::

### 5.3 Covariance Matrices: The Bridge to Statistics {#covariance-matrices}

**Priority: 🟡 Important** - Foundation for Projects 4-5

```{margin}
**covariance**
Measure of how two variables change together; positive means they increase together, negative means one increases as the other decreases
```

```{margin}
**covariance matrix**
Matrix containing all pairwise covariances between random variables; encodes all linear relationships in your data
```

Before diving into matrices, let's understand what **covariance** actually tells us:

**Intuitive Understanding**:
- **Positive covariance**: Variables tend to increase together (mass and luminosity in stars)
- **Negative covariance**: One increases as the other decreases (stellar temperature and radius for giants)
- **Zero covariance**: No linear relationship (doesn't mean independent!)

**Mathematical Definition**:
$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

This measures the average product of deviations from the mean. When both variables are above (or below) their means together, the products are positive, giving positive covariance.

**Relationship to Correlation**:
$$\text{Correlation} = \rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

Correlation is just normalized covariance! It ranges from -1 to +1, making it easier to interpret than raw covariance.

**The Covariance Matrix**

For multiple variables $X_1, ..., X_n$, we organize all pairwise covariances into a matrix:

$$\Sigma_{ij} = \text{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]$$

**Structure and Properties**:
- **Diagonal elements**: $\Sigma_{ii} = \text{Var}(X_i)$ (variances are "self-covariance")
- **Off-diagonal elements**: $\Sigma_{ij}$ (covariances between different variables)
- **Always symmetric**: $\Sigma_{ij} = \Sigma_{ji}$ (order doesn't matter)
- **Always positive semi-definite**: All eigenvalues ≥ 0 (why? see below)

**Why Positive Semi-Definite?**

For any linear combination of variables $Y = a^T X$:
$$\text{Var}(Y) = a^T \Sigma a \geq 0$$

Variance can't be negative! This forces $\Sigma$ to be positive semi-definite. If it has negative eigenvalues, you could construct a linear combination with negative variance—physically impossible!

:::{admonition} 📝 Worked Example: Constructing a Covariance Matrix from Data
:class: note

**Priority: 🟡 Important for Project 4**

Given stellar measurements for 4 stars (mass and luminosity):

```python
# Data: each row is a star, columns are [mass, luminosity]
data = np.array([
    [1.0, 1.2],   # Star 1
    [1.5, 3.1],   # Star 2
    [0.8, 0.6],   # Star 3
    [1.2, 1.9]    # Star 4
])

# Step 1: Compute means
means = np.mean(data, axis=0)  # [1.125, 1.7]

# Step 2: Center the data (subtract means)
centered = data - means
# [[-0.125, -0.5],   # Star 1 below avg mass, below avg lum
#  [ 0.375,  1.4],   # Star 2 above avg mass, above avg lum
#  [-0.325, -1.1],   # Star 3 below avg mass, below avg lum
#  [ 0.075,  0.2]]   # Star 4 slightly above both

# Step 3: Compute covariance matrix
n = len(data)
cov = (centered.T @ centered) / (n - 1)  # n-1 for sample covariance
# [[0.0892, 0.3242],
#  [0.3242, 1.2867]]

# Interpretation:
# - Var(mass) = 0.0892 (spread in masses)
# - Var(luminosity) = 1.2867 (larger spread in luminosities)
# - Cov(mass, lum) = 0.3242 > 0 (positive correlation!)

# Correlation coefficient:
correlation = 0.3242 / (np.sqrt(0.0892) * np.sqrt(1.2867))
# correlation ≈ 0.96 (very strong!)
```

The positive covariance confirms the mass-luminosity relation! More massive stars are more luminous.
:::

:::{admonition} 🔍 Geometric Interpretation of Covariance
:class: note

The covariance matrix defines an **uncertainty ellipsoid** in parameter space:

- **Eigenvectors**: Principal axes of the ellipsoid (directions of maximum/minimum variance)
- **Eigenvalues**: Variance along each principal axis (squared semi-axis lengths)
- **Off-diagonal terms**: Tilt of the ellipsoid (correlations rotate the axes)

In Project 4 (MCMC), you'll sample from multivariate Gaussians with covariance $\Sigma$. Your samples will form an elliptical cloud with shape determined by $\Sigma$!

In Project 5 (GPs), the kernel matrix IS a covariance matrix — it encodes how correlated function values are at different points.
:::

### 5.4 The Multivariate Gaussian Distribution

**Priority: 🟡 Important** - Core of Projects 4-5

```{margin}
**multivariate Gaussian**
Multi-dimensional bell curve defined by mean vector and covariance matrix
```

```{margin}
**Mahalanobis distance**
Scale-invariant distance that accounts for correlations; measures "how many standard deviations away" in correlated space
```

The **multivariate Gaussian** extends the familiar bell curve to multiple dimensions. In 1D, you know the Gaussian as the bell-shaped $e^{-x^2}$ curve. In higher dimensions, it becomes an ellipsoidal cloud of probability.

**The Formula Decoded**:

$$p(\vec{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})\right)$$

Let's understand each piece intuitively:

**The Exponent**: $-\frac{1}{2}(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})$
- This is the **Mahalanobis distance** squared
- In 1D with variance $\sigma^2$: reduces to $-(x-\mu)^2/(2\sigma^2)$
- Measures "how many standard deviations away" but accounts for correlations
- If variables are correlated, being far in one direction might be "cheaper" than another

**The Normalization**: $(2\pi)^{n/2}|\Sigma|^{1/2}$
- Ensures total probability integrates to 1
- $|\Sigma|$ is the determinant—the "volume" of the uncertainty ellipsoid
- Larger determinant = more spread out = lower peak height (volume conserved)

**Geometric Picture**:
- **Mean $\vec{\mu}$**: Center of the probability cloud
- **Covariance $\Sigma$**: Shape and orientation of the ellipsoid
  - Eigenvalues: Lengths of ellipsoid axes
  - Eigenvectors: Directions of axes
- **Contours of constant probability**: Ellipsoids defined by $(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu}) = c$

:::{admonition} 🔍 Why $\Sigma^{-1}$ Appears (Not Just $\Sigma$)
:class: note

The inverse covariance $\Sigma^{-1}$ (called the **precision matrix**) appears because we need to "undo" the correlations to measure true distance.

**Analogy**: Imagine measuring distance on a stretched map. If the map stretches 2× horizontally, a 2cm horizontal distance on the map represents only 1cm in reality. You need to divide by the stretch factor (inverse transform) to get true distance.

Similarly, $\Sigma$ stretches space according to variances and correlations. To measure how "far" a point is from the mean in a statistically meaningful way, we need $\Sigma^{-1}$ to undo this stretching.

**In your projects**:
- **Project 4**: MCMC proposals use multivariate Gaussians—the covariance determines step sizes and directions
- **Project 5**: GP predictions are multivariate Gaussians—the kernel determines the covariance structure
:::

### 5.5 Cholesky Decomposition: The Matrix Square Root {#cholesky-decomposition}

**Priority: 🟡 Important** - Essential for Project 5

```{margin}
**Cholesky decomposition**
Factorization $A = LL^T$ where L is lower triangular; geometrically, finds the "square root" of a positive definite matrix
```

Every positive definite matrix can be factored as:
$$A = LL^T$$

where $L$ is lower triangular with positive diagonal entries.

**Intuitive Understanding**: Think of Cholesky as finding the "square root" of a matrix
- Just as $9 = 3 \times 3$, we have $A = L \times L^T$
- $L$ transforms uncorrelated unit variance variables into correlated variables with covariance $A$

**Why Lower Triangular?**
The triangular structure means each variable depends only on previous ones:
- Variable 1: Independent
- Variable 2: Depends on variable 1
- Variable 3: Depends on variables 1 and 2
- And so on...

This sequential dependency structure makes computation efficient!

**Two Key Applications**:

**1. Solving Linear Systems** (faster than computing inverse):
```python
# Problem: Solve Ax = b where A is positive definite

# Slow way: x = inv(A) @ b (NEVER DO THIS!)
# - Computing inverse: O(n³) operations
# - Numerically unstable
# - Loses sparsity structure

# Fast way using Cholesky:
L = np.linalg.cholesky(A)      # A = LL^T
# Now Ax = b becomes LL^T x = b

# Step 1: Solve Ly = b (forward substitution)
y = np.linalg.solve(L, b)

# Step 2: Solve L^T x = y (back substitution)
x = np.linalg.solve(L.T, y)

# Why faster? Triangular systems solve in O(n²) not O(n³)!
```

**2. Generating Correlated Random Variables** (Projects 4-5):

```python
# Generate samples from N(mu, Sigma)

# The magic: If z ~ N(0, I), then Lz ~ N(0, LL^T) = N(0, Sigma)
L = np.linalg.cholesky(Sigma)
z = np.random.randn(n)          # Standard normal (uncorrelated)
x = mu + L @ z                  # Has mean mu, covariance Sigma

# Why does this work?
# - z has covariance I (independent unit variance)
# - Lz has covariance L·I·L^T = LL^T = Sigma
# - Adding mu shifts the mean without changing covariance
```

:::{admonition} 💡 Geometric Interpretation
:class: note

**Cholesky geometrically transforms a sphere into an ellipsoid**:

1. Start with standard normal samples (spherical cloud)
2. $L$ stretches and rotates to create the correlation structure
3. Result matches the desired covariance ellipsoid

This is why Cholesky is essential for:
- **MCMC** (Project 4): Generate proposal steps with correct correlations
- **GP sampling** (Project 5): Generate functions from the GP prior
- **Data augmentation**: Create synthetic data with realistic correlations

The triangular structure of $L$ means the transformation happens sequentially—each dimension adds its contribution based on previous dimensions.
:::

### Progressive Problems: Positive Definiteness

::::{admonition} 📝 Practice Problems
:class: note

**Level 1**: Verify that the identity matrix is positive definite.

**Level 2**: Prove that if $A$ is positive definite, then $A^{-1}$ is also positive definite.

**Level 3**: Show that the sum of two positive definite matrices is positive definite.

:::{tip} Solutions
:class: dropdown

**Level 1**: For identity matrix $I$ and any $\vec{x} \neq 0$: $\vec{x}^T I \vec{x} = |\vec{x}|^2 > 0$ ✓

**Level 2**: Let $\vec{y} = A^{-1}\vec{x}$ where $\vec{x} \neq 0$. Then $\vec{x} = A\vec{y}$ and $\vec{y} \neq 0$. We have: $\vec{x}^T A^{-1} \vec{x} = \vec{y}^T A \vec{y} > 0$ (since $A$ is positive definite).

**Level 3**: For positive definite $A$ and $B$: $\vec{x}^T(A+B)\vec{x} = \vec{x}^T A\vec{x} + \vec{x}^T B\vec{x} > 0 + 0 = 0$ for all $\vec{x} \neq 0$.
:::
::::

:::{admonition} 📌 Key Takeaway
:class: important

**Positive definite matrices** guarantee physical validity. They ensure energies are positive, distances are non-negative, and probability distributions integrate to 1. When you see a positive definite matrix, think "this represents something that must be positive in the real world."

The **multivariate Gaussian** shows how mean vectors and **covariance matrices** fully characterize multi-dimensional uncertainty. **Cholesky decomposition** is your Swiss Army knife for working with these structures — it's the bridge between uncorrelated randomness and structured correlations, making it essential for sampling, solving, and simulating throughout your projects.
:::

---

## Part 6: Advanced Topics for Your Projects

**Priority: 🟢 Enrichment** - Read as needed for specific projects

:::{admonition} 📅 When You'll Use This
:class: dropdown, hint

These advanced topics appear in later projects or when optimizing code. Read as needed rather than all at once. Revisit these sections after completing related projects—they'll make more sense with practical experience.
:::

### 6.1 Singular Value Decomposition - The Swiss Army Knife {#svd-swiss-army}

```{margin}
**SVD**
Universal decomposition $A = U\Sigma V^T$ that reveals the fundamental action of any matrix
```

```{margin}
**singular values**
Non-negative values σ₁ ≥ σ₂ ≥ ... ≥ 0 measuring importance of each component
```

Every matrix has a **singular value decomposition**:
$$A = U\Sigma V^T$$

where:
- $U$: Left singular vectors (orthonormal output directions)
- $\Sigma$: Diagonal matrix of singular values (stretching factors)
- $V^T$: Right singular vectors (orthonormal input directions)

**Geometric Intuition**: Any matrix transformation can be broken into three steps:
1. **Rotate** (by $V^T$): Align input to principal axes
2. **Stretch** (by $\Sigma$): Scale along each axis by σᵢ
3. **Rotate** (by $U$): Align to output space

This means ANY linear transformation—no matter how complex—is just rotate-stretch-rotate! This decomposition is unique (up to sign ambiguities) and always exists.

**Understanding Rank Through SVD**:
The rank of a matrix equals the number of non-zero singular values. This tells you the true dimensionality:
- Full rank: All σᵢ > 0, no information lost
- Rank deficient: Some σᵢ = 0, transformation loses dimensions
- Numerical rank: Count σᵢ > tolerance (e.g., 10⁻¹⁰) for finite precision

**Why SVD Beats Eigendecomposition**:
| Property | Eigendecomposition | SVD |
|----------|-------------------|-----|
| Works for | Square matrices only | ANY matrix shape |
| Requires | Diagonalizable | Always works |
| Vectors | May be complex | Always real for real matrices |
| Values | Can be negative/complex | Always non-negative real |
| Numerical stability | Can be unstable | Very stable algorithms |

**The Deep Connection**:
For any matrix $A$:
- $A^T A$ has eigenvalues $\lambda_i = \sigma_i^2$ and eigenvectors = columns of $V$
- $AA^T$ has eigenvalues $\lambda_i = \sigma_i^2$ and eigenvectors = columns of $U$
- This is why SVD always exists—symmetric matrices always have eigendecompositions!

:::{admonition} 🔍 Principal Component Analysis (PCA) = SVD of Data
:class: note

**PCA is just SVD applied to centered data!**

Given data matrix $X$ (each row = observation, column = feature):
1. Center the data: $X_c = X - \text{mean}(X)$
2. Apply SVD: $X_c = U\Sigma V^T$
3. Principal components = columns of $V$ (right singular vectors)
4. Variance explained by component $i$ = $\sigma_i^2/(n-1)$
5. Fraction of total variance = $\sigma_i^2 / \sum_j \sigma_j^2$

**Concrete Astrophysical Example**: Decomposing 1000 galaxy spectra
```python
# Each row = one galaxy's spectrum (flux at 4000 wavelengths)
spectra = np.array([...])  # Shape: (1000, 4000)

# Center the data
mean_spectrum = np.mean(spectra, axis=0)
centered = spectra - mean_spectrum

# Apply SVD
U, s, Vt = np.linalg.svd(centered, full_matrices=False)

# First 3 components might capture 95% of variation:
# Component 1: Old stellar population (red, smooth)
# Component 2: Star formation (blue, emission lines)
# Component 3: AGN activity (broad lines, power-law continuum)

# Reconstruct using only k components (dimensionality reduction):
k = 10  # Keep first 10 components
compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
# 1000×4000 matrix stored as 1000×10 + 10 + 10×4000 = 54,010 numbers
# Instead of 4,000,000 numbers—74× compression!
```

The singular values tell you how many components you really need—often just 10-20 capture 99% of thousands of dimensions!
:::

### 6.2 Block Matrices and the Schur Complement {#block-matrices}

```{margin}
**block matrix**
Matrix partitioned into submatrices, often reflecting natural system structure
```

```{margin}
**Schur complement**
The "effective" matrix after eliminating some variables: $S = A - BD^{-1}C$
```

Large systems often have natural block structure:
$$M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$$

**Physical Motivation**: Different parts of your system interact differently.

**Concrete Example: Triple Star System**
Consider a close binary with a distant third star:
$$\begin{pmatrix}
F_{11} & F_{12} & F_{13} \\
F_{21} & F_{22} & F_{23} \\
F_{31} & F_{32} & F_{33}
\end{pmatrix} = 
\begin{pmatrix}
\begin{array}{cc|c}
0 & \text{strong} & \text{weak} \\
\text{strong} & 0 & \text{weak} \\
\hline
\text{weak} & \text{weak} & 0
\end{array}
\end{pmatrix}$$

The 2×2 upper-left block represents strong binary interactions, while off-diagonal blocks show weak coupling to the distant star.

**The Schur Complement in Action**:

Let's solve $\begin{pmatrix} A & B \\ C & D \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} f \\ g \end{pmatrix}$

Instead of solving the full system, we can eliminate $y$:
1. From bottom equation: $y = D^{-1}(g - Cx)$
2. Substitute into top: $Ax + BD^{-1}(g - Cx) = f$
3. Rearrange: $(A - BD^{-1}C)x = f - BD^{-1}g$

The matrix $S = A - BD^{-1}C$ is the Schur complement—it's the "effective $A$" after accounting for $y$'s influence through $D$.

**Simple 2×2 Example**:
$$\begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix} = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$$

Schur complement of $D$: $S = 3 - (1)(1/4)(2) = 3 - 0.5 = 2.5$

This 2.5 is the "effective resistance" in the first variable after the second adjusts optimally.

**Why It Matters for Project 5 (GPs)**:
When you add a new observation to a Gaussian Process:
- Old observations = block $A$
- Covariance with new point = blocks $B, C$  
- New point variance = block $D$
- Prediction uses the Schur complement to efficiently update without re-inverting everything!

### 6.3 The Jacobian Matrix: Local Linear Approximation {#jacobian-matrix}

```{margin}
**Jacobian**
Matrix of all first-order partial derivatives; the best linear approximation at a point
```

For vector function $\vec{f}: \mathbb{R}^n \to \mathbb{R}^m$:
$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

**Intuitive Understanding**: Near any point, nonlinear functions look linear. The Jacobian is that linear approximation:
$$\vec{f}(\vec{x} + \delta\vec{x}) \approx \vec{f}(\vec{x}) + J\cdot\delta\vec{x}$$

**Concrete 2D Example**: 
For the transformation $f(r, \theta) = (r\cos\theta, r\sin\theta)$ (polar to Cartesian):
$$J = \begin{pmatrix} 
\frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta} \\
\frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta}
\end{pmatrix} = \begin{pmatrix}
\cos\theta & -r\sin\theta \\
\sin\theta & r\cos\theta
\end{pmatrix}$$

The determinant $|J| = r$ tells you area scaling—this is why polar area elements are $r\,dr\,d\theta$!

**In Project 2 (N-body Stability)**:
Near an equilibrium configuration $\vec{s}_0$, perturbations $\delta\vec{s}$ evolve as:
$$\frac{d(\delta\vec{s})}{dt} = J|_{\vec{s}_0} \cdot \delta\vec{s}$$

The eigenvalues of $J$ determine the fate:
- All Re(λ) < 0: Stable (perturbations decay)
- Any Re(λ) > 0: Unstable (perturbations grow)
- Re(λ) = 0, Im(λ) ≠ 0: Neutral oscillations

Example: For a star orbiting at Lagrange point L4:
- Eigenvalues have small negative real parts → weakly stable
- Large imaginary parts → oscillates if perturbed
- This is why Trojan asteroids librate around L4/L5!

### 6.4 Matrix Exponentials: Solving Linear Evolution {#matrix-exponentials}

```{margin}
**matrix exponential**
$e^{At}$ propagates linear systems forward in time
```

The matrix exponential solves any linear ODE system:
$$\frac{d\vec{x}}{dt} = A\vec{x} \implies \vec{x}(t) = e^{At}\vec{x}(0)$$

**Three Ways to Understand $e^{At}$**:

**1. Series Definition** (like scalar exponential):
$$e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + ...$$

**2. Through Eigenvalues** (if $A$ is diagonalizable):
If $A = PDP^{-1}$ where $D$ is diagonal with eigenvalues:
$$e^{At} = Pe^{Dt}P^{-1} = P\begin{pmatrix} e^{\lambda_1 t} & & \\ & e^{\lambda_2 t} & \\ & & \ddots \end{pmatrix}P^{-1}$$

**3. Physical Interpretation**: $e^{At}$ is the propagator—it evolves the system from time 0 to time $t$.

**Practical Computation**:
```python
# Never use the series directly—it's inefficient and can be inaccurate
# Instead, use specialized algorithms:

import scipy.linalg as la

A = np.array([[0, 1], [-1, -0.1]])  # Damped oscillator
t = 5.0

# Method 1: Direct computation (best for dense matrices)
expAt = la.expm(A * t)

# Method 2: Via eigendecomposition (good for understanding)
eigvals, eigvecs = la.eig(A)
D = np.diag(np.exp(eigvals * t))
expAt_eig = eigvecs @ D @ la.inv(eigvecs)

# Method 3: For solving ODEs, use ODE solvers instead!
# They're more efficient than computing e^{At} explicitly
```

**Example: Damped Harmonic Oscillator**:
For $\ddot{x} + \gamma\dot{x} + \omega_0^2 x = 0$, rewrite as first-order system:
$$\frac{d}{dt}\begin{pmatrix} x \\ v \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ -\omega_0^2 & -\gamma \end{pmatrix} \begin{pmatrix} x \\ v \end{pmatrix}$$

The eigenvalues $\lambda = \frac{-\gamma \pm \sqrt{\gamma^2 - 4\omega_0^2}}{2}$ determine behavior:
- Underdamped ($\gamma < 2\omega_0$): Complex λ → oscillatory decay
- Critically damped ($\gamma = 2\omega_0$): Repeated real λ → fastest decay
- Overdamped ($\gamma > 2\omega_0$): Two real λ → slow decay

### 6.5 Matrix Norms: How Big is a Matrix?

**Priority: 🟢 Enrichment** - Crucial for understanding stability and convergence

```{margin}
**Frobenius norm**
Sum of squared elements: $||A||_F = \sqrt{\sum_{i,j} |a_{ij}|^2}$
```

```{margin}
**spectral norm**
Maximum amplification factor: $||A||_2 = \sigma_{\max}$
```

Matrix norms measure "size" but different norms capture different aspects:

**Frobenius Norm** (total "energy"):
$$||A||_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{\text{trace}(A^T A)}$$
- Like treating matrix as a long vector
- All elements contribute equally
- Easy to compute but ignores structure

**Spectral Norm** (worst-case amplification):
$$||A||_2 = \max_{||\vec{x}||=1} ||A\vec{x}|| = \sigma_{\max}$$
- Maximum stretch factor for any unit vector
- Equals largest singular value
- Determines stability and convergence

**Why Spectral Norm Matters**:
1. **Stability**: System $x_{n+1} = Ax_n$ is stable iff $||A||_2 < 1$
2. **Error amplification**: Input error $\epsilon$ → output error ≤ $||A||_2 \cdot \epsilon$
3. **Condition number**: $\kappa = ||A||_2 \cdot ||A^{-1}||_2 = \sigma_{\max}/\sigma_{\min}$
4. **Convergence rate**: Iterations converge like $(||A||_2)^n$

**Example: Why Deep Networks Are Hard to Train**:
Consider a 10-layer network where each layer multiplies by matrix $W$:
- If $||W||_2 = 1.1$: Gradients grow as $(1.1)^{10} \approx 2.6×$
- If $||W||_2 = 0.9$: Gradients shrink as $(0.9)^{10} \approx 0.35×$
- Need $||W||_2 \approx 1$ for stable training!

This is why techniques like batch normalization and careful initialization are crucial.

### 6.6 Numerical Implementation Examples {#numerical-implementation}

:::{admonition} 💻 Implementation: Power Method for Largest Eigenvalue
:class: note

**Priority: 🟢 Enrichment** - Foundation for many iterative algorithms

The power method finds the dominant eigenvalue through repeated multiplication:

```python
def power_method(A, num_iterations=100, tolerance=1e-10):
    """
    Find largest eigenvalue and corresponding eigenvector.
    
    Why it works: Any vector can be written as a combination
    of eigenvectors. Repeated multiplication by A amplifies
    the component along the largest eigenvector most strongly.
    
    Convergence rate: |λ₂/λ₁|^k where λ₁, λ₂ are largest eigenvalues
    - If λ₂/λ₁ ≈ 1: Slow convergence (many iterations)
    - If λ₂/λ₁ ≈ 0: Fast convergence (few iterations)
    """
    n = len(A)
    # Random start (likely has component along dominant eigenvector)
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    lambda_old = 0
    convergence_history = []
    
    for iteration in range(num_iterations):
        # The key step: multiply by A
        # After k steps: A^k v ≈ λ₁^k (c₁v₁ + (λ₂/λ₁)^k c₂v₂ + ...)
        # As k→∞, only v₁ term survives
        Av = A @ v
        
        # Normalize to prevent overflow/underflow
        v_new = Av / np.linalg.norm(Av)
        
        # Rayleigh quotient gives best eigenvalue estimate
        # For eigenvector: v^T A v / v^T v = λ
        lambda_est = v_new.T @ A @ v_new
        
        # Track convergence
        error = abs(lambda_est - lambda_old)
        convergence_history.append(error)
        
        if error < tolerance:
            print(f"Converged after {iteration} iterations")
            print(f"Convergence rate: {error/convergence_history[-2]:.3f}")
            break
            
        v = v_new
        lambda_old = lambda_est
    
    return lambda_est, v

# Example: Find dominant mode of coupled oscillators
# Mass-spring system: d²x/dt² = -Kx where K is stiffness matrix
K = np.array([[2, -1, 0],   # Spring between masses 1-2
              [-1, 3, -1],   # Springs on both sides of mass 2
              [0, -1, 2]])   # Spring between masses 2-3

lambda_max, v_max = power_method(K)
print(f"Highest frequency mode: ω = {np.sqrt(lambda_max):.3f}")
print(f"Mode shape: {v_max}")
# This finds the highest frequency oscillation pattern!

# Variant: Inverse power method finds SMALLEST eigenvalue
# Just apply power method to A^(-1)!
```

**Real Applications**:
- **PageRank**: Webpages ranked by dominant eigenvector of link matrix
- **PCA**: Principal component = dominant eigenvector of covariance
- **Stability**: Largest eigenvalue determines system stability
- **Quantum mechanics**: Ground state = smallest eigenvalue of Hamiltonian
:::

:::{admonition} 📌 Key Takeaway
:class: important

These advanced topics aren't just mathematical curiosities—they're the computational workhorses of modern astrophysics:

- **SVD** reveals hidden structure and enables massive data compression (galaxy spectra, CMB analysis)
- **Block matrices** exploit natural hierarchies to solve huge systems efficiently  
- **Jacobians** determine stability of everything from orbits to numerical methods
- **Matrix exponentials** solve linear evolution exactly—the foundation for understanding nonlinear dynamics
- **Matrix norms** quantify stability, convergence, and error propagation
- **Power method** underlies iterative algorithms from PageRank to finding vibrational modes

The beauty is that these tools interconnect: SVD uses eigenvalues, matrix exponentials need eigendecomposition, stability analysis uses Jacobians and norms together. Master these connections and you'll see how a small set of linear algebra concepts powers all of computational astrophysics.

As you progress through projects, return to these sections—concepts that seem abstract now will crystallize when you need them to debug unstable orbits or understand why your neural network won't converge.
:::

---

## Part 7: Numerical Reality - When Mathematics Meets Silicon

**Priority: 🔴 Essential** - Critical for debugging all projects

:::{admonition} 📅 When You'll Use This
:class: dropdown, hint

**Needed from**: Day 1 of coding
**Most critical for**: Debugging numerical errors
**Returns in**: Every project when things go wrong

Read this section early and refer back when debugging.
:::

:::{admonition} 🔭 Where We're Going
:class: note

Pure mathematics assumes infinite precision, but computers work with finite bits. This section reveals the harsh realities of floating-point arithmetic and teaches you to recognize and fix numerical disasters before they ruin your simulations. These aren't edge cases—you WILL encounter these issues in your projects.
:::

### 7.1 The Harsh Truth About Floating-Point Arithmetic {#floating-point-truth}

```{margin}
**floating-point**
Computer representation of real numbers with finite precision using scientific notation in binary
```

```{margin}
**machine epsilon**
Smallest distinguishable floating-point increment (~2.2×10⁻¹⁶ for float64); the gap between 1 and the next representable number
```

Computers can't store infinite decimals, so they approximate real numbers using a finite representation:
$$x = \pm m \times 2^e$$

where $m$ is the mantissa (fractional part) and $e$ is the exponent. Think of it as scientific notation in binary.

**What This Means for 64-bit Doubles**:
- ~16 decimal digits of precision (53 bits for mantissa)
- Largest number: ~10^308 (exponent can go up to 1023)
- Smallest positive normal: ~10^-308 (exponent down to -1022)
- **Machine epsilon**: ~2.2×10^-16 (smallest relative gap)

**The Shocking Truth About 0.1**:
```python
0.1 + 0.2 == 0.3  # False!
print(f"{0.1 + 0.2:.17f}")  # 0.30000000000000004

# Why? 0.1 in binary is actually:
# 0.00011001100110011001100110011... (repeating forever!)
# The computer truncates this at 53 bits, creating rounding error
```

Just like 1/3 = 0.333... repeats forever in decimal, 1/10 repeats forever in binary. The computer must truncate, introducing tiny errors that compound.

**Catastrophic Cancellation - The Silent Killer**:

When you subtract nearly equal numbers, you lose precision catastrophically. Here's why:

```python
# Example: Computing small angles in astronomy
star_position_1 = 89.99999999  # degrees (16 digits total)
star_position_2 = 90.00000001  # degrees (16 digits total)

angle_difference = star_position_2 - star_position_1
# Result: 0.00000002 (only 1 significant digit!)
# We went from 16 digits of precision to 1!

# Better approach: Reformulate to avoid subtraction
# Instead of (A + small) - A, compute small directly
```

**Real Project 2 Example - Gravitational Forces**:
```python
# BAD: Force between very close stars
def gravitational_force_bad(pos1, pos2, m1, m2):
    dx = pos2[0] - pos1[0]  # Catastrophic if positions nearly equal!
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    r_squared = dx**2 + dy**2 + dz**2  # Can become 0 or negative!
    return G * m1 * m2 / r_squared  # Division by ~0!

# GOOD: With softening parameter
def gravitational_force_good(pos1, pos2, m1, m2, epsilon=1e-4):
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    r_squared = dx**2 + dy**2 + dz**2 + epsilon**2  # Never zero!
    return G * m1 * m2 / r_squared
```

:::{admonition} 🔍 Understanding Floating-Point Spacing
:class: note

Floating-point numbers aren't evenly spaced! The gap between consecutive numbers grows with magnitude:

- Near 1: gap ≈ 2.2×10^-16
- Near 1000: gap ≈ 2.2×10^-13
- Near 10^6: gap ≈ 2.2×10^-10

This means:
```python
1.0 + 1e-16 == 1.0  # True! Gap too small to represent
1000.0 + 1e-13 == 1000.0  # True! 
1e6 + 1e-10 == 1e6  # True!
```

**Why this matters**: In Project 2, if your star cluster has positions at ~10^6 cm from origin, you can't resolve position changes smaller than 10^-10 cm. This affects your timestep choice and integration accuracy!
:::

### 7.2 Condition Numbers: Measuring Numerical Danger {#condition-numbers}

```{margin}
**condition number**
κ(A) = σ_max/σ_min, ratio of largest to smallest singular value; measures how much errors amplify
```

The condition number tells you how much input errors get amplified:

$$\frac{||\delta x||}{||x||} \leq \kappa(A) \frac{||\delta b||}{||b||}$$

This means: relative error in solution ≤ condition number × relative error in input.

**Intuitive Understanding**:
- κ = 1: Perfect! Errors don't amplify
- κ = 100: 1% input error → up to 100% output error
- κ = 10^6: Lose 6 digits of accuracy
- κ = 10^16: Complete garbage (for float64)

| κ(A) | Interpretation | What It Means for You |
|------|----------------|----------------------|
| < 10 | Excellent | Trust your results |
| 10-100 | Good | Minor accuracy loss |
| 100-1000 | Acceptable | Check residuals |
| 10^3-10^6 | Problematic | Need careful algorithms |
| > 10^6 | Dangerous | Consider reformulation |
| > 10^10 | Numerically singular | Add regularization |
| ≈ 10^16 | Complete failure | Matrix is effectively rank-deficient |

**Real Example - Why Your Stellar Population Fits Fail**:
```python
# Fitting mass-luminosity relation: L = a*M^b
# Taking logs: log(L) = log(a) + b*log(M)

# Design matrix for 3 stars with similar masses
M = np.array([0.99, 1.00, 1.01])  # Solar masses
X = np.column_stack([np.ones(3), np.log(M)])

# Check condition number
print(f"Condition number: {np.linalg.cond(X):.1e}")
# Output: 4.5e+03 (problematic!)

# Why? The log(M) values are nearly identical!
# log(0.99) ≈ -0.0101, log(1.00) = 0, log(1.01) ≈ 0.0100
# The columns are nearly parallel → high condition number

# Fix: Use centered/scaled variables
M_scaled = (M - M.mean()) / M.std()
X_better = np.column_stack([np.ones(3), M_scaled])
print(f"Better condition number: {np.linalg.cond(X_better):.1e}")
# Output: 2.4e+00 (excellent!)
```

### 7.3 When Linear Algebra Fails - And How to Fix It

Understanding failure modes helps you recognize and fix problems before they crash your code:

**Singular Matrix - Information Lost**:
```python
# What happens: You're trying to solve an impossible system
A = np.array([[1, 2], [2, 4]])  # Second row = 2 × first row
b = np.array([3, 7])  # Not consistent with row relationship!

# This system says: x + 2y = 3 AND 2x + 4y = 7
# But the second equation is just 2× the first, so it should give 6, not 7!

try:
    x = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    print("Singular matrix - no unique solution exists!")
    
    # Fix 1: Use least squares (finds best approximate solution)
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print(f"Least squares solution: {x}")
    print(f"Matrix rank: {rank} (should be 2 for 2×2)")
    
    # Fix 2: Add regularization (Tikhonov/Ridge regression)
    lambda_reg = 1e-3
    A_reg = A.T @ A + lambda_reg * np.eye(2)
    b_reg = A.T @ b
    x_reg = np.linalg.solve(A_reg, b_reg)
```

**Nearly Defective Matrices - Eigenvector Chaos**:
```python
# Jordan block - eigenvalues equal but not enough eigenvectors
A = np.array([[2, 1], 
              [0, 2]])  # Both eigenvalues = 2

# Tiny perturbation causes huge eigenvector changes
A_perturbed = A + 1e-10 * np.random.randn(2, 2)

eigvals1, eigvecs1 = np.linalg.eig(A)
eigvals2, eigvecs2 = np.linalg.eig(A_perturbed)

print(f"Eigenvalue change: {np.abs(eigvals1 - eigvals2).max():.2e}")
print(f"Eigenvector angle change: {np.arccos(np.abs(eigvecs1[:,0] @ eigvecs2[:,0])):.2f} radians")
# Tiny eigenvalue change but large eigenvector change!

# Why this matters: In Project 2, nearly equal eigenvalues mean
# nearly degenerate orbits - small perturbations cause large changes
```

**Loss of Positive Definiteness - The GP Killer**:
```python
# Start with a perfectly good covariance matrix
n = 100
K = np.exp(-0.5 * np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])**2)
print(f"Smallest eigenvalue: {np.linalg.eigvalsh(K).min():.2e}")  # Positive

# After some computations with rounding errors...
K_computed = K @ np.eye(n) @ K / K  # Mathematically = K, but...
print(f"Smallest eigenvalue after ops: {np.linalg.eigvalsh(K_computed).min():.2e}")
# Might be negative due to accumulating rounding errors!

# The Cholesky decomposition will fail:
try:
    L = np.linalg.cholesky(K_computed)
except np.linalg.LinAlgError:
    print("Not positive definite!")
    
    # Fix: Force positive definiteness
    eigvals, eigvecs = np.linalg.eigh(K_computed)
    eigvals = np.maximum(eigvals, 1e-10)  # Threshold negative values
    K_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L = np.linalg.cholesky(K_fixed)  # Now works!
```

### 7.4 Speed vs. Accuracy - The Eternal Tradeoff

:::{admonition} 💻 Never Invert - See The Proof!
:class: warning

Here's definitive proof that computing inverses is both slower AND less accurate:

```python
import numpy as np
import time

# Create test problem
np.random.seed(42)
n = 1000
A = np.random.randn(n, n)
A = A.T @ A + np.eye(n)  # Ensure positive definite with good conditioning
b = np.random.randn(n)

# Method 1: Explicit inverse (NEVER DO THIS)
start = time.time()
A_inv = np.linalg.inv(A)  # O(n³) operations
x_inv = A_inv @ b  # O(n²) operations
time_inv = time.time() - start

# Method 2: Direct solve (ALWAYS DO THIS)
start = time.time()
x_solve = np.linalg.solve(A, b)  # O(n³/3) operations via LU
time_solve = time.time() - start

# Method 3: For multiple right-hand sides
B = np.random.randn(n, 5)  # 5 different b vectors

start = time.time()
X_inv = A_inv @ B  # If you already computed inverse
time_inv_multi = time.time() - start

start = time.time()
X_solve = np.linalg.solve(A, B)  # Still better!
time_solve_multi = time.time() - start

# Compare accuracy
residual_inv = np.linalg.norm(A @ x_inv - b) / np.linalg.norm(b)
residual_solve = np.linalg.norm(A @ x_solve - b) / np.linalg.norm(b)

print(f"Single RHS:")
print(f"  Inverse method: {time_inv:.3f}s, relative error: {residual_inv:.2e}")
print(f"  Solve method:   {time_solve:.3f}s, relative error: {residual_solve:.2e}")
print(f"  Speedup: {time_inv/time_solve:.1f}×, Accuracy gain: {residual_inv/residual_solve:.1f}×")

print(f"\nMultiple RHS:")
print(f"  Inverse method: {time_inv_multi:.3f}s")
print(f"  Solve method:   {time_solve_multi:.3f}s")
print(f"  Even with pre-computed inverse, solve is {time_inv_multi/time_solve_multi:.1f}× faster!")

# Typical output:
# Single RHS:
#   Inverse method: 0.523s, relative error: 8.7e-13
#   Solve method:   0.174s, relative error: 3.1e-14
#   Speedup: 3.0×, Accuracy gain: 28.1×
```

**Why solve is better**:
1. **Fewer operations**: LU decomposition ≈ n³/3 ops vs n³ for inverse
2. **Better stability**: Forward/back substitution accumulates less error
3. **Preserves structure**: Exploits symmetry, bandedness, sparsity
4. **Memory efficient**: No need to store n×n inverse matrix
:::

### 7.5 Troubleshooting Guide - What to Do When Things Break

:::{admonition} 🔧 Common Linear Algebra Problems and Solutions
:class: important

**Problem: "Matrix is singular to working precision"**
- **Symptom**: `np.linalg.solve()` crashes
- **Diagnosis**: Check condition number and rank
  ```python
  print(f"Condition number: {np.linalg.cond(A):.2e}")
  print(f"Rank: {np.linalg.matrix_rank(A)} / {A.shape[0]}")
  ```
- **Fix Options**:
  1. Regularization: `A_reg = A + 1e-6 * np.eye(n)`
  2. Pseudoinverse: `x = np.linalg.pinv(A) @ b`
  3. Least squares: `x = np.linalg.lstsq(A, b, rcond=None)[0]`

**Problem: "Eigenvalues should be real but are complex"**
- **Symptom**: Getting complex eigenvalues for physical system
- **Diagnosis**: Matrix isn't perfectly symmetric due to rounding
  ```python
  asymmetry = np.max(np.abs(A - A.T))
  print(f"Max asymmetry: {asymmetry:.2e}")
  ```
- **Fix**: Force symmetry
  ```python
  A_sym = (A + A.T) / 2
  eigvals = np.linalg.eigvalsh(A_sym)  # For symmetric matrices
  ```

**Problem: "Cholesky decomposition failed" (Project 5 nightmare!)**
- **Symptom**: `np.linalg.cholesky()` raises LinAlgError
- **Diagnosis**: Check minimum eigenvalue
  ```python
  min_eig = np.linalg.eigvalsh(K).min()
  print(f"Min eigenvalue: {min_eig:.2e}")
  if min_eig < 0:
      print(f"Matrix is not positive definite!")
  ```
- **Fix Hierarchy** (try in order):
  ```python
  # Fix 1: Add small jitter
  try:
      L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
  except:
      # Fix 2: Eigenvalue thresholding
      eigvals, eigvecs = np.linalg.eigh(K)
      eigvals = np.maximum(eigvals, 1e-10)
      K_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
      L = np.linalg.cholesky(K_fixed)
  ```

**Problem: "Solution to Ax=b seems wrong"**
- **Symptom**: Large residuals or unphysical results
- **Diagnosis**: Check conditioning and residuals
  ```python
  x = np.linalg.solve(A, b)
  residual = np.linalg.norm(A @ x - b)
  relative_residual = residual / np.linalg.norm(b)
  print(f"Relative residual: {relative_residual:.2e}")
  print(f"Condition number: {np.linalg.cond(A):.2e}")
  ```
- **Fix**: Use SVD for robust solution
  ```python
  U, s, Vt = np.linalg.svd(A, full_matrices=False)
  # Truncate small singular values (regularization)
  threshold = 1e-10 * s.max()
  s_inv = np.where(s > threshold, 1/s, 0)
  x = Vt.T @ (s_inv * (U.T @ b))
  ```

**Problem: "Matrix operations are too slow"**
- **Diagnosis**: Check matrix structure
  ```python
  sparsity = np.count_nonzero(A) / A.size
  print(f"Sparsity: {sparsity:.1%}")
  ```
- **Fix Options**:
  1. **Sparse matrices** (if <10% non-zero):
     ```python
     from scipy.sparse import csr_matrix
     from scipy.sparse.linalg import spsolve
     A_sparse = csr_matrix(A)
     x = spsolve(A_sparse, b)
     ```
  2. **Iterative solvers** (for large systems):
     ```python
     from scipy.sparse.linalg import cg  # Conjugate gradient
     x, info = cg(A, b, tol=1e-6)
     ```
  3. **Exploit structure**: Use specialized solvers for banded, Toeplitz, etc.
:::

:::{admonition} 🎯 Project-Specific Numerical Gotchas
:class: warning

**Project 1**: Fitting power laws to IMF
- Log-transform creates infinities for zero masses
- Fix: Add small offset or use robust fitting

**Project 2**: N-body integration
- Close encounters cause force singularities
- Fix: Gravitational softening parameter

**Project 3**: Monte Carlo sampling
- Random numbers in [0,1) can give log(0) = -∞
- Fix: Use (0,1] or add tiny epsilon

**Project 4**: MCMC
- Proposal covariance can lose positive definiteness
- Fix: Adaptive regularization during burn-in

**Project 5**: Gaussian Processes
- Kernel matrices become ill-conditioned for close points
- Fix: Jitter + careful hyperparameter bounds

**Final Project**: Neural Networks
- Gradient explosion/vanishing from poor initialization
- Fix: Xavier/He initialization + gradient clipping
:::

:::{admonition} 📌 Key Takeaway
:class: important

Computers aren't mathematical ideals—they're finite machines with finite precision. Every number is approximate, every operation loses accuracy, and errors compound rapidly. But with awareness and proper techniques, you can write robust code that handles numerical reality gracefully.

Remember:
- Check condition numbers before trusting results
- Never compute explicit inverses
- Add regularization when matrices are near-singular
- Test your code with pathological cases
- When in doubt, use SVD—it's the most stable decomposition

These aren't edge cases you might encounter—these are daily realities in computational astrophysics. Master them now, and save yourself countless debugging hours later!
:::

---

## Part 8: The Bridge to Machine Learning and Beyond

**Priority: 🟡 Important** - Helps see the big picture

:::{admonition} 📅 When You'll Use This
:class: dropdown, hint

Read this section when transitioning between course phases to understand connections. Return to it after Projects 1-3 to see how classical methods connect to modern ML.
:::

### 8.1 From Classical to Statistical to Learning

Linear algebra provides the mathematical continuity across your entire journey. Watch how the same mathematical objects evolve in meaning as you progress through different computational paradigms:

**The Evolution of Mathematical Objects**:

| Mathematical Object | Classical Physics (Projects 1-3) | Statistical Methods (Projects 4-5) | Machine Learning (Final Project) |
|-------------------|----------------------------------|-------------------------------------|----------------------------------|
| **Vectors** | Positions, velocities, forces | Parameter samples, data points | Feature vectors, gradients |
| **Matrices** | Transformations, rotations | Covariance, kernels | Weight matrices, Jacobians |
| **Eigenvalues** | Stability, oscillation modes | Convergence rates, principal components | Learning rates, network dynamics |
| **Dot products** | Work, projections | Correlations, similarities | Attention scores, kernels |
| **Norms** | Distances, magnitudes | Errors, uncertainties | Loss functions, regularization |
| **Decompositions** | Solving dynamics | Statistical inference | Network compression, analysis |

This isn't coincidence—it's the deep unity of mathematics. The same linear algebra that predicts planetary orbits also powers Google's search algorithm and ChatGPT's language understanding.

**The Conceptual Journey**:

You begin with **deterministic systems** where vectors represent physical states and matrices transform them according to Newton's laws. Everything is precise, predictable, reversible.

Then you encounter **statistical methods** where the same vectors now represent samples from probability distributions, and matrices encode correlations and uncertainties. Randomness enters, but patterns emerge from chaos through the law of large numbers.

Finally, you reach **machine learning** where vectors become learned representations, matrices become trainable parameters, and the same mathematical operations now extract patterns from data rather than evolving physical systems. The mathematics remains constant; only our interpretation evolves.

### 8.2 Linear Algebra in Modern Astronomy

Modern astronomy wouldn't exist without sophisticated linear algebra. Here are three revolutionary discoveries that depended on the techniques you're learning:

**LIGO's Gravitational Wave Detection (2015)**:

When LIGO detected gravitational waves from merging black holes, the signal was buried in noise 1000× stronger. The detection required:
- **Matched filtering**: Computing $(d|h) = \int \frac{d^*(f) h(f)}{S_n(f)} df$ where the noise covariance $S_n(f)$ weights the integral
- **SVD for noise reduction**: Separating instrumental noise modes from signal
- **Eigenanalysis of correlation matrices**: Identifying coherent signals across detectors

Without efficient matrix operations, we'd still be searching for that first "chirp" that confirmed Einstein's century-old prediction.

**The Event Horizon Telescope's Black Hole Image (2019)**:

Creating the first image of a black hole's event horizon required solving an underdetermined inverse problem—reconstructing an image from incomplete interferometric data:

- **Regularized least squares**: $\min ||V\vec{x} - \vec{d}||^2 + \lambda R(\vec{x})$ where $V$ is the visibility matrix
- **Compressed sensing**: Exploiting sparsity in wavelet bases
- **Maximum entropy methods**: Choosing the least-biased image consistent with data

*The linear algebra you're learning literally made black holes visible!*

**Machine Learning Discovers New Exoplanets (Ongoing)**:

The Kepler and TESS missions generated more data than humans could analyze. Machine learning now finds planets we missed:

- **PCA for systematic noise removal**: Separating instrumental effects from transit signals
- **Neural networks for classification**: Weight matrices learning to recognize planet signatures
- **Gaussian Processes for stellar variability**: Modeling star spots to reveal hidden planets

Your final project connects directly to this frontier—the same neural network architecture you'll build is finding new worlds.

### 8.3 Big Picture: Linear Algebra Across Computational Astrophysics

The techniques in this module power every major simulation in modern astrophysics. Understanding these connections helps you see why mastering linear algebra opens doors across the entire field:

**Magnetohydrodynamics (MHD) - The Physics of Cosmic Plasmas**:

From solar flares to accretion disks, MHD simulations model how magnetic fields interact with flowing plasma. The magnetic field evolution follows:
$$\frac{\partial \vec{B}}{\partial t} = \nabla \times (\vec{v} \times \vec{B}) + \eta \nabla^2 \vec{B}$$

Discretizing this gives a matrix equation where eigenvalues determine wave speeds (Alfvén, fast/slow magnetosonic) and stability. When eigenvalues of the linearized MHD operator have positive real parts, magnetic instabilities grow—this is how we predict solar flares and understand jet formation in black hole accretion!

**Cosmological Structure Formation - The Universe's Web**:

Simulating how dark matter halos and galaxies form requires following billions of particles. The key is the **tidal tensor**:
$$T_{ij} = \frac{\partial^2 \Phi}{\partial x_i \partial x_j}$$

Its eigenvalues classify the local geometry:
- 3 positive eigenvalues → void (expansion in all directions)
- 2 positive, 1 negative → sheet/wall
- 1 positive, 2 negative → filament  
- 3 negative → halo (collapse in all directions)

The Cosmic Web's structure—the largest pattern in the universe—emerges from the eigenvalues of 3×3 matrices computed at each point!

**Adaptive Optics - Fixing Atmospheric Blur**:

Ground-based telescopes use deformable mirrors to correct atmospheric turbulence in real-time. The control system solves:
$$\vec{a} = R^{-1} \vec{s}$$

where $\vec{s}$ are wavefront sensor measurements, $R$ is the response matrix, and $\vec{a}$ are actuator commands. This happens 1000× per second! The SVD of $R$ reveals which aberration modes can be corrected and which are lost to noise. Your linear algebra literally sharpens our view of the cosmos.

### 8.4 Preview: Neural Networks as Matrix Compositions

Your final project culminates in building a neural network from scratch. Here's how everything you've learned comes together:

A neural network is fundamentally a composition of linear transformations (matrices) and non-linear activations:

$$\vec{y} = f_L(W_L f_{L-1}(W_{L-1} \cdots f_1(W_1 \vec{x}) \cdots ))$$

**Each Layer Broken Down**:
1. **Linear transformation**: $\vec{z} = W\vec{x} + \vec{b}$ (matrix multiply + bias)
2. **Non-linear activation**: $\vec{a} = f(\vec{z})$ (e.g., ReLU, sigmoid)
3. **Forward propagation**: Compose layers to get output
4. **Backpropagation**: Chain rule through the composition

**Critical Insight**: Without non-linear activations, deep networks collapse to a single matrix:
$$W_L \cdot W_{L-1} \cdots W_1 = W_{\text{effective}}$$

The non-linearities are what allow neural networks to learn complex, non-linear patterns!

**Connecting to Your Physics Background**:

The mathematics of neural network training is remarkably similar to physical systems you understand:
- **Gradient descent** = Following force fields to minimum energy
- **Loss landscape** = Potential energy surface
- **Learning rate** = Timestep in numerical integration
- **Momentum in SGD** = Actual momentum in dynamics
- **Batch normalization** = Maintaining numerical stability

When you train a neural network, you're essentially simulating a particle rolling down a high-dimensional potential energy surface, seeking the global minimum. The same intuition from Project 2's N-body dynamics applies!

:::{admonition} 🎯 The Big Picture
:class: important

Every major computational achievement in astrophysics relies on linear algebra:

- **Gaia**: 1 billion stars → 6 billion phase space coordinates → massive eigenproblems
- **LIGO**: Gravitational waves → matched filtering → matrix operations on strain data
- **Event Horizon Telescope**: Sparse interferometry → regularized inversion → black hole images
- **JWST**: Spectroscopy → matrix decomposition → atmospheric composition of exoplanets
- **Vera Rubin Observatory**: ~20 TB/night → PCA/ML classification → discovering the unexpected

The linear algebra you master here isn't academic exercise—it's the foundation of modern astronomical discovery. Every breakthrough in the next decade will build on these mathematical tools.

When you struggle with eigenvalues or matrix decompositions, remember: you're learning the same mathematics that detected gravitational waves, imaged black holes, and will find signs of life on distant worlds. Master these tools, and you join humanity's quest to understand the cosmos!
:::

:::{tip} 🌟 The More You Know: How Least Squares Found the First Asteroid
:class: dropdown

**Priority: 🟢 Enrichment**

On January 1, 1801, Giuseppe Piazzi discovered Ceres—the first asteroid—but it vanished behind the Sun after just 41 days of observation. The astronomical community faced a crisis: was humanity's first asteroid lost forever?

Carl Friedrich Gauss, then just 24, invented the method of least squares specifically to solve this problem. Working with only 41 noisy position measurements, he needed to determine Ceres' orbital elements (6 parameters defining the ellipse).

The problem was overdetermined (41 observations, 6 unknowns) and the observations contained errors. Gauss's brilliant insight was to minimize the sum of squared residuals:

$$S = \sum_{i=1}^{41} (\text{observed}_i - \text{predicted}_i)^2$$

Taking derivatives and setting them to zero yields the famous normal equations:
$$A^T A \vec{x} = A^T \vec{b}$$

where $A$ encodes the orbital mechanics and $\vec{x}$ contains the orbital elements.

Using only paper and pen, Gauss spent weeks solving this system by hand. His prediction: Ceres would reappear at a specific position on December 31, 1801.

On that exact date — one year after discovery — astronomer Franz von Zach pointed his telescope to Gauss's predicted position. There was Ceres, within 0.5° of the prediction!

This triumph didn't just recover one asteroid — it established least squares as fundamental to all science. Every GPS satellite, every exoplanet discovery, every machine learning model traces back to that moment when linear algebra rescued Ceres from the void.

*Source: Teets, D. A., & Whitehead, K. (1999). "The Discovery of Ceres: How Gauss Became Famous." Mathematics Magazine, 72(2), 83-93.*
:::

::::{admonition} ✅ Test Your Understanding: The Complete Picture
:class: hint

Can you answer these synthesis questions that connect the entire module?

1. How do conservation laws connect to matrix properties?
2. Why must covariance matrices be positive semi-definite?
3. How do eigenvalues determine both physical stability and algorithm convergence?
4. Why does the same math describe classical orbits and neural network training?

Think deeply about these connections before checking the answers—they reveal the profound unity underlying computational astrophysics.

:::{tip} Answers
:class: dropdown

1. **Conservation laws arise from symmetries** (Noether's theorem). These symmetries manifest as matrix properties:
   - Energy conservation → symplectic structure (preserves phase space volume, det=1)
   - Momentum conservation → translation invariance
   - Angular momentum → rotational symmetry (orthogonal transformations)

   **The mathematics enforces the physics:** symplectic integrators conserve energy not by accident but because they preserve the mathematical structure encoding time-translation symmetry.

2. **Covariance matrices represent squared deviations**, and squares cannot be negative. Mathematically, for any linear combination $\vec{y} = \vec{a}^T\vec{x}$:
   $$\text{Var}(\vec{y}) = \vec{a}^T\Sigma\vec{a} \geq 0$$

   If $\Sigma$ had negative eigenvalues, we could construct a linear combination with negative variance—physically impossible! The mathematics protects physical reality.

3. **Eigenvalues determine exponential growth/decay rates**. For linear system $\dot{\vec{x}} = A\vec{x}$:
   $$\vec{x}(t) = e^{At}\vec{x}(0) = \sum_i c_i e^{\lambda_i t}\vec{v}_i$$

   - Physics: $\lambda < 0$ → stable (perturbations decay)
   - Algorithms: Convergence rate ∝ $|\lambda_2/\lambda_1|^k$ (ratio of second-largest to largest)

   It's the same exponential mathematics whether describing orbital stability or MCMC convergence!

4. **Both are optimization problems in high-dimensional spaces**:
   - Classical mechanics: Minimize action $S = \int L \, dt$ where $L = T - V$
   - Neural networks: Minimize loss $\mathcal{L}(\theta)$ over parameters $\theta$

   Both involve:
   - Following gradients (forces in physics, loss gradients in ML)
   - Navigating saddle points (unstable equilibria vs. optimization challenges)
   - Using momentum (physical momentum vs. momentum in SGD)
   - Finding stable minima (bound orbits vs. good parameter regions)

   The optimization landscape's eigenvalues (Hessian) determine behavior in both cases!
:::
::::

:::{admonition} 📌 Module Summary: The Power of Linear Algebra
:class: important

You've journeyed from vectors and matrices through eigenvalues and decompositions to numerical reality and modern applications. The key insight: linear algebra isn't just mathematical machinery—it's the language of physical reality at computational scales.

**What you've mastered**:
- Vectors encode any quantity with direction and magnitude
- Matrices transform, rotate, and scale entire vector spaces
- Eigenvalues reveal hidden stability and dominant modes
- Decompositions solve systems and reveal structure
- Numerical awareness prevents computational disasters
- The same mathematics powers physics and machine learning

**Why it matters**:
Every computational method and machine learning algorithm in astrophysics reduces to linear algebra. Master these foundations and you can:

- Simulate million-body star clusters
- Extract signals from noise
- Infer parameters from data
- Train neural networks
- Contribute to the next astronomical breakthrough

Linear algebra is your passport to computational astrophysics and machine learning. Use it wisely, and the universe's secrets await!
:::

---

## Main Takeaways

:::{admonition} 🎯 Essential Points to Remember
:class: important

Linear algebra is the mathematical infrastructure of computational astrophysics. Here are the key insights to remember:

1. **Vectors encode physical quantities** with magnitude and direction, existing independently of coordinate systems

2. **Matrices are transformations**, not just number grids—they preserve linear structure while changing representations

3. **Eigenvalues and eigenvectors reveal invariant properties** that determine stability, convergence, and fundamental modes

4. **Positive definiteness ensures physical reality** in energy, distance, and probability

5. **Numerical precision limits require careful algorithm design**—use decompositions, never compute inverses

6. **Conservation laws manifest as mathematical structures**—symmetries become properties of vectors and matrices

7. **The same mathematics spans classical, statistical, and machine learning domains**—linear algebra unifies all
:::

---

## Essential SciPy Linear Algebra Reference {#essential-scipy-reference}

**Priority: 🔴 Essential** - Keep this open while coding

```{list-table} Quick Function Reference
:header-rows: 1

* - Task
  - Function
  - When to Use
  - Numerical Stability
* - Solve $A\vec{x}=\vec{b}$
  - `scipy.linalg.solve(A, b)`
  - General square systems
  - ⭐⭐⭐⭐
* - Solve positive definite
  - `scipy.linalg.cho_solve()`
  - Covariance matrices
  - ⭐⭐⭐⭐⭐
* - Least squares
  - `scipy.linalg.lstsq()`
  - Overdetermined systems
  - ⭐⭐⭐⭐⭐
* - Eigenvalues (general)
  - `scipy.linalg.eig()`
  - Non-symmetric matrices
  - ⭐⭐⭐
* - Eigenvalues (symmetric)
  - `scipy.linalg.eigh()`
  - Symmetric/Hermitian
  - ⭐⭐⭐⭐⭐
* - SVD
  - `scipy.linalg.svd()`
  - Any matrix, rank
  - ⭐⭐⭐⭐⭐
* - Matrix exponential
  - `scipy.linalg.expm()`
  - Time evolution
  - ⭐⭐⭐⭐
* - Condition number
  - `numpy.linalg.cond()`
  - Check stability
  - N/A
```

---

## One-Page Cheat Sheet

:::{admonition} 📋 Linear Algebra Quick Reference
:class: tip

### Essential Formulas

**Vector Operations**
- Dot product: $\vec{a} \cdot \vec{b} = \sum_i a_i b_i = |\vec{a}||\vec{b}|\cos\theta$
- Cross product magnitude: $|\vec{a} \times \vec{b}| = |\vec{a}||\vec{b}|\sin\theta$
- Vector norm: $|\vec{v}| = \sqrt{\vec{v} \cdot \vec{v}}$

**Matrix Operations**
- Matrix multiplication: $(AB)_{ij} = \sum_k A_{ik}B_{kj}$
- Transpose: $(A^T)_{ij} = A_{ji}$
- Inverse (2×2): $\begin{pmatrix}a&b\\c&d\end{pmatrix}^{-1} = \frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}$
- Determinant (2×2): $\det\begin{pmatrix}a&b\\c&d\end{pmatrix} = ad-bc$

**Eigenvalues & Eigenvectors**
- Definition: $A\vec{v} = \lambda\vec{v}$
- Characteristic equation: $\det(A - \lambda I) = 0$
- For symmetric matrices: real eigenvalues, orthogonal eigenvectors

**Positive Definite Tests**
1. All eigenvalues > 0
2. All leading principal minors > 0
3. Has Cholesky decomposition $A = LL^T$ (strict positive definiteness required)
4. Quadratic form $\vec{x}^T A \vec{x} > 0$ for all $\vec{x} \neq \vec{0}$

### Critical NumPy/SciPy Commands

| Task | Command | When to Use |
|------|---------|-------------|
| Solve $A\vec{x}=\vec{b}$ | `np.linalg.solve(A, b)` | Linear systems |
| Eigenvalues (symmetric) | `np.linalg.eigh(A)` | Covariance, kernels |
| Cholesky | `np.linalg.cholesky(A)` | Positive definite matrices |
| Condition number | `np.linalg.cond(A)` | Check numerical stability |
| Pseudoinverse | `np.linalg.pinv(A)` | Rank-deficient systems |

### Numerical Stability Rules

1. **Never invert matrices**: Use `solve()` not `inv()`
2. **Check condition numbers**: κ > 10^6 means trouble
3. **Add jitter for stability**: `A_stable = A + 1e-6 * I`
4. **Symmetrize when needed**: `A_sym = (A + A.T) / 2`
5. **Use appropriate precision**: float64 for science, float32 for ML

### Physical Interpretations

| Mathematical Object | Physical Meaning |
|-------------------|------------------|
| Dot product | Work done, projection |
| Cross product | Angular momentum, torque |
| Eigenvalues | Natural frequencies, stability |
| Eigenvectors | Principal axes, normal modes |
| Determinant | Volume scaling, invertibility |
| Symmetric matrix | Reciprocal relationships |
| Positive definite | Valid energy/distance/probability |
| Orthogonal matrix | Rotation, preserves lengths |

### Key Insights by Project

**Project 1**: Vectorization = matrix operations on entire arrays  
**Project 2**: Conservation laws = preserved matrix properties  
**Project 3**: Scattering = transformation matrices  
**Project 4**: MCMC convergence = second eigenvalue  
**Project 5**: GP kernels must be positive definite  
**Final Project**: Neural networks = composed matrix transformations  

### Emergency Debugging

```python
# Matrix is singular or near-singular
if np.linalg.cond(A) > 1e10:
    A = A + 1e-6 * np.eye(len(A))  # Add regularization

# Eigenvalues should be real but aren't
if not np.allclose(A, A.T):
    A = (A + A.T) / 2  # Force symmetry

# Cholesky fails on "positive definite" matrix
try:
    L = np.linalg.cholesky(K)
except np.linalg.LinAlgError:
    # Fix eigenvalues
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 1e-6)
    K = eigvecs @ np.diag(eigvals) @ eigvecs.T
```
:::

---

## Glossary

**basis**: A set of linearly independent vectors that span a vector space; any vector can be uniquely expressed as their linear combination

**block matrix**: A matrix partitioned into submatrices, often exploiting structure for efficient computation

**characteristic equation**: The polynomial equation $\det(A - \lambda I) = 0$ whose roots are the eigenvalues of matrix $A$

**Cholesky decomposition**: Factorization of a positive definite matrix as $A = LL^T$ where $L$ is lower triangular with positive diagonal

**condition number**: The ratio $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ measuring how much a matrix amplifies errors

**conservation law**: A physical quantity that remains constant over time due to an underlying symmetry of the system

**covariance matrix**: Symmetric positive semi-definite matrix containing all pairwise covariances between random variables

**cross product**: Vector operation $\vec{a} \times \vec{b}$ producing a perpendicular vector with magnitude equal to the parallelogram area

**determinant**: Scalar value measuring how a linear transformation scales volumes; zero indicates the transformation is singular

**dot product**: Scalar operation $\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\theta$ measuring projection and alignment between vectors

**eigenvalue**: Scalar $\lambda$ for which there exists a non-zero vector $\vec{v}$ satisfying $A\vec{v} = \lambda\vec{v}$

**eigenvector**: Non-zero vector that is only scaled (not rotated) by a linear transformation

**floating-point**: Computer representation of real numbers using finite precision (typically 64 bits)

**Frobenius norm**: Matrix norm $||A||_F = \sqrt{\sum_{i,j} |a_{ij}|^2}$ analogous to vector 2-norm

**globular cluster**: Spherical collection of 10⁴ to 10⁶ stars gravitationally bound together, orbiting as satellites of galaxies

**Hermitian**: Complex analog of symmetric matrices; satisfies $A = A^*$ (conjugate transpose)

**Jacobian**: Matrix of all first-order partial derivatives of a vector-valued function

**linear algebra**: Branch of mathematics concerning vector spaces and linear transformations between them

**linear independence**: Vectors are linearly independent if the only linear combination equaling zero has all coefficients zero

**linear transformation**: Function between vector spaces preserving addition and scalar multiplication

**machine epsilon**: Smallest floating-point number ε where 1 + ε ≠ 1 in computer arithmetic (~2.2×10⁻¹⁶ for doubles)

**Mahalanobis distance**: Scale-invariant statistical distance $(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})$ accounting for correlations

**matrix**: Rectangular array representing a linear transformation; columns show where basis vectors map

**matrix exponential**: The series $e^A = I + A + A^2/2! + ...$, used for solving linear differential equations

**matrix inverse**: The transformation $A^{-1}$ that undoes $A$: $A^{-1}A = AA^{-1} = I$

**matrix multiplication**: Operation combining transformations: $(AB)\vec{v} = A(B\vec{v})$

**multivariate Gaussian**: Multi-dimensional probability distribution defined by mean vector $\vec{\mu}$ and covariance matrix $\Sigma$

**norm**: Length of a vector: $||\vec{v}|| = \sqrt{\vec{v} \cdot \vec{v}}$

**orthogonal**: Perpendicular; vectors with zero dot product

**orthogonal matrix**: Matrix preserving lengths and angles, satisfying $Q^TQ = QQ^T = I$

**phase space**: Space of all possible states; for N particles in 3D, has 6N dimensions (3 position + 3 velocity per particle)

**positive definite**: Symmetric matrix with all positive eigenvalues, ensuring $\vec{x}^TA\vec{x} > 0$ for all $\vec{x} \neq \vec{0}$

**quadratic form**: Expression $\vec{x}^TA\vec{x}$ where $A$ is symmetric, representing energy, distance, or similar quantities

**rank**: Number of linearly independent rows or columns in a matrix

**regularization**: Adding small values to improve numerical stability, often to matrix diagonals

**singular value decomposition (SVD)**: Universal factorization $A = U\Sigma V^T$ revealing geometric structure of any matrix

**spectral norm**: Matrix norm $||A||_2 = \sigma_{\max}$, the largest singular value

**spectral theorem**: Theorem guaranteeing that symmetric matrices have real eigenvalues and orthogonal eigenvectors

**symmetric matrix**: Matrix equal to its transpose ($A = A^T$), representing order-independent relationships

**symplectic**: Transformation preserving phase space volume; has determinant = 1

**trace**: Sum of diagonal elements of a matrix; equals sum of eigenvalues

**trajectory**: Path traced by a system's state through phase space as it evolves in time

**unitary**: Complex analog of orthogonal matrices; satisfies $U^*U = UU^* = I$

**vector**: Mathematical object with magnitude and direction that transforms according to specific rules

**vector space**: Set equipped with addition and scalar multiplication satisfying eight fundamental axioms

---

## Next Steps

With this mathematical foundation firmly in place, you're ready to tackle the computational challenges of modern astrophysics. In Project 1, you'll immediately apply vector operations and matrix manipulations to handle stellar populations efficiently. In Project 2, you'll see how eigenvalues determine orbital stability. By Project 5, you'll understand why Gaussian Process kernel matrices must be positive definite.

**Remember:** every algorithm you implement ultimately reduces to linear algebra. When numerical issues arise — and they will — return to these foundations. Check condition numbers, verify positive definiteness, use appropriate decompositions. The mathematics you've learned here isn't separate from computation — it IS computation in its purest form.

---

*Welcome to computational astrophysics. You now speak its language.*
