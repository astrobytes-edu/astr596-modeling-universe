---
title: "Module 0b: Statistical Linear Algebra and Numerical Methods"
subtitle: "Advanced Mathematical Foundations for Computational Astrophysics | ASTR 596"
exports:
  - format: pdf
---

## Quick Navigation Guide

### 🔍 Choose Your Learning Path

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🏃 **Fast Track**
Starting Project 4? Read sections marked 🔴

- [Positive Definiteness](#part-5-positive-definite)
- [Covariance Matrices](#covariance-matrices)
- [Numerical Reality](#part-7-numerical-reality)
- [Quick Reference](#quickref)
:::

:::{grid-item-card} 🚶 **Standard Path**
Preparing for Projects 4-5? Read 🔴 and 🟡

- Everything in Fast Track, plus:
- [Multivariate Gaussian](#multivariate-gaussian)
- [Cholesky Decomposition](#cholesky-decomposition)
- [SVD and Applications](#svd-swiss-army)
:::

:::{grid-item-card} 🧗 **Complete Path**
Want mastery? Read all sections including 🟢

- Complete module with:
- Advanced decompositions
- Numerical implementations
- ML connections
:::
::::

### 🎯 Navigation by Project Needs

:::{important} Quick Jump to What You Need by Project
:class: dropdown

**For Project 4 (MCMC)**:
- [Section 5.3: Covariance Matrices](#covariance-matrices) - Proposal distributions
- [Section 5.4: Multivariate Gaussian](#multivariate-gaussian) - Sampling
- [Section 7: Numerical Reality](#part-7-numerical-reality) - Stability

**For Project 5 (Gaussian Processes)**:
- [Section 5: Positive Definite Matrices](#part-5-positive-definite) - Kernel matrices
- [Section 5.5: Cholesky Decomposition](#cholesky-decomposition) - GP implementation
- [Section 6.2: Schur Complement](#block-matrices) - Efficient updates

**For Final Project (Neural Networks)**:
- [Section 6: Advanced Topics](#part-6-advanced) - Matrix norms, Jacobians
- [Section 8: Bridge to ML](#part-8-bridge) - Connecting everything
:::

---

## Learning Objectives

:::{hint} 📅 When to Read This Module
:class: dropdown

- [ ] **Before Project 4**: Read sections marked 🔴 (3-4 hours)
- [ ] **During Projects 4-5**: Deep dive into 🟡 sections
- [ ] **For Final Project**: Review Section 8 connections
- [ ] **Reference**: Return when debugging numerical issues

**Prerequisites**: Complete Module 0a or be comfortable with vectors, matrices, and eigenvalues.
:::

By the end of this module, you will be able to:

- [ ] **Determine** whether a matrix is positive definite using multiple methods 🔴
- [ ] **Construct and interpret** covariance matrices from data 🔴
- [ ] **Apply** Cholesky decomposition for sampling and solving 🟡
- [ ] **Debug** numerical failures in statistical computations 🔴
- [ ] **Connect** linear algebra to MCMC, Gaussian Processes, and neural networks 🟡
- [ ] **Implement** robust algorithms that handle finite precision 🟢

---

## Prerequisites from Module 0a

:::{admonition} 📚 Required Knowledge from Module 0a
:class: note, dropdown

**Priority: 🔴 Essential** - Review if needed

Before starting this module, ensure you understand:

- [ ] Matrix multiplication and transposition
- [ ] Eigenvalues and eigenvectors
- [ ] Symmetric matrices and their properties
- [ ] Matrix determinants and rank
- [ ] Basic matrix decompositions

If unfamiliar, review Module 0a first.
:::

---

## Module Overview

:::{admonition} 🎯 Core Message
:class: important

**Priority: 🔴 Essential**:

This module bridges deterministic linear algebra with statistical methods and machine learning. You'll master the mathematical foundations for uncertainty quantification, probabilistic inference, and learning algorithms—essential for Projects 4-6.
:::

Building on Module 0a's foundations, this module explores how linear algebra enables statistical computation and machine learning. We cover **positive definite matrices** that ensure physical validity (Project 4), **covariance structures** that encode correlations (Project 5), and **numerical methods** that keep computations stable despite finite precision (all projects). The module culminates by showing how these concepts unite classical physics with modern machine learning (Final Project).

:::{admonition} 🎯 Why This Module Matters
:class: important

For Projects 4-6, you'll need:

- **Project 4 (MCMC)**: Covariance matrices for proposals, multivariate Gaussians for sampling
- **Project 5 (Gaussian Processes)**: Positive definite kernels, Cholesky decomposition, numerical stability
- **Final Project (Neural Networks)**: Matrix norms for convergence, Jacobians for backpropagation

These aren't just mathematical abstractions—they're the tools that let you quantify uncertainty in cosmological parameters, interpolate sparse astronomical data, and train neural networks to discover patterns in observations.
:::

## Part 1: Positive Definite Matrices and Statistical Foundations

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

:::{note} 📝 Regularization Throughout Your Projects

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

:::{admonition} 📝 Geometric Interpretation of Covariance
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

:::{admonition} 📝 Why $\Sigma^{-1}$ Appears (Not Just $\Sigma$)
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

:::{admonition} 📝 Principal Component Analysis (PCA) = SVD of Data
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

### 7.1 Condition Numbers: Measuring Numerical Danger {#condition-numbers}

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

### 7.2 When Linear Algebra Fails - And How to Fix It

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

### 7.3 Speed vs. Accuracy - The Eternal Tradeoff

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

### 7.4 Troubleshooting Guide - What to Do When Things Break

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

:::{warning} 🎯 Project-Specific Numerical Gotchas

**Project 4**: MCMC
- Proposal covariance can lose positive definiteness
- **Fix:** Adaptive regularization during burn-in

**Project 5**: Gaussian Processes
- Kernel matrices become ill-conditioned for close points
- **Fix:** Jitter + careful hyperparameter bounds

**Final Project**: Neural Networks
- Gradient explosion/vanishing from poor initialization
- **Fix:** Xavier/He initialization + gradient clipping
:::

:::{admonition} 📌 Key Takeaway
:class: important

Computers aren't mathematical ideals — they're finite machines with finite precision. Every number is approximate, every operation loses accuracy, and errors compound rapidly. But with awareness and proper techniques, you can write robust code that handles numerical reality gracefully.

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

Your final project connects directly to this—the same neural network architecture you'll build is finding new worlds.

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

**Matrix Derivatives for Backpropagation:**

For neural networks, you need derivatives with respect to matrices:
- If L = ||Wx - y||², then ∂L/∂W = 2(Wx - y)x^T
- The chain rule extends: ∂L/∂W₁ = (∂L/∂W₂)(∂W₂/∂W₁)
- This is what JAX's autodiff computes automatically

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

You've journeyed from positive definite matrices through numerical reality to modern applications. The key insight: linear algebra isn't just mathematical machinery—it's the language of physical reality at computational scales.

**What you've mastered**:
- Positive definiteness ensures physical validity
- Covariance matrices encode uncertainty structure
- Cholesky decomposition bridges uncorrelated and correlated worlds
- Numerical precision limits require careful algorithm design
- Advanced decompositions reveal hidden structure
- The same mathematics powers physics and machine learning

**Why it matters**:
Every statistical method and machine learning algorithm in astrophysics relies on these advanced linear algebra concepts. Master these foundations and you can:

- Sample from complex probability distributions
- Build robust Gaussian Process models
- Train stable neural networks
- Debug numerical disasters before they happen
- Contribute to the next astronomical breakthrough

Linear algebra is your passport to computational astrophysics and machine learning. Use it wisely, and the universe's secrets await!
:::

---

## Main Takeaways

:::{admonition} 🎯 Essential Points to Remember
:class: important

This module covered the advanced linear algebra that bridges classical physics with modern machine learning:

1. **Positive definite matrices guarantee physical validity** in energy, distance, and probability

2. **Covariance matrices encode all linear relationships** between random variables

3. **Cholesky decomposition is the bridge** between uncorrelated randomness and structured correlations

4. **Numerical precision is finite** - always check condition numbers and add regularization when needed

5. **Advanced decompositions (SVD, block matrices) reveal hidden structure** and enable efficient computation

6. **The same mathematics spans domains** - eigenvalues determine both orbital stability and neural network convergence

7. **Computers aren't mathematical ideals** - understand floating-point reality to write robust code
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
* - Solve positive definite
  - `scipy.linalg.cho_solve()`
  - Covariance matrices
  - ⭐⭐⭐⭐⭐
* - Least squares
  - `scipy.linalg.lstsq()`
  - Overdetermined systems
  - ⭐⭐⭐⭐⭐
* - SVD
  - `scipy.linalg.svd()`
  - Any matrix, rank
  - ⭐⭐⭐⭐⭐
* - Matrix exponential
  - `scipy.linalg.expm()`
  - Time evolution
  - ⭐⭐⭐⭐
* - Schur decomposition
  - `scipy.linalg.schur()`
  - Stability analysis
  - ⭐⭐⭐⭐
* - Sparse solve
  - `scipy.sparse.linalg.spsolve()`
  - Large sparse systems
  - ⭐⭐⭐⭐⭐
```

---
(quickref)=
## One-Page Cheat Sheet

:::{admonition} 📋 Statistical Linear Algebra Quick Reference
:class: tip

### Essential Formulas

**Positive Definite Matrices**
- Definition: $\vec{x}^T A \vec{x} > 0$ for all $\vec{x} \neq \vec{0}$
- Tests: All eigenvalues > 0, has Cholesky decomposition
- Fix: Add jitter `A_stable = A + 1e-6 * I`

**Covariance Matrices**
- Element: $\Sigma_{ij} = \text{Cov}(X_i, X_j)$
- Always symmetric and positive semi-definite
- Diagonal = variances, off-diagonal = covariances
- Correlation: $\rho_{ij} = \Sigma_{ij}/\sqrt{\Sigma_{ii}\Sigma_{jj}}$

**Multivariate Gaussian**
$$p(\vec{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})\right)$$

**Cholesky Decomposition**
- Factorization: $A = LL^T$ (L is lower triangular)
- Sample from N(μ,Σ): `x = mu + L @ randn(n)`
- Solve Ax=b: Forward then back substitution

### Numerical Reality

**Floating-Point Limits**
- Machine epsilon: ~2.2×10⁻¹⁶ for float64
- Catastrophic cancellation when subtracting nearly equal numbers
- Add softening/regularization to prevent singularities

**Condition Number Guidelines**
- κ < 10: Excellent
- κ = 10³: Lose ~3 digits
- κ = 10⁶: Dangerous
- κ > 10¹⁰: Numerically singular

### Advanced Decompositions

**SVD**: $A = U\Sigma V^T$
- Always exists, most stable
- Rank = number of non-zero singular values
- PCA = SVD of centered data

**Schur Complement**: $S = A - BD^{-1}C$
- Efficient updates in GPs
- Block matrix elimination

### Critical Commands

| Task | Command | Use Case |
|------|---------|----------|
| Check positive definite | `np.linalg.eigvalsh(A).min() > 0` | Before Cholesky |
| Force symmetry | `A = (A + A.T) / 2` | Fix rounding errors |
| Regularize | `A + lambda * I` | Improve conditioning |
| Robust solve | `np.linalg.lstsq()` | When solve() fails |

### Debugging Checklist

□ Matrix singular? → Check rank and condition number
□ Cholesky fails? → Check min eigenvalue, add jitter
□ Complex eigenvalues? → Force symmetry
□ Wrong solution? → Check residuals
□ Too slow? → Consider sparse matrices or iterative solvers

### Key Insights by Project

**Project 4 (MCMC)**: Proposal covariance determines step efficiency
**Project 5 (GPs)**: Kernel matrices must be positive definite
**Final Project**: Weight matrix condition affects gradient flow
:::

---

## Next Steps

With these advanced mathematical foundations in place, you're ready to tackle the statistical and machine learning challenges of modern astrophysics. In Project 4, you'll use covariance matrices to design efficient MCMC samplers. In Project 5, positive definiteness will be crucial for Gaussian Process kernels. By the final project, you'll see how the same linear algebra principles determine neural network training dynamics.

**Remember:** numerical issues aren't bugs to avoid—they're realities to manage. When Cholesky decomposition fails or matrices become ill-conditioned, return to these foundations. Check eigenvalues, add regularization, use appropriate decompositions. The mathematics you've learned here isn't separate from computation—it IS computation in its most robust form.

---

*Master these tools, and the universe's patterns await your discovery.*
