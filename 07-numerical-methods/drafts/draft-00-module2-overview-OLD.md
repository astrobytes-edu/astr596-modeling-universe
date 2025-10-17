# Module 2: Static Problems & Quadrature

## Module Overview

Welcome to the mathematics of equilibrium and measurement in computational astrophysics. This module addresses two fundamental questions: Where do things balance? How do we measure quantities in the universe?

The cosmos is filled with equilibrium points—where pressure balances gravity in stars, where spacecraft can hover between massive bodies, where light itself can orbit black holes. Finding these balance points requires root-finding algorithms. Meanwhile, measuring cosmic quantities—from stellar luminosities to dark matter halos—requires numerical integration across scales from stellar surfaces to galactic boundaries.

## Learning Philosophy

Continuing our "glass-box modeling" approach:
- **Build from first principles** - Derive every algorithm from fundamental mathematics
- **Understand convergence** - Know not just if methods work, but why and when they fail
- **Connect to physics** - Every numerical choice has physical consequences
- **Embrace trade-offs** - No method is universally best; context determines optimality

## Module Learning Objectives

By completing this module, you will be able to:

- [ ] **Implement** root-finding algorithms for astrophysical equilibria with rigorous understanding of convergence
- [ ] **Select** appropriate methods based on problem characteristics and convergence requirements
- [ ] **Apply** numerical integration to measure cosmic quantities with controlled error
- [ ] **Derive** error bounds for quadrature methods from first principles
- [ ] **Connect** quadrature methods to Monte Carlo techniques for high-dimensional problems
- [ ] **Analyze** convergence rates and computational complexity trade-offs
- [ ] **Diagnose** failures in root-finding and integration algorithms
- [ ] **Design** hybrid approaches for challenging problems

## Module Structure

### Part 1: Root Finding - Where Physics Reaches Equilibrium

**Learning Outcomes:**
- Implement bisection, Newton-Raphson, and secant methods from scratch
- Analyze convergence rates (linear, superlinear, quadratic) mathematically
- Diagnose method failures and design robust hybrid approaches
- Apply root finding to Kepler's equation and Lagrange points

**Key Concepts:** Bracketing, tangent approximation, convergence order, condition number

### Part 2: Quadrature - From Photon Counts to Dark Matter Halos

**Learning Outcomes:**
- Derive rectangular, trapezoidal, and Simpson's rules from Taylor series
- Implement Gaussian quadrature for optimal point placement
- Apply Monte Carlo integration for high-dimensional problems
- Choose appropriate methods based on smoothness, dimension, and noise

**Key Concepts:** Truncation error, Richardson extrapolation, curse of dimensionality, variance reduction

### Part 3: Synthesis - The Deep Connections

**Learning Outcomes:**
- Connect root finding and integration as inverse operations
- Apply universal error analysis principles across methods
- Design method selection strategies for real problems
- Recognize these patterns throughout computational physics

**Key Concepts:** Condition number, computational complexity, method selection flowcharts

## Prerequisites

### Required Knowledge
- Module 1 concepts (finite differences, Taylor series, error analysis)
- Calculus (derivatives, integrals, fundamental theorem)
- Basic numerical analysis concepts
- Python programming with NumPy

### Mathematical Maturity
- Comfortable with proofs and derivations
- Able to connect abstract math to physical problems
- Experience with iterative algorithms

## Self-Assessment

Before beginning, verify you can:
- [ ] Explain the difference between truncation and round-off error
- [ ] Use Taylor series to analyze algorithm accuracy
- [ ] Implement iterative algorithms with convergence criteria
- [ ] Recognize when a problem is well- or ill-conditioned

## Connections to Course Projects

The methods you learn here are essential for:
- **Project 2 (N-body)**: Finding orbital perihelion/aphelion, computing orbital periods
- **Project 3 (MCRT)**: Monte Carlo integration of photon paths, finding optical depth boundaries
- **Project 4 (MCMC)**: Integration of posterior distributions, finding credible intervals
- **Project 5 (GP)**: Gaussian quadrature for kernel integrals, hyperparameter optimization
- **Final Project**: Loss function integration, finding neural network minima

## Resources

- **Required Reading**: This module content
- **Reference**: Numerical Recipes, Chapters 9 (root finding) and 4 (integration)
- **Supplemental**: Press et al., "Numerical Methods in Physics"
- **Advanced**: Boyd, "Chebyshev and Fourier Spectral Methods"

## Module Philosophy

Remember: We're not just learning algorithms—we're understanding the fundamental mathematics that makes computational astrophysics possible. Every method has a geometric interpretation, a convergence proof, and a failure mode. Master all three, and you'll be able to tackle any computational challenge in your research.