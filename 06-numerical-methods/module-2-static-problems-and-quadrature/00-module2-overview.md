---
title: "Overview: Static Problems & Quadrature"
subtitle: "Numerical Methods Module 2 | ASTR 596: Modeling the Universe"
exports:
  - format: pdf
---

## Module Overview

This module addresses two fundamental questions: Where do things balance? How do we measure quantities in the universe?

The cosmos is filled with equilibrium points (i.e., the net force on an object is equal to 0) — where pressure balances gravity in stars, where spacecraft can hover between massive bodies, where light itself can orbit black holes. Finding these balance points requires root-finding algorithms. Meanwhile, measuring cosmic quantities — from stellar luminosities to dark matter halos — requires numerical integration (quadrature) across scales from stellar surfaces to galactic boundaries.

While root finding tells us WHERE things happen (equilibrium points, orbital crossing times, energy minima), integration tells us HOW MUCH there is (total energy radiated, mass enclosed, photons absorbed). These two classes of problems form the foundation for all dynamical simulations you'll build later.

## Learning Philosophy

Continuing our "glass-box modeling" approach:

- **Build from first principles** - Derive every algorithm from fundamental mathematics
- **Understand convergence** - Know not just if methods work, but why and when they fail
- **Connect to physics** - Every numerical choice has physical consequences
- **Embrace trade-offs** - No method is universally best; context determines optimality

## Building on Module 1 Foundations

You mastered three key concepts in Module 1 that directly apply here:

1. **Error Analysis**: The truncation vs round-off trade-off governs convergence in root finding and accuracy in integration. The optimal step size h from Module 1 appears here as optimal spacing for numerical integration—same trade-off, different application.

2. **Taylor Series**: Just as it revealed finite difference errors, it now shows us convergence rates and integration accuracy. You'll use the same expansion techniques to understand why Simpson's rule achieves fourth-order accuracy.

3. **Finite Precision**: The same catastrophic cancellation that plagued derivatives can destroy root-finding near f'(x) = 0. You'll need to reformulate problems for numerical stability, just as you did with finite differences.

Watch for these patterns as we extend from computing derivatives at points to finding zeros and measuring areas.

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
- Apply root finding to Kepler's equation, Lagrange points, and stellar structure

**Key Concepts:** Bracketing, tangent approximation, convergence order, condition number

### Part 2: Quadrature - From Photon Counts to Dark Matter Halos

**Learning Outcomes:**

- Derive rectangular, trapezoidal, and Simpson's rules from Taylor series
- Understand optimal point placement through Gaussian quadrature
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

## Self-Assessment

Before beginning, verify you can:

- [ ] Explain the difference between truncation and round-off error
- [ ] Use Taylor series to analyze algorithm accuracy
- [ ] Implement iterative algorithms with convergence criteria
- [ ] Recognize when a problem is well- or ill-conditioned

## Connections to Course Projects

The methods you learn here are essential for:

- **Project 2 (N-body)**: Finding orbital perihelion/aphelion, computing orbital periods, solving Kepler's equation
- **Project 3 (MCRT)**: Monte Carlo integration of photon paths, finding optical depth boundaries
- **Project 4 (MCMC)**: Integration of posterior distributions, finding credible intervals
- **Project 5 (GP)**: Kernel integrals, hyperparameter optimization
- **Final Project**: Loss function minimization, finding neural network optima

## A Note on Software Libraries

In research, you'll often use `scipy.optimize.root_scalar` for root finding and `scipy.integrate.quad` for integration. These are battle-tested and efficient. However, understanding what's happening inside these "black boxes" is crucial for:

- Choosing appropriate methods and tolerances
- Diagnosing when and why they fail
- Modifying algorithms for special cases
- Building custom methods when needed

This module teaches you to build these tools from scratch so you can use library functions intelligently.

## Module Philosophy

**Remember**: We're not just learning algorithms — we're understanding the fundamental mathematics that makes computational astrophysics possible. Every method has a geometric interpretation, a convergence proof, and a failure mode. Master all three, and you'll be able to tackle any computational challenge in your research.
