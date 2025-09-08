---
title: "Overview: ODE Methods & Conservation"
subtitle: "Numerical Methods Module 3 | ASTR 596: Modeling the Universe"
exports:
  - format: pdf
---

## Module Overview

The heart of computational dynamics involves making time flow numerically while preserving the fundamental laws of physics. This module addresses the central challenge of astrophysics: how do we simulate the universe's evolution through billions of years without destroying conservation laws?

The cosmos evolves through differential equations. Planets orbit for gigayears, galaxies merge over cosmic time, stars pulsate through nuclear cycles. But naive numerical integration accumulates errors that grow without bound, eventually destroying the physics we're trying to simulate. This module reveals why "better" isn't always better—why preserving geometric structure matters more than minimizing error, and how to choose methods that keep your universe stable.

## Learning Philosophy

Continuing our "glass-box modeling" approach:
- **Build from failure** - Understand why Euler's method destroys orbits
- **Derive from principles** - Construct Runge-Kutta methods from Taylor series
- **Preserve what matters** - Geometric structure over local accuracy
- **Optimize ruthlessly** - Transform loops to vectorized arrays

## Module Learning Objectives

By completing this module, you will be able to:

- [ ] **Explain** why Euler's method fails catastrophically for long-term orbital dynamics
- [ ] **Derive** the complete family of Runge-Kutta methods from Taylor series expansions
- [ ] **Prove** why symplectic integrators conserve phase space volume and bounded energy
- [ ] **Analyze** stability regions to predict when numerical methods explode
- [ ] **Implement** Euler, RK2, RK4, and Leapfrog integrators from first principles
- [ ] **Choose** appropriate integrators based on problem timescales and conservation requirements
- [ ] **Transform** sequential loop-based code to vectorized array operations
- [ ] **Diagnose** numerical instabilities before they destroy simulations
- [ ] **Design** integration schemes that preserve physical invariants

## Module Structure

### Part 1: The Failure of Naive Integration

**Learning Outcomes:**
- Implement Euler's method and witness its catastrophic energy drift
- Analyze local vs global error accumulation through Taylor series
- Understand why higher accuracy doesn't guarantee better long-term behavior
- Recognize the geometric failure modes in phase space

**Key Concepts:** Local truncation error, global error accumulation, phase space geometry, energy drift

### Part 2: Building Better Methods - Runge-Kutta

**Learning Outcomes:**

- Derive RK2 and RK4 from multivariate Taylor expansions
- Connect RK weights to quadrature rules from Module 2
- Implement adaptive timestep control
- Analyze when higher-order methods help vs harm

**Key Concepts:** Predictor-corrector, order conditions, adaptive control, accuracy vs stability

### Part 3: Symplectic Integration - Geometry Over Accuracy

**Learning Outcomes:**

- Prove symplecticity of leapfrog/Verlet methods
- Understand modified Hamiltonians and bounded energy error
- Implement symplectic integrators for N-body problems
- Choose between accuracy and structure preservation

**Key Concepts:** Symplectic structure, Liouville's theorem, modified Hamiltonian, time reversibility

### Part 4: Stability and Performance

**Learning Outcomes:**

- Analyze stability regions for different methods
- Diagnose and handle stiff equations
- Transform scalar loops to vectorized operations
- Achieve 10-100× performance improvements

**Key Concepts:** Stability regions, stiffness, vectorization, memory layout optimization

## Prerequisites

### Required Knowledge

- Modules 1-2 concepts (derivatives, integration, error analysis)
- Differential equations basics
- Linear algebra (eigenvalues, matrices)
- Python/NumPy proficiency

### Mathematical Maturity
- Comfortable with Taylor series manipulations
- Understanding of phase space concepts
- Ability to connect discrete and continuous mathematics

## Self-Assessment

Before beginning, verify you can:

- [ ] Explain the difference between local and global error
- [ ] Use Taylor series to analyze algorithm accuracy
- [ ] Implement basic numerical integration (trapezoid rule)
- [ ] Work with vector operations in NumPy

## Connections to Course Projects

The methods you learn here are essential for:

- **Project 2 (N-body)**: Core integration methods for orbital dynamics
- **Project 3 (MCRT)**: Ray integration through varying media
- **Project 4 (MCMC)**: Hamiltonian Monte Carlo uses symplectic integration
- **Project 5 (GP)**: Solving SDEs for kernel functions
- **Final Project**: Training dynamics of neural networks

## Module Philosophy

**Remember**: We're not just solving differential equations — we're preserving the fundamental structure of physics through discrete time. Every integration method creates an alternate reality with slightly different physics. Our job is choosing the reality that best preserves what matters for our problem.
