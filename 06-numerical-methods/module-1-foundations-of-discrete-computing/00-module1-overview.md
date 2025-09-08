---
title: "Overview: Foundations of Discrete Computing"
subtitle: "Numerical Methods Module 1| ASTR 596: Modeling the Universe"
exports:
  - format: pdf
---

## The Big Picture: Simulating the Universe on Finite Computers

This module addresses a fundamental challenge: how do we perform calculus — the mathematics of continuous change — on digital computers that can only represent discrete, finite-precision numbers?

In astrophysics, even individual simulations must handle enormous dynamic ranges that strain computational limits. N-body codes track gravitational forces varying by factors of $10^{18}$. Star formation simulations span 6 orders of magnitude in length scale and 11 in density. Stellar evolution models bridge 13 orders of magnitude in time. These ranges push against — and often exceed — the ~$10^{16}$ dynamic range of double-precision arithmetic. This module teaches you to navigate these limitations intelligently, building robust numerical methods from first principles.

## Learning Philosophy

This module embodies our "glass-box modeling" approach:

- **Understand every approximation:** Know exactly what errors you're introducing and why
- **Build from fundamentals:** Derive methods from Taylor series and error analysis
- **Connect to physics:** Every numerical choice has physical consequences
- **Embrace limitations:** Finite precision isn't a bug; it's a feature that teaches us about our models

## Module Learning Objectives

By completing this module, you will be able to:

- [ ] **Analyze** the fundamental trade-offs between truncation and round-off error in numerical calculations
- [ ] **Design** numerical algorithms that minimize the total error for specific astrophysical problems
- [ ] **Predict** error scaling behavior before implementing code
- [ ] **Diagnose** numerical instabilities and reformulate problems for better conditioning
- [ ] **Choose** appropriate numerical methods based on problem characteristics and accuracy requirements
- [ ] **Implement** robust code that handles edge cases and numerical limitations gracefully

## Module Structure

### Part 1: The Fundamental Paradox - Calculus on Computers

**Learning Outcomes:**

- Explain why computers cannot take the limit $h \to 0$ and analyze the implications
- Derive all finite difference approximations from first principles (forward, backward, central, higher-order) from Taylor series.
- Calculate and verify the optimal step size $h$ for different methods
- Implement finite difference methods with appropriate error handling

**Key Concepts:** Finite differences, truncation error, optimal step size, method order

### Part 2: Numbers Aren't Real - Computer Arithmetic & Cosmic Consequences  

**Learning Outcomes:**

- Calculate machine epsilon for different data types
- Identify and classify three types of numerical error (round-off, truncation, propagation)
- Diagnose catastrophic cancellation and reformulate expressions to avoid it
- Predict when round-off errors will dominate calculations

**Key Concepts:** Machine epsilon, floating-point arithmetic, catastrophic cancellation, error propagation

### Part 3: Taylor Series - The Bridge from Continuous to Discrete

**Learning Outcomes:**

- Apply Taylor series to derive numerical methods of different orders
- Verify error predictions empirically through numerical experiments
- Understand why symmetric methods achieve higher accuracy
- Recognize when to use finite differences vs. automatic differentiation

**Key Concepts:** Taylor expansion, error order, symmetry cancellation, method derivation

## Prerequisites

### Required Knowledge

- Calculus through multivariable (derivatives, integrals, Taylor series)
- Basic Python programming (functions, loops, NumPy arrays)
- Elementary differential equations

### Helpful but Not Required

- Previous exposure to numerical methods
- Experience with scientific computing
- Familiarity with astronomical scales and units

## Self-Assessment

Before beginning, verify you can:

- [ ] Take the derivative of $f(x) = x^n \sin(x)$ analytically
- [ ] Expand $e^x$ as a Taylor series around $x = 0$
- [ ] Write a Python function that computes the mean of an array
- [ ] Explain what $\lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ means geometrically

If you're unsure about any of these, review the prerequisite material or ask for help during office hours.

## Connections to Course Projects

The methods you learn here form the foundation for:

- **Project 2 (N-body)**: Computing velocities and accelerations from positions
- **Project 3 (MCRT)**: Handling round-off in photon propagation  
- **Project 4 (MCMC)**: Computing gradients for optimization
- **Project 5 (GP)**: Numerical derivatives of kernel functions
- **Final Project**: Understanding automatic differentiation in neural networks

**Remember:** The goal isn't to memorize formulas but to understand the fundamental trade-offs in numerical computing. These principles will guide you throughout your computational physics career.
