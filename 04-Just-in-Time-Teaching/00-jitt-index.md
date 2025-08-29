---
title: JiTT Mathematical & Conceptual Support
subtitle: "Just-in-Time Teaching | Released When You Need Them"
exports:
  - format: pdf
---

## Mathematical Foundations Delivered Exactly When You Need Them

This module provides targeted mathematical and conceptual support released strategically throughout the course. Rather than front-loading abstract theory, each chapter appears precisely when you need it for your current project—transforming potentially overwhelming mathematics into immediately applicable tools.

:::{tip} The JiTT Philosophy

**_Why Just-in-Time Teaching_?** Cognitive science shows we learn physics and mathematics best when we have concrete problems to solve. Each chapter releases right before you need its concepts, ensuring:

- **Immediate application** reinforces understanding
- **Concrete context** makes abstractions tangible  
- **Reduced cognitive load** by spreading material across the semester
- **Natural spiraling** as concepts reappear in new contexts

The **JiTT** (Just-in-Time Teaching) approach in this course draws from extensive physics education research:

- **Novak et al. (1999)** developed Just-in-Time Teaching at IUPUI, showing 40% reduction in attrition and 90% student preference over traditional formats.
- **Marrs & Novak (2004)** demonstrated JiTT's effectiveness in biology with learning gains jumping from 16.7% (traditional) to 52.3% (JiTT).
- Studies show JiTT students are 8x more likely to transition from "common sense" to scientific thinking.

[Read more about JiTT research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3203712/)
:::

## Module Structure

```{mermaid}
flowchart TD
    A[Week 1-2: Foundations] --> B[Week 3-7: Classical Methods]
    B --> C[Week 8-10: Statistical Inference]
    C --> D[Week 11-12: Bridge to ML]
    D --> E[Week 13-16: Modern ML]
    
    A1[Basic numerics<br/>Stellar physics] --> A
    B1[ODEs, Monte Carlo<br/>Radiative transfer] --> B
    C1[Bayesian, MCMC<br/>Cosmology] --> C
    D1[Kernels, GPs<br/>Function learning] --> D
    E1[Neural networks<br/>Backpropagation] --> E
    
    style A fill:#e3f2fd
    style B fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fce4ec
```

## Release Schedule & Project Alignment

:::{list-table} **When Chapters Unlock**
:header-rows: 1
:widths: 15 25 30 30

* - Week
  - Chapters Released
  - Project Support
  - Key Concepts
* - **1**
  - 1.1 (Stellar), 2.1 (Numerics), 4.1 (Optimization basics)
  - Project 1: Stellar Populations
  - OOP, floating point, vectorization
* - **3**
  - 1.2 (Dynamics), 2.2 (ODEs)
  - Project 2: N-Body
  - Integration methods, stability
* - **4**
  - 3.1 (Probability), 3.2 (Monte Carlo)
  - Project 2: Sampling
  - Distributions, CLT, sampling
* - **5**
  - 1.3 (Rad Transfer), 2.3 (Linear Algebra)
  - Project 3: MCRT
  - Optical depth, scattering
* - **8**
  - 1.4 (Cosmology), 3.3 (Bayesian)
  - Project 4: MCMC
  - Bayes theorem, priors
* - **9**
  - 3.4 (Markov Chains)
  - Project 4: MCMC
  - Detailed balance, convergence
* - **10**
  - 4.1 (Optimization advanced)
  - Project 4: HMC
  - Gradients, Hamiltonians
* - **11**
  - 2.3 (Advanced LA), 3.5 (Multivariate), 4.2-4.3 (Kernels, GPs)
  - Project 5: Gaussian Processes
  - Covariance, kernels, uncertainty
* - **13**
  - 2.4 (Multivariable Calc), 4.4 (Neural Nets)
  - Final Project: Deep Learning
  - Chain rule, architectures
* - **15**
  - 4.5 (Backprop), 4.6 (Training)
  - Final Project: JAX
  - Autodiff, optimization
:::

## Chapter Overview

### Module 1: Astrophysics Foundations

:::{grid} 1 1 2 2

:::{grid-item-card} 📊 **Ch 1.1: Stellar Structure**
Week 1 | Project 1 Support
^^^
- Mass-luminosity relations
- Main sequence physics
- Stellar populations
- HR diagrams
:::

:::{grid-item-card} 🌌 **Ch 1.2: Gravitational Dynamics**
Week 3 | Project 2 Support
^^^
- N-body problem
- Virial theorem
- Plummer spheres
- Cluster evolution
:::

:::{grid-item-card} ☀️ **Ch 1.3: Radiative Transfer**
Week 5 | Project 3 Support
^^^
- Optical depth
- Scattering physics
- Photon transport
- Monte Carlo RT
:::

:::{grid-item-card} 🔭 **Ch 1.4: Cosmological Distances**
Week 8 | Project 4 Support
^^^
- Hubble's law
- Luminosity distance
- Type Ia supernovae
- Distance modulus
:::

::::

### Module 2: Mathematical Tools

:::{grid} 1 1 2 2

:::{grid-item-card} 🔢 **Ch 2.1: Numerical Fundamentals**
Week 1 | Foundation
^^^
- Floating point representation
- Round-off error
- Vectorization
- Algorithm complexity
:::

:::{grid-item-card} 📈 **Ch 2.2: ODEs & Integration**
Week 3 | Project 2 Support
^^^
- Taylor series
- Euler to Leapfrog
- Symplectic integrators
- Energy conservation
:::

:::{grid-item-card} 🔧 **Ch 2.3: Linear Algebra**
Weeks 5, 11 | Projects 3, 5
^^^
- solve() vs inv()
- Cholesky decomposition
- Eigenvalue problems
- Condition numbers
:::

:::{grid-item-card} 🎯 **Ch 2.4: Multivariable Calculus**
Week 13 | Neural Networks
^^^
- Gradients in N-D
- Chain rule
- Jacobians
- Backpropagation math
:::

::::

### Module 3: Probability & Statistics

:::{grid} 1 1 2 2

:::{grid-item-card} 🎲 **Ch 3.1: Probability Foundations**
Week 4 | Monte Carlo Prep
^^^
- PDFs and CDFs
- Change of variables
- Sampling methods
- Expectation values
:::

:::{grid-item-card} 🎯 **Ch 3.2: Monte Carlo Methods**
Week 4 | Projects 2-3
^^^
- Central Limit Theorem
- Error scaling
- Importance sampling
- Variance reduction
:::

:::{grid-item-card} 📊 **Ch 3.3: Bayesian Statistics**
Week 8 | Project 4
^^^
- Bayes theorem
- Prior selection
- Likelihood construction
- Evidence calculation
:::

:::{grid-item-card} 🔗 **Ch 3.4: Markov Chains**
Week 9 | MCMC
^^^
- Markov property
- Detailed balance
- Convergence diagnostics
- Gelman-Rubin test
:::

:::{grid-item-card} 📐 **Ch 3.5: Multivariate Distributions**
Week 11 | Gaussian Processes
^^^
- Multivariate Gaussians
- Covariance matrices
- Marginals & conditionals
- Sampling techniques
:::

::::

### Module 4: Machine Learning Concepts

:::{grid} 1 1 2 2

:::{grid-item-card} 🎢 **Ch 4.1: Optimization**
Weeks 1, 10, 13
^^^
- Gradient descent
- Learning rates
- Momentum methods
- Mini-batch SGD
:::

:::{grid-item-card} 🌀 **Ch 4.2: Kernel Methods**
Week 11 | GPs
^^^
- RBF kernels
- Matérn family
- Kernel trick
- Hyperparameters
:::

:::{grid-item-card} 📉 **Ch 4.3: Gaussian Processes**
Week 11 | Project 5
^^^
- Function distributions
- Posterior predictions
- O(n³) complexity
- Uncertainty quantification
:::

:::{grid-item-card} 🧠 **Ch 4.4: Neural Networks**
Week 13 | Final Project
^^^
- Activation functions
- Weight initialization
- Universal approximation
- Architecture design
:::

:::{grid-item-card} 🔄 **Ch 4.5: Backpropagation**
Week 15 | JAX
^^^
- Computational graphs
- Reverse-mode autodiff
- JAX implementation
- Gradient flow
:::

:::{grid-item-card} 📈 **Ch 4.6: Training Dynamics**
Week 15 | Final Project
^^^
- Overfitting
- Regularization
- Early stopping
- Learning schedules
:::

::::

## How to Use This Module

### For Each Chapter

Every 3-5 page chapter follows this structure:

1. **Physical Intuition** - Why this matters for your project
2. **Mathematical Framework** - Key equations with meaning
3. **Implementation Guide** - Code snippets and pseudocode  
4. **Common Pitfalls** - What typically goes wrong
5. **Self-Check Problems** - Quick verification exercises

### Learning Strategy

:::{tip} **Active Learning Approach**

1. **Preview** when chapter releases - skim for main ideas
2. **Apply** immediately in current project
3. **Debug** using the common pitfalls section
4. **Return** when similar concepts reappear later
5. **Connect** to other chapters as patterns emerge
:::

## Quick Navigation by Project

### Project 1: Stellar Populations
- Ch 1.1: Stellar physics you need
- Ch 2.1: Numerical safety
- Ch 4.1: Basic optimization concepts

### Project 2: N-Body Dynamics  
- Ch 1.2: Gravitational dynamics
- Ch 2.2: Integration methods
- Ch 3.1-3.2: Sampling techniques

### Project 3: Monte Carlo Radiative Transfer
- Ch 1.3: Radiative transfer physics
- Ch 2.3: Vector operations
- Ch 3.2: Monte Carlo convergence

### Project 4: Cosmological Inference
- Ch 1.4: Cosmological distances
- Ch 3.3-3.4: Bayesian/MCMC
- Ch 4.1: Gradient-based optimization

### Project 5: Gaussian Processes
- Ch 2.3: Advanced linear algebra
- Ch 3.5: Multivariate distributions  
- Ch 4.2-4.3: Kernels and GPs

### Final Project: Neural Networks
- Ch 2.4: Multivariable calculus
- Ch 4.4-4.6: Deep learning suite
- All optimization concepts

## Common Patterns Across Chapters

### The Computational Trinity

Three concepts appear everywhere:

1. **Linear Algebra** (Ch 2.3)
   - Solving systems (MCMC, GPs, NNs)
   - Eigenvalues (stability, PCA)
   - Matrix decompositions (numerical methods)

2. **Optimization** (Ch 4.1)
   - Gradient descent (NNs)
   - Maximum likelihood (GPs)
   - HMC (advanced MCMC)

3. **Probability** (Ch 3.1-3.5)
   - Sampling (all Monte Carlo)
   - Uncertainty (GPs, Bayesian)
   - Distributions (everywhere!)

### Numerical Safety Thread

Watch how numerical concerns evolve:

- **Week 1**: Floating point basics
- **Week 3**: ODE integration stability
- **Week 5**: Matrix conditioning
- **Week 8**: Log probabilities for MCMC
- **Week 11**: Cholesky with jitter for GPs
- **Week 13**: Gradient vanishing in NNs

## FAQ

**Q: What if I'm weak in math?**
A: Each chapter assumes only high school mathematics and builds up. The concrete applications make abstractions tangible.

**Q: Should I read ahead?**
A: Focus on current chapters. Later material won't make sense without the project context.

**Q: How much detail should I understand?**
A: Understand concepts and implementation. Detailed proofs are optional unless you're curious.

**Q: Will these be on an exam?**
A: No exams! These concepts appear directly in your projects—that's your assessment.

## Support Resources

:::{admonition} Getting Help
:class: info

- **Conceptual confusion**: Review the physical intuition section
- **Implementation issues**: Check common pitfalls first
- **Mathematical details**: Office hours for derivations
- **Still stuck**: Post on Slack with specific questions
:::

## Learning Outcomes

After completing all JIT chapters, you will:

✅ **Connect physics to mathematics** - See equations as descriptions of reality  
✅ **Implement algorithms correctly** - Translate math to stable code  
✅ **Debug with understanding** - Recognize numerical vs conceptual errors  
✅ **Read research papers** - Understand methods sections  
✅ **Extend techniques** - Apply concepts to new problems

---

*Remember: These chapters appear exactly when you need them. Trust the process—by the time you need differential equations or Bayesian statistics, you'll be ready to learn them with concrete applications in hand.*