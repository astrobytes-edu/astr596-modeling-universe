# Final Project Overview: From Simulation to Surrogate

The culminating project for ASTR 596 where you build a neural network emulator for your N-body simulations and use it for Bayesian inference.

## Project Overview

This capstone project brings your semester full circle, synthesizing everything you've learned:

- **Surrogate Modeling**: Train a neural network to approximate expensive N-body simulations
- **JAX Ecosystem**: Master Equinox (NNs), Optax (optimization), and NumPyro (probabilistic programming)
- **Bayesian Inference**: Use your fast emulator to solve the inverse problem with NUTS
- **Research-Quality Output**: Professional code package and research memo

:::{admonition} The Big Idea
:class: tip
Your N-body simulations are too slow for Bayesian inference (thousands of evaluations). Solution: train a neural network that predicts simulation outcomes in milliseconds, then use it inside NumPyro to infer what initial conditions produced an observed cluster state.
:::

## Learning Objectives

By completing this project, you will:

1. **Design training data** using Latin Hypercube Sampling for efficient parameter space coverage
2. **Build neural network emulators** using Equinox and Optax
3. **Quantify uncertainty** through ensemble methods
4. **Perform Bayesian inference** with NumPyro's NUTS sampler
5. **Produce research-quality work** with professional code structure and documentation

## The Scientific Question

Given the final state of a star cluster—its bound mass fraction, velocity dispersion, and spatial extent—can we infer what initial conditions produced it? You'll vary the initial virial ratio $Q_0$ and Plummer scale radius $a$, train an emulator on the resulting summary statistics, and then recover parameters from held-out "observations."

## Project Components

### Part 1: Generate Training Data

- Run 80–100 N-body simulations using your Project 5 JAX package
- Vary initial conditions $(Q_0, a)$ using Latin Hypercube Sampling
- Compute summary statistics: $f_{\rm bound}$, $\sigma_v$, $r_h$

### Part 2: Neural Network Emulator

- Build an MLP using Equinox
- Train with Optax (Adam optimizer)
- Implement ensemble uncertainty quantification

### Part 3: Evaluate Your Emulator

- Compute accuracy metrics (MAE, RMSE)
- Visualize predicted vs. true values
- Analyze uncertainty and edge behavior

### Part 4: Inference with NumPyro

- Build a probabilistic model with your emulator as the forward model
- Run NUTS to sample the posterior over initial conditions
- Validate parameter recovery on held-out simulations

### Part 5: Package & Document

- Organize as an installable Python package
- Write a research memo with methods, results, and figures

## Resources

- [Project Description](final-project-description.md) — Full requirements, rubric, and code skeletons
- [Equinox Documentation](https://docs.kidger.site/equinox/) — Neural networks as PyTrees
- [Optax Documentation](https://optax.readthedocs.io/) — Gradient-based optimization
- [NumPyro Documentation](https://num.pyro.ai/) — Probabilistic programming

## Timeline

| Week | Focus | Goal |
|------|-------|------|
| Week 1 | Training Data + Emulator | Data generated, NN training |
| Week 2 | Evaluation + Inference | NumPyro pipeline working |
| Week 2.5 | Polish | Package complete, memo submitted |

:::{admonition} Due Date
:class: warning
**Thursday, December 18, 2025, 11:59 PM**

Submit via GitHub Classroom. No late submissions accepted.
:::

## What You'll Emerge With

This project teaches the workflow of modern computational science: expensive simulations → machine learning surrogate → probabilistic inference. The JAX ecosystem you're learning (Equinox, Optax, NumPyro) represents the frontier of scientific ML. You'll finish the course with both deep understanding and practical skills for research careers in astrophysics, data science, and beyond.
