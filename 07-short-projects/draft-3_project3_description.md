# ⚠️ ASTR 596 Project 3: Monte Carlo Radiative Transfer + MCMC + Bayesian Statistics
**Duration**: 4 weeks
**Weight**: 18% of course grade
**Theme**: "Rosen (2016) Direct Radiation + Deep Bayesian Inference"

---

## Project Overview

This project implements sophisticated Monte Carlo radiative transfer following the Rosen (2016) methodology for direct stellar radiation, combined with comprehensive Bayesian parameter estimation. You will build a complete pipeline from stellar cluster heating calculations to statistical inference of dust properties. This project emphasizes both physical understanding of radiative processes and modern statistical methods.

## Learning Objectives

By completing this project, you will:
- **Master radiative transfer physics**: Direct radiation field calculation and dust heating
- **Implement Monte Carlo methods**: Photon transport and statistical sampling
- **Understand Bayesian statistics**: Priors, likelihoods, posteriors, and model comparison
- **Develop MCMC expertise**: Multiple sampling algorithms with convergence diagnostics
- **Apply advanced inference**: Parameter estimation for complex astrophysical models
- **Connect theory to observations**: Synthetic observational data analysis

## Prerequisites from Previous Projects
- **Project 1**: Numerical integration (Planck function), root-finding (temperature balance), blackbody physics
- **Project 2**: Stellar cluster snapshots with realistic mass/luminosity distributions
- **Mathematical Tools**: Statistical sampling, error analysis, performance optimization

---

# Week 1: Direct Radiation Monte Carlo Framework

## Conceptual Introduction (30 min)
- **Radiative Transfer Theory**: Detailed mathematical foundation (see extended theory section)
- **Rosen (2016) Innovation**: Direct radiation field vs diffusion approximation
- **Monte Carlo Philosophy**: Statistical approach to complex integral equations
- **Dust Physics**: Absorption, scattering, and thermal re-emission processes

## Lab Session Objectives
Implement direct radiation field calculation using Project 2 stellar clusters.

### Task 1: Direct Radiation Physics Implementation (60 min)
**Goal**: Build foundation for accurate stellar heating calculations

**Core Physics Modules**:
```python
import numpy as np
from scipy.optimize import brentq
import pickle

# Load stellar cluster data from Project 2
def load_cluster_snapshot(filename, snapshot_index=0):
    """Load stellar cluster from Project 2."""
    