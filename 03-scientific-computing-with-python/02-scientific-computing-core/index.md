---
title: Module Overview
subtitle: Scientific Computing Core | Modeling the Universe
exports:
  - format: pdf
---

## Deepening Your Computational Toolkit

This module builds on the `Python Fundamentals` material and teaches the high-performance numerical tools and workflows used in real research: array programming with NumPy, visualization with Matplotlib, robust numerical computing practices, and performance-aware algorithms.

:::{tip} 📚 How to Use This Module

Treat these chapters as practical, reusable resources: read the concept sections, run the code examples in IPython, complete the exercises to solidify performance and numerical best practices, and refer back to these pages often as your go-to references during assignments and projects.
:::

## Module Overview

### [🧮 Chapter 7: NumPy - The Foundation of Scientific Computing in Python](./07-python-numpy-fundamentals-v2.md)

NumPy arrays, vectorization, broadcasting, indexing, and memory/layout considerations — the foundation for efficient astronomical computation.

### [🖼️ Chapter 8: Matplotlib - Visualizing Your Universe](./08-matplotlib-fundamentals-v1.md)

Create publication-quality figures from NumPy arrays: line plots, images, colormaps, multi-panel layouts, and WCS-aware displays for astronomical data.

### [🔧 Chapter 9: Robust Numerical Computing & Best Practices](./09-python-robust-computing-ORIG.md)

Numerical stability, defensive algorithms, error propagation, and techniques for writing reliable scientific code that produces trustworthy results.

### [🚀 Chapter 10: Advanced Patterns & Performance](./10-python-advanced-oop-ORIG.md)

Advanced object-oriented patterns, performance profiling, C/Fortran integration paths, and packaging/testing strategies for production-grade scientific software.

## Learning Strategy

1. **Run examples interactively** to observe performance differences between naive and vectorized implementations.
2. **Profile before optimizing** — use small benchmarks to find hotspots.
3. **Prefer NumPy idioms** (vectorize, mask, broadcast) over Python loops for data-heavy tasks.
4. **Write tests** for numerical code to guard against subtle regressions.

## Quick Navigation Guide

- Fix performance bottlenecks → [Ch 7: NumPy](./07-python-numpy-fundamentals-v2.md)
- Improve plots and figures → [Ch 8: Matplotlib](./08-matplotlib-fundamentals-v1.md)
- Ensure numerical stability → [Ch 9: Robust Computing](./09-python-robust-computing-ORIG.md)
- Productionize code → [Ch 10: Advanced Patterns](./10-python-advanced-oop-ORIG.md)

## Core Competencies

- Efficient array programming with NumPy
- Vectorized algorithms and broadcasting
- Memory-aware data handling (views vs copies)
- Numerical stability and defensive programming
- Creating publication-quality visualizations with Matplotlib
- Profiling and optimizing performance-critical code
- Packaging and testing scientific libraries

---

**Next:** Start with [Chapter 7: NumPy](./07-python-numpy-fundamentals-v2.md)
