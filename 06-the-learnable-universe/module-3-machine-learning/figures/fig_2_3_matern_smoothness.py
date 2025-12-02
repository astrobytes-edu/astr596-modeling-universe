#!/usr/bin/env python3
"""
Figure 2.3: Matérn Smoothness Comparison - Effect of ν Parameter

Compares Matérn-1/2, 3/2, 5/2 kernels showing how ν controls differentiability.
For each: function samples + first derivatives to visualize smoothness.

Location: After Matérn kernel discussion in Part I (around line 880)
Pedagogical Goal: Show why Matérn-5/2 is often preferred over SE for physics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Set random seed
np.random.seed(789)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

def matern_kernel(X1, X2, nu, ell=0.3, sigma_f2=1.0):
    """Matérn kernel with smoothness parameter nu."""
    dist = cdist(X1, X2, 'euclidean')

    if nu == 0.5:
        K = sigma_f2 * np.exp(-dist / ell)
    elif nu == 1.5:
        scaled_dist = np.sqrt(3) * dist / ell
        K = sigma_f2 * (1 + scaled_dist) * np.exp(-scaled_dist)
    elif nu == 2.5:
        scaled_dist = np.sqrt(5) * dist / ell
        K = sigma_f2 * (1 + scaled_dist + scaled_dist**2 / 3) * np.exp(-scaled_dist)
    else:
        raise ValueError(f"nu={nu} not implemented")

    return K

def matern_derivative_kernel(X1, X2, nu, ell=0.3, sigma_f2=1.0):
    """
    Derivative of Matérn kernel: cov(f'(x), f'(x')).

    For stationary kernels: k''(r) where r = |x - x'|.
    This gives the covariance of derivatives.
    """
    dist = cdist(X1, X2, 'euclidean')
    dist[dist == 0] = 1e-10  # Avoid division by zero

    if nu == 0.5:
        # Matérn-1/2: not differentiable! Set to large variance (noisy derivative)
        # In practice, derivative doesn't exist
        K = np.zeros_like(dist)
    elif nu == 1.5:
        # Matérn-3/2: once differentiable
        scaled_dist = np.sqrt(3) * dist / ell
        K = sigma_f2 * (3 / ell**2) * scaled_dist * np.exp(-scaled_dist)
    elif nu == 2.5:
        # Matérn-5/2: twice differentiable
        scaled_dist = np.sqrt(5) * dist / ell
        K = sigma_f2 * (5 / ell**2) * (1 + scaled_dist) * (1 - scaled_dist / 3) * np.exp(-scaled_dist)
    else:
        raise ValueError(f"nu={nu} not implemented")

    return K

# Matérn variants
matern_configs = [
    {'nu': 0.5, 'name': 'Matérn-1/2', 'color': '#A23B72',
     'desc': 'ν = 1/2: Continuous\nNOT differentiable', 'differentiable': 0},
    {'nu': 1.5, 'name': 'Matérn-3/2', 'color': '#F18F01',
     'desc': 'ν = 3/2: Once\ndifferentiable (C¹)', 'differentiable': 1},
    {'nu': 2.5, 'name': 'Matérn-5/2', 'color': '#6A994E',
     'desc': 'ν = 5/2: Twice\ndifferentiable (C²)', 'differentiable': 2},
]

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

n_samples = 4
n_test = 300
x_test = np.linspace(0, 1, n_test).reshape(-1, 1)

for idx, config in enumerate(matern_configs):
    nu = config['nu']
    color = config['color']

    # Compute kernel matrix
    K = matern_kernel(x_test, x_test, nu=nu, ell=0.15, sigma_f2=1.0)
    K += 1e-8 * np.eye(n_test)

    # Sample functions
    L = np.linalg.cholesky(K)
    f_samples = L @ np.random.randn(n_test, n_samples)

    # Row 1: Function samples
    ax1 = axes[0, idx]
    for i in range(n_samples):
        ax1.plot(x_test, f_samples[:, i], alpha=0.7, linewidth=2, color=color)
    ax1.set_ylabel(r'$f(x)$', fontweight='bold')
    ax1.set_title(f"{config['name']}\n{config['desc']}", fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-3, 3)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)

    # Row 2: Numerical derivatives (finite differences)
    ax2 = axes[1, idx]
    dx = x_test[1] - x_test[0]
    for i in range(n_samples):
        df = np.gradient(f_samples[:, i], dx.item())
        ax2.plot(x_test, df, alpha=0.7, linewidth=2, color=color)

    ax2.set_ylabel(r"$f'(x)$ (numerical)", fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)

    # Add annotation about roughness
    if nu == 0.5:
        ax2.text(0.5, 0.95, '⚠ Rough! Kinks visible',
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax2.set_ylim(-30, 30)
    elif nu == 1.5:
        ax2.text(0.5, 0.95, r"Smooth $f'$, rough $f''$",
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax2.set_ylim(-15, 15)
    else:
        ax2.text(0.5, 0.95, '✓ Very smooth',
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax2.set_ylim(-10, 10)

    # Row 3: Kernel correlation function k(r)
    ax3 = axes[2, idx]
    r = np.linspace(0, 1, 200).reshape(-1, 1)
    X1 = np.array([[0]])
    k_r = matern_kernel(X1, r, nu=nu, ell=0.15, sigma_f2=1.0).ravel()

    ax3.plot(r, k_r, color=color, linewidth=3, label=rf'$k(r)$, $\nu$={nu}')
    ax3.axhline(np.exp(-0.5), color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax3.text(0.8, np.exp(-0.5) + 0.05, r'$e^{-0.5} \approx 0.6$', fontsize=9, color='red')

    ax3.set_xlabel(r'Distance $r$', fontweight='bold')
    ax3.set_ylabel(r'Correlation $k(r)$', fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)

    # Mark lengthscale (approximate, varies by ν)
    ax3.axvline(0.15, color='gray', linestyle=':', alpha=0.5)
    ax3.text(0.15, 0.5, 'ℓ', fontsize=11, color='gray', ha='center')

# Overall title
fig.suptitle(r'Matérn Smoothness Comparison: How $\nu$ Controls Differentiability',
             fontsize=16, fontweight='bold', y=0.995)

# Add explanatory caption
fig.text(0.5, 0.012,
        r"Top row: Function samples $f(x)$. Middle row: Numerical derivatives $f'(x)$ reveal roughness—note how $\nu$=1/2 shows kinks (not differentiable), "
        r'while $\nu$=5/2 is very smooth. Bottom row: Kernel correlation $k(r)$ shows how quickly correlations decay. '
        r'Matérn-5/2 is recommended default: smooth enough for most physics, more realistic than infinitely-smooth SE.',
        ha='center', fontsize=9.5, style='italic', wrap=True)

plt.tight_layout(rect=[0, 0.04, 1, 0.985])

# Save
plt.savefig('fig_2_3_matern_smoothness.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_2_3_matern_smoothness.pdf', bbox_inches='tight')

print("✓ Figure saved: fig_2_3_matern_smoothness.png/pdf")
print("  Comparison: Matérn-1/2, 3/2, 5/2")
print("  For each: function samples + derivatives + correlation k(r)")
print("  Key insight: ν=1/2 rough (kinks), ν=3/2 smooth, ν=5/2 very smooth (recommended)")
print("  Practical advice: Use Matérn-5/2 as default for physics emulation")

# Display (optional)
# plt.show()
