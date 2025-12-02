#!/usr/bin/env python3
"""
Figure 2.1: GP Prior Samples - How Lengthscale Controls Smoothness

Shows random function samples from GP(0, k_SE) with different lengthscales.
Demonstrates intuition: small ℓ → wiggly functions, large ℓ → smooth functions.

Location: After line 318 in Part I
Pedagogical Goal: Build intuition for lengthscale parameter before diving into math
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Define lengthscales to explore
lengthscales = [0.1, 0.3, 1.0]
n_samples = 5  # Number of random function samples per lengthscale
n_test = 200   # Number of test points for smooth curves

# Generate test points
x_test = np.linspace(0, 1, n_test).reshape(-1, 1)

# SE Kernel function
def squared_exponential_kernel(X1, X2, length_scale, signal_variance=1.0):
    """
    Squared Exponential (RBF) kernel.

    k(x, x') = σ_f² exp(-||x - x'||² / (2ℓ²))
    """
    sqdist = cdist(X1, X2, 'sqeuclidean')
    return signal_variance * np.exp(-0.5 * sqdist / length_scale**2)

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('GP Prior Samples: How Lengthscale Controls Function Smoothness',
             fontsize=16, fontweight='bold', y=0.98)

for idx, ell in enumerate(lengthscales):
    # Compute kernel matrix
    K = squared_exponential_kernel(x_test, x_test, length_scale=ell)

    # Add small jitter for numerical stability
    K += 1e-8 * np.eye(n_test)

    # Sample functions from GP prior: f ~ N(0, K)
    # Using Cholesky decomposition: f = L @ z where L L^T = K, z ~ N(0, I)
    L = np.linalg.cholesky(K)
    samples = L @ np.random.randn(n_test, n_samples)

    # Row 1: Plot individual function samples
    ax1 = axes[0, idx]
    for i in range(n_samples):
        ax1.plot(x_test, samples[:, i], alpha=0.7, linewidth=1.5,
                label=f'Sample {i+1}' if idx == 0 and i < 3 else '')
    ax1.set_title(rf'Lengthscale $\ell$ = {ell}', fontweight='bold')
    ax1.set_xlabel(r'Input $x$')
    ax1.set_ylabel(r'Function value $f(x)$')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Add annotation about smoothness
    if ell == 0.1:
        ax1.text(0.5, 0.95, 'Wiggly (high frequency)',
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    elif ell == 1.0:
        ax1.text(0.5, 0.95, 'Smooth (low frequency)',
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Row 2: Plot mean ± 2σ confidence bands
    ax2 = axes[1, idx]
    mean = np.zeros(n_test)  # Prior mean is zero
    std = np.sqrt(np.diag(K))  # Prior std from kernel diagonal

    ax2.fill_between(x_test.ravel(), mean - 2*std, mean + 2*std,
                     alpha=0.3, label=r'$\pm 2\sigma$ (95% confidence)', color='steelblue')
    ax2.plot(x_test, mean, 'k--', linewidth=2, label='Mean (0)')
    ax2.set_xlabel(r'Input $x$')
    ax2.set_ylabel('Prior distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.legend(loc='upper right', fontsize=8)

    # Add vertical correlation length visualization
    # At x=0.5, show where correlation drops to ~0.6 (e^{-0.5})
    corr_dist = ell  # Distance where k drops to e^{-0.5} ≈ 0.6
    if 0.5 + corr_dist <= 1.0:
        ax2.axvline(0.5, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax2.axvline(0.5 + corr_dist, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax2.annotate('', xy=(0.5 + corr_dist, -1.5), xytext=(0.5, -1.5),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax2.text(0.5 + corr_dist/2, -1.8, rf'$\ell$={ell}', ha='center', color='red', fontsize=9)

# Add legend to first plot only
axes[0, 0].legend(loc='upper left', fontsize=8)

# Add explanatory text at bottom (before tight_layout)
fig.text(0.5, 0.015,
         'Key Insight: Small lengthscales → functions vary rapidly (need dense training data). '
         'Large lengthscales → functions vary slowly (sparse data sufficient).',
         ha='center', fontsize=10, style='italic', wrap=True)

# Overall layout - leave space for suptitle (top) and caption (bottom)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])

# Save figure
plt.savefig('fig_2_1_gp_prior_samples.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_2_1_gp_prior_samples.pdf', bbox_inches='tight')
print("✓ Figure saved: fig_2_1_gp_prior_samples.png/pdf")
print(f"  Shape: {fig.get_size_inches()} inches")
print(f"  Lengthscales shown: {lengthscales}")
print(f"  Samples per lengthscale: {n_samples}")

# Display (optional - comment out if running headless)
# plt.show()
