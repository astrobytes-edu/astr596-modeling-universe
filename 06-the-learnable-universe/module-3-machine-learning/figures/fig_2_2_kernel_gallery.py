#!/usr/bin/env python3
"""
Figure 2.2: Kernel Gallery - Comparing Common Kernel Functions

Shows SE, Matérn family (ν=1/2, 3/2, 5/2), and Periodic kernels.
For each kernel: (1) correlation function k(r), (2) sample functions.

Location: After kernel discussion in Part I (around line 850)
Pedagogical Goal: Visual comparison of kernel properties and resulting function smoothness
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv

# Set random seed
np.random.seed(42)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Kernel implementations
def squared_exponential(X1, X2, ell=1.0, sigma_f2=1.0):
    """SE/RBF kernel: infinitely smooth."""
    sqdist = cdist(X1, X2, 'sqeuclidean')
    return sigma_f2 * np.exp(-0.5 * sqdist / ell**2)

def matern_kernel(X1, X2, nu, ell=1.0, sigma_f2=1.0):
    """
    Matérn kernel with smoothness parameter nu.
    nu = 1/2, 3/2, 5/2 have closed forms.
    """
    dist = cdist(X1, X2, 'euclidean')

    if nu == 0.5:
        # Matérn-1/2: continuous but not differentiable
        K = sigma_f2 * np.exp(-dist / ell)
    elif nu == 1.5:
        # Matérn-3/2: once differentiable
        scaled_dist = np.sqrt(3) * dist / ell
        K = sigma_f2 * (1 + scaled_dist) * np.exp(-scaled_dist)
    elif nu == 2.5:
        # Matérn-5/2: twice differentiable
        scaled_dist = np.sqrt(5) * dist / ell
        K = sigma_f2 * (1 + scaled_dist + scaled_dist**2 / 3) * np.exp(-scaled_dist)
    else:
        # General case (slower)
        scaled_dist = np.sqrt(2 * nu) * dist / ell
        scaled_dist[scaled_dist == 0] = 1e-8  # Avoid division by zero
        K = sigma_f2 * (2**(1-nu) / gamma(nu)) * (scaled_dist**nu) * kv(nu, scaled_dist)
        K[dist == 0] = sigma_f2  # Fix diagonal

    return K

def periodic_kernel(X1, X2, period=1.0, ell=1.0, sigma_f2=1.0):
    """Periodic kernel for repeating patterns."""
    dist = cdist(X1, X2, 'euclidean')
    return sigma_f2 * np.exp(-2 * np.sin(np.pi * dist / period)**2 / ell**2)

# Kernel definitions
kernels = {
    'SE (RBF)': {
        'func': lambda X1, X2: squared_exponential(X1, X2, ell=0.3, sigma_f2=1.0),
        'desc': 'Infinitely smooth\n(ν → ∞)',
        'color': '#2E86AB'
    },
    'Matérn-1/2': {
        'func': lambda X1, X2: matern_kernel(X1, X2, nu=0.5, ell=0.3, sigma_f2=1.0),
        'desc': 'Continuous only\n(rough, can have kinks)',
        'color': '#A23B72'
    },
    'Matérn-3/2': {
        'func': lambda X1, X2: matern_kernel(X1, X2, nu=1.5, ell=0.3, sigma_f2=1.0),
        'desc': 'Once differentiable\n(C¹)',
        'color': '#F18F01'
    },
    'Matérn-5/2': {
        'func': lambda X1, X2: matern_kernel(X1, X2, nu=2.5, ell=0.3, sigma_f2=1.0),
        'desc': 'Twice differentiable\n(C², good default)',
        'color': '#C73E1D'
    },
    'Periodic': {
        'func': lambda X1, X2: periodic_kernel(X1, X2, period=1.0, ell=0.2, sigma_f2=1.0),
        'desc': 'Repeating pattern\n(period = 1.0)',
        'color': '#6A994E'
    }
}

# Create figure
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 5, hspace=0.40, wspace=0.32)

kernel_names = list(kernels.keys())

# Row 1: Kernel functions k(r) vs distance r
for idx, name in enumerate(kernel_names):
    ax = fig.add_subplot(gs[0, idx])

    # Compute kernel as function of distance
    r = np.linspace(0, 2, 200).reshape(-1, 1)
    X1 = np.array([[0]])
    X2 = r
    k_r = kernels[name]['func'](X1, X2).ravel()

    ax.plot(r, k_r, color=kernels[name]['color'], linewidth=2.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(np.exp(-0.5), color='red', linestyle=':', alpha=0.5, linewidth=1)

    # Mark ℓ (where correlation drops to e^(-0.5) ≈ 0.6 for SE)
    if name == 'SE (RBF)':
        ax.axvline(0.3, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(0.3, 0.5, 'ℓ', color='red', fontsize=10, ha='center')

    ax.set_xlabel(r'Distance $r$', fontsize=10)
    ax.set_ylabel(r'$k(r)$', fontsize=10)
    ax.set_title(f'{name}\n{kernels[name]["desc"]}', fontweight='bold', fontsize=11)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.2, 1.1)
    ax.grid(True, alpha=0.2)

# Row 2-3: Function samples
n_samples = 3
x_test = np.linspace(0, 1, 200).reshape(-1, 1)

for idx, name in enumerate(kernel_names):
    # Compute kernel matrix
    K = kernels[name]['func'](x_test, x_test)
    K += 1e-8 * np.eye(len(x_test))  # Jitter

    # Sample functions
    L = np.linalg.cholesky(K)
    samples = L @ np.random.randn(len(x_test), n_samples)

    # Plot samples (row 2)
    ax2 = fig.add_subplot(gs[1, idx])
    for i in range(n_samples):
        ax2.plot(x_test, samples[:, i], alpha=0.7, linewidth=1.5,
                color=kernels[name]['color'])
    ax2.set_ylabel(r'$f(x)$', fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, alpha=0.2)
    ax2.set_title('Sample Functions', fontsize=10)

    # Plot confidence bands (row 3)
    ax3 = fig.add_subplot(gs[2, idx])
    mean = np.zeros(len(x_test))
    std = np.sqrt(np.diag(K))

    ax3.fill_between(x_test.ravel(), mean - 2*std, mean + 2*std,
                     alpha=0.3, color=kernels[name]['color'], label=r'$\pm 2\sigma$')
    ax3.plot(x_test, mean, 'k--', linewidth=2, label='Mean (0)')
    ax3.set_xlabel(r'Input $x$', fontsize=10)
    ax3.set_ylabel('Prior distribution', fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-3, 3)
    ax3.grid(True, alpha=0.2)
    ax3.set_title(r'Prior $\pm 2\sigma$ Bands', fontsize=10)
    if idx == 0:
        ax3.legend(loc='upper left', fontsize=8)

# Overall title
fig.suptitle('Kernel Gallery: Common GP Kernels and Their Properties',
             fontsize=16, fontweight='bold', y=0.985)

# Add explanatory caption
fig.text(0.5, 0.012,
        r'Top row: Kernel correlation $k(r)$ vs distance. Middle row: Random function samples showing smoothness. '
        r'Bottom row: Prior confidence bands. SE is infinitely smooth (wiggle-free). Matérn-1/2 is rough (can have kinks). '
        r'Matérn-5/2 balances smoothness with realism (recommended default). Periodic captures repeating patterns.',
        ha='center', fontsize=9.5, style='italic', wrap=True)

# Adjust layout to accommodate title and caption
plt.subplots_adjust(top=0.94, bottom=0.06)

# Save
plt.savefig('fig_2_2_kernel_gallery.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_2_2_kernel_gallery.pdf', bbox_inches='tight')

print("✓ Figure saved: fig_2_2_kernel_gallery.png/pdf")
print("  Kernels shown: SE, Matérn-1/2, Matérn-3/2, Matérn-5/2, Periodic")
print("  Each kernel: correlation function + sample functions + prior bands")
print("  Key insight: Smoothness parameter ν controls differentiability")

# Display (optional)
# plt.show()
