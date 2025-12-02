#!/usr/bin/env python3
"""
Figure 3.2: GP Uncertainty - Interpolation vs Extrapolation

Shows 1D GP regression with training data at x ∈ {1, 3, 5}.
Demonstrates core GP behavior: confident interpolation, uncertain extrapolation.

Location: After line 694 in Part I (N-Body Emulation Example)
Pedagogical Goal: Visualize epistemic uncertainty and why GPs know when they don't know
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Set random seed for reproducibility
np.random.seed(123)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11

# Training data: x ∈ {1, 3, 5}, y values chosen to show non-trivial pattern
x_train = np.array([1.0, 3.0, 5.0]).reshape(-1, 1)
y_train = np.array([0.5, 1.8, 1.2])  # Non-monotonic to show interpolation

# Test points for smooth curve
x_test = np.linspace(-0.5, 6.5, 300).reshape(-1, 1)

# Hyperparameters (chosen for good visualization)
length_scale = 0.8
signal_variance = 1.0
noise_variance = 0.05  # Small noise for clear signal

# SE Kernel
def se_kernel(X1, X2, ell, sigma_f2):
    """Squared Exponential kernel."""
    sqdist = cdist(X1, X2, 'sqeuclidean')
    return sigma_f2 * np.exp(-0.5 * sqdist / ell**2)

# Compute kernel matrices
K_train_train = se_kernel(x_train, x_train, length_scale, signal_variance)
K_train_train += noise_variance * np.eye(len(x_train))  # Add noise

K_test_train = se_kernel(x_test, x_train, length_scale, signal_variance)
K_test_test_diag = signal_variance * np.ones(len(x_test))  # Diagonal of k(x*, x*)

# GP Prediction (using Cholesky decomposition for numerical stability)
L = np.linalg.cholesky(K_train_train)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

# Predictive mean and variance
mu_test = K_test_train @ alpha

# Variance: k** - k* K_y^{-1} k*^T
v = np.linalg.solve(L, K_test_train.T)  # L^{-1} k*
var_test = K_test_test_diag - np.sum(v**2, axis=0)  # Epistemic variance only
std_test = np.sqrt(var_test)

# Total uncertainty (epistemic + aleatoric) for prediction intervals
std_total = np.sqrt(var_test + noise_variance)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Plot uncertainty regions first (so they're behind)
ax.fill_between(x_test.ravel(),
                mu_test - 2*std_test,
                mu_test + 2*std_test,
                alpha=0.2, color='steelblue',
                label=r'$\pm 2\sigma$ epistemic (function uncertainty)')
ax.fill_between(x_test.ravel(),
                mu_test - 2*std_total,
                mu_test + 2*std_total,
                alpha=0.15, color='lightcoral',
                label=r'$\pm 2\sigma$ total (epistemic + noise)')

# Plot predictive mean
ax.plot(x_test, mu_test, 'b-', linewidth=2.5, label=r'Predictive mean $\mu(x)$', zorder=5)

# Plot training data
ax.scatter(x_train, y_train, s=150, c='black', marker='o', edgecolors='white',
          linewidths=2, zorder=10, label='Training data')

# Add vertical lines to mark training locations
for x in x_train.ravel():
    ax.axvline(x, color='gray', linestyle=':', alpha=0.3, linewidth=1)

# Annotate regions
# Interpolation region
ax.annotate('INTERPOLATION REGION\n(confident predictions)',
           xy=(2, -0.5), fontsize=12, ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.6))
ax.annotate('', xy=(1.2, -0.3), xytext=(2.8, -0.3),
           arrowprops=dict(arrowstyle='<->', lw=2, color='green'))

ax.annotate('', xy=(3.2, -0.3), xytext=(4.8, -0.3),
           arrowprops=dict(arrowstyle='<->', lw=2, color='green'))

# Extrapolation regions
ax.annotate('EXTRAPOLATION\n(uncertain)',
           xy=(-0.2, 2.5), fontsize=11, ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.7))
ax.annotate('EXTRAPOLATION\n(uncertain)',
           xy=(6.2, 2.5), fontsize=11, ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.7))

# Add uncertainty arrows showing growth
ax.annotate('', xy=(6.0, mu_test[np.argmin(np.abs(x_test - 6.0))]),
           xytext=(6.0, mu_test[np.argmin(np.abs(x_test - 6.0))] + 2*std_test[np.argmin(np.abs(x_test - 6.0))]),
           arrowprops=dict(arrowstyle='<->', lw=1.5, color='red'))
ax.text(6.15, mu_test[np.argmin(np.abs(x_test - 6.0))] + std_test[np.argmin(np.abs(x_test - 6.0))],
       'Large\nuncertainty', fontsize=10, color='red', va='center')

ax.annotate('', xy=(2.0, mu_test[np.argmin(np.abs(x_test - 2.0))]),
           xytext=(2.0, mu_test[np.argmin(np.abs(x_test - 2.0))] + 2*std_test[np.argmin(np.abs(x_test - 2.0))]),
           arrowprops=dict(arrowstyle='<->', lw=1.5, color='darkgreen'))
ax.text(1.75, mu_test[np.argmin(np.abs(x_test - 2.0))] + std_test[np.argmin(np.abs(x_test - 2.0))],
       'Small\nuncert.', fontsize=10, color='darkgreen', va='center', ha='right')

# Formatting
ax.set_xlabel(r'Input $x$', fontweight='bold')
ax.set_ylabel(r'Output $f(x)$', fontweight='bold')
ax.set_title('GP Uncertainty: Confident Interpolation vs Uncertain Extrapolation',
            fontweight='bold', fontsize=16)
ax.legend(loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-1, 3.5)

# Add explanatory caption
fig.text(0.5, 0.015,
        r'Key Insight: GP uncertainty $\sigma(x)$ is small near training data (interpolation) and grows far from data (extrapolation). '
        r'This automatic uncertainty quantification tells you when predictions are trustworthy!',
        ha='center', fontsize=10, style='italic', wrap=True)

# Save
plt.tight_layout(rect=[0, 0.045, 1, 1])
plt.savefig('fig_3_2_gp_uncertainty.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_3_2_gp_uncertainty.pdf', bbox_inches='tight')

print("✓ Figure saved: fig_3_2_gp_uncertainty.png/pdf")
print(f"  Training points: x = {x_train.ravel()}, y = {y_train}")
print(f"  Hyperparameters: ℓ={length_scale}, σ_f²={signal_variance}, σ_n²={noise_variance}")
print(f"  Min/Max epistemic std: {std_test.min():.3f} / {std_test.max():.3f}")

# Display (optional)
# plt.show()
