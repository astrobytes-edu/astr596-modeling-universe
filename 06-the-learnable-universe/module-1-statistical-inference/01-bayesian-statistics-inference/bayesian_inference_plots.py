"""
Educational Plots for ASTR 596 Module 5, Part 2
Generate figures illustrating Bayesian inference concepts

Run this script to create all plots, then insert them as images in course materials.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from pathlib import Path

# Set output directory
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style for clean, publication-quality plots with consistent appearance
plt.style.use('seaborn-v0_8-darkgrid')
STYLE_CONFIG = {
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',  # Light gray background
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
}
plt.rcParams.update(STYLE_CONFIG)

# ==============================================================================
# PLOT 1: Likelihood Function (Section 2.2)
# Shows how likelihood peaks at MLE and its width represents constraint
# ==============================================================================

def plot_likelihood_function():
    """Illustrate the likelihood function for Cepheid distance."""
    
    # Observed data
    period = 10.0  # days
    observed_mag = 25.3
    mag_error = 0.05
    
    # Physics model: P-L relation + distance modulus
    def predicted_magnitude(distance, extinction=0.0):
        absolute_mag = -2.43 * (np.log10(period) - 1) - 4.05
        return absolute_mag + 5*np.log10(distance) - 5 + extinction
    
    # Likelihood function
    def likelihood(distance, extinction=0.0):
        pred_mag = predicted_magnitude(distance, extinction)
        total_sigma = np.sqrt(mag_error**2 + 0.15**2)
        return np.exp(-0.5 * ((observed_mag - pred_mag)/total_sigma)**2)
    
    # Evaluate over distance range
    distances = np.linspace(1e6, 5e6, 500)  # parsecs
    likelihoods = np.array([likelihood(d) for d in distances])

    # Normalize likelihood to peak = 1.0 for clarity
    likelihoods_norm = likelihoods / np.max(likelihoods)

    # Find peak and confidence intervals
    peak_idx = np.argmax(likelihoods_norm)
    peak_distance = distances[peak_idx]

    # Find 68% (1-sigma) confidence interval
    # Integrate from edges until we reach 68% of total probability
    cumulative = np.cumsum(likelihoods_norm)
    cumulative /= cumulative[-1]

    # Find indices where cumulative probability crosses 16% and 84%
    idx_low = np.argmin(np.abs(cumulative - 0.16))
    idx_high = np.argmin(np.abs(cumulative - 0.84))
    dist_low = distances[idx_low] / 1e6
    dist_high = distances[idx_high] / 1e6

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot normalized likelihood
    ax.plot(distances/1e6, likelihoods_norm, 'b-', linewidth=2.5, label='Likelihood (normalized)')

    # Shade 68% confidence region
    mask = (distances >= distances[idx_low]) & (distances <= distances[idx_high])
    ax.fill_between(distances[mask]/1e6, 0, likelihoods_norm[mask],
                     alpha=0.25, color='blue', label='68% confidence region')

    # Mark peak
    ax.axvline(peak_distance/1e6, color='red', linestyle='--',
               linewidth=2, label=f'MLE: {peak_distance/1e6:.2f} Mpc')

    # Mark confidence interval bounds
    ax.axvline(dist_low, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(dist_high, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Distance (Mpc)', fontsize=12)
    ax.set_ylabel('Normalized Likelihood  P(data | distance)', fontsize=12)
    ax.set_title('Likelihood Function for Cepheid Distance Measurement', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Add annotation
    ax.annotate('Peak = Most likely distance\ngiven the data',
                xy=(peak_distance/1e6, 1.0),
                xytext=(peak_distance/1e6 + 0.5, 0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red')

    ax.annotate(f'Uncertainty:\n{dist_low:.2f} – {dist_high:.2f} Mpc',
                xy=(peak_distance/1e6, 0.3),
                xytext=(2.5, 0.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                fontsize=9, color='gray',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig_part2_likelihood_1d.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ==============================================================================
# PLOT 2: Distance-Extinction Degeneracy (Section 2.2)
# Shows how likelihood has degeneracies in 2D parameter space
# ==============================================================================

def plot_degeneracy():
    """Illustrate distance-extinction degeneracy and how priors break it."""

    period = 10.0
    observed_mag = 25.3
    mag_error = 0.05

    def likelihood_2d(distance, extinction):
        absolute_mag = -2.43 * (np.log10(period) - 1) - 4.05
        pred_mag = absolute_mag + 5*np.log10(distance) - 5 + extinction
        total_sigma = np.sqrt(mag_error**2 + 0.15**2)
        return np.exp(-0.5 * ((observed_mag - pred_mag)/total_sigma)**2)

    # Create grid
    distances = np.linspace(2.5e6, 3.5e6, 200)
    extinctions = np.linspace(0, 0.6, 200)
    D, A = np.meshgrid(distances, extinctions)

    # Evaluate likelihood
    L = np.vectorize(likelihood_2d)(D, A)
    L_norm = L / np.max(L)  # Normalize

    # Define priors
    # Gaussian prior on distance (from previous observations)
    d_prior_mean, d_prior_std = 3.0e6, 0.15e6  # 3.0 ± 0.15 Mpc
    prior_d = np.exp(-0.5 * ((distances - d_prior_mean) / d_prior_std)**2)

    # Exponential prior on extinction (most stars have low extinction)
    a_scale = 0.15  # mag
    prior_a = (1/a_scale) * np.exp(-extinctions / a_scale)

    # 2D prior
    Prior = np.outer(prior_a, prior_d)
    Prior /= np.max(Prior)

    # Posterior
    Posterior = L_norm * Prior
    Posterior /= np.max(Posterior)

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # LEFT PANEL: Likelihood only (shows degeneracy)
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
    cs1 = ax1.contourf(D/1e6, A, L_norm, levels=levels, cmap='viridis', alpha=0.8)
    ax1.contour(D/1e6, A, L_norm, levels=levels, colors='black', alpha=0.4, linewidths=0.8)

    # Highlight degeneracy ridge with thick white outline
    max_indices = np.argmax(L_norm, axis=0)
    ridge_extinctions = extinctions[max_indices]
    ax1.plot(distances/1e6, ridge_extinctions, 'w-', linewidth=5, alpha=0.8)
    ax1.plot(distances/1e6, ridge_extinctions, 'r-', linewidth=3,
            label='Degeneracy ridge')

    ax1.set_xlabel('Distance (Mpc)', fontsize=12)
    ax1.set_ylabel('Extinction $A_V$ (mag)', fontsize=12)
    ax1.set_title('Likelihood Only\n(Distance-Extinction Degeneracy)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Annotations positioned ON the ridge
    ax1.annotate('Close + dusty', xy=(2.65, 0.47), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'))
    ax1.annotate('Far + clear', xy=(3.25, 0.15), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='black'))

    # Add colorbar for left panel
    cbar1 = plt.colorbar(cs1, ax=ax1, label='Normalized Likelihood')

    # RIGHT PANEL: Posterior (degeneracy broken by prior)
    cs2 = ax2.contourf(D/1e6, A, Posterior, levels=levels, cmap='YlOrRd', alpha=0.8)
    ax2.contour(D/1e6, A, Posterior, levels=levels, colors='black', alpha=0.4, linewidths=0.8)

    # Mark the peak of posterior
    peak_idx = np.unravel_index(np.argmax(Posterior), Posterior.shape)
    peak_d = D[peak_idx] / 1e6
    peak_a = A[peak_idx]
    ax2.plot(peak_d, peak_a, 'r*', markersize=20, markeredgecolor='white',
             markeredgewidth=2, label='Posterior peak')

    # Show prior contours in blue
    prior_levels = [0.3, 0.6, 0.9]
    ax2.contour(D/1e6, A, Prior, levels=prior_levels, colors='blue',
                linestyles='--', linewidths=2, alpha=0.6)

    ax2.set_xlabel('Distance (Mpc)', fontsize=12)
    ax2.set_ylabel('Extinction $A_V$ (mag)', fontsize=12)
    ax2.set_title('Posterior = Likelihood × Prior\n(Degeneracy Broken)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add colorbar for right panel
    cbar2 = plt.colorbar(cs2, ax=ax2, label='Normalized Posterior')

    # Legend showing prior influence
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', linestyle='--', linewidth=2),
                    Line2D([0], [0], marker='*', color='w', markerfacecolor='r',
                           markersize=15, markeredgecolor='white')]
    ax2.legend(custom_lines, ['Prior contours', 'Posterior peak'],
               loc='upper left', fontsize=10)

    plt.suptitle('How Priors Break Parameter Degeneracies',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig_part2_degeneracy_2d.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ==============================================================================
# PLOT 3: Types of Priors (Section 2.3)
# Compare weak, informative, and physical constraint priors
# ==============================================================================

def plot_prior_types():
    """Illustrate different types of priors."""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 1. Uniform (uninformative) prior
    ax = axes[0, 0]
    x = np.linspace(0, 100, 1000)
    prior = np.ones_like(x) / 100
    ax.fill_between(x, 0, prior, alpha=0.3, color='gray')
    ax.plot(x, prior, 'k-', linewidth=2)
    ax.set_title('Uniform Prior\n"Uninformative"', fontweight='bold')
    ax.set_xlabel('Distance (kpc)')
    ax.set_ylabel('P(distance)')
    ax.text(50, 0.012, 'All distances\nequally plausible', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.set_ylim([0, 0.015])
    
    # 2. Gaussian (informative) prior
    ax = axes[0, 1]
    x = np.linspace(0, 100, 1000)
    mean, sigma = 50, 2
    prior = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/sigma)**2)
    ax.fill_between(x, 0, prior, alpha=0.3, color='blue')
    ax.plot(x, prior, 'b-', linewidth=2)
    ax.set_title('Gaussian Prior\n"Informative" (from previous data)', fontweight='bold')
    ax.set_xlabel('Distance (kpc)')
    ax.set_ylabel('P(distance)')
    ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean} kpc')
    ax.text(50, 0.15, f'σ = {sigma} kpc', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.legend()
    
    # 3. Exponential (population) prior
    ax = axes[1, 0]
    x = np.linspace(0, 3, 1000)
    scale = 0.3
    prior = (1/scale) * np.exp(-x/scale)
    ax.fill_between(x, 0, prior, alpha=0.3, color='green')
    ax.plot(x, prior, 'g-', linewidth=2)
    ax.set_title('Exponential Prior\n"Population model" (e.g., extinction)', fontweight='bold')
    ax.set_xlabel('Extinction A_V (mag)')
    ax.set_ylabel('P(extinction)')
    ax.text(1.5, 2, 'Low values\nmore common', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # 4. Jeffrey's prior (scale-invariant for distances)
    ax = axes[1, 1]
    x = np.linspace(1, 100, 1000)  # Start at 1 to avoid singularity at 0
    # Jeffrey's prior for distance: P(d) ∝ 1/d
    prior = 1.0 / x
    # Normalize over the range (improper prior, but we normalize for display)
    prior /= np.trapezoid(prior, x)

    ax.fill_between(x, 0, prior, alpha=0.3, color='purple')
    ax.plot(x, prior, color='purple', linewidth=2)
    ax.set_title("Jeffrey's Prior\n\"Scale-invariant\"", fontweight='bold')
    ax.set_xlabel('Distance (kpc)')
    ax.set_ylabel('P(distance)')
    ax.text(30, np.max(prior)*0.6,
            r'$P(d) \propto 1/d$' + '\n(favors closer distances)\nScale-invariant transformation',
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    ax.set_xlim([0, 100])
    ax.set_ylim(0, np.max(prior)*1.1)
    
    plt.suptitle('Types of Priors in Bayesian Astronomy',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig_part2_prior_types.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ==============================================================================
# PLOT 4: Prior vs Data Strength (Section 2.3)
# Show when prior dominates vs when data dominates
# ==============================================================================

def plot_prior_data_balance():
    """Illustrate the balance between prior and data strength."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.linspace(40, 60, 1000)
    dx = x[1] - x[0]

    # Prior (informative)
    prior_mean, prior_std = 50, 2
    prior = (1/(prior_std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-prior_mean)/prior_std)**2)

    # Set consistent y-axis limits for all panels
    y_max = 0.25

    # Scenario 1: Strong prior, weak data
    ax = axes[0]
    data_mean, data_std = 47, 5  # Wide likelihood
    likelihood = (1/(data_std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-data_mean)/data_std)**2)
    posterior = prior * likelihood
    posterior = posterior / (np.sum(posterior) * dx)  # Properly normalize

    # Calculate posterior stats
    post_mean = np.sum(x * posterior * dx)
    post_std = np.sqrt(np.sum((x - post_mean)**2 * posterior * dx))

    ax.plot(x, prior, 'b-', linewidth=2, label='Prior (strong)')
    ax.plot(x, likelihood, 'g--', linewidth=2, label='Likelihood (weak)')
    ax.plot(x, posterior, 'r-', linewidth=3, label='Posterior')
    ax.axvline(prior_mean, color='blue', linestyle=':', alpha=0.5)
    ax.axvline(data_mean, color='green', linestyle=':', alpha=0.5)
    ax.set_title('Prior Dominates', fontsize=12)
    ax.set_xlabel('Distance (kpc)')
    ax.set_ylabel('Probability Density')
    ax.legend(fontsize=9)
    ax.set_ylim(0, y_max)
    ax.text(0.05, 0.95, f'σ_prior = {prior_std:.1f} kpc\nσ_data = {data_std:.1f} kpc\nσ_post = {post_std:.1f} kpc',
            transform=ax.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Scenario 2: Balanced
    ax = axes[1]
    data_mean, data_std = 48, 2  # Similar width
    likelihood = (1/(data_std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-data_mean)/data_std)**2)
    posterior = prior * likelihood
    posterior = posterior / (np.sum(posterior) * dx)

    # Calculate posterior stats
    post_mean = np.sum(x * posterior * dx)
    post_std = np.sqrt(np.sum((x - post_mean)**2 * posterior * dx))

    ax.plot(x, prior, 'b-', linewidth=2, label='Prior')
    ax.plot(x, likelihood, 'g--', linewidth=2, label='Likelihood')
    ax.plot(x, posterior, 'r-', linewidth=3, label='Posterior')
    ax.axvline(prior_mean, color='blue', linestyle=':', alpha=0.5)
    ax.axvline(data_mean, color='green', linestyle=':', alpha=0.5)
    ax.set_title('Balanced Information', fontsize=12)
    ax.set_xlabel('Distance (kpc)')
    ax.set_ylabel('Probability Density')
    ax.legend(fontsize=9)
    ax.set_ylim(0, y_max)
    ax.text(0.05, 0.95, f'σ_prior = {prior_std:.1f} kpc\nσ_data = {data_std:.1f} kpc\nσ_post = {post_std:.1f} kpc',
            transform=ax.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Scenario 3: Weak prior, strong data
    ax = axes[2]
    data_mean, data_std = 47, 0.5  # Narrow likelihood
    likelihood = (1/(data_std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-data_mean)/data_std)**2)
    posterior = prior * likelihood
    posterior = posterior / (np.sum(posterior) * dx)

    # Calculate posterior stats
    post_mean = np.sum(x * posterior * dx)
    post_std = np.sqrt(np.sum((x - post_mean)**2 * posterior * dx))

    ax.plot(x, prior, 'b-', linewidth=2, label='Prior (weak)')
    ax.plot(x, likelihood, 'g--', linewidth=2, label='Likelihood (strong)')
    ax.plot(x, posterior, 'r-', linewidth=3, label='Posterior')
    ax.axvline(prior_mean, color='blue', linestyle=':', alpha=0.5)
    ax.axvline(data_mean, color='green', linestyle=':', alpha=0.5)
    ax.set_title('Data Dominates', fontsize=12)
    ax.set_xlabel('Distance (kpc)')
    ax.set_ylabel('Probability Density')
    ax.legend(fontsize=9)
    ax.set_ylim(0, y_max)
    ax.text(0.05, 0.95, f'σ_prior = {prior_std:.1f} kpc\nσ_data = {data_std:.1f} kpc\nσ_post = {post_std:.1f} kpc',
            transform=ax.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle('The Balance Between Prior Knowledge and New Data',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig_part2_prior_data_balance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ==============================================================================
# PLOT 5: Bayesian Updating (Section 2.4)
# Visual representation of Bayes' theorem: prior × likelihood = posterior
# ==============================================================================

def plot_bayesian_updating():
    """Illustrate Bayesian updating: Prior × Likelihood = Posterior."""

    fig = plt.figure(figsize=(8, 10), constrained_layout=True)
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, figure=fig)

    x = np.linspace(40, 60, 1000)
    dx = x[1] - x[0]

    # Prior
    prior_mean, prior_std = 50, 2
    prior = (1/(prior_std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-prior_mean)/prior_std)**2)

    # Likelihood
    data_mean, data_std = 48, 1.5
    likelihood = (1/(data_std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-data_mean)/data_std)**2)

    # Posterior (properly normalized)
    posterior_unnorm = prior * likelihood
    posterior = posterior_unnorm / (np.trapezoid(posterior_unnorm, x))

    # Calculate posterior statistics correctly
    post_mean = np.trapezoid(x * posterior, x)
    post_std = np.sqrt(np.trapezoid((x - post_mean)**2 * posterior, x))

    # Plot 1: Prior
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(x, 0, prior, alpha=0.3, color='blue')
    ax1.plot(x, prior, 'b-', linewidth=2)
    ax1.axvline(prior_mean, color='blue', linestyle='--', linewidth=2)
    ax1.set_title('Prior: P(θ)\nWhat we knew before', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=10)
    ax1.text(prior_mean, np.max(prior)*0.7, f'{prior_mean} ± {prior_std} kpc',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax1.set_xlim(40, 60)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Likelihood
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(x, 0, likelihood, alpha=0.3, color='green')
    ax2.plot(x, likelihood, 'g-', linewidth=2)
    ax2.axvline(data_mean, color='green', linestyle='--', linewidth=2)
    ax2.set_title('Likelihood: P(D|θ)\nWhat the data tells us', fontsize=11)
    ax2.set_ylabel('Likelihood', fontsize=10)
    ax2.text(data_mean, np.max(likelihood)*0.7, f'Data peak: {data_mean} kpc',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax2.set_xlim(40, 60)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Multiplication process
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(x, prior/np.max(prior), 'b-', linewidth=2, label='Prior (normalized)', alpha=0.7)
    ax3.plot(x, likelihood/np.max(likelihood), 'g-', linewidth=2, label='Likelihood (normalized)', alpha=0.7)
    ax3.plot(x, posterior/np.max(posterior), 'r-', linewidth=3, label='Posterior (normalized)')
    ax3.set_title("Bayes' Theorem: Prior × Likelihood = Posterior", fontsize=12)
    ax3.set_ylabel('Normalized Probability', fontsize=10)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(40, 60)

    # Plot 4: Posterior result
    ax4 = fig.add_subplot(gs[2, :])
    ax4.fill_between(x, 0, posterior, alpha=0.3, color='red')
    ax4.plot(x, posterior, 'r-', linewidth=3)
    ax4.axvline(post_mean, color='red', linestyle='--', linewidth=2)
    ax4.axvspan(post_mean - post_std, post_mean + post_std, alpha=0.2, color='red')
    ax4.set_title('Posterior: P(θ|D)\nWhat we know after combining prior + data', fontsize=12)
    ax4.set_xlabel('Distance (kpc)', fontsize=11)
    ax4.set_ylabel('Probability Density', fontsize=10)
    ax4.set_xlim(40, 60)
    ax4.grid(True, alpha=0.3)

    # Add information gain annotation
    info_text = f"Information Gain:\n"
    info_text += f"Prior: σ = {prior_std:.2f} kpc\n"
    info_text += f"Posterior: σ = {post_std:.2f} kpc\n"
    info_text += f"Uncertainty reduced by {prior_std/post_std:.2f}×"
    ax4.text(0.98, 0.98, info_text, transform=ax4.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
             fontsize=9)

    output_path = OUTPUT_DIR / 'fig_part2_bayesian_updating.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ==============================================================================
# PLOT 6: 2D Posterior with Marginalization (Section 2.4)
# Show joint posterior and marginal distributions
# ==============================================================================

def plot_2d_posterior_marginalization():
    """Illustrate 2D posterior and marginalization."""

    # Create figure with improved layout
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(3, 3, width_ratios=[3, 1, 0.3], height_ratios=[1, 3, 0.15],
                  hspace=0.15, wspace=0.15, figure=fig)

    # Main 2D posterior plot
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Generate 2D posterior (correlated)
    distances = np.linspace(45, 55, 200)
    extinctions = np.linspace(0, 0.5, 200)
    D, A = np.meshgrid(distances, extinctions)

    # Create correlated posterior
    d_mean, a_mean = 49, 0.2
    d_std, a_std = 1.5, 0.08
    correlation = 0.7

    cov = np.array([[d_std**2, correlation*d_std*a_std],
                    [correlation*d_std*a_std, a_std**2]])

    pos = np.dstack((D, A))
    mean = np.array([d_mean, a_mean])

    from scipy.stats import multivariate_normal
    posterior = multivariate_normal(mean, cov).pdf(pos)

    # Plot 2D posterior with filled contours
    levels = np.linspace(0.1, 1.0, 10) * np.max(posterior)
    cs = ax_main.contourf(D, A, posterior, levels=levels, cmap='YlOrRd', alpha=0.9)
    ax_main.contour(D, A, posterior, levels=levels, colors='black', alpha=0.3, linewidths=0.8)

    ax_main.set_xlabel('Distance (kpc)', fontsize=12)
    ax_main.set_ylabel('Extinction $A_V$ (mag)', fontsize=12)
    ax_main.set_title('Joint Posterior: P(d, $A_V$ | data)', fontsize=12)
    ax_main.grid(True, alpha=0.3)

    # Add correlation annotation with clearer visualization
    ax_main.plot([46, 52], [0.08, 0.35], 'b--', linewidth=2.5, alpha=0.8)
    ax_main.text(49, 0.05, 'Positive correlation:\nHigher extinction\n→ closer distance',
                 ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='blue'))

    # Marginal over extinction (top plot)
    marginal_d = np.trapezoid(posterior, extinctions, axis=0)
    marginal_d /= np.trapezoid(marginal_d, distances)

    ax_top.fill_between(distances, 0, marginal_d, alpha=0.4, color='blue')
    ax_top.plot(distances, marginal_d, 'b-', linewidth=2.5)
    ax_top.axvline(d_mean, color='red', linestyle='--', linewidth=2)
    ax_top.set_ylabel('$P(d|\\mathrm{data})$', fontsize=11, fontweight='bold')
    ax_top.set_title('Marginal over Extinction', fontsize=11, style='italic')
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.3)

    # Marginal over distance (right plot)
    marginal_a = np.trapezoid(posterior, distances, axis=1)
    marginal_a /= np.trapezoid(marginal_a, extinctions)

    ax_right.fill_betweenx(extinctions, 0, marginal_a, alpha=0.4, color='green')
    ax_right.plot(marginal_a, extinctions, 'g-', linewidth=2.5)
    ax_right.axhline(a_mean, color='red', linestyle='--', linewidth=2)
    ax_right.set_xlabel('$P(A_V|\\mathrm{data})$', fontsize=11, fontweight='bold')
    ax_right.set_title('Marginal\nover\nDistance', fontsize=10, style='italic')
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.3)

    # Add colorbar in the proper location
    cbar_ax = fig.add_subplot(gs[1, 2])
    cbar = plt.colorbar(cs, cax=cbar_ax, label='Posterior Density')
    cbar.set_label('Posterior Density', fontsize=10)

    # Add overall title
    fig.suptitle('2D Posterior with Marginal Distributions', fontsize=14, fontweight='bold', y=0.98)

    output_path = OUTPUT_DIR / 'fig_part2_2d_posterior_marginals.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ==============================================================================
# MAIN: Generate all plots
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING EDUCATIONAL PLOTS FOR MODULE 5, PART 2")
    print("="*70 + "\n")
    
    print("Section 2.2: Likelihood")
    plot_likelihood_function()
    plot_degeneracy()
    
    print("\nSection 2.3: Priors")
    plot_prior_types()
    plot_prior_data_balance()
    
    print("\nSection 2.4: Bayes' Theorem")
    plot_bayesian_updating()
    plot_2d_posterior_marginalization()
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nFiles created:")
    print("  1. fig_part2_likelihood_1d.png")
    print("  2. fig_part2_degeneracy_2d.png")
    print("  3. fig_part2_prior_types.png")
    print("  4. fig_part2_prior_data_balance.png")
    print("  5. fig_part2_bayesian_updating.png")
    print("  6. fig_part2_2d_posterior_marginals.png")
    print("\nInsert these figures into your course materials as:")
    print("  ![Caption](fig_part2_likelihood_1d.png)")
    print("="*70 + "\n")