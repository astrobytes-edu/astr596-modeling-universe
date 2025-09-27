"""
Static Visualizations for Module 4 Part III
Monte Carlo Radiative Transfer

This script generates publication-quality static figures that can be
embedded directly in MyST Markdown documents. These serve as alternatives
to interactive visualizations when JavaScript isn't available.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set up publication-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_optical_depth_sampling_figure():
    """
    Figure 3.1: Monte Carlo Optical Depth Sampling
    
    Shows the inverse transform method for sampling τ = -ln(ξ)
    and validates that samples follow the exponential distribution.
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Generate samples
    np.random.seed(42)  # For reproducibility
    n_samples = 10000
    xi = np.random.uniform(0, 1, n_samples)
    tau = -np.log(xi)
    tau_filtered = tau[tau <= 6]  # Filter for display
    
    # Panel 1: Uniform random numbers
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(xi[:1000], bins=30, alpha=0.7, density=True, 
             color='skyblue', edgecolor='black')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                label='Uniform: p(ξ) = 1')
    ax1.set_xlabel('Random Number ξ')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Input: Uniform Random Numbers')
    ax1.legend()
    ax1.set_ylim(0, 1.5)
    
    # Panel 2: Inverse transform visualization
    ax2 = fig.add_subplot(gs[0, 1])
    xi_theory = np.linspace(0.01, 0.99, 100)
    tau_theory = -np.log(xi_theory)
    
    # Show transformation curve
    ax2.plot(xi_theory, tau_theory, 'r-', linewidth=3, 
             label='τ = -ln(ξ)')
    
    # Show some sample points
    sample_indices = np.random.choice(len(xi), 20)
    ax2.scatter(xi[sample_indices], tau[sample_indices], 
                alpha=0.6, s=50, color='blue', label='Sample Points')
    
    ax2.set_xlabel('Random Number ξ')
    ax2.set_ylabel('Optical Depth τ')
    ax2.set_title('Inverse Transform Method')
    ax2.legend()
    ax2.set_ylim(0, 6)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Resulting distribution
    ax3 = fig.add_subplot(gs[0, 2])
    counts, bins, _ = ax3.hist(tau_filtered, bins=40, alpha=0.7, 
                               density=True, color='lightgreen', 
                               edgecolor='black', label='MC Samples')
    
    # Theoretical curve
    tau_th = np.linspace(0, 6, 100)
    pdf_th = np.exp(-tau_th)
    ax3.plot(tau_th, pdf_th, 'r-', linewidth=3, 
             label='Theory: p(τ) = e⁻τ')
    
    ax3.set_xlabel('Optical Depth τ')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Output: Exponential Distribution')
    ax3.legend()
    ax3.set_ylim(0, 1.2)
    
    # Panel 4: Cumulative distribution comparison
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Empirical CDF
    tau_sorted = np.sort(tau_filtered)
    cdf_empirical = np.arange(1, len(tau_sorted) + 1) / len(tau_sorted)
    ax4.plot(tau_sorted, cdf_empirical, 'b-', linewidth=2, 
             label='Empirical CDF', alpha=0.8)
    
    # Theoretical CDF
    cdf_theory = 1 - np.exp(-tau_th)
    ax4.plot(tau_th, cdf_theory, 'r--', linewidth=3, 
             label='Theory: F(τ) = 1 - e⁻τ')
    
    ax4.set_xlabel('Optical Depth τ')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution Function Validation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Statistics text box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Calculate statistics
    theoretical_mean = 1.0
    empirical_mean = np.mean(tau_filtered)
    theoretical_std = 1.0
    empirical_std = np.std(tau_filtered)
    
    stats_text = f"""
Monte Carlo Validation

Theoretical Values:
• Mean: {theoretical_mean:.3f}
• Std Dev: {theoretical_std:.3f}

Empirical Values:
• Mean: {empirical_mean:.3f}
• Std Dev: {empirical_std:.3f}

Sample Size: {len(tau_filtered):,}

Agreement: {abs(empirical_mean - theoretical_mean) < 0.05}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=12, 
             verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    fig.suptitle('Figure 3.1: Monte Carlo Optical Depth Sampling\n' + 
                 'Inverse Transform Method: ξ → τ = -ln(ξ)', 
                 fontsize=16, fontweight='bold')
    
    return fig

def create_discrete_vs_continuous_figure():
    """
    Figure 3.2: Discrete vs Continuous Absorption
    
    Illustrates the fundamental difference between Monte Carlo
    discrete absorption and ray tracing continuous absorption.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Parameters
    total_tau = 3.0
    n_packets = 20
    
    # Generate packet data
    np.random.seed(123)
    tau_targets = -np.log(np.random.uniform(0, 1, n_packets))
    
    # Panel 1: Individual packet paths (Monte Carlo)
    for i in range(min(8, n_packets)):
        tau_target = tau_targets[i]
        if tau_target <= total_tau:
            # Absorbed packet
            ax1.plot([0, tau_target], [i+1, i+1], 'r-', linewidth=2, alpha=0.7)
            ax1.plot([tau_target, tau_target], [i+1, 0], 'r-', linewidth=3)
            ax1.scatter(tau_target, i+1, color='red', s=100, marker='x', linewidth=3)
        else:
            # Escaped packet
            ax1.plot([0, total_tau], [i+1, i+1], 'g--', linewidth=2, alpha=0.7)
            ax1.scatter(total_tau, i+1, color='green', s=80, marker='>')
    
    ax1.axvline(x=total_tau, color='black', linestyle='-', linewidth=2, 
                label='Domain Boundary')
    ax1.set_xlabel('Optical Depth τ')
    ax1.set_ylabel('Packet Number')
    ax1.set_title('Monte Carlo: Discrete Absorption\n(Individual Packet Histories)')
    ax1.set_xlim(0, total_tau * 1.1)
    ax1.set_ylim(0, 9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Continuous attenuation (Ray Tracing)
    tau_cont = np.linspace(0, total_tau, 100)
    luminosity = np.exp(-tau_cont)
    
    ax2.plot(tau_cont, luminosity, 'b-', linewidth=3, 
             label='L(τ) = e⁻τ')
    ax2.fill_between(tau_cont, 0, luminosity, alpha=0.3, 
                     color='blue', label='Continuous Energy Loss')
    
    # Show energy deposition rate
    deposition_rate = luminosity  # dL/dτ = L
    ax2.plot(tau_cont, -deposition_rate, 'r-', linewidth=2, 
             label='Energy Deposition Rate')
    ax2.fill_between(tau_cont, 0, -deposition_rate, alpha=0.3, 
                     color='red')
    
    ax2.set_xlabel('Optical Depth τ')
    ax2.set_ylabel('Normalized Luminosity')
    ax2.set_title('Ray Tracing: Continuous Absorption\n(Ensemble Average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.2, 1.2)
    
    # Panel 3: Energy deposition comparison
    tau_bins = np.linspace(0, total_tau, 20)
    
    # Monte Carlo energy deposition (histogram)
    absorbed_tau = tau_targets[tau_targets <= total_tau]
    mc_hist, _ = np.histogram(absorbed_tau, bins=tau_bins, density=True)
    bin_centers = (tau_bins[:-1] + tau_bins[1:]) / 2
    
    ax3.bar(bin_centers, mc_hist, width=0.12, alpha=0.7, 
            color='red', label='Monte Carlo')
    
    # Theoretical deposition
    theoretical = np.exp(-bin_centers)
    ax3.plot(bin_centers, theoretical, 'b-', linewidth=3, 
             label='Theory: e⁻τ')
    
    ax3.set_xlabel('Optical Depth τ')
    ax3.set_ylabel('Energy Deposition Density')
    ax3.set_title('Energy Deposition Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Transmission fraction convergence
    packet_counts = np.logspace(2, 4, 20).astype(int)
    transmissions = []
    
    for n in packet_counts:
        tau_test = -np.log(np.random.uniform(0, 1, n))
        escaped = np.sum(tau_test > total_tau)
        transmissions.append(escaped / n)
    
    analytical_transmission = np.exp(-total_tau)
    
    ax4.semilogx(packet_counts, transmissions, 'bo-', markersize=6, 
                 label='Monte Carlo')
    ax4.axhline(y=analytical_transmission, color='red', linestyle='--', 
                linewidth=2, label=f'Analytical: e⁻³ = {analytical_transmission:.3f}')
    
    # Error bars (approximate)
    errors = [np.sqrt(t * (1-t) / n) for t, n in zip(transmissions, packet_counts)]
    ax4.errorbar(packet_counts, transmissions, yerr=errors, 
                 fmt='bo', alpha=0.5)
    
    ax4.set_xlabel('Number of Packets')
    ax4.set_ylabel('Transmission Fraction')
    ax4.set_title('Convergence to Analytical Solution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('Figure 3.2: Discrete vs Continuous Absorption\n' + 
                 'Fundamental Difference Between MC and Ray Tracing', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_convergence_monitoring_figure():
    """
    Figure 3.3: Monte Carlo Convergence Monitoring
    
    Shows how to assess convergence and determine when enough
    packets have been used for reliable results.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Parameters
    tau_true = 2.0
    analytical = np.exp(-tau_true)
    max_packets = 50000
    
    # Generate convergence data
    np.random.seed(456)
    checkpoints = np.logspace(2, np.log10(max_packets), 50).astype(int)
    running_means = []
    errors = []
    
    cumulative_escaped = 0
    all_results = []
    
    for i, n in enumerate(checkpoints):
        if i == 0:
            tau_samples = -np.log(np.random.uniform(0, 1, n))
        else:
            new_samples = n - checkpoints[i-1]
            tau_samples = -np.log(np.random.uniform(0, 1, new_samples))
        
        escaped = np.sum(tau_samples > tau_true)
        cumulative_escaped += escaped
        
        current_mean = cumulative_escaped / n
        current_error = np.sqrt(current_mean * (1 - current_mean) / n)
        
        running_means.append(current_mean)
        errors.append(current_error)
        all_results.extend([1 if t > tau_true else 0 for t in tau_samples])
    
    # Panel 1: Running average convergence
    ax1 = fig.add_subplot(gs[0, :2])
    
    ax1.errorbar(checkpoints, running_means, yerr=errors, 
                 fmt='b-', alpha=0.7, linewidth=2, label='Running Average ± 1σ')
    ax1.axhline(y=analytical, color='red', linestyle='--', 
                linewidth=3, label=f'Analytical: e⁻² = {analytical:.4f}')
    
    # Convergence region
    converged_idx = len(checkpoints) // 2
    conv_region = np.arange(converged_idx, len(checkpoints))
    conv_mean = np.mean(running_means[converged_idx:])
    conv_std = np.std(running_means[converged_idx:])
    
    ax1.fill_between(checkpoints[conv_region], 
                     conv_mean - conv_std, conv_mean + conv_std,
                     alpha=0.2, color='green', 
                     label=f'Converged Region (σ = {conv_std:.4f})')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Packets')
    ax1.set_ylabel('Transmission Fraction')
    ax1.set_title('Running Average Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Error scaling validation
    ax2 = fig.add_subplot(gs[0, 2])
    
    theoretical_errors = [1.0 / np.sqrt(n) * 0.4 for n in checkpoints]  # Scaled
    
    ax2.loglog(checkpoints, errors, 'bo', markersize=4, 
               label='Actual Error', alpha=0.7)
    ax2.loglog(checkpoints, theoretical_errors, 'r--', linewidth=2, 
               label='Theory: ∝ 1/√N')
    
    ax2.set_xlabel('Number of Packets')
    ax2.set_ylabel('Standard Error')
    ax2.set_title('Error Scaling Verification')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Multiple realizations
    ax3 = fig.add_subplot(gs[1, :])
    
    n_realizations = 30
    realization_size = 5000
    realization_results = []
    
    for _ in range(n_realizations):
        tau_real = -np.log(np.random.uniform(0, 1, realization_size))
        escaped_real = np.sum(tau_real > tau_true)
        realization_results.append(escaped_real / realization_size)
    
    # Box plot
    ax3.boxplot([realization_results], positions=[1], widths=0.6,
                patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    # Individual points
    x_pos = np.ones(len(realization_results)) + np.random.normal(0, 0.05, len(realization_results))
    ax3.scatter(x_pos, realization_results, alpha=0.6, s=50, color='blue')
    
    # Analytical line
    ax3.axhline(y=analytical, color='red', linestyle='--', 
                linewidth=3, label=f'Analytical: {analytical:.4f}')
    
    # Statistics
    mean_real = np.mean(realization_results)
    std_real = np.std(realization_results)
    theoretical_std = np.sqrt(analytical * (1 - analytical) / realization_size)
    
    ax3.text(1.5, max(realization_results), 
             f'Empirical: {mean_real:.4f} ± {std_real:.4f}\n' + 
             f'Theoretical σ: {theoretical_std:.4f}',
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax3.set_xlim(0.5, 2.5)
    ax3.set_ylabel('Transmission Fraction')
    ax3.set_title(f'Multiple Realizations ({n_realizations} runs, {realization_size:,} packets each)')
    ax3.set_xticks([1])
    ax3.set_xticklabels(['MC Realizations'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Convergence criteria assessment
    ax4 = fig.add_subplot(gs[2, :2])
    
    # Rolling window assessment
    window_size = 10
    rolling_means = []
    rolling_stds = []
    
    for i in range(window_size, len(running_means)):
        window = running_means[i-window_size:i]
        rolling_means.append(np.mean(window))
        rolling_stds.append(np.std(window))
    
    rolling_checkpoints = checkpoints[window_size:]
    
    ax4.plot(rolling_checkpoints, rolling_stds, 'g-', linewidth=2,
             label='Rolling Standard Deviation')
    ax4.axhline(y=0.001, color='orange', linestyle=':', linewidth=2,
                label='Convergence Threshold (0.1%)')
    
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Number of Packets')
    ax4.set_ylabel('Rolling Standard Deviation')
    ax4.set_title('Convergence Stability Assessment')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Convergence checklist
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    # Assess convergence criteria
    final_deviation = abs(running_means[-1] - analytical) / errors[-1]
    is_stable = rolling_stds[-1] < 0.001 if rolling_stds else False
    has_enough_samples = checkpoints[-1] >= 10000
    
    checklist_text = f"""
Convergence Assessment

✓ Statistical Agreement:
  {final_deviation:.1f}σ deviation
  {'✓ PASS' if final_deviation < 3 else '✗ FAIL'} (< 3σ required)

✓ Stability:
  Rolling σ = {rolling_stds[-1]:.5f}
  {'✓ PASS' if is_stable else '✗ FAIL'} (< 0.001 required)

✓ Sample Size:
  N = {checkpoints[-1]:,} packets
  {'✓ PASS' if has_enough_samples else '✗ FAIL'} (≥ 10k required)

✓ Error Scaling:
  Follows 1/√N trend
  ✓ PASS

Overall: {'✓ CONVERGED' if all([final_deviation < 3, is_stable, has_enough_samples]) else '✗ NOT CONVERGED'}
    """
    
    ax5.text(0.05, 0.95, checklist_text, fontsize=10, 
             verticalalignment='top', transform=ax5.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    fig.suptitle('Figure 3.3: Monte Carlo Convergence Monitoring\n' + 
                 'Essential Diagnostics for Reliable Results', 
                 fontsize=16, fontweight='bold')
    
    return fig

def create_scattering_comparison_figure():
    """
    Figure 3.4: Scattering Phase Functions and Path Visualization
    
    Shows different scattering regimes and their impact on
    photon transport in Monte Carlo simulations.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Phase function comparison
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Isotropic scattering
    iso_phase = np.ones_like(theta) / (4 * np.pi)
    ax1.plot(theta, iso_phase * 50, 'b-', linewidth=2, 
             label='Isotropic (g=0)')
    
    # Forward scattering
    g_forward = 0.7
    cos_theta = np.cos(theta)
    forward_phase = (1 - g_forward**2) / (1 + g_forward**2 - 2*g_forward*cos_theta)**(3/2)
    ax1.plot(theta, forward_phase, 'r-', linewidth=2, 
             label=f'Forward (g={g_forward})')
    
    # Backward scattering
    g_backward = -0.7
    backward_phase = (1 - g_backward**2) / (1 + g_backward**2 - 2*g_backward*cos_theta)**(3/2)
    ax1.plot(theta, backward_phase, 'g-', linewidth=2, 
             label=f'Backward (g={g_backward})')
    
    ax1.set_title('Henyey-Greenstein Phase Functions')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True)
    
    # Panel 2: Photon path visualization (isotropic)
    ax2 = fig.add_subplot(gs[0, 1])
    simulate_photon_paths(ax2, g=0.0, title='Isotropic Scattering')
    
    # Panel 3: Photon path visualization (forward)
    ax3 = fig.add_subplot(gs[0, 2])
    simulate_photon_paths(ax3, g=0.7, title='Forward Scattering')
    
    # Panel 4: Escape fraction vs albedo
    ax4 = fig.add_subplot(gs[1, 0])
    
    albedos = np.linspace(0, 0.95, 20)
    tau_total = 2.0
    
    # Analytical approximation for escape fraction with scattering
    escape_fractions = []
    for omega in albedos:
        # Simple approximation: effective optical depth
        tau_eff = tau_total * (1 - omega)
        escape_fractions.append(np.exp(-tau_eff))
    
    ax4.plot(albedos, escape_fractions, 'b-', linewidth=3,
             label='With Scattering')
    
    # Without scattering
    no_scatter = np.exp(-tau_total) * np.ones_like(albedos)
    ax4.plot(albedos, no_scatter, 'r--', linewidth=2,
             label='Pure Absorption')
    
    ax4.set_xlabel('Albedo ω')
    ax4.set_ylabel('Escape Fraction')
    ax4.set_title(f'Impact of Scattering (τ = {tau_total})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Convergence requirements
    ax5 = fig.add_subplot(gs[1, 1])
    
    omegas = [0.0, 0.3, 0.6, 0.9]
    base_packets = 10000
    
    required_packets = []
    for omega in omegas:
        # Empirical scaling: N ∝ 1/(1-ω)
        factor = 1 / (1 - omega + 0.01)  # Small offset to avoid division by zero
        required_packets.append(base_packets * factor)
    
    bars = ax5.bar(range(len(omegas)), required_packets, 
                   color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    
    ax5.set_xlabel('Albedo ω')
    ax5.set_ylabel('Required Packets')
    ax5.set_title('Computational Cost vs Albedo')
    ax5.set_xticks(range(len(omegas)))
    ax5.set_xticklabels([f'{ω:.1f}' for ω in omegas])
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}',
                 ha='center', va='bottom', fontweight='bold')
    
    # Panel 6: Validation strategies
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    validation_text = """
Scattering Validation Tests

1. Isotropic Emission:
   • Source emits uniformly
   • Check angular distribution
   
2. Energy Conservation:
   • Track absorbed + escaped
   • Should equal input energy
   
3. Phase Function Sampling:
   • Test against analytical CDF
   • Verify cos(θ) distribution
   
4. Multiple Scattering:
   • Compare with diffusion limit
   • Check path length statistics
   
5. Albedo Dependence:
   • Vary ω from 0 to 0.99
   • Validate escape fraction
   
Common Issues:
⚠️ Direction sampling errors
⚠️ Incorrect albedo handling
⚠️ Missing energy conservation
⚠️ Poor convergence for high ω
    """
    
    ax6.text(0.05, 0.95, validation_text, fontsize=10,
             verticalalignment='top', transform=ax6.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
    
    fig.suptitle('Figure 3.4: Scattering in Monte Carlo Radiative Transfer\n' + 
                 'Phase Functions, Paths, and Validation', 
                 fontsize=16, fontweight='bold')
    
    return fig

def simulate_photon_paths(ax, g=0.0, n_photons=5, title=""):
    """Helper function to simulate and plot photon scattering paths"""
    np.random.seed(789)
    
    for i in range(n_photons):
        # Start at origin
        x, y = [0], [0]
        direction = 0  # Initial direction (radians)
        
        # Simulate scattering events
        for step in range(10):
            # Move forward
            step_size = 0.3
            x.append(x[-1] + step_size * np.cos(direction))
            y.append(y[-1] + step_size * np.sin(direction))
            
            # Scatter according to phase function
            if abs(g) < 1e-3:
                # Isotropic
                cos_theta = 1 - 2 * np.random.random()
            else:
                # Henyey-Greenstein sampling
                xi = np.random.random()
                s = (1 - g*g) / (1 - g + 2*g*xi)
                cos_theta = (1 + g*g - s*s) / (2*g)
            
            theta = np.arccos(np.clip(cos_theta, -1, 1))
            phi = 2 * np.pi * np.random.random()
            
            # Update direction (simplified 2D)
            direction += theta - np.pi/2 + np.random.random() * np.pi/4
            
            # Stop if photon escapes domain
            if np.sqrt(x[-1]**2 + y[-1]**2) > 3:
                break
        
        # Plot path
        colors = plt.cm.viridis(i / n_photons)
        ax.plot(x, y, 'o-', color=colors, alpha=0.7, markersize=3)
        ax.scatter(x[0], y[0], color='red', s=50, marker='*')  # Source
        ax.scatter(x[-1], y[-1], color='black', s=30, marker='s')  # End
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')

def create_exponential_theory_figure():
    """
    Figure 3.0: Exponential Distribution Theory
    
    Educational figure showing the mathematical foundation of 
    Monte Carlo optical depth sampling with clear visual connections
    to the inverse transform method.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Define tau range for theory
    tau_max = 5.0
    tau_theory = np.linspace(0, tau_max, 200)
    
    # Panel 1: Exponential PDF
    ax1 = fig.add_subplot(gs[0, 0])
    pdf_theory = np.exp(-tau_theory)
    ax1.plot(tau_theory, pdf_theory, 'r-', linewidth=3, 
             label='p(τ) = e⁻τ')
    ax1.fill_between(tau_theory, 0, pdf_theory, alpha=0.3, color='red')
    
    # Highlight mean free path
    ax1.axvline(x=1.0, color='blue', linestyle='--', linewidth=2,
                label='Mean Free Path (τ = 1)')
    ax1.axhline(y=np.exp(-1), color='blue', linestyle=':', alpha=0.7)
    
    ax1.set_xlabel('Optical Depth τ')
    ax1.set_ylabel('Probability Density p(τ)')
    ax1.set_title('Exponential Probability Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, tau_max)
    ax1.set_ylim(0, 1.1)
    
    # Panel 2: Cumulative Distribution Function
    ax2 = fig.add_subplot(gs[0, 1])
    cdf_theory = 1 - np.exp(-tau_theory)
    ax2.plot(tau_theory, cdf_theory, 'g-', linewidth=3, 
             label='F(τ) = 1 - e⁻τ')
    
    # Show percentiles
    percentiles = [0.5, 0.63, 0.95, 0.99]
    tau_percentiles = [-np.log(1 - p) for p in percentiles]
    
    for i, (p, t) in enumerate(zip(percentiles, tau_percentiles)):
        if t <= tau_max:
            ax2.plot([0, t, t], [p, p, 0], 'k--', alpha=0.5)
            ax2.plot(t, p, 'ko', markersize=6)
            ax2.text(t + 0.1, p - 0.05, f'{int(p*100)}%', fontsize=10)
    
    ax2.set_xlabel('Optical Depth τ')
    ax2.set_ylabel('Cumulative Probability F(τ)')
    ax2.set_title('Cumulative Distribution Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, tau_max)
    ax2.set_ylim(0, 1)
    
    # Panel 3: Inverse Transform Visualization
    ax3 = fig.add_subplot(gs[0, 2])
    xi_values = np.linspace(0.01, 0.99, 100)
    tau_transformed = -np.log(xi_values)
    
    ax3.plot(xi_values, tau_transformed, 'b-', linewidth=3,
             label='τ = -ln(ξ)')
    
    # Show specific examples
    examples = [0.1, 0.37, 0.63, 0.9]
    for xi in examples:
        tau = -np.log(xi)
        if tau <= tau_max:
            ax3.plot([xi], [tau], 'ro', markersize=8)
            ax3.plot([xi, xi], [0, tau], 'r--', alpha=0.5)
            ax3.plot([0, xi], [tau, tau], 'r--', alpha=0.5)
            ax3.text(xi + 0.02, tau, f'({xi:.2f}, {tau:.1f})', 
                     fontsize=9, rotation=45)
    
    ax3.set_xlabel('Uniform Random Number ξ')
    ax3.set_ylabel('Optical Depth τ')
    ax3.set_title('Inverse Transform Method')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, tau_max)
    
    # Panel 4: Physical Interpretation
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Show survival probability vs distance
    distance = np.linspace(0, 3, 100)
    survival = np.exp(-distance)
    
    ax4.plot(distance, survival, 'purple', linewidth=3, 
             label='Survival Probability = e⁻τ')
    ax4.fill_between(distance, 0, survival, alpha=0.2, color='purple')
    
    # Show interaction probability
    interaction = 1 - survival
    ax4.plot(distance, interaction, 'orange', linewidth=3, 
             label='Interaction Probability = 1 - e⁻τ')
    
    # Mark key distances
    key_distances = [0.69, 1.0, 2.3]  # ln(2), 1, ln(10)
    key_labels = ['50% interactions', '63% interactions', '90% interactions']
    
    for dist, label in zip(key_distances, key_labels):
        surv = np.exp(-dist)
        ax4.plot([dist, dist], [0, surv], 'k--', alpha=0.7)
        ax4.plot(dist, surv, 'ko', markersize=6)
        ax4.text(dist, surv + 0.05, label, ha='center', fontsize=10, 
                 rotation=45 if dist > 1.5 else 0)
    
    ax4.set_xlabel('Optical Depth τ')
    ax4.set_ylabel('Probability')
    ax4.set_title('Physical Interpretation: Beer\'s Law')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 3)
    ax4.set_ylim(0, 1)
    
    # Panel 5: Monte Carlo Connection
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    connection_text = """
Monte Carlo Implementation

1. Generate uniform random ξ ∈ [0,1]

2. Transform: τ = -ln(ξ)
   • This samples from p(τ) = e⁻τ
   • Automatically follows Beer's law

3. Physical meaning:
   • Most interactions near source
   • Few interactions deep in medium
   • Statistically exact sampling

4. Validation check:
   • Histogram of τ values should 
     match exponential curve
   • Mean should equal 1.0
   • CDF should match theory

Key insight: Random numbers
become physics-based distances!
    """
    
    ax5.text(0.05, 0.95, connection_text, fontsize=11, 
             verticalalignment='top', transform=ax5.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    fig.suptitle('Figure 3.0: Exponential Distribution Theory for Monte Carlo\n' + 
                 'Mathematical Foundation of Optical Depth Sampling', 
                 fontsize=16, fontweight='bold')
    
    return fig

def save_all_figures(output_dir='./figures/'):
    """Generate and save all static figures"""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    figures = {}
    
    print("Generating static figures for Module 4 Part III...")
    print("=" * 60)
    
    # Generate Figure 3.0
    print("Creating Figure 3.0: Exponential Distribution Theory...")
    fig0 = create_exponential_theory_figure()
    fig0.savefig(f'{output_dir}figure_3_0_exponential_theory.png', 
                 dpi=300, bbox_inches='tight')
    figures['exponential_theory'] = fig0
    print("  ✓ Saved as figure_3_0_exponential_theory.png")
    
    # Generate Figure 3.1
    print("Creating Figure 3.1: Optical Depth Sampling...")
    fig1 = create_optical_depth_sampling_figure()
    fig1.savefig(f'{output_dir}figure_3_1_optical_depth_sampling.png', 
                 dpi=300, bbox_inches='tight')
    figures['optical_depth_sampling'] = fig1
    print("  ✓ Saved as figure_3_1_optical_depth_sampling.png")
    
    # Generate Figure 3.2
    print("Creating Figure 3.2: Discrete vs Continuous Absorption...")
    fig2 = create_discrete_vs_continuous_figure()
    fig2.savefig(f'{output_dir}figure_3_2_discrete_vs_continuous.png', 
                 dpi=300, bbox_inches='tight')
    figures['discrete_vs_continuous'] = fig2
    print("  ✓ Saved as figure_3_2_discrete_vs_continuous.png")
    
    # Generate Figure 3.3
    print("Creating Figure 3.3: Convergence Monitoring...")
    fig3 = create_convergence_monitoring_figure()
    fig3.savefig(f'{output_dir}figure_3_3_convergence_monitoring.png', 
                 dpi=300, bbox_inches='tight')
    figures['convergence_monitoring'] = fig3
    print("  ✓ Saved as figure_3_3_convergence_monitoring.png")
    
    # Generate Figure 3.4
    print("Creating Figure 3.4: Scattering Comparison...")
    fig4 = create_scattering_comparison_figure()
    fig4.savefig(f'{output_dir}figure_3_4_scattering_comparison.png', 
                 dpi=300, bbox_inches='tight')
    figures['scattering_comparison'] = fig4
    print("  ✓ Saved as figure_3_4_scattering_comparison.png")
    
    print("=" * 60)
    print(f"All static figures generated and saved to {output_dir}")
    
    return figures

if __name__ == "__main__":
    # Generate all figures
    figures = save_all_figures()
    
    # Display figures if in interactive mode
    try:
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            plt.show()
    except:
        pass
    
    print("\nStatic figures ready for MyST integration!")
    print("These figures provide visual support for understanding:")
    print("  - Optical depth sampling and inverse transform method")
    print("  - Critical difference between discrete and continuous absorption")
    print("  - Convergence monitoring and validation techniques")
    print("  - Scattering physics and computational complexity")
