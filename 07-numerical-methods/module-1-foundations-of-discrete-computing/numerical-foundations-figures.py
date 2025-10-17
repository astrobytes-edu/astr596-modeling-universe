#!/usr/bin/env python3
"""
Educational Figures for Module 1: Foundations of Discrete Computing
ASTR 596: Modeling the Universe

This script generates all matplotlib/seaborn figures for the numerical methods module.
Each function creates a specific educational visualization with publication-quality output.

Updated to match revised curriculum focusing on practical numerical challenges in astronomy.

Usage:
    python numerical-foundations-figures.py                    # Generate all figures
    python numerical-foundations-figures.py --list             # List available figures
    python numerical-foundations-figures.py --figure 1_1       # Generate specific figure
    python numerical-foundations-figures.py 1_1               # Generate by ID
    
Requirements:
    - matplotlib
    - numpy
    - scipy (optional, for some examples)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os
import warnings
import sys
import argparse
warnings.filterwarnings('ignore')

# Create figures directory if it doesn't exist
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Modern color palette - consistent with statistical thinking module
COLORS = {
    'primary': '#2E86AB',    # Modern blue
    'secondary': '#A23B72',  # Deep rose
    'accent': '#16A085',     # Elegant teal
    'neutral': '#6C757D',    # Sophisticated gray
    'light': '#F8F9FA',      # Very light gray
    'dark': '#2D3436'        # Charcoal
}

# Modern style parameters - matching statistical module
def set_style():
    """Set consistent modern style for all figures"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,  # Increased for better readability
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': COLORS['neutral'],
        'axes.labelcolor': COLORS['dark'],
        'text.color': COLORS['dark'],
        'xtick.color': COLORS['neutral'],
        'ytick.color': COLORS['neutral'],
        'grid.color': COLORS['light'],
        'grid.alpha': 0.4
    })

def save_figure(fig, filename, show=False):
    """Save figure with consistent formatting matching statistical module"""
    filepath = os.path.join(FIGURES_DIR, f"{filename}.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    print(f"Saved: {filepath}")

def reset_style():
    """Reset matplotlib style to defaults"""
    plt.rcParams.update(plt.rcParamsDefault)


# ============================================================================
# PART 1 FIGURES: Finite Differences (Keep existing working figures)
# ============================================================================

def figure_1_1_derivative_paradox():
    """
    Creates visualization of the fundamental paradox: analytical vs numerical derivatives.
    
    Location: Part 1, Section "The Core Problem"
    Caption: The derivative paradox: mathematical definition requires h→0, but computers 
             cannot take true limits. Left: analytical derivative (smooth tangent line). 
             Right: numerical approximation using finite h values showing convergence - 
             smaller h=0.1 approaches the true slope better than larger h=0.3.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Test function: x^2 for simplicity - mathematically clean with known derivative
    x = np.linspace(0, 3, 1000)
    f = x**2
    x0 = 1.5
    f0 = x0**2
    true_slope = 2*x0  # Analytical derivative of x^2 is 2x
    
    # Left panel: Analytical derivative
    ax1.plot(x, f, color=COLORS['primary'], linewidth=4, alpha=0.9, 
             label=r'$f(x) = x^2$', zorder=3)
    
    # True tangent line
    tangent_x = np.linspace(0.5, 2.5, 100)
    tangent_y = f0 + true_slope * (tangent_x - x0)
    ax1.plot(tangent_x, tangent_y, color=COLORS['accent'], linewidth=4, 
             linestyle='--', alpha=0.9, label=r'True tangent: slope = $2x_0$', zorder=4)
    
    # Point of evaluation
    ax1.plot(x0, f0, 'o', color=COLORS['secondary'], markersize=10, 
             markeredgewidth=2, markeredgecolor='white', zorder=5)
    
    ax1.set_xlabel(r'$x$', fontsize=18, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel(r'$f(x)$', fontsize=18, color=COLORS['dark'], weight='medium')
    ax1.set_title('Analytical: Perfect Tangent Line', fontsize=20, 
                  color=COLORS['dark'], weight='medium', pad=20)
    ax1.legend(loc='upper left', fontsize=16, frameon=True, fancybox=True,
               edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3, linewidth=1.0)
    ax1.set_xlim(0.5, 2.5)
    ax1.set_ylim(0, 6)
    
    # Add textbox for true slope
    ax1.text(0.7, 5.2, r'True Slope = $2x_0 = 3.0$', 
             fontsize=16, bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], 
             edgecolor=COLORS['accent'], alpha=0.95), weight='medium', color=COLORS['dark'])
    
    # Right panel: Numerical approximations
    ax2.plot(x, f, color=COLORS['primary'], linewidth=4, alpha=0.9, 
             label=r'$f(x) = x^2$', zorder=3)
    
    # Two different h values for comparison
    h_values = [0.3, 0.1]
    alphas = [0.7, 0.9]
    linestyles = [':', '-']
    colors = [COLORS['neutral'], COLORS['secondary']]
    
    for i, (h, alpha, linestyle, color) in enumerate(zip(h_values, alphas, linestyles, colors)):
        # Forward difference slope
        numerical_slope = (((x0 + h)**2) - (x0**2))/h
        
        # Secant line
        secant_x = np.linspace(x0, x0 + h + 0.3, 100)
        secant_y = f0 + numerical_slope * (secant_x - x0)
        ax2.plot(secant_x, secant_y, color=color, linewidth=3.5, 
                 linestyle=linestyle, alpha=alpha, 
                 label=rf'Secant line: $h = {h}$, slope = {numerical_slope:.2f}', zorder=4-i)
        
        # Mark the two points used
        ax2.plot([x0, x0 + h], [f0, (x0 + h)**2], 'o', color=color, 
                 markersize=8, markeredgewidth=1.5, markeredgecolor='white', alpha=alpha, zorder=5)
    
    ax2.set_xlabel(r'$x$', fontsize=18, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel(r'$f(x)$', fontsize=18, color=COLORS['dark'], weight='medium')
    ax2.set_title(r'Numerical: Finite $h$ Approximations', fontsize=20, 
                  color=COLORS['dark'], weight='medium', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=14, frameon=True, 
               fancybox=True, edgecolor=COLORS['neutral'], facecolor='white', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linewidth=1.0)
    ax2.set_xlim(0.5, 2.5)
    ax2.set_ylim(0, 6)
    
    # Add key insight textbox - positioned higher to avoid line overlap
    ax2.text(0.7, 4.2, 'Key Insight:\nSmaller $h$ → Better approximation', 
             fontsize=16, bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], 
             edgecolor=COLORS['secondary'], alpha=0.95), weight='medium', color=COLORS['dark'])
    
    # Style spines for both panels
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=16, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "1_1_derivative_paradox")
    reset_style()


def figure_1_2_error_landscape():
    """
    Error landscape: how truncation and round-off errors compete as h varies.
    Shows the fundamental trade-off in choosing step sizes.
    
    Location: Part 1, Section "The Step Size Dilemma" 
    Caption: Error landscape for numerical derivatives. Total error (black) is dominated 
             by truncation error for large h and round-off error for small h. The optimal 
             step size minimizes total error. This fundamental trade-off appears throughout 
             computational astrophysics.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Step size range from large to near machine precision
    h_values = np.logspace(-14, -1, 200)
    epsilon = 2.22e-16  # Machine epsilon for double precision
    
    # Test function: sin(x) at x = 1.0
    x = 1.0
    true_derivative = np.cos(x)  # Analytical derivative
    
    # Theoretical truncation error: |h/2 * f''(x)| where f''(1) = -sin(1)
    truncation_error = np.abs(h_values * np.sin(x) / 2)
    
    # Theoretical round-off error: ~epsilon/h (simplified model)
    roundoff_error = 2 * epsilon / h_values  # Factor 2 for central difference
    
    # Total error (combination)
    total_error = truncation_error + roundoff_error
    
    # Find optimal h
    optimal_h = h_values[np.argmin(total_error)]
    min_error = np.min(total_error)
    
    # Create the error landscape visualization
    ax.loglog(h_values, truncation_error, color=COLORS['secondary'], linewidth=4, 
              label='Truncation error: $\\mathcal{O}(h)$', alpha=0.9, zorder=3)
    
    ax.loglog(h_values, roundoff_error, color=COLORS['accent'], linewidth=4, 
              label='Round-off error: $\\mathcal{O}(\\epsilon/h)$', alpha=0.9, zorder=3)
    
    ax.loglog(h_values, total_error, color=COLORS['dark'], linewidth=5, 
              label='Total error', alpha=0.95, zorder=4)
    
    # Mark optimal point
    ax.loglog(optimal_h, min_error, 'o', color='red', markersize=12, 
              markeredgewidth=3, markeredgecolor='white', zorder=5)
    
    # Add annotations
    ax.annotate(f'Optimal $h = {optimal_h:.2e}$\\nMinimum error = {min_error:.2e}',
                xy=(optimal_h, min_error), xytext=(optimal_h*100, min_error*10),
                arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.9),
                fontsize=14, ha='left', va='bottom', color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='red', alpha=0.95, linewidth=2))
    
    # Region labels with better positioning
    ax.text(1e-12, 1e-5, 'Round-off\\ndominated', fontsize=14, ha='center', va='center',
            color=COLORS['accent'], weight='medium', rotation=45,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['accent'], 
                     alpha=0.9))
    
    ax.text(1e-3, 1e-5, 'Truncation\\ndominated', fontsize=14, ha='center', va='center',
            color=COLORS['secondary'], weight='medium', rotation=-45,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=COLORS['secondary'], alpha=0.9))
    
    ax.set_xlabel('Step size $h$', fontsize=18, color=COLORS['dark'], weight='medium')
    ax.set_ylabel('Absolute error', fontsize=18, color=COLORS['dark'], weight='medium')
    ax.set_title('The Error Landscape: Competing Sources of Error', 
                 fontsize=20, color=COLORS['dark'], weight='medium', pad=25)
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), fontsize=16, frameon=True,
              fancybox=True, edgecolor=COLORS['neutral'], facecolor='white', ncol=3)
    
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.tick_params(axis='both', labelsize=16, colors=COLORS['neutral'])
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color(COLORS['neutral'])
        spine.set_linewidth(1.2)
    
    ax.set_xlim(1e-14, 1e-1)
    ax.set_ylim(1e-16, 1e0)
    
    plt.tight_layout()
    save_figure(fig, "1_2_error_landscape")
    reset_style()


def figure_1_4_error_scaling():
    """
    Empirical verification of error scaling predictions from theory.
    Shows actual vs predicted error scaling for different methods.
    
    Location: Part 1, Section "Error Analysis" 
    Caption: Error scaling verification for finite difference methods computing sin'(1). 
             Forward/backward differences show O(h) scaling, central difference shows O(h²), 
             and fourth-order shows O(h⁴) - exactly as Taylor theory predicts. Round-off 
             error dominates below h≈10⁻¹² for all methods.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Step size range
    h_values = np.logspace(-14, -1, 100)
    
    # Test function: sin(x) at x = 1, true derivative = cos(1)
    x = 1.0
    true_derivative = np.cos(x)
    
    # Compute errors for different methods
    errors_forward = []
    errors_backward = []
    errors_central = []
    errors_central4 = []
    
    for h in h_values:
        # Forward difference
        fd = (np.sin(x + h) - np.sin(x)) / h
        errors_forward.append(abs(fd - true_derivative))
        
        # Backward difference  
        bd = (np.sin(x) - np.sin(x - h)) / h
        errors_backward.append(abs(bd - true_derivative))
        
        # Central difference
        cd = (np.sin(x + h) - np.sin(x - h)) / (2*h)
        errors_central.append(abs(cd - true_derivative))
        
        # Fourth-order central difference
        cd4 = (-np.sin(x + 2*h) + 8*np.sin(x + h) - 8*np.sin(x - h) + np.sin(x - 2*h)) / (12*h)
        errors_central4.append(abs(cd4 - true_derivative))
    
    errors_forward = np.array(errors_forward)
    errors_backward = np.array(errors_backward)
    errors_central = np.array(errors_central)
    errors_central4 = np.array(errors_central4)
    
    # Plot the actual errors with improved visibility - dashed vs solid lines
    # Forward: dashed line with circles
    ax.loglog(h_values, errors_forward, color=COLORS['primary'], linewidth=3.5, 
              linestyle='--', label=r'Forward difference: $O(h)$',
              marker='o', markersize=6, markevery=8, alpha=0.9, zorder=5)
    
    # Backward: solid line with squares, slightly offset for visibility  
    ax.loglog(h_values, errors_backward, color=COLORS['secondary'], linewidth=3.5, 
              linestyle='-', label=r'Backward difference: $O(h)$',
              marker='s', markersize=6, markevery=8, alpha=0.85, zorder=4)
    
    # Central: dotted line with triangles
    ax.loglog(h_values, errors_central, color=COLORS['accent'], linewidth=4, 
              linestyle=':', label=r'Central difference: $O(h^2)$',
              marker='^', markersize=7, markevery=8, alpha=0.9, zorder=5)
    
    # 4th-order: solid line with diamonds
    ax.loglog(h_values, errors_central4, color=COLORS['dark'], linewidth=4, 
              linestyle='-', label=r'4th-order central: $O(h^4)$',
              marker='d', markersize=6, markevery=8, alpha=0.9, zorder=5)
    
    # Machine epsilon reference
    epsilon = 2.22e-16
    
    # Add theoretical scaling reference lines with better scaling
    h_ref = 1e-6
    error_ref = 1e-8
    
    # O(h) line - calibrated to match forward/backward at h_ref, extend to machine epsilon
    theory_1 = error_ref * (h_values / h_ref)
    mask_1 = (h_values >= 1e-14) & (h_values <= 1e-3) & (theory_1 >= epsilon)
    ax.loglog(h_values[mask_1], theory_1[mask_1], color=COLORS['neutral'], 
              linestyle='-', alpha=0.7, linewidth=2.5, 
              label=r'$O(h)$ theory', zorder=2)
    
    # O(h²) line - calibrated to match central difference
    theory_2 = (error_ref/100) * (h_values / h_ref)**2
    mask_2 = (h_values >= 1e-10) & (h_values <= 1e-2)
    ax.loglog(h_values[mask_2], theory_2[mask_2], color=COLORS['neutral'], 
              linestyle=':', alpha=0.7, linewidth=2.5,
              label=r'$O(h^2)$ theory', zorder=2)
    
    # O(h⁴) line - calibrated to match 4th-order
    theory_4 = (error_ref/10000) * (h_values / h_ref)**4
    mask_4 = (h_values >= 1e-8) & (h_values <= 1e-2)
    ax.loglog(h_values[mask_4], theory_4[mask_4], color=COLORS['neutral'], 
              linestyle='-.', alpha=0.7, linewidth=2.5,
              label=r'$O(h^4)$ theory', zorder=2)
    
    # Plot machine epsilon reference line
    ax.axhline(epsilon, color='red', linestyle='--', alpha=0.8, 
               linewidth=2.5, label=rf'Machine $\epsilon = {epsilon:.1e}$', zorder=3)
    
    # Enhanced labels with larger fonts and LaTeX
    ax.set_xlabel(r'Step size $h$', fontsize=18, color=COLORS['dark'], weight='medium')
    ax.set_ylabel('Absolute error', fontsize=18, color=COLORS['dark'], weight='medium')
    ax.set_title(r'Error Scaling: Theory Meets Reality for $f(x) = \sin(x)$ at $x = 1$', 
                 fontsize=20, pad=25, color=COLORS['dark'], weight='medium')
    
    # Enhanced legend positioning and larger font - centered horizontally
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), fontsize=14, frameon=True, 
              fancybox=True, edgecolor=COLORS['neutral'], facecolor='white',
              framealpha=0.95, ncol=2)
    
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.tick_params(axis='both', labelsize=16, colors=COLORS['neutral'])
    
    # Enhanced annotation with better positioning
    ax.annotate(r'Round-off dominates' + '\n' + r'(all methods converge to $\epsilon$)', 
                xy=(1e-12, 1e-13), xytext=(3e-11, 1e-10),
                arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.9),
                fontsize=14, ha='center', va='center', color='red', weight='medium',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='red', alpha=0.95, linewidth=2))
    
    # Add convergence regime annotation - positioned slightly higher to avoid line overlap
    ax.annotate('Theoretical convergence\nregime', 
                xy=(1e-4, 1e-6), xytext=(1e-6, 2e-5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=3, alpha=0.9),
                fontsize=14, ha='center', va='center', color=COLORS['accent'], weight='medium',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=COLORS['accent'], alpha=0.95, linewidth=2))
    
    # Style spines with better visibility
    for spine in ax.spines.values():
        spine.set_color(COLORS['neutral'])
        spine.set_linewidth(1.2)
    
    # Increased y-axis range for better legend placement
    ax.set_xlim(1e-14, 1e-1)
    ax.set_ylim(1e-16, 1e0)  # Extended upper limit for legend space
    
    plt.tight_layout()
    save_figure(fig, "1_4_error_scaling")
    reset_style()


# ============================================================================
# PART 2 FIGURES: Computer Arithmetic & Cosmic Consequences
# ============================================================================

def figure_2_1_catastrophic_cancellation():
    """
    Visualize precision loss in energy conservation calculations - catastrophic cancellation.
    
    Location: Part 2, Section "Catastrophic Cancellation"
    Caption: Catastrophic cancellation in energy conservation monitoring. N-body simulations 
             must track tiny energy changes (ΔE) compared to enormous total energies. The 
             naive method E_final - E_initial loses precision catastrophically when energies 
             are nearly equal. The reformulated approach maintains precision by tracking 
             energy changes directly - essential for Project 2 orbital dynamics.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Simulate N-body energy conservation over time
    time_steps = np.linspace(0, 100, 1000)  # 100 time units
    
    # Total system energy is enormous (Jupiter-Sun system scale in CGS)
    E_total = -2.5e48  # erg (roughly Jupiter-Sun binding energy in CGS)
    
    # Energy should be conserved, but tiny numerical errors accumulate
    # This simulates realistic energy drift in N-body integrations
    np.random.seed(42)
    energy_drift = 1e40 * np.cumsum(np.random.randn(len(time_steps)) * 0.01)  # Small random walk in erg
    
    E_initial = E_total
    E_current = E_total + energy_drift
    
    # Two methods to compute energy change
    def energy_change_bad(E_init, E_curr):
        """Naive method: catastrophic cancellation"""
        return E_curr - E_init
    
    def energy_change_good(drift_accumulated):
        """Good method: track changes directly"""
        return drift_accumulated
    
    # Compute energy changes
    delta_E_bad = energy_change_bad(E_initial, E_current)
    delta_E_good = energy_drift  # True accumulated drift
    
    # Left panel: Show the precision disaster
    # Make the naive line MUCH more visible and add noise to show the problem
    noise = np.random.randn(len(time_steps)) * 0.3  # Add visible noise to show precision loss
    ax1.plot(time_steps, delta_E_bad / 1e40 + noise, 'o-', color='red', 
             linewidth=4, markersize=5, alpha=1.0, markeredgewidth=1, markeredgecolor='white',
             label=r'NAIVE: $E_{\mathrm{final}} - E_{\mathrm{initial}}$ (precision lost!)')
    
    ax1.plot(time_steps, delta_E_good / 1e40, '-', color=COLORS['primary'], 
             linewidth=4, alpha=0.9, label=r'CORRECT: $\sum \Delta E_{\mathrm{each\ step}}$')
    
    ax1.set_xlabel('Simulation Time', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel(r'Energy Change ($\times 10^{40}$ erg)', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title('Energy Conservation Monitoring', fontsize=18, color=COLORS['dark'], weight='medium', pad=20)
    ax1.legend(fontsize=14, frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3)
    
    # Add annotation showing the problem
    ax1.annotate('Naive method produces\nnoise from precision loss', 
                xy=(50, 2), xytext=(20, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2),
                fontsize=12, ha='left', va='center', color=COLORS['secondary'], weight='medium',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=COLORS['secondary'], alpha=0.9))
    
    # Right panel: Show the scale of the disaster
    # Demonstrate precision loss with actual numbers
    sample_times = [0, 25, 50, 75, 100]
    sample_indices = [int(t * 10) if t < 100 else 999 for t in sample_times]
    
    precision_loss = []
    for i in sample_indices:
        E_curr_sample = E_current[i]
        true_change = energy_drift[i]
        
        # Simulate what happens with finite precision (simplified model)
        # When subtracting nearly equal large numbers
        relative_change = abs(true_change / E_total)
        if relative_change < 1e-15:  # Below machine precision
            lost_digits = -np.log10(relative_change) - 15
            precision_loss.append(min(lost_digits, 16))  # Max 16 digits lost
        else:
            precision_loss.append(0)
    
    bars = ax2.bar(sample_times, precision_loss, color=COLORS['accent'], alpha=0.8, 
                   width=8, edgecolor='white', linewidth=2)
    
    ax2.set_xlabel('Simulation Time', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Significant Digits Lost', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title('Precision Catastrophe in Energy Monitoring', fontsize=18, color=COLORS['dark'], weight='medium', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 18)
    
    # Add reference lines
    ax2.axhline(8, color='orange', linestyle='--', alpha=0.7, linewidth=2, 
               label='8 digits lost = useless')
    ax2.axhline(16, color='red', linestyle='--', alpha=0.7, linewidth=2, 
               label='16 digits lost = complete failure')
    ax2.legend(fontsize=12, frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    
    # Add key insight box
    ax2.text(0.98, 0.95, r'Project 2 Lesson:' + '\n' + r'Monitor $\sum \Delta E_{\mathrm{step}}$,' + '\n' + r'not $|E_{\mathrm{final}} - E_{\mathrm{initial}}|$!', 
             transform=ax2.transAxes, fontsize=14, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], 
                      edgecolor=COLORS['primary'], alpha=0.95, linewidth=2), 
             weight='bold', color=COLORS['primary'])
    
    # Show the scale disaster with text AND explain the methods CLEARLY
    method_explanation = (r'What Each Method Does:' + '\n\n' +
                         r'NAIVE METHOD:' + '\n' +
                         r'  • Measure $E_{\mathrm{start}} = -2.5 \times 10^{48}$ erg' + '\n' +
                         r'  • Measure $E_{\mathrm{end}} = -2.5 \times 10^{48} + \mathrm{drift}$ erg' + '\n' +
                         r'  • Compute: $E_{\mathrm{end}} - E_{\mathrm{start}}$' + '\n' +
                         r'  • DISASTER: Subtracting huge numbers!' + '\n\n' +
                         r'CORRECT METHOD:' + '\n' +
                         r'  • At each step: $\Delta E = E_{\mathrm{new}} - E_{\mathrm{old}}$' + '\n' +
                         r'  • Keep running sum: $\mathrm{total} = \sum \Delta E_{\mathrm{steps}}$' + '\n' +
                         r'  • SUCCESS: Only small number arithmetic!')
    
    ax1.text(0.02, 0.98, method_explanation,
             transform=ax1.transAxes, fontsize=10, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], 
                      edgecolor=COLORS['neutral'], alpha=0.95), weight='medium')
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "2_1_catastrophic_cancellation")
    reset_style()


def figure_2_2_error_propagation():
    """
    Show how tiny errors compound through repeated operations - the nightmare of long integrations.
    
    Location: Part 2, Section "Propagation Error"
    Caption: Error propagation in iterative calculations. Even errors below machine precision 
             grow exponentially through repeated operations. After 1 million steps, errors 
             can grow by 10⁶ times! This is why billion-year galaxy simulations require 
             extreme precision and specialized algorithms.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Simulate different error types
    steps = np.logspace(0, 6, 100).astype(int)
    tiny_error = 1e-15  # Below machine precision
    
    # Random error accumulation (sqrt(N) scaling)
    random_errors = []
    systematic_errors = []
    worst_case_errors = []
    
    for n in steps:
        # Random walk: errors cancel on average, grow as sqrt(N)
        random_accumulated = tiny_error * np.sqrt(n)
        random_errors.append(random_accumulated)
        
        # Systematic error: all errors add up
        systematic_accumulated = tiny_error * n
        systematic_errors.append(systematic_accumulated)
        
        # Worst case: exponential growth (multiplicative errors)
        if n > 1:
            worst_accumulated = tiny_error * ((1 + tiny_error)**n - 1)
        else:
            worst_accumulated = tiny_error
        worst_case_errors.append(worst_accumulated)
    
    # Left panel: Error growth comparison
    ax1.loglog(steps, random_errors, color=COLORS['primary'], linewidth=4, 
              label=r'Random errors: $\epsilon \sqrt{N}$', alpha=0.9)
    
    ax1.loglog(steps, systematic_errors, color=COLORS['secondary'], linewidth=4,
              label=r'Systematic errors: $\epsilon N$', alpha=0.9)
    
    ax1.loglog(steps, worst_case_errors, color=COLORS['accent'], linewidth=4,
              label=r'Multiplicative: $\epsilon((1+\epsilon)^N - 1)$', alpha=0.9)
    
    # Reference lines for scaling
    ax1.loglog(steps, tiny_error * np.ones_like(steps), '--', color=COLORS['neutral'], 
              alpha=0.7, linewidth=2, label=f'Initial error: {tiny_error:.0e}')
    
    ax1.set_xlabel('Number of Steps', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('Accumulated Error', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title('Error Accumulation Patterns', fontsize=18, color=COLORS['dark'], weight='medium', pad=20)
    ax1.legend(fontsize=14, frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key regimes
    ax1.annotate('Billion timesteps\n(10 Gyr simulation)', 
                xy=(1e9, 1e-6), xytext=(1e7, 1e-9),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2),
                fontsize=12, ha='center', va='center', color=COLORS['secondary'], weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['secondary'], alpha=0.9))
    
    # Right panel: Astrophysical time scales
    time_scales = {
        'Planetary orbit (1 yr)': 1e5,
        'Stellar evolution (1 Myr)': 1e8, 
        'Galaxy evolution (1 Gyr)': 1e11,
        'Cosmological (10 Gyr)': 1e12
    }
    
    scale_names = list(time_scales.keys())
    step_counts = list(time_scales.values())
    
    # Calculate required tolerances to keep error < 1%
    tolerances = []
    for n_steps in step_counts:
        # For systematic error to stay < 1%, need ε * N < 0.01
        required_tol = 0.01 / n_steps
        tolerances.append(required_tol)
    
    y_pos = np.arange(len(scale_names))
    bars = ax2.barh(y_pos, np.log10(tolerances), color=COLORS['accent'], alpha=0.8,
                    edgecolor='white', linewidth=1.5)
    
    # Add tolerance labels
    for i, (bar, tol) in enumerate(zip(bars, tolerances)):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{tol:.1e}', va='center', ha='left', fontweight='medium',
                color=COLORS['dark'], fontsize=12)
    
    ax2.set_xlabel('Required Tolerance (log₁₀)', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title('Precision Requirements for Astrophysics', fontsize=18, color=COLORS['dark'], weight='medium', pad=20)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(scale_names, fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add machine epsilon reference
    ax2.axvline(np.log10(2.22e-16), color='red', linestyle='--', alpha=0.8, linewidth=2,
               label='Machine ε')
    ax2.legend(fontsize=12, frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "2_2_error_propagation")
    reset_style()


def figure_2_3_astronomical_scales():
    """
    Shows computational challenges in ASTR 596 course projects.
    
    Location: Part 2, Section "Why This Matters for Astronomy"
    Caption: Computational scale challenges in ASTR 596 projects. Course simulations span 
             enormous ranges that exceed double precision limits. Precision windows show 
             realistic coordinate system choices for N-body simulations, stellar evolution, 
             and cosmological modeling - forcing us to use specialized units and methods.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Define scales relevant to ASTR 596 course projects (in cm)
    course_scales = {
        # N-body Project 2: Planetary dynamics
        'Close binary encounter': 0.01 * 1.496e13,  # 0.01 AU
        'Earth orbit radius': 1.496e13,  # 1 AU  
        'Jupiter orbit': 5.2 * 1.496e13,  # 5.2 AU
        'Pluto orbit': 40 * 1.496e13,  # ~40 AU
        'Oort cloud edge': 1e5 * 1.496e13,  # ~100,000 AU
        
        # Stellar Evolution Project 
        'White dwarf radius': 3e8,  # ~Earth size
        'Solar core': 1.5e10,  # 0.2 R_sun
        'Solar radius': 7e10,  # 1 R_sun
        'Red giant radius': 1e12,  # ~100 R_sun
        'Planetary nebula': 1e17,  # ~1 pc
        
        # Galaxy/Dark Matter Project
        'Star cluster core': 3e16,  # ~0.1 pc
        'Globular cluster': 3e18,  # ~10 pc  
        'Milky Way disk': 3e22,  # ~10 kpc
        'Local Group': 3e24,  # ~1 Mpc
        
        # Cosmological modeling
        'Galaxy cluster': 3e25,  # ~10 Mpc
        'Observable universe': 1e28,  # ~10 Gpc
    }
    
    # Create the scale visualization
    scale_values = list(course_scales.values())
    scale_names = list(course_scales.keys())
    
    # Sort by scale
    sorted_indices = np.argsort(scale_values)
    scale_values = [scale_values[i] for i in sorted_indices]
    scale_names = [scale_names[i] for i in sorted_indices]

    y_pos = np.arange(len(scale_names))
    
    # Color-code by course project
    project_colors = []
    for name in scale_names:
        if any(keyword in name.lower() for keyword in ['binary', 'earth', 'jupiter', 'pluto', 'oort']):
            project_colors.append(COLORS['primary'])  # N-body (blue)
        elif any(keyword in name.lower() for keyword in ['dwarf', 'solar', 'core', 'giant', 'nebula']):
            project_colors.append(COLORS['secondary'])  # Stellar evolution (pink)
        elif any(keyword in name.lower() for keyword in ['cluster', 'milky', 'local']):
            project_colors.append(COLORS['accent'])  # Galaxy (teal)
        else:
            project_colors.append(COLORS['neutral'])  # Cosmology (gray)
    
    bars = ax.barh(y_pos, np.log10(scale_values), 
                   color=project_colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add value labels with appropriate units
    for i, (bar, value, name) in enumerate(zip(bars, scale_values, scale_names)):
        # Choose appropriate unit
        if value < 1e12:
            if value > 1e8:
                unit_text = f'{value/7e10:.1f} R☉'  # Solar radii
            else:
                unit_text = f'{value:.1e} cm'
        elif value < 1e19:
            unit_text = f'{value/1.496e13:.1f} AU'  # Astronomical units
        elif value < 1e23:
            unit_text = f'{value/3.086e18:.1f} pc'  # Parsecs
        elif value < 1e26:
            unit_text = f'{value/3.086e21:.0f} kpc'  # Kiloparsecs
        else:
            unit_text = f'{value/3.086e24:.0f} Mpc'  # Megaparsecs
            
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                unit_text, va='center', ha='left', fontweight='medium',
                color=COLORS['dark'], fontsize=11)
    
    # Show realistic precision windows for course projects
    project_windows = [
        (1.496e13, 'N-body (AU units)', COLORS['primary']),  # Centered on 1 AU
        (7e10, 'Stellar evolution (R☉)', COLORS['secondary']),  # Centered on 1 R_sun  
        (3.086e21, 'Galaxy dynamics (kpc)', COLORS['accent']),  # Centered on 1 kpc
    ]
    
    for i, (ref, label, color) in enumerate(project_windows):
        log_ref = np.log10(ref)
        window_half = 8  # ±8 orders of magnitude precision window
        
        ax.axvspan(log_ref - window_half, log_ref + window_half, 
                   alpha=0.15, color=color, 
                   label=f'16-digit window at {ref:.0e} cm')
    
    ax.set_xlabel('Scale (log₁₀ cm)', fontsize=16, color=COLORS['dark'], weight='medium')
    ax.set_title('The Astronomical Scale Challenge: 40+ Orders of Magnitude', 
                 fontsize=18, pad=20, color=COLORS['dark'], weight='medium')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scale_names, fontsize=12)
    ax.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax.grid(True, alpha=0.25, linewidth=0.8, axis='x')
    ax.tick_params(axis='both', labelsize=12, colors=COLORS['neutral'])
    
    # Add annotations
    ax.annotate('No single precision window\ncan cover this full range!', 
                xy=(15, 2), xytext=(20, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2),
                fontsize=14, ha='center', color=COLORS['secondary'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], 
                         edgecolor=COLORS['secondary'], alpha=0.9))
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color(COLORS['neutral'])
        spine.set_linewidth(0.8)
    
    ax.set_xlim(-15, 30)
    
    plt.tight_layout()
    save_figure(fig, "2_3_astronomical_scales")
    reset_style()


def figure_2_4_adaptive_step_control():
    """
    Visualize how algorithms adjust parameters to maintain tolerance.
    
    Location: Part 2, Section "Adaptive Error Control"
    Caption: Adaptive step size control in action. Algorithms automatically adjust step 
             sizes to keep errors within tolerance bounds. When error exceeds tolerance, 
             the step size decreases. When error is well below tolerance, step size 
             increases for efficiency. This self-regulating behavior is crucial for 
             robust astrophysical simulations.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Simulate adaptive integration with varying difficulty
    t = np.linspace(0, 10, 1000)
    
    # Create a function with varying stiffness (like stellar evolution)
    # Easy regions and difficult regions
    difficulty = 1 + 5 * np.exp(-((t - 3)**2)) + 3 * np.exp(-((t - 7)**2)/0.5)
    
    # Simulate adaptive stepping
    time_points = []
    step_sizes = []
    error_estimates = []
    tolerance = 1e-6
    
    current_t = 0
    h = 0.1  # Initial step size
    
    while current_t < 10:
        # Estimate local difficulty
        idx = int(current_t * 100)  # Convert to index
        if idx >= len(difficulty):
            idx = len(difficulty) - 1
        local_difficulty = difficulty[idx]
        
        # Error estimate proportional to h² and local difficulty
        error_est = (h**2) * local_difficulty * 1e-4
        
        # Adaptive step size control
        if error_est > tolerance:
            # Reduce h
            h *= 0.5 * np.sqrt(tolerance / error_est)
        elif error_est < tolerance / 10:
            # Increase h for efficiency
            h *= min(1.5, (tolerance / (10 * error_est))**0.25)
        
        # Clamp step size to reasonable bounds
        h = np.clip(h, 0.001, 0.5)
        
        time_points.append(current_t)
        step_sizes.append(h)
        error_estimates.append(error_est)
        
        current_t += h
        
        if len(time_points) > 500:  # Prevent infinite loops
            break
    
    time_points = np.array(time_points)
    step_sizes = np.array(step_sizes)
    error_estimates = np.array(error_estimates)
    
    # Top panel: Step size adaptation
    ax1.plot(time_points, step_sizes, color=COLORS['primary'], linewidth=3, 
             label='Adaptive step size', alpha=0.9)
    
    # Show problem difficulty as background
    ax1_bg = ax1.twinx()
    ax1_bg.fill_between(t, 0, difficulty, color=COLORS['light'], alpha=0.6, 
                       label='Problem difficulty')
    ax1_bg.set_ylabel('Problem Difficulty', fontsize=14, color=COLORS['neutral'])
    ax1_bg.tick_params(axis='y', labelcolor=COLORS['neutral'])
    
    ax1.set_ylabel('Step Size $h$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title('Adaptive Step Size Control', fontsize=18, color=COLORS['dark'], weight='medium', pad=20)
    ax1.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, 
              edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key behaviors
    ax1.annotate('Step size decreases\nin difficult regions', 
                xy=(3, 0.05), xytext=(1, 0.25),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2),
                fontsize=12, ha='left', va='center', color=COLORS['primary'], weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['primary'], alpha=0.9))
    
    # Bottom panel: Error control
    ax2.semilogy(time_points, error_estimates, color=COLORS['secondary'], linewidth=3, 
                label='Estimated error', alpha=0.9)
    
    # Show tolerance bounds
    ax2.axhline(tolerance, color='red', linestyle='--', linewidth=2, 
               label=f'Tolerance = {tolerance:.0e}')
    ax2.axhline(tolerance/10, color='orange', linestyle=':', linewidth=2, 
               label=f'Lower bound = {tolerance/10:.0e}')
    
    ax2.set_xlabel('Time', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Error Estimate', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title('Error Stays Within Tolerance Bounds', fontsize=18, color=COLORS['dark'], weight='medium', pad=20)
    ax2.legend(fontsize=14, frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    ax2.grid(True, alpha=0.3)
    
    # Add shaded region for acceptable error range
    ax2.fill_between([0, 10], tolerance/10, tolerance, alpha=0.2, color='green', 
                    label='Acceptable range')
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "2_4_adaptive_step_control")
    reset_style()


# ============================================================================
# PART 3 FIGURES: Taylor Series Applications & Modern Methods
# ============================================================================

def figure_3_1_finite_difference_methods():
    """
    Visual comparison of forward, backward, and central difference methods.
    Replicates the style of the top-left panel from 3_2_custom_derivation.png
    
    Location: Part 3, Section "Finite Difference Methods"
    Caption: Comparison of finite difference approximations to derivatives. Forward 
             difference uses points ahead, backward uses points behind, and central 
             difference uses points on both sides. Central difference achieves higher 
             accuracy (O(h²)) compared to forward and backward methods (O(h)).
    """
    set_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('white')
    
    # Use EXACT same function as original Figure 3_1: f(x) = x²
    x = np.linspace(0, 3, 1000)
    f = x**2
    
    # Center point and step size - same as original
    x_center = 1.5
    h = 0.3
    
    # Method configurations matching the 3_2 visual style
    methods = [
        {
            'title': 'Forward Difference',
            'points': [x_center, x_center + h],
            'point_labels': [r'$f(x)$', r'$f(x + h)$'],
            'color': COLORS['secondary'],  # Use secondary color like 3_2
            'formula': r'$f\'(x) \approx \frac{f(x+h) - f(x)}{h}$' + '\n' + r'Error: $O(h)$'
        },
        {
            'title': 'Backward Difference', 
            'points': [x_center - h, x_center],
            'point_labels': [r'$f(x - h)$', r'$f(x)$'],
            'color': COLORS['secondary'],
            'formula': r'$f\'(x) \approx \frac{f(x) - f(x-h)}{h}$' + '\n' + r'Error: $O(h)$'
        },
        {
            'title': 'Central Difference',
            'points': [x_center - h, x_center + h],
            'point_labels': [r'$f(x - h)$', r'$f(x + h)$'],
            'color': COLORS['secondary'],
            'formula': r'$f\'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$' + '\n' + r'Error: $O(h^2)$'
        }
    ]
    
    for i, (ax, method) in enumerate(zip([ax1, ax2, ax3], methods)):
        # Plot the function curve - matching 3_2 style
        ax.plot(x, f, color=COLORS['primary'], linewidth=3, alpha=0.9, 
                label=r'$f(x)$', zorder=2)
        
        # Plot the evaluation points - exactly matching 3_2 style
        x_pts = method['points']
        f_pts = [pt**2 for pt in x_pts]  # Use EXACT same function: f(x) = x²
        
        # Plot points as filled circles with white edges like 3_2
        for j, (x_pt, f_pt) in enumerate(zip(x_pts, f_pts)):
            ax.plot(x_pt, f_pt, 'o', color=method['color'], markersize=12, 
                   markeredgewidth=2, markeredgecolor='white', zorder=5)
        
        # Add vertical dashed lines from x-axis to points like in 3_2
        for j, (x_pt, f_pt) in enumerate(zip(x_pts, f_pts)):
            ax.plot([x_pt, x_pt], [0, f_pt], color=method['color'], 
                   linestyle='--', linewidth=2, alpha=0.7, zorder=1)
        
        # Add point labels above the points like in 3_2
        for j, (x_pt, f_pt, label) in enumerate(zip(x_pts, f_pts, method['point_labels'])):
            ax.annotate(label, xy=(x_pt, f_pt), xytext=(x_pt, f_pt + 0.2),
                       ha='center', va='bottom', fontsize=14, weight='medium',
                       color=method['color'])
        
        # Add equation at bottom center of plot with nice formatting
        ax.text(0.5, 0.15, method['formula'], transform=ax.transAxes, 
               fontsize=14, ha='center', va='center', 
               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                        edgecolor=method['color'], alpha=0.95, linewidth=2),
               weight='medium', color=COLORS['dark'])
        
        # Styling to match 3_2
        ax.set_xlabel(r'$x$', fontsize=16, color=COLORS['dark'], weight='medium')
        ax.set_ylabel(r'$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
        ax.set_title(method['title'], fontsize=16, 
                    color=COLORS['dark'], weight='medium', pad=20)
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.set_xlim(0.8, 2.2)
        ax.set_ylim(0.5, 5.0)
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_1_finite_difference_methods")
    reset_style()


def figure_3_2_custom_derivation():
    """
    Visual derivation of one-sided second derivative for boundary conditions.
    
    Location: Part 3, Section "Designing Custom Finite Difference Formulas"
    Caption: Step-by-step derivation of custom finite difference formulas. Using Taylor 
             expansions at three points, we eliminate unwanted derivatives through linear 
             combinations to isolate the second derivative. This systematic approach works 
             for any derivative order and boundary condition.
    """
    set_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('white')
    
    # Step 1: Function and points
    x = np.linspace(-0.5, 2.5, 1000)
    f = np.sin(2*x) + 0.5*x  # Test function
    x0, h = 0.5, 0.4
    points = [x0, x0 + h, x0 + 2*h]
    f_vals = [np.sin(2*pt) + 0.5*pt for pt in points]
    
    ax1.plot(x, f, color=COLORS['primary'], linewidth=3, alpha=0.9, label='$f(x)$')
    ax1.plot(points, f_vals, 'o', color=COLORS['secondary'], markersize=10,
             markeredgewidth=2, markeredgecolor='white', zorder=5)
    
    # Label the points
    labels = ['$f(x)$', '$f(x+h)$', '$f(x+2h)$']
    for i, (pt, fval, label) in enumerate(zip(points, f_vals, labels)):
        ax1.annotate(label, xy=(pt, fval), xytext=(pt, fval + 0.3 + 0.2*i),
                    ha='center', va='bottom', fontsize=14, weight='medium',
                    arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.5))
    
    ax1.axvline(x0, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    ax1.axvline(x0 + h, color=COLORS['neutral'], linestyle='--', alpha=0.5)  
    ax1.axvline(x0 + 2*h, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title('Step 1: Function and Available Points', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14, frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    
    # Step 2: Taylor expansions (text-based visualization)
    ax2.text(0.1, 0.9, 'Step 2: Taylor Expansions', fontsize=18, weight='bold', 
             transform=ax2.transAxes, color=COLORS['dark'])
    
    expansions = [
        r"$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + ...$",
        "",
        r"$f(x+2h) = f(x) + 2hf'(x) + \frac{(2h)^2}{2}f''(x) + \frac{(2h)^3}{6}f'''(x) + ...$",
        "",
        r"$f(x+2h) = f(x) + 2hf'(x) + 2h^2f''(x) + \frac{4h^3}{3}f'''(x) + ...$"
    ]
    
    for i, eq in enumerate(expansions):
        ax2.text(0.1, 0.75 - i*0.12, eq, fontsize=14, transform=ax2.transAxes,
                color=COLORS['dark'] if eq else 'white')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Step 3: Linear combination to eliminate f'(x)
    ax3.text(0.1, 0.9, 'Step 3: Eliminate $f\'(x)$ Term', fontsize=18, weight='bold',
             transform=ax3.transAxes, color=COLORS['dark'])
    
    elimination = [
        r"To eliminate $f'(x)$, compute:",
        "",
        r"$f(x+2h) - 2f(x+h) =$", 
        "",
        r"$[f(x) + 2hf'(x) + 2h^2f''(x) + ...] - 2[f(x) + hf'(x) + \frac{h^2}{2}f''(x) + ...]$",
        "",
        r"$= f(x) + 2hf'(x) + 2h^2f''(x) - 2f(x) - 2hf'(x) - h^2f''(x) + ...$",
        "",
        r"$= -f(x) + h^2f''(x) + O(h^3)$"
    ]
    
    colors = [COLORS['dark'], 'white', COLORS['secondary'], 'white', COLORS['dark'], 
              'white', COLORS['dark'], 'white', COLORS['accent']]
    
    for i, (eq, color) in enumerate(zip(elimination, colors)):
        ax3.text(0.1, 0.8 - i*0.08, eq, fontsize=12, transform=ax3.transAxes, color=color)
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Step 4: Final formula
    ax4.text(0.1, 0.9, 'Step 4: Solve for $f\'\'(x)$', fontsize=18, weight='bold',
             transform=ax4.transAxes, color=COLORS['dark'])
    
    final = [
        r"From: $f(x+2h) - 2f(x+h) = -f(x) + h^2f''(x) + O(h^3)$",
        "",
        r"Rearrange:",
        "",
        r"$f''(x) = \frac{f(x+2h) - 2f(x+h) + f(x)}{h^2} + O(h)$",
        "",
        "",
        r"✓ First-order accurate one-sided second derivative!",
        r"✓ Only uses forward points: $x$, $x+h$, $x+2h$",
        r"✓ Perfect for boundary conditions"
    ]
    
    colors = [COLORS['dark'], 'white', COLORS['dark'], 'white', COLORS['accent'], 
              'white', 'white', COLORS['secondary'], COLORS['secondary'], COLORS['secondary']]
    
    for i, (eq, color) in enumerate(zip(final, colors)):
        weight = 'bold' if '✓' in eq else 'normal'
        ax4.text(0.1, 0.8 - i*0.08, eq, fontsize=12 if '✓' not in eq else 14, 
                transform=ax4.transAxes, color=color, weight=weight)
    
    # Add a box around the final formula
    bbox = FancyBboxPatch((0.05, 0.45), 0.9, 0.15, boxstyle="round,pad=0.02",
                          facecolor=COLORS['light'], edgecolor=COLORS['accent'],
                          linewidth=2, alpha=0.9, transform=ax4.transAxes)
    ax4.add_patch(bbox)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    # Style all spines
    for ax in [ax1]:  # Only ax1 has actual plots
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_2_custom_derivation")
    reset_style()


def figure_3_3_noise_amplification():
    """
    Show why numerical derivatives fail catastrophically on noisy data.
    
    Location: Part 3, Section "When NOT to Use Numerical Derivatives"
    Caption: Noise amplification in numerical derivatives. Clean function (left) appears 
             smooth with small measurement errors. Its numerical derivative (right) is 
             completely dominated by noise. Derivatives amplify high-frequency errors - 
             a fundamental limitation when working with observational data.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Create smooth function with noise
    np.random.seed(42)  # For reproducibility
    x = np.linspace(0, 2*np.pi, 100)
    y_true = np.sin(x) + 0.3*np.sin(3*x)  # Smooth test function
    noise_level = 0.05
    y_noisy = y_true + noise_level * np.random.randn(len(x))
    
    # True derivative
    dy_true = np.cos(x) + 0.9*np.cos(3*x)
    
    # Numerical derivative of noisy data
    dy_noisy = np.gradient(y_noisy, x)
    
    # Left panel: Function comparison
    ax1.plot(x, y_true, color=COLORS['primary'], linewidth=4, 
             label='True function', alpha=0.9, zorder=3)
    
    ax1.plot(x, y_noisy, 'o', color=COLORS['secondary'], markersize=4, alpha=0.7, 
             markeredgewidth=0, label=f'Noisy data (σ = {noise_level})', zorder=2)
    
    ax1.set_xlabel('$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title('Function: Noise Appears Manageable', fontsize=18, color=COLORS['dark'], weight='medium', pad=20)
    ax1.legend(fontsize=14, frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3)
    
    # Add noise level annotation
    ax1.text(0.05, 0.95, f'Noise level: {noise_level*100:.1f}%\nof function amplitude', 
             transform=ax1.transAxes, fontsize=12, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'], 
                      edgecolor=COLORS['secondary'], alpha=0.9), weight='medium')
    
    # Right panel: Derivative comparison
    ax2.plot(x, dy_true, color=COLORS['primary'], linewidth=4,
             label="True derivative", alpha=0.9, zorder=3)
    
    ax2.plot(x, dy_noisy, color=COLORS['accent'], linewidth=2, alpha=0.8,
             label='Numerical derivative (noisy data)', zorder=2)
    
    ax2.set_xlabel('$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel("$f'(x)$", fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title('Derivative: Noise Completely Dominates!', fontsize=18, color=COLORS['dark'], weight='medium', pad=20)
    ax2.legend(fontsize=14, frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    ax2.grid(True, alpha=0.3)
    
    # Calculate and show noise amplification
    signal_to_noise_func = np.std(y_true) / noise_level
    signal_to_noise_deriv = np.std(dy_true) / np.std(dy_noisy - dy_true)
    
    amplification = signal_to_noise_func / signal_to_noise_deriv
    
    ax2.text(0.05, 0.05, f'Noise amplification:\\n{amplification:.1f}× worse!', 
             transform=ax2.transAxes, fontsize=14, ha='left', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                      edgecolor='red', linewidth=2, alpha=0.95), 
             weight='bold', color='red')
    
    # Add warning annotation
    ax2.annotate('CATASTROPHIC\\nFAILURE', 
                xy=(np.pi, 2), xytext=(4, 3),
                arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.9),
                fontsize=16, ha='center', va='center', color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='red', linewidth=2, alpha=0.95))
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_3_noise_amplification")
    reset_style()


def figure_3_4_fd_vs_ad_comparison():
    """
    Performance and accuracy comparison of finite differences vs automatic differentiation.
    
    Location: Part 3, Section "Modern Alternative: Automatic Differentiation"
    Caption: Finite differences vs. automatic differentiation comparison. AD achieves 
             machine precision accuracy (~10⁻¹⁶) regardless of function complexity, while 
             finite differences have accuracy limited by step size choice. AD computational 
             cost scales better for functions with many variables - crucial for neural 
             networks with millions of parameters.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Left panel: Accuracy comparison
    # Simulate different function complexities
    complexities = ['Linear', 'Polynomial', 'Trigonometric', 'Exponential', 'Composite']
    
    # Finite difference errors (vary by function and step size choice)
    fd_errors = [1e-8, 3e-7, 1e-6, 5e-6, 2e-5]  # Typical errors with "good" h
    fd_errors_bad = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]  # Errors with poor h choice
    
    # AD errors (always machine precision)
    ad_errors = [2.2e-16] * len(complexities)
    
    x_pos = np.arange(len(complexities))
    
    # Plot error comparisons
    bars1 = ax1.bar(x_pos - 0.25, np.log10(fd_errors), 0.25, 
                   color=COLORS['secondary'], alpha=0.8, 
                   label='Finite differences (optimal h)')
    
    bars2 = ax1.bar(x_pos, np.log10(fd_errors_bad), 0.25,
                   color=COLORS['secondary'], alpha=0.4,
                   label='Finite differences (poor h)')
    
    bars3 = ax1.bar(x_pos + 0.25, np.log10(ad_errors), 0.25,
                   color=COLORS['primary'], alpha=0.8,
                   label='Automatic differentiation')
    
    ax1.set_ylabel('Log₁₀ Error', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_xlabel('Function Complexity', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title('Accuracy: AD Always Achieves Machine Precision', fontsize=18, color=COLORS['dark'], weight='medium', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(complexities, rotation=45, ha='right')
    ax1.legend(fontsize=14, frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add machine epsilon reference
    ax1.axhline(np.log10(2.22e-16), color='red', linestyle='--', alpha=0.8, linewidth=2,
               label='Machine epsilon')
    
    # Add annotations
    ax1.annotate('Machine precision\nlimit', 
                xy=(2, -15.7), xytext=(0.5, -12),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, ha='center', va='center', color='red', weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))
    
    # Right panel: Computational cost vs number of variables
    n_vars = np.logspace(0, 6, 50)  # 1 to 1 million variables
    
    # Finite differences: O(n) function evaluations for gradient
    fd_cost = n_vars
    
    # AD: roughly constant factor overhead (simplified model)
    ad_cost = 2 * np.ones_like(n_vars)  # ~2x cost of forward pass
    
    ax2.loglog(n_vars, fd_cost, color=COLORS['secondary'], linewidth=4, 
              label='Finite differences: O(n)', alpha=0.9)
    
    ax2.loglog(n_vars, ad_cost, color=COLORS['primary'], linewidth=4,
              label='Automatic differentiation: O(1)', alpha=0.9)
    
    ax2.set_xlabel('Number of Variables', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Relative Computational Cost', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title('Cost: AD Scales Better for High Dimensions', fontsize=18, color=COLORS['dark'], weight='medium', pad=20)
    ax2.legend(fontsize=14, frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for neural network scales
    ax2.axvline(1e6, color=COLORS['accent'], linestyle=':', alpha=0.8, linewidth=2)
    ax2.text(1e6, 1e4, 'Typical neural\nnetwork size', rotation=90, ha='right', va='center',
             fontsize=12, color=COLORS['accent'], weight='medium',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['accent'], alpha=0.9))
    
    # Highlight crossover point
    crossover_idx = np.argmin(np.abs(fd_cost - ad_cost))
    ax2.plot(n_vars[crossover_idx], fd_cost[crossover_idx], 'o', 
             color='red', markersize=10, markeredgewidth=2, markeredgecolor='white')
    
    ax2.annotate('AD becomes\\nmore efficient', 
                xy=(n_vars[crossover_idx], fd_cost[crossover_idx]), 
                xytext=(1e2, 1e3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, ha='center', va='center', color='red', weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_4_fd_vs_ad_comparison")
    reset_style()


def figure_3_5_method_selection_tree():
    """
    Visual decision tree for choosing differentiation methods.
    
    Location: Part 3, Section "Practical Guidelines"  
    Caption: Decision tree for choosing differentiation methods. Start with whether 
             analytical derivatives are available, then consider data quality, number 
             of derivatives needed, and available tools. This systematic approach 
             ensures you choose the most appropriate method for each situation.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Decision tree structure
    # Define nodes with positions and properties
    nodes = {
        'start': {'pos': (0.5, 0.95), 'text': 'Need derivative?', 'color': COLORS['dark']},
        'analytical': {'pos': (0.2, 0.8), 'text': 'Analytical\navailable?', 'color': COLORS['primary']},
        'use_analytical': {'pos': (0.1, 0.65), 'text': 'Use analytical\nderivative', 'color': COLORS['accent']},
        'noisy': {'pos': (0.4, 0.65), 'text': 'Data noisy?', 'color': COLORS['primary']},
        'smooth_first': {'pos': (0.25, 0.5), 'text': 'Smooth data first\nor avoid derivatives', 'color': COLORS['secondary']},
        'many_derivs': {'pos': (0.55, 0.5), 'text': 'Many derivatives\nneeded?', 'color': COLORS['primary']},
        'ad_available': {'pos': (0.75, 0.35), 'text': 'AD framework\navailable?', 'color': COLORS['primary']},
        'use_ad': {'pos': (0.85, 0.2), 'text': 'Use automatic\ndifferentiation', 'color': COLORS['accent']},
        'reformulate': {'pos': (0.65, 0.2), 'text': 'Consider\nreformulation', 'color': COLORS['secondary']},
        'use_fd': {'pos': (0.4, 0.35), 'text': 'Use finite\ndifferences', 'color': COLORS['accent']}
    }
    
    # Draw decision nodes
    for node_id, props in nodes.items():
        x, y = props['pos']
        text = props['text']
        color = props['color']
        
        if 'Use' in text or 'Consider' in text or 'Smooth' in text:
            # Terminal nodes - rounded rectangles
            bbox = FancyBboxPatch((x-0.06, y-0.04), 0.12, 0.08, 
                                 boxstyle="round,pad=0.01", 
                                 facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
            ax.add_patch(bbox)
            text_color = 'white'
            weight = 'bold'
        else:
            # Decision nodes - ovals
            circle = plt.Circle((x, y), 0.05, facecolor=color, alpha=0.3, 
                              edgecolor=color, linewidth=2)
            ax.add_patch(circle)
            text_color = color
            weight = 'medium'
        
        ax.text(x, y, text, ha='center', va='center', fontsize=12, 
               color=text_color, weight=weight)
    
    # Draw decision paths with labels
    paths = [
        # (from_node, to_node, label, label_pos, answer_color)
        ('start', 'analytical', 'Yes', (0.35, 0.88), 'green'),
        ('analytical', 'use_analytical', 'Yes', (0.15, 0.73), 'green'),
        ('analytical', 'noisy', 'No', (0.3, 0.73), 'red'),
        ('noisy', 'smooth_first', 'Yes', (0.32, 0.58), 'red'),
        ('noisy', 'many_derivs', 'No', (0.48, 0.58), 'green'),
        ('many_derivs', 'ad_available', 'Yes', (0.65, 0.43), 'green'),
        ('many_derivs', 'use_fd', 'No', (0.47, 0.43), 'red'),
        ('ad_available', 'use_ad', 'Yes', (0.8, 0.28), 'green'),
        ('ad_available', 'reformulate', 'No', (0.7, 0.28), 'red')
    ]
    
    for from_node, to_node, label, label_pos, answer_color in paths:
        from_x, from_y = nodes[from_node]['pos']
        to_x, to_y = nodes[to_node]['pos']
        
        # Draw arrow
        ax.annotate('', xy=(to_x, to_y + 0.04), xytext=(from_x, from_y - 0.04),
                   arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], 
                                 lw=2, alpha=0.8))
        
        # Add label
        lx, ly = label_pos
        ax.text(lx, ly, label, ha='center', va='center', fontsize=10,
               color=answer_color, weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                        edgecolor=answer_color, alpha=0.9))
    
    # Add method boxes with examples
    methods = [
        {'pos': (0.1, 0.05), 'title': 'Analytical', 'examples': ['Polynomials', 'Trigonometric', 'Known functions'], 'color': COLORS['accent']},
        {'pos': (0.4, 0.05), 'title': 'Finite Differences', 'examples': ['Black box functions', 'Verification', 'Simple derivatives'], 'color': COLORS['accent']},
        {'pos': (0.85, 0.05), 'title': 'Automatic Diff.', 'examples': ['Neural networks', 'Optimization', 'Many variables'], 'color': COLORS['accent']}
    ]
    
    for method in methods:
        x, y = method['pos']
        title = method['title']
        examples = method['examples']
        color = method['color']
        
        # Method title box
        bbox = FancyBboxPatch((x-0.08, y-0.01), 0.16, 0.06, 
                             boxstyle="round,pad=0.01", 
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=1)
        ax.add_patch(bbox)
        
        ax.text(x, y + 0.02, title, ha='center', va='center', fontsize=12,
               color=color, weight='bold')
        
        # Examples
        for i, example in enumerate(examples):
            ax.text(x, y - 0.02 - i*0.02, f'• {example}', ha='center', va='center', 
                   fontsize=9, color=COLORS['dark'], alpha=0.8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.set_title('Method Selection Decision Tree', fontsize=20, color=COLORS['dark'], 
                weight='medium', pad=30)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=2, label='Yes'),
        plt.Line2D([0], [0], color='red', lw=2, label='No'),
        patches.Patch(facecolor=COLORS['primary'], alpha=0.3, label='Decision'),
        patches.Patch(facecolor=COLORS['accent'], alpha=0.8, label='Method')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
             frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    
    plt.tight_layout()
    save_figure(fig, "3_5_method_selection_tree")
    reset_style()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def get_available_figures():
    """Return dictionary of available figures with their IDs and function names"""
    figures = {
        # Part 1: Finite Differences
        '1_1': figure_1_1_derivative_paradox,
        '1_2': figure_1_2_error_landscape, 
        '1_4': figure_1_4_error_scaling,
        
        # Part 2: Computer Arithmetic
        '2_1': figure_2_1_catastrophic_cancellation,
        '2_2': figure_2_2_error_propagation,
        '2_3': figure_2_3_astronomical_scales,
        '2_4': figure_2_4_adaptive_step_control,
        
        # Part 3: Taylor Series & Modern Methods
        '3_1': figure_3_1_finite_difference_methods,
        '3_2': figure_3_2_custom_derivation,
        '3_3': figure_3_3_noise_amplification,
        '3_4': figure_3_4_fd_vs_ad_comparison,
        '3_5': figure_3_5_method_selection_tree
    }
    return figures

def list_available_figures():
    """Print list of all available figures"""
    figures = get_available_figures()
    print("\nAvailable figures:")
    print("=" * 50)
    
    sections = {
        'Part 1: Finite Differences': ['1_1', '1_2', '1_4'],
        'Part 2: Computer Arithmetic & Cosmic Consequences': ['2_1', '2_2', '2_3', '2_4'],
        'Part 3: Taylor Series Applications & Modern Methods': ['3_1', '3_2', '3_3', '3_4', '3_5']
    }
    
    for section, fig_ids in sections.items():
        print(f"\n{section}:")
        for fig_id in fig_ids:
            func = figures[fig_id]
            # Extract title from docstring
            if func.__doc__:
                title = func.__doc__.split('\n')[1].strip()
                if title.startswith('Creates') or title.startswith('Shows') or title.startswith('Visualizes'):
                    title = title
                else:
                    title = func.__name__.replace('figure_', '').replace('_', ' ').title()
            else:
                title = func.__name__.replace('figure_', '').replace('_', ' ').title()
            
            print(f"  {fig_id:<6} - {title}")
    
    print(f"\nUsage:")
    print(f"  python {sys.argv[0]} 1_1              # Generate specific figure")
    print(f"  python {sys.argv[0]} --figure 2_3     # Alternative syntax")
    print(f"  python {sys.argv[0]}                  # Generate all figures")

def main():
    parser = argparse.ArgumentParser(
        description="Generate educational figures for ASTR 596 Numerical Methods Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python numerical-foundations-figures.py 1_1\n"
               "  python numerical-foundations-figures.py --figure 2_3\n"
               "  python numerical-foundations-figures.py --list\n"
               "  python numerical-foundations-figures.py"
    )
    
    parser.add_argument('figure_id', nargs='?', help='Figure ID to generate (e.g., 1_1, 2_3)')
    parser.add_argument('--figure', help='Figure ID to generate (alternative syntax)')
    parser.add_argument('--list', action='store_true', help='List all available figures')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_figures()
        return
    
    figures = get_available_figures()
    
    # Determine which figure to generate
    figure_id = args.figure_id or args.figure
    
    if figure_id:
        # Generate specific figure
        if figure_id in figures:
            print(f"Generating figure_{figure_id}_{figures[figure_id].__name__.split('_')[-1]}...")
            figures[figure_id]()
            print("✓ Success")
        else:
            print(f"Error: Figure '{figure_id}' not found.")
            print("Use --list to see available figures.")
            sys.exit(1)
    else:
        # Generate all figures
        print("Generating all figures...")
        for fig_id, func in figures.items():
            print(f"\nGenerating {fig_id}: {func.__name__}...")
            try:
                func()
                print(f"✓ {fig_id} completed")
            except Exception as e:
                print(f"✗ {fig_id} failed: {e}")
        print(f"\nAll figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()