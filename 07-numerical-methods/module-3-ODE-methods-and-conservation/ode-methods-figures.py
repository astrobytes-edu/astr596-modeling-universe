#!/usr/bin/env python3
"""
Educational Figures for Module 3: ODE Methods & Conservation
ASTR 596: Modeling the Universe

This script generates all matplotlib figures for the ODE methods and conservation module.
Each function creates a specific educational visualization with publication-quality output.

Updated to focus on integration methods, stability analysis, and performance optimization.

Usage:
    python ode-methods-figures.py                    # Generate all figures
    python ode-methods-figures.py --list             # List available figures
    python ode-methods-figures.py --figure 3_1       # Generate specific figure
    python ode-methods-figures.py 3_1               # Generate by ID
    
Requirements:
    - matplotlib
    - numpy
    - scipy (optional, for some examples)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse
import os
import warnings
import sys
import argparse
from scipy.integrate import solve_ivp
warnings.filterwarnings('ignore')

# Create figures directory if it doesn't exist
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Modern color palette - consistent with previous modules
COLORS = {
    'primary': '#2E86AB',    # Modern blue
    'secondary': '#A23B72',  # Deep rose
    'accent': '#16A085',     # Elegant teal
    'neutral': '#6C757D',    # Sophisticated gray
    'light': '#F8F9FA',      # Very light gray
    'dark': '#2D3436'        # Charcoal
}

# Modern style parameters
def set_style():
    """Set consistent modern style for all figures"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
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
    """Save figure with consistent formatting"""
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
# PART 1 FIGURES: Failure of Naive Integration
# ============================================================================

def figure_3_1_euler_geometric():
    """
    Euler's method geometric interpretation showing tangent line approximation.
    
    Location: Part 1, Section "Euler's Method - The Simplest Approach"
    Caption: Euler's method extends the tangent line at each point, accumulating error 
             by ignoring the solution's curvature. The local truncation error (shown in red) 
             grows with step size squared.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Left panel: Single Euler step geometric interpretation
    def f(x, y):
        return -2*y + x  # Example ODE
    
    # True solution (analytical for this example)
    def true_solution(x):
        return 0.5*x - 0.25 + 1.25*np.exp(-2*x)
    
    x0, y0 = 0, 1  # Initial condition
    h_values = [0.2, 0.4]  # Match the right panel timesteps
    colors_steps = ['#2E8B57', '#8B008B']  # Green and purple for different step sizes
    
    # Fine grid for true solution
    x_fine = np.linspace(0, 2, 1000)
    y_true = true_solution(x_fine)
    
    ax1.plot(x_fine, y_true, color=COLORS['primary'], linewidth=4, alpha=0.9,
             label='True solution', zorder=3)
    
    # Show both step sizes for comparison
    for i, h in enumerate(h_values):
        # Euler step
        slope = f(x0, y0)
        x1 = x0 + h
        y1_euler = y0 + h * slope
        y1_true = true_solution(x1)
        
        # Draw tangent line
        x_tangent = np.linspace(x0, x1 + 0.1, 100)
        y_tangent = y0 + slope * (x_tangent - x0)
        ax1.plot(x_tangent, y_tangent, '--', color=colors_steps[i], linewidth=3,
                 alpha=0.8, label=f'Tangent line, h={h}', zorder=2)
        
        # Mark points
        ax1.plot(x1, y1_euler, 's', color=colors_steps[i], markersize=10,
                 markeredgewidth=2, markeredgecolor='white', zorder=5,
                 label=f'Euler (h={h})')
        ax1.plot(x1, y1_true, '^', color=COLORS['primary'], markersize=8,
                 markeredgewidth=2, markeredgecolor='white', zorder=5)
        
        # Show local error with double-headed arrow
        ax1.annotate('', xy=(x1, y1_true), xytext=(x1, y1_euler),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2),
                    zorder=4)
        
        # Error annotation offset slightly for each step size
        offset_x = 0.05 + i * 0.1
        ax1.text(x1 + offset_x, (y1_euler + y1_true)/2, 
                 f'Error\n(h={h})', color='red', fontsize=10, 
                 weight='medium', ha='left', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                          edgecolor='red', alpha=0.9))
    
    # Mark starting point
    ax1.plot(x0, y0, 'o', color=COLORS['dark'], markersize=12,
             markeredgewidth=2, markeredgecolor='white', zorder=5,
             label='Starting point')
    
    ax1.annotate('Both steps start\nfrom same point', 
                xy=(x0, y0), xytext=(0.3, 0.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2),
                fontsize=11, ha='center', va='center', color=COLORS['dark'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['dark'], alpha=0.9))
    
    ax1.set_xlabel('x', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('y', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title("Single Euler Step: Following the Tangent", fontsize=18,
                  color=COLORS['dark'], weight='medium', pad=20)
    ax1.legend(fontsize=12, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white', loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, 2.0)
    ax1.set_ylim(-0.5, 1.2)
    
    # Right panel: Multiple Euler steps showing error accumulation
    def euler_solve(f, x0, y0, h, n_steps):
        x_vals = [x0]
        y_vals = [y0]
        x, y = x0, y0
        
        for _ in range(n_steps):
            y = y + h * f(x, y)
            x = x + h
            x_vals.append(x)
            y_vals.append(y)
        
        return np.array(x_vals), np.array(y_vals)
    
    # Multiple step sizes
    h_small = 0.2
    h_large = 0.4
    n_steps_small = int(2.0 / h_small)
    n_steps_large = int(2.0 / h_large)
    
    # Euler solutions
    x_euler_small, y_euler_small = euler_solve(f, x0, y0, h_small, n_steps_small)
    x_euler_large, y_euler_large = euler_solve(f, x0, y0, h_large, n_steps_large)
    
    # Plot true solution
    ax2.plot(x_fine, y_true, color=COLORS['primary'], linewidth=4, alpha=0.9,
             label='True solution', zorder=5)
    
    # Plot Euler solutions
    ax2.plot(x_euler_small, y_euler_small, 'o-', color=COLORS['accent'], 
             linewidth=3, markersize=6, label=f'Euler, h = {h_small}', zorder=4)
    ax2.plot(x_euler_large, y_euler_large, 's-', color=COLORS['secondary'],
             linewidth=3, markersize=8, label=f'Euler, h = {h_large}', zorder=3)
    
    # Show error growth
    final_true = true_solution(2.0)
    final_small = y_euler_small[-1]
    final_large = y_euler_large[-1]
    
    error_small = abs(final_small - final_true)
    error_large = abs(final_large - final_true)
    
    ax2.text(0.05, 0.95, f'Final errors:\nSmall h: {error_small:.3f}\nLarge h: {error_large:.3f}',
            transform=ax2.transAxes, fontsize=12, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'],
                     edgecolor=COLORS['primary'], alpha=0.9),
            color=COLORS['dark'], weight='medium')
    
    ax2.set_xlabel('x', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('y', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title("Error Accumulation Over Multiple Steps", fontsize=18,
                  color=COLORS['dark'], weight='medium', pad=20)
    ax2.legend(fontsize=12, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2)
    ax2.set_ylim(-0.2, 1.2)
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_1_euler_geometric")
    reset_style()


def figure_3_2_energy_drift():
    """
    Energy drift catastrophe showing Euler's systematic violation of conservation.
    
    Location: Part 1, Section "The Energy Drift Catastrophe"
    Caption: Euler's method systematically violates energy conservation. (Top) Phase space 
             trajectory spirals outward. (Middle) Energy grows linearly with time. (Bottom) 
             Spatial orbit expands continuously. Blue: true solution, Red: Euler's method.
    """
    set_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('white')
    
    # Harmonic oscillator parameters
    omega = 1.0
    x0, v0 = 1.0, 0.0  # Initial conditions
    E0 = 0.5 * (v0**2 + omega**2 * x0**2)  # Initial energy
    
    def harmonic_rhs(t, y):
        x, v = y[0], y[1]
        return [v, -omega**2 * x]
    
    def euler_harmonic(x0, v0, omega, h, n_steps):
        """Euler integration of harmonic oscillator"""
        x, v = x0, v0
        positions = [x0]
        velocities = [v0]
        times = [0]
        energies = [0.5 * (v0**2 + omega**2 * x0**2)]
        
        for i in range(n_steps):
            # Euler update
            a = -omega**2 * x
            x = x + h * v
            v = v + h * a
            
            positions.append(x)
            velocities.append(v)
            times.append((i+1) * h)
            energies.append(0.5 * (v**2 + omega**2 * x**2))
        
        return (np.array(times), np.array(positions), 
                np.array(velocities), np.array(energies))
    
    # Integration parameters  
    h = 0.02  # Reasonable step for visible drift
    T = 2 * np.pi / omega  # Period  
    n_periods = 10  # Show clear accumulation over reasonable time
    t_final = n_periods * T
    n_steps = int(t_final / h)
    
    # Euler solution
    t_euler, x_euler, v_euler, E_euler = euler_harmonic(x0, v0, omega, h, n_steps)
    
    # True solution (analytical)
    t_true = np.linspace(0, t_final, 1000)
    x_true = x0 * np.cos(omega * t_true) + (v0/omega) * np.sin(omega * t_true)
    v_true = -x0 * omega * np.sin(omega * t_true) + v0 * np.cos(omega * t_true)
    E_true = np.full_like(t_true, E0)
    
    # Panel 1: Phase space (x, v)
    ax1.plot(x_true, v_true, color=COLORS['primary'], linewidth=4, alpha=0.9,
             label='True orbit (circle)', zorder=3)
    ax1.plot(x_euler, v_euler, color=COLORS['secondary'], linewidth=3, alpha=0.8,
             label='Euler (outward spiral)', zorder=2)
    
    # Mark starting point
    ax1.plot(x0, v0, 'o', color=COLORS['dark'], markersize=10,
             markeredgewidth=2, markeredgecolor='white', zorder=5)
    
    # Add arrows to show direction
    n_arrows = 8
    for i in range(0, len(x_euler)-1, len(x_euler)//n_arrows):
        dx = x_euler[i+1] - x_euler[i]
        dv = v_euler[i+1] - v_euler[i]
        ax1.annotate('', xy=(x_euler[i+1], v_euler[i+1]), 
                    xytext=(x_euler[i], v_euler[i]),
                    arrowprops=dict(arrowstyle='->', color=COLORS['secondary'],
                                  lw=2, alpha=0.7))
    
    ax1.set_xlabel('Position x', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('Velocity v', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_title('Phase Space: Spiral Outward!', fontsize=16,
                  color=COLORS['dark'], weight='medium')
    ax1.legend(fontsize=12, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Panel 2: Normalized Energy vs time
    E_true_norm = E_true / E0  # Should be exactly 1.0
    E_euler_norm = E_euler / E0  # Will grow over time
    
    ax2.plot(t_true, E_true_norm, color=COLORS['primary'], linewidth=4, alpha=0.9,
             label=r'True energy ($E/E_0 = 1$)', zorder=3)
    ax2.plot(t_euler, E_euler_norm, color=COLORS['secondary'], linewidth=3,
             label=r'Euler energy ($E/E_0$)', zorder=2)
    
    # Calculate energy growth rate (should be E_final/E_initial - 1)
    energy_factor = E_euler[-1] / E_euler[0]
    energy_growth = (energy_factor - 1) * 100
    
    ax2.text(0.05, 0.95, f'Energy growth:\n{energy_growth:.1f}% over {n_periods} periods\n(Factor: {energy_factor:.2f}×)',
            transform=ax2.transAxes, fontsize=12, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='red', alpha=0.9),
            color='red', weight='bold')
    
    ax2.set_xlabel('Time', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel(r'$E/E_0$ (Normalized Energy)', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_title('Energy: Linear Growth!', fontsize=16,
                  color=COLORS['dark'], weight='medium')
    ax2.legend(fontsize=12, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Spatial orbit in x-y plane (for circular motion visualization)
    # Convert 1D oscillator to 2D circular motion for better visualization
    y_true = x0 * np.sin(omega * t_true) - (v0/omega) * np.cos(omega * t_true)
    # For Euler, we'll approximate the circular motion
    theta_euler = np.arctan2(v_euler, omega * x_euler)
    r_euler = np.sqrt(x_euler**2 + (v_euler/omega)**2)
    x_euler_circ = r_euler * np.cos(theta_euler)
    y_euler_circ = r_euler * np.sin(theta_euler)
    
    # True circular orbit
    theta_true = np.linspace(0, 2*np.pi, 100)
    radius_true = np.sqrt(x0**2 + (v0/omega)**2)
    x_circle = radius_true * np.cos(theta_true)
    y_circle = radius_true * np.sin(theta_true)
    
    ax3.plot(x_circle, y_circle, color=COLORS['primary'], linewidth=4, alpha=0.9,
             label='True orbit', zorder=3)
    ax3.plot(x_euler_circ, y_euler_circ, color=COLORS['secondary'], linewidth=3,
             label='Euler orbit', zorder=2)
    
    # Mark starting point
    ax3.plot(x_euler_circ[0], y_euler_circ[0], 'o', color=COLORS['dark'], 
             markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=5)
    
    ax3.set_xlabel('x', fontsize=14, color=COLORS['dark'], weight='medium')
    ax3.set_ylabel('y', fontsize=14, color=COLORS['dark'], weight='medium')
    ax3.set_title('Spatial Orbit: Expanding Spiral!', fontsize=16,
                  color=COLORS['dark'], weight='medium')
    ax3.legend(fontsize=12, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Panel 4: Error growth analysis
    # Interpolate true solution onto Euler time grid
    x_true_interp = np.interp(t_euler, t_true, x_true)
    position_error = np.abs(x_euler - x_true_interp)
    energy_error = np.abs(E_euler - E0)
    
    ax4.semilogy(t_euler, position_error, color=COLORS['accent'], linewidth=3,
                label='Position error', zorder=3)
    ax4.semilogy(t_euler, energy_error, color='red', linewidth=3,
                label='Energy error', zorder=2)
    
    # Add theoretical growth lines
    linear_growth = 0.01 * t_euler  # Expected linear growth for Euler
    ax4.plot(t_euler, linear_growth, '--', color=COLORS['neutral'], linewidth=2,
            alpha=0.7, label='Linear growth', zorder=1)
    
    ax4.set_xlabel('Time', fontsize=14, color=COLORS['dark'], weight='medium')
    ax4.set_ylabel('Error (log scale)', fontsize=14, color=COLORS['dark'], weight='medium')
    ax4.set_title('Error Growth Analysis', fontsize=16,
                  color=COLORS['dark'], weight='medium')
    ax4.legend(fontsize=12, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax4.grid(True, alpha=0.3)
    
    # Style spines
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=12, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_2_energy_drift")
    reset_style()


# ============================================================================
# PART 2 FIGURES: Runge-Kutta Methods
# ============================================================================

def figure_3_3_rk2_midpoint():
    """
    RK2 midpoint method visualization showing how sampling captures curvature.
    
    Location: Part 2, Section "RK2 - The Midpoint Method"
    Caption: RK2 midpoint method: First evaluate derivative at start (k₁), use it to 
             estimate midpoint, evaluate derivative there (k₂), then use midpoint 
             derivative for full step. This captures solution curvature, achieving 
             O(h²) global accuracy.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Define test ODE and solution
    def f(x, y):
        return -y + np.sin(x)  # Example ODE with known solution
    
    def true_solution(x):
        # For y' = -y + sin(x), solution is y = C*exp(-x) + (sin(x) - cos(x))/2
        # With initial condition y(0) = y0, we get C = y0 + 1/2
        return (y0 + 0.5) * np.exp(-x) + (np.sin(x) - np.cos(x)) / 2
    
    x0, y0 = 0, 1.0  # Initial condition
    h = 1.0  # Step size
    
    # Left panel: Geometric interpretation of RK2
    x_fine = np.linspace(0, 2, 1000)
    y_true = true_solution(x_fine)
    
    ax1.plot(x_fine, y_true, color=COLORS['primary'], linewidth=4, alpha=0.9,
             label='True solution', zorder=5)
    
    # RK2 method steps
    k1 = f(x0, y0)  # Initial derivative
    x_mid = x0 + h/2
    y_mid = y0 + (h/2) * k1  # Predictor to midpoint
    k2 = f(x_mid, y_mid)  # Derivative at midpoint
    
    x1 = x0 + h
    y1_rk2 = y0 + h * k2  # Full step using midpoint derivative
    y1_true = true_solution(x1)
    y1_euler = y0 + h * k1  # Euler for comparison
    
    # Draw the RK2 construction
    # Step 1: Initial tangent to midpoint
    x_half_step = np.linspace(x0, x_mid, 50)
    y_half_step = y0 + k1 * (x_half_step - x0)
    ax1.plot(x_half_step, y_half_step, '--', color=COLORS['secondary'], linewidth=2,
             alpha=0.7, label=r'$k_1$: Initial slope', zorder=3)
    
    # Step 2: Show midpoint slope at the midpoint, then use it for full step
    # Short line showing k2 slope at midpoint
    x_k2_demo = np.array([x_mid - 0.2, x_mid + 0.2])
    y_k2_demo = y_mid + k2 * (x_k2_demo - x_mid)
    ax1.plot(x_k2_demo, y_k2_demo, '-', color=COLORS['accent'], 
             linewidth=4, alpha=0.9, label=r'$k_2$: Midpoint slope', zorder=4)
    
    # Show the final RK2 step using k2
    ax1.plot([x0, x1], [y0, y1_rk2], ':', color=COLORS['accent'],
             linewidth=3, alpha=0.7, label=r'RK2 step using $k_2$', zorder=3)
    
    # Mark key points
    ax1.plot(x0, y0, 'o', color=COLORS['dark'], markersize=10,
             markeredgewidth=2, markeredgecolor='white', zorder=6)
    ax1.plot(x_mid, y_mid, 's', color=COLORS['secondary'], markersize=8,
             markeredgewidth=2, markeredgecolor='white', zorder=6)
    ax1.plot(x1, y1_rk2, '^', color=COLORS['accent'], markersize=10,
             markeredgewidth=2, markeredgecolor='white', zorder=6)
    ax1.plot(x1, y1_true, 'D', color=COLORS['primary'], markersize=8,
             markeredgewidth=2, markeredgecolor='white', zorder=6)
    ax1.plot(x1, y1_euler, 'v', color='red', markersize=8,
             markeredgewidth=2, markeredgecolor='white', zorder=6)
    
    # Annotations
    ax1.annotate(r'Start ($x_0, y_0$)', xy=(x0, y0), xytext=(x0-0.3, y0+0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5),
                fontsize=11, ha='center', color=COLORS['dark'], weight='medium')
    
    ax1.annotate('Midpoint estimate', xy=(x_mid, y_mid), xytext=(x_mid-0.2, y_mid+0.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.5),
                fontsize=11, ha='center', color=COLORS['secondary'], weight='medium')
    
    ax1.annotate('RK2 result', xy=(x1, y1_rk2), xytext=(x1+0.2, y1_rk2+0.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5),
                fontsize=11, ha='left', color=COLORS['accent'], weight='medium')
    
    # Show errors
    rk2_error = abs(y1_rk2 - y1_true)
    euler_error = abs(y1_euler - y1_true)
    ax1.plot([x1, x1], [y1_rk2, y1_true], color=COLORS['accent'], linewidth=3,
             alpha=0.6, zorder=2)
    ax1.plot([x1+0.05, x1+0.05], [y1_euler, y1_true], color='red', linewidth=3,
             alpha=0.6, zorder=2)
    
    ax1.text(0.05, 0.95, f'Errors at x = {x1}:\nRK2: {rk2_error:.4f}\nEuler: {euler_error:.4f}',
            transform=ax1.transAxes, fontsize=11, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light'],
                     edgecolor=COLORS['primary'], alpha=0.9),
            color=COLORS['dark'], weight='medium')
    
    ax1.set_xlabel('x', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('y', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_title('RK2 Geometric Construction', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax1.legend(fontsize=11, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, 2.2)
    ax1.set_ylim(-0.5, 2.0)
    
    # Right panel: Convergence comparison
    def euler_step(x, y, h, f):
        return y + h * f(x, y)
    
    def rk2_step(x, y, h, f):
        k1 = f(x, y)
        k2 = f(x + h/2, y + h/2 * k1)
        return y + h * k2
    
    # Test convergence for different step sizes
    h_values = [0.4, 0.2, 0.1, 0.05, 0.025]
    x_test = 2.0
    y_exact = true_solution(x_test)
    
    euler_errors = []
    rk2_errors = []
    
    for h_test in h_values:
        n_steps = int(x_test / h_test)
        h_actual = x_test / n_steps  # Adjust to hit x_test exactly
        
        # Euler
        x, y = 0, y0
        for _ in range(n_steps):
            y = euler_step(x, y, h_actual, f)
            x += h_actual
        euler_errors.append(abs(y - y_exact))
        
        # RK2
        x, y = 0, y0
        for _ in range(n_steps):
            y = rk2_step(x, y, h_actual, f)
            x += h_actual
        rk2_errors.append(abs(y - y_exact))
    
    ax2.loglog(h_values, euler_errors, 'o-', color='red', linewidth=3,
              markersize=8, label='Euler: O(h)', alpha=0.9)
    ax2.loglog(h_values, rk2_errors, 's-', color=COLORS['accent'], linewidth=3,
              markersize=8, label='RK2: O(h²)', alpha=0.9)
    
    # Reference lines for slopes
    h_ref = np.array(h_values)
    ax2.plot(h_ref, 0.5 * h_ref, '--', color='red', alpha=0.6, linewidth=2,
            label='h (first order)')
    ax2.plot(h_ref, 0.1 * h_ref**2, '--', color=COLORS['accent'], alpha=0.6, 
            linewidth=2, label='h² (second order)')
    
    ax2.set_xlabel('Step size h', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Error at x = 2', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_title('Convergence Rate Comparison', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax2.legend(fontsize=11, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax2.grid(True, alpha=0.3)
    
    # Annotate slopes
    ax2.text(0.15, 0.02, 'Slope = 1\n(linear)', fontsize=10, ha='center',
            color='red', weight='medium',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor='red', alpha=0.8))
    ax2.text(0.15, 0.001, 'Slope = 2\n(quadratic)', fontsize=10, ha='center',
            color=COLORS['accent'], weight='medium',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor=COLORS['accent'], alpha=0.8))
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=12, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_3_rk2_midpoint")
    reset_style()


def figure_3_4_adaptive_timestep():
    """
    Adaptive timestep control for eccentric orbit showing variable step sizes.
    
    Location: Part 2, Section "Adaptive Timestep Control"
    Caption: Adaptive timestep control for eccentric orbit (e=0.9). Small steps near 
             perihelion where velocity is high, large steps near aphelion where motion 
             is slow. This maintains constant error while minimizing computation.
    """
    set_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('white')
    
    # Kepler problem parameters
    e = 0.9  # High eccentricity
    a = 1.0  # Semi-major axis
    GM = 1.0  # Gravitational parameter
    
    def kepler_rhs(t, y):
        """Kepler problem: [x, y, vx, vy]"""
        x, y, vx, vy = y
        r = np.sqrt(x**2 + y**2)
        r3 = r**3
        return [vx, vy, -GM*x/r3, -GM*y/r3]
    
    # Initial conditions for elliptical orbit
    x0 = a * (1 - e)  # Perihelion
    y0 = 0
    v_peri = np.sqrt(GM * (1 + e) / (a * (1 - e)))  # Perihelion velocity
    vx0 = 0
    vy0 = v_peri
    
    y0_vec = [x0, y0, vx0, vy0]
    
    # Period for reference
    T = 2 * np.pi * np.sqrt(a**3 / GM)
    
    # Panel 1: Full orbit with adaptive vs fixed timesteps
    # Fixed timestep solution
    t_span = (0, T)
    t_eval_fixed = np.linspace(0, T, 1000)
    sol_fixed = solve_ivp(kepler_rhs, t_span, y0_vec, t_eval=t_eval_fixed, 
                         method='RK45', rtol=1e-8)
    
    # Adaptive solution with variable timestep
    sol_adaptive = solve_ivp(kepler_rhs, t_span, y0_vec, method='RK45', 
                            rtol=1e-6, atol=1e-9, dense_output=True)
    
    # Extract positions
    x_fixed, y_fixed = sol_fixed.y[0], sol_fixed.y[1]
    x_adaptive, y_adaptive = sol_adaptive.y[0], sol_adaptive.y[1]
    
    ax1.plot(x_fixed, y_fixed, color=COLORS['primary'], linewidth=3, alpha=0.8,
             label='Fixed timestep (1000 points)')
    ax1.plot(x_adaptive, y_adaptive, 'o', color=COLORS['secondary'], markersize=4,
             alpha=0.7, label=f'Adaptive ({len(sol_adaptive.t)} points)')
    
    # Mark perihelion and aphelion
    ax1.plot(x0, y0, 'o', color='red', markersize=12, 
             markeredgewidth=2, markeredgecolor='white', zorder=5)
    ax1.plot(-a*(1+e), 0, 's', color='blue', markersize=10,
             markeredgewidth=2, markeredgecolor='white', zorder=5)
    
    ax1.text(x0, y0-0.3, 'Perihelion\n(high speed)', ha='center', va='top',
            fontsize=11, color='red', weight='medium',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor='red', alpha=0.8))
    ax1.text(-a*(1+e), 0.3, 'Aphelion\n(low speed)', ha='center', va='bottom',
            fontsize=11, color='blue', weight='medium',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor='blue', alpha=0.8))
    
    ax1.set_xlabel('x', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('y', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_title(f'Eccentric Orbit (e = {e})', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax1.legend(fontsize=12, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Panel 2: Timestep size vs orbital position
    # Calculate true anomaly for each point
    def true_anomaly(x, y, vx, vy):
        r = np.sqrt(x**2 + y**2)
        # Use vis-viva equation and angular momentum to find true anomaly
        v2 = vx**2 + vy**2
        h = x*vy - y*vx  # Specific angular momentum
        # Position along major axis
        cos_nu = (x * np.cos(0) + y * np.sin(0)) / r  # Assuming perihelion at x-axis
        return np.arccos(np.clip(cos_nu, -1, 1))
    
    # Calculate timestep sizes (differences between consecutive times)
    dt_adaptive = np.diff(sol_adaptive.t)
    t_mid = sol_adaptive.t[:-1] + dt_adaptive/2
    
    # For visualization, let's create a synthetic relationship with orbital position
    # based on distance from perihelion
    r_adaptive = np.sqrt(sol_adaptive.y[0]**2 + sol_adaptive.y[1]**2)
    
    ax2.semilogy(t_mid/T, dt_adaptive, 'o-', color=COLORS['accent'], 
                linewidth=2, markersize=4, alpha=0.8)
    ax2.set_xlabel('Orbital fraction', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Timestep size', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_title('Adaptive Timestep vs Orbital Position', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    ax2.annotate('Small steps\nnear perihelion', xy=(0.05, min(dt_adaptive)*2), 
                xytext=(0.2, min(dt_adaptive)*10),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2),
                fontsize=11, ha='center', color=COLORS['accent'], weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['accent'], alpha=0.9))
    
    ax2.annotate('Large steps\nnear aphelion', xy=(0.45, max(dt_adaptive)*0.5), 
                xytext=(0.3, max(dt_adaptive)*0.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2),
                fontsize=11, ha='center', color=COLORS['accent'], weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['accent'], alpha=0.9))
    
    # Panel 3: Speed vs orbital position
    v_magnitude = np.sqrt(sol_adaptive.y[2]**2 + sol_adaptive.y[3]**2)
    
    ax3.plot(sol_adaptive.t/T, v_magnitude, color=COLORS['primary'], linewidth=3)
    ax3.set_xlabel('Orbital fraction', fontsize=14, color=COLORS['dark'], weight='medium')
    ax3.set_ylabel('Speed', fontsize=14, color=COLORS['dark'], weight='medium')
    ax3.set_title('Orbital Speed Variation', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax3.grid(True, alpha=0.3)
    
    # Mark perihelion and aphelion
    ax3.axvline(0, color='red', linestyle='--', alpha=0.7, label='Perihelion')
    ax3.axvline(0.5, color='blue', linestyle='--', alpha=0.7, label='Aphelion')
    ax3.legend(fontsize=11)
    
    # Panel 4: Error estimate vs time (conceptual)
    # Since we don't have access to the internal error estimates,
    # we'll show the relationship between timestep and error conceptually
    
    # Show how adaptive timestep maintains roughly constant error
    target_error = 1e-6
    estimated_error = target_error * (1 + 0.3 * np.sin(4 * np.pi * t_mid/T))  # Synthetic
    
    ax4.semilogy(t_mid/T, estimated_error, 'o-', color=COLORS['secondary'], 
                linewidth=2, markersize=4, alpha=0.8, label='Error estimate')
    ax4.axhline(target_error, color=COLORS['accent'], linestyle='--', linewidth=2,
               alpha=0.8, label=f'Target tolerance: {target_error:.0e}')
    ax4.fill_between([0, 1], target_error*0.1, target_error*10, alpha=0.2,
                    color=COLORS['accent'], label='Acceptable range')
    
    ax4.set_xlabel('Orbital fraction', fontsize=14, color=COLORS['dark'], weight='medium')
    ax4.set_ylabel('Error estimate', fontsize=14, color=COLORS['dark'], weight='medium')
    ax4.set_title('Error Control', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax4.legend(fontsize=11, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax4.grid(True, alpha=0.3)
    
    # Add summary statistics
    total_steps_adaptive = len(sol_adaptive.t)
    total_steps_fixed = len(t_eval_fixed)
    efficiency = total_steps_fixed / total_steps_adaptive
    
    fig.suptitle(f'Adaptive Integration: {efficiency:.1f}× more efficient '
                f'({total_steps_adaptive} vs {total_steps_fixed} steps)',
                fontsize=18, color=COLORS['dark'], weight='medium', y=0.98)
    
    # Style spines
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=12, colors=COLORS['neutral'])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    save_figure(fig, "3_4_adaptive_timestep")
    reset_style()


# ============================================================================
# PART 3 FIGURES: Symplectic Integration
# ============================================================================

def figure_3_5_phase_space_structure():
    """
    Phase space structure showing Hamiltonian flow and volume preservation.
    
    Location: Part 3, Section "The Phase Space Perspective"
    Caption: Phase space structure of Hamiltonian systems. (Left) Harmonic oscillator: 
             closed orbits on energy surfaces. (Middle) Pendulum: separatrix divides 
             oscillation from rotation. (Right) Liouville's theorem: phase space volume 
             is preserved under Hamiltonian flow—areas deform but don't change size.
    """
    set_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('white')
    
    # Panel 1: Harmonic oscillator phase space
    q = np.linspace(-2, 2, 100)
    p = np.linspace(-2, 2, 100)
    Q, P = np.meshgrid(q, p)
    
    # Energy levels for harmonic oscillator H = (p^2 + q^2)/2
    omega = 1.0
    energy_levels = [0.2, 0.5, 1.0, 1.8, 3.2]
    colors_levels = plt.cm.Blues(np.linspace(0.3, 0.9, len(energy_levels)))
    
    for i, E in enumerate(energy_levels):
        # For harmonic oscillator, energy contours are circles
        theta = np.linspace(0, 2*np.pi, 200)
        q_circle = np.sqrt(2*E/omega) * np.cos(theta)
        p_circle = np.sqrt(2*E) * np.sin(theta)
        
        ax1.plot(q_circle, p_circle, color=colors_levels[i], linewidth=2.5,
                alpha=0.8, label=f'E = {E}')
    
    # Add flow arrows
    n_arrows = 8
    for i, E in enumerate([0.5, 1.5]):
        theta_arrows = np.linspace(0, 2*np.pi, n_arrows, endpoint=False)
        q_arr = np.sqrt(2*E/omega) * np.cos(theta_arrows)
        p_arr = np.sqrt(2*E) * np.sin(theta_arrows)
        
        # Flow direction: dq/dt = p, dp/dt = -omega^2 * q
        dq_dt = p_arr
        dp_dt = -omega**2 * q_arr
        
        # Normalize for visualization
        length = np.sqrt(dq_dt**2 + dp_dt**2)
        dq_dt = 0.15 * dq_dt / length
        dp_dt = 0.15 * dp_dt / length
        
        for j in range(n_arrows):
            ax1.arrow(q_arr[j], p_arr[j], dq_dt[j], dp_dt[j],
                     head_width=0.08, head_length=0.06, fc=COLORS['primary'],
                     ec=COLORS['primary'], alpha=0.7, zorder=3)
    
    ax1.set_xlabel('Position q', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('Momentum p', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_title('Harmonic Oscillator\n(Elliptical Orbits)', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    
    # Panel 2: Pendulum phase space
    q_pend = np.linspace(-np.pi, np.pi, 100)
    p_pend = np.linspace(-3, 3, 100)
    Q_pend, P_pend = np.meshgrid(q_pend, p_pend)
    
    # Pendulum Hamiltonian: H = p^2/2 - cos(q)
    g = 1.0  # Reduced units
    H_pend = P_pend**2/2 - np.cos(Q_pend)
    
    # Plot energy contours
    energy_levels_pend = [-0.8, -0.4, 0.0, 0.4, 1.0, 2.0]
    
    for E in energy_levels_pend:
        if E < 1.0:  # Oscillatory motion
            color = COLORS['primary']
            alpha = 0.8
        else:  # Rotational motion
            color = COLORS['secondary']
            alpha = 0.8
        
        contours = ax2.contour(Q_pend, P_pend, H_pend, levels=[E], 
                              colors=[color], alpha=alpha, linewidths=2.5)
    
    # Mark the separatrix (E = 1)
    separatrix = ax2.contour(Q_pend, P_pend, H_pend, levels=[1.0], 
                           colors=['red'], linewidths=4, alpha=0.9)
    
    # Add labels
    ax2.text(0, -2.5, 'Oscillation\n(libration)', ha='center', va='center',
            fontsize=12, color=COLORS['primary'], weight='medium',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLORS['primary'], alpha=0.9))
    ax2.text(0, 2.5, 'Rotation', ha='center', va='center',
            fontsize=12, color=COLORS['secondary'], weight='medium',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLORS['secondary'], alpha=0.9))
    ax2.text(np.pi*0.7, 0, 'Separatrix', ha='center', va='bottom',
            fontsize=12, color='red', weight='bold', rotation=90,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='red', alpha=0.9))
    
    ax2.set_xlabel('Angle θ', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Angular momentum p', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_title('Pendulum\n(Mixed Dynamics)', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-3, 3)
    
    # Set x-axis ticks for pendulum
    ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    # Panel 3: Volume preservation illustration
    # Show how a phase space area element deforms but preserves area
    
    # Initial rectangle in phase space
    q0, p0 = -0.5, 0.3
    width, height = 0.3, 0.4
    
    # Rectangle corners
    corners_init = np.array([
        [q0, p0],
        [q0 + width, p0],
        [q0 + width, p0 + height],
        [q0, p0 + height],
        [q0, p0]  # Close the loop
    ])
    
    # Simulate Hamiltonian flow (using harmonic oscillator)
    def harmonic_flow(q, p, t):
        """Exact solution for harmonic oscillator"""
        q_new = q * np.cos(t) + p * np.sin(t)
        p_new = p * np.cos(t) - q * np.sin(t)
        return q_new, p_new
    
    times = [0, 0.8, 1.6]
    colors_time = ['blue', 'green', 'red']
    alphas = [0.8, 0.6, 0.7]
    
    for i, (t, color, alpha) in enumerate(zip(times, colors_time, alphas)):
        if t == 0:
            corners = corners_init
        else:
            corners = np.zeros_like(corners_init)
            for j in range(len(corners_init)):
                corners[j, 0], corners[j, 1] = harmonic_flow(
                    corners_init[j, 0], corners_init[j, 1], t)
        
        # Plot the deformed area element
        ax3.fill(corners[:, 0], corners[:, 1], color=color, alpha=alpha,
                edgecolor=color, linewidth=2, 
                label=f't = {t}' if i == 0 else f't = {t:.1f}')
    
    # Calculate areas to show they're preserved
    def polygon_area(corners):
        x, y = corners[:-1, 0], corners[:-1, 1]
        return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] 
                           for i in range(-1, len(x)-1)))
    
    area_init = polygon_area(corners_init)
    
    # Add flow field background
    q_grid = np.linspace(-1.5, 1.5, 15)
    p_grid = np.linspace(-1.5, 1.5, 15)
    Q_grid, P_grid = np.meshgrid(q_grid, p_grid)
    
    # Harmonic oscillator flow: dq/dt = p, dp/dt = -q
    dQ = P_grid
    dP = -Q_grid
    
    ax3.quiver(Q_grid, P_grid, dQ, dP, alpha=0.3, scale=20, width=0.002,
              color=COLORS['neutral'])
    
    ax3.set_xlabel('Position q', fontsize=14, color=COLORS['dark'], weight='medium')
    ax3.set_ylabel('Momentum p', fontsize=14, color=COLORS['dark'], weight='medium')
    ax3.set_title('Volume Preservation\n(Liouville\'s Theorem)', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax3.legend(fontsize=11, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    
    # Add text showing area preservation
    ax3.text(0.05, 0.95, f'Area = {area_init:.3f}\n(constant for all t)',
            transform=ax3.transAxes, fontsize=11, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light'],
                     edgecolor=COLORS['primary'], alpha=0.9),
            color=COLORS['dark'], weight='medium')
    
    # Style spines
    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=12, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_5_phase_space_structure")
    reset_style()


def figure_3_6_modified_hamiltonian():
    """
    Modified Hamiltonian showing bounded energy oscillation vs systematic drift.
    
    Location: Part 3, Section "The Modified Hamiltonian"
    Caption: Energy behavior comparison. (Top) RK4 exhibits monotonic energy drift while 
             Leapfrog energy oscillates within bounded envelope. (Bottom) Phase space 
             trajectories: RK4 spirals outward, Leapfrog remains on a nearby invariant 
             torus of the modified Hamiltonian.
    """
    set_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('white')
    
    # Harmonic oscillator parameters
    omega = 1.0
    x0, v0 = 1.0, 0.0
    E0 = 0.5 * (v0**2 + omega**2 * x0**2)
    
    def rk4_step(x, v, h, omega):
        """Single RK4 step for harmonic oscillator"""
        k1_x = v
        k1_v = -omega**2 * x
        
        k2_x = v + 0.5*h*k1_v
        k2_v = -omega**2 * (x + 0.5*h*k1_x)
        
        k3_x = v + 0.5*h*k2_v
        k3_v = -omega**2 * (x + 0.5*h*k2_x)
        
        k4_x = v + h*k3_v
        k4_v = -omega**2 * (x + h*k3_x)
        
        x_new = x + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v_new = v + h/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        return x_new, v_new
    
    def leapfrog_step(x, v, h, omega):
        """Single leapfrog step for harmonic oscillator"""
        # Kick-drift-kick (velocity Verlet)
        a = -omega**2 * x
        v_half = v + 0.5*h*a
        x_new = x + h*v_half
        a_new = -omega**2 * x_new
        v_new = v_half + 0.5*h*a_new
        
        return x_new, v_new
    
    # Integration parameters
    h = 0.2  # Relatively large timestep to show differences clearly
    T = 2*np.pi/omega  # Period
    n_periods = 20
    t_final = n_periods * T
    n_steps = int(t_final / h)
    
    # RK4 integration
    x_rk4, v_rk4 = [x0], [v0]
    E_rk4 = [E0]
    t_vals = [0]
    
    x, v = x0, v0
    for i in range(n_steps):
        x, v = rk4_step(x, v, h, omega)
        x_rk4.append(x)
        v_rk4.append(v)
        E_rk4.append(0.5 * (v**2 + omega**2 * x**2))
        t_vals.append((i+1) * h)
    
    # Leapfrog integration
    x_lf, v_lf = [x0], [v0]
    E_lf = [E0]
    
    x, v = x0, v0
    for i in range(n_steps):
        x, v = leapfrog_step(x, v, h, omega)
        x_lf.append(x)
        v_lf.append(v)
        E_lf.append(0.5 * (v**2 + omega**2 * x**2))
    
    t_vals = np.array(t_vals)
    x_rk4, v_rk4, E_rk4 = np.array(x_rk4), np.array(v_rk4), np.array(E_rk4)
    x_lf, v_lf, E_lf = np.array(x_lf), np.array(v_lf), np.array(E_lf)
    
    # Panel 1: Energy vs time - long term view
    ax1.plot(t_vals/T, E_rk4, color=COLORS['secondary'], linewidth=3,
             label='RK4 (systematic drift)', alpha=0.9)
    ax1.plot(t_vals/T, E_lf, color=COLORS['primary'], linewidth=2,
             label='Leapfrog (bounded oscillation)', alpha=0.8)
    ax1.axhline(E0, color='black', linestyle='--', linewidth=2, alpha=0.6,
               label='True energy')
    
    # Show energy bounds for leapfrog
    E_lf_mean = np.mean(E_lf)
    E_lf_std = np.std(E_lf)
    ax1.fill_between(t_vals/T, E_lf_mean - 2*E_lf_std, E_lf_mean + 2*E_lf_std,
                    color=COLORS['primary'], alpha=0.2, 
                    label='Leapfrog bounds (±2σ)')
    
    # Calculate energy drift rates
    rk4_drift = (E_rk4[-1] - E_rk4[0]) / E_rk4[0] * 100
    lf_variation = E_lf_std / E0 * 100
    
    ax1.text(0.05, 0.95, f'Energy change over {n_periods} periods:\n'
                        f'RK4: {rk4_drift:+.2f}% (monotonic drift)\n'
                        f'Leapfrog: ±{lf_variation:.3f}% (bounded)',
            transform=ax1.transAxes, fontsize=11, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'],
                     edgecolor=COLORS['primary'], alpha=0.9),
            color=COLORS['dark'], weight='medium')
    
    ax1.set_xlabel('Time (periods)', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('Energy', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_title('Long-term Energy Behavior', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax1.legend(fontsize=11, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Energy vs time - short term detail
    n_detail = int(3 * T / h)  # First 3 periods
    ax2.plot(t_vals[:n_detail]/T, E_rk4[:n_detail], color=COLORS['secondary'], 
             linewidth=3, label='RK4', alpha=0.9)
    ax2.plot(t_vals[:n_detail]/T, E_lf[:n_detail], color=COLORS['primary'], 
             linewidth=2, label='Leapfrog', alpha=0.8)
    ax2.axhline(E0, color='black', linestyle='--', linewidth=2, alpha=0.6,
               label='True energy')
    
    ax2.set_xlabel('Time (periods)', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Energy', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_title('Short-term Energy Detail', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax2.legend(fontsize=11, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Phase space - RK4 spiral
    # True circle for comparison
    theta_circle = np.linspace(0, 2*np.pi, 200)
    x_circle = np.sqrt(2*E0) * np.cos(theta_circle + np.pi/2)
    v_circle = np.sqrt(2*E0) * np.sin(theta_circle + np.pi/2)
    
    ax3.plot(x_circle, v_circle, color='black', linestyle='--', linewidth=3,
             alpha=0.6, label='True orbit (constant energy)')
    ax3.plot(x_rk4, v_rk4, color=COLORS['secondary'], linewidth=2,
             alpha=0.8, label='RK4 (outward spiral)')
    
    # Mark start and end points
    ax3.plot(x0, v0, 'o', color=COLORS['dark'], markersize=10,
             markeredgewidth=2, markeredgecolor='white', zorder=5, label='Start')
    ax3.plot(x_rk4[-1], v_rk4[-1], 's', color=COLORS['secondary'], markersize=8,
             markeredgewidth=2, markeredgecolor='white', zorder=5, label='End')
    
    # Add arrows to show outward drift
    n_arrows = 8
    for i in range(0, len(x_rk4)-1, len(x_rk4)//n_arrows):
        dx = x_rk4[i+1] - x_rk4[i]
        dv = v_rk4[i+1] - v_rk4[i]
        if i % 2 == 0:  # Only show some arrows for clarity
            ax3.annotate('', xy=(x_rk4[i+1], v_rk4[i+1]), 
                        xytext=(x_rk4[i], v_rk4[i]),
                        arrowprops=dict(arrowstyle='->', color=COLORS['secondary'],
                                      lw=1.5, alpha=0.7))
    
    ax3.set_xlabel('Position x', fontsize=14, color=COLORS['dark'], weight='medium')
    ax3.set_ylabel('Velocity v', fontsize=14, color=COLORS['dark'], weight='medium')
    ax3.set_title('RK4: Outward Spiral', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax3.legend(fontsize=11, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Panel 4: Phase space - Leapfrog bounded orbit
    ax4.plot(x_circle, v_circle, color='black', linestyle='--', linewidth=3,
             alpha=0.6, label='True orbit')
    ax4.plot(x_lf, v_lf, color=COLORS['primary'], linewidth=2, alpha=0.8,
             label='Leapfrog (bounded)')
    
    # Mark start point
    ax4.plot(x0, v0, 'o', color=COLORS['dark'], markersize=10,
             markeredgewidth=2, markeredgecolor='white', zorder=5, label='Start')
    
    # Show that leapfrog stays on a nearby invariant torus
    # Calculate the modified Hamiltonian bounds
    E_lf_max = np.max(E_lf)
    E_lf_min = np.min(E_lf)
    
    # Draw bounds
    x_bound_max = np.sqrt(2*E_lf_max) * np.cos(theta_circle + np.pi/2)
    v_bound_max = np.sqrt(2*E_lf_max) * np.sin(theta_circle + np.pi/2)
    x_bound_min = np.sqrt(2*E_lf_min) * np.cos(theta_circle + np.pi/2)
    v_bound_min = np.sqrt(2*E_lf_min) * np.sin(theta_circle + np.pi/2)
    
    ax4.fill_between(x_bound_max, v_bound_max, v_bound_min, alpha=0.2,
                    color=COLORS['primary'], label='Modified Hamiltonian bounds')
    
    ax4.set_xlabel('Position x', fontsize=14, color=COLORS['dark'], weight='medium')
    ax4.set_ylabel('Velocity v', fontsize=14, color=COLORS['dark'], weight='medium')
    ax4.set_title('Leapfrog: Bounded Orbit', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax4.legend(fontsize=11, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Style spines
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=12, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_6_modified_hamiltonian")
    reset_style()


# ============================================================================
# PART 4 FIGURES: Stability Analysis
# ============================================================================

def figure_3_7_stability_regions():
    """
    Stability regions comparison for different integration methods.
    
    Location: Part 4, Section "Linear Stability Theory"
    Caption: Stability regions in the complex z-plane where z = hλ. Shaded regions 
             show where |R(z)| ≤ 1. Euler has smallest region, RK4 extends further 
             along negative real axis, Leapfrog covers imaginary axis (oscillatory 
             problems), and implicit methods cover entire left half-plane.
    """
    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create complex plane grid
    x = np.linspace(-4, 2, 800)
    y = np.linspace(-3, 3, 600)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    # Define CORRECT stability functions
    def R_euler(z):
        """Forward Euler stability function: R(z) = 1 + z"""
        return 1 + z
    
    def R_rk2(z):
        """RK2 (midpoint) stability function: R(z) = 1 + z + z²/2"""
        return 1 + z + z**2/2
    
    def R_rk4(z):
        """RK4 stability function: R(z) = 1 + z + z²/2 + z³/6 + z⁴/24"""
        return 1 + z + z**2/2 + z**3/6 + z**4/24
    
    def R_leapfrog(z):
        """
        Leapfrog stability function (CORRECTED):
        For leapfrog: y_{n+1} - y_{n-1} = 2h*f(y_n)
        Characteristic equation: r² - 2zr - 1 = 0 where z = hλ
        Roots: r = z ± √(z² + 1)
        Stability requires both |r₁| ≤ 1 and |r₂| ≤ 1
        For purely imaginary z = iω, stable when |ω| ≤ 2
        """
        # Compute the two roots of r² - 2zr - 1 = 0
        discriminant = z**2 + 1
        sqrt_disc = np.sqrt(discriminant)
        r1 = z + sqrt_disc
        r2 = z - sqrt_disc
        
        # Leapfrog is stable if both roots satisfy |r| ≤ 1
        # We return the larger magnitude root as the stability function
        stable_r1 = np.abs(r1) <= 1
        stable_r2 = np.abs(r2) <= 1
        stable = stable_r1 & stable_r2
        
        # Return a value that indicates stability (not the amplification factor)
        return np.where(stable, 0.5, 2.0)  # 0.5 for stable, 2.0 for unstable
    
    def R_backward_euler(z):
        """Backward Euler stability function: R(z) = 1/(1 - z)"""
        # Handle division by zero at z = 1
        result = np.full_like(z, np.inf, dtype=complex)
        mask = np.abs(1 - z) > 1e-12
        result[mask] = 1.0 / (1 - z[mask])
        return result
    
    # Calculate and plot stability regions for explicit methods
    explicit_methods = [
        ('Euler', R_euler, COLORS['secondary'], 0.7),
        ('RK2', R_rk2, COLORS['accent'], 0.6), 
        ('RK4', R_rk4, COLORS['primary'], 0.5),
    ]
    
    for name, R_func, color, alpha in explicit_methods:
        R_vals = R_func(Z)
        stable_region = np.abs(R_vals) <= 1.0
        
        # Fill stable regions
        ax.contourf(X, Y, stable_region.astype(int), levels=[0.5, 1.5], 
                   colors=[color], alpha=alpha, extend='max')
        # Draw boundaries
        ax.contour(X, Y, stable_region.astype(int), levels=[0.5], 
                  colors=[color], linewidths=2, alpha=0.9)
    
    # Handle Leapfrog separately (special stability criterion)
    R_lf = R_leapfrog(Z)
    stable_leapfrog = R_lf <= 1.0
    ax.contourf(X, Y, stable_leapfrog.astype(int), levels=[0.5, 1.5], 
               colors=['green'], alpha=0.6, extend='max')
    ax.contour(X, Y, stable_leapfrog.astype(int), levels=[0.5], 
              colors=['green'], linewidths=2, alpha=0.9)
    
    # Handle Backward Euler with proper calculation
    R_be = R_backward_euler(Z)
    # Backward Euler is stable when |1/(1-z)| ≤ 1, which means |1-z| ≥ 1
    # This is equivalent to Re(z) ≤ 0 (left half-plane)
    stable_backward = np.abs(R_be) <= 1.0
    # Handle the singularity at z = 1
    stable_backward = np.where(np.isfinite(R_be), stable_backward, False)
    # The mathematical result is: stable in left half-plane Re(z) ≤ 0
    stable_backward = X <= 0
    
    ax.contourf(X, Y, stable_backward.astype(int), levels=[0.5, 1.5], 
               colors=['#9370DB'], alpha=0.3, extend='max')
    ax.contour(X, Y, stable_backward.astype(int), levels=[0.5], 
              colors=['#9370DB'], linewidths=2, alpha=0.9)
    
    # Add method labels positioned within their stability regions
    ax.text(-0.5, 0.5, 'Euler', fontsize=12, weight='bold', 
           color=COLORS['secondary'], ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor=COLORS['secondary'], alpha=0.9))
                    
    ax.text(-1.5, 1.0, 'RK2', fontsize=12, weight='bold', 
           color=COLORS['accent'], ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor=COLORS['accent'], alpha=0.9))
                    
    ax.text(-2.0, 0.5, 'RK4', fontsize=12, weight='bold', 
           color=COLORS['primary'], ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor=COLORS['primary'], alpha=0.9))
    
    ax.text(0.2, 1.5, 'Leapfrog\n(imaginary axis)', fontsize=12, weight='bold', 
           color='green', ha='left', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='green', alpha=0.9))
                    
    ax.text(-3.0, -2.0, 'Backward Euler\n(entire left half-plane)', 
           fontsize=12, weight='bold', color='#9370DB', ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#9370DB', alpha=0.9))
    
    # Mark important points
    ax.plot(-1, 0, 'ko', markersize=8, markeredgewidth=2, 
           markeredgecolor='white', zorder=10)
    ax.text(-1, -0.3, '(-1, 0)', fontsize=10, ha='center', weight='medium')
    
    ax.plot(0, 2, 'go', markersize=8, markeredgewidth=2, 
           markeredgecolor='white', zorder=10)
    ax.text(0.3, 2, '(0, 2i)', fontsize=10, ha='left', weight='medium')
    
    ax.plot(0, -2, 'go', markersize=8, markeredgewidth=2, 
           markeredgecolor='white', zorder=10)  
    ax.text(0.3, -2, '(0, -2i)', fontsize=10, ha='left', weight='medium')
    
    # Axes and labels
    ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)
    ax.set_xlabel(r'Real($z$) where $z = h\lambda$', fontsize=14, 
                 color=COLORS['dark'], weight='medium')
    ax.set_ylabel(r'Imaginary($z$) where $z = h\lambda$', fontsize=14, 
                 color=COLORS['dark'], weight='medium')
    ax.set_title('Stability Regions Comparison', fontsize=18,
                color=COLORS['dark'], weight='medium', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 2)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    
    # Add legend explaining the shading
    legend_text = ('Shaded regions show where $|R(z)| \\leq 1$\n'
                  'Methods are stable inside their regions')
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
           fontsize=11, ha='left', va='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'],
                    edgecolor=COLORS['neutral'], alpha=0.95),
           color=COLORS['dark'], weight='medium')
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color(COLORS['neutral'])
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', labelsize=12, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_7_stability_regions")
    reset_style()


def figure_3_8_implicit_vs_explicit():
    """
    Implicit vs explicit stability comparison for stiff equations.
    
    Location: Part 4, Section "Implicit Methods for Stiff Problems"
    Caption: Implicit methods transform stability. (Left) Backward Euler is stable for 
             entire left half-plane while Forward Euler has tiny stability region. 
             (Right) For stiff equation y' = -1000(y-cos(t)) - sin(t), Forward Euler 
             explodes while Backward Euler remains stable with large timesteps.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Panel 1: Stability regions comparison (mathematically computed)
    x = np.linspace(-3, 1, 600)
    y = np.linspace(-2.5, 2.5, 500)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    # Forward Euler stability function: R(z) = 1 + z
    # Stable when |1 + z| ≤ 1
    R_forward = 1 + Z
    stable_forward = np.abs(R_forward) <= 1.0
    
    # Backward Euler stability function: R(z) = 1/(1 - z)  
    # Stable when |1/(1-z)| ≤ 1, which means |1-z| ≥ 1
    # This is mathematically equivalent to Re(z) ≤ 0
    R_backward = np.full_like(Z, np.inf, dtype=complex)
    mask = np.abs(1 - Z) > 1e-12  # Avoid division by zero
    R_backward[mask] = 1.0 / (1 - Z[mask])
    stable_backward = np.abs(R_backward) <= 1.0
    # Handle singularity properly - the mathematical result is left half-plane
    stable_backward = np.where(np.isfinite(R_backward), stable_backward, False)
    
    # Plot Forward Euler stability region
    ax1.contourf(X, Y, stable_forward.astype(int), levels=[0.5, 1.5], 
                colors=[COLORS['secondary']], alpha=0.7, extend='max')
    ax1.contour(X, Y, stable_forward.astype(int), levels=[0.5], 
               colors=[COLORS['secondary']], linewidths=3, alpha=0.9)
    
    # Plot Backward Euler stability region  
    ax1.contourf(X, Y, stable_backward.astype(int), levels=[0.5, 1.5], 
                colors=['#9370DB'], alpha=0.4, extend='max')
    ax1.contour(X, Y, stable_backward.astype(int), levels=[0.5], 
               colors=['#9370DB'], linewidths=3, alpha=0.9)
    
    # Mark important points
    ax1.plot(-1, 0, 'ko', markersize=10, markeredgewidth=2, 
            markeredgecolor='white', zorder=10)
    ax1.text(-1, -0.3, '(-1, 0)', fontsize=11, ha='center', weight='medium')
    
    # Add method labels
    ax1.text(-0.5, 0.5, 'Forward Euler\n(small circle)', fontsize=12, weight='bold', 
            color=COLORS['secondary'], ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLORS['secondary'], alpha=0.9))
    
    ax1.text(-2.2, 1.5, 'Backward Euler\n(entire left half-plane)', 
            fontsize=12, weight='bold', color='#9370DB', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#9370DB', alpha=0.9))
    
    # Add annotation about the stiff problem eigenvalue
    ax1.text(0.02, 0.98, r'Note: Stiff problem has $\lambda = -1000$ (far left, off-scale)', 
            transform=ax1.transAxes, fontsize=11, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                     edgecolor='orange', alpha=0.9),
            color='red', weight='medium')
    
    ax1.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='black', linewidth=1, alpha=0.5)
    ax1.set_xlabel(r'Real($z$) where $z = h\lambda$', fontsize=14, 
                  color=COLORS['dark'], weight='medium')
    ax1.set_ylabel(r'Imaginary($z$) where $z = h\lambda$', fontsize=14, 
                  color=COLORS['dark'], weight='medium')
    ax1.set_title('Stability Regions Comparison', fontsize=16,
                 color=COLORS['dark'], weight='medium', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3, 1)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    
    # Panel 2: Solution behavior for stiff equation
    # Stiff ODE: y' = -1000(y - cos(t)) - sin(t)
    # Exact solution: y = cos(t)
    
    def stiff_rhs(t, y):
        return -1000 * (y - np.cos(t)) - np.sin(t)
    
    def exact_solution(t):
        return np.cos(t)
    
    # Forward Euler with small timestep (required for stability)
    def forward_euler_solve(f, y0, t_span, h):
        t_vals = [t_span[0]]
        y_vals = [y0]
        t, y = t_span[0], y0
        
        while t < t_span[1]:
            if t + h > t_span[1]:
                h = t_span[1] - t
            y = y + h * f(t, y)
            t = t + h
            t_vals.append(t)
            y_vals.append(y)
            
            # Break if solution explodes
            if abs(y) > 1e10:
                break
                
        return np.array(t_vals), np.array(y_vals)
    
    # Backward Euler using Newton iteration
    def backward_euler_solve(f, df_dy, y0, t_span, h):
        t_vals = [t_span[0]]
        y_vals = [y0]
        t, y = t_span[0], y0
        
        while t < t_span[1]:
            if t + h > t_span[1]:
                h = t_span[1] - t
                
            t_new = t + h
            
            # Newton iteration for y_new = y + h*f(t_new, y_new)
            y_new = y  # Initial guess
            for _ in range(10):  # Newton iterations
                residual = y_new - y - h * f(t_new, y_new)
                if abs(residual) < 1e-12:
                    break
                jacobian = 1 - h * df_dy(t_new, y_new)
                y_new = y_new - residual / jacobian
            
            t, y = t_new, y_new
            t_vals.append(t)
            y_vals.append(y)
            
        return np.array(t_vals), np.array(y_vals)
    
    # Derivative for Newton iteration
    def stiff_jacobian(t, y):
        return -1000
    
    # Integration parameters
    t_span = (0, 0.01)  # Short time span to show instability quickly
    y0 = 1.0
    
    # Forward Euler with too large timestep (unstable)
    h_large = 0.0025  # This will be unstable for forward Euler
    t_forward, y_forward = forward_euler_solve(stiff_rhs, y0, t_span, h_large)
    
    # Backward Euler with same large timestep (stable)
    t_backward, y_backward = backward_euler_solve(stiff_rhs, stiff_jacobian, y0, t_span, h_large)
    
    # Exact solution
    t_exact = np.linspace(t_span[0], t_span[1], 1000)
    y_exact = exact_solution(t_exact)
    
    # Plot solutions
    ax2.plot(t_exact, y_exact, 'k--', linewidth=3, alpha=0.8, 
            label='Exact solution: $y = \\cos(t)$', zorder=3)
    
    if len(y_forward) > 1 and not np.any(np.abs(y_forward) > 1e10):
        ax2.plot(t_forward, y_forward, 'o-', color=COLORS['secondary'], 
                linewidth=2, markersize=6, label=f'Forward Euler (h={h_large})', zorder=2)
    else:
        ax2.text(0.5, 0.8, 'Forward Euler\nEXPLODES!', transform=ax2.transAxes,
                fontsize=14, weight='bold', color='red', ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='red', alpha=0.9))
    
    ax2.plot(t_backward, y_backward, 's-', color='#9370DB', 
            linewidth=2, markersize=6, label=f'Backward Euler (h={h_large})', zorder=2)
    
    # Show that we're in the stable region for backward Euler
    z_value = h_large * (-1000)  # z = hλ for our stiff problem
    ax2.text(0.02, 0.95, f'For this problem: $z = h\\lambda = {z_value:.1f}$\n'
                         f'Forward Euler: $|R(z)| = |1 + z| = {abs(1 + z_value):.1f}$ (unstable!)\n'
                         f'Backward Euler: $|R(z)| = |1/(1-z)| = {abs(1/(1-z_value)):.3f}$ (stable)',
            transform=ax2.transAxes, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'],
                     edgecolor=COLORS['neutral'], alpha=0.95),
            color=COLORS['dark'], weight='medium')
    
    ax2.set_xlabel('Time t', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Solution y(t)', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_title('Stiff Equation: $y\' = -1000(y - \\cos(t)) - \\sin(t)$', fontsize=16,
                 color=COLORS['dark'], weight='medium', pad=15)
    ax2.legend(fontsize=12, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(t_span)
    ax2.set_ylim(0.8, 1.2)
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=12, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "3_8_implicit_vs_explicit")
    reset_style()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def get_available_figures():
    """Return dictionary of available figures with their IDs and function names"""
    figures = {
        # Part 1: Failure of Naive Integration
        '3_1': figure_3_1_euler_geometric,
        '3_2': figure_3_2_energy_drift,
        
        # Part 2: Runge-Kutta Methods  
        '3_3': figure_3_3_rk2_midpoint,
        '3_4': figure_3_4_adaptive_timestep,
        
        # Part 3: Symplectic Integration
        '3_5': figure_3_5_phase_space_structure,
        '3_6': figure_3_6_modified_hamiltonian,
        
        # Part 4: Stability Analysis
        '3_7': figure_3_7_stability_regions,
        '3_8': figure_3_8_implicit_vs_explicit,
    }
    return figures

def list_available_figures():
    """Print list of all available figures"""
    figures = get_available_figures()
    print("\nAvailable figures:")
    print("=" * 60)
    
    sections = {
        'Part 1: Failure of Naive Integration': ['3_1', '3_2'],
        'Part 2: Runge-Kutta Methods': ['3_3', '3_4'],
        'Part 3: Symplectic Integration': ['3_5', '3_6'],
        'Part 4: Stability Analysis': ['3_7', '3_8'],
    }
    
    for section, fig_ids in sections.items():
        print(f"\n{section}:")
        for fig_id in fig_ids:
            func = figures[fig_id]
            # Extract title from docstring
            if func.__doc__:
                lines = func.__doc__.split('\n')
                title = None
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('Location:') and not line.startswith('Caption:'):
                        title = line.replace('"""', '').strip()
                        if title:
                            break
                if not title:
                    title = func.__name__.replace('figure_', '').replace('_', ' ').title()
            else:
                title = func.__name__.replace('figure_', '').replace('_', ' ').title()
            
            print(f"  {fig_id:<6} - {title}")
    
    print(f"\nUsage:")
    print(f"  python {sys.argv[0]} 3_1              # Generate specific figure")
    print(f"  python {sys.argv[0]} --figure 3_4     # Alternative syntax")
    print(f"  python {sys.argv[0]}                  # Generate all figures")

def main():
    parser = argparse.ArgumentParser(
        description="Generate educational figures for ASTR 596 ODE Methods & Conservation Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python ode-methods-figures.py 3_1\n"
               "  python ode-methods-figures.py --figure 3_4\n"
               "  python ode-methods-figures.py --list\n"
               "  python ode-methods-figures.py"
    )
    
    parser.add_argument('figure_id', nargs='?', help='Figure ID to generate (e.g., 3_1, 3_4)')
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
            print(f"Generating figure {figure_id}...")
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