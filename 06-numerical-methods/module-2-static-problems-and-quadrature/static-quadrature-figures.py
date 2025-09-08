#!/usr/bin/env python3
"""
Educational Figures for Module 2: Static Problems & Quadrature
ASTR 596: Modeling the Universe

This script generates all matplotlib/seaborn figures for the static problems and quadrature module.
Each function creates a specific educational visualization with publication-quality output.

Updated to match curriculum focusing on root finding, integration methods, and their applications.

Usage:
    python static-quadrature-figures.py                    # Generate all figures
    python static-quadrature-figures.py --list             # List available figures
    python static-quadrature-figures.py --figure 2_1       # Generate specific figure
    python static-quadrature-figures.py 2_1               # Generate by ID
    
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

# Modern color palette - consistent with numerical foundations module
COLORS = {
    'primary': '#2E86AB',    # Modern blue
    'secondary': '#A23B72',  # Deep rose
    'accent': '#16A085',     # Elegant teal
    'neutral': '#6C757D',    # Sophisticated gray
    'light': '#F8F9FA',      # Very light gray
    'dark': '#2D3436'        # Charcoal
}

# Modern style parameters - matching numerical foundations
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
    """Save figure with consistent formatting matching numerical module"""
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
# PART 1 FIGURES: Root Finding
# ============================================================================

def figure_2_1_root_methods_comparison():
    """
    Geometric comparison of root finding methods.
    
    Location: Part 1, Section "Building Intuition: The Geometry of Root Finding"
    Caption: Three geometric approaches to root finding: Bisection (bracketing), Newton 
             (tangent lines), and Secant (interpolation). The convergence plot shows how 
             quickly each method reduces error—note the different slopes corresponding to 
             linear, superlinear, and quadratic convergence.
    """
    set_style()
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Define test function: f(x) = x^3 - 2x - 5 with root near x = 2.1
    def f(x):
        return x**3 - 2*x - 5
    
    def fprime(x):
        return 3*x**2 - 2
    
    x_fine = np.linspace(1.5, 2.5, 1000)
    y_fine = f(x_fine)
    
    # Find true root for reference
    from scipy.optimize import brentq
    true_root = brentq(f, 1.5, 2.5)
    
    # Create 2x2 subplot layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top left: Bisection method
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('white')
    
    ax1.plot(x_fine, y_fine, color=COLORS['primary'], linewidth=3, alpha=0.9, 
             label=r'$f(x) = x^3 - 2x - 5$', zorder=3)
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax1.axvline(true_root, color='red', linestyle=':', alpha=0.7, linewidth=2, 
               label=f'True root: {true_root:.3f}')
    
    # Bisection iterations
    a, b = 1.5, 2.5
    for i in range(4):
        c = (a + b) / 2
        fc = f(c)
        
        # Draw bracket interval
        ax1.fill_between([a, b], -8, 8, alpha=0.2 - i*0.05, 
                        color=COLORS['secondary'], zorder=0)
        
        # Mark midpoint
        ax1.plot(c, fc, 'o', color=COLORS['secondary'], markersize=8,
                markeredgewidth=2, markeredgecolor='white', zorder=4)
        
        # Annotate iteration
        if i < 2:  # Only label first few to avoid clutter
            ax1.text(c, fc + 1, f'c_{i}', ha='center', va='bottom',
                    fontsize=12, color=COLORS['secondary'], weight='medium')
        
        # Update bracket
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    ax1.set_xlabel(r'$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel(r'$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title('Bisection: Bracketing Approach', fontsize=16, 
                  color=COLORS['dark'], weight='medium', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1.4, 2.6)
    ax1.set_ylim(-8, 8)
    
    # Top right: Newton's method
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('white')
    
    ax2.plot(x_fine, y_fine, color=COLORS['primary'], linewidth=3, alpha=0.9, 
             label=r'$f(x) = x^3 - 2x - 5$', zorder=3)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax2.axvline(true_root, color='red', linestyle=':', alpha=0.7, linewidth=2)
    
    # Newton iterations starting from x0 = 2.3
    x = 2.3
    for i in range(4):
        fx = f(x)
        fpx = fprime(x)
        
        # Draw tangent line
        x_tangent = np.linspace(x - 0.3, x + 0.3, 100)
        y_tangent = fx + fpx * (x_tangent - x)
        ax2.plot(x_tangent, y_tangent, color=COLORS['accent'], linewidth=2.5,
                alpha=0.8 - i*0.15, linestyle='--', zorder=2)
        
        # Mark current point
        ax2.plot(x, fx, 'o', color=COLORS['accent'], markersize=8,
                markeredgewidth=2, markeredgecolor='white', zorder=4)
        
        # Find x-intercept (next iterate)
        x_next = x - fx / fpx
        ax2.plot(x_next, 0, 's', color=COLORS['accent'], markersize=6,
                markeredgewidth=1, markeredgecolor='white', alpha=0.8, zorder=4)
        
        # Draw vertical line to show next point
        if i < 3:
            ax2.plot([x_next, x_next], [0, f(x_next)], color=COLORS['neutral'],
                    linestyle=':', alpha=0.5, zorder=1)
        
        if i < 2:  # Label first iterations
            ax2.text(x, fx + 1, f'x_{i}', ha='center', va='bottom',
                    fontsize=12, color=COLORS['accent'], weight='medium')
        
        x = x_next
    
    ax2.set_xlabel(r'$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel(r'$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title('Newton: Tangent Line Approach', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1.4, 2.6)
    ax2.set_ylim(-8, 8)
    
    # Bottom left: Secant method
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('white')
    
    ax3.plot(x_fine, y_fine, color=COLORS['primary'], linewidth=3, alpha=0.9, 
             label=r'$f(x) = x^3 - 2x - 5$', zorder=3)
    ax3.axhline(0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax3.axvline(true_root, color='red', linestyle=':', alpha=0.7, linewidth=2)
    
    # Secant method starting with x0 = 1.8, x1 = 2.4
    x0, x1 = 1.8, 2.4
    for i in range(3):
        f0, f1 = f(x0), f(x1)
        
        # Draw secant line
        slope = (f1 - f0) / (x1 - x0)
        x_secant = np.linspace(min(x0, x1) - 0.2, max(x0, x1) + 0.2, 100)
        y_secant = f0 + slope * (x_secant - x0)
        ax3.plot(x_secant, y_secant, color=COLORS['secondary'], linewidth=2.5,
                alpha=0.8 - i*0.15, linestyle='-', zorder=2)
        
        # Mark points
        ax3.plot([x0, x1], [f0, f1], 'o', color=COLORS['secondary'], markersize=8,
                markeredgewidth=2, markeredgecolor='white', zorder=4)
        
        # Find x-intercept
        x2 = x0 - f0 * (x1 - x0) / (f1 - f0)
        ax3.plot(x2, 0, 's', color=COLORS['secondary'], markersize=6,
                markeredgewidth=1, markeredgecolor='white', alpha=0.8, zorder=4)
        
        # Update points
        x0, x1 = x1, x2
    
    ax3.set_xlabel(r'$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax3.set_ylabel(r'$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax3.set_title('Secant: Interpolation Approach', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1.4, 2.6)
    ax3.set_ylim(-8, 8)
    
    # Bottom right: Convergence comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('white')
    
    # Simulate convergence for all three methods
    iterations = np.arange(1, 11)
    
    # Bisection: linear convergence
    bisection_error = 0.5 * (0.5)**iterations
    
    # Newton: quadratic convergence (starting from reasonable error)
    newton_error = [0.1]
    for i in range(1, 10):
        newton_error.append(newton_error[-1]**2 * 0.5)  # Roughly quadratic
    newton_error = np.array(newton_error)
    
    # Secant: superlinear convergence (golden ratio)
    phi = (1 + np.sqrt(5)) / 2
    secant_error = 0.1 * (0.3)**iterations**((phi-1))
    
    ax4.semilogy(iterations, bisection_error, 'o-', color=COLORS['secondary'],
                linewidth=3, markersize=6, label='Bisection (linear)', alpha=0.9)
    ax4.semilogy(iterations, newton_error, 's-', color=COLORS['accent'],
                linewidth=3, markersize=6, label='Newton (quadratic)', alpha=0.9)
    ax4.semilogy(iterations, secant_error, '^-', color=COLORS['primary'],
                linewidth=3, markersize=6, label='Secant (superlinear)', alpha=0.9)
    
    # Add reference lines
    ax4.plot([1, 10], [0.5, 0.5*0.5**9], '--', color=COLORS['neutral'],
            alpha=0.6, label=r'$O(0.5^n)$ reference')
    
    ax4.set_xlabel('Iteration Number', fontsize=16, color=COLORS['dark'], weight='medium')
    ax4.set_ylabel('Absolute Error', fontsize=16, color=COLORS['dark'], weight='medium')
    ax4.set_title('Convergence Rate Comparison', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax4.legend(fontsize=12, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax4.grid(True, alpha=0.3)
    
    # Style all axes
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "2_1_root_methods_comparison")
    reset_style()


def figure_2_2_newton_failure_modes():
    """
    Newton's method failure modes illustration.
    
    Location: Part 1, Section "When Newton Fails"
    Caption: Newton's method failure modes: (a) Cycling between points, (b) Divergence 
             from poor initial guess, (c) Vertical tangent at root, (d) Slow convergence 
             for repeated roots. These cases illustrate why we need alternative methods.
    """
    set_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('white')
    
    # Panel 1: Cycling behavior
    def f1(x):
        return x**3 - 2*x + 2  # No real roots, creates cycling
    
    def f1_prime(x):
        return 3*x**2 - 2
    
    x = np.linspace(-2.5, 2.5, 1000)
    y1 = f1(x)
    
    ax1.plot(x, y1, color=COLORS['primary'], linewidth=3, alpha=0.9, zorder=3)
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax1.grid(True, alpha=0.3)
    
    # Simulate cycling starting from x = 0
    x_cycle = [0]
    for i in range(6):
        xi = x_cycle[-1]
        f_xi = f1(xi)
        fp_xi = f1_prime(xi)
        if fp_xi != 0:
            x_next = xi - f_xi / fp_xi
            x_cycle.append(x_next)
        
        # Draw tangent
        if i < 4:  # Only show first few
            x_tan = np.linspace(xi - 1, xi + 1, 100)
            y_tan = f_xi + fp_xi * (x_tan - xi)
            ax1.plot(x_tan, y_tan, '--', color=COLORS['secondary'], 
                    alpha=0.7 - i*0.1, linewidth=2, zorder=2)
            ax1.plot(xi, f_xi, 'o', color=COLORS['secondary'], markersize=8,
                    markeredgewidth=2, markeredgecolor='white', zorder=4)
    
    # Show cycling pattern with arrows
    for i in range(min(4, len(x_cycle)-1)):
        ax1.annotate('', xy=(x_cycle[i+1], f1(x_cycle[i+1])), 
                    xytext=(x_cycle[i], f1(x_cycle[i])),
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent'], 
                                  lw=2, alpha=0.8))
    
    ax1.set_title('(a) Cycling Behavior', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_xlabel(r'$x$', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel(r'$f(x)$', fontsize=14, color=COLORS['dark'], weight='medium')
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-5, 5)
    
    # Panel 2: Divergence from poor initial guess
    def f2(x):
        return np.arctan(x)  # Well-behaved but Newton can diverge
    
    def f2_prime(x):
        return 1 / (1 + x**2)
    
    x = np.linspace(-5, 5, 1000)
    y2 = f2(x)
    
    ax2.plot(x, y2, color=COLORS['primary'], linewidth=3, alpha=0.9, zorder=3)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax2.grid(True, alpha=0.3)
    
    # Starting from poor initial guess x0 = 2
    x_div = 2.0
    for i in range(4):
        f_xi = f2(x_div)
        fp_xi = f2_prime(x_div)
        
        # Draw tangent
        x_tan = np.linspace(x_div - 1, x_div + 3, 100)
        y_tan = f_xi + fp_xi * (x_tan - x_div)
        ax2.plot(x_tan, y_tan, '--', color=COLORS['secondary'], 
                alpha=0.7 - i*0.1, linewidth=2, zorder=2)
        ax2.plot(x_div, f_xi, 'o', color=COLORS['secondary'], markersize=8,
                markeredgewidth=2, markeredgecolor='white', zorder=4)
        
        # Next iterate (diverges)
        x_next = x_div - f_xi / fp_xi
        ax2.plot(x_next, 0, 's', color=COLORS['accent'], markersize=6,
                alpha=0.8, zorder=4)
        
        x_div = x_next
        if abs(x_div) > 5:  # Stop if diverging
            break
    
    ax2.set_title('(b) Divergence from Poor Guess', fontsize=16, 
                  color=COLORS['dark'], weight='medium')
    ax2.set_xlabel(r'$x$', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel(r'$f(x)$', fontsize=14, color=COLORS['dark'], weight='medium')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-2, 2)
    
    # Panel 3: Vertical tangent (infinite derivative)
    def f3(x):
        return np.sign(x) * np.abs(x)**(1/3)  # f(x) = x^(1/3)
    
    x_pos = np.linspace(0.001, 2, 500)
    x_neg = np.linspace(-2, -0.001, 500)
    y3_pos = f3(x_pos)
    y3_neg = f3(x_neg)
    
    ax3.plot(x_pos, y3_pos, color=COLORS['primary'], linewidth=3, alpha=0.9, zorder=3)
    ax3.plot(x_neg, y3_neg, color=COLORS['primary'], linewidth=3, alpha=0.9, zorder=3)
    ax3.axhline(0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax3.axvline(0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax3.grid(True, alpha=0.3)
    
    # Show vertical tangent at origin
    ax3.axvline(0, color=COLORS['secondary'], linewidth=4, alpha=0.8,
               linestyle='--', label="f'(0) = ∞", zorder=2)
    
    # Mark the problematic point
    ax3.plot(0, 0, 'o', color='red', markersize=12, 
            markeredgewidth=3, markeredgecolor='white', zorder=5)
    
    ax3.text(0.1, 0.5, "f'(0) = ∞\nNewton fails!", fontsize=12, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='red', alpha=0.9), color='red', weight='medium')
    
    ax3.set_title(r'(c) Vertical Tangent: $f(x) = x^{1/3}$', fontsize=16,
                  color=COLORS['dark'], weight='medium')
    ax3.set_xlabel(r'$x$', fontsize=14, color=COLORS['dark'], weight='medium')
    ax3.set_ylabel(r'$f(x)$', fontsize=14, color=COLORS['dark'], weight='medium')
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-1.5, 1.5)
    
    # Panel 4: Slow convergence for repeated roots
    def f4(x):
        return (x - 1)**2  # Double root at x = 1
    
    def f4_prime(x):
        return 2*(x - 1)
    
    x = np.linspace(-0.5, 2.5, 1000)
    y4 = f4(x)
    
    ax4.plot(x, y4, color=COLORS['primary'], linewidth=3, alpha=0.9, zorder=3)
    ax4.axhline(0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax4.axvline(1, color='red', linestyle=':', alpha=0.7, linewidth=2,
               label='Double root at x=1')
    ax4.grid(True, alpha=0.3)
    
    # Newton iterations showing slow convergence
    x_slow = 1.8
    iterates = [x_slow]
    for i in range(5):
        f_xi = f4(x_slow)
        fp_xi = f4_prime(x_slow)
        
        # Draw tangent
        x_tan = np.linspace(x_slow - 0.3, x_slow + 0.3, 100)
        y_tan = f_xi + fp_xi * (x_tan - x_slow)
        ax4.plot(x_tan, y_tan, '--', color=COLORS['secondary'], 
                alpha=0.7 - i*0.1, linewidth=2, zorder=2)
        ax4.plot(x_slow, f_xi, 'o', color=COLORS['secondary'], markersize=8,
                markeredgewidth=2, markeredgecolor='white', zorder=4)
        
        if fp_xi != 0:
            x_next = x_slow - f_xi / fp_xi
            iterates.append(x_next)
            ax4.plot(x_next, 0, 's', color=COLORS['accent'], markersize=6,
                    alpha=0.8, zorder=4)
            x_slow = x_next
    
    # Show slow linear convergence
    ax4.text(0.2, 0.4, 'Only LINEAR\nconvergence for\nrepeated roots!', 
            fontsize=12, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light'], 
                     edgecolor=COLORS['secondary'], alpha=0.9), 
            color=COLORS['secondary'], weight='medium')
    
    ax4.set_title(r'(d) Repeated Root: $f(x) = (x-1)^2$', fontsize=16,
                  color=COLORS['dark'], weight='medium')
    ax4.set_xlabel(r'$x$', fontsize=14, color=COLORS['dark'], weight='medium')
    ax4.set_ylabel(r'$f(x)$', fontsize=14, color=COLORS['dark'], weight='medium')
    ax4.set_xlim(-0.5, 2.5)
    ax4.set_ylim(0, 1.0)
    
    # Style all axes
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=12, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "2_2_newton_failure_modes")
    reset_style()


def figure_2_3_hybrid_method_performance():
    """
    Hybrid method combining bisection and Newton for robustness.
    
    Location: Part 1, Section "Hybrid Methods: Best of All Worlds"
    Caption: Hybrid method combining bisection and Newton for f(x) = x^20 - 1. Pure Newton 
             fails due to poor conditioning, pure bisection is slow, but the hybrid approach 
             switches methods adaptively for both robustness and speed.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Test function: f(x) = x^20 - 1 (poorly conditioned near root x = 1)
    def f(x):
        return x**20 - 1
    
    def fprime(x):
        return 20 * x**19
    
    # Left panel: Show the function and its conditioning
    x = np.linspace(0.8, 1.2, 1000)
    y = f(x)
    
    ax1.plot(x, y, color=COLORS['primary'], linewidth=4, alpha=0.9, 
             label=r'$f(x) = x^{20} - 1$', zorder=3)
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax1.axvline(1, color='red', linestyle=':', alpha=0.7, linewidth=2, 
               label='True root: x = 1')
    ax1.grid(True, alpha=0.3)
    
    # Show derivative to illustrate conditioning
    ax1_twin = ax1.twinx()
    fprime_vals = fprime(x)
    ax1_twin.plot(x, fprime_vals, color=COLORS['accent'], linewidth=3, 
                 alpha=0.7, linestyle='--', label=r"$f'(x) = 20x^{19}$")
    
    # Highlight the conditioning problem
    ax1.text(0.85, 0.5, 'Near x=1:\nf(x) ≈ 0 but\nf\'(x) = 20\n(Well conditioned)', 
            fontsize=12, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'], 
                     edgecolor=COLORS['primary'], alpha=0.9), 
            color=COLORS['primary'], weight='medium')
    
    ax1.text(1.05, -0.5, 'Poor initial guess\nleads to divergence', 
            fontsize=12, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='red', alpha=0.9), 
            color='red', weight='medium')
    
    ax1.set_xlabel(r'$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel(r'$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1_twin.set_ylabel(r"$f'(x)$", fontsize=16, color=COLORS['accent'], weight='medium')
    ax1.set_title(r'Challenge: $f(x) = x^{20} - 1$', fontsize=18,
                  color=COLORS['dark'], weight='medium', pad=20)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12,
              frameon=True, fancybox=True, edgecolor=COLORS['neutral'], facecolor='white')
    
    ax1.set_xlim(0.8, 1.2)
    ax1.set_ylim(-1, 1)
    ax1_twin.set_ylim(0, 25)
    
    # Right panel: Convergence comparison
    iterations = np.arange(1, 21)
    
    # Pure Newton (fails from poor guess)
    newton_error = []
    x_newton = 1.1  # Poor initial guess
    for i in range(20):
        error = abs(x_newton - 1.0)
        newton_error.append(error)
        
        fx = f(x_newton)
        fpx = fprime(x_newton)
        
        if abs(fpx) < 1e-14:  # Avoid division by zero
            newton_error.extend([newton_error[-1]] * (20 - i - 1))
            break
            
        x_newton = x_newton - fx / fpx
        
        # Simulate divergence if goes too far
        if abs(x_newton - 1) > 1:
            newton_error.extend([10] * (20 - i - 1))
            break
    
    newton_error = np.array(newton_error[:20])
    
    # Pure bisection (slow but steady)
    bisection_error = []
    a, b = 0.8, 1.2
    for i in range(20):
        c = (a + b) / 2
        error = abs(c - 1.0)
        bisection_error.append(error)
        
        fc = f(c)
        fa = f(a)
        
        if fa * fc < 0:
            b = c
        else:
            a = c
    
    bisection_error = np.array(bisection_error)
    
    # Hybrid method (switch strategies)
    hybrid_error = []
    x_hybrid = 1.1
    a_bracket, b_bracket = 0.8, 1.2
    
    for i in range(20):
        error = abs(x_hybrid - 1.0)
        hybrid_error.append(error)
        
        fx = f(x_hybrid)
        fpx = fprime(x_hybrid)
        
        # Decision logic: use Newton if well-conditioned and inside bracket
        if (abs(fpx) > 1e-10 and 
            a_bracket < x_hybrid < b_bracket and 
            error > 1e-12):
            # Try Newton step
            x_newton_step = x_hybrid - fx / fpx
            
            # Accept Newton step if it stays in bracket and reduces error
            if (a_bracket < x_newton_step < b_bracket and 
                abs(x_newton_step - 1) < error):
                x_hybrid = x_newton_step
            else:
                # Fall back to bisection
                c = (a_bracket + b_bracket) / 2
                x_hybrid = c
                fc = f(c)
                fa = f(a_bracket)
                
                if fa * fc < 0:
                    b_bracket = c
                else:
                    a_bracket = c
        else:
            # Use bisection to maintain bracket
            c = (a_bracket + b_bracket) / 2
            x_hybrid = c
            fc = f(c)
            fa = f(a_bracket)
            
            if fa * fc < 0:
                b_bracket = c
            else:
                a_bracket = c
    
    hybrid_error = np.array(hybrid_error)
    
    # Plot convergence comparison
    ax2.semilogy(iterations, newton_error, 'o--', color='red', linewidth=3,
                markersize=6, label='Pure Newton (fails)', alpha=0.9)
    ax2.semilogy(iterations, bisection_error, 's-', color=COLORS['secondary'],
                linewidth=3, markersize=6, label='Pure bisection (slow)', alpha=0.9)
    ax2.semilogy(iterations, hybrid_error, '^-', color=COLORS['primary'],
                linewidth=3, markersize=6, label='Hybrid (robust + fast)', alpha=0.9)
    
    # Add annotations
    ax2.annotate('Newton diverges\nfrom poor guess', 
                xy=(5, newton_error[4]), xytext=(8, 1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, ha='center', va='center', color='red', weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='red', alpha=0.9))
    
    ax2.annotate('Bisection: guaranteed\nbut slow convergence', 
                xy=(15, bisection_error[14]), xytext=(12, 1e-3),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2),
                fontsize=12, ha='center', va='center', color=COLORS['secondary'], 
                weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['secondary'], alpha=0.9))
    
    ax2.annotate('Hybrid: best of both\nworlds!', 
                xy=(10, hybrid_error[9]), xytext=(15, 1e-6),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2),
                fontsize=12, ha='center', va='center', color=COLORS['primary'], 
                weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light'],
                         edgecolor=COLORS['primary'], alpha=0.9))
    
    ax2.set_xlabel('Iteration Number', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Absolute Error', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title('Convergence Comparison', fontsize=18,
                  color=COLORS['dark'], weight='medium', pad=20)
    ax2.legend(fontsize=14, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-16, 10)
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    # Style twin axis
    for spine in ax1_twin.spines.values():
        spine.set_color(COLORS['neutral'])
        spine.set_linewidth(1.2)
    ax1_twin.tick_params(axis='both', labelsize=14, colors=COLORS['accent'])
    
    plt.tight_layout()
    save_figure(fig, "2_3_hybrid_method_performance")
    reset_style()


# ============================================================================
# PART 2 FIGURES: Quadrature
# ============================================================================

def figure_2_4_integration_methods_geometry():
    """
    Four geometric approaches to numerical integration.
    
    Location: Part 2, Section "Building Integration Methods"
    Caption: Four geometric approaches to numerical integration. Rectangle and midpoint 
             rules use constant approximations, trapezoidal uses linear interpolation, 
             and Simpson's uses parabolic interpolation. Higher-order interpolation 
             generally yields better accuracy for smooth functions.
    """
    set_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('white')
    
    # Test function: sin(x) + 0.5 from 0 to π
    def f(x):
        return np.sin(x) + 0.5
    
    a, b = 0, np.pi
    x_fine = np.linspace(a, b, 1000)
    y_fine = f(x_fine)
    true_integral = 2.5  # Analytical result
    
    # Panel 1: Rectangle rule (left)
    n = 5
    x_rect = np.linspace(a, b, n+1)
    h = (b - a) / n
    
    ax1.plot(x_fine, y_fine, color=COLORS['primary'], linewidth=4, alpha=0.9, 
             label=r'$f(x) = \sin(x) + 0.5$', zorder=5)
    ax1.fill_between(x_fine, 0, y_fine, alpha=0.2, color=COLORS['primary'], zorder=1)
    
    # Draw rectangles
    rect_sum = 0
    for i in range(n):
        x_left = x_rect[i]
        height = f(x_left)
        rect_sum += height * h
        
        # Draw rectangle
        rect = patches.Rectangle((x_left, 0), h, height, 
                               linewidth=2, edgecolor=COLORS['secondary'],
                               facecolor=COLORS['secondary'], alpha=0.4, zorder=3)
        ax1.add_patch(rect)
        
        # Mark sample points
        ax1.plot(x_left, height, 'o', color=COLORS['secondary'], markersize=8,
                markeredgewidth=2, markeredgecolor='white', zorder=4)
    
    ax1.set_xlabel(r'$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel(r'$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title(f'Rectangle Rule (Left)\nApprox = {rect_sum:.3f}', 
                  fontsize=16, color=COLORS['dark'], weight='medium', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(a, b)
    ax1.set_ylim(0, 1.8)
    
    # Panel 2: Midpoint rule
    x_mid = np.linspace(a + h/2, b - h/2, n)
    
    ax2.plot(x_fine, y_fine, color=COLORS['primary'], linewidth=4, alpha=0.9, 
             label=r'$f(x) = \sin(x) + 0.5$', zorder=5)
    ax2.fill_between(x_fine, 0, y_fine, alpha=0.2, color=COLORS['primary'], zorder=1)
    
    # Draw midpoint rectangles
    mid_sum = 0
    for i in range(n):
        x_center = x_mid[i]
        height = f(x_center)
        mid_sum += height * h
        
        # Draw rectangle centered on midpoint
        rect = patches.Rectangle((x_center - h/2, 0), h, height,
                               linewidth=2, edgecolor=COLORS['accent'],
                               facecolor=COLORS['accent'], alpha=0.4, zorder=3)
        ax2.add_patch(rect)
        
        # Mark sample points
        ax2.plot(x_center, height, 'o', color=COLORS['accent'], markersize=8,
                markeredgewidth=2, markeredgecolor='white', zorder=4)
    
    ax2.set_xlabel(r'$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel(r'$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title(f'Midpoint Rule\nApprox = {mid_sum:.3f}', 
                  fontsize=16, color=COLORS['dark'], weight='medium', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(a, b)
    ax2.set_ylim(0, 1.8)
    
    # Panel 3: Trapezoidal rule
    ax3.plot(x_fine, y_fine, color=COLORS['primary'], linewidth=4, alpha=0.9, 
             label=r'$f(x) = \sin(x) + 0.5$', zorder=5)
    ax3.fill_between(x_fine, 0, y_fine, alpha=0.2, color=COLORS['primary'], zorder=1)
    
    # Draw trapezoids
    trap_sum = 0
    for i in range(n):
        x1, x2 = x_rect[i], x_rect[i+1]
        y1, y2 = f(x1), f(x2)
        trap_sum += (y1 + y2) * h / 2
        
        # Draw trapezoid
        verts = [(x1, 0), (x1, y1), (x2, y2), (x2, 0)]
        trap = patches.Polygon(verts, linewidth=2, edgecolor=COLORS['secondary'],
                             facecolor=COLORS['secondary'], alpha=0.4, zorder=3)
        ax3.add_patch(trap)
        
        # Draw connecting line
        ax3.plot([x1, x2], [y1, y2], '-', color=COLORS['secondary'], 
                linewidth=3, alpha=0.8, zorder=4)
        
        # Mark sample points
        ax3.plot([x1, x2], [y1, y2], 'o', color=COLORS['secondary'], markersize=8,
                markeredgewidth=2, markeredgecolor='white', zorder=5)
    
    ax3.set_xlabel(r'$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax3.set_ylabel(r'$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax3.set_title(f'Trapezoidal Rule\nApprox = {trap_sum:.3f}', 
                  fontsize=16, color=COLORS['dark'], weight='medium', pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(a, b)
    ax3.set_ylim(0, 1.8)
    
    # Panel 4: Simpson's rule (need even n)
    n_simp = 6  # Must be even
    x_simp = np.linspace(a, b, n_simp+1)
    h_simp = (b - a) / n_simp
    
    ax4.plot(x_fine, y_fine, color=COLORS['primary'], linewidth=4, alpha=0.9, 
             label=r'$f(x) = \sin(x) + 0.5$', zorder=5)
    ax4.fill_between(x_fine, 0, y_fine, alpha=0.2, color=COLORS['primary'], zorder=1)
    
    # Draw parabolic segments
    simp_sum = 0
    for i in range(0, n_simp, 2):
        x0, x1, x2 = x_simp[i], x_simp[i+1], x_simp[i+2]
        y0, y1, y2 = f(x0), f(x1), f(x2)
        
        # Simpson's rule for this segment
        segment_integral = (h_simp / 3) * (y0 + 4*y1 + y2)
        simp_sum += segment_integral
        
        # Draw parabola through three points
        x_para = np.linspace(x0, x2, 100)
        # Lagrange interpolation for parabola
        y_para = (y0 * (x_para - x1) * (x_para - x2) / ((x0 - x1) * (x0 - x2)) +
                  y1 * (x_para - x0) * (x_para - x2) / ((x1 - x0) * (x1 - x2)) +
                  y2 * (x_para - x0) * (x_para - x1) / ((x2 - x0) * (x2 - x1)))
        
        ax4.plot(x_para, y_para, '-', color=COLORS['accent'], linewidth=3, 
                alpha=0.8, zorder=4)
        ax4.fill_between(x_para, 0, y_para, alpha=0.4, color=COLORS['accent'], zorder=3)
        
        # Mark sample points
        ax4.plot([x0, x1, x2], [y0, y1, y2], 'o', color=COLORS['accent'], 
                markersize=8, markeredgewidth=2, markeredgecolor='white', zorder=5)
    
    ax4.set_xlabel(r'$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax4.set_ylabel(r'$f(x)$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax4.set_title(f'Simpson\'s Rule\nApprox = {simp_sum:.3f}', 
                  fontsize=16, color=COLORS['dark'], weight='medium', pad=15)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(a, b)
    ax4.set_ylim(0, 1.8)
    
    # Add true integral annotation to each panel
    true_text = f'True = {true_integral:.3f}'
    for ax in [ax1, ax2, ax3, ax4]:
        ax.text(0.02, 0.95, true_text, transform=ax.transAxes, 
               fontsize=12, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=COLORS['primary'], alpha=0.9), 
               color=COLORS['primary'], weight='medium')
    
    # Style all axes
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "2_4_integration_methods_geometry")
    reset_style()


def figure_2_5_simpson_vs_trapezoid_scaling():
    """
    Error scaling comparison for integration methods.
    
    Location: Part 2, Section "Method 3: Simpson's Rule"
    Caption: Error scaling comparison for Trapezoidal (O(h²)) and Simpson's (O(h⁴)) rules. 
             Note the different slopes on the log-log plot corresponding to different 
             convergence orders. Simpson's rule achieves machine precision with far fewer points.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Test function: exp(x) from 0 to 1
    def f(x):
        return np.exp(x)
    
    a, b = 0, 1
    true_integral = np.exp(1) - 1  # Analytical result
    
    # Range of n values
    n_values = 2**np.arange(1, 12)  # Powers of 2 for clear scaling
    h_values = (b - a) / n_values
    
    # Calculate errors
    trap_errors = []
    simp_errors = []
    
    for n in n_values:
        h = (b - a) / n
        x_points = np.linspace(a, b, n+1)
        y_points = f(x_points)
        
        # Trapezoidal rule
        trap_result = h * (y_points[0]/2 + np.sum(y_points[1:-1]) + y_points[-1]/2)
        trap_error = abs(trap_result - true_integral)
        trap_errors.append(trap_error)
        
        # Simpson's rule (need even n)
        if n % 2 == 0:
            simp_result = h/3 * (y_points[0] + 
                               4*np.sum(y_points[1:-1:2]) + 
                               2*np.sum(y_points[2:-1:2]) + 
                               y_points[-1])
            simp_error = abs(simp_result - true_integral)
            simp_errors.append(simp_error)
        else:
            simp_errors.append(np.nan)
    
    trap_errors = np.array(trap_errors)
    simp_errors = np.array(simp_errors)
    
    # Left panel: Error vs n
    # Only plot Simpson where n is even
    valid_simp = ~np.isnan(simp_errors)
    
    ax1.loglog(n_values, trap_errors, 'o-', color=COLORS['secondary'], 
              linewidth=3, markersize=8, label='Trapezoidal: O(h²)', alpha=0.9)
    ax1.loglog(n_values[valid_simp], simp_errors[valid_simp], 's-', 
              color=COLORS['primary'], linewidth=3, markersize=8, 
              label='Simpson\'s: O(h⁴)', alpha=0.9)
    
    # Add reference lines for scaling
    # O(h²) reference
    h2_ref = 0.1 * h_values**2
    ax1.loglog(n_values, h2_ref, '--', color=COLORS['neutral'], 
              alpha=0.7, linewidth=2, label='h² reference')
    
    # O(h⁴) reference  
    h4_ref = 0.01 * h_values**4
    ax1.loglog(n_values, h4_ref, ':', color=COLORS['neutral'], 
              alpha=0.7, linewidth=2, label='h⁴ reference')
    
    # Machine precision reference
    ax1.axhline(2.22e-16, color='red', linestyle='-.', alpha=0.8, linewidth=2,
               label='Machine ε')
    
    ax1.set_xlabel('Number of Intervals (n)', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('Absolute Error', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title(r'Error Scaling: $\int_0^1 e^x dx$', fontsize=18,
                  color=COLORS['dark'], weight='medium', pad=20)
    ax1.legend(fontsize=14, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 3000)
    ax1.set_ylim(1e-16, 1e-1)
    
    # Add annotations for slopes
    ax1.annotate('Slope = -2\n(quadratic)', 
                xy=(32, trap_errors[7]), xytext=(100, 1e-6),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2),
                fontsize=12, ha='center', va='center', color=COLORS['secondary'], 
                weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['secondary'], alpha=0.9))
    
    ax1.annotate('Slope = -4\n(quartic)', 
                xy=(64, simp_errors[10]), xytext=(200, 1e-10),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2),
                fontsize=12, ha='center', va='center', color=COLORS['primary'], 
                weight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light'],
                         edgecolor=COLORS['primary'], alpha=0.9))
    
    # Right panel: Efficiency comparison
    # Show work vs accuracy trade-off
    target_errors = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    trap_n_needed = []
    simp_n_needed = []
    
    for target in target_errors:
        # Find minimum n for trapezoidal
        trap_idx = np.where(trap_errors <= target)[0]
        if len(trap_idx) > 0:
            trap_n_needed.append(n_values[trap_idx[0]])
        else:
            trap_n_needed.append(np.nan)
        
        # Find minimum n for Simpson's
        simp_idx = np.where((simp_errors <= target) & valid_simp)[0]
        if len(simp_idx) > 0:
            simp_n_needed.append(n_values[simp_idx[0]])
        else:
            simp_n_needed.append(np.nan)
    
    trap_n_needed = np.array(trap_n_needed)
    simp_n_needed = np.array(simp_n_needed)
    
    # Only plot where both methods can achieve the target
    both_valid = ~(np.isnan(trap_n_needed) | np.isnan(simp_n_needed))
    target_errors = np.array(target_errors)
    
    ax2.loglog(target_errors[both_valid], trap_n_needed[both_valid], 'o-',
              color=COLORS['secondary'], linewidth=3, markersize=8,
              label='Trapezoidal rule', alpha=0.9)
    ax2.loglog(target_errors[both_valid], simp_n_needed[both_valid], 's-',
              color=COLORS['primary'], linewidth=3, markersize=8,
              label='Simpson\'s rule', alpha=0.9)
    
    # Add efficiency annotation
    crossover_idx = 2  # Around 1e-6 error level
    if crossover_idx < len(target_errors) and crossover_idx < len(trap_n_needed):
        ratio = trap_n_needed[crossover_idx] / simp_n_needed[crossover_idx]
        ax2.annotate(f'Simpson needs {ratio:.0f}×\nfewer points!', 
                    xy=(target_errors[crossover_idx], simp_n_needed[crossover_idx]), 
                    xytext=(1e-8, 20),
                    arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2),
                    fontsize=14, ha='center', va='center', color=COLORS['primary'], 
                    weight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'],
                             edgecolor=COLORS['primary'], alpha=0.9))
    
    ax2.set_xlabel('Target Error', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Intervals Needed (n)', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title('Efficiency Comparison', fontsize=18,
                  color=COLORS['dark'], weight='medium', pad=20)
    ax2.legend(fontsize=14, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1e-12, 1e-2)
    ax2.set_ylim(1, 10000)
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "2_5_simpson_vs_trapezoid_scaling")
    reset_style()


def figure_2_6_monte_carlo_convergence():
    """
    Monte Carlo integration visualization and convergence.
    
    Location: Part 2, Section "Method 5: Monte Carlo Integration"
    Caption: Monte Carlo integration: (Left) Random sampling of a 2D function. Unlike grid 
             methods, points are distributed randomly. (Right) Convergence of the integral 
             estimate showing characteristic N^(-1/2) scaling with statistical error bars.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
    
    # Left panel: 2D function visualization with random sampling
    # Use a 2D Gaussian-like function
    def f_2d(x, y):
        return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.2)
    
    # Create a grid for visualization
    x_grid = np.linspace(0, 1, 100)
    y_grid = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f_2d(X, Y)
    
    # Plot the function as a heatmap
    im = ax1.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.7)
    ax1.contour(X, Y, Z, levels=10, colors='white', alpha=0.5, linewidths=0.8)
    
    # Add random sample points
    np.random.seed(42)
    n_samples = 200
    x_random = np.random.uniform(0, 1, n_samples)
    y_random = np.random.uniform(0, 1, n_samples)
    z_random = f_2d(x_random, y_random)
    
    # Color points by function value
    scatter = ax1.scatter(x_random, y_random, c=z_random, s=30, 
                         cmap='Reds', alpha=0.8, edgecolors='white', 
                         linewidth=0.5, zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label(r'$f(x,y)$', fontsize=14, color=COLORS['dark'], weight='medium')
    cbar.ax.tick_params(labelsize=12, colors=COLORS['neutral'])
    
    ax1.set_xlabel(r'$x$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel(r'$y$', fontsize=16, color=COLORS['dark'], weight='medium')
    ax1.set_title(r'Monte Carlo Sampling: $f(x,y) = e^{-(x-0.5)^2-(y-0.5)^2)/0.2}$', 
                  fontsize=16, color=COLORS['dark'], weight='medium', pad=15)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    
    # Add annotation explaining the method
    ax1.text(0.02, 0.98, f'N = {n_samples} random samples\nEstimate = Volume × Average', 
            transform=ax1.transAxes, fontsize=12, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor=COLORS['primary'], alpha=0.95), 
            color=COLORS['primary'], weight='medium')
    
    # Right panel: Convergence with error bars
    # True integral (computed numerically with high precision)
    from scipy import integrate
    
    def f_1d(x):
        return np.sin(x) * np.exp(-x/2)  # 1D function for clearer demonstration
    
    true_integral, _ = integrate.quad(f_1d, 0, np.pi)
    
    # Monte Carlo simulation
    n_points = np.logspace(1, 5, 50).astype(int)
    n_runs = 20  # Multiple runs for error bars
    
    mc_estimates = []
    mc_errors = []
    
    np.random.seed(42)
    for n in n_points:
        estimates = []
        for run in range(n_runs):
            x_mc = np.random.uniform(0, np.pi, n)
            y_mc = f_1d(x_mc)
            estimate = np.pi * np.mean(y_mc)  # Volume × average
            estimates.append(estimate)
        
        estimates = np.array(estimates)
        mean_estimate = np.mean(estimates)
        std_error = np.std(estimates)
        
        mc_estimates.append(mean_estimate)
        mc_errors.append(std_error)
    
    mc_estimates = np.array(mc_estimates)
    mc_errors = np.array(mc_errors)
    
    # Plot convergence with error bars
    ax2.errorbar(n_points, np.abs(mc_estimates - true_integral), 
                yerr=mc_errors, fmt='o', color=COLORS['primary'], 
                linewidth=2, markersize=6, capsize=3, alpha=0.8,
                label='Monte Carlo estimate')
    
    # Add theoretical N^(-1/2) reference line
    # Estimate variance from the data
    variance_estimate = np.var(f_1d(np.random.uniform(0, np.pi, 10000)))
    theoretical_error = np.sqrt(variance_estimate * np.pi**2 / n_points)
    
    ax2.loglog(n_points, theoretical_error, '--', color=COLORS['secondary'], 
              linewidth=3, alpha=0.8, label=r'$N^{-1/2}$ theoretical')
    
    # Add machine precision reference
    ax2.axhline(2.22e-16, color='red', linestyle='-.', alpha=0.6, linewidth=2,
               label='Machine ε')
    
    # Mark specific points
    convergence_points = [100, 1000, 10000]
    for i, n in enumerate(convergence_points):
        idx = np.argmin(np.abs(n_points - n))
        error = np.abs(mc_estimates[idx] - true_integral)
        if i == 1:  # Annotate middle point
            ax2.annotate(f'N = {n:,}\nError ≈ {error:.4f}', 
                        xy=(n, error), xytext=(n*3, error*10),
                        arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2),
                        fontsize=12, ha='left', va='center', color=COLORS['accent'], 
                        weight='medium',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor=COLORS['accent'], alpha=0.9))
    
    ax2.set_xlabel('Number of Samples (N)', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Absolute Error', fontsize=16, color=COLORS['dark'], weight='medium')
    ax2.set_title(r'MC Convergence: $\int_0^\pi \sin(x)e^{-x/2} dx$', fontsize=16,
                  color=COLORS['dark'], weight='medium', pad=15)
    ax2.legend(fontsize=14, frameon=True, fancybox=True,
              edgecolor=COLORS['neutral'], facecolor='white', loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(10, 100000)
    ax2.set_ylim(1e-6, 1e-1)
    
    # Add key insight
    ax2.text(0.02, 0.05, 'Key insight:\nError ∝ N^(-1/2)\nindependent of dimension!', 
            transform=ax2.transAxes, fontsize=12, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'], 
                     edgecolor=COLORS['primary'], alpha=0.95), 
            color=COLORS['primary'], weight='bold')
    
    # Style spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=14, colors=COLORS['neutral'])
    
    plt.tight_layout()
    save_figure(fig, "2_6_monte_carlo_convergence")
    reset_style()


def figure_2_7_integration_method_decision_tree():
    """
    Decision flowchart for selecting integration methods.
    
    Location: Part 2, Section "Choosing the Right Integration Method"
    Caption: Decision tree for selecting integration methods. The key factors are 
             dimensionality, smoothness, and noise level. Examples show typical 
             applications for each method.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Decision tree structure - positions for nodes
    nodes = {
        'start': {'pos': (0.5, 0.95), 'text': 'Need to integrate?', 'type': 'start'},
        'dimension': {'pos': (0.5, 0.8), 'text': 'Dimension > 4?', 'type': 'decision'},
        'monte_carlo': {'pos': (0.8, 0.65), 'text': 'Use Monte Carlo\nIntegration', 'type': 'method'},
        'smooth': {'pos': (0.2, 0.65), 'text': 'Function smooth?', 'type': 'decision'},
        'simpson': {'pos': (0.05, 0.5), 'text': 'Use Simpson\'s Rule\n(high accuracy)', 'type': 'method'},
        'noisy': {'pos': (0.35, 0.5), 'text': 'Data noisy?', 'type': 'decision'},
        'trapezoid': {'pos': (0.55, 0.35), 'text': 'Use Trapezoidal Rule\n(robust)', 'type': 'method'},
        'adaptive': {'pos': (0.15, 0.35), 'text': 'Try Simpson with\nadaptive refinement', 'type': 'method'},
        
        # Special cases
        'experimental': {'pos': (0.75, 0.2), 'text': 'Irregular spacing?\nUse Trapezoidal', 'type': 'method'},
        'oscillatory': {'pos': (0.25, 0.2), 'text': 'Highly oscillatory?\nUse specialized methods', 'type': 'method'},
        'gaussian': {'pos': (0.05, 0.2), 'text': 'Very smooth?\nConsider Gaussian\nquadrature', 'type': 'method'}
    }
    
    # Draw nodes
    for node_id, props in nodes.items():
        x, y = props['pos']
        text = props['text']
        node_type = props['type']
        
        if node_type == 'start':
            # Start node - large circle
            circle = plt.Circle((x, y), 0.06, facecolor=COLORS['primary'], 
                              alpha=0.8, edgecolor='white', linewidth=3, zorder=3)
            ax.add_patch(circle)
            text_color = 'white'
            weight = 'bold'
            fontsize = 14
            
        elif node_type == 'decision':
            # Decision node - diamond shape (using circle for simplicity)
            circle = plt.Circle((x, y), 0.05, facecolor=COLORS['accent'], 
                              alpha=0.3, edgecolor=COLORS['accent'], linewidth=2, zorder=3)
            ax.add_patch(circle)
            text_color = COLORS['dark']
            weight = 'medium'
            fontsize = 12
            
        else:  # method node
            # Method node - rounded rectangle
            bbox = FancyBboxPatch((x-0.08, y-0.04), 0.16, 0.08, 
                                 boxstyle="round,pad=0.01", 
                                 facecolor=COLORS['secondary'], alpha=0.8, 
                                 edgecolor='white', linewidth=2, zorder=3)
            ax.add_patch(bbox)
            text_color = 'white'
            weight = 'bold'
            fontsize = 11
        
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
               color=text_color, weight=weight, zorder=4)
    
    # Draw connections with labels
    connections = [
        ('start', 'dimension', 'Yes', (0.5, 0.875), 'green'),
        ('dimension', 'monte_carlo', 'Yes', (0.65, 0.725), 'green'),
        ('dimension', 'smooth', 'No', (0.35, 0.725), 'red'),
        ('smooth', 'simpson', 'Yes', (0.125, 0.575), 'green'),
        ('smooth', 'noisy', 'No', (0.275, 0.575), 'red'),
        ('noisy', 'trapezoid', 'Yes', (0.45, 0.425), 'red'),
        ('noisy', 'adaptive', 'No', (0.25, 0.425), 'green'),
    ]
    
    for from_node, to_node, label, label_pos, color in connections:
        from_x, from_y = nodes[from_node]['pos']
        to_x, to_y = nodes[to_node]['pos']
        
        # Draw arrow
        ax.annotate('', xy=(to_x, to_y + 0.04), xytext=(from_x, from_y - 0.04),
                   arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], 
                                 lw=2, alpha=0.8), zorder=2)
        
        # Add label
        lx, ly = label_pos
        ax.text(lx, ly, label, ha='center', va='center', fontsize=10,
               color=color, weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                        edgecolor=color, alpha=0.9), zorder=4)
    
    # Add examples and characteristics for each method
    examples = [
        {'pos': (0.05, 0.08), 'title': 'Simpson\'s Rule', 
         'examples': ['Theoretical functions', 'Smooth data', 'High accuracy needed'],
         'color': COLORS['secondary']},
        {'pos': (0.35, 0.08), 'title': 'Trapezoidal Rule', 
         'examples': ['Experimental data', 'Irregular spacing', 'Noisy measurements'],
         'color': COLORS['secondary']},
        {'pos': (0.8, 0.08), 'title': 'Monte Carlo', 
         'examples': ['High-dimensional', 'Statistical mechanics', 'Bayesian inference'],
         'color': COLORS['secondary']}
    ]
    
    for example in examples:
        x, y = example['pos']
        title = example['title']
        ex_list = example['examples']
        color = example['color']
        
        # Title box
        bbox = FancyBboxPatch((x-0.08, y+0.02), 0.16, 0.04, 
                             boxstyle="round,pad=0.005", 
                             facecolor=color, alpha=0.2, 
                             edgecolor=color, linewidth=1, zorder=1)
        ax.add_patch(bbox)
        
        ax.text(x, y+0.04, title, ha='center', va='center', fontsize=11,
               color=color, weight='bold')
        
        # Examples
        for i, ex in enumerate(ex_list):
            ax.text(x, y - i*0.025, f'• {ex}', ha='center', va='center', 
                   fontsize=9, color=COLORS['dark'], alpha=0.8)
    
    # Add complexity/accuracy chart
    chart_x, chart_y = 0.75, 0.45
    chart_width, chart_height = 0.2, 0.15
    
    # Background for chart
    rect = patches.Rectangle((chart_x - chart_width/2, chart_y - chart_height/2), 
                           chart_width, chart_height, linewidth=1, 
                           edgecolor=COLORS['neutral'], facecolor=COLORS['light'], 
                           alpha=0.3, zorder=1)
    ax.add_patch(rect)
    
    # Chart title
    ax.text(chart_x, chart_y + chart_height/2 + 0.02, 'Accuracy vs Cost', 
           ha='center', va='bottom', fontsize=11, weight='bold', color=COLORS['dark'])
    
    # Plot methods on accuracy vs computational cost
    methods_chart = {
        'Rectangle': (0.3, 0.2, COLORS['neutral']),
        'Trapezoid': (0.6, 0.4, COLORS['secondary']),
        'Simpson': (0.8, 0.6, COLORS['primary']),
        'Monte Carlo': (0.4, 0.8, COLORS['accent'])  # High cost for low-D, but scalable
    }
    
    for method, (accuracy, cost, color) in methods_chart.items():
        plot_x = chart_x - chart_width/2 + accuracy * chart_width
        plot_y = chart_y - chart_height/2 + cost * chart_height
        
        ax.plot(plot_x, plot_y, 'o', color=color, markersize=8, 
               markeredgewidth=1.5, markeredgecolor='white', zorder=3)
        
        if method == 'Simpson':  # Label one for clarity
            ax.text(plot_x + 0.02, plot_y, method, ha='left', va='center',
                   fontsize=9, color=color, weight='medium')
    
    # Chart axes labels
    ax.text(chart_x, chart_y - chart_height/2 - 0.02, 'Accuracy →', 
           ha='center', va='top', fontsize=9, color=COLORS['neutral'])
    ax.text(chart_x - chart_width/2 - 0.02, chart_y, 'Cost ↑', 
           ha='right', va='center', fontsize=9, color=COLORS['neutral'], rotation=90)
    
    # Main title
    ax.set_title('Integration Method Selection Guide', fontsize=20, color=COLORS['dark'], 
                weight='medium', pad=30)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=3, label='Yes'),
        plt.Line2D([0], [0], color='red', lw=3, label='No'),
        patches.Patch(facecolor=COLORS['accent'], alpha=0.3, label='Decision'),
        patches.Patch(facecolor=COLORS['secondary'], alpha=0.8, label='Method')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
             frameon=True, fancybox=True, edgecolor=COLORS['neutral'], 
             facecolor='white', bbox_to_anchor=(0, 1))
    
    # Set limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    save_figure(fig, "2_7_integration_method_decision_tree")
    reset_style()


# ============================================================================
# PART 3 FIGURES: Synthesis
# ============================================================================

def figure_2_8_error_landscapes_unified():
    """
    Unified view of error analysis principles across numerical methods.
    
    Location: Part 3, Section "The Deep Connections"
    Caption: Unified error analysis framework spanning root finding and integration. 
             All methods share the fundamental trade-off between truncation error 
             (from approximations) and round-off error (from finite precision). 
             Understanding these universal patterns enables method selection and 
             parameter optimization across computational physics.
    """
    set_style()
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Create complex subplot layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1],
                         hspace=0.4, wspace=0.3)
    
    # Top row: Different method error landscapes
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Middle row: Convergence comparison and condition number effects
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Bottom row: Unified framework summary
    ax6 = fig.add_subplot(gs[2, :])
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor('white')
    
    # Panel 1: Root finding error landscape (Newton's method)
    h_values = np.logspace(-16, -1, 200)
    epsilon = 2.22e-16
    
    # Simulation of Newton's method error components
    # Truncation error: dominated by function evaluation errors
    truncation_rf = h_values  # Simplified model
    
    # Round-off error: when f'(x) is small, amplifies round-off
    condition_number = 2.0  # Moderate conditioning
    roundoff_rf = condition_number * epsilon / h_values
    
    total_rf = truncation_rf + roundoff_rf
    optimal_h_rf = h_values[np.argmin(total_rf)]
    
    ax1.loglog(h_values, truncation_rf, color=COLORS['secondary'], linewidth=3,
              label='Truncation', alpha=0.9)
    ax1.loglog(h_values, roundoff_rf, color=COLORS['accent'], linewidth=3,
              label='Round-off', alpha=0.9)
    ax1.loglog(h_values, total_rf, color=COLORS['dark'], linewidth=4,
              label='Total error', alpha=0.9)
    ax1.axvline(optimal_h_rf, color='red', linestyle=':', alpha=0.8,
               label=f'Optimal h = {optimal_h_rf:.2e}')
    
    ax1.set_xlabel('Step size h', fontsize=12, color=COLORS['dark'], weight='medium')
    ax1.set_ylabel('Error', fontsize=12, color=COLORS['dark'], weight='medium')
    ax1.set_title('Root Finding\n(Newton Method)', fontsize=14, 
                  color=COLORS['dark'], weight='medium')
    ax1.legend(fontsize=10, frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1e-16, 1e-1)
    ax1.set_ylim(1e-16, 1e0)
    
    # Panel 2: Integration error landscape (Trapezoidal rule)
    # Truncation error: O(h²) for trapezoidal
    truncation_int = 0.1 * h_values**2
    
    # Round-off error: accumulation of rounding in sum
    n_points = 1 / h_values  # Number of intervals
    roundoff_int = epsilon * n_points
    
    total_int = truncation_int + roundoff_int
    optimal_h_int = h_values[np.argmin(total_int)]
    
    ax2.loglog(h_values, truncation_int, color=COLORS['secondary'], linewidth=3,
              label='Truncation', alpha=0.9)
    ax2.loglog(h_values, roundoff_int, color=COLORS['accent'], linewidth=3,
              label='Round-off', alpha=0.9)
    ax2.loglog(h_values, total_int, color=COLORS['dark'], linewidth=4,
              label='Total error', alpha=0.9)
    ax2.axvline(optimal_h_int, color='red', linestyle=':', alpha=0.8,
               label=f'Optimal h = {optimal_h_int:.2e}')
    
    ax2.set_xlabel('Step size h', fontsize=12, color=COLORS['dark'], weight='medium')
    ax2.set_ylabel('Error', fontsize=12, color=COLORS['dark'], weight='medium')
    ax2.set_title('Numerical Integration\n(Trapezoidal Rule)', fontsize=14,
                  color=COLORS['dark'], weight='medium')
    ax2.legend(fontsize=10, frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1e-16, 1e-1)
    ax2.set_ylim(1e-16, 1e0)
    
    # Panel 3: Monte Carlo "error landscape" 
    n_samples = np.logspace(1, 8, 100)
    
    # Monte Carlo has no "truncation" in the traditional sense, but statistical error
    statistical_error = 1 / np.sqrt(n_samples)  # N^(-1/2) scaling
    
    # "Round-off" represents machine precision limits on sums
    mc_roundoff = epsilon * n_samples  # Accumulation in averaging
    
    # Total dominated by statistical error until very large N
    total_mc = np.sqrt(statistical_error**2 + mc_roundoff**2)
    
    ax3.loglog(n_samples, statistical_error, color=COLORS['secondary'], linewidth=3,
              label='Statistical', alpha=0.9)
    ax3.loglog(n_samples, mc_roundoff, color=COLORS['accent'], linewidth=3,
              label='Round-off', alpha=0.9)
    ax3.loglog(n_samples, total_mc, color=COLORS['dark'], linewidth=4,
              label='Total error', alpha=0.9)
    
    ax3.set_xlabel('Sample size N', fontsize=12, color=COLORS['dark'], weight='medium')
    ax3.set_ylabel('Error', fontsize=12, color=COLORS['dark'], weight='medium')
    ax3.set_title('Monte Carlo Integration\n(Statistical Error)', fontsize=14,
                  color=COLORS['dark'], weight='medium')
    ax3.legend(fontsize=10, frameon=True, fancybox=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(10, 1e8)
    ax3.set_ylim(1e-8, 1)
    
    # Panel 4: Universal convergence patterns
    iterations = np.arange(1, 16)
    
    # Different convergence rates
    linear_conv = 0.5**iterations  # Bisection
    superlinear_conv = 0.1 * (0.3)**(iterations**(1.618/1.618))  # Secant approximation
    quadratic_conv = []
    error = 0.1
    for i in iterations:
        quadratic_conv.append(error)
        error = error**2 * 0.5  # Newton
        if error < 1e-16:
            error = 1e-16
    quadratic_conv = np.array(quadratic_conv)
    
    ax4.semilogy(iterations, linear_conv, 'o-', color=COLORS['secondary'], 
                linewidth=3, markersize=6, label='Linear (Bisection)', alpha=0.9)
    ax4.semilogy(iterations, superlinear_conv, 's-', color=COLORS['accent'], 
                linewidth=3, markersize=6, label='Superlinear (Secant)', alpha=0.9)
    ax4.semilogy(iterations, quadratic_conv, '^-', color=COLORS['primary'], 
                linewidth=3, markersize=6, label='Quadratic (Newton)', alpha=0.9)
    
    # Add integration methods for comparison
    # Simpson's rule: O(h⁴) error vs h
    h_int = 0.5**iterations  # Halving step size
    simpson_error = 0.1 * h_int**4
    ax4.semilogy(iterations, simpson_error, 'd-', color=COLORS['dark'], 
                linewidth=3, markersize=6, label='Quartic (Simpson)', alpha=0.9)
    
    ax4.set_xlabel('Iteration / Step Refinement', fontsize=14, color=COLORS['dark'], weight='medium')
    ax4.set_ylabel('Error', fontsize=14, color=COLORS['dark'], weight='medium')
    ax4.set_title('Universal Convergence Patterns Across Methods', fontsize=16,
                  color=COLORS['dark'], weight='medium')
    ax4.legend(fontsize=12, frameon=True, fancybox=True, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(1e-16, 1)
    
    # Panel 5: Condition number effects
    condition_numbers = [1, 10, 100, 1000]
    colors_cond = [COLORS['primary'], COLORS['accent'], COLORS['secondary'], 'red']
    
    for i, (kappa, color) in enumerate(zip(condition_numbers, colors_cond)):
        # Error amplification with condition number
        base_error = 1e-12
        amplified_error = base_error * kappa
        
        # Show as horizontal lines at different levels
        ax5.axhline(amplified_error, color=color, linewidth=3, alpha=0.8,
                   label=f'κ = {kappa}')
        
        # Add text annotation
        ax5.text(0.1, amplified_error * 1.5, f'κ = {kappa}', 
                fontsize=11, color=color, weight='medium')
    
    ax5.axhline(2.22e-16, color='black', linestyle='--', alpha=0.6,
               label='Machine ε')
    
    ax5.set_ylabel('Achievable Accuracy', fontsize=14, color=COLORS['dark'], weight='medium')
    ax5.set_title('Condition Number\nLimits Accuracy', fontsize=14,
                  color=COLORS['dark'], weight='medium')
    ax5.set_yscale('log')
    ax5.set_ylim(1e-16, 1e-6)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xlim(0, 1)
    ax5.set_xticks([])
    
    # Panel 6: Unified framework summary (text-based)
    ax6.text(0.5, 0.8, 'Universal Principles of Numerical Error Analysis', 
            ha='center', va='center', fontsize=18, weight='bold', 
            color=COLORS['dark'], transform=ax6.transAxes)
    
    principles = [
        '1. All methods balance TRUNCATION ERROR (approximation) vs ROUND-OFF ERROR (precision)',
        '2. Optimal parameters exist where these errors are balanced',
        '3. CONDITION NUMBERS determine sensitivity to perturbations',
        '4. Higher-order methods achieve faster convergence but may be less robust',
        '5. The same error analysis framework applies across all numerical methods'
    ]
    
    for i, principle in enumerate(principles):
        color = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
                COLORS['dark'], COLORS['neutral']][i]
        ax6.text(0.05, 0.6 - i*0.12, principle, ha='left', va='center', 
                fontsize=13, color=color, weight='medium', 
                transform=ax6.transAxes)
    
    # Add connecting arrows between different error types
    ax6.annotate('', xy=(0.3, 0.3), xytext=(0.15, 0.3),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=3),
                transform=ax6.transAxes)
    ax6.text(0.225, 0.35, 'Same\nTrade-off', ha='center', va='bottom',
            fontsize=12, weight='bold', color=COLORS['primary'],
            transform=ax6.transAxes)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    # Style all numerical axes
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        for spine in ax.spines.values():
            spine.set_color(COLORS['neutral'])
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=11, colors=COLORS['neutral'])
    
    plt.suptitle('The Unified Mathematics of Computational Error', 
                fontsize=20, color=COLORS['dark'], weight='medium', y=0.98)
    
    plt.tight_layout()
    save_figure(fig, "2_8_error_landscapes_unified")
    reset_style()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def get_available_figures():
    """Return dictionary of available figures with their IDs and function names"""
    figures = {
        # Part 1: Root Finding
        '2_1': figure_2_1_root_methods_comparison,
        '2_2': figure_2_2_newton_failure_modes,
        '2_3': figure_2_3_hybrid_method_performance,
        
        # Part 2: Quadrature
        '2_4': figure_2_4_integration_methods_geometry,
        '2_5': figure_2_5_simpson_vs_trapezoid_scaling,
        '2_6': figure_2_6_monte_carlo_convergence,
        '2_7': figure_2_7_integration_method_decision_tree,
        
        # Part 3: Synthesis
        '2_8': figure_2_8_error_landscapes_unified
    }
    return figures

def list_available_figures():
    """Print list of all available figures"""
    figures = get_available_figures()
    print("\nAvailable figures:")
    print("=" * 60)
    
    sections = {
        'Part 1: Root Finding': ['2_1', '2_2', '2_3'],
        'Part 2: Quadrature': ['2_4', '2_5', '2_6', '2_7'],
        'Part 3: Synthesis': ['2_8']
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
                        title = line
                        break
                if not title:
                    title = func.__name__.replace('figure_', '').replace('_', ' ').title()
            else:
                title = func.__name__.replace('figure_', '').replace('_', ' ').title()
            
            print(f"  {fig_id:<6} - {title}")
    
    print(f"\nUsage:")
    print(f"  python {sys.argv[0]} 2_1              # Generate specific figure")
    print(f"  python {sys.argv[0]} --figure 2_4     # Alternative syntax")
    print(f"  python {sys.argv[0]}                  # Generate all figures")

def main():
    parser = argparse.ArgumentParser(
        description="Generate educational figures for ASTR 596 Static Problems & Quadrature Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python static-quadrature-figures.py 2_1\n"
               "  python static-quadrature-figures.py --figure 2_4\n"
               "  python static-quadrature-figures.py --list\n"
               "  python static-quadrature-figures.py"
    )
    
    parser.add_argument('figure_id', nargs='?', help='Figure ID to generate (e.g., 2_1, 2_4)')
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