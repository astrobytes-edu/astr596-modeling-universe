#!/usr/bin/env python3
"""
Statistical Foundations Module - Complete Collection of Computational Demos
===========================================================================

This module contains all computational demos from the statistical foundations course,
organized by topic for easy exploration and learning.

Course: ASTR 596: Modeling the Universe
Module: Statistical Foundations - How Nature Computes

Contents:
1. Temperature Emergence from Particle Statistics  
2. Pressure from Chaotic Molecular Collisions
3. Central Limit Theorem in Action
4. Maximum Entropy Distributions
5. Correlation and Velocity Ellipsoids
6. Marginalization Visualization
7. Ergodic vs Non-Ergodic Systems
8. Law of Large Numbers Convergence
9. Error Propagation Through Calculations
10. Bayesian Learning and Inference
11. Monte Carlo π Estimation
12. Random Sampling Methods
13. Power Law Distribution Sampling
14. Plummer Sphere Spatial Sampling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create figures directory if it doesn't exist
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# 1. TEMPERATURE EMERGENCE FROM PARTICLE STATISTICS
# =============================================================================

def demo_temperature_emergence():
    """
    Demonstrate how temperature emerges as a meaningful concept only with
    multiple particles. Shows the Maxwell-Boltzmann distribution convergence.
    """
    print("=" * 60)
    print("1. TEMPERATURE EMERGENCE FROM PARTICLE STATISTICS")
    print("=" * 60)
    
    # Physical constants
    T_true = 300  # K
    m_H = 1.67e-24  # g (hydrogen mass)
    k_B = 1.38e-16  # erg/K
    sigma_true = np.sqrt(k_B * T_true / m_H) / 1e5  # Convert to km/s

    # Modern color palette - minimalist and professional
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose
        'accent': '#F18F01',     # Warm orange
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters with larger, more readable fonts
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,  # Increased from 10
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral'],
        'grid.color': colors['light'],
        'grid.alpha': 0.4
    })

    # Show how temperature becomes meaningful with more particles (powers of 10)
    N_values = [1, 10, 100, 1000, 10000, 100000]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), facecolor='white')
    fig.patch.set_facecolor('white')

    for idx, N in enumerate(N_values):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor('white')
        
        # Generate N particle velocities
        velocities = np.random.normal(0, sigma_true, N)
        
        # For N=1, can't define temperature
        if N == 1:
            ax.scatter([velocities[0]], [0], s=120, color=colors['accent'], 
                      alpha=0.9, zorder=3, edgecolors=colors['dark'], linewidth=1.5)
            ax.axhline(0, color=colors['neutral'], linestyle='-', alpha=0.5, linewidth=0.8)
            ax.set_title(f'$N = {N}$: No Temperature Concept\nSingle velocity: {velocities[0]:.1f} km/s', 
                        fontsize=14, pad=12, color=colors['dark'], weight='medium')
            ax.set_xlim(-3*sigma_true, 3*sigma_true)
            ax.set_ylim(-0.15, 0.15)
            ax.grid(True, alpha=0.3, linewidth=0.5)
        else:
            # Plot histogram with modern styling
            n_bins = min(25, max(8, N//20))  # Adaptive binning
            counts, bins, patches = ax.hist(velocities, bins=n_bins, density=True, 
                                           alpha=0.65, edgecolor='white', linewidth=0.8,
                                           color=colors['primary'], rwidth=0.9)
            
            # Overlay theoretical Maxwell-Boltzmann with elegant styling
            v_theory = np.linspace(-3*sigma_true, 3*sigma_true, 300)
            pdf_theory = (1/np.sqrt(2*np.pi*sigma_true**2)) * np.exp(-v_theory**2/(2*sigma_true**2))
            ax.plot(v_theory, pdf_theory, color=colors['secondary'], linewidth=2.5, 
                   label='Maxwell–Boltzmann\n$T = 300$ K', alpha=0.9)
            
            # Calculate "temperature" from variance
            T_measured = m_H * np.var(velocities) * 1e10 / k_B
            error = abs(T_measured - T_true) / T_true * 100
            
            # Proper mathematical notation for powers of 10
            if N == 10:
                N_str = '10'
            elif N == 100:
                N_str = '10^2'
            elif N == 1000:
                N_str = '10^3'
            elif N == 10000:
                N_str = '10^4'
            elif N == 100000:
                N_str = '10^5'
            else:
                N_str = str(N)
            
            ax.set_title(f'$N = {N_str}$: $T_{{\\rm measured}} = {T_measured:.0f}$ K\n(Error: {error:.1f}%)', 
                        fontsize=14, pad=12, color=colors['dark'], weight='medium')
            
            # Elegant legend
            ax.legend(fontsize=11, loc='upper right', frameon=True, 
                     fancybox=True, shadow=False, framealpha=0.95,
                     edgecolor=colors['neutral'], facecolor='white')
            ax.grid(True, alpha=0.25, linewidth=0.5)
        
        # Clean axis styling
        ax.set_xlabel('Velocity (km s$^{-1}$)', fontsize=12, color=colors['dark'], weight='medium')
        ax.set_ylabel('Probability Density', fontsize=12, color=colors['dark'], weight='medium')
        ax.tick_params(axis='both', which='major', labelsize=11, 
                      colors=colors['neutral'], width=0.8, length=4)
        
        # Set consistent x-axis limits
        ax.set_xlim(-3*sigma_true, 3*sigma_true)
        
        # Add subtle frame
        for spine in ax.spines.values():
            spine.set_color(colors['neutral'])
            spine.set_linewidth(0.8)

    # Professional title with proper spacing
    plt.suptitle('Temperature Emerges from Statistical Distributions', 
                fontsize=20, y=0.98, color=colors['dark'], weight='semibold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.45, wspace=0.30)
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_DIR, '01_temperature_emergence_from_statistics.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset matplotlib style to defaults for other functions
    plt.rcParams.update(plt.rcParamsDefault)

    # Interactive temperature demo
    print("\nInteractive Temperature Demonstration:")
    print("-" * 50)
    
    # Physical constants for hydrogen gas at 300 K
    sigma_theoretical = np.sqrt(k_B * T_true / m_H) / 1e5

    print(f"True temperature: {T_true} K")
    print(f"Theoretical velocity dispersion: {sigma_theoretical:.2f} km/s")
    print("-" * 50)

    # Test with different numbers of particles (matching figure)
    particle_counts = [10, 100, 1000, 10000, 100000]

    for N in particle_counts:
        # Generate random velocities from Maxwell-Boltzmann distribution
        velocities = np.random.normal(0, sigma_theoretical, N)
        
        # Calculate temperature from the variance of velocities
        # T = m * <v²> / k_B, where <v²> is the variance
        T_measured = m_H * np.var(velocities) * 1e10 / k_B
        
        # Calculate error
        error_percent = abs(T_measured - T_true) / T_true * 100
        
        # Format N with proper mathematical notation
        if N == 10:
            N_str = '10'
        elif N == 100:
            N_str = '10²'
        elif N == 1000:
            N_str = '10³'
        elif N == 10000:
            N_str = '10⁴'
        elif N == 100000:
            N_str = '10⁵'
        else:
            N_str = str(N)
        print(f"N = {N_str:>6} particles: T = {T_measured:6.1f} K (Error: {error_percent:4.1f}%)")

    print("-" * 50)
    print("Key insight: Temperature is only meaningful for collections of particles!")
    print("With more particles, our statistical estimate gets more accurate.")

# =============================================================================
# 2. PRESSURE FROM CHAOTIC MOLECULAR COLLISIONS
# =============================================================================

def create_simple_pressure_illustration():
    """
    Create a simple, elegant illustration of chaos → order averaging.
    """
    print("\n" + "=" * 60)
    print("2. SIMPLE PRESSURE ILLUSTRATION")  
    print("=" * 60)
    
    # Modern color palette
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'accent': '#F18F01',
        'neutral': '#6C757D',
        'light': '#F8F9FA',
        'dark': '#2D3436'
    }
    
    # Set modern style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })
    
    # Generate physically realistic collision data
    np.random.seed(42)
    N = 2000  # More collisions for better chaos visualization
    
    # Physical constants
    T = 300  # K (room temperature)
    m_H = 1.67e-24  # g (hydrogen mass)
    k_B = 1.38e-16  # erg/K
    
    # Generate velocities from Maxwell-Boltzmann distribution
    sigma_v = np.sqrt(k_B * T / m_H)  # velocity distribution width
    velocities = np.abs(np.random.normal(0, sigma_v, N))  # |v| for collisions
    momentum_transfers = 2 * m_H * velocities  # elastic collision: Δp = 2mv
    
    # Normalize for visualization
    momentum_norm = momentum_transfers / (m_H * sigma_v)  # dimensionless units
    
    # Create simple illustration
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    
    # Running average for smooth pressure line
    running_avg = np.cumsum(momentum_norm) / np.arange(1, N+1)
    
    # Plot many individual collisions with size proportional to momentum
    # Show more points but subsample for performance
    show_every = 5  # Show every 5th collision
    indices = range(0, N, show_every)
    collision_indices = [i for i in indices]
    collision_momenta = momentum_norm[::show_every]
    
    # Size proportional to momentum (with reasonable scaling)
    sizes = 15 + 50 * (collision_momenta / np.max(collision_momenta))  # Scale: 15-65 pixels
    
    ax.scatter(collision_indices, collision_momenta, s=sizes, alpha=0.5, 
              color=colors['neutral'], edgecolors='white', linewidth=0.5,
              label='Individual Collisions (Maxwell-Boltzmann)')
    
    # Plot smooth running average  
    ax.plot(range(N), running_avg, color=colors['primary'], linewidth=3.5, 
           label='Running Average → Steady Pressure', zorder=10)
    
    # Add theoretical average line (should match the data!)
    theoretical_avg = np.mean(momentum_norm)  # True average of our data
    ax.axhline(y=theoretical_avg, color=colors['secondary'], linestyle='--', 
              linewidth=2.5, alpha=0.9, label='True Average (Pressure)', zorder=10)
    
    # Clean, professional annotation - better positioned
    ax.text(0.25, 0.95, 'Individual molecular collisions\nfollow Maxwell-Boltzmann statistics', 
           transform=ax.transAxes, fontsize=11, color=colors['neutral'], 
           ha='center', va='top', bbox=dict(boxstyle='round,pad=0.5', 
           facecolor='white', edgecolor=colors['light'], alpha=0.9))
    
    ax.text(0.75, 0.35, 'Statistical averaging\n→ steady macroscopic pressure!', 
           transform=ax.transAxes, fontsize=12, color=colors['primary'], 
           ha='center', va='center', weight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['light'], 
           edgecolor=colors['primary'], alpha=0.9))
    
    ax.set_xlabel('Number of Collisions', fontsize=12, color=colors['dark'])
    ax.set_ylabel('Momentum Transfer', fontsize=12, color=colors['dark'])
    ax.set_title('How Molecular Chaos Becomes Steady Pressure', 
                fontsize=16, color=colors['dark'], weight='semibold', pad=20)
    
    ax.legend(fontsize=11, frameon=True, fancybox=True, 
             edgecolor=colors['neutral'], facecolor='white')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color(colors['neutral'])
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '02_simple_pressure_illustration.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset style
    plt.rcParams.update(plt.rcParamsDefault)

def demo_pressure_emergence():
    """
    Show how steady pressure emerges from chaotic molecular collisions
    through statistical averaging.
    """
    print("\n" + "=" * 60)
    print("2. PRESSURE FROM CHAOTIC MOLECULAR COLLISIONS")
    print("=" * 60)
    
    # Parameters
    N_molecules = 100000
    T = 300  # K
    m_H = 1.67e-24  # g (hydrogen mass)
    k_B = 1.38e-16  # erg/K
    sigma = np.sqrt(k_B * T / m_H)  # velocity distribution width

    # Generate random molecular collisions
    np.random.seed(42)  # for reproducibility
    collision_times = np.sort(np.random.uniform(0, 1, N_molecules))  
    collision_velocities = np.abs(np.random.normal(0, sigma, N_molecules))
    momentum_transfers = 2 * m_H * collision_velocities

    # Show the chaos: plot individual collisions
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Top: Individual collisions are random
    axes[0].scatter(collision_times[:1000], momentum_transfers[:1000], 
                    s=1, alpha=0.5)
    axes[0].set_ylabel('Momentum\nTransfer (g·cm/s)')
    axes[0].set_title('Individual Collisions: Pure Chaos')
    axes[0].set_xlim(0, 0.01)  # zoom in to see first 1% of collisions

    # Middle: Average over small time windows
    window_size = 100
    n_windows = N_molecules // window_size
    windowed_pressure = []
    window_times = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        avg_momentum = np.mean(momentum_transfers[start_idx:end_idx])
        windowed_pressure.append(avg_momentum)
        window_times.append(np.mean(collision_times[start_idx:end_idx]))

    axes[1].plot(window_times, windowed_pressure, 'b-', alpha=0.7)
    axes[1].axhline(y=np.mean(momentum_transfers), color='r', 
                    linestyle='--', label=f'True average')
    axes[1].set_ylabel('Average Momentum\nTransfer (g·cm/s)')
    axes[1].set_title(f'Averaged over {window_size} collisions: Fluctuations Shrink')
    axes[1].legend()

    # Bottom: Average over large time windows  
    window_size = 10000
    n_windows = N_molecules // window_size
    windowed_pressure = []
    window_times = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        avg_momentum = np.mean(momentum_transfers[start_idx:end_idx])
        windowed_pressure.append(avg_momentum)
        window_times.append(np.mean(collision_times[start_idx:end_idx]))

    axes[2].plot(window_times, windowed_pressure, 'g-', linewidth=2)
    axes[2].axhline(y=np.mean(momentum_transfers), color='r', 
                    linestyle='--', label=f'True average')
    axes[2].set_ylabel('Average Momentum\nTransfer (g·cm/s)')
    axes[2].set_title(f'Averaged over {window_size} collisions: Steady Pressure!')
    axes[2].set_xlabel('Time (arbitrary units)')
    axes[2].legend()

    plt.tight_layout()
    plt.suptitle('Order from Chaos: How Averaging Creates Pressure', 
                 y=1.02, fontsize=14)
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_DIR, '02_pressure_emergence_from_chaos.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

    # Quantitative analysis
    print("Chaos → Order through Averaging")
    print("-" * 40)
    individual_var = np.std(momentum_transfers)/np.mean(momentum_transfers)
    print(f"Individual collisions vary by: {individual_var:.1%}")

    for window_size in [10, 100, 1000, 10000]:
        n_windows = N_molecules // window_size
        window_avgs = []
        for i in range(n_windows):
            start = i * window_size
            end = (i + 1) * window_size
            window_avg = np.mean(momentum_transfers[start:end])
            window_avgs.append(window_avg)
        
        variation = np.std(window_avgs)/np.mean(window_avgs)
        print(f"Averaged over {window_size:5d}: vary by {variation:5.2%}")

# =============================================================================
# 3. CENTRAL LIMIT THEOREM IN ACTION
# =============================================================================

def demo_central_limit_theorem():
    """
    Demonstrate how non-Gaussian distributions become Gaussian through summation.
    Clean, focused version showing the key transformation.
    """
    print("\n" + "=" * 60)
    print("3. CENTRAL LIMIT THEOREM IN ACTION")
    print("=" * 60)
    
    # Modern color palette - matching other demos
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose
        'accent': '#16A085',     # Elegant teal instead of orange
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters with larger fonts for online readability
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,  # Increased from 12
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })

    # Create a cleaner 2x3 layout focusing on key transitions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    fig.patch.set_facecolor('white')

    # Original distribution: Exponential (very skewed!)
    np.random.seed(42)  # For reproducibility

    # Show key transitions: 1, 5, 20, 100, 500, 2000
    sum_sizes = [1, 5, 20, 100, 500, 2000]

    for idx, n_sum in enumerate(sum_sizes):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor('white')
        
        # Generate enough samples - ensure consistency for larger N
        # Use more samples for larger N to avoid binning artifacts
        if n_sum <= 20:
            n_realizations = 8000
        else:
            n_realizations = 6000  # Slightly fewer for computational efficiency but still robust
        
        # Generate sums of random variables
        sums = []
        for _ in range(n_realizations):
            # Use exponential distribution (highly skewed)
            sample = np.random.exponential(scale=1.0, size=n_sum)
            sums.append(np.sum(sample))
        
        # Standardize: subtract mean, divide by std
        sums = np.array(sums)
        expected_mean = n_sum * 1.0  # E[X] = 1 for exp(1)
        expected_std = np.sqrt(n_sum) * 1.0  # Var[X] = 1 for exp(1)
        sums_standardized = (sums - expected_mean) / expected_std
        
        # Plot histogram with beautiful styling
        n_bins = min(30, max(15, len(sums_standardized)//60))
        ax.hist(sums_standardized, bins=n_bins, density=True, 
                alpha=0.7, color=colors['primary'], edgecolor='white', 
                linewidth=0.5, rwidth=0.95)
        
        # Overlay theoretical Gaussian
        x = np.linspace(-4, 4, 200)
        gaussian = stats.norm.pdf(x, 0, 1)
        ax.plot(x, gaussian, color=colors['secondary'], linewidth=3, 
               label='Standard Gaussian', alpha=0.9, zorder=10)
        
        # Calculate goodness of fit - Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.kstest(sums_standardized, 'norm')
        fit_quality = 1 - ks_statistic  # Convert to "goodness" (0=bad, 1=perfect)
        
        # Intelligent titles with fit quality metrics
        fit_percent = fit_quality * 100
        
        if n_sum == 1:
            ax.set_title(f'$N = 1$: Exponential (Fit: {fit_percent:.1f}%)', 
                        fontsize=16, pad=18, color=colors['dark'], weight='medium')  # Larger titles
            ax.annotate('Very skewed!', xy=(2, 0.25), xytext=(2.5, 0.35),
                       arrowprops=dict(arrowstyle='->', color=colors['accent'], lw=1.5),
                       fontsize=13, color=colors['accent'], weight='bold')  # Larger annotations
        elif n_sum == 5:
            ax.set_title(f'$N = 5$: Starting to Center (Fit: {fit_percent:.1f}%)', 
                        fontsize=16, pad=18, color=colors['dark'], weight='medium')
        elif n_sum == 20:
            ax.set_title(f'$N = 20$: Bell Shape Emerging (Fit: {fit_percent:.1f}%)', 
                        fontsize=16, pad=18, color=colors['dark'], weight='medium')
        elif n_sum == 100:
            ax.set_title(f'$N = 100$: Nearly Gaussian! (Fit: {fit_percent:.1f}%)', 
                        fontsize=16, pad=18, color=colors['dark'], weight='medium')
        elif n_sum == 500:
            ax.set_title(f'$N = 500$: Excellent Match (Fit: {fit_percent:.1f}%)', 
                        fontsize=16, pad=18, color=colors['dark'], weight='medium')
        else:  # 2000
            ax.set_title(f'$N = 2000$: Perfect Gaussian! (Fit: {fit_percent:.1f}%)', 
                        fontsize=16, pad=18, color=colors['dark'], weight='medium')
            ax.annotate('Perfect!', xy=(-0.5, 0.35), xytext=(-2, 0.35),
                       arrowprops=dict(arrowstyle='->', color=colors['accent'], lw=1.5),
                       fontsize=13, color=colors['accent'], weight='bold')
        
        # Consistent axis limits and styling
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.45)
        ax.set_xlabel('Standardized Value', fontsize=13, color=colors['dark'])  # Larger labels
        ax.set_ylabel('Density', fontsize=13, color=colors['dark'])
        ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])  # Larger tick labels
        ax.grid(True, alpha=0.2, linewidth=0.5)
        
        # Add legend to top-right panel
        if idx == 2:  # Top right
            ax.legend(fontsize=13, loc='upper right', frameon=True,  # Larger legend
                     fancybox=True, edgecolor=colors['neutral'], 
                     facecolor='white', framealpha=0.95)
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_color(colors['neutral'])
            spine.set_linewidth(0.8)

    # Professional main title - larger for online readability
    plt.suptitle('Central Limit Theorem: Any Distribution → Gaussian', 
                fontsize=22, y=0.95, color=colors['dark'], weight='semibold')  # Increased from 18
    
    # Add educational subtitle - larger, with practical insight
    fig.text(0.5, 0.91, 'Practical convergence by N≈500 • Further increases give diminishing returns',
             ha='center', va='center', fontsize=15, color=colors['neutral'],  # Increased from 13
             style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.83, hspace=0.35, wspace=0.25)
    
    # Save with high quality
    plt.savefig(os.path.join(FIGURES_DIR, '03_central_limit_theorem_in_action.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)

# =============================================================================
# 4. MAXIMUM ENTROPY DISTRIBUTIONS
# =============================================================================

def demo_maximum_entropy():
    """
    Show why Maxwell-Boltzmann distributions emerge naturally from maximum entropy principle.
    This is the deep reason why we see exponential energy distributions in stellar atmospheres.
    """
    print("\n" + "=" * 60)
    print("4. MAXIMUM ENTROPY DISTRIBUTIONS")
    print("=" * 60)
    
    # Modern color palette - consistent with other figures
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose  
        'accent': '#16A085',     # Elegant teal
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })
    
    # Create focused figure showing the key insight
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Physical setup: stellar atmosphere with fixed temperature
    mean_energy = 3.0  # kT units
    x = np.linspace(0, 12, 1000)
    
    # Panel 1: Biased assumption - all particles at mean energy
    ax = axes[0]
    ax.set_facecolor('white')
    ax.axvline(mean_energy, color=colors['secondary'], linewidth=6, alpha=0.9, 
              label='All particles identical')
    ax.axvline(mean_energy, color='white', linewidth=2, alpha=0.8)  # White line on top
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel('Energy (units of kT)', fontsize=15, color=colors['dark'])
    ax.set_ylabel('Probability Density', fontsize=15, color=colors['dark'])
    ax.set_title('Biased Assumption\n"All particles have same energy"', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    
    # Add text annotation - moved to the right
    ax.text(0.65, 0.8, 'Entropy = 0\n(Maximum bias!)', transform=ax.transAxes,
           fontsize=16, ha='center', va='center', color=colors['secondary'], 
           weight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
           edgecolor=colors['secondary'], alpha=0.95))
    
    # Panel 2: Maximum entropy - exponential/Boltzmann
    ax = axes[1]  
    ax.set_facecolor('white')
    kT = mean_energy  # Use consistent energy scale: kT = 3 units
    # Properly normalized exponential distribution: p(E) = (1/kT) * exp(-E/kT)
    boltzmann = (1/kT) * np.exp(-x/kT)  
    ax.plot(x, boltzmann, color=colors['accent'], linewidth=4.5, alpha=0.95,
           label='Maxwell-Boltzmann')
    ax.fill_between(x, boltzmann, alpha=0.3, color=colors['accent'])
    
    # Correct entropy for exponential: S = 1 + ln(kT) 
    entropy_boltzmann = 1 + np.log(kT)  # = 1 + ln(3) ≈ 2.10
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('Energy (units of kT)', fontsize=15, color=colors['dark'])
    ax.set_ylabel('Probability Density', fontsize=15, color=colors['dark'])
    ax.set_title('Maximum Entropy Solution\n"Least biased distribution"', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    
    # Add key insight
    ax.text(0.6, 0.75, f'Entropy = {entropy_boltzmann:.2f}\n(Maximum possible!)', 
           transform=ax.transAxes, fontsize=16, ha='center', va='center', 
           color=colors['accent'], weight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
           edgecolor=colors['accent'], alpha=0.95))
    
    # Panel 3: Why this matters - astrophysical applications
    ax = axes[2]
    ax.set_facecolor('white')
    
    # Show how this connects to stellar atmospheres - energy level populations
    energies = np.array([0, 1, 2, 3, 4, 5])  # Include ground state (E=0)
    kT_stellar = 2.0  # Stellar atmosphere temperature scale
    # Boltzmann factor: n_i/n_0 = exp(-(E_i - E_0)/kT) = exp(-E_i/kT) for E_0=0
    populations = np.exp(-energies/kT_stellar)
    populations = populations / populations[0]  # Normalize to ground state
    
    bars = ax.bar(energies, populations, color=colors['primary'], alpha=0.8, 
                 edgecolor='white', linewidth=2, width=0.6)
    
    # Theoretical curve showing exponential decay
    x_fine = np.linspace(-0.3, 5.3, 100)
    theory = np.exp(-x_fine/kT_stellar)
    ax.plot(x_fine, theory, color=colors['secondary'], linewidth=4, alpha=0.9,
           linestyle='--', label='Boltzmann Law')
    
    ax.set_xlabel('Energy Level (above ground state)', fontsize=15, color=colors['dark'])
    ax.set_ylabel('Population Ratio (n/n₀)', fontsize=15, color=colors['dark'])
    ax.set_title('Stellar Atmospheres\n"Why we see exponential populations"', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    ax.set_xticks(energies)
    ax.set_ylim(0, 1.1)
    
    # Add physical insight
    ax.text(0.5, 0.85, 'This explains:\n• Spectral line ratios\n• Ionization fractions\n• Opacity sources', 
           transform=ax.transAxes, fontsize=14, ha='center', va='top', 
           color=colors['primary'], weight='medium',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['light'], 
           edgecolor=colors['primary'], alpha=0.95))
    
    # Style all axes
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(colors['neutral'])
            spine.set_linewidth(1.0)
    
    # Professional main title with optimal white space
    plt.suptitle('Maximum Entropy Principle: Why Stellar Atmospheres Follow Maxwell-Boltzmann', 
                fontsize=19, y=0.95, color=colors['dark'], weight='semibold')
    
    # Educational subtitle with good spacing  
    fig.text(0.5, 0.88, 'Given only mean energy (temperature), nature chooses the least biased distribution → exponential energy populations',
             ha='center', va='center', fontsize=14, color=colors['neutral'], style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.68, wspace=0.38, hspace=0.20)
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_DIR, '04_maximum_entropy_distributions.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)

# =============================================================================
# 5. CORRELATION AND VELOCITY ELLIPSOIDS
# =============================================================================

def demo_correlation():
    """
    Demonstrate correlation and velocity ellipsoids - fundamental for stellar dynamics.
    Shows how correlation shapes velocity distributions in stellar systems.
    """
    print("\n" + "=" * 60)
    print("5. CORRELATION AND VELOCITY ELLIPSOIDS")
    print("=" * 60)
    
    # Modern color palette - consistent with other figures
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose  
        'accent': '#16A085',     # Elegant teal
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })
    
    def draw_ellipse(ax, cov, center=(0, 0), n_std=2, color='red', alpha=0.3, linewidth=2):
        """Draw confidence ellipse from covariance matrix"""
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigenvals)
        ellipse = plt.matplotlib.patches.Ellipse(center, width, height, angle=angle,
                                                facecolor=color, alpha=alpha, 
                                                edgecolor=color, linewidth=linewidth)
        ax.add_patch(ellipse)
        return width, height, angle
    
    # Create sophisticated figure showing velocity ellipsoids
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1, 1],
                         hspace=0.35, wspace=0.3)
    
    # Define correlation values and their astrophysical context
    correlations = [0.0, 0.5, 0.8, 0.95]
    context_labels = ['Isotropic\n(Spherical Halo)', 'Weak Streaming\n(Thick Disk)', 
                     'Strong Stream\n(Thin Disk)', 'Tidal Stream\n(Disrupted System)']
    panel_colors = [colors['neutral'], colors['accent'], colors['primary'], colors['secondary']]
    
    # Top row: Scatter plots with ellipsoids
    for i, (rho, context, color) in enumerate(zip(correlations, context_labels, panel_colors)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('white')
        
        # Create realistic covariance matrix
        sigma_x, sigma_y = 40, 30  # Different dispersions (km/s)
        cov = [[sigma_x**2, rho * sigma_x * sigma_y], 
               [rho * sigma_x * sigma_y, sigma_y**2]]
        mean = [0, 0]
        
        # Generate stellar velocities
        np.random.seed(42 + i)  # Reproducible
        n_stars = 1500
        velocities = np.random.multivariate_normal(mean, cov, n_stars)
        vx, vy = velocities.T
        
        # Create scatter plot with speed coloring
        speeds = np.sqrt(vx**2 + vy**2)
        scatter = ax.scatter(vx, vy, c=speeds, s=12, alpha=0.6, cmap='viridis_r', 
                           edgecolors='none', rasterized=True)
        
        # Draw confidence ellipsoids
        for n_std, alpha, lw in [(3, 0.15, 1.5), (2, 0.25, 2), (1, 0.4, 2.5)]:
            draw_ellipse(ax, cov, n_std=n_std, color=color, alpha=alpha, linewidth=lw)
        
        # Styling
        ax.set_xlabel('$v_x$ (km/s)', fontsize=14, color=colors['dark'], weight='medium')
        ax.set_ylabel('$v_y$ (km/s)', fontsize=14, color=colors['dark'], weight='medium')
        ax.set_title(f'{context}\n$\\rho$ = {rho}', fontsize=15, pad=15, 
                    color=colors['dark'], weight='medium')
        ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_xlim(-150, 150)
        ax.set_ylim(-120, 120)
        ax.set_aspect('equal')
        
        # Add measured correlation
        measured_rho = np.corrcoef(vx, vy)[0, 1]
        ax.text(0.02, 0.98, f'Measured: {measured_rho:.2f}', 
               transform=ax.transAxes, fontsize=12, color=colors['dark'],
               verticalalignment='top', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
               edgecolor=color, alpha=0.9))
    
    # Bottom row: Principal component analysis and ellipse parameters
    for i, (rho, context, color) in enumerate(zip(correlations, context_labels, panel_colors)):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor('white')
        
        # Covariance matrix analysis
        sigma_x, sigma_y = 40, 30
        cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y], 
                       [rho * sigma_x * sigma_y, sigma_y**2]])
        
        # Eigenanalysis for principal axes
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        major_axis = np.sqrt(eigenvals[1])  # Larger eigenvalue
        minor_axis = np.sqrt(eigenvals[0])  # Smaller eigenvalue
        angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
        
        # Draw the ellipse and axes
        theta = np.linspace(0, 2*np.pi, 100)
        ellipse_x = 2 * major_axis * np.cos(theta)
        ellipse_y = 2 * minor_axis * np.sin(theta)
        
        # Rotate ellipse
        cos_angle = np.cos(np.radians(angle))
        sin_angle = np.sin(np.radians(angle))
        x_rot = ellipse_x * cos_angle - ellipse_y * sin_angle
        y_rot = ellipse_x * sin_angle + ellipse_y * cos_angle
        
        ax.plot(x_rot, y_rot, color=color, linewidth=3, alpha=0.9)
        ax.fill(x_rot, y_rot, color=color, alpha=0.2)
        
        # Draw principal axes
        major_vec = eigenvecs[:, 1] * major_axis * 2
        minor_vec = eigenvecs[:, 0] * minor_axis * 2
        
        ax.arrow(0, 0, major_vec[0], major_vec[1], head_width=8, head_length=10, 
                fc=colors['dark'], ec=colors['dark'], linewidth=2, alpha=0.8)
        ax.arrow(0, 0, minor_vec[0], minor_vec[1], head_width=8, head_length=10, 
                fc=colors['neutral'], ec=colors['neutral'], linewidth=2, alpha=0.8)
        
        # Labels and statistics
        ax.set_xlabel('Principal Axis 1', fontsize=13, color=colors['dark'], weight='medium')
        ax.set_ylabel('Principal Axis 2', fontsize=13, color=colors['dark'], weight='medium')
        ax.set_title(f'Ellipse Parameters\nAxis Ratio: {major_axis/minor_axis:.2f}', 
                    fontsize=14, pad=15, color=colors['dark'], weight='medium')
        ax.tick_params(axis='both', labelsize=11, colors=colors['neutral'])
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_aspect('equal')
        
        # Set limits based on major axis
        lim = major_axis * 2.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        
        # Add angle annotation
        ax.text(0.02, 0.02, f'Angle: {angle:.1f}°', 
               transform=ax.transAxes, fontsize=11, color=colors['dark'],
               bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
               edgecolor=colors['neutral'], alpha=0.9))
    
    # Bottom: Summary analysis panel
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    # Create table of ellipse properties
    summary_text = """
    **Velocity Ellipsoids in Stellar Systems:** Correlation fundamentally shapes the geometry of velocity distributions.
    
    • **Isotropic (ρ=0):** Circular distribution - no preferred direction (halo stars, globular clusters)
    • **Moderate (ρ=0.5):** Elliptical - mild asymmetric drift (thick disk populations)  
    • **Strong (ρ=0.8):** Highly elongated - significant streaming motion (thin disk, spiral arms)
    • **Extreme (ρ≥0.95):** Nearly linear - coherent flows (stellar streams, tidal debris)
    
    **Key Insight:** The shape encodes the dynamical history - relaxed systems are round, disturbed systems are elongated.
    """
    
    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14, 
                   color=colors['dark'], style='italic', weight='medium',
                   bbox=dict(boxstyle='round,pad=1.0', facecolor=colors['light'], 
                   edgecolor=colors['neutral'], alpha=0.95, linewidth=1))
    
    # Style all axes
    for ax in fig.get_axes():
        if ax.get_subplotspec() is not None:
            for spine in ax.spines.values():
                spine.set_color(colors['neutral'])
                spine.set_linewidth(0.8)
    
    # Professional main title
    plt.suptitle('Correlation and Velocity Ellipsoids: How Stellar Dynamics Shape Observable Distributions', 
                fontsize=22, y=0.97, color=colors['dark'], weight='semibold')
    
    # Educational subtitle
    fig.text(0.5, 0.93, 'From isotropic halos to streaming disk populations • Correlation encodes dynamical history',
             ha='center', va='center', fontsize=16, color=colors['neutral'], style='italic')
    
    # Save with high quality
    plt.savefig(os.path.join(FIGURES_DIR, '05_correlation_and_velocity_ellipsoids.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Quantitative analysis
    print("\n🎯 Velocity Ellipsoid Analysis:")
    print("-" * 70)
    for i, (rho, context) in enumerate(zip(correlations, context_labels)):
        sigma_x, sigma_y = 40, 30
        cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y], 
                       [rho * sigma_x * sigma_y, sigma_y**2]])
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        major_axis = np.sqrt(eigenvals[1])
        minor_axis = np.sqrt(eigenvals[0])
        angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
        
        print(f"{context:20s} (ρ={rho:4.2f}): "
              f"axes = {major_axis:5.1f}:{minor_axis:5.1f} km/s, "
              f"ratio = {major_axis/minor_axis:4.2f}, "
              f"angle = {angle:6.1f}°")
    
    print("\n✓ Correlation shapes velocity ellipsoids fundamentally!")
    print("✓ Essential for: stellar dynamics, galactic archaeology, N-body simulations")
    print("✓ Ellipsoid shape reveals dynamical state and evolutionary history")

# =============================================================================
# 6. MARGINALIZATION VISUALIZATION
# =============================================================================

def demo_marginalization():
    """
    Demonstrate marginalization - the art of extracting 1D information from multi-dimensional distributions.
    Shows stellar velocity applications with modern astrophysical context.
    """
    print("\n" + "=" * 60)
    print("6. MARGINALIZATION VISUALIZATION")
    print("=" * 60)
    
    # Modern color palette - consistent with other figures
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose  
        'accent': '#16A085',     # Elegant teal
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })
    
    # Create comprehensive figure showing marginalization in astrophysical context
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Create custom grid layout for better aesthetics
    gs = fig.add_gridspec(3, 4, width_ratios=[3, 1, 0.2, 1.2], height_ratios=[1, 3, 0.8],
                         hspace=0.3, wspace=0.3)
    
    # ========== Panel 1: Main joint distribution ==========
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_joint.set_facecolor('white')
    
    # Create stellar velocity distribution (more realistic)
    # Represent v_x and v_y velocities with correlation (streaming motion)
    mean_vel = [15, 25]  # km/s - typical stellar velocities
    cov_vel = [[120, 60], [60, 180]]  # km/s covariance with correlation
    rv = stats.multivariate_normal(mean_vel, cov_vel)
    
    # Create high-resolution grid
    v_x = np.linspace(-40, 70, 120)
    v_y = np.linspace(-30, 80, 120)
    V_X, V_Y = np.meshgrid(v_x, v_y)
    pos = np.dstack((V_X, V_Y))
    Z = rv.pdf(pos)
    
    # Beautiful contour plot with custom colormap
    levels = 20
    contour = ax_joint.contourf(V_X, V_Y, Z, levels=levels, cmap='viridis', alpha=0.9)
    contour_lines = ax_joint.contour(V_X, V_Y, Z, levels=8, colors='white', alpha=0.6, linewidths=1)
    
    ax_joint.set_xlabel('Velocity $v_x$ (km/s)', fontsize=15, color=colors['dark'], weight='medium')
    ax_joint.set_ylabel('Velocity $v_y$ (km/s)', fontsize=15, color=colors['dark'], weight='medium')
    ax_joint.set_title('Joint Distribution $P(v_x, v_y)$\nStellar Velocity Distribution', 
                      fontsize=18, pad=20, color=colors['dark'], weight='medium')
    ax_joint.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    ax_joint.grid(True, alpha=0.3, linewidth=0.8)
    
    # ========== Panel 2: X marginal (top) ==========
    ax_margx = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_margx.set_facecolor('white')
    
    # Calculate marginal by integration
    marginal_vx = np.trapz(Z, v_y, axis=0)
    marginal_vx = marginal_vx / np.trapz(marginal_vx, v_x)
    
    # Plot with modern styling
    ax_margx.plot(v_x, marginal_vx, color=colors['primary'], linewidth=3.5, alpha=0.9)
    ax_margx.fill_between(v_x, marginal_vx, alpha=0.4, color=colors['primary'], edgecolor='none')
    
    ax_margx.set_ylabel('$P(v_x)$', fontsize=15, color=colors['dark'], weight='medium')
    ax_margx.set_title('$P(v_x) = \\int P(v_x, v_y) dv_y$\nRadial Velocity Distribution', 
                      fontsize=16, pad=20, color=colors['dark'], weight='medium')
    ax_margx.tick_params(axis='x', labelbottom=False)
    ax_margx.tick_params(axis='y', labelsize=13, colors=colors['neutral'])
    ax_margx.grid(True, alpha=0.3, linewidth=0.8)
    
    # Add statistical info
    mean_vx = np.trapz(v_x * marginal_vx, v_x)
    std_vx = np.sqrt(np.trapz((v_x - mean_vx)**2 * marginal_vx, v_x))
    ax_margx.axvline(mean_vx, color=colors['secondary'], linewidth=2.5, linestyle='--', alpha=0.8)
    ax_margx.text(0.02, 0.85, f'μ = {mean_vx:.1f} km/s\nσ = {std_vx:.1f} km/s', 
                 transform=ax_margx.transAxes, fontsize=12, color=colors['dark'],
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                 edgecolor=colors['neutral'], alpha=0.9))
    
    # ========== Panel 3: Y marginal (right) ==========
    ax_margy = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    ax_margy.set_facecolor('white')
    
    # Calculate marginal by integration
    marginal_vy = np.trapz(Z, v_x, axis=1)
    marginal_vy = marginal_vy / np.trapz(marginal_vy, v_y)
    
    # Plot with modern styling
    ax_margy.plot(marginal_vy, v_y, color=colors['secondary'], linewidth=3.5, alpha=0.9)
    ax_margy.fill_betweenx(v_y, marginal_vy, alpha=0.4, color=colors['secondary'], edgecolor='none')
    
    ax_margy.set_xlabel('$P(v_y)$', fontsize=15, color=colors['dark'], weight='medium')
    ax_margy.set_title('$P(v_y) = \\int P(v_x, v_y) dv_x$\nTangential Component', 
                      fontsize=16, pad=20, color=colors['dark'], weight='medium')
    ax_margy.tick_params(axis='y', labelleft=False)
    ax_margy.tick_params(axis='x', labelsize=13, colors=colors['neutral'])
    ax_margy.grid(True, alpha=0.3, linewidth=0.8)
    
    # Add statistical info
    mean_vy = np.trapz(v_y * marginal_vy, v_y)
    std_vy = np.sqrt(np.trapz((v_y - mean_vy)**2 * marginal_vy, v_y))
    ax_margy.axhline(mean_vy, color=colors['primary'], linewidth=2.5, linestyle='--', alpha=0.8)
    ax_margy.text(0.05, 0.02, f'μ = {mean_vy:.1f} km/s\nσ = {std_vy:.1f} km/s', 
                 transform=ax_margy.transAxes, fontsize=12, color=colors['dark'],
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                 edgecolor=colors['neutral'], alpha=0.9))
    
    # ========== Panel 4: Colorbar ==========
    ax_cb = fig.add_subplot(gs[1, 2])
    cbar = plt.colorbar(contour, cax=ax_cb, label='')
    cbar.set_label('Probability Density\n(s²/km²)', fontsize=13, color=colors['dark'], weight='medium')
    cbar.ax.tick_params(labelsize=12, colors=colors['neutral'])
    
    # ========== Panel 5: Astrophysical application example ==========
    ax_app = fig.add_subplot(gs[1, 3])
    ax_app.set_facecolor('white')
    
    # Show practical example: line-of-sight velocity measurements
    np.random.seed(42)
    n_stars = 500
    # Sample from the joint distribution
    stellar_velocities = rv.rvs(n_stars)
    v_los = stellar_velocities[:, 0]  # Line-of-sight component (marginalized!)
    
    # Create histogram of observed velocities
    bins = np.linspace(-40, 70, 25)
    n, bins_edges, patches = ax_app.hist(v_los, bins=bins, density=True, alpha=0.7, 
                                        color=colors['accent'], edgecolor='white', 
                                        linewidth=1)
    
    # Overlay theoretical marginal
    ax_app.plot(v_x, marginal_vx, color=colors['dark'], linewidth=3, alpha=0.9,
               label='Theoretical P(v_x)')
    
    ax_app.set_xlabel('Line-of-sight Velocity (km/s)', fontsize=13, color=colors['dark'], weight='medium')
    ax_app.set_ylabel('Normalized Count', fontsize=13, color=colors['dark'], weight='medium')
    ax_app.set_title('Observed Stellar Velocities\n(Marginalization in Action)', 
                    fontsize=16, pad=20, color=colors['dark'], weight='medium')
    ax_app.legend(fontsize=12, frameon=True, fancybox=True,
                 edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax_app.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
    ax_app.grid(True, alpha=0.3, linewidth=0.8)
    
    # Add annotation
    ax_app.text(0.05, 0.85, f'N = {n_stars} stars\nOnly v_x measured\n(v_y unobservable)', 
               transform=ax_app.transAxes, fontsize=11, color=colors['dark'],
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
               edgecolor=colors['accent'], alpha=0.9))
    
    # ========== Panel 6: Educational explanation ==========
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')
    ax_text.text(0.5, 0.5, 
                'Marginalization extracts observable 1D distributions from unobservable multi-dimensional reality.\n'
                'Example: Stellar surveys measure only line-of-sight velocities, automatically marginalizing over perpendicular components.\n'
                'Key insight: The shape we observe (1D marginal) depends on correlations in the full distribution.',
                ha='center', va='center', fontsize=14, color=colors['dark'], style='italic',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=colors['light'], 
                edgecolor=colors['neutral'], alpha=0.95, linewidth=1))
    
    # Style all axes
    for ax in [ax_joint, ax_margx, ax_margy, ax_app]:
        for spine in ax.spines.values():
            spine.set_color(colors['neutral'])
            spine.set_linewidth(0.8)
    
    # Professional main title
    plt.suptitle('Marginalization: The Art of Extracting 1D Information from Multi-dimensional Reality', 
                fontsize=20, y=0.96, color=colors['dark'], weight='semibold')
    
    # Educational subtitle
    fig.text(0.5, 0.92, 'From 2D stellar velocity distributions to 1D observable quantities • Integration as information extraction',
             ha='center', va='center', fontsize=15, color=colors['neutral'], style='italic')
    
    # Save with high quality
    plt.savefig(os.path.join(FIGURES_DIR, '06_marginalization_visualization.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Quantitative analysis with astrophysical context
    print("\n🎯 Marginalization Analysis - Stellar Velocities:")
    print("-" * 70)
    print(f"Joint distribution peak: v_x = {mean_vel[0]:.1f} km/s, v_y = {mean_vel[1]:.1f} km/s")
    print(f"Radial velocity marginal:     μ = {mean_vx:.1f} km/s, σ = {std_vx:.1f} km/s") 
    print(f"Tangential velocity marginal: μ = {mean_vy:.1f} km/s, σ = {std_vy:.1f} km/s")
    print(f"Velocity correlation: ρ = {cov_vel[0][1]/np.sqrt(cov_vel[0][0]*cov_vel[1][1]):.2f}")
    print(f"Sample verification: {n_stars} stars, observed <v_x> = {np.mean(v_los):.1f} km/s")
    
    print("\n✓ Marginalization lets us extract 1D observables from multi-D reality!")
    print("✓ Essential for: spectroscopy, proper motions, luminosity functions")
    print("✓ What we measure ≠ what exists - marginalization bridges this gap")

# =============================================================================
# 7. ERGODIC VS NON-ERGODIC SYSTEMS
# =============================================================================

def demo_ergodicity():
    """
    Demonstrate ergodicity: why MCMC sampling works and thermal equilibrium is reachable.
    Shows the fundamental principle that time averages equal ensemble averages.
    """
    print("\n" + "=" * 60)
    print("7. ERGODICITY: THE FOUNDATION OF MCMC AND THERMAL EQUILIBRIUM")
    print("=" * 60)
    
    # Color scheme matching other figures
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose  
        'accent': '#16A085',     # Elegant teal
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'font.family': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
    })
    
    # Create sophisticated figure showing ergodicity in action
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Create grid layout with more generous spacing
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6], width_ratios=[1, 1, 1],
                         hspace=0.55, wspace=0.35, top=0.85, bottom=0.08)
    
    # === TOP ROW: Maxwell-Boltzmann Distribution Sampling ===
    # Show how MCMC samples produce the correct equilibrium distribution
    
    # Physical parameters for hydrogen gas at room temperature
    T = 300  # Kelvin
    m_H = 1.67e-24  # grams
    k_B = 1.38e-16  # erg/K
    sigma_v = np.sqrt(k_B * T / m_H)  # theoretical velocity width
    
    # Time evolution of velocity sampling
    time_steps = [100, 1000, 10000]
    sample_colors = [colors['accent'], colors['primary'], colors['secondary']]
    
    for i, (n_samples, color) in enumerate(zip(time_steps, sample_colors)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('white')
        
        # Generate cumulative samples (as MCMC would)
        np.random.seed(42)
        velocities = np.random.normal(0, sigma_v, n_samples)
        
        # Histogram of samples
        counts, bins, _ = ax.hist(velocities/1e5, bins=40, density=True, alpha=0.7, 
                                 color=color, edgecolor='white', linewidth=0.8)
        
        # Theoretical Maxwell-Boltzmann (1D)
        v_range = np.linspace(-4*sigma_v, 4*sigma_v, 1000)
        mb_theory = (1/np.sqrt(2*np.pi*sigma_v**2)) * np.exp(-0.5*(v_range**2/sigma_v**2))
        ax.plot(v_range/1e5, mb_theory*1e5, '--', linewidth=3, color=colors['dark'], 
               label='Maxwell-Boltzmann\nTheory', alpha=0.9)
        
        # Calculate fit quality
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        theory_at_bins = (1/np.sqrt(2*np.pi*sigma_v**2)) * np.exp(-0.5*((bin_centers*1e5)**2/sigma_v**2)) * 1e5
        chi_sq = np.sum((counts - theory_at_bins)**2 / theory_at_bins)
        fit_quality = max(0, 100 - chi_sq)
        
        # Styling and annotations
        ax.set_xlabel('Velocity (km/s)', fontsize=14, color=colors['dark'], weight='medium')
        if i == 0:
            ax.set_ylabel('Probability Density', fontsize=14, color=colors['dark'], weight='medium')
        ax.set_title(f'After {n_samples:,} MCMC Steps\nFit Quality: {fit_quality:.1f}%', 
                    fontsize=15, pad=15, color=colors['dark'], weight='medium')
        
        ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, max(counts)*1.2)
        
        if i == 2:  # Only show legend on rightmost plot
            ax.legend(frameon=False, fontsize=11, loc='upper right')
    
    # === MIDDLE ROW: Time vs Ensemble Averages ===
    # Demonstrate the ergodic principle directly
    
    # Left: Time average convergence
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor('white')
    
    # Generate long time series
    np.random.seed(42)
    long_series = np.random.normal(0, sigma_v, 50000)
    time_points = np.arange(1, len(long_series)+1)
    
    # Calculate cumulative time average of kinetic energy
    cumulative_avg = np.cumsum(0.5 * m_H * long_series**2) / time_points
    theoretical_avg = 0.5 * k_B * T  # <E> = (1/2)kT for 1D
    
    # Plot convergence
    ax.semilogx(time_points[::50], cumulative_avg[::50], linewidth=2.5, 
               color=colors['primary'], label='Time Average', alpha=0.8)
    ax.axhline(theoretical_avg, linestyle='--', linewidth=3, color=colors['secondary'], 
              label='Theoretical\nEnsemble Average', alpha=0.9)
    
    # Shade convergence region
    tolerance = 0.05 * theoretical_avg
    ax.fill_between([100, 50000], theoretical_avg - tolerance, theoretical_avg + tolerance,
                   alpha=0.15, color=colors['secondary'], label='±5% Tolerance')
    
    ax.set_xlabel('Time Steps', fontsize=14, color=colors['dark'], weight='medium')
    ax.set_ylabel('⟨Kinetic Energy⟩ (erg)', fontsize=14, color=colors['dark'], weight='medium')
    ax.set_title('Ergodic Convergence:\nTime Average → Ensemble Average', 
                fontsize=15, pad=15, color=colors['dark'], weight='medium')
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
    ax.grid(True, alpha=0.25, linewidth=0.8)
    
    # Middle: Ensemble distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor('white')
    
    # Generate ensemble of initial conditions
    np.random.seed(123)
    n_trajectories = 1000
    trajectory_energies = []
    
    for _ in range(n_trajectories):
        initial_v = np.random.normal(0, sigma_v)
        # Each trajectory samples from same distribution
        trajectory_energies.append(0.5 * m_H * initial_v**2)
    
    # Histogram of ensemble energies
    ax.hist(trajectory_energies, bins=40, density=True, alpha=0.7, color=colors['accent'],
           edgecolor='white', linewidth=0.8)
    
    # Theoretical energy distribution (exponential)
    E_range = np.linspace(0, max(trajectory_energies), 1000)
    energy_dist = (1/(k_B*T)) * np.exp(-E_range/(k_B*T))
    ax.plot(E_range, energy_dist, '--', linewidth=3, color=colors['dark'],
           label='Maxwell-Boltzmann\nEnergy Distribution', alpha=0.9)
    
    ax.axvline(theoretical_avg, linestyle=':', linewidth=2.5, color=colors['secondary'],
              label=f'⟨E⟩ = ½kT', alpha=0.9)
    
    ax.set_xlabel('Kinetic Energy (erg)', fontsize=14, color=colors['dark'], weight='medium')
    ax.set_ylabel('Probability Density', fontsize=14, color=colors['dark'], weight='medium')
    ax.set_title('Ensemble Distribution:\nMany Initial Conditions', 
                fontsize=15, pad=15, color=colors['dark'], weight='medium')
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
    ax.grid(True, alpha=0.25, linewidth=0.8)
    
    # Right: Non-ergodic example (trapped system)
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor('white')
    
    # Simulate trapped particle (harmonic oscillator with fixed energy)
    t = np.linspace(0, 10*np.pi, 5000)
    E_trap = 0.1 * k_B * T  # Fixed energy much less than kT
    A = np.sqrt(2*E_trap/m_H)  # Amplitude
    x_trap = A * np.sin(t)
    v_trap = A * np.cos(t)
    
    # Time series shows oscillation without equilibration
    ax.plot(t[:1000]/np.pi, (0.5*m_H*v_trap[:1000]**2)/(k_B*T), linewidth=2, 
           color=colors['secondary'], label='Trapped Particle', alpha=0.8)
    ax.axhline(0.5, linestyle='--', linewidth=2.5, color=colors['dark'], 
              label='⟨E⟩/kT = ½ (equilibrium)', alpha=0.7)
    ax.axhline(E_trap/(k_B*T), linestyle=':', linewidth=2.5, color=colors['accent'],
              label=f'Actual E/kT = {E_trap/(k_B*T):.2f}', alpha=0.8)
    
    ax.set_xlabel('Time (units of π)', fontsize=14, color=colors['dark'], weight='medium')
    ax.set_ylabel('Energy / kT', fontsize=14, color=colors['dark'], weight='medium')
    ax.set_title('Non-Ergodic System:\nTrapped, Cannot Reach Equilibrium', 
                fontsize=15, pad=15, color=colors['dark'], weight='medium')
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_ylim(0, 0.6)
    
    # === BOTTOM ROW: Key Insights ===
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    # Educational summary boxes
    insights = [
        ("🔄 Ergodic Hypothesis", 
         "Time averages = Ensemble averages\nThis is why MCMC sampling works!",
         colors['primary']),
        ("🎯 MCMC Connection", 
         "Long chains sample equilibrium distribution\nErgodicity guarantees convergence",
         colors['accent']),
        ("🚫 Non-Ergodic Failure", 
         "Trapped systems never equilibrate\nMCMC chains get stuck, won't converge",
         colors['secondary']),
        ("🌡️ Temperature Meaning", 
         "T sets equilibrium distribution width\nSame T → Same statistics (if ergodic)",
         colors['neutral'])
    ]
    
    for i, (title, description, color) in enumerate(insights):
        x_pos = 0.05 + i * 0.23
        # Create rounded rectangle background
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x_pos, 0.1), 0.2, 0.8, boxstyle="round,pad=0.02",
                           facecolor=color, alpha=0.1, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        
        # Add text
        ax.text(x_pos + 0.1, 0.7, title, ha='center', va='center', fontsize=13,
               weight='bold', color=colors['dark'], wrap=True)
        ax.text(x_pos + 0.1, 0.35, description, ha='center', va='center', fontsize=11,
               color=colors['dark'], wrap=True)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Overall title with better positioning
    fig.suptitle('Ergodicity: The Statistical Foundation of Thermal Equilibrium and MCMC\n'
                'Why Time Averages Equal Ensemble Averages (When Systems Can Explore All States)',
                fontsize=16, y=0.92, color=colors['dark'], weight='semibold')
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_DIR, '07_ergodic_vs_nonergodic_systems.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print educational summary
    print("Key insight: Ergodicity is WHY statistical mechanics works!")
    print("• Ergodic systems: Time averages = Ensemble averages")
    print("• This allows us to predict equilibrium properties from single long simulations")
    print("• MCMC sampling relies on ergodicity - chains must explore all probable states")
    print("• Non-ergodic systems get trapped and never reach equilibrium")
    print("• Temperature controls the equilibrium distribution all ergodic systems reach")

# =============================================================================
# 8. LAW OF LARGE NUMBERS CONVERGENCE
# =============================================================================

def demo_law_of_large_numbers():
    """
    Demonstrate the Law of Large Numbers - the fundamental reason stellar modeling works.
    Shows how 10^57 particles create statistical certainty rather than chaos.
    """
    print("\n" + "=" * 60)
    print("8. LAW OF LARGE NUMBERS CONVERGENCE")
    print("=" * 60)
    
    # Modern color palette - consistent with other figures
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose  
        'accent': '#16A085',     # Elegant teal
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })
    
    # Create comprehensive figure showing why stellar modeling is possible
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Panel 1 (top-left): Distribution narrowing with increasing N
    ax = axes[0, 0]
    ax.set_facecolor('white')
    
    # Show how distributions narrow as N increases
    N_values = [10, 100, 1000, 10000]
    sample_colors = [colors['secondary'], colors['accent'], colors['primary'], colors['dark']]
    alphas = [0.8, 0.7, 0.6, 0.5]
    
    for i, (N, color, alpha) in enumerate(zip(N_values, sample_colors, alphas)):
        # Generate many sample means
        n_trials = 2000
        sample_means = []
        for _ in range(n_trials):
            # Sample from exponential distribution (stellar energies!)
            sample = np.random.exponential(scale=1.0, size=N)
            sample_means.append(np.mean(sample))
        
        # Plot histogram of sample means
        bins = np.linspace(0.5, 1.5, 40)
        ax.hist(sample_means, bins=bins, alpha=alpha, density=True, 
               color=color, edgecolor='white', linewidth=0.5,
               label=f'N = {N:,} (σ = {np.std(sample_means):.3f})')
    
    # Theoretical mean
    ax.axvline(x=1.0, color=colors['dark'], linewidth=3.5, linestyle='-', alpha=0.9,
              label='True Mean = 1.0')
    
    ax.set_xlabel('Sample Mean', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_ylabel('Probability Density', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_title('Distributions Narrow as N Increases\n"Averages become predictable"', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.legend(fontsize=12, frameon=True, fancybox=True,
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    ax.set_xlim(0.5, 1.5)
    
    # Panel 2 (top-right): Universal 1/√N scaling
    ax = axes[0, 1]
    ax.set_facecolor('white')
    
    # Generate scaling data
    N_range = np.logspace(1, 6, 30).astype(int)
    measured_stds = []
    
    for N in N_range:
        # Calculate standard error of the mean
        n_trials = 200 
        sample_means = []
        for _ in range(n_trials):
            sample = np.random.exponential(scale=1.0, size=N)
            sample_means.append(np.mean(sample))
        measured_stds.append(np.std(sample_means))
    
    # Plot measured vs theoretical
    ax.loglog(N_range, measured_stds, 'o', color=colors['primary'], 
             markersize=6, markerfacecolor='white', markeredgecolor=colors['primary'],
             markeredgewidth=2, alpha=0.8, label='Measured Standard Error')
    
    # Theoretical 1/√N line (properly normalized)
    theoretical = measured_stds[10] * np.sqrt(N_range[10]) / np.sqrt(N_range)
    ax.loglog(N_range, theoretical, '--', color=colors['secondary'], 
             linewidth=4, alpha=0.9, label=r'Theoretical: $1/\sqrt{N}$ scaling')
    
    ax.set_xlabel('Sample Size N', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_ylabel('Standard Error of Mean', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_title('Universal Statistical Law\n$\\sigma_{\\bar{x}} \\propto 1/\\sqrt{N}$', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.legend(fontsize=13, frameon=True, fancybox=True,
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    
    # Panel 3 (bottom-left): Real-time convergence
    ax = axes[1, 0]
    ax.set_facecolor('white')
    
    # Show running average converging to true value
    N_max = 15000
    np.random.seed(42)  # For reproducible demo
    # Use stellar energy distribution (exponential)
    samples = np.random.exponential(scale=1.0, size=N_max)
    running_means = np.cumsum(samples) / np.arange(1, N_max + 1)
    
    # Calculate confidence bounds
    sample_range = np.arange(1, N_max + 1)
    upper_bound = 1.0 + 1.0/np.sqrt(sample_range)  # True mean ± 1 standard error
    lower_bound = 1.0 - 1.0/np.sqrt(sample_range)
    
    # Plot convergence
    ax.plot(sample_range, running_means, color=colors['accent'], linewidth=2.5, alpha=0.9,
           label='Running Average')
    ax.axhline(y=1.0, color=colors['secondary'], linewidth=3.5, linestyle='-', alpha=0.9,
              label='True Mean = 1.0')
    ax.fill_between(sample_range, lower_bound, upper_bound, 
                   color=colors['neutral'], alpha=0.2, label='±1σ Bounds')
    
    ax.set_xlabel('Number of Samples', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_ylabel('Running Average', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_title('Convergence to True Value\n"More data = more certainty"', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.legend(fontsize=13, frameon=True, fancybox=True,
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    ax.set_xlim(0, N_max)
    ax.set_ylim(0.7, 1.3)
    
    # Panel 4 (bottom-right): Physical scales - WHY stellar modeling works
    ax = axes[1, 1]
    ax.set_facecolor('white')
    
    # Physical systems and their particle counts
    systems = ['Lab Sample\n(N ≈ 10²³)', 'Atmosphere\n(N ≈ 10⁴⁴)', 'Star Interior\n(N ≈ 10⁵⁷)', 'Galaxy\n(N ≈ 10⁶⁸)']
    N_phys = [1e23, 1e44, 1e57, 1e68]
    fluctuations = [1/np.sqrt(N) for N in N_phys]
    system_colors = [colors['secondary'], colors['accent'], colors['primary'], colors['dark']]
    
    bars = ax.bar(range(len(systems)), fluctuations, color=system_colors, alpha=0.8,
                 edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, fluct) in enumerate(zip(bars, fluctuations)):
        height = bar.get_height()
        # Format very small numbers nicely
        if fluct < 1e-20:
            label_text = f'{fluct:.0e}'
        else:
            label_text = f'{fluct:.1e}'
        ax.text(bar.get_x() + bar.get_width()/2., height * 2,
                label_text, ha='center', va='bottom', fontsize=12, 
                color=colors['dark'], weight='bold')
    
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, fontsize=12, color=colors['dark'])
    ax.set_ylabel('Relative Fluctuation (σ/μ)', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_yscale('log')
    ax.set_title('Why Stellar Physics is Predictable\n"Large N → Statistical Certainty"', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    ax.set_ylim(1e-35, 1e-10)
    
    # Add explanatory text box
    ax.text(0.5, 0.02, 'Stars have ~10⁵⁷ particles\nFluctuations are ~10⁻²⁸·⁵\nSmaller than quantum uncertainty!', 
           transform=ax.transAxes, ha='center', va='bottom', fontsize=12,
           color=colors['dark'], style='italic', weight='medium',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
           edgecolor=colors['neutral'], alpha=0.95))
    
    # Style all axes
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color(colors['neutral'])
            spine.set_linewidth(0.8)
    
    # Professional main title
    plt.suptitle('Law of Large Numbers: Why $10^{57}$ Particles Create Predictability', 
                fontsize=22, y=0.96, color=colors['dark'], weight='semibold')
    
    # Educational subtitle
    fig.text(0.5, 0.92, 'The statistical foundation that makes stellar modeling possible • Chaos at microscale → Order at macroscale',
             ha='center', va='center', fontsize=16, color=colors['neutral'], style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.35, wspace=0.25)
    
    # Save with high quality
    plt.savefig(os.path.join(FIGURES_DIR, '08_law_of_large_numbers_convergence.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Quantitative analysis
    print("\n🎯 Fluctuation Scaling Analysis:")
    print("-" * 60)
    physical_systems = [
        ("Laboratory sample", 1e23),
        ("Stellar atmosphere", 1e44), 
        ("Stellar core", 1e57),
        ("Galaxy", 1e68)
    ]
    
    for system, N in physical_systems:
        fluct = 1/np.sqrt(N)
        print(f"{system:20s}: N = {N:.0e}, fluctuations = {fluct:.1e} ({fluct*100:.1e}%)")
    
    print("\n✓ This is WHY we can model 10⁵⁷ particles with just 4 equations!")
    print("✓ Large numbers don't create complexity - they create certainty!")

# =============================================================================
# 9. ERROR PROPAGATION THROUGH CALCULATIONS
# =============================================================================

def demo_error_propagation():
    """
    Show how uncertainties propagate through stellar luminosity calculation.
    Modern aesthetic: professional styling with consistent color palette.
    """
    print("\n" + "=" * 60)
    print("9. ERROR PROPAGATION THROUGH CALCULATIONS")
    print("=" * 60)
    
    # Set modern style parameters
    plt.style.use('default')
    colors = {
        'primary': '#2E86AB',    # Deep blue
        'secondary': '#A23B72',  # Deep rose
        'accent': '#F18F01',     # Orange
        'success': '#C73E1D',    # Red-orange
        'neutral': '#6C757D',    # Gray
        'light': '#E9F4F8'       # Light blue
    }
    
    # Example: Calculate stellar luminosity from radius and temperature
    # L = 4π R² σ T⁴ (Stefan-Boltzmann law)

    # "Measurements" with uncertainties
    R_mean = 1.0  # Solar radii
    R_sigma = 0.05  # 5% uncertainty
    T_mean = 5778  # K (solar temperature)
    T_sigma = 50  # K uncertainty

    # Method 1: Error propagation formula
    # L = 4π R² σ T⁴, so:
    # ∂L/∂R = 8π R σ T⁴
    # ∂L/∂T = 16π R² σ T³

    sigma_SB = 5.67e-8  # Stefan-Boltzmann constant
    L_mean = 4 * np.pi * R_mean**2 * sigma_SB * T_mean**4

    dL_dR = 8 * np.pi * R_mean * sigma_SB * T_mean**4
    dL_dT = 16 * np.pi * R_mean**2 * sigma_SB * T_mean**3

    L_sigma_formula = np.sqrt((dL_dR * R_sigma)**2 + (dL_dT * T_sigma)**2)
    L_relative_error_formula = L_sigma_formula / L_mean

    # Method 2: Monte Carlo error propagation
    n_samples = 10000
    R_samples = np.random.normal(R_mean, R_sigma, n_samples)
    T_samples = np.random.normal(T_mean, T_sigma, n_samples)
    L_samples = 4 * np.pi * R_samples**2 * sigma_SB * T_samples**4

    L_mean_MC = np.mean(L_samples)
    L_sigma_MC = np.std(L_samples)
    L_relative_error_MC = L_sigma_MC / L_mean_MC

    # Create figure with modern styling
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor('white')

    # Set consistent font styling
    font_title = {'fontsize': 16, 'fontweight': 'bold'}
    font_label = {'fontsize': 14}
    font_text = {'fontsize': 12}

    # Top left: Input distributions
    ax = axes[0, 0]
    ax.hist(R_samples/R_mean, bins=50, alpha=0.7, density=True, 
            color=colors['primary'], label='R/R☉', edgecolor='white', linewidth=0.5)
    ax.hist(T_samples/T_mean, bins=50, alpha=0.7, density=True, 
            color=colors['secondary'], label='T/T☉', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Normalized Value', **font_label)
    ax.set_ylabel('Probability Density', **font_label)
    ax.set_title('Input Measurement Uncertainties', **font_title, pad=20)
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(frameon=False, fontsize=12)
    ax.tick_params(labelsize=11)

    # Top right: Output distribution  
    ax = axes[0, 1]
    ax.hist(L_samples/L_mean, bins=50, density=True, alpha=0.8, 
            color=colors['accent'], edgecolor='white', linewidth=0.5)
    ax.axvline(1, color=colors['success'], linestyle='--', linewidth=2.5, 
               label='Expected value', alpha=0.8)
    ax.set_xlabel('L/L☉ (Normalized Luminosity)', **font_label)
    ax.set_ylabel('Probability Density', **font_label)
    ax.set_title(f'Resulting Luminosity Distribution\nRelative Error: σ_L/L = {L_relative_error_MC:.2%}', 
                **font_title, pad=20)
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(frameon=False, fontsize=12)
    ax.tick_params(labelsize=11)

    # Bottom left: Error contributions
    ax = axes[1, 0]
    # Calculate individual contributions
    L_from_R_only = 4 * np.pi * R_samples**2 * sigma_SB * T_mean**4
    L_from_T_only = 4 * np.pi * R_mean**2 * sigma_SB * T_samples**4
    R_contribution = np.std(L_from_R_only) / L_mean
    T_contribution = np.std(L_from_T_only) / L_mean

    contributions = [R_contribution*100, T_contribution*100, L_relative_error_MC*100]
    labels = ['Radius\nUncertainty', 'Temperature\nUncertainty', 'Combined\nTotal']
    bar_colors = [colors['primary'], colors['secondary'], colors['accent']]

    bars = ax.bar(labels, contributions, color=bar_colors, alpha=0.8, 
                  edgecolor='white', linewidth=1)
    ax.set_ylabel('Relative Error (%)', **font_label)
    ax.set_title('Error Contribution Analysis', **font_title, pad=20)

    # Add values on bars with better formatting
    for bar, val in zip(bars, contributions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold', fontfamily='Inter')

    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax.tick_params(labelsize=11)

    # Bottom right: Method comparison and insights
    ax = axes[1, 1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Title and comparison
    ax.text(0.5, 0.95, 'Method Comparison & Key Insights', 
            ha='center', va='top', **font_title, color=colors['primary'])
    
    ax.text(0.05, 0.80, 'Analytical vs. Monte Carlo:', 
            ha='left', va='top', fontsize=13, fontweight='bold', fontfamily='Inter')
    ax.text(0.05, 0.72, f'• Formula method: σ_L/L = {L_relative_error_formula:.3f}', 
            ha='left', va='top', **font_text)
    ax.text(0.05, 0.65, f'• Monte Carlo: σ_L/L = {L_relative_error_MC:.3f}', 
            ha='left', va='top', **font_text)
    ax.text(0.05, 0.58, f'• Agreement: {abs(L_relative_error_formula-L_relative_error_MC)/L_relative_error_MC*100:.1f}% difference', 
            ha='left', va='top', **font_text)
    
    ax.text(0.05, 0.45, 'Physical Insight:', 
            ha='left', va='top', fontsize=13, fontweight='bold', fontfamily='Inter', color=colors['secondary'])
    ax.text(0.05, 0.37, f'• T⁴ dependence amplifies temperature errors', 
            ha='left', va='top', **font_text)
    ax.text(0.05, 0.30, f'• Temperature contributes {T_contribution/L_relative_error_MC:.0%} of total error', 
            ha='left', va='top', **font_text)
    ax.text(0.05, 0.23, f'• Despite 5% radius vs 0.9% temp uncertainty,', 
            ha='left', va='top', **font_text)
    ax.text(0.05, 0.16, f'  temperature dominates by {T_contribution/R_contribution:.1f}× factor', 
            ha='left', va='top', **font_text, color=colors['success'])
    
    ax.text(0.05, 0.05, 'Stefan-Boltzmann: L = 4πR²σT⁴', 
            ha='left', va='top', fontsize=11, fontfamily='Inter', 
            style='italic', color=colors['neutral'])
    
    ax.axis('off')

    # Overall title and layout
    fig.suptitle('Error Propagation in Stellar Luminosity Calculations', 
                fontsize=18, fontweight='bold', fontfamily='Inter', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_DIR, '09_error_propagation_through_calculations.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

    print("Error Propagation Analysis:")
    print("-" * 50)
    print(f"Input uncertainties:")
    print(f"  R: {R_sigma/R_mean:.1%}")
    print(f"  T: {T_sigma/T_mean:.1%}")
    print(f"\nOutput uncertainty in L:")
    print(f"  Formula method: {L_relative_error_formula:.2%}")
    print(f"  Monte Carlo: {L_relative_error_MC:.2%}")
    print(f"\nError contributions:")
    print(f"  From R uncertainty: {R_contribution:.2%}")
    print(f"  From T uncertainty: {T_contribution:.2%}")
    print(f"  Ratio T/R contribution: {T_contribution/R_contribution:.1f}x")

# =============================================================================
# 10. BAYESIAN LEARNING AND INFERENCE
# =============================================================================

def demo_bayesian_learning():
    """
    Demonstrate Bayesian inference updating beliefs about stellar temperature.
    """
    print("\n" + "=" * 60)
    print("10. BAYESIAN LEARNING AND INFERENCE")
    print("=" * 60)
    
    # Example: Inferring a star's temperature from its color
    # Color index B-V is related to temperature, but with uncertainty

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # True stellar temperature (unknown to us)
    T_true = 5800  # K (Sun-like)

    # Prior: Based on stellar population studies
    # Most stars are cool (M-dwarfs), fewer hot stars
    T_range = np.linspace(3000, 10000, 1000)

    # Prior: Log-normal distribution (more cool stars)
    prior_mean = np.log(4500)
    prior_std = 0.4
    prior = stats.lognorm.pdf(T_range, s=prior_std, scale=np.exp(prior_mean))
    prior = prior / np.trapz(prior, T_range)  # Normalize

    # Observation: Color measurement with uncertainty
    # Simplified relation: B-V ≈ 5000/T (very approximate!)
    observed_color = 5000/T_true + np.random.normal(0, 0.1)
    color_uncertainty = 0.1

    # Likelihood: P(data|temperature)
    def likelihood(T, observed_color, uncertainty):
        expected_color = 5000/T  # Our model
        return stats.norm.pdf(observed_color, expected_color, uncertainty)

    like = np.array([likelihood(T, observed_color, color_uncertainty) for T in T_range])
    like = like / np.trapz(like, T_range)  # Normalize for visualization

    # Posterior: Prior × Likelihood (then normalize)
    posterior_unnorm = prior * like
    posterior = posterior_unnorm / np.trapz(posterior_unnorm, T_range)

    # Visualization
    # Row 1: Single observation
    axes[0, 0].plot(T_range, prior, 'b-', lw=2)
    axes[0, 0].fill_between(T_range, prior, alpha=0.3)
    axes[0, 0].axvline(T_true, color='red', linestyle='--', alpha=0.5, label='True T')
    axes[0, 0].set_title('Prior\n(Population knowledge)')
    axes[0, 0].set_xlabel('Temperature (K)')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].legend()

    axes[0, 1].plot(T_range, like, 'g-', lw=2)
    axes[0, 1].fill_between(T_range, like, alpha=0.3, color='green')
    axes[0, 1].axvline(T_true, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title(f'Likelihood\n(Color = {observed_color:.2f})')
    axes[0, 1].set_xlabel('Temperature (K)')

    axes[0, 2].plot(T_range, posterior, 'purple', lw=2)
    axes[0, 2].fill_between(T_range, posterior, alpha=0.3, color='purple')
    axes[0, 2].axvline(T_true, color='red', linestyle='--', alpha=0.5)
    axes[0, 2].set_title('Posterior\n(Updated belief)')
    axes[0, 2].set_xlabel('Temperature (K)')

    # Row 2: Multiple observations (sequential updating)
    n_obs = 5
    colors = plt.cm.viridis(np.linspace(0, 1, n_obs))

    # Start with prior
    current_posterior = prior.copy()

    axes[1, 0].plot(T_range, prior, 'k-', lw=2, label='Initial prior')

    for i in range(n_obs):
        # New observation
        new_color = 5000/T_true + np.random.normal(0, 0.1)
        
        # Likelihood for this observation
        new_like = np.array([likelihood(T, new_color, color_uncertainty) for T in T_range])
        
        # Update: posterior becomes new prior
        current_posterior = current_posterior * new_like
        current_posterior = current_posterior / np.trapz(current_posterior, T_range)
        
        axes[1, 0].plot(T_range, current_posterior, color=colors[i], 
                        alpha=0.7, label=f'After obs {i+1}')

    axes[1, 0].axvline(T_true, color='red', linestyle='--', alpha=0.5, label='True T')
    axes[1, 0].set_title('Sequential Updating')
    axes[1, 0].set_xlabel('Temperature (K)')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].legend(fontsize=8)

    # Show uncertainty reduction
    axes[1, 1].set_title('Posterior Uncertainty vs N')
    uncertainties = []
    n_obs_range = range(0, 21)

    current_posterior = prior.copy()
    for n in n_obs_range:
        # Calculate standard deviation
        mean = np.trapz(T_range * current_posterior, T_range)
        std = np.sqrt(np.trapz((T_range - mean)**2 * current_posterior, T_range))
        uncertainties.append(std)
        
        if n < 20:  # Get one more observation
            new_color = 5000/T_true + np.random.normal(0, 0.1)
            new_like = np.array([likelihood(T, new_color, color_uncertainty) for T in T_range])
            current_posterior = current_posterior * new_like
            current_posterior = current_posterior / np.trapz(current_posterior, T_range)

    axes[1, 1].plot(n_obs_range, uncertainties, 'bo-')
    axes[1, 1].set_xlabel('Number of Observations')
    axes[1, 1].set_ylabel('Posterior Std Dev (K)')
    axes[1, 1].grid(True, alpha=0.3)

    # Compare Bayesian vs Frequentist
    axes[1, 2].text(0.1, 0.9, 'Bayesian vs Frequentist', fontsize=12, weight='bold', 
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.75, 'Bayesian:', fontsize=11, weight='bold',
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.65, '• Probability of parameters given data', fontsize=10,
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.55, '• Incorporates prior knowledge', fontsize=10,
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.45, '• Updates with each observation', fontsize=10,
                    transform=axes[1, 2].transAxes)

    axes[1, 2].text(0.1, 0.30, 'Frequentist:', fontsize=11, weight='bold',
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.20, '• Probability of data given parameters', fontsize=10,
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.10, '• No prior (or uniform prior)', fontsize=10,
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.00, '• Focus on long-run frequencies', fontsize=10,
                    transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')

    plt.suptitle('Bayesian Inference: Learning from Data', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_DIR, '10_bayesian_learning_and_inference.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

    print("Bayesian Learning Summary:")
    print("-" * 50)
    print(f"True temperature: {T_true} K")
    print(f"Prior peak: {T_range[np.argmax(prior)]:.0f} K")
    print(f"Prior std: {np.sqrt(np.trapz((T_range - np.trapz(T_range*prior, T_range))**2 * prior, T_range)):.0f} K")
    print(f"Posterior peak (1 obs): {T_range[np.argmax(posterior)]:.0f} K")
    print(f"Posterior std (1 obs): {np.sqrt(np.trapz((T_range - np.trapz(T_range*posterior, T_range))**2 * posterior, T_range)):.0f} K")
    print(f"Posterior std (5 obs): {uncertainties[5]:.0f} K")
    print(f"Posterior std (20 obs): {uncertainties[20]:.0f} K")

# =============================================================================
# 11. MONTE CARLO π ESTIMATION
# =============================================================================

def demo_monte_carlo_pi():
    """
    Estimate π using Monte Carlo sampling - the foundation of computational astrophysics.
    Shows visual dartboard method + convergence analysis with modern styling.
    """
    print("\n" + "=" * 60)
    print("11. MONTE CARLO π ESTIMATION")
    print("=" * 60)
    
    def estimate_pi_monte_carlo(n_samples):
        """
        Estimate π using Monte Carlo sampling.
        
        Method: Generate random points in [-1,1] x [-1,1].
        Check if they fall inside unit circle (x² + y² ≤ 1).
        π/4 = (points inside circle) / (total points)
        """
        # Generate random points in square [-1, 1] x [-1, 1]
        x = np.random.uniform(-1, 1, n_samples)
        y = np.random.uniform(-1, 1, n_samples)
        
        # Check if points are inside unit circle
        inside_circle = (x**2 + y**2) <= 1
        
        # Estimate π
        # π/4 = fraction inside circle, so π = 4 * fraction
        pi_estimate = 4 * np.sum(inside_circle) / n_samples
        
        return pi_estimate, x, y, inside_circle

    # Test with increasing sample sizes for convergence analysis
    sample_sizes = [100, 1000, 10000, 100000, 1000000]
    pi_estimates = []
    errors = []
    
    print(f"True value: π = {np.pi:.10f}")
    print("-" * 50)
    
    for n in sample_sizes:
        pi_est, _, _, _ = estimate_pi_monte_carlo(n)
        error = abs(pi_est - np.pi)
        relative_error = error / np.pi * 100
        pi_estimates.append(pi_est)
        errors.append(error)
        print(f"N = {n:7d}: π ≈ {pi_est:.6f}, error = {error:.6f} ({relative_error:.2f}%)")
    
    # Modern color palette - consistent with other figures
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose  
        'accent': '#16A085',     # Elegant teal
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Panel 1 (top-left): Visual demonstration with fewer points for clarity
    n_vis_small = 1000
    pi_est_small, x_small, y_small, inside_small = estimate_pi_monte_carlo(n_vis_small)
    
    ax = axes[0, 0]
    ax.set_facecolor('white')
    
    # Plot points with our modern colors - inside points as teal, outside as rose
    ax.scatter(x_small[inside_small], y_small[inside_small], 
              c=colors['accent'], s=8, alpha=0.8, edgecolors='none',
              label=f'Inside Circle: {np.sum(inside_small)}', rasterized=True)
    ax.scatter(x_small[~inside_small], y_small[~inside_small], 
              c=colors['secondary'], s=8, alpha=0.6, edgecolors='none',
              label=f'Outside Circle: {np.sum(~inside_small)}', rasterized=True)
    
    # Draw perfect circle with thick line
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=colors['dark'], linewidth=3.5, alpha=0.9)
    
    # Draw square boundary
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 
           color=colors['dark'], linewidth=3.5, alpha=0.9)
    
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.set_xlabel('x coordinate', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_ylabel('y coordinate', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_title(f'Monte Carlo Dartboard Method\nN = {n_vis_small:,} samples, π ≈ {pi_est_small:.4f}', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.legend(fontsize=13, frameon=True, fancybox=True, 
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    
    # Panel 2 (top-right): Higher resolution demonstration  
    n_vis_large = 10000
    pi_est_large, x_large, y_large, inside_large = estimate_pi_monte_carlo(n_vis_large)
    
    ax = axes[0, 1]
    ax.set_facecolor('white')
    
    # Smaller points for density, with transparency
    ax.scatter(x_large[inside_large], y_large[inside_large], 
              c=colors['accent'], s=2, alpha=0.7, edgecolors='none', rasterized=True)
    ax.scatter(x_large[~inside_large], y_large[~inside_large], 
              c=colors['secondary'], s=2, alpha=0.5, edgecolors='none', rasterized=True)
    
    # Draw circle and square
    ax.plot(np.cos(theta), np.sin(theta), color=colors['dark'], linewidth=3.5, alpha=0.9)
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 
           color=colors['dark'], linewidth=3.5, alpha=0.9)
    
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.set_xlabel('x coordinate', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_ylabel('y coordinate', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_title(f'High Density Sampling\nN = {n_vis_large:,} samples, π ≈ {pi_est_large:.4f}', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    
    # Add accuracy annotation
    accuracy = abs(pi_est_large - np.pi) / np.pi * 100
    ax.text(0.05, 0.95, f'Accuracy: {accuracy:.2f}% error', 
           transform=ax.transAxes, fontsize=14, color=colors['dark'], weight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
           edgecolor=colors['neutral'], alpha=0.95))
    
    # Panel 3 (bottom-left): Error scaling analysis
    ax = axes[1, 0]
    ax.set_facecolor('white')
    
    # Plot measured errors
    ax.loglog(sample_sizes, errors, 'o-', color=colors['primary'], linewidth=3, 
             markersize=8, markerfacecolor='white', markeredgecolor=colors['primary'],
             markeredgewidth=2, label='Measured Error')
    
    # Plot theoretical 1/√N scaling - normalized to match data
    theoretical_scaling = errors[2] * np.sqrt(sample_sizes[2]) / np.sqrt(sample_sizes)
    ax.loglog(sample_sizes, theoretical_scaling, '--', color=colors['secondary'], 
             linewidth=3.5, alpha=0.9, label=r'Theoretical $1/\sqrt{N}$ scaling')
    
    ax.set_xlabel('Sample Size N', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_ylabel('Absolute Error |π - π̂|', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_title('Monte Carlo Error Scaling\nFundamental $1/\\sqrt{N}$ Convergence', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.legend(fontsize=14, frameon=True, fancybox=True,
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    
    # Panel 4 (bottom-right): Convergence of estimates
    ax = axes[1, 1]
    ax.set_facecolor('white')
    
    # Plot π estimates vs sample size
    ax.semilogx(sample_sizes, pi_estimates, 'o-', color=colors['accent'], 
               linewidth=3, markersize=8, markerfacecolor='white', 
               markeredgecolor=colors['accent'], markeredgewidth=2, 
               label='π Estimates')
    
    # True value of π as horizontal line
    ax.axhline(y=np.pi, color=colors['secondary'], linewidth=3.5, 
              linestyle='-', alpha=0.9, label=f'True π = {np.pi:.6f}')
    
    # Error bounds
    upper_bound = np.pi + 1/np.sqrt(np.array(sample_sizes))
    lower_bound = np.pi - 1/np.sqrt(np.array(sample_sizes))
    ax.fill_between(sample_sizes, lower_bound, upper_bound, 
                   color=colors['neutral'], alpha=0.2, label='Expected ±1σ bounds')
    
    ax.set_xlabel('Sample Size N', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_ylabel('π Estimate', fontsize=15, color=colors['dark'], weight='medium')
    ax.set_title('Convergence to True Value\nEstimates Approach π as N → ∞', 
                fontsize=18, pad=25, color=colors['dark'], weight='medium')
    ax.legend(fontsize=13, frameon=True, fancybox=True,
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis='both', labelsize=13, colors=colors['neutral'])
    
    # Style all axes
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color(colors['neutral'])
            spine.set_linewidth(0.8)
    
    # Professional main title
    plt.suptitle('Monte Carlo Method: Estimating π Through Random Sampling', 
                fontsize=22, y=0.96, color=colors['dark'], weight='semibold')
    
    # Educational subtitle
    fig.text(0.5, 0.92, 'Foundation technique for computational astrophysics • Random points + geometry = statistical precision',
             ha='center', va='center', fontsize=16, color=colors['neutral'], style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.35, wspace=0.25)
    
    # Save with high quality
    plt.savefig(os.path.join(FIGURES_DIR, '11_monte_carlo_pi_estimation.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)
    
    print("\n🎯 Key Insights:")
    print("✓ Monte Carlo error decreases as 1/√N (fundamental limitation)")
    print("✓ Need 100× more samples for 10× better accuracy")  
    print("✓ Method works for any integral - basis for all computational astrophysics")
    print("✓ Trade computational time for statistical precision")

# =============================================================================
# 12. RANDOM SAMPLING METHODS
# =============================================================================

def demo_random_sampling():
    """
    Demonstrate inverse transform and rejection sampling methods.
    """
    print("\n" + "=" * 60)
    print("12. RANDOM SAMPLING METHODS")
    print("=" * 60)
    
    # Example 1: Exponential distribution (inverse transform)
    print("Inverse Transform Method - Exponential Distribution:")
    print("-" * 50)
    
    def sample_exponential(lambda_param, n_samples):
        """Sample exponential distribution using inverse transform"""
        u = np.random.uniform(0, 1, n_samples)
        return -np.log(u) / lambda_param
    
    # Sample and compare to theory
    lambda_val = 2.0
    n_samples = 10000
    exp_samples = sample_exponential(lambda_val, n_samples)
    
    print(f"Lambda = {lambda_val}")
    print(f"Theoretical mean: {1/lambda_val:.3f}")
    print(f"Sample mean: {np.mean(exp_samples):.3f}")
    print(f"Theoretical std: {1/lambda_val:.3f}")
    print(f"Sample std: {np.std(exp_samples):.3f}")
    
    # Example 2: Rejection sampling for arbitrary function
    print("\nRejection Sampling Method:")
    print("-" * 50)
    
    def rejection_sample(f, x_min, x_max, f_max, n_samples):
        """Sample using rejection method"""
        samples = []
        n_tries = 0
        
        while len(samples) < n_samples:
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(0, f_max)
            n_tries += 1
            
            if y <= f(x):
                samples.append(x)
        
        efficiency = n_samples / n_tries
        return np.array(samples), efficiency
    
    # Example: Sample from x^2 on [0,1]
    def quadratic(x):
        return x**2
    
    quad_samples, efficiency = rejection_sample(quadratic, 0, 1, 1, 1000)
    print(f"Quadratic function f(x) = x²")
    print(f"Sampling efficiency: {efficiency:.1%}")
    print(f"Theoretical mean: {3/4:.3f}")  # ∫₀¹ x · 3x² dx = 3/4
    print(f"Sample mean: {np.mean(quad_samples):.3f}")
    
    # Modern color palette - consistent with other figures
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose  
        'accent': '#16A085',     # Elegant teal
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Top left: Exponential samples
    ax = axes[0, 0]
    ax.set_facecolor('white')
    ax.hist(exp_samples, bins=60, density=True, alpha=0.7, 
            color=colors['primary'], edgecolor='white', linewidth=0.5,
            label=f'Samples (N={n_samples:,})')
    x_exp = np.linspace(0, 4, 200)
    ax.plot(x_exp, lambda_val * np.exp(-lambda_val * x_exp), 
            color=colors['secondary'], linewidth=3.5, alpha=0.9, label='Theoretical PDF')
    ax.set_xlabel('Value', fontsize=13, color=colors['dark'])
    ax.set_ylabel('Probability Density', fontsize=13, color=colors['dark'])
    ax.set_title('Inverse Transform Method\nExponential Distribution', 
                fontsize=16, pad=18, color=colors['dark'], weight='medium')
    ax.legend(fontsize=12, frameon=True, fancybox=True, 
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
    
    # Top right: CDF comparison
    ax = axes[0, 1]
    ax.set_facecolor('white')
    cdf_theory = 1 - np.exp(-lambda_val * x_exp)
    ax.plot(x_exp, cdf_theory, color=colors['secondary'], linewidth=3.5, 
           alpha=0.9, label='Theoretical CDF')
    # Empirical CDF
    x_sorted = np.sort(exp_samples)
    y_ecdf = np.arange(1, len(x_sorted)+1) / len(x_sorted)
    ax.plot(x_sorted, y_ecdf, color=colors['primary'], alpha=0.8, linewidth=2.5,
           label='Empirical CDF')
    ax.set_xlabel('Value', fontsize=13, color=colors['dark'])
    ax.set_ylabel('Cumulative Probability', fontsize=13, color=colors['dark'])
    ax.set_title('CDF Validation\nTheory vs. Sample', 
                fontsize=16, pad=18, color=colors['dark'], weight='medium')
    ax.legend(fontsize=12, frameon=True, fancybox=True,
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
    
    # Bottom left: Rejection sampling visualization
    ax = axes[1, 0]
    ax.set_facecolor('white')
    x_quad = np.linspace(0, 1, 200)
    ax.plot(x_quad, quadratic(x_quad), color=colors['dark'], linewidth=3.5, 
           alpha=0.9, label='Target: $f(x) = x^2$', zorder=10)
    ax.fill_between(x_quad, quadratic(x_quad), alpha=0.2, color=colors['dark'])
    
    # Show some rejected and accepted points
    np.random.seed(42)  # For reproducible visualization
    x_test = np.random.uniform(0, 1, 300)
    y_test = np.random.uniform(0, 1, 300)
    accepted = y_test <= quadratic(x_test)
    
    ax.scatter(x_test[~accepted], y_test[~accepted], c=colors['secondary'], s=25, 
               alpha=0.6, label='Rejected', marker='x', linewidth=1.5)
    ax.scatter(x_test[accepted], y_test[accepted], c=colors['accent'], s=20, 
               alpha=0.8, label='Accepted', edgecolors='white', linewidth=0.5)
    ax.set_xlabel('x', fontsize=13, color=colors['dark'])
    ax.set_ylabel('y (uniform random)', fontsize=13, color=colors['dark'])
    ax.set_title(f'Rejection Sampling Method\nEfficiency: {efficiency:.1%}', 
                fontsize=16, pad=18, color=colors['dark'], weight='medium')
    ax.legend(fontsize=12, frameon=True, fancybox=True, loc='upper left',
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
    
    # Bottom right: Quadratic samples
    ax = axes[1, 1]
    ax.set_facecolor('white')
    ax.hist(quad_samples, bins=40, density=True, alpha=0.7, color=colors['accent'],
            edgecolor='white', linewidth=0.5, label=f'Samples (N={len(quad_samples):,})')
    # Theoretical PDF: f(x) = 3x² (normalized)
    ax.plot(x_quad, 3 * x_quad**2, color=colors['secondary'], linewidth=3.5, 
           alpha=0.9, label='Theoretical: $3x^2$')
    ax.set_xlabel('Value', fontsize=13, color=colors['dark'])
    ax.set_ylabel('Probability Density', fontsize=13, color=colors['dark'])
    ax.set_title('Rejection Sampling Result\nQuadratic Distribution', 
                fontsize=16, pad=18, color=colors['dark'], weight='medium')
    ax.legend(fontsize=12, frameon=True, fancybox=True,
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
    
    # Style all axes
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color(colors['neutral'])
            spine.set_linewidth(0.8)
    
    # Professional main title
    plt.suptitle('Random Sampling Methods: From Uniform to Any Distribution', 
                fontsize=20, y=0.96, color=colors['dark'], weight='semibold')
    
    # Educational subtitle
    fig.text(0.5, 0.92, 'Inverse Transform (left) uses CDF inversion • Rejection Sampling (right) accepts/rejects uniform draws',
             ha='center', va='center', fontsize=15, color=colors['neutral'], style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.35, wspace=0.25)
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_DIR, '12_random_sampling_methods.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)

# =============================================================================
# 13. POWER LAW DISTRIBUTION SAMPLING
# =============================================================================

def demo_power_law_sampling():
    """
    Demonstrate sampling from power law distributions - the Universal Law of Astrophysics!
    From stellar masses to dark matter halos, power laws are everywhere.
    """
    print("\n" + "=" * 60)
    print("13. POWER LAW DISTRIBUTION SAMPLING")
    print("=" * 60)
    
    # Modern color palette - consistent with other figures
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose  
        'accent': '#16A085',     # Elegant teal
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })
    
    def sample_power_law(alpha, x_min, x_max, n_samples):
        """Sample from power law distribution p(x) ∝ x^(-alpha) using inverse transform."""
        u = np.random.uniform(0, 1, n_samples)
        
        if abs(alpha - 1.0) < 1e-10:
            # Special case: p(x) ∝ x^(-1)
            samples = x_min * (x_max/x_min)**u
        else:
            # General case: p(x) ∝ x^(-alpha)
            samples = (x_min**(1-alpha) + u*(x_max**(1-alpha) - x_min**(1-alpha)))**(1/(1-alpha))
        
        return samples

    # Focus on Salpeter IMF with different sample sizes to show convergence
    alpha = 2.35  # Salpeter slope
    x_min, x_max = 0.5, 120  # Realistic stellar mass range (solar masses)
    
    # Different sample sizes to demonstrate convergence
    sample_sizes = [100, 1000, 10000]
    size_labels = ['Small Sample', 'Medium Sample', 'Large Sample']
    
    # Create educational 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Theoretical Salpeter power law (for overlay on all panels)
    x_theory = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    norm = (alpha-1)/(x_min**(1-alpha) - x_max**(1-alpha))
    p_theory = norm * x_theory**(-alpha)
    
    for idx, (n_samples, size_label) in enumerate(zip(sample_sizes, size_labels)):
        ax = axes[idx]
        ax.set_facecolor('white')
        
        # Generate Salpeter IMF stellar mass samples
        np.random.seed(42 + idx)  # Reproducible but different per panel
        samples = sample_power_law(alpha, x_min, x_max, n_samples)
        
        # Create histogram with appropriate binning for sample size
        if n_samples <= 1000:
            n_bins = 30
        elif n_samples <= 10000:
            n_bins = 40
        else:
            n_bins = 50
            
        bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins)
        ax.hist(samples, bins=bins, alpha=0.7, density=True, 
               color=colors['primary'], edgecolor='white', linewidth=0.5,
               label=f'N = {n_samples:,} stars')
        
        # Overlay theoretical Salpeter power law on all panels
        ax.plot(x_theory, p_theory, color=colors['secondary'], linewidth=4, 
               alpha=0.9, label=f'Theory: $\\propto M^{{-{alpha}}}$', zorder=10)
        
        # Modern styling
        ax.set_xscale('log')
        ax.set_yscale('log') 
        ax.set_xlabel('Stellar Mass ($M_\\odot$)', fontsize=15, color=colors['dark'], labelpad=12)
        ax.set_ylabel('Probability Density', fontsize=15, color=colors['dark'], labelpad=12)
        ax.set_title(f'{size_label}\nSalpeter IMF Sampling', 
                    fontsize=16, pad=25, color=colors['dark'], weight='medium')
        
        # Professional legend and grid
        ax.legend(fontsize=12, frameon=True, fancybox=True,
                 edgecolor=colors['neutral'], facecolor='white', framealpha=0.95,
                 loc='upper right')
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.tick_params(axis='both', labelsize=12, colors=colors['neutral'])
        
        # Add sample-specific statistics
        mean_mass = np.mean(samples)
        median_mass = np.median(samples)
        low_mass_fraction = np.sum(samples < 1.0) / len(samples) * 100
        
        stats_text = f'Mean: {mean_mass:.1f} $M_\\odot$\n'
        stats_text += f'Median: {median_mass:.2f} $M_\\odot$\n'
        stats_text += f'M < 1$M_\\odot$: {low_mass_fraction:.0f}%'
        
        ax.text(0.95, 0.45, stats_text, transform=ax.transAxes, 
               fontsize=11, va='top', ha='right', color=colors['dark'],
               bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['light'], 
                        edgecolor=colors['neutral'], alpha=0.9))
        
        
        # Add educational notes based on sample size - same level positioning
        if idx == 0:  # Small sample
            note_text = 'Noisy!\nNeed more stars'
            ax.text(0.95, 0.75, note_text, transform=ax.transAxes, 
                   fontsize=11, va='center', ha='right', color=colors['secondary'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=colors['secondary'], alpha=0.9))
        elif idx == 2:  # Large sample
            note_text = 'Smooth!\nExcellent match\nto theory'
            ax.text(0.95, 0.75, note_text, transform=ax.transAxes, 
                   fontsize=11, va='center', ha='right', color=colors['accent'], weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=colors['accent'], alpha=0.9))
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_color(colors['neutral'])
            spine.set_linewidth(0.8)

    # Professional main title with educational focus
    plt.suptitle('Salpeter Initial Mass Function: How Sample Size Affects Accuracy', 
                fontsize=20, y=0.95, color=colors['dark'], weight='semibold')
    
    # Educational subtitle about convergence
    fig.text(0.5, 0.88, 'Larger samples → smoother histograms → better match to theoretical power law',
             ha='center', va='center', fontsize=15, color=colors['neutral'], style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.70, wspace=0.15, hspace=0.15)
    
    # Save figure
    plt.savefig(os.path.join(FIGURES_DIR, '13_power_law_distribution_sampling.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset matplotlib defaults
    plt.rcParams.update(plt.rcParamsDefault)

    print("\n🌟 Astrophysical Power Laws - The Universal Pattern:")
    print("=" * 55)
    print("• Stellar Initial Mass Function (IMF): α ≈ 2.35 (Salpeter)")
    print("• Galaxy luminosity function: Similar slopes")
    print("• Dark matter halo masses: α ≈ 1.9")  
    print("• Asteroid size distribution: α ≈ 2.5")
    print("• Planetary mass distribution: α ≈ 1.5")
    print("\n🔑 Key Insight: Nature loves power laws!")
    print("   Most mass/energy is in the smallest objects!")

# =============================================================================
# 14. PLUMMER SPHERE SPATIAL SAMPLING  
# =============================================================================

def demo_plummer_sampling():
    """
    Demonstrate spatial sampling from Plummer sphere density profiles.
    Shows how to sample 3D positions for realistic star cluster initial conditions.
    """
    print("\n" + "=" * 60)
    print("14. PLUMMER SPHERE SPATIAL SAMPLING")
    print("=" * 60)
    
    # Modern color palette - consistent with other figures
    colors = {
        'primary': '#2E86AB',    # Modern blue
        'secondary': '#A23B72',  # Deep rose  
        'accent': '#16A085',     # Elegant teal
        'neutral': '#6C757D',    # Sophisticated gray
        'light': '#F8F9FA',      # Very light gray
        'dark': '#2D3436'        # Charcoal
    }
    
    # Set modern style parameters with LARGER fonts
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 16,  # Increased base font
        'axes.linewidth': 1.2,  # Thicker axes
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': colors['neutral'],
        'axes.labelcolor': colors['dark'],
        'text.color': colors['dark'],
        'xtick.color': colors['neutral'],
        'ytick.color': colors['neutral']
    })
    
    def sample_plummer_radius(a, n_samples):
        """
        Sample radii from Plummer sphere using inverse transform.
        
        The CDF is F(r) = r³/(r² + a²)^(3/2)
        """
        u = np.random.uniform(0, 1, n_samples)
        # Solve u = r³/(r² + a²)^(3/2) for r
        # Let s = (r/a)², then u = s^(3/2)/(1 + s)^(3/2) = (s/(1+s))^(3/2)
        # So s/(1+s) = u^(2/3), which gives s = u^(2/3)/(1 - u^(2/3))
        u_pow = u**(2/3)
        s = u_pow / (1 - u_pow)
        r = a * np.sqrt(s)
        return r
    
    def sample_plummer_3d(a, n_samples):
        """Sample 3D positions from Plummer sphere"""
        # Sample radii
        r = sample_plummer_radius(a, n_samples)
        
        # Sample random directions on sphere
        # Method: sample from uniform on sphere surface
        theta = np.arccos(1 - 2*np.random.uniform(0, 1, n_samples))  # polar angle
        phi = 2*np.pi*np.random.uniform(0, 1, n_samples)  # azimuthal angle
        
        # Convert to Cartesian
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return x, y, z, r
    
    # Sample from Plummer sphere with educational parameters
    a = 1.0  # Plummer radius (scale length)
    n_samples = 10000  # Reduced to 10,000 stars as requested
    x, y, z, r = sample_plummer_3d(a, n_samples)
    
    # Theoretical Plummer density profile
    def plummer_density(r, a):
        return (3/(4*np.pi*a**3)) * (1 + (r/a)**2)**(-5/2)
    
    # Theoretical enclosed mass fraction (this is the CDF we sample from!)
    def plummer_mass(r, a):
        return r**3 / (r**2 + a**2)**(3/2)
    
    # Educational output
    print(f"🌟 Plummer Sphere Demonstration:")
    print(f"   Generated {n_samples:,} stellar positions")
    print(f"   Plummer radius: a = {a} (scale length)")
    print(f"   Half-mass radius theory: {1.3*a:.2f}a")
    print(f"   Half-mass radius sample: {np.median(r):.2f}a")
    
    # Define colormap range for consistency across all plots - FIXED LIMITS
    r_min, r_max = 0.0, 4.0  # Fixed colormap limits 0 to 4
    
    # Create modern 2x2 figure with white background and MORE space
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Top left: 2D projection with modern styling
    ax = axes[0, 0]
    ax.set_facecolor('white')
    
    # Color stars by radius for depth perception - REVERSED colormap 0 to 4
    scatter = ax.scatter(x, y, s=4, alpha=0.8, c=r, cmap='viridis_r', 
                        rasterized=True, edgecolors='none', vmin=0.0, vmax=4.0)
    
    # Professional styling with MASSIVE fonts
    ax.set_xlabel('Position x/a', fontsize=22, color=colors['dark'], labelpad=18, weight='bold')
    ax.set_ylabel('Position y/a', fontsize=22, color=colors['dark'], labelpad=18, weight='bold')
    ax.set_title('2D Projection: Face-On View\n(Color = Distance from Center)', 
                fontsize=24, pad=30, color=colors['dark'], weight='bold')
    ax.set_aspect('equal')
    
    # Add visible reference circles with thicker lines
    for i, radius in enumerate([a, 2*a, 3*a]):
        circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', 
                           alpha=0.7, color=colors['neutral'], linewidth=3.0)  # Much thicker
        ax.add_patch(circle)
        if i == 1:  # Only label middle circle to avoid clutter
            ax.text(radius*0.7, radius*0.7, f'r = {radius:.0f}a', 
                   fontsize=14, ha='center', va='center', weight='bold',  # Larger, bold
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.9, edgecolor=colors['neutral'], linewidth=1.5))
    
    # Clean axis styling with MASSIVE ticks
    ax.grid(True, alpha=0.3, linewidth=1.0, color=colors['neutral'])
    ax.tick_params(axis='both', which='major', labelsize=18, 
                  colors=colors['neutral'], width=1.5, length=8)
    
    # Set limits to -4a to 4a as requested
    max_extent = 4*a
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    
    # Top right: Radial density profile with modern styling
    ax = axes[0, 1]
    ax.set_facecolor('white')
    
    # Bin the samples radially - handle Poisson noise properly  
    r_bins = np.logspace(-1.5, 1, 35)  # Start from 0.03 to avoid empty inner bins
    r_centers = np.sqrt(r_bins[1:] * r_bins[:-1])  # Geometric mean for log bins
    hist, _ = np.histogram(r, bins=r_bins)
    
    # Convert to number density ρ(r) properly
    # Volume of spherical shell: V = (4/3)π(r₂³ - r₁³) 
    shell_volumes = (4/3) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    # Sample density = (particles in shell) / (shell volume) / (total particles)
    # This gives density per unit volume, normalized so ∫ρ(r)dV = 1
    density_sample = hist / shell_volumes / n_samples
    
    # Only plot bins with reasonable statistics (>5 particles)
    good_bins = hist >= 5
    r_centers = r_centers[good_bins]
    density_sample = density_sample[good_bins]
    
    # Theoretical density - match the data range
    r_theory = np.logspace(-1.5, 1, 200)
    density_theory = plummer_density(r_theory, a)
    
    # Plot with modern colors and styling
    ax.loglog(r_centers/a, density_sample * a**3, 'o', 
             color=colors['primary'], alpha=0.8, markersize=6, 
             markerfacecolor=colors['primary'], markeredgecolor='white',
             markeredgewidth=0.5, label='Sample Data')
    ax.loglog(r_theory/a, density_theory * a**3, 
             color=colors['secondary'], linewidth=3.5, alpha=0.9,
             label='Theoretical: $(1+r^2/a^2)^{-5/2}$')
    
    # Professional styling with MASSIVE fonts
    ax.set_xlabel('Radius (r/a)', fontsize=22, color=colors['dark'], labelpad=18, weight='bold')
    ax.set_ylabel('Normalized Density ($\\rho a^3$)', fontsize=22, color=colors['dark'], labelpad=18, weight='bold')
    ax.set_title('Density Profile Verification\n(Log-Log Scale)', 
                fontsize=24, pad=30, color=colors['dark'], weight='bold')
    
    # Professional legend with MASSIVE font
    ax.legend(fontsize=18, frameon=True, fancybox=True,
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95,
             loc='upper right')
    ax.grid(True, alpha=0.25, linewidth=0.5, color=colors['neutral'])
    ax.tick_params(axis='both', which='major', labelsize=18, 
                  colors=colors['neutral'], width=1.5, length=8)
    ax.tick_params(axis='both', which='minor', labelsize=16, 
                  colors=colors['neutral'], width=1.0, length=4)
    
    # Set proper axis limits - consistent with colorbar 0 to 4
    ax.set_xlim(0.1, 4.0)  # Match colorbar range 0-4
    ax.set_ylim(0.002, 1.0)  # Max y-limit set to 1
    
    # Bottom left: Enclosed mass (CDF) with modern styling
    ax = axes[1, 0]
    ax.set_facecolor('white')
    
    # Calculate enclosed mass fraction (this is the CDF!)
    r_sorted = np.sort(r)
    mass_enclosed = np.arange(1, len(r_sorted)+1) / len(r_sorted)
    
    # Theoretical enclosed mass
    r_theory = np.linspace(0.01, 8, 300)
    mass_theory = plummer_mass(r_theory, a)
    
    # Plot with modern styling
    ax.plot(r_sorted/a, mass_enclosed, 
           color=colors['primary'], linewidth=2.5, alpha=0.8,
           label='Sample CDF', zorder=5)
    ax.plot(r_theory/a, mass_theory, 
           color=colors['secondary'], linewidth=3.5, alpha=0.9,
           label='Theoretical CDF', zorder=10)
    
    # Mark half-mass radius with educational annotation - MOVED TO RIGHT
    ax.axhline(0.5, color=colors['accent'], linestyle='--', alpha=0.8, linewidth=2)
    ax.axvline(1.3, color=colors['accent'], linestyle='--', alpha=0.8, linewidth=2)
    ax.text(4.0, 0.52, '$r_{1/2} = 1.3a$\n(Half-mass radius)', 
           fontsize=12, color=colors['accent'], weight='medium',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor=colors['accent'], alpha=0.9))
    
    # Professional styling with MASSIVE fonts
    ax.set_xlabel('Radius (r/a)', fontsize=22, color=colors['dark'], labelpad=18, weight='bold')
    ax.set_ylabel('Enclosed Mass Fraction', fontsize=22, color=colors['dark'], labelpad=18, weight='bold')
    ax.set_title('Cumulative Distribution Function\n(Mass Profile)', 
                fontsize=24, pad=30, color=colors['dark'], weight='bold')
    
    # Professional legend with MASSIVE font
    ax.legend(fontsize=18, frameon=True, fancybox=True,
             edgecolor=colors['neutral'], facecolor='white', framealpha=0.95,
             loc='lower right')
    ax.grid(True, alpha=0.25, linewidth=0.5, color=colors['neutral'])
    ax.tick_params(axis='both', which='major', labelsize=18, 
                  colors=colors['neutral'], width=1.5, length=8)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.02)
    
    # Bottom right: 3D structure with modern styling
    ax = axes[1, 1]
    ax.set_facecolor('white')
    
    # Create density-based visualization
    from matplotlib.patches import Circle
    
    # Plot as scatter with consistent REVERSED viridis colormap - FIXED 0 to 4
    scatter = ax.scatter(x, z, s=5, alpha=0.7, c=r, cmap='viridis_r', 
                        rasterized=True, edgecolors='none', vmin=0.0, vmax=4.0)
    
    # Add reference circles with same format as top plot
    for i, radius in enumerate([a, 2*a, 3*a]):
        circle = Circle((0, 0), radius, fill=False, linestyle='--', 
                       alpha=0.7, color=colors['neutral'], linewidth=3.0)  # Match top plot
        ax.add_patch(circle)
        if i == 1:  # Only label middle circle like top plot
            ax.text(radius*0.7, radius*0.7, f'r = {radius:.0f}a', 
                   fontsize=14, ha='center', va='center', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.9, edgecolor=colors['neutral'], linewidth=1.5))
    
    # Professional styling with MASSIVE fonts
    ax.set_xlabel('Position x/a', fontsize=22, color=colors['dark'], labelpad=18, weight='bold')
    ax.set_ylabel('Position z/a', fontsize=22, color=colors['dark'], labelpad=18, weight='bold')
    ax.set_title('Edge-On View: Spherical Structure\n(Color = Distance from Center)', 
                fontsize=24, pad=30, color=colors['dark'], weight='bold')
    ax.set_aspect('equal')
    
    # Set limits and styling
    max_extent = 4*a
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.grid(True, alpha=0.25, linewidth=0.5, color=colors['neutral'])
    ax.tick_params(axis='both', which='major', labelsize=18, 
                  colors=colors['neutral'], width=1.5, length=8)
    
    # Add colorbar with FIXED LIMITS 0-5 and MASSIVE fonts
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Distance from Center (r/a)', fontsize=18, 
                   color=colors['dark'], labelpad=18, weight='bold')
    cbar.ax.tick_params(labelsize=16, colors=colors['neutral'])
    # FORCE colorbar limits to 0-4 for consistency
    cbar.mappable.set_clim(vmin=0.0, vmax=4.0)
    
    # Style all axes consistently
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color(colors['neutral'])
            spine.set_linewidth(0.8)
    
    # Professional main title
    plt.suptitle('Plummer Sphere Sampling: From 1D CDF to 3D Stellar Positions', 
                fontsize=20, y=0.94, color=colors['dark'], weight='semibold')  # Moved down from 0.96
    
    # Educational subtitle - moved closer to main title
    fig.text(0.5, 0.91, 'Inverse transform sampling creates realistic star cluster initial conditions',
             ha='center', va='center', fontsize=15, color=colors['neutral'], style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, hspace=0.45, wspace=0.30)  # Reduced top margin from 0.85 to 0.82
    
    # Save figure with high quality
    plt.savefig(os.path.join(FIGURES_DIR, '14_plummer_sphere_spatial_sampling.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset matplotlib style to defaults
    plt.rcParams.update(plt.rcParamsDefault)
    
    print("\nVerification:")
    print("-" * 30)
    print(f"Central density (r < 0.1a): {np.sum(r < 0.1*a)/n_samples*100:.1f}% of particles")
    print(f"Within 1 scale radius: {np.sum(r < a)/n_samples*100:.1f}% of particles")  
    print(f"Within 2 scale radii: {np.sum(r < 2*a)/n_samples*100:.1f}% of particles")
    print(f"Mean radius: {np.mean(r)/a:.2f} a")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_demos():
    """
    Run all computational demos in sequence.
    """
    print("STATISTICAL FOUNDATIONS - COMPLETE COMPUTATIONAL DEMOS")
    print("=" * 70)
    print("Running all demonstrations from the Statistical Foundations module...")
    print("This may take a few moments - generating plots and calculations.")
    print("=" * 70)
    
    # Run all demonstrations
    demo_temperature_emergence()
    demo_pressure_emergence() 
    demo_central_limit_theorem()
    demo_maximum_entropy()
    demo_correlation()
    demo_marginalization()
    demo_ergodicity()
    demo_law_of_large_numbers()
    demo_error_propagation()
    demo_bayesian_learning()
    demo_monte_carlo_pi()
    demo_random_sampling()
    demo_power_law_sampling()
    demo_plummer_sampling()
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED!")
    print("=" * 70)
    print("\nYou have now seen all the key statistical concepts in action:")
    print("✓ Temperature emergence from distributions")
    print("✓ Pressure from molecular chaos")
    print("✓ Central Limit Theorem creating Gaussians")  
    print("✓ Maximum entropy distributions")
    print("✓ Correlation effects on joint distributions")
    print("✓ Marginalization reducing dimensions")
    print("✓ Ergodic vs non-ergodic behavior")
    print("✓ Law of large numbers convergence")
    print("✓ Error propagation through calculations")
    print("✓ Bayesian learning from data")
    print("✓ Monte Carlo methods")
    print("✓ Random sampling techniques") 
    print("✓ Power law distributions (stellar IMF)")
    print("✓ Spatial sampling (Plummer spheres)")
    print("\nThese are the foundations for all your projects!")
    
    # Print summary of saved figures
    print(f"\n📁 All figures saved to: {FIGURES_DIR}")
    print("📊 Figure files created:")
    figures_created = [
        "01_temperature_emergence_from_statistics.png",
        "02_pressure_emergence_from_chaos.png", 
        "03_central_limit_theorem_in_action.png",
        "04_maximum_entropy_distributions.png",
        "05_correlation_and_velocity_ellipsoids.png",
        "06_marginalization_visualization.png",
        "07_ergodic_vs_nonergodic_systems.png",
        "08_law_of_large_numbers_convergence.png",
        "09_error_propagation_through_calculations.png",
        "10_bayesian_learning_and_inference.png",
        "11_monte_carlo_pi_estimation.png",
        "12_random_sampling_methods.png",
        "13_power_law_distribution_sampling.png",
        "14_plummer_sphere_spatial_sampling.png"
    ]
    for i, fig_name in enumerate(figures_created, 1):
        print(f"   {i:2d}. {fig_name}")
    print(f"\n✅ Total: {len(figures_created)} publication-quality figures created!")

def list_available_demos():
    """List all available demonstration functions."""
    demos = [
        ("demo_temperature_emergence", "Temperature emergence from particle statistics"),
        ("demo_pressure_emergence", "Pressure from chaotic molecular collisions"),
        ("demo_central_limit_theorem", "Central Limit Theorem in action"),
        ("demo_maximum_entropy", "Maximum entropy distributions"),
        ("demo_correlation", "Correlation and velocity ellipsoids"),
        ("demo_marginalization", "Marginalization visualization"),
        ("demo_ergodicity", "Ergodic vs non-ergodic systems"),
        ("demo_law_of_large_numbers", "Law of large numbers convergence"),
        ("demo_error_propagation", "Error propagation through calculations"),
        ("demo_bayesian_learning", "Bayesian learning and inference"),
        ("demo_monte_carlo_pi", "Monte Carlo π estimation"),
        ("demo_random_sampling", "Random sampling methods"),
        ("demo_power_law_sampling", "Power law distribution sampling"),
        ("demo_plummer_sampling", "Plummer sphere spatial sampling"),
    ]
    
    print("📊 Available Statistical Foundations Demos:")
    print("=" * 50)
    for i, (func_name, description) in enumerate(demos, 1):
        print(f"{i:2d}. {func_name:<30} - {description}")
    print("\nUsage:")
    print("import statistical_foundations")
    print("statistical_foundations.demo_temperature_emergence()  # Run individual demo")
    print("statistical_foundations.run_all_demos()              # Run all demos")
    print(f"\n📁 Figures will be saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    run_all_demos()