"""
Module 2a: Statistical Mechanics Visualizations
Using ASTR596 Plotting Utilities for consistent course styling
Author: Anna Rosen
Course: ASTR 596: Modeling the Universe
Date: January 2025

This module creates pedagogical figures for the Statistical Foundations module,
demonstrating key concepts in statistical mechanics and thermodynamics.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.stats as stats
from importlib import import_module

# Import ASTR596 plotting utilities
sys.path.insert(0, '.')
plotting_utils = import_module('astr596-plotting-utils')
ASTR596Plotter = plotting_utils.ASTR596Plotter


class StatMechVisualizations:
    """
    Generate all figures for Module 2a: Statistical Foundations
    Using ASTR596 plotting utilities for consistent styling.
    """
    
    def __init__(self, save_dir='./module2a_figures/'):
        """
        Initialize with ASTR596 plotter and physical constants.
        
        Parameters:
        -----------
        save_dir : str
            Directory for saving figures
        """
        # Initialize the ASTR596 plotter with default style
        self.plotter = ASTR596Plotter(style='default', save_dir=save_dir)
        
        # Use physical constants from the plotter
        self.k_B = self.plotter.constants['k_B']  # Boltzmann constant
        self.m_H = self.plotter.constants['m_H']  # Hydrogen mass
        
        # We'll use the plotter's color palettes throughout
        print(f"Initialized StatMech visualizations with save directory: {save_dir}")
    
    def fig1_temperature_emergence(self):
        """
        Figure 1: Temperature Emergence from Ensemble
        Shows how temperature becomes meaningful as N increases.
        """
        # Create figure with better spacing to prevent overlap
        fig, gs = self.plotter.create_figure_with_gridspec(
            figsize=(16, 11), nrows=3, ncols=3, hspace=0.45, wspace=0.4
        )
        
        N_values = [1, 2, 5, 10, 50, 100, 500, 1000, 10000]
        T_true = 300  # K
        sigma = np.sqrt(self.k_B * T_true / self.m_H)
        
        for idx, N in enumerate(N_values):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            
            # Generate velocities
            np.random.seed(42 + idx)
            velocities = np.random.normal(0, sigma, N) / 1e5  # Convert to km/s
            
            if N == 1:
                # Single particle visualization - cleaner
                ax.scatter([velocities[0]], [0], s=400, 
                          color=self.plotter.colors_main[2], zorder=5,
                          edgecolor='black', linewidth=2, alpha=0.8)
                ax.arrow(0, 0, velocities[0], 0, head_width=0.15, 
                        head_length=abs(velocities[0])*0.05, 
                        fc=self.plotter.colors_main[7], 
                        ec=self.plotter.colors_main[7], linewidth=2, alpha=0.7)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlim(-20, 20)
                ax.set_title(f'N = 1: Single particle\n' + 
                           f'v = {velocities[0]:.1f} km/s\n' +
                           '(No temperature!)', fontsize=11, fontweight='bold', color='red')
                ax.set_xlabel('Velocity (km/s)', fontsize=10)
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.grid(True, alpha=0.3, axis='x', linestyle=':')
                
            elif N == 2:
                # Two particles - clearer visualization
                colors = [self.plotter.colors_categorical[0], self.plotter.colors_categorical[1]]
                for i, v in enumerate(velocities):
                    y_pos = 0.3 - i * 0.6
                    ax.scatter([v], [y_pos], s=300, color=colors[i], 
                             edgecolor='black', linewidth=1.5, alpha=0.8, zorder=5)
                    ax.arrow(0, y_pos, v, 0, head_width=0.1, 
                            head_length=abs(v)*0.05, fc=colors[i], 
                            ec=colors[i], alpha=0.5, linewidth=1.5)
                    ax.text(v, y_pos + 0.15, f'{v:.1f}', ha='center', fontsize=9)
                
                mean_v = np.mean(velocities)
                ax.axvline(mean_v, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlim(-20, 20)
                ax.set_title(f'N = 2: Two particles\n' +
                           f'Mean = {mean_v:.1f} km/s\n' +
                           '(Temperature emerging)', fontsize=11, fontweight='bold')
                ax.set_xlabel('Velocity (km/s)', fontsize=10)
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.grid(True, alpha=0.3, axis='x', linestyle=':')
                
            else:
                # Histogram for larger N - cleaner with better colors
                bins = max(15, min(25, int(np.sqrt(N))))
                counts, bin_edges, patches = ax.hist(velocities, bins=bins, 
                                            density=True, alpha=0.8, 
                                            edgecolor='black', linewidth=0.5)
                
                # Use single color scheme for clarity
                for patch in patches:
                    patch.set_facecolor(self.plotter.colors_main[3])
                    patch.set_alpha(0.7)
                
                # Overlay theoretical Maxwell-Boltzmann
                v_range = np.linspace(-20, 20, 200)
                theory = stats.norm.pdf(v_range, 0, sigma/1e5)
                ax.plot(v_range, theory, color='black', 
                       lw=2.5, label='Theory', alpha=0.9, linestyle='--')
                
                # Measure temperature from variance
                T_measured = self.m_H * np.var(velocities * 1e5) / self.k_B
                T_error = T_measured / np.sqrt(2*N)
                
                # Better title formatting
                ax.set_title(f'N = {N:,}\n' + 
                           r'$T = $' + f'{T_measured:.0f} ± {T_error:.0f} K', 
                           fontsize=11, fontweight='bold')
                
                ax.set_xlim(-20, 20)
                ax.set_xlabel('Velocity (km/s)', fontsize=10)
                ax.set_ylabel('Probability Density', fontsize=10)
                
                # Place legend carefully to avoid data
                if idx == 6:  # First large N plot
                    ax.legend(loc='upper left', fontsize=9, framealpha=0.95,
                            edgecolor='black', fancybox=False)
                
                ax.grid(True, alpha=0.3, linestyle=':')
        
        # Main title with better positioning - increased y to add space
        fig.suptitle('Temperature Emerges from Ensemble Statistics', 
                    fontsize=18, fontweight='bold', y=0.99)
        
        # Subtitle - also moved up
        fig.text(0.5, 0.95, 
                '"Temperature is a collective property - it doesn\'t exist for one particle"',
                ha='center', fontsize=13, style='italic', color='#444')
        
        # Educational insight box
        fig.text(0.5, 0.01, 
                r'Key Insight: As N increases, fluctuations decrease as $1/\sqrt{N}$ → Temperature becomes well-defined',
                ha='center', fontsize=12, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', 
                         edgecolor='#1565C0', linewidth=2))
        
        return fig
    
    def fig2_pressure_from_chaos(self):
        """
        Figure 2: Pressure Emerges from Random Collisions
        Shows how individual chaotic collisions average to steady pressure.
        """
        fig, gs = self.plotter.create_figure_with_gridspec(
            figsize=(15, 9), nrows=2, ncols=2, hspace=0.35, wspace=0.3
        )
        
        T = 300  # K
        sigma = np.sqrt(self.k_B * T / self.m_H)
        
        # Panel 1: Individual particle trajectories
        ax1 = fig.add_subplot(gs[0, :])
        n_particles = 5
        time_steps = 100
        
        colors = self.plotter.get_color_sequence(n_particles, 'categorical')
        for i in range(n_particles):
            np.random.seed(i)
            velocities = np.random.normal(0, sigma, time_steps) / 1e5
            momentum = self.m_H * np.abs(velocities) * 1e5
            time = np.arange(time_steps)
            ax1.plot(time, momentum * 1e19, alpha=0.7, linewidth=1.5, 
                    label=f'Particle {i+1}', color=colors[i])
        
        self.plotter.apply_style(ax1, 
                               xlabel='Time (arbitrary units)',
                               ylabel=r'Momentum Transfer ($\times 10^{-19}$ g$\cdot$cm/s)',
                               title='Individual Particle Collisions: Pure Chaos',
                               legend=True)
        
        # Panel 2: Distribution of momentum transfers
        ax2 = fig.add_subplot(gs[1, 0])
        all_momentum = []
        for _ in range(10000):
            v = np.random.normal(0, sigma)
            all_momentum.append(2 * self.m_H * np.abs(v))
        
        all_momentum = np.array(all_momentum) * 1e19
        ax2.hist(all_momentum, bins=50, density=True, alpha=0.7, 
                color=self.plotter.colors_main[3], edgecolor='black', linewidth=0.5)
        mean_val = np.mean(all_momentum)
        ax2.axvline(mean_val, color=self.plotter.colors_main[8], linestyle='--', 
                   linewidth=2, label=rf'Mean = {mean_val:.2f}')
        
        self.plotter.apply_style(ax2,
                               xlabel=r'Momentum Transfer ($\times 10^{-19}$ g$\cdot$cm/s)',
                               ylabel='Probability Density',
                               title='Distribution of Individual Transfers',
                               legend=True)
        
        # Panel 3: Convergence to steady pressure
        ax3 = fig.add_subplot(gs[1, 1])
        n_samples_list = [10, 100, 1000, 10000]
        colors = self.plotter.get_color_sequence(len(n_samples_list), 'temperature')
        
        for idx, n_samples in enumerate(n_samples_list):
            np.random.seed(100 + idx)
            averages = []
            for i in range(1, min(max(1, n_samples + 1), 1000)):
                sample = np.random.normal(0, sigma, i)
                avg_pressure = np.mean(self.m_H * sample**2)
                averages.append(avg_pressure)
            
            x = np.arange(1, len(averages) + 1)
            ax3.semilogx(x, np.array(averages) * 1e10, alpha=0.7, 
                        linewidth=1.5, label=rf'$N = {n_samples}$',
                        color=colors[idx])
        
        theoretical = self.k_B * T * 1e10
        ax3.axhline(theoretical, color=self.plotter.colors_main[0], 
                   linestyle='--', linewidth=2, label=r'Theory: $k_BT$')
        
        self.plotter.apply_style(ax3,
                               xlabel='Number of Particles Averaged',
                               ylabel=r'Pressure per Particle ($\times 10^{-10}$ erg/cm$^3$)',
                               title='Convergence to Steady Pressure',
                               legend=True)
        
        fig.suptitle(r'From Molecular Chaos to Macroscopic Pressure: $P = nk_BT$',
                    fontsize=16, fontweight='bold', y=0.96)
        
        return fig
    
    def fig2b_wall_collision_visual(self):
        """
        Figure 2b: Gas Molecules with Wall - Isotropic Velocities
        Shows gas molecules with isotropic velocity distribution hitting wall.
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        
        T = 300  # K
        sigma = np.sqrt(self.k_B * T / self.m_H)
        
        # Wall visualization
        ax.axvline(x=0, color=self.plotter.colors_main[0], linewidth=15, alpha=0.8)
        ax.text(-0.5, 5, 'WALL', rotation=90, fontsize=14, fontweight='bold',
                va='center', ha='center', color='white')
        
        # Particles with ISOTROPIC velocities
        n_particles = 60
        np.random.seed(42)
        
        for i in range(n_particles):
            x = np.random.uniform(0.5, 12)
            y = np.random.uniform(0, 10)
            
            # Generate isotropic velocities using Maxwell-Boltzmann distribution
            # Speed from Rayleigh distribution (3D speed distribution)
            speed = np.random.rayleigh(sigma/1e5) * 0.0003
            # Random angle for 2D projection
            angle = np.random.uniform(0, 2*np.pi)
            
            # Velocity components (scaled for visualization)
            vx = speed * np.cos(angle) * 10
            vy = speed * np.sin(angle) * 10
            
            # Color particles based on their kinetic energy
            ke = 0.5 * self.m_H * (vx**2 + vy**2)
            color_intensity = min(1, ke / (3 * self.k_B * T) * 1e10)
            particle_color = self.plotter.colors_temperature[int(color_intensity * (len(self.plotter.colors_temperature) - 1))]
            
            # Draw particle
            circle = Circle((x, y), 0.2, color=particle_color, alpha=0.7)
            ax.add_patch(circle)
            
            # Draw velocity arrow
            ax.arrow(x, y, vx, vy, head_width=0.15, head_length=0.08, 
                     fc=self.plotter.colors_main[7], ec=self.plotter.colors_main[7], 
                     alpha=0.6, linewidth=0.5)
        
        # Add velocity distribution inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
        
        # Show velocity distribution
        velocities = []
        for _ in range(1000):
            speed = np.random.rayleigh(sigma/1e5) * 0.0003
            angle = np.random.uniform(0, 2*np.pi)
            vx = speed * np.cos(angle) * 10
            velocities.append(vx)
        
        axins.hist(velocities, bins=30, density=True, alpha=0.7, 
                  color=self.plotter.colors_main[3], edgecolor='black', linewidth=0.5)
        axins.set_xlabel(r'$v_x$ (scaled)', fontsize=9)
        axins.set_ylabel('Probability', fontsize=9)
        axins.set_title('Velocity Distribution', fontsize=10)
        axins.tick_params(labelsize=8)
        
        ax.set_xlim(-1, 13)
        ax.set_ylim(-0.5, 10.5)
        ax.set_aspect('equal')
        
        self.plotter.apply_style(ax,
                               xlabel='Distance from Wall',
                               ylabel='Position',
                               title='Isotropic Gas Molecules Creating Pressure on Wall',
                               grid=False)
        
        # Add text explaining isotropic velocities
        ax.text(9, 1, 
               'Isotropic velocities:\nRandom directions\nMaxwell-Boltzmann speeds',
               fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(r'Molecular Bombardment: Isotropic Velocities Create Pressure',
                    fontsize=16, fontweight='bold')
        
        return fig
    
    def fig3_central_limit_theorem(self):
        """
        Figure 3: Central Limit Theorem in Action
        Shows convergence to Gaussian for any distribution.
        """
        fig, gs = self.plotter.create_figure_with_gridspec(
            figsize=(16, 11), nrows=3, ncols=3, hspace=0.4, wspace=0.35
        )
        
        n_samples = 10000
        
        # Different starting distributions
        distributions = [
            ('Uniform', lambda n: np.random.uniform(-1, 1, n), 
             self.plotter.colors_categorical[0]),
            ('Exponential', lambda n: np.random.exponential(1, n), 
             self.plotter.colors_categorical[1]),
            ('Bimodal', lambda n: np.concatenate([
                np.random.normal(-2, 0.3, n//2),
                np.random.normal(2, 0.3, n//2)
            ]), self.plotter.colors_categorical[2])
        ]
        
        N_values = [1, 2, 30]  # Number of variables to sum
        
        for col_idx, (name, dist_func, color) in enumerate(distributions):
            for row_idx, N in enumerate(N_values):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                
                if N == 1:
                    # Original distribution
                    samples = dist_func(n_samples)
                    ax.hist(samples, bins=50, density=True, alpha=0.7, 
                           color=color, edgecolor='black', linewidth=0.5)
                    ax.set_title(f'{name} Distribution (N = {N})', 
                               fontweight='bold', fontsize=11)
                    ax.set_ylim(0, 0.6)  # Consistent y-limits
                    
                else:
                    # Sum of N samples
                    sums = []
                    for _ in range(n_samples // N):
                        sums.append(np.sum(dist_func(N)))
                    
                    sums = np.array(sums)
                    # Normalize for comparison
                    std_sums = np.std(sums)
                    if std_sums > 0:
                        sums_normalized = (sums - np.mean(sums)) / std_sums
                    else:
                        sums_normalized = sums - np.mean(sums)
                    
                    # Plot histogram
                    counts, bins_edges, patches = ax.hist(sums_normalized, bins=40, 
                                                          density=True, alpha=0.7, 
                                                          edgecolor='black', linewidth=0.5)
                    
                    # Color by deviation from Gaussian
                    x_mid = (bins_edges[:-1] + bins_edges[1:]) / 2
                    gaussian_vals = stats.norm.pdf(x_mid)
                    
                    # Create gradient colormap
                    cmap = self.plotter.create_colormap([
                        self.plotter.color_reject, 
                        self.plotter.color_highlight,
                        self.plotter.color_accept
                    ])
                    
                    for count, patch, gauss in zip(counts, patches, gaussian_vals):
                        if gauss > 1e-10:
                            deviation = abs(count - gauss) / gauss
                        else:
                            deviation = 0
                        color_intensity = max(0, min(1, 1 - deviation))
                        patch.set_facecolor(cmap(color_intensity))
                    
                    # Overlay Gaussian
                    x = np.linspace(-4, 4, 100)
                    ax.plot(x, stats.norm.pdf(x), color=self.plotter.colors_main[8], 
                           lw=2.5, label='Standard Gaussian', alpha=0.9)
                    
                    # Simple title without KS statistic
                    ax.set_title(f'Sum of {N} {name}', 
                               fontweight='bold', fontsize=11)
                    
                    # Set consistent y-limits for normalized distributions
                    ax.set_ylim(0, 0.45)
                    ax.set_xlim(-4, 4)
                    
                    if col_idx == 2 and row_idx == 1:  # Only one legend
                        ax.legend(loc='upper right', fontsize=9)
                
                self.plotter.apply_style(ax, xlabel='Normalized Value', 
                                       ylabel='Probability Density')
                
                # Add arrow between rows for middle column
                if row_idx < 2 and col_idx == 1:
                    ax.annotate('', xy=(0.5, -0.45), xytext=(0.5, -0.25),
                              xycoords='axes fraction',
                              arrowprops=dict(arrowstyle='->', lw=2, 
                                            color=self.plotter.colors_main[8]))
        
        fig.suptitle(r'Central Limit Theorem: All Roads Lead to Gaussian',
                    fontsize=16, fontweight='bold', y=0.96)
        
        return fig
    
    def fig4_maxwell_boltzmann_3d(self):
        """
        Figure 4: Maxwell-Boltzmann Distribution
        Shows velocity and energy distributions at different temperatures.
        Scientifically accurate representations for educational purposes.
        """
        fig, gs = self.plotter.create_figure_with_gridspec(
            figsize=(16, 10), nrows=2, ncols=3, hspace=0.3, wspace=0.35
        )
        
        T_values = [100, 300, 1000]  # Kelvin
        colors = self.plotter.colors_temperature[:3]  # Cold to hot
        
        # Panel 1: 1D velocity component - properly normalized
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Calculate appropriate x-range based on highest temperature
        sigma_max = np.sqrt(self.k_B * max(T_values) / self.m_H) / 1e5  # km/s for T=1000K
        v_limit = 4 * sigma_max  # Show ±4σ of widest distribution
        v = np.linspace(-v_limit, v_limit, 1000)  # km/s
        
        for T, color in zip(T_values, colors):
            # 1D Maxwell-Boltzmann: f(v_x) = sqrt(m/2πkT) * exp(-mv_x^2/2kT)
            sigma_1d = np.sqrt(self.k_B * T / self.m_H) / 1e5  # km/s
            # Using scipy's normal distribution which is already normalized
            f_1d = stats.norm.pdf(v, 0, sigma_1d)
            
            # Scale for display (probability density per km/s)
            ax1.plot(v, f_1d * 1e3, label=rf'$T = {T}$ K', linewidth=2.5, color=color, alpha=0.8)
            ax1.fill_between(v, 0, f_1d * 1e3, alpha=0.2, color=color)
        
        ax1.set_xlim(-v_limit, v_limit)  # Adaptive limits
        ax1.set_xlabel(r'Velocity $v_x$ (km/s)', fontsize=11)
        ax1.set_ylabel(r'Probability Density ($\times 10^{-3}$ per km/s)', fontsize=11)
        ax1.set_title('1D Velocity Component', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        # Panel 2: 3D speed distribution - Maxwell speed distribution
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculate appropriate speed range based on highest temperature
        v_thermal_max = np.sqrt(2 * self.k_B * max(T_values) / self.m_H) / 1e5  # km/s
        v_speed_limit = 5 * v_thermal_max  # Show up to 5× thermal velocity
        v_speed = np.linspace(0, v_speed_limit, 1000)  # km/s
        
        for T, color in zip(T_values, colors):
            # Maxwell speed distribution: f(v) = 4π * (m/2πkT)^(3/2) * v^2 * exp(-mv^2/2kT)
            v_thermal = np.sqrt(2 * self.k_B * T / self.m_H) / 1e5  # thermal velocity in km/s
            
            # Use Maxwell distribution from scipy.stats
            maxwell = stats.maxwell(scale=v_thermal/np.sqrt(2))
            f_speed = maxwell.pdf(v_speed)
            
            # Scale for display
            ax2.plot(v_speed, f_speed * 1e3, label=rf'$T = {T}$ K', 
                    linewidth=2.5, color=color, alpha=0.8)
            ax2.fill_between(v_speed, 0, f_speed * 1e3, alpha=0.2, color=color)
            
            # Mark characteristic speeds
            v_mp = v_thermal  # Most probable speed = sqrt(2kT/m)
            v_mean = v_thermal * np.sqrt(8/np.pi)  # Mean speed
            v_rms = v_thermal * np.sqrt(3/2)  # RMS speed
            
            ax2.axvline(v_mp, color=color, linestyle=':', alpha=0.5, linewidth=1)
        
        ax2.set_xlim(0, v_speed_limit)  # Adaptive limits
        ax2.set_xlabel(r'Speed $|\vec{v}|$ (km/s)', fontsize=11)
        ax2.set_ylabel(r'Probability Density ($\times 10^{-3}$ per km/s)', fontsize=11)
        ax2.set_title('3D Speed Distribution (Maxwell)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle=':')
        
        # Panel 3: Energy distribution - Maxwell-Boltzmann energy distribution
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Plot in units of kT for clarity
        E_over_kT = np.linspace(0, 8, 1000)  # Energy in units of kT
        
        # Maxwell-Boltzmann energy distribution in units of kT
        # P(E/kT) = 2 * sqrt(E/kT / π) * exp(-E/kT)
        # This is universal - same shape for all temperatures!
        f_E = 2 * np.sqrt(E_over_kT / np.pi) * np.exp(-E_over_kT)
        
        # Plot the universal curve
        ax3.plot(E_over_kT, f_E, 'k-', linewidth=3, label='Universal curve', alpha=0.8)
        ax3.fill_between(E_over_kT, 0, f_E, alpha=0.3, color='gray')
        
        # Mark important points
        ax3.axvline(1, color='red', linestyle=':', alpha=0.7, linewidth=2, label=r'$E = k_BT$')
        ax3.axvline(0.5, color='blue', linestyle=':', alpha=0.7, linewidth=2, label=r'$E = \frac{1}{2}k_BT$ (peak)')
        
        # Add temperature scale on top x-axis (in CGS units)
        ax3_top = ax3.twiny()
        ax3_top.set_xlabel('Energy at T = 300K (×10⁻¹⁴ erg)', fontsize=10, color='gray')
        # k_B = 1.38e-16 erg/K, so k_B*T at 300K = 4.14e-14 erg
        ax3_top.set_xlim(0, 8 * self.k_B * 300 * 1e14)  # Scale to 10^-14 erg
        
        ax3.set_xlim(0, 8)
        ax3.set_ylim(0, 0.5)
        ax3.set_xlabel(r'Energy / $k_BT$ (dimensionless)', fontsize=11)
        ax3.set_ylabel('Probability Density', fontsize=11)
        ax3.set_title('Kinetic Energy Distribution', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle=':')
        
        # Panels 4-6: 2D velocity space visualization with adaptive limits
        for idx, (T, color) in enumerate(zip(T_values, colors)):
            ax = fig.add_subplot(gs[1, idx])
            
            # Create 2D grid with temperature-dependent limits
            sigma_km = np.sqrt(self.k_B * T / self.m_H) / 1e5  # km/s
            v_limit = 3 * sigma_km  # Show ±3σ
            vx = vy = np.linspace(-v_limit, v_limit, 100)
            VX, VY = np.meshgrid(vx, vy)
            
            # 2D Maxwell-Boltzmann distribution
            sigma = np.sqrt(self.k_B * T / self.m_H) / 1e5  # km/s
            # Prevent numerical underflow
            exponent = np.clip(-(VX**2 + VY**2) / (2 * sigma**2), -700, 700)
            Z = (1 / (2 * np.pi * sigma**2)) * np.exp(exponent)
            
            # Create custom colormap for each temperature
            cmap = self.plotter.create_colormap(['white', color])
            
            # Plot as contour
            levels = np.linspace(0, Z.max(), 10)
            ax.contourf(VX, VY, Z, levels=levels, cmap=cmap, alpha=0.8)
            ax.contour(VX, VY, Z, levels=5, colors='black', linewidths=0.5, alpha=0.5)
            
            # Add RMS speed circle
            v_rms = np.sqrt(3 * self.k_B * T / self.m_H) / 1e5
            circle = Circle((0, 0), v_rms, fill=False, 
                          edgecolor=self.plotter.colors_main[0], 
                          linewidth=2, linestyle='--', label=r'$v_{\rm rms}$')
            ax.add_patch(circle)
            
            ax.set_xlabel(r'$v_x$ (km/s)', fontsize=11)
            ax.set_ylabel(r'$v_y$ (km/s)', fontsize=11)
            ax.set_title(rf'2D Velocity Space: $T = {T}$ K', fontsize=11, fontweight='bold')
            if idx == 0:
                ax.legend(loc='upper right', fontsize=9)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle=':')
        
        fig.suptitle(r'Maxwell-Boltzmann: Temperature Sets the Velocity Scale',
                    fontsize=16, fontweight='bold', y=0.97)
        
        return fig
    
    def fig5_ergodicity_demo(self):
        """
        Figure 5: Ergodicity - Time Average Equals Ensemble Average
        Clear demonstration with coin flip and oscillator examples.
        """
        fig, gs = self.plotter.create_figure_with_gridspec(
            figsize=(16, 10), nrows=2, ncols=3, hspace=0.35, wspace=0.35
        )
        
        # Top panel: Coin flip convergence
        ax1 = fig.add_subplot(gs[0, :])
        
        n_flips = 1000
        np.random.seed(42)
        flips = np.random.choice([0, 1], n_flips)
        time_avg = np.cumsum(flips) / np.arange(1, n_flips + 1)
        
        # Plot with better styling
        ax1.fill_between(range(n_flips), 
                        0.5 - 1/np.sqrt(np.arange(1, n_flips + 1)),
                        0.5 + 1/np.sqrt(np.arange(1, n_flips + 1)),
                        alpha=0.2, color='gray', 
                        label=r'$1/\sqrt{N}$ bounds')
        
        ax1.plot(time_avg, color=self.plotter.colors_main[2], linewidth=2.5, 
                label='Time Average (one coin)', alpha=0.9)
        ax1.axhline(0.5, color='black', linestyle='--', 
                   linewidth=2, label='Theoretical = 0.5', alpha=0.7)
        
        ax1.set_xlabel('Number of Flips', fontsize=11)
        ax1.set_ylabel('Running Average', fontsize=11)
        ax1.set_title('Coin Flip: Time Average Converges', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', framealpha=0.95)
        ax1.set_xlim(0, n_flips)  # Explicitly set x-limits
        ax1.set_ylim(0.2, 0.8)  # Wider y-limits to see early fluctuations better
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        # Example 2: Harmonic oscillator - Time evolution
        ax2 = fig.add_subplot(gs[1, 0])
        
        t = np.linspace(0, 20, 1000)
        E_total = 1.0
        omega = 2 * np.pi
        phase = np.pi/4  # Fixed phase for reproducibility
        
        KE = E_total * np.sin(omega * t + phase)**2
        time_avg_KE = np.cumsum(KE) / np.arange(1, len(KE) + 1)
        
        ax2.plot(t, KE, color=self.plotter.colors_main[3], alpha=0.3, 
                linewidth=1, label='Instantaneous KE')
        ax2.plot(t, time_avg_KE, color=self.plotter.colors_main[2], 
                linewidth=3, label='Time Average')
        ax2.axhline(0.5, color='black', linestyle='--', 
                   linewidth=2, label=r'Theory: $\langle KE \rangle = E/2$', alpha=0.7)
        
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('KE / Total Energy', fontsize=11)
        ax2.set_title('Time Average: Following One System', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.set_xlim(0, 20)  # Explicitly set x-limits
        ax2.set_ylim(-0.1, 1.1)  # Slightly expanded to see oscillations clearly
        
        # Example 3: Ensemble average
        ax3 = fig.add_subplot(gs[1, 1])
        
        n_systems = 1000
        np.random.seed(43)
        t_snapshot = 2.5  # Fixed time
        phases = np.random.uniform(0, 2*np.pi, n_systems)
        KE_ensemble = E_total * np.sin(omega * t_snapshot + phases)**2
        ensemble_avg = np.cumsum(KE_ensemble) / np.arange(1, n_systems + 1)
        
        ax3.fill_between(range(n_systems),
                        0.5 - 1/(2*np.sqrt(np.arange(1, n_systems + 1))),
                        0.5 + 1/(2*np.sqrt(np.arange(1, n_systems + 1))),
                        alpha=0.2, color='gray', 
                        label=r'$1/\sqrt{N}$ bounds')
        
        ax3.plot(ensemble_avg, color=self.plotter.colors_main[5], 
                linewidth=3, label='Ensemble Average')
        ax3.axhline(0.5, color='black', linestyle='--', 
                   linewidth=2, label=r'Theory: $E/2$', alpha=0.7)
        
        ax3.set_xlabel('Number of Systems', fontsize=11)
        ax3.set_ylabel('Average KE / Total Energy', fontsize=11)
        ax3.set_title('Ensemble Average: Many Systems at Once', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax3.grid(True, alpha=0.3, linestyle=':')
        ax3.set_xlim(0, n_systems)  # Explicitly set x-limits
        ax3.set_ylim(0.3, 0.7)  # Slightly wider to show convergence
        
        # Example 4: Visual comparison - Bottom right
        ax4 = fig.add_subplot(gs[1, 2])
        
        # Generate both averages
        sample_sizes = np.logspace(1, 3, 50).astype(int)
        time_results = []
        ensemble_results = []
        
        for n in sample_sizes:
            # Time average
            t_long = np.linspace(0, n/10, n)
            KE_time = E_total * np.sin(omega * t_long + phase)**2
            time_results.append(np.mean(KE_time))
            
            # Ensemble average  
            phases_ens = np.random.uniform(0, 2*np.pi, n)
            KE_ens = E_total * np.sin(omega * t_snapshot + phases_ens)**2
            ensemble_results.append(np.mean(KE_ens))
        
        ax4.semilogx(sample_sizes, time_results, 'o-',
                    color=self.plotter.colors_main[2], 
                    alpha=0.8, linewidth=2, markersize=4, label='Time Average')
        ax4.semilogx(sample_sizes, ensemble_results, 's-',
                    color=self.plotter.colors_main[5], 
                    alpha=0.8, linewidth=2, markersize=4, label='Ensemble Average')
        ax4.axhline(0.5, color='black', linestyle='--', 
                   linewidth=2, label='Theory', alpha=0.7)
        
        ax4.set_xlabel('Sample Size (N)', fontsize=11)
        ax4.set_ylabel('Average KE/E', fontsize=11)
        ax4.set_title('Both Methods → Same Result\n(Time avg over N periods vs N-system ensemble)', fontsize=11, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax4.grid(True, alpha=0.3, linestyle=':')
        ax4.set_ylim(0.4, 0.6)
        
        fig.suptitle(r'Ergodicity: The Foundation of Statistical Mechanics',
                    fontsize=16, fontweight='bold', y=0.96)
        
        return fig
    
    def fig6_error_scaling(self):
        """
        Figure 6: Universal 1/√N Error Scaling
        Demonstrates Monte Carlo convergence rate.
        """
        fig, gs = self.plotter.create_figure_with_gridspec(
            figsize=(15, 9), nrows=2, ncols=2, hspace=0.35, wspace=0.35
        )
        
        # Panel 1: Error scaling for different distributions
        ax1 = fig.add_subplot(gs[0, :])
        
        N_values = np.logspace(1, 5, 30).astype(int)
        
        # Make all distributions have same variance for fair comparison
        distributions = {
            'Gaussian': (lambda n: np.random.normal(0, 1, n), 
                        self.plotter.colors_categorical[0]),
            'Exponential': (lambda n: np.random.exponential(1, n) - 1, 
                           self.plotter.colors_categorical[1]),
            'Uniform': (lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n), 
                       self.plotter.colors_categorical[2])
        }
        
        first_error = None
        for name, (dist_func, color) in distributions.items():
            errors = []
            # First compute the true standard deviation of the distribution
            large_sample = dist_func(10000)
            true_std = np.std(large_sample)
            
            for N in N_values:
                # Theoretical standard error of the mean: σ/√N
                # We can estimate this empirically with multiple trials
                n_trials = 100  # Use consistent number of trials
                means = []
                for _ in range(n_trials):
                    sample = dist_func(N)
                    means.append(np.mean(sample))
                
                # This estimates the standard error of the mean
                error = np.std(means) if len(means) > 1 else true_std/np.sqrt(N)
                errors.append(error)
            
            errors = np.array(errors)
            if first_error is None and errors[0] > 1e-10:
                first_error = errors[0]
            mask = errors > 0
            ax1.loglog(N_values[mask], errors[mask], 'o-', label=name, 
                      alpha=0.8, markersize=6, linewidth=2, color=color)
        
        # Theoretical 1/√N
        theoretical = 1.0 / np.sqrt(N_values)
        if first_error is not None and theoretical[0] > 0:
            theoretical = theoretical * first_error / theoretical[0]
        else:
            theoretical = theoretical * 0.1
        ax1.loglog(N_values, theoretical, '--', linewidth=2.5, 
                  label=r'$1/\sqrt{N}$ scaling', alpha=0.7, 
                  color=self.plotter.colors_main[0])
        
        self.plotter.apply_style(ax1,
                               xlabel=r'Number of Samples ($N$)',
                               ylabel='Standard Error',
                               title=r'Universal Error Scaling: $\sigma \propto 1/\sqrt{N}$',
                               legend=True)
        
        # Panel 2: Practical implications
        ax2 = fig.add_subplot(gs[1, 0])
        
        accuracy_improvement = [1, 10, 100, 1000]
        samples_needed = [1, 100, 10000, 1000000]
        
        colors_bars = self.plotter.get_color_sequence(4, 'sequential')
        bars = ax2.bar(range(len(accuracy_improvement)), 
                      np.log10(samples_needed), 
                      color=colors_bars, edgecolor='black', linewidth=1.5)
        
        ax2.set_xticks(range(len(accuracy_improvement)))
        ax2.set_xticklabels([rf'${a}\times$' for a in accuracy_improvement])
        
        # Add value labels on bars
        for bar, val in zip(bars, samples_needed):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val:,}', ha='center', va='bottom', fontsize=10)
        
        self.plotter.apply_style(ax2,
                               xlabel='Desired Accuracy Improvement',
                               ylabel=r'$\log_{10}$(Samples Needed)',
                               title='Cost of Accuracy: Quadratic Scaling')
        
        # Panel 3: Monte Carlo π estimation
        ax3 = fig.add_subplot(gs[1, 1])
        
        np.random.seed(42)
        n_points = np.logspace(1, 4, 20).astype(int)
        estimates = []
        
        for n in n_points:
            x = np.random.uniform(-1, 1, n)
            y = np.random.uniform(-1, 1, n)
            inside = (x**2 + y**2) <= 1
            pi_estimate = 4 * np.sum(inside) / n
            estimates.append(pi_estimate)
        
        ax3.semilogx(n_points, estimates, 'o-', 
                    color=self.plotter.colors_main[3], 
                    linewidth=2, markersize=6, label='MC Estimate')
        ax3.axhline(np.pi, color=self.plotter.colors_main[8], 
                   linestyle='--', linewidth=2, label=r'True value ($\pi$)')
        ax3.fill_between(n_points, 
                        np.pi - 1/np.sqrt(n_points),
                        np.pi + 1/np.sqrt(n_points),
                        alpha=0.3, color=self.plotter.colors_main[0], 
                        label=r'$1/\sqrt{N}$ bounds')
        
        self.plotter.apply_style(ax3,
                               xlabel='Number of Samples',
                               ylabel='Estimated Value',
                               title=r'Monte Carlo Estimation of $\pi$',
                               legend=True)
        ax3.set_ylim(2.5, 3.7)
        
        fig.suptitle(r'The $1/\sqrt{N}$ Law: Why Monte Carlo Works',
                    fontsize=16, fontweight='bold', y=0.96)
        
        # Add key insight box
        self.plotter.add_text_box(ax3,
                                 r'Key: 10× accuracy = 100× samples!',
                                 location='lower right',
                                 style='warning')
        
        return fig
    
    def fig7_sampling_methods_comparison(self):
        """
        Figure 7: Sampling Methods Comparison
        Demonstrates inverse transform and rejection sampling.
        """
        fig, gs = self.plotter.create_figure_with_gridspec(
            figsize=(16, 12), nrows=3, ncols=2, hspace=0.35, wspace=0.3
        )
        
        # Panel 1: Inverse Transform - Concept
        ax1 = fig.add_subplot(gs[0, 0])
        
        x = np.linspace(-3, 3, 1000)
        cdf = stats.norm.cdf(x)
        
        ax1.plot(x, cdf, color=self.plotter.colors_main[2], 
                linewidth=3, label='CDF')
        
        # Show mapping arrows
        u_samples = [0.1, 0.3, 0.5, 0.7, 0.9]
        for u in u_samples:
            x_val = stats.norm.ppf(u)
            ax1.arrow(-3.5, u, 3.5 + x_val, 0, head_width=0.03, 
                     head_length=0.1, fc=self.plotter.colors_main[7], 
                     ec=self.plotter.colors_main[7], alpha=0.5)
            ax1.plot([x_val, x_val], [0, u], '--', 
                    color=self.plotter.colors_main[7], alpha=0.5)
            ax1.scatter([x_val], [0], color=self.plotter.colors_main[8], 
                       s=50, zorder=5)
            ax1.text(-3.8, u, rf'$u={u}$', fontsize=9, va='center')
        
        self.plotter.apply_style(ax1,
                               xlabel=r'$x$',
                               ylabel=r'$F(x) = u$',
                               title='Inverse Transform Method')
        ax1.set_xlim(-4, 3)
        
        # Panel 2: Inverse Transform - Result
        ax2 = fig.add_subplot(gs[0, 1])
        
        n_samples = 1000
        np.random.seed(42)
        u = np.random.uniform(0.001, 0.999, n_samples)
        x_samples = np.clip(stats.norm.ppf(u), -10, 10)
        
        ax2.hist(x_samples, bins=30, density=True, alpha=0.7, 
                color=self.plotter.colors_main[3], edgecolor='black', 
                label='Samples')
        x_range = np.linspace(-4, 4, 100)
        ax2.plot(x_range, stats.norm.pdf(x_range), 
                color=self.plotter.colors_main[8], 
                linewidth=2, label='Target PDF')
        
        self.plotter.apply_style(ax2,
                               xlabel=r'$x$',
                               ylabel='Probability Density',
                               title=rf'Result: {n_samples} samples',
                               legend=True)
        
        # Panel 3: Rejection Sampling - Concept
        ax3 = fig.add_subplot(gs[1, 0])
        
        def target_pdf(x):
            return 0.3 * stats.norm.pdf(x, -1.5, 0.5) + \
                   0.7 * stats.norm.pdf(x, 1, 0.7)
        
        x_range = np.linspace(-4, 4, 1000)
        y_target = target_pdf(x_range)
        
        ax3.fill_between(x_range, 0, y_target, alpha=0.3, 
                        color=self.plotter.colors_main[2], label='Target PDF')
        ax3.plot(x_range, y_target, color=self.plotter.colors_main[2], 
                linewidth=2)
        
        # Envelope
        M = 0.5
        ax3.axhline(M, color=self.plotter.colors_main[7], linestyle='--', 
                   linewidth=2, label=rf'Envelope $M = {M}$')
        ax3.fill_between(x_range, y_target, M, alpha=0.2, 
                        color=self.plotter.colors_main[7])
        
        # Show random points
        np.random.seed(42)
        n_points = 100
        x_random = np.random.uniform(-4, 4, n_points)
        y_random = np.random.uniform(0, M, n_points)
        
        accept = y_random <= target_pdf(x_random)
        ax3.scatter(x_random[accept], y_random[accept], 
                   c=self.plotter.color_accept, 
                   s=20, alpha=0.6, label='Accept')
        ax3.scatter(x_random[~accept], y_random[~accept], 
                   c=self.plotter.color_reject, 
                   s=20, alpha=0.6, label='Reject')
        
        self.plotter.apply_style(ax3,
                               xlabel=r'$x$',
                               ylabel=r'$y$',
                               title='Rejection Sampling',
                               legend=True)
        ax3.set_xlim(-4, 4)
        
        # Panel 4: Rejection Sampling - Result
        ax4 = fig.add_subplot(gs[1, 1])
        
        samples = []
        n_tries = 0
        target_samples = 1000
        max_tries = 100000
        
        while len(samples) < target_samples and n_tries < max_tries:
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(0, M)
            n_tries += 1
            if y <= target_pdf(x):
                samples.append(x)
        
        efficiency = len(samples) / n_tries if n_tries > 0 else 0
        
        ax4.hist(samples, bins=30, density=True, alpha=0.7, 
                color=self.plotter.color_accept, edgecolor='black', 
                label='Accepted samples')
        ax4.plot(x_range, y_target, color=self.plotter.colors_main[2], 
                linewidth=2, label='Target PDF')
        
        self.plotter.apply_style(ax4,
                               xlabel=r'$x$',
                               ylabel='Probability Density',
                               title=rf'Efficiency = {efficiency:.1%}',
                               legend=True)
        
        # Panel 5-6: Comparison table with better formatting
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create comparison data
        methods_data = [
            ['Method', 'Pros', 'Cons', 'Best For'],
            ['Inverse\nTransform', 
             '• Exact\n• One-to-one\n• No rejection',
             '• Need CDF\n• CDF inversion\n  can be hard',
             'Simple distributions'],
            ['Rejection\nSampling',
             '• Works for any PDF\n• Simple\n• No CDF needed',
             '• Can be inefficient\n• Wastes samples',
             'Complex distributions']
        ]
        
        table = ax5.table(cellText=methods_data, loc='center', 
                         cellLoc='left', colWidths=[0.15, 0.35, 0.35, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 3.5)  # Increased height for better readability
        
        # Style the header row
        for i in range(4):
            table[(0, i)].set_facecolor(self.plotter.colors_main[1])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows with alternating colors
        for i in range(1, 3):
            for j in range(4):
                if i == 1:
                    table[(i, j)].set_facecolor('#F8F9FA')
                else:
                    table[(i, j)].set_facecolor('#ECEFF4')
        
        fig.suptitle(r'Sampling Methods: From Uniform to Any Distribution',
                    fontsize=16, fontweight='bold', y=0.96)
        
        return fig
    
    def generate_all_figures(self, save=True):
        """
        Generate and save all figures for the module.
        
        Parameters:
        -----------
        save : bool
            Whether to save the figures
        
        Returns:
        --------
        figures : dict
            Dictionary of figure objects
        """
        print("\n" + "=" * 60)
        print("GENERATING MODULE 2a: STATISTICAL MECHANICS FIGURES")
        print("=" * 60)
        
        # Define all figures to generate
        figure_specs = [
            ('fig1_temperature_emergence', self.fig1_temperature_emergence),
            ('fig2_pressure_chaos', self.fig2_pressure_from_chaos),
            ('fig2b_wall_collision', self.fig2b_wall_collision_visual),
            ('fig3_central_limit', self.fig3_central_limit_theorem),
            ('fig4_maxwell_boltzmann', self.fig4_maxwell_boltzmann_3d),
            ('fig5_ergodicity', self.fig5_ergodicity_demo),
            ('fig6_error_scaling', self.fig6_error_scaling),
            ('fig7_sampling_methods', self.fig7_sampling_methods_comparison)
        ]
        
        figures = {}
        
        for name, fig_method in figure_specs:
            print(f"\nGenerating: {name}...")
            try:
                fig = fig_method()
                figures[name] = fig
                
                if save:
                    # Save using ASTR596 plotter's save method
                    self.plotter.save_figure(fig, name, formats=['png', 'svg'])
                    
                    # Generate caption for course materials
                    caption_info = {
                        'fig1_temperature_emergence': (
                            'Temperature Emergence from Ensemble',
                            'Demonstrates how temperature becomes a meaningful concept only for ensembles of particles, not individual ones.',
                            'The error in temperature measurement decreases as 1/√N.'
                        ),
                        'fig2_pressure_chaos': (
                            'From Molecular Chaos to Pressure',
                            'Shows how random molecular collisions create steady macroscopic pressure.',
                            'Individual chaos averages to P = nkT.'
                        ),
                        'fig2b_wall_collision': (
                            'Isotropic Gas Molecules Creating Pressure',
                            'Visualization of gas molecules with isotropic (random direction) velocities hitting a wall.',
                            'Colors represent kinetic energy. Inset shows velocity distribution.'
                        ),
                        'fig3_central_limit': (
                            'Central Limit Theorem',
                            'Any distribution becomes Gaussian when summed, explaining why normal distributions are ubiquitous in nature.',
                            'Colors indicate deviation from perfect Gaussian.'
                        ),
                        'fig4_maxwell_boltzmann': (
                            'Maxwell-Boltzmann Distribution',
                            'Velocity and energy distributions at different temperatures.',
                            'Higher temperature means broader distribution and higher speeds.'
                        ),
                        'fig5_ergodicity': (
                            'Ergodicity Principle',
                            'Time average equals ensemble average for ergodic systems.',
                            'This principle underlies all Monte Carlo methods.'
                        ),
                        'fig6_error_scaling': (
                            'Universal 1/√N Error Scaling',
                            'Monte Carlo error decreases as 1/√N regardless of distribution.',
                            '10× accuracy requires 100× more samples.'
                        ),
                        'fig7_sampling_methods': (
                            'Sampling Methods Comparison',
                            'Inverse transform vs rejection sampling for generating distributions.',
                            'Each method has different efficiency and requirements.'
                        )
                    }
                    
                    if name in caption_info:
                        title, desc, detail = caption_info[name]
                        # Extract figure number, handling both fig1 and fig2b formats
                        if 'b' in name:
                            fig_num = name[3:5]  # e.g., '2b' from 'fig2b_...'
                        else:
                            fig_num = int(name[3])  # e.g., 1 from 'fig1_...'
                        captions = self.plotter.generate_caption(
                            figure_number=fig_num,
                            title=title,
                            description=desc,
                            details=detail
                        )
                        print(f"  Caption: {captions['full'][:80]}...")
                    
            except Exception as e:
                print(f"  ✗ Error generating {name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print(f"COMPLETE: Generated {len(figures)} figures")
        print(f"Location: {self.plotter.save_dir}")
        print("=" * 60)
        
        return figures


# Example usage and testing
def main():
    """Main function to generate all module figures."""
    
    # Create visualizer instance
    viz = StatMechVisualizations(save_dir='./module2a_figures/')
    
    # Generate all figures
    figures = viz.generate_all_figures(save=True)
    
    # Don't display figures - user will view files
    print("\nFigures saved to disk. View them in ./module2a_figures/")
    # plt.show()  # Commented out to prevent popup
    
    return figures


if __name__ == "__main__":
    main()