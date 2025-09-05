"""
Module 2a Visualization Suite - Updated with LaTeX and Scientific Colors
Generates pedagogical figures for Statistical Foundations module
Author: Anna Rosen + Claude AI
Course: ASTR 596: Modeling the Universe
Date: January 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set up beautiful plotting style
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.facecolor'] = '#fafafa'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed

class StatMechVisualizations:
    """Generate all figures for Module 2a: Statistical Foundations"""
    
    def __init__(self, save_dir='./figures/'):
        """Initialize with output directory for figures"""
        self.save_dir = save_dir
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        
        self.k_B = 1.38e-16  # erg/K (Boltzmann constant)
        self.m_H = 1.67e-24  # g (hydrogen mass)
        
        # Define sophisticated color palettes inspired by scientific visualization
        # Main palette: Deep space to stellar (Nord-inspired + scientific)
        self.main_colors = [
            '#2E3440',  # Dark space gray
            '#5E81AC',  # Deep space blue  
            '#81A1C1',  # Stellar blue
            '#88C0D0',  # Nebula cyan
            '#8FBCBB',  # Quantum teal
            '#A3BE8C',  # Aurora green
            '#EBCB8B',  # Solar yellow
            '#D08770',  # Plasma orange
            '#BF616A',  # Stellar red
            '#B48EAD',  # Cosmic purple
        ]
        
        # Temperature palette: Cold to hot
        self.temp_palette = [
            '#5E81AC',  # Cold blue (100 K)
            '#A3BE8C',  # Medium green (300 K)
            '#D08770',  # Hot orange (1000 K)
        ]
        
        # Distribution colors
        self.dist_colors = [
            '#8FBCBB',  # Teal for uniform
            '#88C0D0',  # Cyan for exponential  
            '#B48EAD',  # Purple for bimodal
        ]
        
        # Acceptance/rejection colors
        self.accept_color = '#A3BE8C'  # Green
        self.reject_color = '#BF616A'  # Red
    
    def fig1_temperature_emergence(self):
        """
        Figure 1: Temperature Emergence from Ensemble
        Shows how temperature becomes meaningful as N increases.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        N_values = [1, 2, 5, 10, 50, 100, 500, 1000, 10000]
        T_true = 300  # K
        sigma = np.sqrt(self.k_B * T_true / self.m_H)
        
        for idx, N in enumerate(N_values):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            
            # Generate velocities
            np.random.seed(42 + idx)  # For reproducibility
            velocities = np.random.normal(0, sigma, N) / 1e5  # Convert to km/s
            
            if N == 1:
                # Single particle - show as a point with arrow
                ax.arrow(0, 0, velocities[0], 0, head_width=0.3, 
                        head_length=0.5, fc=self.main_colors[8], 
                        ec=self.main_colors[8], linewidth=2)
                ax.scatter([velocities[0]], [0], s=200, color=self.main_colors[8], zorder=5)
                ax.set_ylim(-1, 1)
                ax.set_xlim(-30, 30)
                ax.set_title(rf'$N = {N}$: Single particle' + '\n' + 
                           rf'$v = {velocities[0]:.1f}$ km/s' + '\n' +
                           r'(No temperature!)', 
                           fontsize=10, fontweight='bold')
                ax.set_xlabel(r'Velocity (km/s)')
                ax.set_yticks([])
                
            elif N == 2:
                # Two particles - show both
                for i, v in enumerate(velocities):
                    color_idx = (3 + i) % len(self.main_colors)  # Use modulo for safe indexing
                    ax.arrow(0, 0.5, v, 0, head_width=0.2, 
                            head_length=0.5, fc=self.main_colors[color_idx], 
                            ec=self.main_colors[color_idx], alpha=0.7)
                ax.set_ylim(0, 1)
                ax.set_xlim(-30, 30)
                ax.set_title(rf'$N = {N}$: Two particles' + '\n' +
                           r'(Temperature emerging)', 
                           fontsize=10)
                ax.set_xlabel(r'Velocity (km/s)')
                ax.set_yticks([])
                
            else:
                # Show histogram with increasing resolution
                bins = min(max(5, N//5), 30)
                counts, bins_edges, patches = ax.hist(velocities, bins=bins, 
                                               density=True, alpha=0.7, 
                                               edgecolor='black', linewidth=0.5)
                
                # Color bars by temperature - use gradient
                norm = plt.Normalize(vmin=counts.min() if len(counts) > 0 else 0, 
                                   vmax=counts.max() if len(counts) > 0 else 1)
                cmap = LinearSegmentedColormap.from_list('temp', self.temp_palette)
                for count, patch in zip(counts, patches):
                    color_intensity = norm(count)
                    patch.set_facecolor(cmap(color_intensity))
                
                # Overlay theoretical Maxwell-Boltzmann
                v_range = np.linspace(-30, 30, 200)
                theory = stats.norm.pdf(v_range, 0, sigma/1e5)
                ax.plot(v_range, theory, color=self.main_colors[8], lw=2, 
                       label='Maxwell-Boltzmann', alpha=0.8)
                
                # Measure temperature from variance
                T_measured = self.m_H * np.var(velocities * 1e5) / self.k_B
                T_error = T_measured / np.sqrt(2*N)
                
                ax.set_title(rf'$N = {N}$: $T = {T_measured:.0f} \pm {T_error:.0f}$ K', 
                           fontsize=10)
                ax.set_xlabel(r'Velocity (km/s)')
                ax.set_ylabel(r'Probability Density')
                
                if idx >= 6:
                    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle(r'Temperature Emerges from Ensemble Statistics' + '\n' + 
                    r'"Temperature doesn\'t exist for one particle"', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Add text annotation
        fig.text(0.5, 0.02, 
                r'As $N$ increases: (1) Distribution shape stabilizes, ' + 
                r'(2) Temperature measurement uncertainty decreases as $1/\sqrt{N}$, ' + 
                r'(3) Individual chaos $\rightarrow$ Collective order',
                ha='center', fontsize=11, style='italic', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#ECEFF4', alpha=0.7))
        
        return fig
    
    def fig2_pressure_from_chaos(self):
        """
        Figure 2: Pressure Emerges from Random Collisions
        Enhanced visualization with particle simulation
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        T = 300  # K
        sigma = np.sqrt(self.k_B * T / self.m_H)
        
        # Panel 1: Individual particle trajectories (chaos)
        ax1 = fig.add_subplot(gs[0, :])
        n_particles = 5
        time_steps = 100
        
        for i in range(n_particles):
            np.random.seed(i)
            velocities = np.random.normal(0, sigma, time_steps) / 1e5
            momentum = self.m_H * np.abs(velocities) * 1e5
            time = np.arange(time_steps)
            color_idx = (2 + i) % len(self.main_colors)  # Use modulo for safe indexing
            color = self.main_colors[color_idx]
            ax1.plot(time, momentum * 1e19, alpha=0.7, linewidth=1.5, 
                    label=f'Particle {i+1}', color=color)
        
        ax1.set_ylabel(r'Momentum Transfer' + '\n' + r'($\times 10^{-19}$ g$\cdot$cm/s)', fontsize=11)
        ax1.set_xlabel(r'Time (arbitrary units)')
        ax1.set_title(r'Individual Particle Collisions: Pure Chaos', fontweight='bold')
        ax1.legend(loc='upper right', ncol=5, frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Distribution of momentum transfers
        ax2 = fig.add_subplot(gs[1, 0])
        all_momentum = []
        for _ in range(10000):
            v = np.random.normal(0, sigma)
            all_momentum.append(2 * self.m_H * np.abs(v))
        
        all_momentum = np.array(all_momentum) * 1e19
        ax2.hist(all_momentum, bins=50, density=True, alpha=0.7, 
                color=self.main_colors[3], edgecolor='black', linewidth=0.5)
        mean_val = np.mean(all_momentum)
        ax2.axvline(mean_val, color=self.main_colors[8], linestyle='--', 
                   linewidth=2, label=rf'Mean = {mean_val:.2f}')
        ax2.set_xlabel(r'Momentum Transfer ($\times 10^{-19}$ g$\cdot$cm/s)')
        ax2.set_ylabel(r'Probability Density')
        ax2.set_title(r'Distribution of Individual Transfers')
        ax2.legend()
        
        # Panel 3: Convergence to steady pressure
        ax3 = fig.add_subplot(gs[1, 1])
        n_samples_list = [10, 100, 1000, 10000]
        
        for idx, n_samples in enumerate(n_samples_list):
            np.random.seed(100 + idx)  # Ensure reproducibility
            averages = []
            for i in range(1, min(max(1, n_samples + 1), 1000)):  # Ensure at least 1 iteration
                sample = np.random.normal(0, sigma, i)
                avg_pressure = np.mean(self.m_H * sample**2)
                averages.append(avg_pressure)
            
            x = np.arange(1, len(averages) + 1)
            color_idx = idx % len(self.temp_palette) if len(self.temp_palette) > 0 else 0
            ax3.semilogx(x, np.array(averages) * 1e10, alpha=0.7, 
                        linewidth=1.5, label=rf'$N = {n_samples}$')
        
        theoretical = self.k_B * T * 1e10
        ax3.axhline(theoretical, color=self.main_colors[0], linestyle='--', linewidth=2, 
                   label=rf'Theory: $k_BT$')
        ax3.set_xlabel(r'Number of Particles Averaged')
        ax3.set_ylabel(r'Pressure per Particle' + '\n' + r'($\times 10^{-10}$ erg/cm$^3$)')
        ax3.set_title(r'Convergence to Steady Pressure')
        ax3.legend(loc='right')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Visual representation of pressure on wall
        ax4 = fig.add_subplot(gs[2, :])
        
        # Simulate particles hitting wall
        n_particles = 50
        np.random.seed(42)
        
        # Wall on the left
        ax4.axvline(x=0, color=self.main_colors[0], linewidth=10, alpha=0.8)
        ax4.text(-0.5, 5, 'WALL', rotation=90, fontsize=12, fontweight='bold',
                va='center', ha='center', color='white')
        
        # Particles approaching wall
        for i in range(n_particles):
            x = np.random.uniform(0.5, 10)
            y = np.random.uniform(0, 10)
            vx = -np.random.uniform(0.5, 2)  # Moving toward wall
            vy = np.random.uniform(-0.5, 0.5)
            
            # Draw particle
            circle = Circle((x, y), 0.15, color=self.main_colors[2], alpha=0.6)
            ax4.add_patch(circle)
            
            # Draw velocity arrow
            ax4.arrow(x, y, vx, vy, head_width=0.1, head_length=0.05, 
                     fc=self.main_colors[7], ec=self.main_colors[7], alpha=0.5)
        
        ax4.set_xlim(-1, 11)
        ax4.set_ylim(-0.5, 10.5)
        ax4.set_aspect('equal')
        ax4.set_xlabel(r'Distance from Wall')
        ax4.set_ylabel(r'Position')
        ax4.set_title(r'Molecular Bombardment Creates Pressure: Random Impacts $\rightarrow$ Steady Force', 
                     fontweight='bold')
        ax4.grid(False)
        
        plt.suptitle(r'From Molecular Chaos to Macroscopic Pressure' + '\n' + 
                    r'$P = nk_BT$ emerges from pure statistics',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def fig3_central_limit_theorem(self):
        """
        Figure 3: Central Limit Theorem in Action
        Interactive visualization showing convergence to Gaussian
        """
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        n_samples = 10000
        
        # Different starting distributions
        distributions = [
            ('Uniform', lambda n: np.random.uniform(-1, 1, n), self.dist_colors[0]),
            ('Exponential', lambda n: np.random.exponential(1, n) - 1, self.dist_colors[1]),
            ('Bimodal', lambda n: np.concatenate([
                np.random.normal(-2, 0.3, n//2),
                np.random.normal(2, 0.3, n//2)
            ]), self.dist_colors[2])
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
                    ax.set_title(rf'{name} Distribution' + '\n' + rf'($N = {N}$)', fontweight='bold')
                    
                else:
                    # Sum of N samples
                    sums = []
                    for _ in range(n_samples // N):
                        sums.append(np.sum(dist_func(N)))
                    
                    sums = np.array(sums)
                    # Normalize
                    if np.std(sums) > 0:
                        sums_normalized = (sums - np.mean(sums)) / np.std(sums)
                    else:
                        sums_normalized = sums - np.mean(sums)
                    
                    # Plot histogram
                    counts, bins_edges, patches = ax.hist(sums_normalized, bins=40, 
                                                   density=True, alpha=0.7, 
                                                   edgecolor='black', linewidth=0.5)
                    
                    # Color by deviation from Gaussian
                    x_mid = (bins_edges[:-1] + bins_edges[1:]) / 2
                    gaussian_vals = stats.norm.pdf(x_mid)
                    
                    # Create custom colormap from distribution color to green
                    cmap = LinearSegmentedColormap.from_list('gauss_dev', 
                                                            [self.reject_color, '#EBCB8B', self.accept_color])
                    
                    for count, patch, gauss in zip(counts, patches, gaussian_vals):
                        if gauss > 1e-10:  # Use small epsilon to avoid division by zero
                            deviation = abs(count - gauss) / gauss
                        else:
                            deviation = 0
                        color_intensity = max(0, min(1, 1 - deviation))
                        patch.set_facecolor(cmap(color_intensity))
                    
                    # Overlay Gaussian
                    x = np.linspace(-4, 4, 100)
                    ax.plot(x, stats.norm.pdf(x), color=self.main_colors[8], lw=2.5, 
                           label='Standard Gaussian', alpha=0.9)
                    
                    # Calculate KS statistic
                    ks_stat, _ = stats.kstest(sums_normalized, 'norm')
                    
                    ax.set_title(rf'Sum of {N} {name}' + '\n' + rf'KS = {ks_stat:.4f}', 
                               fontweight='bold')
                    
                    if col_idx == 2:
                        ax.legend(loc='upper right', frameon=True, fancybox=True)
                
                ax.set_xlabel(r'Value')
                ax.set_ylabel(r'Probability Density')
                ax.grid(True, alpha=0.3)
                
                # Add arrow between rows
                if row_idx < 2 and col_idx == 1:
                    ax.annotate('', xy=(0.5, -0.35), xytext=(0.5, -0.15),
                              xycoords='axes fraction',
                              arrowprops=dict(arrowstyle='->', lw=2, color=self.main_colors[8]))
        
        plt.suptitle(r'Central Limit Theorem: All Distributions Converge to Gaussian' + '\n' + 
                    r'This is why we can predict macroscopic behavior despite microscopic chaos',
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Add explanatory text
        fig.text(0.5, 0.01, 
                r'As $N$ increases, ANY distribution becomes Gaussian when summed. ' +
                r'Colors show deviation from perfect Gaussian (green = close, red = far).',
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#ECEFF4', alpha=0.7))
        
        return fig
    
    def fig4_maxwell_boltzmann_3d(self):
        """
        Figure 4: Maxwell-Boltzmann Distribution
        Enhanced visualization with physical insights
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.3)
        
        T_values = [100, 300, 1000]
        
        # Panel 1: 1D velocity component
        ax1 = fig.add_subplot(gs[0, 0])
        v = np.linspace(-3000, 3000, 1000)  # km/s
        
        for T, color in zip(T_values, self.temp_palette):
            sigma_1d = np.sqrt(self.k_B * T / self.m_H) / 1e5  # km/s
            f_1d = stats.norm.pdf(v, 0, sigma_1d)
            ax1.plot(v, f_1d, label=rf'$T = {T}$ K', linewidth=2.5, color=color)
            ax1.fill_between(v, 0, f_1d, alpha=0.2, color=color)
        
        ax1.set_xlabel(r'Velocity $v_x$ (km/s)')
        ax1.set_ylabel(r'Probability Density')
        ax1.set_title(r'1D Velocity Component' + '\n' + r'(Single direction)', fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: 3D speed distribution
        ax2 = fig.add_subplot(gs[0, 1])
        v_speed = np.linspace(0, 5000, 1000)  # km/s
        
        for T, color in zip(T_values, self.temp_palette):
            # Maxwell-Boltzmann speed distribution
            v_cms = v_speed * 1e5  # Convert to cm/s
            factor = 4 * np.pi * (self.m_H / (2 * np.pi * self.k_B * T))**(3/2)
            # Clip exponent to prevent underflow
            exponent = np.clip(-self.m_H * v_cms**2 / (2 * self.k_B * T), -700, 700)
            f_3d = factor * v_cms**2 * np.exp(exponent)
            f_3d /= 1e5  # Normalize for km/s units
            
            ax2.plot(v_speed, f_3d * 1e3, label=rf'$T = {T}$ K', linewidth=2.5, color=color)
            ax2.fill_between(v_speed, 0, f_3d * 1e3, alpha=0.2, color=color)
            
            # Mark most probable speed
            v_mp = np.sqrt(2 * self.k_B * T / self.m_H) / 1e5
            ax2.axvline(v_mp, color=color, linestyle=':', alpha=0.7)
        
        ax2.set_xlabel(r'Speed $|\vec{v}|$ (km/s)')
        ax2.set_ylabel(r'Probability Density ($\times 10^{-3}$)')
        ax2.set_title(r'3D Speed Distribution' + '\n' + r'(Magnitude of velocity)', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Energy distribution
        ax3 = fig.add_subplot(gs[0, 2])
        E = np.linspace(0, 10, 1000)  # In units of kT
        
        for T, color in zip(T_values, self.temp_palette):
            # Maxwell-Boltzmann energy distribution
            # Handle E=0 case for sqrt
            sqrt_term = np.where(E > 0, np.sqrt(E/np.pi), 0)
            f_E = 2 * sqrt_term * np.exp(-E)
            ax3.plot(E, f_E, label=rf'$T = {T}$ K', linewidth=2.5, color=color)
            ax3.fill_between(E, 0, f_E, alpha=0.2, color=color)
        
        ax3.set_xlabel(r'Energy (units of $k_BT$)')
        ax3.set_ylabel(r'Probability Density')
        ax3.set_title(r'Kinetic Energy Distribution' + '\n' + r'($E = \frac{1}{2}mv^2$)', fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4-6: 2D velocity space visualization
        for idx, (T, color) in enumerate(zip(T_values, self.temp_palette)):
            ax = fig.add_subplot(gs[1, idx])
            
            # Create 2D grid
            vx = vy = np.linspace(-2000, 2000, 100)
            VX, VY = np.meshgrid(vx, vy)
            
            # 2D Maxwell-Boltzmann
            sigma = np.sqrt(self.k_B * T / self.m_H) / 1e5
            # Clip exponent to prevent underflow
            exponent = np.clip(-(VX**2 + VY**2) / (2 * sigma**2), -700, 700)
            Z = (1 / (2 * np.pi * sigma**2)) * np.exp(exponent)
            
            # Create custom colormap for each temperature
            cmap = LinearSegmentedColormap.from_list('temp_map', ['white', color])
            
            # Plot as contour
            levels = np.linspace(0, Z.max(), 10)
            cs = ax.contourf(VX, VY, Z, levels=levels, cmap=cmap, alpha=0.8)
            ax.contour(VX, VY, Z, levels=5, colors='black', linewidths=0.5, alpha=0.5)
            
            # Add circular markers for characteristic speeds
            v_rms = np.sqrt(3 * self.k_B * T / self.m_H) / 1e5
            circle = Circle((0, 0), v_rms, fill=False, edgecolor=self.main_colors[0], 
                          linewidth=2, linestyle='--', label=r'$v_{\rm rms}$')
            ax.add_patch(circle)
            
            ax.set_xlabel(r'$v_x$ (km/s)')
            ax.set_ylabel(r'$v_y$ (km/s)')
            ax.set_title(rf'2D Velocity Space' + '\n' + rf'$T = {T}$ K', fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(loc='upper right')
        
        plt.suptitle(r'Maxwell-Boltzmann: Temperature Controls Distribution Width' + '\n' + 
                    r'Higher $T$ $\rightarrow$ Broader distribution $\rightarrow$ More velocity diversity',
                    fontsize=16, fontweight='bold', y=1.00)
        
        return fig
    
    def fig5_ergodicity_demo(self):
        """
        Figure 5: Ergodicity - Time Average Equals Ensemble Average
        Clear demonstration with coin flip and oscillator examples
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # Example 1: Coin flips
        ax1 = fig.add_subplot(gs[0, :])
        
        # Time average (one coin, many flips)
        n_flips = 1000
        np.random.seed(42)
        flips = np.random.choice([0, 1], n_flips)
        time_avg = np.cumsum(flips) / np.arange(1, n_flips + 1)
        
        ax1.plot(time_avg, color=self.main_colors[2], linewidth=2, 
                label='Time Average (one coin)', alpha=0.8)
        ax1.axhline(0.5, color=self.main_colors[8], linestyle='--', linewidth=2, 
                   label='Theoretical = 0.5')
        ax1.fill_between(range(n_flips), 
                        0.5 - 1/np.sqrt(np.arange(1, n_flips + 1)),
                        0.5 + 1/np.sqrt(np.arange(1, n_flips + 1)),
                        alpha=0.3, color=self.main_colors[0], label=r'$1/\sqrt{N}$ bounds')
        
        ax1.set_xlabel(r'Number of Flips')
        ax1.set_ylabel(r'Running Average')
        ax1.set_title(r'Coin Flip: Time Average Converges to Expected Value', 
                     fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.3, 0.7)
        
        # Example 2: Harmonic oscillator energy
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Time evolution
        t = np.linspace(0, 20, 1000)
        E_total = 1.0
        omega = 2 * np.pi
        phase = np.random.uniform(0, 2*np.pi)
        
        KE = E_total * np.sin(omega * t + phase)**2
        time_avg_KE = np.cumsum(KE) / np.arange(1, len(KE) + 1)
        
        ax2.plot(t, KE, color=self.main_colors[3], alpha=0.3, linewidth=1, 
                label='Instantaneous KE')
        ax2.plot(t, time_avg_KE, color=self.main_colors[3], linewidth=2.5, 
                label='Time Average')
        ax2.axhline(0.5, color=self.main_colors[8], linestyle='--', linewidth=2,
                   label=r'Theory: $E/2$')
        
        ax2.set_xlabel(r'Time')
        ax2.set_ylabel(r'Kinetic Energy / Total Energy')
        ax2.set_title(r'Time Average: Following One System', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Example 3: Ensemble average
        ax3 = fig.add_subplot(gs[1, 1])
        
        n_systems = 1000
        phases = np.random.uniform(0, 2*np.pi, n_systems)
        KE_ensemble = E_total * np.sin(phases)**2
        
        ensemble_avg = np.cumsum(KE_ensemble) / np.arange(1, n_systems + 1)
        
        ax3.plot(ensemble_avg, color=self.main_colors[5], linewidth=2.5, 
                label='Ensemble Average')
        ax3.axhline(0.5, color=self.main_colors[8], linestyle='--', linewidth=2,
                   label=r'Theory: $E/2$')
        ax3.fill_between(range(n_systems),
                        0.5 - 1/np.sqrt(np.arange(1, n_systems + 1)),
                        0.5 + 1/np.sqrt(np.arange(1, n_systems + 1)),
                        alpha=0.3, color=self.main_colors[0], label=r'$1/\sqrt{N}$ bounds')
        
        ax3.set_xlabel(r'Number of Systems')
        ax3.set_ylabel(r'Average Kinetic Energy / Total Energy')
        ax3.set_title(r'Ensemble Average: Many Systems at One Instant', 
                     fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Example 4: Visual comparison
        ax4 = fig.add_subplot(gs[2, :])
        
        # Generate both averages
        n_points = 100
        time_results = []
        ensemble_results = []
        
        for n in range(1, n_points):
            # Time average
            t_sample = np.linspace(0, n, min(n*10, 1000))
            KE_t = E_total * np.sin(omega * t_sample + phase)**2
            time_results.append(np.mean(KE_t))
            
            # Ensemble average
            phases_e = np.random.uniform(0, 2*np.pi, min(n*10, 1000))
            KE_e = E_total * np.sin(phases_e)**2
            ensemble_results.append(np.mean(KE_e))
        
        ax4.plot(time_results, color=self.main_colors[3], linewidth=2, 
                label='Time Average', alpha=0.8)
        ax4.plot(ensemble_results, color=self.main_colors[5], linewidth=2, 
                label='Ensemble Average', alpha=0.8)
        ax4.axhline(0.5, color=self.main_colors[8], linestyle='--', linewidth=2, 
                label='Theory')
        
        ax4.set_xlabel(r'Sample Size')
        ax4.set_ylabel(r'Average Value')
        ax4.set_title(r'Ergodicity: Both Averaging Methods Converge to Same Result' + '\n' +
                     r'This is why MCMC works!', fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0.35, 0.65)
        
        plt.suptitle(r'Ergodicity: The Bridge Between Theory and Computation',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def fig6_error_scaling(self):
        """
        Figure 6: Universal 1/√N Error Scaling
        Clear demonstration of Monte Carlo convergence
        """
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Error scaling for different distributions
        ax1 = fig.add_subplot(gs[0, :])
        
        N_values = np.logspace(1, 5, 30).astype(int)
        
        distributions = {
            'Gaussian': (lambda n: np.random.normal(0, 1, n), self.main_colors[2]),
            'Exponential': (lambda n: np.random.exponential(1, n), self.main_colors[5]),
            'Uniform': (lambda n: np.random.uniform(-1, 1, n), self.main_colors[7])
        }
        
        first_error = None  # Store first error for theoretical curve normalization
        for name, (dist_func, color) in distributions.items():
            errors = []
            for N in N_values:
                # Run multiple trials
                means = []
                for _ in range(min(50, 10000//N)):  # Adjust trials for performance
                    sample = dist_func(N)
                    means.append(np.mean(sample))
                
                error = np.std(means) if len(means) > 1 else 1e-10  # Avoid zero for log plot
                errors.append(error)
            
            errors = np.array(errors)
            if first_error is None and errors[0] > 1e-10:  # Store first valid error
                first_error = errors[0]
            mask = errors > 0  # Only plot non-zero errors
            ax1.loglog(N_values[mask], errors[mask], 'o-', label=name, alpha=0.8, 
                      markersize=6, linewidth=2, color=color)
        
        # Theoretical 1/√N
        theoretical = 1.0 / np.sqrt(N_values)
        # Normalize theoretical curve safely using first_error
        if first_error is not None and theoretical[0] > 0:
            theoretical = theoretical * first_error / theoretical[0]
        else:
            theoretical = theoretical * 0.1  # Use default scaling
        ax1.loglog(N_values, theoretical, '--', linewidth=2.5, 
                  label=r'$1/\sqrt{N}$ scaling', alpha=0.7, color=self.main_colors[0])
        
        ax1.set_xlabel(r'Number of Samples ($N$)', fontsize=12)
        ax1.set_ylabel(r'Standard Error', fontsize=12)
        ax1.set_title(r'Universal Error Scaling: All Distributions Follow $1/\sqrt{N}$', 
                     fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Panel 2: Practical implications
        ax2 = fig.add_subplot(gs[1, 0])
        
        accuracy_improvement = [1, 10, 100, 1000]
        samples_needed = [1, 100, 10000, 1000000]
        
        colors_bars = [self.main_colors[i*2 % len(self.main_colors)] for i in range(4)]
        bars = ax2.bar(range(len(accuracy_improvement)), 
                      np.log10(samples_needed), 
                      color=colors_bars,
                      edgecolor='black', linewidth=1.5)
        
        ax2.set_xticks(range(len(accuracy_improvement)))
        ax2.set_xticklabels([rf'${a}\times$' for a in accuracy_improvement])
        ax2.set_ylabel(r'$\log_{10}$(Samples Needed)', fontsize=12)
        ax2.set_xlabel(r'Desired Accuracy Improvement', fontsize=12)
        ax2.set_title(r'Cost of Accuracy: Quadratic Scaling', fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, samples_needed):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val:,}', ha='center', va='bottom', fontsize=10)
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Convergence visualization  
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Simulate Monte Carlo estimation of π
        np.random.seed(42)
        n_points = np.logspace(1, 4, 20).astype(int)
        estimates = []
        
        for n in n_points:
            # Estimate π using Monte Carlo
            x = np.random.uniform(-1, 1, n)
            y = np.random.uniform(-1, 1, n)
            inside = (x**2 + y**2) <= 1
            pi_estimate = 4 * np.sum(inside) / n
            estimates.append(pi_estimate)
        
        ax3.semilogx(n_points, estimates, 'o-', color=self.main_colors[3], 
                    linewidth=2, markersize=6, label=r'MC Estimate')
        ax3.axhline(np.pi, color=self.main_colors[8], linestyle='--', linewidth=2, 
                   label=r'True value ($\pi$)')
        ax3.fill_between(n_points, 
                        np.pi - 1/np.sqrt(n_points),
                        np.pi + 1/np.sqrt(n_points),
                        alpha=0.3, color=self.main_colors[0], label=r'$1/\sqrt{N}$ bounds')
        
        ax3.set_xlabel(r'Number of Samples', fontsize=12)
        ax3.set_ylabel(r'Estimated Value', fontsize=12)
        ax3.set_title(r'Monte Carlo Estimation of $\pi$', fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(2.5, 3.7)
        
        plt.suptitle(r'The Universal $1/\sqrt{N}$ Scaling: Foundation of Monte Carlo Methods',
                    fontsize=16, fontweight='bold', y=1.02)
        
        return fig
    
    def fig7_sampling_methods_comparison(self):
        """
        Figure 7: Sampling Methods Comparison
        Clear visualization of inverse transform and rejection sampling
        """
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # Panel 1: Inverse Transform - Concept
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Show CDF and mapping
        x = np.linspace(-3, 3, 1000)
        cdf = stats.norm.cdf(x)
        
        ax1.plot(x, cdf, color=self.main_colors[2], linewidth=3, label='CDF')
        
        # Show mapping arrows
        u_samples = [0.1, 0.3, 0.5, 0.7, 0.9]
        for u in u_samples:
            x_val = stats.norm.ppf(u)
            ax1.arrow(-3.5, u, 3.5 + x_val, 0, head_width=0.03, 
                     head_length=0.1, fc=self.main_colors[7], ec=self.main_colors[7], alpha=0.5)
            ax1.plot([x_val, x_val], [0, u], '--', color=self.main_colors[7], alpha=0.5)
            ax1.scatter([x_val], [0], color=self.main_colors[8], s=50, zorder=5)
            ax1.text(-3.8, u, rf'$u={u}$', fontsize=9, va='center')
        
        ax1.set_xlabel(r'$x$', fontsize=11)
        ax1.set_ylabel(r'$F(x) = u$', fontsize=11)
        ax1.set_title(r'Inverse Transform Method' + '\n' + r'Map uniform $[0,1]$ to any distribution', 
                     fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-4, 3)
        
        # Panel 2: Inverse Transform - Example
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Generate samples using inverse transform
        n_samples = 1000
        np.random.seed(42)
        # Avoid extreme values that could cause numerical issues
        u = np.random.uniform(0.001, 0.999, n_samples)
        x_samples = np.clip(stats.norm.ppf(u), -10, 10)  # Clip to reasonable range
        
        ax2.hist(x_samples, bins=30, density=True, alpha=0.7, 
                color=self.main_colors[3], edgecolor='black', label='Samples')
        x_range = np.linspace(-4, 4, 100)
        ax2.plot(x_range, stats.norm.pdf(x_range), color=self.main_colors[8], 
                linewidth=2, label='Target PDF')
        
        ax2.set_xlabel(r'$x$', fontsize=11)
        ax2.set_ylabel(r'Probability Density', fontsize=11)
        ax2.set_title(rf'Result: {n_samples} samples' + '\n' + r'Perfect match!', 
                     fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Rejection Sampling - Concept
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Target: mixture of Gaussians
        def target_pdf(x):
            return 0.3 * stats.norm.pdf(x, -1.5, 0.5) + \
                   0.7 * stats.norm.pdf(x, 1, 0.7)
        
        x_range = np.linspace(-4, 4, 1000)
        y_target = target_pdf(x_range)
        
        ax3.fill_between(x_range, 0, y_target, alpha=0.3, color=self.main_colors[2], 
                        label='Target PDF')
        ax3.plot(x_range, y_target, color=self.main_colors[2], linewidth=2)
        
        # Envelope
        M = 0.5
        ax3.axhline(M, color=self.main_colors[7], linestyle='--', linewidth=2, 
                   label=rf'Envelope $M = {M}$')
        ax3.fill_between(x_range, y_target, M, alpha=0.2, color=self.main_colors[7])
        
        # Show some random points
        np.random.seed(42)
        n_points = 100
        x_random = np.random.uniform(-4, 4, n_points)
        y_random = np.random.uniform(0, M, n_points)
        
        accept = y_random <= target_pdf(x_random)
        ax3.scatter(x_random[accept], y_random[accept], c=self.accept_color, 
                   s=20, alpha=0.6, label='Accept')
        ax3.scatter(x_random[~accept], y_random[~accept], c=self.reject_color, 
                   s=20, alpha=0.6, label='Reject')
        
        ax3.set_xlabel(r'$x$', fontsize=11)
        ax3.set_ylabel(r'$y$', fontsize=11)
        ax3.set_title(r'Rejection Sampling' + '\n' + r'"Throwing darts at the distribution"', 
                     fontweight='bold')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-4, 4)
        
        # Panel 4: Rejection Sampling - Result
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Generate samples using rejection
        samples = []
        n_tries = 0
        target_samples = 1000
        max_tries = 100000  # Add maximum iterations to prevent infinite loop
        while len(samples) < target_samples and n_tries < max_tries:
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(0, M)
            n_tries += 1
            if y <= target_pdf(x):
                samples.append(x)
        
        # Handle case where we didn't get enough samples
        if len(samples) < target_samples:
            print(f"Warning: Only generated {len(samples)} samples after {max_tries} attempts")
        
        efficiency = len(samples) / n_tries if n_tries > 0 else 0
        
        ax4.hist(samples, bins=30, density=True, alpha=0.7, 
                color=self.accept_color, edgecolor='black', label='Accepted samples')
        ax4.plot(x_range, y_target, color=self.main_colors[2], linewidth=2, label='Target PDF')
        
        ax4.set_xlabel(r'$x$', fontsize=11)
        ax4.set_ylabel(r'Probability Density', fontsize=11)
        ax4.set_title(rf'Result: Efficiency = {efficiency:.1%}' + '\n' + 
                     rf'({n_tries} tries for {len(samples)} samples)', 
                     fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panel 5-6: Comparison table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        table_data = [
            ['Method', 'Pros', 'Cons', 'Best For'],
            ['Inverse Transform', 
             '• Exact\n• One-to-one mapping\n• No rejection',
             '• Need analytical CDF\n• CDF inversion can be hard',
             'Simple distributions\n(exponential, uniform)'],
            ['Rejection Sampling',
             '• Works for any PDF\n• Simple to implement\n• No CDF needed',
             '• Can be inefficient\n• Wastes samples\n• Need good envelope',
             'Complex distributions\n(multimodal, arbitrary)']
        ]
        
        table = ax5.table(cellText=table_data, loc='center', 
                         cellLoc='left', colWidths=[0.15, 0.35, 0.35, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.5)
        
        # Style the header row
        for i in range(4):
            table[(0, i)].set_facecolor(self.main_colors[1])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style the data rows
        for i in range(1, 3):
            for j in range(4):
                if i == 1:
                    table[(i, j)].set_facecolor('#E5E9F0')
                else:
                    table[(i, j)].set_facecolor('#ECEFF4')
        
        plt.suptitle(r'Sampling Methods: From Uniform Random to Any Distribution',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def generate_all_figures(self, save=True):
        """Generate and optionally save all figures"""
        print("Generating Statistical Mechanics Visualizations...")
        print("=" * 50)
        
        # Define figure names and their generation methods
        figure_specs = [
            ('fig1_temperature_emergence', self.fig1_temperature_emergence),
            ('fig2_pressure_chaos', self.fig2_pressure_from_chaos),
            ('fig3_central_limit', self.fig3_central_limit_theorem),
            ('fig4_maxwell_boltzmann', self.fig4_maxwell_boltzmann_3d),
            ('fig5_ergodicity', self.fig5_ergodicity_demo),
            ('fig6_error_scaling', self.fig6_error_scaling),
            ('fig7_sampling_methods', self.fig7_sampling_methods_comparison)
        ]
        
        figures = {}
        
        for name, fig_method in figure_specs:
            print(f"Generating: {name}...")
            try:
                fig = fig_method()
                figures[name] = fig
                
                if save and self.save_dir:
                    # Save as both SVG and PNG
                    svg_filename = f'{self.save_dir}{name}.svg'
                    png_filename = f'{self.save_dir}{name}.png'
                    
                    # Save SVG
                    fig.savefig(svg_filename, format='svg', bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
                    print(f"  ✓ Saved SVG: {svg_filename}")
                    
                    # Save PNG (150 dpi for web, 300 for print)
                    fig.savefig(png_filename, format='png', dpi=150, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
                    print(f"  ✓ Saved PNG: {png_filename}")
                    
            except Exception as e:
                print(f"  ✗ Error generating {name}: {str(e)}")
        
        print("=" * 50)
        print(f"Successfully generated {len(figures)} figures!")
        
        return figures

# Usage
if __name__ == "__main__":
    # Create visualizer instance with figures directory
    viz = StatMechVisualizations(save_dir='./figures/')
    
    # Generate all figures (save=True by default now)
    all_figs = viz.generate_all_figures(save=True)
    
    # Display all figures
    plt.show()