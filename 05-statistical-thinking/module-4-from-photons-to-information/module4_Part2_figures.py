"""
Module 4 Part II: Radiative Transfer Figures
=============================================
This module generates publication-quality figures for the Mathematical Foundations
of Radiative Transfer section. Each figure is designed to build physical intuition
while maintaining mathematical rigor.

Author: Statistical Thinking ASTR 596
Requirements: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow, Wedge, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set up aesthetic parameters for all figures
plt.style.use('default')
COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',     # Purple-pink
    'tertiary': '#F18F01',      # Orange
    'quaternary': '#C73E1D',    # Red
    'light': '#F6F6F6',         # Light gray background
    'dark': '#2D3436',          # Dark gray text
    'green': '#00B894',         # Teal-green
    'yellow': '#FDCB6E'         # Warm yellow
}

def setup_axes(ax, title='', xlabel='', ylabel='', grid=False):
    """Configure axes with consistent styling."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color=COLORS['dark'])
    ax.set_xlabel(xlabel, fontsize=12, color=COLORS['dark'])
    ax.set_ylabel(ylabel, fontsize=12, color=COLORS['dark'])
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=COLORS['dark'], which='both')


def create_figure_2_1_1():
    """
    Create Figure 2.1.1: Solid Angle and Intensity Geometry
    
    This figure illustrates the fundamental geometric concepts underlying specific intensity.
    It shows how radiation is measured within a pencil beam defined by a solid angle, 
    and how spherical coordinates are used for angular integration.
    
    Caption: Specific intensity measures radiation within a pencil beam (center) coming from 
    a particular direction defined by solid angle dΩ (left). To calculate total flux, we 
    integrate over all directions using spherical coordinates (right). The cos θ factor 
    accounts for the projected area.
    
    Returns:
        fig: matplotlib figure object
    """
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor('white')
    
    # Create three subplots
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # Left panel: Solid angle concept
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.set_title('Solid Angle Concept', fontsize=12, fontweight='bold', color=COLORS['dark'])
    
    # Create cone for solid angle
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, 1, 50)
    theta, z = np.meshgrid(theta, z)
    r = z * 0.3  # Cone radius increases with z
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax1.plot_surface(x, y, z, alpha=0.3, color=COLORS['primary'], edgecolor='none')
    
    # Add observer point
    ax1.scatter([0], [0], [0], s=100, color=COLORS['quaternary'], edgecolors='black', linewidth=2)
    ax1.text(0, 0, -0.1, 'Observer', fontsize=10, ha='center')
    
    # Add patch on sky
    theta_patch = np.linspace(0, 2*np.pi, 30)
    x_patch = 0.3 * np.cos(theta_patch)
    y_patch = 0.3 * np.sin(theta_patch)
    z_patch = np.ones_like(x_patch)
    ax1.plot(x_patch, y_patch, z_patch, 'k-', linewidth=2)
    ax1.plot_surface(x_patch.reshape(1,-1), y_patch.reshape(1,-1), 
                     z_patch.reshape(1,-1), alpha=0.5, color=COLORS['tertiary'])
    
    # Add labels
    ax1.text(0, 0.4, 1, r'$d\Omega = \frac{dA}{r^2}$', fontsize=11, ha='center')
    ax1.text(0, 0, 0.5, r'$r$', fontsize=10, ha='center')
    
    ax1.set_xlim([-0.5, 0.5])
    ax1.set_ylim([-0.5, 0.5])
    ax1.set_zlim([0, 1.2])
    ax1.set_box_aspect([1,1,2])
    ax1.view_init(elev=20, azim=45)
    ax1.set_axis_off()
    
    # Center panel: Pencil beam geometry
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title('Pencil Beam of Radiation', fontsize=12, fontweight='bold', color=COLORS['dark'])
    ax2.set_xlim([-1, 3])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Draw area element
    rect = Rectangle((-0.3, -0.5), 0.6, 1, facecolor=COLORS['light'], 
                    edgecolor=COLORS['dark'], linewidth=2)
    ax2.add_patch(rect)
    ax2.text(0, -0.8, r'$dA$', fontsize=12, ha='center', fontweight='bold')
    
    # Draw pencil beam
    beam_x = [0, 2.5]
    beam_y1 = [0.3, 0.8]
    beam_y2 = [-0.3, -0.8]
    
    ax2.fill_between(beam_x, beam_y1, beam_y2, alpha=0.3, color=COLORS['secondary'])
    ax2.plot(beam_x, beam_y1, 'k--', linewidth=1, alpha=0.7)
    ax2.plot(beam_x, beam_y2, 'k--', linewidth=1, alpha=0.7)
    
    # Add angle indicator
    arc = mpatches.Arc((0, 0), 1.5, 1.5, angle=0, theta1=-20, theta2=20, 
                      linewidth=2, color=COLORS['quaternary'])
    ax2.add_patch(arc)
    ax2.text(0.9, 0, r'$\theta$', fontsize=12, color=COLORS['quaternary'], fontweight='bold')
    
    # Add normal vector
    ax2.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc=COLORS['green'], ec=COLORS['green'])
    ax2.text(-0.2, 1.1, r'$\hat{n}$', fontsize=12, color=COLORS['green'], fontweight='bold')
    
    # Add intensity arrow
    ax2.arrow(2.2, 0, 0.3, 0, head_width=0.15, head_length=0.08, 
             fc=COLORS['secondary'], ec=COLORS['secondary'], linewidth=2)
    ax2.text(2.7, 0, r'$I_\nu$', fontsize=14, ha='center', va='center', fontweight='bold')
    
    # Add labels
    ax2.text(1.2, 1.2, r'$d\Omega$', fontsize=12, ha='center', color=COLORS['secondary'])
    
    # Right panel: Spherical coordinates
    ax3 = fig.add_subplot(gs[2], projection='3d')
    ax3.set_title('Spherical Coordinate System', fontsize=12, fontweight='bold', color=COLORS['dark'])
    
    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax3.plot_wireframe(x, y, z, alpha=0.1, color='gray', linewidth=0.5)
    
    # Highlight integration element
    theta_elem = np.pi/3
    phi_elem = np.pi/4
    dtheta = 0.2
    dphi = 0.3
    
    # Create surface element
    theta_range = np.linspace(theta_elem, theta_elem + dtheta, 10)
    phi_range = np.linspace(phi_elem, phi_elem + dphi, 10)
    theta_mesh, phi_mesh = np.meshgrid(theta_range, phi_range)
    
    x_elem = np.sin(theta_mesh) * np.cos(phi_mesh)
    y_elem = np.sin(theta_mesh) * np.sin(phi_mesh)
    z_elem = np.cos(theta_mesh)
    
    ax3.plot_surface(x_elem, y_elem, z_elem, alpha=0.7, color=COLORS['tertiary'])
    
    # Add coordinate labels
    ax3.text(1.2, 0, 0, r'$\phi$', fontsize=12, color=COLORS['primary'], fontweight='bold')
    ax3.text(0, 0, 1.2, r'$\theta$', fontsize=12, color=COLORS['quaternary'], fontweight='bold')
    
    # Add differential element label
    ax3.text(0.7, 0.7, 0.5, r'$d\Omega = \sin\theta d\theta d\phi$', 
            fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor=COLORS['light'], alpha=0.8))
    
    # Draw axes
    ax3.plot([0, 1.5], [0, 0], [0, 0], 'k-', linewidth=2, alpha=0.5)
    ax3.plot([0, 0], [0, 1.5], [0, 0], 'k-', linewidth=2, alpha=0.5)
    ax3.plot([0, 0], [0, 0], [0, 1.5], 'k-', linewidth=2, alpha=0.5)
    
    ax3.set_xlim([-1, 1.5])
    ax3.set_ylim([-1, 1.5])
    ax3.set_zlim([-1, 1.5])
    ax3.view_init(elev=20, azim=45)
    ax3.set_axis_off()
    
    # Add main title
    fig.suptitle('Figure 2.1.1: Solid Angle and Intensity Geometry', 
                fontsize=16, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    return fig


def create_figure_2_1_2():
    """
    Create Figure 2.1.2: Intensity Conservation vs Flux Dilution
    
    This figure demonstrates the crucial distinction between specific intensity (which remains
    constant along rays in vacuum) and flux (which decreases as 1/r²). It shows three observers
    at different distances and graphs both quantities to illustrate the relationship.
    
    Caption: While specific intensity remains constant along rays in vacuum (top line), the flux 
    we measure decreases as 1/r² (bottom curve) because the solid angle subtended by the source 
    shrinks with distance. This is why distant stars appear fainter but not dimmer per unit 
    solid angle - a crucial distinction for understanding surface brightness.
    
    Returns:
        fig: matplotlib figure object
    """
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3, 
                          height_ratios=[1.2, 1], width_ratios=[2, 1])
    
    # Top panel: Three observers at different distances
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Observers at Different Distances from a Star', 
                 fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax1.set_xlim([-0.5, 10])
    ax1.set_ylim([-2, 2])
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Draw star
    star = Circle((0, 0), 0.3, color=COLORS['yellow'], ec='orange', linewidth=2)
    ax1.add_patch(star)
    ax1.text(0, -0.6, 'Star', fontsize=12, ha='center', fontweight='bold')
    
    # Draw light rays
    angles = np.linspace(-30, 30, 7) * np.pi/180
    for angle in angles:
        x_ray = [0.3*np.cos(angle), 9.5]
        y_ray = [0.3*np.sin(angle), 9.5*np.tan(angle)]
        ax1.plot(x_ray, y_ray, 'y-', alpha=0.3, linewidth=1)
    
    # Add observers and solid angles
    distances = [2, 4, 8]
    observer_colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    
    for i, (d, color) in enumerate(zip(distances, observer_colors)):
        # Observer
        ax1.scatter([d], [0], s=150, color=color, edgecolors='black', 
                   linewidth=2, zorder=5)
        ax1.text(d, -0.4, f'r = {d}', fontsize=11, ha='center')
        
        # Solid angle cone (simplified 2D representation)
        angle_width = np.arctan(0.3/d)
        y_top = d * np.tan(angle_width)
        y_bottom = -y_top
        
        # Draw cone lines
        ax1.plot([d, 0.3], [0, 0.3], '--', color=color, alpha=0.7, linewidth=1.5)
        ax1.plot([d, 0.3], [0, -0.3], '--', color=color, alpha=0.7, linewidth=1.5)
        
        # Label solid angle
        ax1.text(d, 0.8, f'Ω ∝ 1/r²', fontsize=10, ha='center', 
                color=color, bbox=dict(boxstyle="round,pad=0.2", 
                facecolor='white', alpha=0.8))
    
    # Bottom left: Intensity and Flux vs Distance
    ax2 = fig.add_subplot(gs[1, 0])
    setup_axes(ax2, title='Intensity and Flux vs Distance', 
              xlabel='Distance (r)', ylabel='Relative Value', grid=True)
    
    r = np.linspace(0.5, 10, 100)
    intensity = np.ones_like(r)  # Constant
    flux = 1/r**2  # Inverse square law
    
    # Normalize flux for visualization
    flux = flux / flux[0]
    
    ax2.plot(r, intensity, '-', color=COLORS['primary'], linewidth=3, label='Intensity (I)')
    ax2.plot(r, flux, '-', color=COLORS['quaternary'], linewidth=3, label='Flux (F ∝ 1/r²)')
    
    # Add observer points
    for d, color in zip(distances, observer_colors):
        ax2.scatter([d], [1], s=100, color=color, edgecolors='black', 
                   linewidth=2, zorder=5)
        ax2.scatter([d], [1/d**2 * 4], s=100, color=color, edgecolors='black', 
                   linewidth=2, zorder=5)
    
    ax2.fill_between(r, 0, flux, alpha=0.2, color=COLORS['quaternary'])
    ax2.set_xlim([0, 10])
    ax2.set_ylim([0, 1.2])
    ax2.legend(loc='upper right', fontsize=11)
    
    # Add annotations
    ax2.annotate('Intensity stays constant\n(surface brightness preserved)', 
                xy=(7, 1), xytext=(5, 0.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5),
                fontsize=10, color=COLORS['primary'])
    
    ax2.annotate('Flux decreases as 1/r²\n(total collected energy)', 
                xy=(4, 0.0625), xytext=(6, 0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['quaternary'], lw=1.5),
                fontsize=10, color=COLORS['quaternary'])
    
    # Bottom right: The relationship I × Ω = F
    ax3 = fig.add_subplot(gs[1, 1])
    setup_axes(ax3, title='The Key Relationship', xlabel='', ylabel='')
    ax3.axis('off')
    
    # Create visual equation
    ax3.text(0.5, 0.7, r'$F = I \times \Omega$', fontsize=20, ha='center', 
            fontweight='bold', color=COLORS['dark'])
    
    ax3.text(0.5, 0.5, r'Flux = Intensity × Solid Angle', fontsize=14, 
            ha='center', style='italic')
    
    # Add explanatory boxes
    ax3.text(0.5, 0.3, 'Intensity (I): Constant along rays', fontsize=11, 
            ha='center', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor=COLORS['primary'], alpha=0.2))
    
    ax3.text(0.5, 0.15, 'Solid Angle (Ω): Decreases as 1/r²', fontsize=11, 
            ha='center', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor=COLORS['tertiary'], alpha=0.2))
    
    ax3.text(0.5, 0, 'Flux (F): Decreases as 1/r²', fontsize=11, 
            ha='center', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor=COLORS['quaternary'], alpha=0.2))
    
    # Add main title
    fig.suptitle('Figure 2.1.2: Intensity Conservation vs Flux Dilution', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig


def create_figure_2_2_1():
    """
    Create Figure 2.2.1: The Radiative Transfer Equation Geometry
    
    This figure illustrates the physical processes described by the radiative transfer equation.
    It shows how intensity changes along a ray due to emission and absorption, introduces the
    concept of optical depth, and demonstrates the three opacity regimes.
    
    Caption: The radiative transfer equation describes the balance between emission (adding photons) 
    and absorption (removing photons) as radiation travels through a medium. The natural variable τ 
    (optical depth) counts the number of mean free paths, determining whether we see through the 
    medium (optically thin) or only its surface (optically thick).
    
    Returns:
        fig: matplotlib figure object
    """
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3,
                          height_ratios=[1.5, 1, 1])
    
    # Main panel: RTE cylinder diagram
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Radiative Transfer Along a Ray', fontsize=14, 
                 fontweight='bold', color=COLORS['dark'])
    ax1.set_xlim([-1, 10])
    ax1.set_ylim([-2, 2])
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Draw cylindrical volume element
    cylinder_x = [3, 6]
    cylinder_y = [-0.8, 0.8]
    
    # Draw cylinder body
    rect = Rectangle((3, -0.8), 3, 1.6, facecolor=COLORS['light'], 
                    edgecolor=COLORS['dark'], linewidth=2, alpha=0.5)
    ax1.add_patch(rect)
    
    # Draw end caps
    ellipse1 = mpatches.Ellipse((3, 0), 0.3, 1.6, facecolor=COLORS['light'], 
                                edgecolor=COLORS['dark'], linewidth=2)
    ellipse2 = mpatches.Ellipse((6, 0), 0.3, 1.6, facecolor=COLORS['light'], 
                                edgecolor=COLORS['dark'], linewidth=2)
    ax1.add_patch(ellipse1)
    ax1.add_patch(ellipse2)
    
    # Add incident intensity
    arrow_in = FancyArrow(0.5, 0, 2.3, 0, width=0.3, head_width=0.5, 
                         head_length=0.2, fc=COLORS['primary'], 
                         ec=COLORS['primary'], linewidth=2)
    ax1.add_patch(arrow_in)
    ax1.text(1.5, 0.6, r'$I_\nu$', fontsize=14, fontweight='bold', color=COLORS['primary'])
    
    # Add emergent intensity
    arrow_out = FancyArrow(6.2, 0, 2.3, 0, width=0.3, head_width=0.5, 
                          head_length=0.2, fc=COLORS['secondary'], 
                          ec=COLORS['secondary'], linewidth=2)
    ax1.add_patch(arrow_out)
    ax1.text(7.5, 0.6, r'$I_\nu + dI_\nu$', fontsize=14, fontweight='bold', 
            color=COLORS['secondary'])
    
    # Add emission arrows (pointing inward)
    emission_angles = [45, 135, 225, 315]
    for angle in emission_angles:
        angle_rad = angle * np.pi/180
        x_start = 4.5 + 1.5*np.cos(angle_rad)
        y_start = 1.3*np.sin(angle_rad)
        dx = -0.7*np.cos(angle_rad)
        dy = -0.7*np.sin(angle_rad)
        
        ax1.arrow(x_start, y_start, dx, dy, head_width=0.15, head_length=0.1, 
                 fc=COLORS['green'], ec=COLORS['green'], linewidth=1.5, alpha=0.8)
    
    ax1.text(4.5, 1.8, 'Emission', fontsize=12, ha='center', 
            color=COLORS['green'], fontweight='bold')
    ax1.text(4.5, -1.5, r'$j_\nu dV$', fontsize=12, ha='center', color=COLORS['green'])
    
    # Add absorption arrows (pointing outward)
    absorption_angles = [0, 90, 180, 270]
    for angle in absorption_angles:
        angle_rad = angle * np.pi/180
        x_start = 4.5 + 0.8*np.cos(angle_rad)
        y_start = 0.8*np.sin(angle_rad)
        dx = 0.7*np.cos(angle_rad)
        dy = 0.7*np.sin(angle_rad)
        
        ax1.arrow(x_start, y_start, dx, dy, head_width=0.15, head_length=0.1, 
                 fc=COLORS['quaternary'], ec=COLORS['quaternary'], 
                 linewidth=1.5, alpha=0.8)
    
    ax1.text(4.5, -2.2, 'Absorption', fontsize=12, ha='center', 
            color=COLORS['quaternary'], fontweight='bold')
    ax1.text(4.5, -2.5, r'$\kappa_\nu \rho I_\nu dV$', fontsize=12, ha='center', 
            color=COLORS['quaternary'])
    
    # Add labels
    ax1.text(4.5, 0, r'$ds$', fontsize=13, ha='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add RTE equation
    ax1.text(4.5, -3.5, r'$\frac{dI_\nu}{ds} = -\kappa_\nu \rho I_\nu + j_\nu$', 
            fontsize=16, ha='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['yellow'], alpha=0.3))
    
    # Middle panel: Optical depth illustration
    ax2 = fig.add_subplot(gs[1, :])
    setup_axes(ax2, title='Intensity Decay Through Optical Depth', 
              xlabel='Optical Depth (τ)', ylabel='I/I₀', grid=True)
    
    tau = np.linspace(0, 5, 100)
    I_pure_absorption = np.exp(-tau)
    I_with_source = np.exp(-tau) + 0.3*(1 - np.exp(-tau))
    
    ax2.plot(tau, I_pure_absorption, '-', color=COLORS['quaternary'], 
            linewidth=3, label='Pure Absorption (S=0)')
    ax2.plot(tau, I_with_source, '-', color=COLORS['primary'], 
            linewidth=3, label='With Source (S=0.3I₀)')
    
    # Mark important tau values
    tau_markers = [1, 2, 3]
    for t in tau_markers:
        ax2.axvline(t, color='gray', linestyle=':', alpha=0.5)
        ax2.text(t, 1.02, f'τ={t}', fontsize=10, ha='center')
    
    # Add shading for different regimes
    ax2.axvspan(0, 1, alpha=0.1, color=COLORS['green'], label='Optically Thin')
    ax2.axvspan(1, 3, alpha=0.1, color=COLORS['yellow'], label='Transition')
    ax2.axvspan(3, 5, alpha=0.1, color=COLORS['quaternary'], label='Optically Thick')
    
    ax2.set_xlim([0, 5])
    ax2.set_ylim([0, 1.1])
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add annotations for e^(-tau) values
    ax2.annotate(f'e⁻¹ ≈ 0.37', xy=(1, np.exp(-1)), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                fontsize=10, color='gray')
    ax2.annotate(f'e⁻³ ≈ 0.05', xy=(3, np.exp(-3)), xytext=(2.5, 0.2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                fontsize=10, color='gray')
    
    # Bottom panels: Three opacity regimes
    titles = ['Optically Thin (τ << 1)', 'Transition (τ ≈ 1)', 'Optically Thick (τ >> 1)']
    descriptions = ['See through medium\n+ source contribution', 
                   'See both source\nand medium', 
                   'See only surface\n(τ ≈ 1 layer)']
    tau_vals = [0.1, 1.0, 5.0]
    
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.set_title(titles[i], fontsize=11, fontweight='bold', color=COLORS['dark'])
        ax.set_xlim([0, 3])
        ax.set_ylim([0, 2])
        ax.axis('off')
        
        # Draw medium
        if i == 0:  # Thin
            alpha = 0.2
            color = COLORS['green']
        elif i == 1:  # Transition
            alpha = 0.5
            color = COLORS['yellow']
        else:  # Thick
            alpha = 0.9
            color = COLORS['quaternary']
        
        rect = Rectangle((0.5, 0.5), 2, 1, facecolor=color, alpha=alpha, 
                        edgecolor=COLORS['dark'], linewidth=2)
        ax.add_patch(rect)
        
        # Draw rays
        if i == 0:  # Most rays pass through
            for y in np.linspace(0.7, 1.3, 5):
                ax.arrow(0, y, 2.8, 0, head_width=0.08, head_length=0.1, 
                        fc='gray', ec='gray', alpha=0.7, linewidth=1)
        elif i == 1:  # Some rays absorbed
            for j, y in enumerate(np.linspace(0.7, 1.3, 5)):
                if j % 2 == 0:
                    ax.arrow(0, y, 2.8, 0, head_width=0.08, head_length=0.1, 
                            fc='gray', ec='gray', alpha=0.5, linewidth=1)
                else:
                    ax.arrow(0, y, 1.5, 0, head_width=0.08, head_length=0.1, 
                            fc='gray', ec='gray', alpha=0.3, linewidth=1)
        else:  # No rays pass through
            for y in np.linspace(0.7, 1.3, 5):
                ax.arrow(0, y, 0.8, 0, head_width=0.08, head_length=0.1, 
                        fc='gray', ec='gray', alpha=0.2, linewidth=1)
        
        ax.text(1.5, 0.2, descriptions[i], fontsize=10, ha='center', 
               style='italic', color=COLORS['dark'])
        ax.text(1.5, 1.7, f'τ = {tau_vals[i]}', fontsize=11, ha='center', 
               fontweight='bold', color=color)
    
    # Add main title
    fig.suptitle('Figure 2.2.1: The Radiative Transfer Equation Geometry', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig


def create_figure_2_3_1():
    """
    Create Figure 2.3.1: Three Scattering Regimes
    
    This figure compares three different scattering scenarios: pure absorption, mixed 
    absorption/scattering (typical of interstellar dust), and pure scattering. It illustrates
    how the albedo parameter affects photon paths and the resulting radiation field.
    
    Caption: The albedo ω determines the fate of interacting photons. Pure absorption (left) 
    removes photons from all directions. Pure scattering (right) conserves photons but 
    redistributes them, creating halos. Real dust (center) does both, making the radiative 
    transfer problem both non-local and non-conservative.
    
    Returns:
        fig: matplotlib figure object
    """
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.2,
                          height_ratios=[1.5, 0.3, 1])
    
    # Parameters for three cases
    cases = [
        {'omega': 0.0, 'title': 'Pure Absorption\n(ω = 0)', 'color': COLORS['quaternary']},
        {'omega': 0.6, 'title': 'Mixed Case (Typical Dust)\n(ω = 0.6)', 'color': COLORS['tertiary']},
        {'omega': 1.0, 'title': 'Pure Scattering\n(ω = 1)', 'color': COLORS['primary']}
    ]
    
    np.random.seed(42)  # For reproducible random paths
    
    for col, case in enumerate(cases):
        # Top panel: Photon paths visualization
        ax1 = fig.add_subplot(gs[0, col])
        ax1.set_title(case['title'], fontsize=12, fontweight='bold', color=COLORS['dark'])
        ax1.set_xlim([-2, 2])
        ax1.set_ylim([-2, 2])
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # Draw medium as a circle
        medium = Circle((0, 0), 1.5, facecolor=case['color'], alpha=0.2, 
                       edgecolor=case['color'], linewidth=2)
        ax1.add_patch(medium)
        
        # Draw source star
        star = Circle((0, 0), 0.1, color=COLORS['yellow'], edgecolor='orange', linewidth=2)
        ax1.add_patch(star)
        
        # Simulate photon paths
        n_photons = 15
        for i in range(n_photons):
            # Initial angle
            angle = 2*np.pi*i/n_photons
            x, y = [0], [0]
            
            # Trace photon path
            step_size = 0.15
            max_steps = 50
            
            for step in range(max_steps):
                # Current position
                r = np.sqrt(x[-1]**2 + y[-1]**2)
                
                if r > 1.5:  # Escaped medium
                    break
                
                # Probability of interaction
                if np.random.random() < 0.1:  # Interaction occurs
                    if np.random.random() < case['omega']:  # Scattering
                        # Change direction randomly
                        angle = 2*np.pi*np.random.random()
                        # Add scatter point
                        ax1.scatter(x[-1], y[-1], s=20, color=case['color'], 
                                   alpha=0.7, edgecolors='none')
                    else:  # Absorption
                        # Stop path here
                        ax1.scatter(x[-1], y[-1], s=30, marker='x', 
                                   color=COLORS['quaternary'], alpha=0.8)
                        break
                
                # Continue path
                x.append(x[-1] + step_size*np.cos(angle))
                y.append(y[-1] + step_size*np.sin(angle))
            
            # Draw path
            if len(x) > 1:
                alpha = 0.3 + 0.5*(i/n_photons)  # Vary transparency
                ax1.plot(x, y, '-', color='gray', alpha=alpha, linewidth=1)
        
        # Add halo for scattering cases
        if case['omega'] > 0:
            halo_intensity = case['omega']
            for r in [1.7, 1.9, 2.1]:
                halo = Circle((0, 0), r, facecolor='none', edgecolor=case['color'], 
                            alpha=halo_intensity*0.3, linewidth=15)
                ax1.add_patch(halo)
        
        # Middle panel: Parameters
        ax2 = fig.add_subplot(gs[1, col])
        ax2.axis('off')
        
        # Display parameters
        param_text = f"Albedo: ω = {case['omega']:.1f}\n"
        param_text += f"Absorption: {(1-case['omega']):.1f}\n"
        param_text += f"Scattering: {case['omega']:.1f}"
        
        ax2.text(0.5, 0.5, param_text, fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=case['color'], alpha=0.2))
        
        # Bottom panel: Intensity profile
        ax3 = fig.add_subplot(gs[2, col])
        setup_axes(ax3, title='Radial Intensity Profile', 
                  xlabel='Distance from Source', ylabel='Intensity')
        
        r = np.linspace(0, 3, 100)
        
        if case['omega'] == 0:  # Pure absorption
            intensity = np.exp(-r)
        elif case['omega'] == 1:  # Pure scattering
            # Intensity spreads out but conserves total energy
            intensity = np.exp(-r/3) + 0.2*np.exp(-r**2/4)  # Direct + scattered
        else:  # Mixed case
            intensity = 0.4*np.exp(-r) + case['omega']*0.3*np.exp(-r**2/3)
        
        ax3.plot(r, intensity, '-', color=case['color'], linewidth=3)
        ax3.fill_between(r, 0, intensity, alpha=0.3, color=case['color'])
        
        # Add vertical line at medium boundary
        ax3.axvline(1.5, color='gray', linestyle='--', alpha=0.5)
        ax3.text(1.5, 0.9, 'Medium\nedge', fontsize=9, ha='center', color='gray')
        
        ax3.set_xlim([0, 3])
        ax3.set_ylim([0, 1])
        
        # Add physical interpretation
        if col == 0:
            interpretation = 'No halo\nSharp shadows\nEnergy lost'
        elif col == 1:
            interpretation = 'Diffuse halo\nSoft shadows\nPartial energy loss'
        else:
            interpretation = 'Bright halo\nFilled shadows\nEnergy conserved'
        
        ax3.text(2.2, 0.7, interpretation, fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add overall description
    fig.text(0.5, 0.02, 'Physical Effects: Pure absorption creates sharp shadows with no scattered light. ' +
             'Pure scattering conserves photons but redistributes them spatially, filling in shadows. ' +
             'Real interstellar dust (ω ≈ 0.6) exhibits both behaviors, creating complex radiation fields.',
             fontsize=11, ha='center', style='italic', wrap=True)
    
    # Add main title
    fig.suptitle('Figure 2.3.1: Three Scattering Regimes', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig


def save_all_figures(output_dir='./figures/'):
    """
    Generate and save all figures for Module 4 Part II.
    
    This function creates all four figures illustrating radiative transfer concepts
    and saves them as high-resolution PNG files suitable for inclusion in the module.
    
    Parameters:
        output_dir (str): Directory path where figures will be saved
    
    Returns:
        dict: Dictionary containing figure objects with descriptive keys
    """
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    figures = {}
    
    print("Generating Module 4 Part II Radiative Transfer Figures...")
    print("=" * 60)
    
    # Generate Figure 2.1.1
    print("Creating Figure 2.1.1: Solid Angle and Intensity Geometry...")
    fig1 = create_figure_2_1_1()
    fig1.savefig(f'{output_dir}figure_2_1_1_solid_angle.png', dpi=150, bbox_inches='tight')
    figures['solid_angle'] = fig1
    print("  ✓ Saved as figure_2_1_1_solid_angle.png")
    
    # Generate Figure 2.1.2
    print("Creating Figure 2.1.2: Intensity Conservation vs Flux Dilution...")
    fig2 = create_figure_2_1_2()
    fig2.savefig(f'{output_dir}figure_2_1_2_intensity_flux.png', dpi=150, bbox_inches='tight')
    figures['intensity_flux'] = fig2
    print("  ✓ Saved as figure_2_1_2_intensity_flux.png")
    
    # Generate Figure 2.2.1
    print("Creating Figure 2.2.1: Radiative Transfer Equation Geometry...")
    fig3 = create_figure_2_2_1()
    fig3.savefig(f'{output_dir}figure_2_2_1_rte_geometry.png', dpi=150, bbox_inches='tight')
    figures['rte_geometry'] = fig3
    print("  ✓ Saved as figure_2_2_1_rte_geometry.png")
    
    # Generate Figure 2.3.1
    print("Creating Figure 2.3.1: Three Scattering Regimes...")
    fig4 = create_figure_2_3_1()
    fig4.savefig(f'{output_dir}figure_2_3_1_scattering_regimes.png', dpi=150, bbox_inches='tight')
    figures['scattering_regimes'] = fig4
    print("  ✓ Saved as figure_2_3_1_scattering_regimes.png")
    
    print("=" * 60)
    print(f"All figures successfully generated and saved to {output_dir}")
    print("\nFigure Descriptions:")
    print("• Figure 2.1.1: Illustrates solid angle concept and spherical coordinates")
    print("• Figure 2.1.2: Demonstrates why flux decreases while intensity is conserved")
    print("• Figure 2.2.1: Shows RTE physics with emission/absorption balance")
    print("• Figure 2.3.1: Compares absorption, scattering, and mixed regimes")
    
    return figures


if __name__ == "__main__":
    """
    Main execution: Generate all figures when script is run directly.
    
    This script generates publication-quality figures for teaching radiative transfer
    concepts in ASTR 596 Module 4 Part II. The figures build progressively from
    geometric concepts through the radiative transfer equation to scattering physics.
    """
    
    print("\n" + "="*70)
    print("  ASTR 596 - Module 4 Part II: Radiative Transfer Figures")
    print("  Mathematical Foundations of Radiative Transfer")
    print("="*70 + "\n")
    
    # Generate all figures
    figures = save_all_figures()
    
    # Display figures if matplotlib is in interactive mode
    try:
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            print("\nDisplaying figures in separate windows...")
            plt.show()
    except:
        pass
    
    print("\n✓ Figure generation complete!")
    print("These figures provide visual support for understanding:")
    print("  - How specific intensity encodes complete radiation information")
    print("  - The relationship between intensity, flux, and solid angle")
    print("  - The physical meaning of the radiative transfer equation")
    print("  - How scattering couples different directions in the radiation field")
    print("\nUse these figures alongside the module text for optimal learning.")
