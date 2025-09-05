"""
Demonstration of ASTR596 Plotting Utilities
Shows all features and capabilities of the plotting module
Author: Anna Rosen
Date: January 2025
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from importlib import import_module

# Add current directory to path and import from the actual filename
sys.path.insert(0, '.')
plotting_utils = import_module('astr596-plotting-utils')
ASTR596Plotter = plotting_utils.ASTR596Plotter
plot_spectrum = plotting_utils.plot_spectrum
plot_hr_diagram = plotting_utils.plot_hr_diagram
plot_power_spectrum = plotting_utils.plot_power_spectrum

def demo_basic_plotting():
    """Demonstrate basic plotting features."""
    print("\n=== DEMO 1: Basic Plotting Features ===")
    
    # Initialize plotter
    plotter = ASTR596Plotter(style='default', save_dir='./demo_figures/')
    
    # Create figure with multiple subplots
    fig, axes = plotter.create_figure(figsize=(15, 10), nrows=2, ncols=3)
    axes = axes.flatten()
    
    # Demo 1.1: Line plots with different color palettes
    ax = axes[0]
    x = np.linspace(0, 10, 100)
    colors = plotter.get_color_sequence(5, 'main')
    for i, color in enumerate(colors):
        y = np.sin(x + i*np.pi/4) * np.exp(-x/10)
        ax.plot(x, y, color=color, label=f'Wave {i+1}', linewidth=2)
    plotter.apply_style(ax, xlabel='Time (s)', ylabel='Amplitude', 
                       title='Main Color Palette', legend=True)
    
    # Demo 1.2: Temperature color gradient
    ax = axes[1]
    temps = np.linspace(100, 10000, 7)
    colors = plotter.get_color_sequence(7, 'temperature')
    for i, (T, color) in enumerate(zip(temps, colors)):
        y = stats.norm.pdf(x, loc=5, scale=1+i*0.2)
        ax.fill_between(x, 0, y, color=color, alpha=0.6, label=f'{T:.0f} K')
    plotter.apply_style(ax, xlabel='Velocity (km/s)', ylabel='Probability',
                       title='Temperature Palette', legend=True)
    ax.legend(ncol=2, fontsize=8)
    
    # Demo 1.3: Diverging colors for positive/negative
    ax = axes[2]
    x = np.linspace(-3, 3, 100)
    y = x**2 - 2*x + np.random.normal(0, 0.5, 100)
    colors = plotter.colors_diverging
    colors_to_use = [colors[0] if yi < 0 else colors[-1] for yi in y]
    ax.scatter(x, y, c=colors_to_use, s=20, alpha=0.6)
    ax.axhline(0, color=plotter.color_neutral, linestyle='--', linewidth=1)
    plotter.apply_style(ax, xlabel='X', ylabel='Y',
                       title='Diverging Colors (pos/neg)')
    
    # Demo 1.4: Plot with error bars and shaded regions
    ax = axes[3]
    x = np.linspace(0, 10, 20)
    y = np.exp(-x/5) + np.random.normal(0, 0.05, 20)
    yerr = 0.1 * y
    plotter.plot_with_errors(ax, x, y, yerr=yerr, 
                            color=plotter.colors_main[2],
                            label='Data', fill_between=True)
    # Add shaded regions
    plotter.add_shaded_region(ax, 2, 4, color=plotter.color_highlight, 
                             alpha=0.2, label='Region of Interest')
    plotter.apply_style(ax, xlabel='Distance (kpc)', ylabel='Flux',
                       title='Error Bars & Shaded Regions', legend=True)
    
    # Demo 1.5: Categorical data comparison
    ax = axes[4]
    categories = ['Galaxy', 'Star', 'Nebula', 'Cluster']
    values = [23, 45, 12, 67]
    colors = plotter.get_color_sequence(4, 'categorical')
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{val}', ha='center', va='bottom')
    plotter.apply_style(ax, xlabel='Object Type', ylabel='Count',
                       title='Categorical Colors')
    
    # Demo 1.6: Accept/Reject visualization
    ax = axes[5]
    np.random.seed(42)
    x_accept = np.random.normal(2, 0.5, 100)
    y_accept = np.random.normal(2, 0.5, 100)
    x_reject = np.random.normal(-1, 0.5, 50)
    y_reject = np.random.normal(-1, 0.5, 50)
    
    ax.scatter(x_accept, y_accept, color=plotter.color_accept, 
              s=30, alpha=0.6, label='Accepted')
    ax.scatter(x_reject, y_reject, color=plotter.color_reject,
              s=30, alpha=0.6, label='Rejected')
    ax.axhline(0, color=plotter.color_neutral, linestyle='--', alpha=0.5)
    ax.axvline(0, color=plotter.color_neutral, linestyle='--', alpha=0.5)
    plotter.apply_style(ax, xlabel='Parameter 1', ylabel='Parameter 2',
                       title='Accept/Reject Classification', legend=True)
    
    plt.suptitle('ASTR596 Plotting Utils: Color Palettes Demo', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plotter.save_figure(fig, 'demo_basic_features', formats=['png', 'svg'])
    print("✓ Saved: demo_basic_features")
    
    return fig


def demo_advanced_features():
    """Demonstrate advanced plotting features."""
    print("\n=== DEMO 2: Advanced Features ===")
    
    plotter = ASTR596Plotter(style='default', save_dir='./demo_figures/')
    
    # Create figure with GridSpec for complex layout
    fig, gs = plotter.create_figure_with_gridspec(figsize=(14, 10), 
                                                  nrows=3, ncols=2,
                                                  hspace=0.3, wspace=0.25)
    
    # Panel 1: Scientific notation and annotations
    ax1 = fig.add_subplot(gs[0, :])
    x = np.logspace(-3, 3, 100)
    y = x**2 * np.exp(-x/10)
    ax1.loglog(x, y, color=plotter.colors_main[1], linewidth=2)
    
    # Add annotations
    peak_idx = np.argmax(y)
    plotter.add_arrow_annotation(ax1, 'Peak', 
                                xy=(x[peak_idx], y[peak_idx]),
                                xytext=(x[peak_idx]*10, y[peak_idx]*0.1))
    
    plotter.apply_style(ax1, xlabel=r'Energy (keV)', 
                       ylabel=r'Intensity (counts/s)',
                       title='Logarithmic Plot with Annotations')
    plotter.format_axis_scientific(ax1, axis='both')
    
    # Panel 2: Inset plot
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.linspace(0, 10, 1000)
    y = np.sin(2*np.pi*x) * np.exp(-x/5)
    ax2.plot(x, y, color=plotter.colors_main[3], linewidth=2)
    
    # Create inset
    ax_inset = plotter.create_inset_axis(ax2, bounds=[0.6, 0.6, 0.35, 0.35])
    ax_inset.plot(x[:200], y[:200], color=plotter.colors_main[7], linewidth=1)
    ax_inset.set_xlim(0, 2)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.set_title('Zoom', fontsize=8)
    
    plotter.apply_style(ax2, xlabel='Time (s)', ylabel='Signal',
                       title='Main Plot with Inset')
    
    # Panel 3: Text boxes with different styles
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    ax3.plot(x, y1, label='sin(x)', color=plotter.colors_main[2])
    ax3.plot(x, y2, label='cos(x)', color=plotter.colors_main[5])
    
    # Add different text box styles
    plotter.add_text_box(ax3, 'Info: Phase shift = π/2', 
                        location='upper right', style='info')
    plotter.add_text_box(ax3, r'$\phi = \omega t$', 
                        location='lower left', style='equation')
    
    plotter.apply_style(ax3, xlabel='x', ylabel='y',
                       title='Text Box Styles', legend=True)
    
    # Panel 4: Scale bar demonstration
    ax4 = fig.add_subplot(gs[2, 0])
    # Simulate an image/map
    np.random.seed(42)
    data = np.random.randn(100, 100)
    im = ax4.imshow(data, cmap=plotter.create_colormap(plotter.colors_temperature),
                   aspect='auto', extent=[0, 50, 0, 50])
    
    # Add scale bar
    plotter.add_scalebar(ax4, length=10, label='10 pc', 
                        location='lower right', color='white')
    
    plotter.apply_style(ax4, xlabel='RA (arcsec)', ylabel='Dec (arcsec)',
                       title='Image with Scale Bar', grid=False)
    plt.colorbar(im, ax=ax4, label='Intensity')
    
    # Panel 5: Multiple comparison
    ax5 = fig.add_subplot(gs[2, 1])
    data_sets = [
        {'x': np.linspace(0, 10, 50), 
         'y': np.random.exponential(2, 50), 
         'label': 'Dataset A'},
        {'x': np.linspace(0, 10, 50), 
         'y': np.random.normal(3, 1, 50), 
         'label': 'Dataset B'},
        {'x': np.linspace(0, 10, 50), 
         'y': np.random.gamma(2, 1, 50), 
         'label': 'Dataset C'},
    ]
    plotter.plot_comparison(ax5, data_sets, plot_type='scatter')
    plotter.apply_style(ax5, xlabel='Time', ylabel='Value',
                       title='Multi-Dataset Comparison', legend=True)
    
    plt.suptitle('ASTR596 Plotting Utils: Advanced Features Demo',
                fontsize=16, fontweight='bold', y=1.00)
    
    # Save figure
    plotter.save_figure(fig, 'demo_advanced_features', formats=['png'])
    print("✓ Saved: demo_advanced_features")
    
    return fig


def demo_astrophysics_plots():
    """Demonstrate specialized astrophysics plotting functions."""
    print("\n=== DEMO 3: Astrophysics Plots ===")
    
    plotter = ASTR596Plotter(style='default', save_dir='./demo_figures/')
    
    # Create a figure with subplots for different astro plots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Spectrum plot
    ax1 = plt.subplot(3, 2, 1)
    wavelength = np.linspace(3000, 8000, 500)  # Angstroms
    flux = (1 + 0.5*np.sin(wavelength/500)) * np.exp(-((wavelength-5000)/2000)**2)
    flux += np.random.normal(0, 0.02, len(wavelength))
    flux_err = np.ones_like(flux) * 0.03
    
    plotter.plot_with_errors(ax1, wavelength, flux, yerr=flux_err,
                            color=plotter.colors_main[1], marker='',
                            linestyle='-', fill_between=True)
    # Add emission lines
    for line_wave in [4861, 6563]:  # Hβ and Hα
        ax1.axvline(line_wave, color=plotter.color_highlight, 
                   linestyle='--', alpha=0.5)
        ax1.text(line_wave, 1.1, f'{line_wave}Å', 
                rotation=90, ha='right', va='bottom', fontsize=8)
    
    plotter.apply_style(ax1, xlabel='Wavelength (Å)', 
                       ylabel='Flux (arbitrary)',
                       title='Stellar Spectrum')
    
    # 2. H-R Diagram
    ax2 = plt.subplot(3, 2, 2)
    np.random.seed(42)
    # Main sequence
    n_ms = 200
    log_temp_ms = np.random.uniform(3.5, 4.2, n_ms)
    log_lum_ms = 4.5 * (log_temp_ms - 3.5) + np.random.normal(0, 0.3, n_ms)
    
    # Giants
    n_giants = 50
    log_temp_g = np.random.uniform(3.4, 3.7, n_giants)
    log_lum_g = np.random.uniform(2, 3.5, n_giants)
    
    # Plot with temperature colors
    temp_colors = plotter.create_colormap(plotter.colors_temperature[::-1])
    scatter = ax2.scatter(log_temp_ms, log_lum_ms, c=log_temp_ms, 
                         cmap=temp_colors, s=20, alpha=0.6, label='Main Sequence')
    ax2.scatter(log_temp_g, log_lum_g, color=plotter.colors_main[7],
               s=50, alpha=0.6, label='Giants', marker='^')
    
    ax2.invert_xaxis()  # Temperature decreases to the right
    plotter.apply_style(ax2, xlabel=r'$\log(T_{\rm eff}/{\rm K})$',
                       ylabel=r'$\log(L/L_\odot)$',
                       title='H-R Diagram', legend=True)
    plt.colorbar(scatter, ax=ax2, label='Temperature')
    
    # 3. Power Spectrum
    ax3 = plt.subplot(3, 2, 3)
    k = np.logspace(-2, 1, 100)
    power = 1000 * k**(-1.5) * (1 + np.random.normal(0, 0.1, len(k)))
    theory = 1000 * k**(-1.5)
    
    ax3.loglog(k, power, 'o', color=plotter.colors_main[3],
              markersize=3, alpha=0.5, label='Data')
    ax3.loglog(k, theory, '-', color=plotter.colors_main[8],
              linewidth=2, label='Theory: k^{-1.5}')
    
    plotter.apply_style(ax3, xlabel=r'$k$ (h/Mpc)',
                       ylabel=r'$P(k)$ (Mpc/h)$^3$',
                       title='Power Spectrum', legend=True)
    
    # 4. Light Curve
    ax4 = plt.subplot(3, 2, 4)
    time = np.linspace(0, 100, 500)
    # Periodic variable star
    magnitude = 12 + 0.5*np.sin(2*np.pi*time/10) + \
                0.2*np.sin(2*np.pi*time/3) + \
                np.random.normal(0, 0.02, len(time))
    
    ax4.plot(time, magnitude, '.', color=plotter.colors_main[4],
            markersize=2, alpha=0.5)
    ax4.invert_yaxis()  # Magnitudes: brighter is smaller number
    
    # Mark phases
    for phase in [10, 20, 30, 40]:
        ax4.axvline(phase, color=plotter.color_neutral, 
                   linestyle=':', alpha=0.3)
    
    plotter.apply_style(ax4, xlabel='Time (days)',
                       ylabel='Magnitude',
                       title='Variable Star Light Curve')
    
    # 5. Galaxy rotation curve
    ax5 = plt.subplot(3, 2, 5)
    r = np.linspace(0.1, 30, 100)  # kpc
    # Observed (flat)
    v_obs = 220 * (1 - np.exp(-r/5))  # km/s
    # Keplerian (without dark matter)
    v_kep = 220 * np.sqrt(5/r) * (1 - np.exp(-r/5))
    
    ax5.plot(r, v_obs, color=plotter.colors_main[2], 
            linewidth=2.5, label='Observed')
    ax5.plot(r, v_kep, '--', color=plotter.colors_main[8],
            linewidth=2, label='Expected (no DM)')
    ax5.fill_between(r, v_kep, v_obs, alpha=0.2, 
                     color=plotter.colors_main[9],
                     label='Dark Matter')
    
    plotter.apply_style(ax5, xlabel='Radius (kpc)',
                       ylabel='Rotation Velocity (km/s)',
                       title='Galaxy Rotation Curve', legend=True)
    
    # 6. Cosmological evolution
    ax6 = plt.subplot(3, 2, 6)
    z = np.linspace(0, 5, 100)
    # Hubble parameter evolution
    H_z = 70 * np.sqrt(0.3*(1+z)**3 + 0.7)  # Simple ΛCDM
    
    ax6.plot(z, H_z, color=plotter.colors_main[5], linewidth=2.5)
    ax6.fill_between(z, H_z-5, H_z+5, alpha=0.3,
                     color=plotter.colors_main[5],
                     label='1σ uncertainty')
    
    # Mark important epochs
    plotter.add_shaded_region(ax6, 1, 2, color=plotter.color_highlight,
                             alpha=0.1, label='Peak SF epoch')
    
    plotter.apply_style(ax6, xlabel='Redshift (z)',
                       ylabel='H(z) (km/s/Mpc)',
                       title='Hubble Parameter Evolution', legend=True)
    
    plt.suptitle('ASTR596 Plotting Utils: Astrophysics Visualizations',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save figure
    plotter.save_figure(fig, 'demo_astrophysics_plots', formats=['png', 'svg'])
    print("✓ Saved: demo_astrophysics_plots")
    
    return fig


def demo_presentation_style():
    """Demonstrate presentation style with larger fonts."""
    print("\n=== DEMO 4: Presentation Style ===")
    
    # Use presentation style for bigger fonts
    plotter = ASTR596Plotter(style='presentation', save_dir='./demo_figures/')
    
    fig, ax = plotter.create_figure(figsize=(12, 8))
    
    # Simple but impactful plot
    x = np.linspace(0, 10, 100)
    colors = plotter.get_color_sequence(3, 'main')
    
    for i, (label, color) in enumerate(zip(['Early', 'Peak', 'Late'], colors)):
        y = stats.norm.pdf(x, loc=3+i*2, scale=0.8)
        ax.fill_between(x, 0, y, color=color, alpha=0.6, label=f'{label} Phase')
        ax.plot(x, y, color=color, linewidth=3)
    
    # Add key annotations
    plotter.add_arrow_annotation(ax, 'Star Formation\nBegins',
                                xy=(3, 0.45), xytext=(1, 0.35))
    plotter.add_arrow_annotation(ax, 'Peak Activity',
                                xy=(5, 0.45), xytext=(5, 0.6))
    plotter.add_arrow_annotation(ax, 'Quenching',
                                xy=(7, 0.45), xytext=(9, 0.35))
    
    plotter.apply_style(ax, xlabel='Time (Gyr)',
                       ylabel='Star Formation Rate',
                       title='Galaxy Evolution Phases',
                       legend=True)
    
    # Add info box
    plotter.add_text_box(ax, 'Key Insight:\nSFR peaks at z~2',
                        location='upper right', style='warning')
    
    # Save with presentation DPI
    plotter.save_figure(fig, 'demo_presentation_style', 
                       formats=['png'], dpi=150)
    print("✓ Saved: demo_presentation_style")
    
    return fig


def demo_data_validation():
    """Demonstrate data validation and error handling."""
    print("\n=== DEMO 5: Data Validation ===")
    
    plotter = ASTR596Plotter(save_dir='./demo_figures/')
    
    fig, axes = plotter.create_figure(figsize=(12, 5), nrows=1, ncols=2)
    
    # Generate data with some issues
    x = np.linspace(0, 10, 100)
    y_good = np.exp(-x/5) + np.random.normal(0, 0.1, 100)
    y_bad = y_good.copy()
    y_bad[20:25] = np.nan  # Add NaN values
    y_bad[50] = np.inf     # Add infinity
    
    # Plot 1: Show raw data with issues
    ax1 = axes[0]
    ax1.plot(x, y_bad, 'o', color=plotter.color_reject, 
            markersize=3, label='Raw (with issues)')
    
    # Clean the data
    try:
        y_cleaned = plotter.validate_data(y_bad, allow_inf=False)
    except ValueError as e:
        print(f"  Validation caught: {e}")
        # Remove bad values
        mask = np.isfinite(y_bad)
        x_clean = x[mask]
        y_cleaned = y_bad[mask]
        
    ax1.plot(x_clean, y_cleaned, color=plotter.color_accept,
            linewidth=2, label='Cleaned')
    
    plotter.apply_style(ax1, xlabel='X', ylabel='Y',
                       title='Data Validation Example', legend=True)
    
    # Plot 2: Show statistics
    ax2 = axes[1]
    
    # Calculate statistics
    stats_text = f"""Data Statistics:
    Original points: {len(y_bad)}
    NaN values: {np.sum(np.isnan(y_bad))}
    Inf values: {np.sum(np.isinf(y_bad))}
    Valid points: {np.sum(np.isfinite(y_bad))}
    
    After cleaning:
    Points: {len(y_cleaned)}
    Mean: {np.mean(y_cleaned):.3f}
    Std: {np.std(y_cleaned):.3f}
    """
    
    ax2.text(0.5, 0.5, stats_text, transform=ax2.transAxes,
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor=plotter.color_background,
                     edgecolor=plotter.color_neutral))
    ax2.axis('off')
    ax2.set_title('Validation Report', fontweight='bold')
    
    plt.suptitle('Data Validation and Cleaning', fontsize=14, fontweight='bold')
    
    plotter.save_figure(fig, 'demo_data_validation', formats=['png'])
    print("✓ Saved: demo_data_validation")
    
    return fig


def demo_caption_generation():
    """Demonstrate caption generation for course materials."""
    print("\n=== DEMO 6: Caption Generation ===")
    
    plotter = ASTR596Plotter(save_dir='./demo_figures/')
    
    # Create a simple figure
    fig, ax = plotter.create_figure(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x/10)
    ax.plot(x, y, color=plotter.colors_main[2], linewidth=2)
    plotter.apply_style(ax, xlabel='Time', ylabel='Amplitude',
                       title='Damped Oscillation')
    
    # Generate captions
    captions = plotter.generate_caption(
        figure_number=1,
        title='Damped Oscillation',
        description='Example of an exponentially damped sine wave, showing how oscillations decay over time.',
        details='The decay constant is τ = 10 time units.'
    )
    
    print("\n📝 Generated Captions:")
    print("\nFull Caption:")
    print(captions['full'])
    print("\nMyST Markdown:")
    print(captions['myst'])
    
    # Add caption to figure
    fig.text(0.5, -0.05, captions['full'], ha='center', fontsize=10,
            wrap=True, style='italic')
    
    plotter.save_figure(fig, 'demo_caption_generation', formats=['png'])
    print("\n✓ Saved: demo_caption_generation")
    
    return fig


def run_all_demos():
    """Run all demonstration functions."""
    print("=" * 60)
    print("ASTR596 PLOTTING UTILS - COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    
    # Run all demos
    demos = [
        demo_basic_plotting,
        demo_advanced_features,
        demo_astrophysics_plots,
        demo_presentation_style,
        demo_data_validation,
        demo_caption_generation
    ]
    
    figures = []
    for demo_func in demos:
        try:
            fig = demo_func()
            figures.append(fig)
        except Exception as e:
            print(f"  ✗ Error in {demo_func.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print(f"DEMO COMPLETE: Generated {len(figures)} demonstration figures")
    print("Check ./demo_figures/ directory for outputs")
    print("=" * 60)
    
    # Show all figures
    plt.show()
    
    return figures


if __name__ == "__main__":
    # Run all demonstrations
    figures = run_all_demos()