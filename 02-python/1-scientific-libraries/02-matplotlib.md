# Matplotlib: Publication-Quality Astronomical Visualizations

## Learning Objectives
By the end of this chapter, you will:
- Understand the anatomy of a matplotlib figure
- Create publication-ready astronomical plots
- Move beyond `plt.plot()` to object-oriented plotting
- Master complex layouts and projections
- Design effective colormaps for scientific data
- Build interactive visualizations for data exploration
- Export figures for journals and presentations

## The Anatomy of a Figure

### Understanding Every Component

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke

# Based on the famous anatomy diagram by Nicolas P. Rougier
def create_anatomy_figure():
    """
    Create a figure showing all matplotlib components.
    This is essential for understanding how to control every aspect.
    """
    
    np.random.seed(123)
    
    # Create data
    X = np.linspace(0.5, 3.5, 100)
    Y1 = 3 + np.cos(X)
    Y2 = 1 + np.cos(1 + X/0.75)/2
    Y3 = np.random.uniform(Y1, Y2, len(X))
    
    # Create figure with explicit size and DPI
    fig = plt.figure(figsize=(8, 8), dpi=100, facecolor='white')
    
    # Add axes with specific position [left, bottom, width, height]
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Configure spines (the box around the plot)
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Set limits
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    
    # Configure major and minor ticks
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    
    # Tick parameters
    ax.tick_params(which='major', length=10, width=1.5, direction='inout')
    ax.tick_params(which='minor', length=5, width=1.0, direction='inout')
    
    # Grid
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, 
            color='0.7', alpha=0.7, zorder=0)
    
    # Plot data with different zorders (drawing order)
    line1 = ax.plot(X, Y1, color='#4444FF', linewidth=2, 
                    label='Blue signal', zorder=10)
    line2 = ax.plot(X, Y2, color='#FF4444', linewidth=2, 
                    label='Red signal', zorder=9)
    scatter = ax.scatter(X, Y3, facecolor='white', edgecolor='black', 
                        s=30, linewidth=1, zorder=11)
    
    # Labels and title
    ax.set_xlabel('X axis label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y axis label', fontsize=12, fontweight='bold')
    ax.set_title('Anatomy of a Figure', fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, 
             shadow=True, borderpad=1, columnspacing=1)
    
    # Add annotations to identify components
    def annotate_component(x, y, text, xy_text=None):
        if xy_text is None:
            xy_text = (x, y - 0.3)
        ax.annotate(text, xy=(x, y), xytext=xy_text,
                   fontsize=10, fontweight='bold', color='green',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                            alpha=0.7))
    
    # Annotate key components
    annotate_component(2, 0, 'X axis', (2, -0.5))
    annotate_component(0, 2, 'Y axis', (-0.5, 2))
    annotate_component(2, 2.8, 'Line plot')
    annotate_component(3.2, 1.7, 'Scatter plot')
    annotate_component(3.5, 3.5, 'Legend')
    annotate_component(2, 4.3, 'Title', (2, 4.5))
    
    plt.suptitle('Every component can be customized!', 
                 fontsize=10, style='italic', y=0.02)
    
    return fig, ax

fig, ax = create_anatomy_figure()
plt.show()

print("\nKey figure components:")
print("- Figure: The entire window/page")
print("- Axes: The plotting area with data")
print("- Axis: The x or y axis")
print("- Spines: The borders of the axes")
print("- Ticks: Marks on the axes")
print("- Grid: Reference lines")
print("- Legend: Key for multiple datasets")
print("- Title/Labels: Text descriptions")
```

## Stop Using plt.plot()! Use Object-Oriented Interface

### Why Object-Oriented is Better

```python
# ❌ BAD: Procedural interface with plt
def bad_plotting_example():
    """What NOT to do - using plt.xxx() for everything."""
    
    plt.figure()
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bad Practice')
    # Problems:
    # - Can't control multiple figures
    # - Unclear which axes you're modifying
    # - Hard to create complex layouts
    # - Difficult to reuse or modify

# ✅ GOOD: Object-oriented interface
def good_plotting_example():
    """Professional approach using OO interface."""
    
    # Create figure and axes explicitly
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Now we have full control over each subplot
    for i, ax in enumerate(axes.flat):
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x * (i+1))
        
        # Each axes is independent
        ax.plot(x, y, label=f'$\sin({i+1}x)$')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Harmonic {i+1}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Can modify specific axes properties
        if i == 0:
            ax.set_facecolor('#f0f0f0')
    
    fig.suptitle('Object-Oriented Plotting', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    return fig, axes

fig, axes = good_plotting_example()
plt.show()
```

### Building Complex Layouts

```python
def complex_astronomical_figure():
    """Create a complex multi-panel astronomical figure."""
    
    # Create custom layout using GridSpec
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main image panel (2x2)
    ax_image = fig.add_subplot(gs[0:2, 0:2])
    
    # Histogram panels
    ax_hist_x = fig.add_subplot(gs[2, 0:2], sharex=ax_image)
    ax_hist_y = fig.add_subplot(gs[0:2, 2], sharey=ax_image)
    
    # Color bar panel
    ax_cbar = fig.add_subplot(gs[2, 2])
    
    # Generate synthetic galaxy image
    np.random.seed(42)
    x = np.linspace(-50, 50, 200)
    y = np.linspace(-50, 50, 200)
    X, Y = np.meshgrid(x, y)
    
    # Sersic profile for galaxy
    n = 4  # de Vaucouleurs profile
    r_eff = 15
    R = np.sqrt(X**2 + Y**2)
    b_n = 1.9992 * n - 0.3271  # Approximation
    intensity = np.exp(-b_n * ((R/r_eff)**(1/n) - 1))
    
    # Add noise
    noise = np.random.normal(0, 0.02, intensity.shape)
    galaxy = intensity + noise
    
    # Main image
    im = ax_image.imshow(galaxy, extent=[-50, 50, -50, 50], 
                         cmap='viridis', origin='lower',
                         vmin=0, vmax=1)
    ax_image.set_xlabel('X [pixels]')
    ax_image.set_ylabel('Y [pixels]')
    ax_image.set_title('Synthetic Galaxy')
    
    # Add contours
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    ax_image.contour(X, Y, intensity, levels=levels, colors='white', 
                     linewidths=0.5, alpha=0.5)
    
    # X-axis histogram (surface brightness profile)
    profile_x = np.sum(galaxy, axis=0)
    ax_hist_x.plot(x, profile_x, 'k-', linewidth=1)
    ax_hist_x.fill_between(x, profile_x, alpha=0.3)
    ax_hist_x.set_ylabel('Integrated Flux')
    ax_hist_x.set_xlabel('X [pixels]')
    ax_hist_x.grid(True, alpha=0.3)
    
    # Y-axis histogram
    profile_y = np.sum(galaxy, axis=1)
    ax_hist_y.plot(profile_y, y, 'k-', linewidth=1)
    ax_hist_y.fill_betweenx(y, profile_y, alpha=0.3)
    ax_hist_y.set_xlabel('Integrated Flux')
    ax_hist_y.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    
    # Remove tick labels where needed
    ax_hist_x.set_xticklabels([])
    ax_hist_y.set_yticklabels([])
    
    return fig

fig = complex_astronomical_figure()
plt.show()
```

## Publication-Quality Figures

### Setting Up for Publications

```python
def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    
    # Define publication styles
    MNRAS_style = {
        'font.size': 9,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.figsize': (3.5, 3.5),  # Single column
        'figure.dpi': 300,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
    }
    
    ApJ_style = {
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times'],
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'figure.figsize': (7, 5),  # Two column
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.format': 'eps',
    }
    
    # Apply style
    plt.rcParams.update(MNRAS_style)
    
    return MNRAS_style, ApJ_style

def create_publication_figure():
    """Example of publication-ready figure."""
    
    # Setup style
    setup_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # Panel (a): Color-magnitude diagram
    np.random.seed(42)
    n_stars = 500
    
    # Main sequence
    b_v_ms = np.random.uniform(-0.3, 1.8, n_stars)
    v_mag_ms = 4.8 + 5.5 * b_v_ms + np.random.normal(0, 0.3, n_stars)
    
    # Red giants
    n_giants = 50
    b_v_rg = np.random.uniform(0.8, 1.5, n_giants)
    v_mag_rg = np.random.uniform(-1, 3, n_giants)
    
    # Plot with proper markers and colors
    ax1.scatter(b_v_ms, v_mag_ms, s=1, c='k', alpha=0.5, rasterized=True)
    ax1.scatter(b_v_rg, v_mag_rg, s=10, c='red', marker='^', 
                edgecolors='darkred', linewidth=0.5, label='Red Giants')
    
    ax1.set_xlabel('$(B-V))
    ax1.set_ylabel('$M_V)
    ax1.set_xlim(-0.5, 2.0)
    ax1.set_ylim(12, -2)  # Inverted for magnitudes
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.legend(loc='upper left', frameon=False)
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, 
             fontweight='bold', va='top')
    
    # Panel (b): Light curve
    time = np.linspace(0, 10, 200)
    flux = 1.0 + 0.02 * np.sin(2 * np.pi * time / 2.4)  # Transit
    flux += np.random.normal(0, 0.002, len(time))  # Noise
    
    # Binned data
    n_bins = 20
    bin_edges = np.linspace(0, 10, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_flux = []
    binned_err = []
    
    for i in range(n_bins):
        mask = (time >= bin_edges[i]) & (time < bin_edges[i+1])
        if np.any(mask):
            binned_flux.append(np.mean(flux[mask]))
            binned_err.append(np.std(flux[mask]) / np.sqrt(np.sum(mask)))
        else:
            binned_flux.append(np.nan)
            binned_err.append(np.nan)
    
    # Plot with error bars
    ax2.errorbar(bin_centers, binned_flux, yerr=binned_err, 
                 fmt='o', markersize=3, capsize=2, capthick=1,
                 elinewidth=1, label='Binned data')
    ax2.plot(time, flux, 'k-', alpha=0.2, linewidth=0.5, 
             rasterized=True, label='Raw data')
    
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Relative Flux')
    ax2.set_ylim(0.975, 1.005)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.legend(loc='lower right', frameon=False)
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, 
             fontweight='bold', va='top')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save in multiple formats
    for fmt in ['pdf', 'eps', 'png']:
        filename = f'publication_figure.{fmt}'
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {filename}")
    
    return fig

# fig = create_publication_figure()
print("Publication figure example ready to run")
```

## Scientific Colormaps

### Choosing and Creating Colormaps

```python
def explore_scientific_colormaps():
    """Demonstrate proper colormap selection for astronomical data."""
    
    # Create test data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Different types of data
    intensity_data = np.exp(-(X**2 + Y**2))  # Always positive
    velocity_data = X * np.exp(-(X**2 + Y**2)/2)  # Positive and negative
    phase_data = np.angle(X + 1j*Y)  # Cyclic
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Good colormaps for intensity
    im1 = axes[0, 0].imshow(intensity_data, cmap='viridis')
    axes[0, 0].set_title('Intensity: viridis (good)')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(intensity_data, cmap='inferno')
    axes[0, 1].set_title('Intensity: inferno (good)')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # Bad colormap for intensity (but commonly used)
    im3 = axes[0, 2].imshow(intensity_data, cmap='jet')
    axes[0, 2].set_title('Intensity: jet (BAD!)')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Diverging colormaps for velocity
    im4 = axes[1, 0].imshow(velocity_data, cmap='RdBu_r', 
                            vmin=-velocity_data.max(), vmax=velocity_data.max())
    axes[1, 0].set_title('Velocity: RdBu_r (good)')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    # Cyclic colormap for phase
    im5 = axes[1, 1].imshow(phase_data, cmap='hsv')
    axes[1, 1].set_title('Phase: hsv (cyclic)')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    # Custom colormap for special data
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create custom "temperature" colormap
    colors = ['#000033', '#000055', '#0000BB', '#0E4C92', 
              '#2E8B57', '#FFD700', '#FF8C00', '#FF4500', '#8B0000']
    n_bins = 256
    cmap_custom = LinearSegmentedColormap.from_list('temperature', colors, N=n_bins)
    
    im6 = axes[1, 2].imshow(intensity_data, cmap=cmap_custom)
    axes[1, 2].set_title('Custom temperature map')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    fig.suptitle('Colormap Selection for Scientific Data', fontsize=14)
    fig.tight_layout()
    
    return fig

def create_colorblind_friendly_plot():
    """Create plots that work for colorblind readers."""
    
    # Use colorblind-friendly palette
    colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', 
              '#ECE133', '#56B4E9', '#F0E442']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.linspace(0, 2*np.pi, 100)
    
    for i, (color, style) in enumerate(zip(colors[:4], ['-', '--', '-.', ':'])):
        y = np.sin(x + i*np.pi/4)
        ax.plot(x, y, color=color, linestyle=style, linewidth=2,
                label=f'Dataset {i+1}')
    
    ax.set_xlabel('Phase')
    ax.set_ylabel('Amplitude')
    ax.set_title('Colorblind-Friendly Plot Design')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text about the principle
    ax.text(0.02, 0.02, 'Use both color AND line styles for distinction',
            transform=ax.transAxes, fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig

fig1 = explore_scientific_colormaps()
fig2 = create_colorblind_friendly_plot()
plt.show()
```

## Advanced Plot Types for Astronomy

### Specialized Astronomical Plots

```python
def create_astronomical_plots():
    """Create specialized plots common in astronomy."""
    
    fig = plt.figure(figsize=(12, 10))
    
    # 1. Sky projection plot
    ax1 = plt.subplot(2, 3, 1, projection='mollweide')
    
    # Generate random sky positions
    n_objects = 1000
    ra = np.random.uniform(-np.pi, np.pi, n_objects)
    dec = np.random.uniform(-np.pi/2, np.pi/2, n_objects)
    
    ax1.scatter(ra, dec, s=1, alpha=0.5, c=np.random.random(n_objects))
    ax1.set_xlabel('RA')
    ax1.set_ylabel('Dec')
    ax1.set_title('Sky Distribution (Mollweide)')
    ax1.grid(True)
    
    # 2. Corner plot for MCMC
    ax2 = plt.subplot(2, 3, 2)
    
    # Simulate MCMC samples
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    samples = np.random.multivariate_normal(mean, cov, 5000)
    
    from matplotlib.patches import Ellipse
    
    # Plot 2D histogram
    h, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    ax2.imshow(h.T, extent=extent, origin='lower', cmap='Blues', aspect='auto')
    
    # Add confidence ellipses
    for n_std in [1, 2, 3]:
        ellipse = Ellipse(mean, 2*n_std, 2*n_std*np.sqrt(1-0.5**2),
                         angle=45, facecolor='none', 
                         edgecolor='red', linewidth=1)
        ax2.add_patch(ellipse)
    
    ax2.set_xlabel('Parameter 1')
    ax2.set_ylabel('Parameter 2')
    ax2.set_title('MCMC Posterior')
    
    # 3. Phase-folded light curve
    ax3 = plt.subplot(2, 3, 3)
    
    # Generate transit light curve
    time = np.linspace(0, 30, 1000)
    period = 3.52
    phase = (time % period) / period
    
    flux = np.ones_like(time)
    in_transit = (phase > 0.48) & (phase < 0.52)
    flux[in_transit] = 1 - 0.02 * np.exp(-(((phase[in_transit] - 0.5)/0.01)**2))
    flux += np.random.normal(0, 0.002, len(flux))
    
    # Plot phase-folded
    ax3.scatter(phase, flux, s=1, alpha=0.5, c='k', rasterized=True)
    ax3.scatter(phase - 1, flux, s=1, alpha=0.5, c='k', rasterized=True)
    ax3.scatter(phase + 1, flux, s=1, alpha=0.5, c='k', rasterized=True)
    
    # Binned curve
    bins = np.linspace(0, 1, 50)
    digitized = np.digitize(phase, bins)
    binned = [flux[digitized == i].mean() for i in range(1, len(bins))]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax3.plot(bin_centers, binned, 'r-', linewidth=2)
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Relative Flux')
    ax3.set_title('Phase-folded Transit')
    ax3.set_xlim(-0.2, 1.2)
    
    # 4. Spectrum plot
    ax4 = plt.subplot(2, 3, 4)
    
    # Generate synthetic spectrum
    wave = np.linspace(4000, 7000, 3000)
    continuum = 1.0
    flux_spectrum = continuum + np.random.normal(0, 0.02, len(wave))
    
    # Add absorption lines
    lines = [4861, 6563, 5892]  # H-beta, H-alpha, Na D
    for line in lines:
        flux_spectrum -= 0.3 * np.exp(-((wave - line)/2)**2)
    
    ax4.plot(wave, flux_spectrum, 'k-', linewidth=0.5)
    ax4.fill_between(wave, flux_spectrum, continuum, 
                     where=(flux_spectrum < continuum), 
                     color='blue', alpha=0.3)
    
    # Mark lines
    for line in lines:
        ax4.axvline(line, color='red', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Wavelength [Å]')
    ax4.set_ylabel('Normalized Flux')
    ax4.set_title('Stellar Spectrum')
    ax4.set_ylim(0.6, 1.1)
    
    # 5. Radial profile
    ax5 = plt.subplot(2, 3, 5)
    
    # Generate galaxy profile
    r = np.logspace(-1, 2, 100)
    I_e = 100  # Intensity at r_eff
    r_eff = 10  # Effective radius
    n = 4  # Sersic index (de Vaucouleurs)
    
    b_n = 1.9992 * n - 0.3271
    intensity = I_e * np.exp(-b_n * ((r/r_eff)**(1/n) - 1))
    
    ax5.semilogy(r, intensity, 'b-', linewidth=2, label='Sersic n=4')
    ax5.axvline(r_eff, color='red', linestyle='--', label='$R_{eff})
    ax5.set_xlabel('Radius [arcsec]')
    ax5.set_ylabel('Surface Brightness')
    ax5.set_title('Galaxy Profile')
    ax5.legend()
    ax5.grid(True, alpha=0.3, which='both')
    
    # 6. Polarization vector plot
    ax6 = plt.subplot(2, 3, 6)
    
    # Generate polarization field
    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 10)
    X, Y = np.meshgrid(x, y)
    
    # Polarization from magnetic field
    P = np.sqrt(X**2 + Y**2) * 0.1
    theta = np.arctan2(Y, X) + np.pi/2  # Perpendicular to radial
    
    # Plot vectors
    U = P * np.cos(theta)
    V = P * np.sin(theta)
    
    ax6.quiver(X, Y, U, V, P, cmap='plasma')
    ax6.set_xlabel('X [pc]')
    ax6.set_ylabel('Y [pc]')
    ax6.set_title('Polarization Vectors')
    ax6.set_aspect('equal')
    
    fig.suptitle('Specialized Astronomical Plots', fontsize=14)
    fig.tight_layout()
    
    return fig

fig = create_astronomical_plots()
plt.show()
```

## Interactive Plots

### Making Plots Interactive

```python
from matplotlib.widgets import Slider, Button, CheckButtons

def create_interactive_hr_diagram():
    """Create an interactive HR diagram with controls."""
    
    # Generate cluster data
    np.random.seed(42)
    n_stars = 1000
    
    # Initial parameters
    age_gyr = 1.0
    metallicity = 0.02
    distance_modulus = 0.0
    
    def generate_cluster(age, z, dm):
        """Generate cluster CMD based on parameters."""
        # Simplified isochrone model
        masses = np.random.uniform(0.1, 2, n_stars)
        
        # Main sequence turn-off depends on age
        m_turnoff = 2.5 / (age ** 0.7)
        
        # Generate colors and magnitudes
        b_v = 0.5 * (masses - 0.5) + np.random.normal(0, 0.05, n_stars)
        
        # Absolute magnitude
        M_V = 4.8 + 5 * (masses - 1) + np.random.normal(0, 0.1, n_stars)
        
        # Apply turn-off
        is_evolved = masses > m_turnoff
        M_V[is_evolved] -= 2 * (masses[is_evolved] - m_turnoff)
        b_v[is_evolved] += 0.5
        
        # Apply metallicity effect
        M_V += 0.2 * np.log10(z / 0.02)
        
        # Apply distance modulus
        m_V = M_V + dm
        
        return b_v, m_V
    
    # Create figure and initial plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    
    b_v, m_V = generate_cluster(age_gyr, metallicity, distance_modulus)
    scatter = ax.scatter(b_v, m_V, s=1, alpha=0.5, c=b_v, cmap='RdYlBu_r')
    
    ax.set_xlabel('B - V')
    ax.set_ylabel('V magnitude')
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(20, 5)  # Inverted
    ax.grid(True, alpha=0.3)
    ax.set_title('Interactive HR Diagram')
    
    # Create sliders
    ax_age = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_metal = plt.axes([0.2, 0.10, 0.6, 0.03])
    ax_dm = plt.axes([0.2, 0.05, 0.6, 0.03])
    
    slider_age = Slider(ax_age, 'Age [Gyr]', 0.1, 10.0, valinit=age_gyr)
    slider_metal = Slider(ax_metal, 'Z', 0.001, 0.05, valinit=metallicity)
    slider_dm = Slider(ax_dm, 'DM', 0, 15, valinit=distance_modulus)
    
    def update(val):
        """Update plot when sliders change."""
        age = slider_age.val
        z = slider_metal.val
        dm = slider_dm.val
        
        b_v, m_V = generate_cluster(age, z, dm)
        
        # Update scatter plot data
        scatter.set_offsets(np.c_[b_v, m_V])
        scatter.set_array(b_v)
        
        ax.set_title(f'Age={age:.1f} Gyr, Z={z:.3f}, DM={dm:.1f}')
        fig.canvas.draw_idle()
    
    slider_age.on_changed(update)
    slider_metal.on_changed(update)
    slider_dm.on_changed(update)
    
    # Add reset button
    ax_reset = plt.axes([0.85, 0.15, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset')
    
    def reset(event):
        slider_age.reset()
        slider_metal.reset()
        slider_dm.reset()
    
    button_reset.on_clicked(reset)
    
    return fig

# Create interactive plot
# fig = create_interactive_hr_diagram()
# plt.show()
print("Interactive HR diagram ready - uncomment to run")
```

## Animation for Time Series

### Creating Scientific Animations

```python
from matplotlib.animation import FuncAnimation

def create_orbit_animation():
    """Animate a binary star orbit."""
    
    # Orbital parameters
    period = 2.0
    e = 0.5  # Eccentricity
    a = 1.0  # Semi-major axis
    
    # Time array
    t = np.linspace(0, period, 100)
    
    # Calculate orbital positions (simplified)
    E = np.linspace(0, 2*np.pi, 100)  # Eccentric anomaly
    r = a * (1 - e * np.cos(E))
    theta = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
    
    x1 = r * np.cos(theta) * 0.3  # Star 1 (more massive)
    y1 = r * np.sin(theta) * 0.3
    
    x2 = -r * np.cos(theta) * 0.7  # Star 2 (less massive)
    y2 = -r * np.sin(theta) * 0.7
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Orbit plot
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X [AU]')
    ax1.set_ylabel('Y [AU]')
    ax1.set_title('Binary Orbit')
    ax1.grid(True, alpha=0.3)
    
    # Plot orbits
    ax1.plot(x1, y1, 'b-', alpha=0.3, label='Star 1')
    ax1.plot(x2, y2, 'r-', alpha=0.3, label='Star 2')
    
    # Stars
    star1, = ax1.plot([], [], 'bo', markersize=10)
    star2, = ax1.plot([], [], 'ro', markersize=8)
    trail1, = ax1.plot([], [], 'b-', alpha=0.5, linewidth=1)
    trail2, = ax1.plot([], [], 'r-', alpha=0.5, linewidth=1)
    
    ax1.legend()
    
    # Radial velocity plot
    ax2.set_xlim(0, period)
    ax2.set_ylim(-50, 50)
    ax2.set_xlabel('Time [years]')
    ax2.set_ylabel('Radial Velocity [km/s]')
    ax2.set_title('Radial Velocity Curves')
    ax2.grid(True, alpha=0.3)
    
    # Calculate RV curves
    rv1 = -30 * np.sin(theta)
    rv2 = 30 * np.sin(theta)
    
    ax2.plot(t, rv1, 'b-', alpha=0.3)
    ax2.plot(t, rv2, 'r-', alpha=0.3)
    
    rv1_line, = ax2.plot([], [], 'b-', linewidth=2)
    rv2_line, = ax2.plot([], [], 'r-', linewidth=2)
    time_marker = ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    def init():
        star1.set_data([], [])
        star2.set_data([], [])
        trail1.set_data([], [])
        trail2.set_data([], [])
        rv1_line.set_data([], [])
        rv2_line.set_data([], [])
        return star1, star2, trail1, trail2, rv1_line, rv2_line, time_marker
    
    def animate(frame):
        # Update star positions
        star1.set_data([x1[frame]], [y1[frame]])
        star2.set_data([x2[frame]], [y2[frame]])
        
        # Update trails (last 10 points)
        trail_length = 10
        start = max(0, frame - trail_length)
        trail1.set_data(x1[start:frame+1], y1[start:frame+1])
        trail2.set_data(x2[start:frame+1], y2[start:frame+1])
        
        # Update RV plot
        rv1_line.set_data(t[:frame+1], rv1[:frame+1])
        rv2_line.set_data(t[:frame+1], rv2[:frame+1])
        time_marker.set_xdata([t[frame]])
        
        return star1, star2, trail1, trail2, rv1_line, rv2_line, time_marker
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(t), interval=50, blit=True)
    
    # Save animation
    # anim.save('binary_orbit.mp4', writer='ffmpeg', fps=20)
    
    return fig, anim

# fig, anim = create_orbit_animation()
# plt.show()
print("Animation example ready - uncomment to run")
```

## Best Practices Summary

### Figure Quality Checklist

```python
def figure_quality_guidelines():
    """Print guidelines for publication-quality figures."""
    
    guidelines = """
    PUBLICATION FIGURE CHECKLIST
    ============================
    
    1. RESOLUTION AND SIZE
    □ DPI ≥ 300 for print, ≥ 150 for web
    □ Figure size matches journal column width
    □ Fonts readable at final size (≥ 8pt)
    
    2. COLORS
    □ Colorblind-friendly palette
    □ Works in grayscale
    □ Avoid pure red/green combinations
    □ Use ColorBrewer or viridis-family colormaps
    
    3. LINES AND MARKERS
    □ Line width ≥ 1pt
    □ Different line styles for B&W printing
    □ Markers large enough to distinguish
    □ Avoid overlapping elements
    
    4. LABELS AND LEGENDS
    □ All axes labeled with units
    □ Legend doesn't obscure data
    □ Panel labels (a), (b), (c) if multi-panel
    □ Consistent notation throughout paper
    
    5. DATA PRESENTATION
    □ Error bars when appropriate
    □ Significance levels marked
    □ Sample size indicated
    □ Outliers handled appropriately
    
    6. FILE FORMATS
    □ Vector format (PDF/EPS) for line plots
    □ Rasterize dense scatter plots
    □ PNG for web, PDF for print
    □ Consistent format throughout paper
    
    7. ACCESSIBILITY
    □ Alt text for online versions
    □ Clear caption explaining figure
    □ Data available in table if needed
    □ Consider readers with visual impairments
    """
    
    print(guidelines)
    
    # Example implementation
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    
    # Good practices demonstrated
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) + np.random.normal(0, 0.1, 100)
    y2 = np.cos(x) + np.random.normal(0, 0.1, 100)
    
    # Use both color and style
    ax.plot(x, y1, 'C0-', linewidth=1.5, label='Dataset 1')
    ax.plot(x, y2, 'C1--', linewidth=1.5, label='Dataset 2')
    
    # Clear labels with units
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [arbitrary units]')
    
    # Legend outside plot area
    ax.legend(loc='upper right', frameon=False)
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set reasonable limits
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 1.5)
    
    fig.tight_layout()
    
    return fig

figure_quality_guidelines()
```

## Try It Yourself

### Exercise 1: Multi-Panel Astronomical Figure

```python
def create_grb_figure():
    """
    Create a publication-quality figure for a GRB observation.
    
    Requirements:
    - 4 panels showing: light curve, spectrum, image, SED
    - Proper layout with GridSpec
    - Publication-ready styling
    - Appropriate colormaps
    - Error bars and uncertainties
    """
    # Your code here
    pass
```

### Exercise 2: Interactive Spectrum Explorer

```python
def spectrum_explorer():
    """
    Create an interactive spectrum visualization with:
    - Slider for smoothing
    - Line identification on click
    - Zoom and pan capabilities
    - Continuum fitting options
    """
    # Your code here
    pass
```

### Exercise 3: Animated Galaxy Merger

```python
def animate_galaxy_merger():
    """
    Create an animation showing:
    - Two galaxies approaching
    - Tidal tails formation
    - Final merged system
    - Include time counter and scale bar
    """
    # Your code here
    pass
```

## Key Takeaways

✅ **Always use object-oriented interface** - More control and clarity  
✅ **Understand figure anatomy** - Every component can be customized  
✅ **Design for your audience** - Consider colorblind readers and B&W printing  
✅ **Choose colormaps carefully** - Viridis for intensity, RdBu for diverging  
✅ **Make publication-ready figures** - High DPI, clear labels, proper sizing  
✅ **Use specialized projections** - Mollweide for sky, polar for angles  
✅ **Add interactivity thoughtfully** - Sliders and widgets for exploration  
✅ **Export appropriately** - Vector for line art, raster for images  

## Next Chapter Preview
SciPy: Numerical methods and scientific algorithms for astronomical data analysis.