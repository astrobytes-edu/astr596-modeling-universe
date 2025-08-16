# Astronomy-Specific Plotting Examples for ASTR 596

This document contains advanced astronomy visualization examples that can be used as project assignments or demonstrations. These examples require students to apply the concepts from Chapter 8 to create publication-quality astronomical figures.

## Hertzsprung-Russell Diagram

The HR diagram is fundamental to stellar astronomy and demonstrates several important visualization concepts: inverted axes, logarithmic scales, density visualization, and color mapping.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

def create_hr_diagram(n_stars=10000, show_regions=True):
    """
    Create a publication-quality Hertzsprung-Russell diagram.
    
    This demonstrates:
    - Inverted x-axis (hot stars on left)
    - Logarithmic scales on both axes
    - Density visualization for large datasets
    - Annotation of stellar populations
    - Custom colormaps for temperature
    
    Parameters
    ----------
    n_stars : int
        Number of stars to simulate
    show_regions : bool
        Whether to annotate stellar populations
    
    Returns
    -------
    fig, ax : matplotlib objects
        Figure and axes for further customization
    """
    # Set up temperature colormap (blue = hot, red = cool)
    colors = ['#0000FF', '#00FFFF', '#FFFFFF', '#FFFF00', '#FF0000']
    n_bins = 256
    temp_cmap = LinearSegmentedColormap.from_list('stellar_temp', colors, N=n_bins)
    
    # Generate synthetic stellar populations
    np.random.seed(42)
    
    # Main sequence (90% of stars)
    n_ms = int(0.9 * n_stars)
    # Temperature distribution peaks around G-type stars
    ms_temp = 10**(np.random.normal(3.75, 0.15, n_ms))  # Peak at ~5600K
    # Mass-luminosity relation: L ∝ M^3.5 with scatter
    ms_mass = (ms_temp / 5778)**2  # Approximate mass from temperature
    ms_lum = ms_mass**3.5 * 10**(np.random.normal(0, 0.2, n_ms))
    
    # Red giants (8% of stars)
    n_rg = int(0.08 * n_stars)
    rg_temp = 10**(np.random.uniform(3.5, 3.7, n_rg))  # 3000-5000K
    rg_lum = 10**(np.random.uniform(1.5, 3.5, n_rg))  # 30-3000 L☉
    
    # White dwarfs (2% of stars)
    n_wd = n_stars - n_ms - n_rg
    wd_temp = 10**(np.random.uniform(3.7, 4.3, n_wd))  # 5000-20000K
    wd_lum = 10**(np.random.uniform(-4, -2, n_wd))  # 0.0001-0.01 L☉
    
    # Combine all populations
    temps = np.concatenate([ms_temp, rg_temp, wd_temp])
    lums = np.concatenate([ms_lum, rg_lum, wd_lum])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use hexbin for density visualization with large datasets
    hb = ax.hexbin(temps, lums, xscale='log', yscale='log',
                   gridsize=50, cmap='YlOrRd', mincnt=1,
                   edgecolors='none', linewidths=0.2)
    
    # Critical: Invert x-axis (hot stars on left)
    ax.invert_xaxis()
    
    # Set axis properties
    ax.set_xlabel('Effective Temperature (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Luminosity (L☉)', fontsize=14, fontweight='bold')
    ax.set_title('Hertzsprung-Russell Diagram', fontsize=16, fontweight='bold')
    
    # Set appropriate limits
    ax.set_xlim(40000, 2000)  # Note: reversed because axis is inverted
    ax.set_ylim(1e-4, 1e6)
    
    # Add grid
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="minor", ls=":", alpha=0.1)
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label('Number of Stars', fontsize=12)
    
    # Annotate stellar populations if requested
    if show_regions:
        # Main sequence band
        ms_patch = patches.Polygon([(25000, 1e4), (3000, 0.01), 
                                   (2500, 0.001), (20000, 1e3)],
                                  closed=True, fill=False, edgecolor='blue',
                                  linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(ms_patch)
        ax.text(10000, 100, 'Main Sequence', fontsize=12, 
                color='blue', fontweight='bold', rotation=-45)
        
        # Red giant branch
        ax.annotate('Red Giants', xy=(4000, 100), xytext=(3000, 1000),
                   fontsize=12, color='red', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        # White dwarf region
        ax.annotate('White Dwarfs', xy=(10000, 0.001), xytext=(15000, 0.00001),
                   fontsize=12, color='gray', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        
        # Add spectral classes on top
        spectral_classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
        spectral_temps = [30000, 20000, 8500, 6500, 5500, 4500, 3000]
        for sc, st in zip(spectral_classes, spectral_temps):
            if st >= 2000 and st <= 40000:  # Within plot range
                ax.text(st, 2e5, sc, fontsize=11, ha='center', 
                       fontweight='bold', color='navy')
    
    # Add reference stars
    sun = ax.scatter([5778], [1], s=200, c='yellow', marker='*',
                    edgecolors='orange', linewidths=2, zorder=5,
                    label='Sun')
    
    # Add legend
    ax.legend(loc='lower left', fontsize=11)
    
    plt.tight_layout()
    
    return fig, ax

# Example usage:
# fig, ax = create_hr_diagram(n_stars=50000, show_regions=True)
# plt.show()
```

## Light Curves with Phase Folding

Light curves are essential for studying variable stars, exoplanets, and transient events. This example shows how to create publication-quality light curve plots with phase folding.

```python
def create_transit_light_curve(duration_hours=3, period_days=2.5, depth=0.01,
                               noise_level=0.001, n_transits=5):
    """
    Create a simulated exoplanet transit light curve with phase folding.
    
    Demonstrates:
    - Time series visualization
    - Error bars and uncertainties
    - Phase folding techniques
    - Multiple panel layouts
    - Residual plots
    
    Parameters
    ----------
    duration_hours : float
        Transit duration in hours
    period_days : float
        Orbital period in days
    depth : float
        Transit depth (fractional)
    noise_level : float
        Photometric noise level
    n_transits : int
        Number of transits to simulate
    
    Returns
    -------
    fig : matplotlib figure
        Complete figure with raw and folded light curves
    """
    # Generate time array covering multiple transits
    total_time = period_days * n_transits
    time = np.linspace(0, total_time, 2000)
    
    # Create transit model (simplified box model)
    flux = np.ones_like(time)
    transit_duration_days = duration_hours / 24
    
    for i in range(n_transits):
        transit_center = period_days * (i + 0.5)
        in_transit = np.abs(time - transit_center) < transit_duration_days / 2
        flux[in_transit] = 1 - depth
    
    # Add realistic noise
    noise = np.random.normal(0, noise_level, len(time))
    observed_flux = flux + noise
    uncertainties = np.full_like(time, noise_level)
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(14, 10))
    
    # Panel 1: Raw light curve
    ax1 = plt.subplot(3, 1, 1)
    ax1.errorbar(time, observed_flux, yerr=uncertainties, 
                fmt='o', markersize=2, alpha=0.5, ecolor='gray',
                elinewidth=0.5, capsize=0, label='Observed data')
    ax1.plot(time, flux, 'r-', linewidth=2, label='Transit model')
    ax1.set_xlabel('Time (days)', fontsize=12)
    ax1.set_ylabel('Relative Flux', fontsize=12)
    ax1.set_title('Raw Light Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Calculate phase
    phase = (time % period_days) / period_days
    # Center on transit (phase 0.5 -> 0)
    phase_centered = phase - 0.5
    phase_centered[phase_centered < -0.5] += 1
    
    # Panel 2: Phase-folded light curve
    ax2 = plt.subplot(3, 1, 2)
    
    # Sort by phase for cleaner plotting
    sort_idx = np.argsort(phase_centered)
    
    ax2.errorbar(phase_centered[sort_idx], observed_flux[sort_idx], 
                yerr=uncertainties[sort_idx],
                fmt='o', markersize=2, alpha=0.3, ecolor='gray',
                elinewidth=0.5, capsize=0, label='Folded data')
    
    # Overplot model
    model_phase = np.linspace(-0.5, 0.5, 1000)
    model_flux = np.ones_like(model_phase)
    transit_phase_width = transit_duration_days / period_days
    in_transit_model = np.abs(model_phase) < transit_phase_width / 2
    model_flux[in_transit_model] = 1 - depth
    
    ax2.plot(model_phase, model_flux, 'r-', linewidth=2, label='Model')
    ax2.set_xlabel('Orbital Phase', fontsize=12)
    ax2.set_ylabel('Relative Flux', fontsize=12)
    ax2.set_title(f'Phase-Folded Light Curve (Period = {period_days:.2f} days)',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Zoom in on transit
    ax2.set_xlim(-0.1, 0.1)
    
    # Panel 3: Residuals
    ax3 = plt.subplot(3, 1, 3)
    residuals = observed_flux - flux
    
    ax3.errorbar(phase_centered[sort_idx], residuals[sort_idx] * 1e6,
                yerr=uncertainties[sort_idx] * 1e6,
                fmt='o', markersize=2, alpha=0.5, ecolor='gray',
                elinewidth=0.5, capsize=0)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax3.set_xlabel('Orbital Phase', fontsize=12)
    ax3.set_ylabel('Residuals (ppm)', fontsize=12)
    ax3.set_title('Residuals', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.1, 0.1)
    
    # Add statistics box
    rms = np.std(residuals) * 1e6
    ax3.text(0.02, 0.95, f'RMS = {rms:.1f} ppm\nχ² = {np.sum((residuals/noise_level)**2)/len(residuals):.2f}',
            transform=ax3.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Transit Light Curve Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

# Example usage:
# fig = create_transit_light_curve(duration_hours=2.5, period_days=3.2, depth=0.015)
# plt.show()
```

## Spectral Energy Distributions (SEDs)

SEDs are crucial for understanding the physics of astronomical objects. This example shows how to create professional SED plots with multiple components.

```python
def create_sed_plot(teff=5778, add_components=True):
    """
    Create a spectral energy distribution plot with multiple components.
    
    Demonstrates:
    - Log-log plots for wide wavelength/flux ranges
    - Multiple components (stellar, disk, etc.)
    - Observational data with error bars
    - Proper wavelength/frequency axes
    - Color coding by wavelength region
    
    Parameters
    ----------
    teff : float
        Effective temperature in Kelvin
    add_components : bool
        Whether to show individual components
    
    Returns
    -------
    fig, ax : matplotlib objects
        Figure and axes objects
    """
    # Wavelength range from UV to far-IR (0.1 to 1000 microns)
    wavelength = np.logspace(-1, 3, 1000)  # microns
    
    # Planck function for stellar photosphere
    def planck(wav, temp):
        """Planck function in wavelength space."""
        from scipy.constants import h, c, k
        wav_m = wav * 1e-6  # Convert to meters
        B = (2 * h * c**2 / wav_m**5) / (np.exp(h * c / (wav_m * k * temp)) - 1)
        return B * 1e-6  # Convert to per micron
    
    # Stellar photosphere
    stellar_flux = planck(wavelength, teff)
    stellar_flux *= (6.96e8 / 1.496e11)**2  # Scale by (R_sun/AU)^2
    
    # Optional disk component (simplified)
    if add_components:
        # Inner disk (hot dust)
        disk_inner = planck(wavelength, 1500) * 1e-3
        # Outer disk (cold dust)
        disk_outer = planck(wavelength, 50) * 1e-5
        total_flux = stellar_flux + disk_inner + disk_outer
    else:
        total_flux = stellar_flux
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot components
    ax.loglog(wavelength, stellar_flux, 'b-', linewidth=2, 
              label=f'Stellar Photosphere (T={teff}K)')
    
    if add_components:
        ax.loglog(wavelength, disk_inner, 'r--', linewidth=1.5,
                 label='Inner Disk (Hot Dust)', alpha=0.7)
        ax.loglog(wavelength, disk_outer, 'g--', linewidth=1.5,
                 label='Outer Disk (Cold Dust)', alpha=0.7)
        ax.loglog(wavelength, total_flux, 'k-', linewidth=2.5,
                 label='Total SED', alpha=0.8)
    
    # Add simulated photometric observations
    # Wavelengths for common bands (microns)
    bands = {
        'U': 0.365, 'B': 0.445, 'V': 0.551, 'R': 0.658, 'I': 0.806,  # Optical
        'J': 1.22, 'H': 1.63, 'K': 2.19,  # Near-IR
        'W1': 3.4, 'W2': 4.6, 'W3': 12, 'W4': 22,  # WISE
        'MIPS24': 24, 'MIPS70': 70, 'MIPS160': 160  # Spitzer
    }
    
    # Simulate observations with errors
    obs_wav = []
    obs_flux = []
    obs_err = []
    
    for band, wav in bands.items():
        if wav >= wavelength.min() and wav <= wavelength.max():
            obs_wav.append(wav)
            # Interpolate model at this wavelength
            model_flux = np.interp(wav, wavelength, total_flux)
            # Add realistic scatter
            scatter = 10**(np.random.normal(0, 0.05))
            obs_flux.append(model_flux * scatter)
            obs_err.append(model_flux * 0.1)  # 10% errors
    
    obs_wav = np.array(obs_wav)
    obs_flux = np.array(obs_flux)
    obs_err = np.array(obs_err)
    
    # Plot observations
    ax.errorbar(obs_wav, obs_flux, yerr=obs_err, fmt='o', 
               markersize=8, markerfacecolor='orange',
               markeredgecolor='darkred', markeredgewidth=1,
               ecolor='gray', elinewidth=1, capsize=3,
               label='Photometric Observations', zorder=5)
    
    # Shade wavelength regions
    ax.axvspan(0.1, 0.4, alpha=0.1, color='violet', label='UV')
    ax.axvspan(0.4, 0.7, alpha=0.1, color='yellow', label='Optical')
    ax.axvspan(0.7, 5, alpha=0.1, color='red', label='Near-IR')
    ax.axvspan(5, 40, alpha=0.1, color='orange', label='Mid-IR')
    ax.axvspan(40, 1000, alpha=0.1, color='brown', label='Far-IR')
    
    # Labels and formatting
    ax.set_xlabel('Wavelength (μm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Flux Density (W m⁻² μm⁻¹)', fontsize=14, fontweight='bold')
    ax.set_title('Spectral Energy Distribution', fontsize=16, fontweight='bold')
    
    # Set reasonable limits
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(1e-16, 1e-10)
    
    # Add grid
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="minor", ls=":", alpha=0.1)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add secondary x-axis for frequency
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    # Frequency = c/wavelength
    freq_ticks = np.array([3e9, 3e10, 3e11, 3e12, 3e13, 3e14, 3e15])
    wav_ticks = 3e14 / freq_ticks  # c in μm/s
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(wav_ticks)
    ax2.set_xticklabels([f'{f/1e12:.0f}' for f in freq_ticks])
    ax2.set_xlabel('Frequency (THz)', fontsize=12)
    
    plt.tight_layout()
    
    return fig, ax

# Example usage:
# fig, ax = create_sed_plot(teff=5778, add_components=True)
# plt.show()
```

## Radial Profiles and Surface Brightness

Radial profiles are essential for studying galaxies, stellar atmospheres, and other extended objects.

```python
def create_radial_profile(galaxy_type='spiral'):
    """
    Create galaxy surface brightness radial profile plots.
    
    Demonstrates:
    - Semi-log plots for exponential profiles
    - Multiple component fitting
    - Error regions and uncertainties
    - Proper axis scaling for astronomical units
    
    Parameters
    ----------
    galaxy_type : str
        Type of galaxy ('spiral', 'elliptical', 'dwarf')
    
    Returns
    -------
    fig : matplotlib figure
        Complete figure with profile and residuals
    """
    # Generate radial distances
    r = np.logspace(-1, 2, 100)  # 0.1 to 100 kpc
    
    # Surface brightness profiles (mag/arcsec^2)
    if galaxy_type == 'spiral':
        # Exponential disk + bulge
        I0_disk = 20.5  # Central surface brightness
        h_disk = 3.5    # Scale length (kpc)
        disk = I0_disk + 1.086 * r / h_disk  # mag/arcsec^2
        
        I0_bulge = 18.0
        r_eff = 1.0  # Effective radius
        bulge = I0_bulge + 3.33 * ((r/r_eff)**0.25 - 1)  # de Vaucouleurs
        
        # Combine (in linear space, then back to mags)
        total_flux = 10**(-0.4*disk) + 10**(-0.4*bulge)
        total = -2.5 * np.log10(total_flux)
        
    elif galaxy_type == 'elliptical':
        # Pure de Vaucouleurs profile
        I0 = 17.5
        r_eff = 5.0
        total = I0 + 3.33 * ((r/r_eff)**0.25 - 1)
        disk = np.full_like(r, np.nan)
        bulge = total
        
    else:  # dwarf
        # Simple exponential
        I0 = 22.0
        h = 1.0
        total = I0 + 1.086 * r / h
        disk = total
        bulge = np.full_like(r, np.nan)
    
    # Add observational scatter
    noise = np.random.normal(0, 0.1, len(r))
    observed = total + noise
    errors = np.full_like(r, 0.1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main profile plot
    ax1.errorbar(r, observed, yerr=errors, fmt='o', markersize=4,
                alpha=0.6, ecolor='gray', elinewidth=0.5, capsize=2,
                label='Observed', zorder=3)
    
    # Plot components
    if not np.all(np.isnan(disk)):
        ax1.plot(r, disk, 'b--', linewidth=2, label='Disk component', alpha=0.7)
    if not np.all(np.isnan(bulge)):
        ax1.plot(r, bulge, 'r--', linewidth=2, label='Bulge component', alpha=0.7)
    
    ax1.plot(r, total, 'k-', linewidth=2.5, label='Total model')
    
    # Formatting
    ax1.set_xscale('log')
    ax1.set_xlabel('Radius (kpc)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Surface Brightness (mag/arcsec²)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{galaxy_type.capitalize()} Galaxy Radial Profile',
                  fontsize=14, fontweight='bold')
    ax1.invert_yaxis()  # Brighter is up
    ax1.set_ylim(28, 16)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add scale indicators
    ax1.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
    ax1.text(1, 17, '1 kpc', rotation=90, va='bottom', ha='right', alpha=0.5)
    
    if galaxy_type == 'spiral':
        ax1.axvline(x=h_disk, color='blue', linestyle=':', alpha=0.5)
        ax1.text(h_disk, 17, f'h = {h_disk} kpc', rotation=90, 
                va='bottom', ha='right', color='blue', alpha=0.7)
    
    # Residuals panel
    residuals = observed - total
    
    ax2.errorbar(r, residuals, yerr=errors, fmt='o', markersize=4,
                alpha=0.6, ecolor='gray', elinewidth=0.5, capsize=2)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax2.axhspan(-0.1, 0.1, alpha=0.2, color='gray')  # 1-sigma region
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Radius (kpc)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals\n(mag/arcsec²)', fontsize=10)
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    
    return fig

# Example usage:
# fig = create_radial_profile('spiral')
# plt.show()
```

## Color-Magnitude Diagrams

Color-magnitude diagrams are essential for studying stellar populations in clusters and galaxies.

```python
def create_cmd(cluster_age=1e9, distance_modulus=0, metallicity=0.0):
    """
    Create a color-magnitude diagram for a stellar cluster.
    
    Demonstrates:
    - Density plots for crowded fields
    - Isochrone overlays
    - Proper magnitude systems
    - Selection boxes for different populations
    
    Parameters
    ----------
    cluster_age : float
        Age in years
    distance_modulus : float
        Distance modulus (m-M)
    metallicity : float
        [Fe/H] in dex
    
    Returns
    -------
    fig, ax : matplotlib objects
        Figure and axes
    """
    # Simulate stellar population
    np.random.seed(42)
    n_stars = 5000
    
    # Main sequence turnoff depends on age
    turnoff_mass = (cluster_age / 1e10)**(-1/2.5)  # Approximate
    
    # Generate stellar masses (Salpeter IMF)
    masses = np.random.pareto(2.35, n_stars) * 0.1 + 0.1
    masses = masses[masses < 10]  # Limit to < 10 solar masses
    
    # Convert to magnitudes (simplified)
    # Absolute magnitude
    M_V = 4.83 - 2.5 * np.log10((masses/1)**3.5)  # Main sequence L-M relation
    
    # Color depends on mass/temperature
    B_V = 1.5 - 0.5 * np.log10(masses)  # Approximate B-V color
    
    # Add distance modulus
    V = M_V + distance_modulus
    
    # Add photometric errors
    V_err = 0.01 * 10**(0.2 * (V - 20))  # Larger errors for fainter stars
    B_V_err = V_err * 1.5
    
    V += np.random.normal(0, V_err)
    B_V += np.random.normal(0, B_V_err)
    
    # Add some red giants
    n_giants = 100
    V_giants = np.random.uniform(17, 19, n_giants) + distance_modulus
    B_V_giants = np.random.uniform(1.0, 1.8, n_giants)
    
    # Combine all stars
    V_all = np.concatenate([V, V_giants])
    B_V_all = np.concatenate([B_V, B_V_giants])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Use hexbin for density in crowded regions
    hb = ax.hexbin(B_V_all, V_all, gridsize=50, cmap='YlOrRd',
                   mincnt=1, edgecolors='none')
    
    # Invert y-axis (bright stars at top)
    ax.invert_yaxis()
    
    # Labels
    ax.set_xlabel('B - V (mag)', fontsize=14, fontweight='bold')
    ax.set_ylabel('V (mag)', fontsize=14, fontweight='bold')
    ax.set_title(f'Color-Magnitude Diagram\nAge = {cluster_age/1e9:.1f} Gyr, [Fe/H] = {metallicity:.1f}',
                 fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set limits
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(25, 15)
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Number of Stars', fontsize=12)
    
    # Add population labels
    ax.text(0.0, 20, 'Main Sequence', fontsize=12, color='blue',
            fontweight='bold', rotation=-45)
    ax.text(1.4, 18, 'Red Giant\nBranch', fontsize=12, color='red',
            fontweight='bold', ha='center')
    
    # Add selection box for red giants
    from matplotlib.patches import Rectangle
    rgb_box = Rectangle((0.9, 16.5), 1.0, 3.0, fill=False,
                        edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(rgb_box)
    
    # Add main sequence turnoff point
    turnoff_V = 4.83 - 2.5 * np.log10((turnoff_mass/1)**3.5) + distance_modulus
    turnoff_BV = 1.5 - 0.5 * np.log10(turnoff_mass)
    ax.scatter([turnoff_BV], [turnoff_V], s=200, c='green', marker='*',
              edgecolors='darkgreen', linewidths=2, zorder=5,
              label=f'Turnoff (M={turnoff_mass:.1f} M☉)')
    
    # Add theoretical isochrone (simplified)
    iso_masses = np.logspace(-1, np.log10(turnoff_mass), 100)
    iso_MV = 4.83 - 2.5 * np.log10((iso_masses/1)**3.5)
    iso_V = iso_MV + distance_modulus
    iso_BV = 1.5 - 0.5 * np.log10(iso_masses)
    
    ax.plot(iso_BV, iso_V, 'g-', linewidth=2, alpha=0.7,
           label=f'Isochrone ({cluster_age/1e9:.1f} Gyr)')
    
    # Legend
    ax.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    return fig, ax

# Example usage:
# fig, ax = create_cmd(cluster_age=5e9, distance_modulus=12.5, metallicity=-0.5)
# plt.show()
```

## Period-Luminosity Relations (Leavitt Law)

The period-luminosity relation for Cepheid variables is fundamental for distance measurements.

```python
def create_period_luminosity_plot(add_types=True, show_instability_strip=True):
    """
    Create a period-luminosity diagram for pulsating variables.
    
    Demonstrates:
    - Log-linear plots
    - Multiple populations with different relations
    - Fitting and displaying power laws
    - Distance determination examples
    
    Parameters
    ----------
    add_types : bool
        Show different types of variables
    show_instability_strip : bool
        Show the instability strip region
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    # Generate synthetic Cepheid data
    np.random.seed(42)
    
    # Classical Cepheids
    n_cepheids = 100
    log_P_cep = np.random.uniform(0.5, 2.0, n_cepheids)  # log(days)
    # Leavitt law: M_V = -2.81 * log(P) - 1.43
    M_V_cep = -2.81 * log_P_cep - 1.43 + np.random.normal(0, 0.15, n_cepheids)
    
    # Type II Cepheids (fainter)
    n_type2 = 50
    log_P_type2 = np.random.uniform(0.0, 1.5, n_type2)
    M_V_type2 = -2.0 * log_P_type2 - 0.5 + np.random.normal(0, 0.2, n_type2)
    
    # RR Lyrae (shorter periods, nearly constant luminosity)
    n_rrl = 80
    log_P_rrl = np.random.uniform(-0.5, 0.0, n_rrl)
    M_V_rrl = 0.6 + np.random.normal(0, 0.1, n_rrl)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main P-L plot
    if add_types:
        ax1.scatter(10**log_P_cep, M_V_cep, s=50, alpha=0.7,
                   label='Classical Cepheids', color='blue')
        ax1.scatter(10**log_P_type2, M_V_type2, s=40, alpha=0.6,
                   label='Type II Cepheids', color='green')
        ax1.scatter(10**log_P_rrl, M_V_rrl, s=30, alpha=0.6,
                   label='RR Lyrae', color='red')
    else:
        ax1.scatter(10**log_P_cep, M_V_cep, s=50, alpha=0.7, color='blue')
    
    # Fit and plot Leavitt law
    from scipy.optimize import curve_fit
    
    def leavitt_law(P, a, b):
        return a * np.log10(P) + b
    
    popt, pcov = curve_fit(leavitt_law, 10**log_P_cep, M_V_cep)
    perr = np.sqrt(np.diag(pcov))
    
    P_fit = np.logspace(-0.5, 2.5, 100)
    M_V_fit = leavitt_law(P_fit, *popt)
    
    ax1.plot(P_fit, M_V_fit, 'b-', linewidth=2,
            label=f'Fit: $M_V = {popt[0]:.2f} \log P {popt[1]:+.2f})
    
    # Show 1-sigma uncertainty band
    M_V_upper = leavitt_law(P_fit, popt[0] + perr[0], popt[1] - perr[1])
    M_V_lower = leavitt_law(P_fit, popt[0] - perr[0], popt[1] + perr[1])
    ax1.fill_between(P_fit, M_V_upper, M_V_lower, alpha=0.2, color='blue')
    
    # Formatting
    ax1.set_xscale('log')
    ax1.invert_yaxis()  # Brighter at top
    ax1.set_xlabel('Period (days)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Absolute Magnitude $M_V, fontsize=14, fontweight='bold')
    ax1.set_title('Period-Luminosity Relation (Leavitt Law)',
                 fontsize=16, fontweight='bold')
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11)
    
    # Add reference lines for specific periods
    reference_periods = [1, 10, 30]
    for P_ref in reference_periods:
        M_ref = leavitt_law(P_ref, *popt)
        ax1.axvline(x=P_ref, color='gray', linestyle=':', alpha=0.5)
        ax1.text(P_ref, -6, f'{P_ref}d', ha='center', va='top',
                fontsize=9, color='gray')
    
    # Distance determination example (bottom panel)
    # Simulate observing a Cepheid
    P_obs = 25  # days
    m_V_obs = 15.5  # apparent magnitude
    M_V_pred = leavitt_law(P_obs, *popt)
    distance_modulus = m_V_obs - M_V_pred
    distance_pc = 10**(1 + distance_modulus/5)
    
    ax2.scatter([P_obs], [m_V_obs], s=200, color='orange', marker='*',
               edgecolors='darkorange', linewidths=2, zorder=5)
    
    # Show range of periods for scale
    P_range = np.logspace(-0.5, 2.5, 100)
    m_V_range = leavitt_law(P_range, *popt) + distance_modulus
    ax2.plot(P_range, m_V_range, 'orange', linewidth=2,
            label=f'd = {distance_pc/1000:.1f} kpc')
    
    ax2.set_xscale('log')
    ax2.invert_yaxis()
    ax2.set_xlabel('Period (days)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Apparent Magnitude $m_V, fontsize=14, fontweight='bold')
    ax2.set_title(f'Distance Determination Example: P = {P_obs}d, $m_V$ = {m_V_obs}',
                 fontsize=12)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Add text box with calculation
    textstr = f'Observed: P = {P_obs} days, $m_V$ = {m_V_obs}\n'
    textstr += f'Predicted: $M_V$ = {M_V_pred:.2f}\n'
    textstr += f'Distance modulus: μ = {distance_modulus:.2f}\n'
    textstr += f'Distance: d = {distance_pc:.0f} pc = {distance_pc/1000:.1f} kpc'
    
    ax2.text(0.98, 0.95, textstr, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)

# Example usage:
# fig, axes = create_period_luminosity_plot(add_types=True)
# plt.show()
```

## Galaxy Rotation Curves

Rotation curves are crucial evidence for dark matter and understanding galaxy dynamics.

```python
def create_rotation_curve(galaxy_type='spiral', add_dark_matter=True):
    """
    Create a galaxy rotation curve showing the dark matter problem.
    
    Demonstrates:
    - Combining theoretical models with observations
    - Error bars and systematic uncertainties
    - Component decomposition
    - Scientific annotation
    
    Parameters
    ----------
    galaxy_type : str
        Type of galaxy
    add_dark_matter : bool
        Include dark matter halo
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    # Radial distances
    r = np.linspace(0.1, 30, 100)  # kpc
    
    # Observed rotation curve (flat at large radii)
    v_obs = 220 * (1 - np.exp(-r/2))  # km/s, asymptotes to 220 km/s
    # Add realistic scatter
    v_obs += np.random.normal(0, 10, len(r))
    v_err = 5 + 5 * r/30  # Errors increase with radius
    
    # Theoretical components
    # 1. Bulge (concentrated in center)
    M_bulge = 1e10  # Solar masses
    r_bulge = 1.0   # kpc
    v_bulge = np.sqrt(4.3e-6 * M_bulge * r / (r**2 + r_bulge**2))
    
    # 2. Disk (exponential)
    M_disk = 5e10
    r_disk = 3.0
    # Simplified disk contribution
    v_disk = np.sqrt(4.3e-6 * M_disk * r / (r + r_disk)**2) * 2
    
    # 3. Dark matter halo (NFW profile)
    if add_dark_matter:
        M_halo = 1e12
        r_s = 10  # Scale radius
        # Simplified NFW
        x = r / r_s
        v_halo = np.sqrt(4.3e-6 * M_halo / r * 
                        (np.log(1 + x) - x/(1 + x)))
    else:
        v_halo = np.zeros_like(r)
    
    # Total theoretical curve
    v_total = np.sqrt(v_bulge**2 + v_disk**2 + v_halo**2)
    v_no_dm = np.sqrt(v_bulge**2 + v_disk**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot observations
    ax.errorbar(r, v_obs, yerr=v_err, fmt='o', markersize=5,
               color='black', ecolor='gray', elinewidth=1,
               capsize=3, alpha=0.7, label='Observed', zorder=5)
    
    # Plot components
    ax.plot(r, v_bulge, 'b--', linewidth=1.5, alpha=0.7, label='Bulge')
    ax.plot(r, v_disk, 'g--', linewidth=1.5, alpha=0.7, label='Disk')
    
    if add_dark_matter:
        ax.plot(r, v_halo, 'r--', linewidth=1.5, alpha=0.7, label='Dark Matter Halo')
        ax.plot(r, v_total, 'r-', linewidth=2.5, label='Total (with DM)')
        ax.plot(r, v_no_dm, 'b-', linewidth=2, alpha=0.5, label='Total (no DM)')
    else:
        ax.plot(r, v_no_dm, 'b-', linewidth=2.5, label='Total (visible matter only)')
    
    # Formatting
    ax.set_xlabel('Radius (kpc)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rotation Velocity (km/s)', fontsize=14, fontweight='bold')
    ax.set_title(f'{galaxy_type.capitalize()} Galaxy Rotation Curve',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    
    # Set limits
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 300)
    
    # Add annotations
    if add_dark_matter:
        # Highlight the discrepancy
        ax.annotate('', xy=(20, v_no_dm[np.argmin(np.abs(r - 20))]),
                   xytext=(20, v_obs[np.argmin(np.abs(r - 20))]),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(21, 150, 'Dark matter\nrequired', fontsize=11,
               color='red', fontweight='bold')
    
    # Add text box with galaxy parameters
    textstr = f'Galaxy Parameters:\n'
    textstr += f'$M_{{bulge}}$ = {M_bulge:.1e} M☉\n'
    textstr += f'$M_{{disk}}$ = {M_disk:.1e} M☉\n'
    if add_dark_matter:
        textstr += f'$M_{{halo}}$ = {M_halo:.1e} M☉\n'
        textstr += f'$M_{{dark}}/M_{{visible}}$ = {M_halo/(M_bulge + M_disk):.1f}'
    
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    return fig, ax

# Example usage:
# fig, ax = create_rotation_curve('spiral', add_dark_matter=True)
# plt.show()
```

## Usage Notes for Instructors

These examples are designed to be challenging enough that students must understand the underlying concepts to modify them effectively. Consider using them as:

1. **Project Templates**: Give students the basic structure but require them to add features (e.g., add RR Lyrae to the HR diagram, include reddening in the CMD)

2. **Debugging Exercises**: Introduce deliberate bugs (wrong scale, inverted axes, incorrect units) and have students fix them

3. **Data Integration**: Provide real astronomical data and have students adapt these templates to visualize it

4. **Comparative Studies**: Have students create similar plots for different objects (e.g., rotation curves for different galaxy types)

5. **Publication Preparation**: Use these as starting points for creating figures for their own research papers

Each example demonstrates multiple advanced concepts while remaining grounded in real astronomical applications. Students who can successfully modify and extend these examples will have mastered both Matplotlib and scientific visualization principles.