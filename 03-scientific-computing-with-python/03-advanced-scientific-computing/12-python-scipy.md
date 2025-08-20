---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# ⚠️ Chapter 12: SciPy - Scientific Python

## Learning Objectives
By the end of this chapter, you will:
- Master interpolation and integration for astronomical data
- Solve ODEs for orbital dynamics and stellar evolution
- Apply optimization algorithms to fitting problems
- Use signal processing for time series and spectra
- Implement statistical tests for data analysis
- Apply special functions in astrophysical calculations
- Leverage sparse matrices for large-scale problems

## Introduction: SciPy's Role in Astronomy

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, special, integrate, optimize, interpolate, signal, stats
from scipy.sparse import csr_matrix, linalg as sparse_linalg
import warnings
warnings.filterwarnings('ignore')

def why_scipy():
    """Demonstrate SciPy's value for astronomical computations."""
    
    print("SciPy provides optimized algorithms for:")
    print("1. Physical constants")
    print(f"   Speed of light: {constants.c:.0f} m/s")
    print(f"   Gravitational constant: {constants.G:.4e} m³/kg/s²")
    print(f"   Stefan-Boltzmann: {constants.Stefan_Boltzmann:.4e} W/m²/K⁴")
    
    print("\n2. Special functions")
    # Bessel functions for diffraction patterns
    x = np.linspace(0, 20, 100)
    airy_pattern = (2 * special.j1(x) / x)**2
    print(f"   Airy disk calculation uses Bessel function J₁")
    
    print("\n3. Optimized numerical algorithms")
    print("   - Integration: adaptive quadrature, ODE solvers")
    print("   - Optimization: least squares, MCMC")
    print("   - Signal processing: filters, FFT, wavelets")
    print("   - Statistics: distributions, tests, correlations")

why_scipy()
```

## Interpolation for Astronomical Data

### 1D and 2D Interpolation

```python
def interpolation_examples():
    """Interpolation methods for astronomical data."""
    
    # 1D Spectral interpolation
    print("1. Spectral Line Interpolation")
    
    # Original spectrum (low resolution)
    wave_obs = np.array([4000, 4500, 5000, 5500, 6000, 6500, 7000])
    flux_obs = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 0.92, 0.88])
    
    # Different interpolation methods
    wave_high = np.linspace(4000, 7000, 1000)
    
    # Linear interpolation
    interp_linear = interpolate.interp1d(wave_obs, flux_obs, kind='linear')
    flux_linear = interp_linear(wave_high)
    
    # Cubic spline
    interp_cubic = interpolate.interp1d(wave_obs, flux_obs, kind='cubic')
    flux_cubic = interp_cubic(wave_high)
    
    # Akima spline (less oscillation)
    from scipy.interpolate import Akima1DInterpolator
    interp_akima = Akima1DInterpolator(wave_obs, flux_obs)
    flux_akima = interp_akima(wave_high)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, flux, title in zip(axes, 
                               [flux_linear, flux_cubic, flux_akima],
                               ['Linear', 'Cubic', 'Akima']):
        ax.plot(wave_high, flux, 'b-', linewidth=1, label='Interpolated')
        ax.plot(wave_obs, flux_obs, 'ro', markersize=6, label='Observed')
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('Flux')
        ax.set_title(f'{title} Interpolation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2D Image interpolation
    print("\n2. PSF Interpolation Across Detector")
    
    # PSF measurements at specific positions
    x_psf = np.array([0, 512, 1024, 0, 512, 1024, 0, 512, 1024])
    y_psf = np.array([0, 0, 0, 512, 512, 512, 1024, 1024, 1024])
    fwhm_psf = np.array([2.1, 2.0, 2.2, 2.0, 1.9, 2.1, 2.2, 2.1, 2.3])
    
    # Interpolate across entire detector
    x_grid = np.arange(0, 1025, 32)
    y_grid = np.arange(0, 1025, 32)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Different 2D interpolation methods
    from scipy.interpolate import griddata
    
    # Linear interpolation
    fwhm_linear = griddata((x_psf, y_psf), fwhm_psf, 
                          (X_grid, Y_grid), method='linear')
    
    # Cubic interpolation
    fwhm_cubic = griddata((x_psf, y_psf), fwhm_psf, 
                         (X_grid, Y_grid), method='cubic')
    
    # Radial basis function
    from scipy.interpolate import Rbf
    rbf = Rbf(x_psf, y_psf, fwhm_psf, function='thin_plate')
    fwhm_rbf = rbf(X_grid, Y_grid)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, data, title in zip(axes, 
                               [fwhm_linear, fwhm_cubic, fwhm_rbf],
                               ['Linear', 'Cubic', 'RBF']):
        im = ax.contourf(X_grid, Y_grid, data, levels=20, cmap='viridis')
        ax.scatter(x_psf, y_psf, c='red', s=50, marker='x')
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title(f'{title} PSF Interpolation')
        plt.colorbar(im, ax=ax, label='FWHM [pixels]')
    
    plt.tight_layout()
    plt.show()
    
    return wave_high, flux_cubic

wavelengths, spectrum = interpolation_examples()
```

### Extrapolation and Edge Effects

```python
def extrapolation_pitfalls():
    """Demonstrate dangers of extrapolation."""
    
    # Observed data (limited range)
    redshift = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    distance = np.array([450, 1400, 2200, 3100, 4000])  # Mpc
    
    # Try to extrapolate
    z_range = np.linspace(0, 2, 100)
    
    # Different extrapolation approaches
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Linear extrapolation (dangerous!)
    f_linear = interpolate.interp1d(redshift, distance, 
                                   kind='linear', fill_value='extrapolate')
    d_linear = f_linear(z_range)
    
    axes[0].plot(z_range, d_linear, 'b-', label='Linear extrap')
    axes[0].plot(redshift, distance, 'ro', markersize=8, label='Data')
    axes[0].axvspan(1.0, 2.0, alpha=0.2, color='red', label='Danger zone')
    axes[0].set_xlabel('Redshift')
    axes[0].set_ylabel('Distance [Mpc]')
    axes[0].set_title('Linear Extrapolation (BAD)')
    axes[0].legend()
    
    # Polynomial fit (also dangerous!)
    poly_coef = np.polyfit(redshift, distance, 3)
    d_poly = np.polyval(poly_coef, z_range)
    
    axes[1].plot(z_range, d_poly, 'g-', label='Polynomial')
    axes[1].plot(redshift, distance, 'ro', markersize=8, label='Data')
    axes[1].axvspan(1.0, 2.0, alpha=0.2, color='red')
    axes[1].set_xlabel('Redshift')
    axes[1].set_title('Polynomial Extrapolation (WORSE)')
    axes[1].legend()
    
    # Physical model (better)
    from scipy.integrate import quad
    
    def luminosity_distance(z, H0=70, Om0=0.3, OL0=0.7):
        """Calculate luminosity distance using cosmology."""
        c = 3e5  # km/s
        
        def integrand(zp):
            return 1 / np.sqrt(Om0*(1+zp)**3 + OL0)
        
        integral, _ = quad(integrand, 0, z)
        return c/H0 * (1+z) * integral
    
    d_model = [luminosity_distance(z) for z in z_range]
    
    axes[2].plot(z_range, d_model, 'k-', label='ΛCDM model')
    axes[2].plot(redshift, distance, 'ro', markersize=8, label='Data')
    axes[2].set_xlabel('Redshift')
    axes[2].set_title('Physical Model (GOOD)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Lesson: Use physical models, not blind extrapolation!")

extrapolation_pitfalls()
```

## Integration Techniques

### Numerical Integration Methods

```python
def integration_showcase():
    """Compare integration methods for astronomical applications."""
    
    # 1. Integrate stellar spectrum to get bolometric luminosity
    print("1. Bolometric Luminosity Calculation")
    
    # Planck function
    def planck(wavelength, T):
        """Planck function in wavelength space."""
        h = constants.h
        c = constants.c
        k = constants.k
        
        wavelength = wavelength * 1e-9  # Convert nm to m
        
        with np.errstate(over='ignore'):
            B = (2*h*c**2/wavelength**5) / (np.exp(h*c/(wavelength*k*T)) - 1)
        
        return B
    
    # Integrate over all wavelengths
    T_star = 5778  # Solar temperature
    
    # Method 1: Fixed quadrature
    from scipy.integrate import quad
    L_quad, error = quad(lambda lam: planck(lam, T_star), 10, 100000)
    
    # Method 2: Adaptive quadrature with specified tolerance
    from scipy.integrate import quad_vec
    wavelengths = np.logspace(1, 5, 1000)  # 10 nm to 100 μm
    integrand = planck(wavelengths, T_star)
    L_trapz = np.trapz(integrand, wavelengths)
    
    # Method 3: Romberg integration
    from scipy.integrate import romberg
    L_romberg = romberg(lambda lam: planck(lam, T_star), 10, 100000)
    
    print(f"  Quadrature: L = {L_quad:.3e} W/m²/sr")
    print(f"  Trapezoid:  L = {L_trapz:.3e} W/m²/sr")
    print(f"  Romberg:    L = {L_romberg:.3e} W/m²/sr")
    print(f"  Stefan-Boltzmann: σT⁴ = {constants.Stefan_Boltzmann * T_star**4:.3e} W/m²")
    
    # 2. Mass enclosed in galaxy profile
    print("\n2. Galaxy Mass Profile")
    
    def density_profile(r, rho0=1e7, rs=10):
        """NFW dark matter density profile."""
        x = r / rs
        return rho0 / (x * (1 + x)**2)
    
    def mass_enclosed(R, rho0=1e7, rs=10):
        """Integrate density to get enclosed mass."""
        
        def integrand(r):
            return 4 * np.pi * r**2 * density_profile(r, rho0, rs)
        
        mass, _ = quad(integrand, 0, R)
        return mass
    
    radii = np.logspace(0, 3, 50)  # 1 to 1000 kpc
    masses = [mass_enclosed(R) for R in radii]
    
    # Analytical solution for NFW
    def mass_nfw_analytical(R, rho0=1e7, rs=10):
        """Analytical NFW enclosed mass."""
        x = R / rs
        return 4 * np.pi * rho0 * rs**3 * (np.log(1+x) - x/(1+x))
    
    masses_analytical = [mass_nfw_analytical(R) for R in radii]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(radii, masses, 'b-', linewidth=2, label='Numerical')
    ax.loglog(radii, masses_analytical, 'r--', linewidth=2, label='Analytical')
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Enclosed Mass [M☉]')
    ax.set_title('NFW Profile Mass Integration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

integration_showcase()
```

### Multi-dimensional Integration

```python
def multidimensional_integration():
    """Multi-dimensional integration for complex geometries."""
    
    # Calculate volume and mass of triaxial galaxy
    print("Triaxial Galaxy Mass Calculation")
    
    def galaxy_density(x, y, z, a=20, b=10, c=5, rho0=1e8):
        """Density in triaxial galaxy."""
        r_ell = np.sqrt((x/a)**2 + (y/b)**2 + (z/c)**2)
        if r_ell > 1:
            return 0
        return rho0 * np.exp(-r_ell)
    
    # Method 1: Triple integral
    from scipy.integrate import tplquad
    
    mass, error = tplquad(
        galaxy_density,
        -5, 5,  # z limits
        lambda z: -10, lambda z: 10,  # y limits
        lambda z, y: -20, lambda z, y: 20  # x limits
    )
    
    print(f"  Total mass (tplquad): {mass:.3e} M☉")
    print(f"  Relative error: {error/mass:.3e}")
    
    # Method 2: Monte Carlo integration
    def monte_carlo_integrate(n_samples=100000):
        """Monte Carlo integration for complex shapes."""
        # Sample in bounding box
        x = np.random.uniform(-20, 20, n_samples)
        y = np.random.uniform(-10, 10, n_samples)
        z = np.random.uniform(-5, 5, n_samples)
        
        # Evaluate density
        densities = np.array([galaxy_density(xi, yi, zi) 
                             for xi, yi, zi in zip(x, y, z)])
        
        # Volume of bounding box
        volume = 40 * 20 * 10
        
        # Monte Carlo estimate
        mass_mc = volume * np.mean(densities)
        error_mc = volume * np.std(densities) / np.sqrt(n_samples)
        
        return mass_mc, error_mc
    
    mass_mc, error_mc = monte_carlo_integrate()
    print(f"  Total mass (Monte Carlo): {mass_mc:.3e} ± {error_mc:.3e} M☉")

multidimensional_integration()
```

## Differential Equations for Dynamics

### Solving ODEs: Orbital Mechanics

```python
def orbital_dynamics():
    """Solve orbital dynamics with various methods."""
    
    def three_body_system(t, state, m1, m2, m3):
        """
        Three-body gravitational system.
        State = [x1, y1, z1, x2, y2, z2, x3, y3, z3,
                vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]
        """
        G = constants.G
        
        # Unpack positions and velocities
        r1 = state[0:3]
        r2 = state[3:6]
        r3 = state[6:9]
        v1 = state[9:12]
        v2 = state[12:15]
        v3 = state[15:18]
        
        # Calculate distances
        r12 = np.linalg.norm(r2 - r1)
        r13 = np.linalg.norm(r3 - r1)
        r23 = np.linalg.norm(r3 - r2)
        
        # Calculate accelerations
        a1 = G*m2*(r2-r1)/r12**3 + G*m3*(r3-r1)/r13**3
        a2 = G*m1*(r1-r2)/r12**3 + G*m3*(r3-r2)/r23**3
        a3 = G*m1*(r1-r3)/r13**3 + G*m2*(r2-r3)/r23**3
        
        # Return derivatives
        return np.concatenate([v1, v2, v3, a1, a2, a3])
    
    # Set up figure-8 solution (approximate)
    x0 = 0.97000436
    y0 = 0.0
    vx0 = -0.93240737
    vy0 = -0.86473146
    
    # Initial conditions (scaled units where G=1, m=1)
    state0 = np.array([
        x0, -y0, 0,     # Body 1 position
        -x0, y0, 0,     # Body 2 position
        0, 0, 0,        # Body 3 position
        vx0/2, vy0/2, 0,  # Body 1 velocity
        vx0/2, vy0/2, 0,  # Body 2 velocity
        -vx0, -vy0, 0     # Body 3 velocity
    ])
    
    # Solve with different methods
    t_span = (0, 6)
    t_eval = np.linspace(0, 6, 1000)
    
    # RK45 (adaptive step)
    from scipy.integrate import solve_ivp
    
    sol_rk45 = solve_ivp(
        three_body_system, t_span, state0, 
        method='RK45', t_eval=t_eval, 
        args=(1, 1, 1), rtol=1e-10
    )
    
    # DOP853 (8th order, good for long integration)
    sol_dop853 = solve_ivp(
        three_body_system, t_span, state0,
        method='DOP853', t_eval=t_eval,
        args=(1, 1, 1), rtol=1e-10
    )
    
    # Plot orbits
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # RK45 solution
    for i, color in enumerate(['red', 'blue', 'green']):
        axes[0].plot(sol_rk45.y[3*i], sol_rk45.y[3*i+1], 
                    color=color, linewidth=1, alpha=0.7,
                    label=f'Body {i+1}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Three-Body Orbits (RK45)')
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Energy conservation check
    def total_energy(state, m1, m2, m3):
        """Calculate total energy of system."""
        G = 1  # Scaled units
        
        r1, r2, r3 = state[0:3], state[3:6], state[6:9]
        v1, v2, v3 = state[9:12], state[12:15], state[15:18]
        
        # Kinetic energy
        KE = 0.5 * (m1*np.sum(v1**2) + m2*np.sum(v2**2) + m3*np.sum(v3**2))
        
        # Potential energy
        r12 = np.linalg.norm(r2 - r1)
        r13 = np.linalg.norm(r3 - r1)
        r23 = np.linalg.norm(r3 - r2)
        PE = -G * (m1*m2/r12 + m1*m3/r13 + m2*m3/r23)
        
        return KE + PE
    
    energies = [total_energy(sol_rk45.y[:, i], 1, 1, 1) 
                for i in range(len(t_eval))]
    
    axes[1].plot(t_eval, energies, 'k-', linewidth=1)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Total Energy')
    axes[1].set_title('Energy Conservation')
    axes[1].grid(True, alpha=0.3)
    
    drift = (energies[-1] - energies[0]) / energies[0]
    axes[1].text(0.5, 0.95, f'Relative drift: {drift:.3e}',
                transform=axes[1].transAxes, ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return sol_rk45

solution = orbital_dynamics()
```

## Optimization and Fitting

### Least Squares Fitting

```python
def optimization_examples():
    """Optimization methods for astronomical data fitting."""
    
    # Generate synthetic supernova light curve
    np.random.seed(42)
    
    def supernova_model(t, t0, A, tau_rise, tau_fall):
        """Simplified supernova light curve model."""
        phase = t - t0
        flux = np.zeros_like(t)
        
        # Rising phase
        mask_rise = (phase >= 0) & (phase < tau_rise)
        flux[mask_rise] = A * (1 - np.exp(-phase[mask_rise]/tau_rise*3))
        
        # Falling phase
        mask_fall = phase >= tau_rise
        flux[mask_fall] = A * np.exp(-(phase[mask_fall]-tau_rise)/tau_fall)
        
        return flux
    
    # True parameters
    true_params = dict(t0=50, A=100, tau_rise=15, tau_fall=30)
    
    # Generate data
    t_obs = np.linspace(0, 150, 50)
    flux_true = supernova_model(t_obs, **true_params)
    flux_err = 5 + 0.05 * flux_true  # Heteroscedastic errors
    flux_obs = flux_true + np.random.normal(0, flux_err)
    
    # 1. Simple least squares
    from scipy.optimize import curve_fit
    
    popt, pcov = curve_fit(
        supernova_model, t_obs, flux_obs,
        p0=[45, 90, 10, 25],  # Initial guess
        sigma=flux_err,
        absolute_sigma=True
    )
    
    perr = np.sqrt(np.diag(pcov))
    
    print("Least Squares Fit:")
    param_names = ['t0', 'A', 'tau_rise', 'tau_fall']
    for name, val, err, true in zip(param_names, popt, perr, true_params.values()):
        print(f"  {name}: {val:.2f} ± {err:.2f} (true: {true})")
    
    # 2. Robust fitting (handles outliers)
    from scipy.optimize import least_squares
    
    def residuals(params, t, flux, flux_err):
        model = supernova_model(t, *params)
        return (flux - model) / flux_err
    
    # Add outliers
    flux_outliers = flux_obs.copy()
    flux_outliers[10] += 50  # Bad pixel
    flux_outliers[30] -= 40  # Cosmic ray
    
    # Huber loss (robust to outliers)
    result_robust = least_squares(
        residuals, [45, 90, 10, 25],
        args=(t_obs, flux_outliers, flux_err),
        loss='huber'
    )
    
    print("\nRobust Fit (with outliers):")
    for name, val in zip(param_names, result_robust.x):
        print(f"  {name}: {val:.2f}")
    
    # 3. Global optimization (for complex χ² surfaces)
    from scipy.optimize import differential_evolution
    
    def chi2(params, t, flux, flux_err):
        model = supernova_model(t, *params)
        return np.sum(((flux - model) / flux_err)**2)
    
    bounds = [(40, 60), (50, 150), (5, 25), (20, 40)]
    
    result_global = differential_evolution(
        chi2, bounds,
        args=(t_obs, flux_obs, flux_err),
        seed=42
    )
    
    print("\nGlobal Optimization:")
    for name, val in zip(param_names, result_global.x):
        print(f"  {name}: {val:.2f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    t_model = np.linspace(0, 150, 500)
    
    # Original fit
    axes[0, 0].errorbar(t_obs, flux_obs, yerr=flux_err, 
                       fmt='ko', markersize=4, alpha=0.5, label='Data')
    axes[0, 0].plot(t_model, supernova_model(t_model, *popt), 
                   'b-', linewidth=2, label='Least Squares')
    axes[0, 0].plot(t_model, supernova_model(t_model, **true_params), 
                   'r--', linewidth=2, label='True')
    axes[0, 0].set_xlabel('Time [days]')
    axes[0, 0].set_ylabel('Flux')
    axes[0, 0].set_title('Standard Least Squares')
    axes[0, 0].legend()
    
    # With outliers
    axes[0, 1].errorbar(t_obs, flux_outliers, yerr=flux_err,
                       fmt='ko', markersize=4, alpha=0.5, label='Data+outliers')
    axes[0, 1].plot(t_model, supernova_model(t_model, *result_robust.x),
                   'g-', linewidth=2, label='Robust fit')
    axes[0, 1].plot(t_model, supernova_model(t_model, *popt),
                   'b--', linewidth=2, alpha=0.5, label='Standard fit')
    axes[0, 1].set_xlabel('Time [days]')
    axes[0, 1].set_ylabel('Flux')
    axes[0, 1].set_title('Robust Fitting')
    axes[0, 1].legend()
    
    # Residuals
    residuals_standard = flux_obs - supernova_model(t_obs, *popt)
    residuals_robust = flux_outliers - supernova_model(t_obs, *result_robust.x)
    
    axes[1, 0].scatter(t_obs, residuals_standard/flux_err, alpha=0.5)
    axes[1, 0].axhline(0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Time [days]')
    axes[1, 0].set_ylabel('Normalized Residuals')
    axes[1, 0].set_title('Residuals (Standard)')
    axes[1, 0].set_ylim(-5, 5)
    
    axes[1, 1].scatter(t_obs, residuals_robust/flux_err, alpha=0.5)
    axes[1, 1].axhline(0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Time [days]')
    axes[1, 1].set_ylabel('Normalized Residuals')
    axes[1, 1].set_title('Residuals (Robust)')
    axes[1, 1].set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.show()
    
    return popt, pcov

params, covariance = optimization_examples()
```

## Signal Processing for Time Series

### Fourier Analysis and Filtering

```python
def signal_processing_astronomy():
    """Signal processing for astronomical time series."""
    
    # Generate synthetic variable star data
    np.random.seed(42)
    
    # Time sampling (uneven, realistic for ground-based)
    n_nights = 200
    t_obs = []
    for night in range(n_nights):
        # Random number of observations per night
        n_obs = np.random.poisson(5)
        if n_obs > 0:
            # Observations within 6-hour window
            t_night = night + np.random.uniform(0, 0.25, n_obs)
            t_obs.extend(t_night)
    
    t_obs = np.array(sorted(t_obs))
    
    # Multi-periodic signal
    P1, P2 = 2.3456, 3.7890  # Periods in days
    A1, A2 = 0.1, 0.05  # Amplitudes
    
    signal = (A1 * np.sin(2*np.pi*t_obs/P1) + 
             A2 * np.sin(2*np.pi*t_obs/P2 + 1.5))
    
    # Add noise and trends
    noise = np.random.normal(0, 0.02, len(t_obs))
    trend = 0.0001 * t_obs  # Linear trend
    flux = 1.0 + signal + noise + trend
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # 1. Raw data
    axes[0, 0].scatter(t_obs, flux, s=1, alpha=0.5)
    axes[0, 0].set_xlabel('Time [days]')
    axes[0, 0].set_ylabel('Flux')
    axes[0, 0].set_title('Raw Light Curve')
    
    # 2. Lomb-Scargle periodogram (for uneven sampling)
    from scipy.signal import lombscargle
    
    frequencies = np.linspace(0.1, 2, 1000)
    periods = 1 / frequencies
    
    # Normalize data
    flux_norm = flux - np.mean(flux)
    
    # Compute periodogram
    power = lombscargle(t_obs, flux_norm, 2*np.pi*frequencies)
    
    # Normalize power
    power = power / power.max()
    
    axes[0, 1].plot(periods, power, 'k-', linewidth=1)
    axes[0, 1].axvline(P1, color='red', linestyle='--', alpha=0.5, label=f'P1={P1:.3f}')
    axes[0, 1].axvline(P2, color='blue', linestyle='--', alpha=0.5, label=f'P2={P2:.3f}')
    axes[0, 1].set_xlabel('Period [days]')
    axes[0, 1].set_ylabel('Power')
    axes[0, 1].set_title('Lomb-Scargle Periodogram')
    axes[0, 1].set_xlim(0, 10)
    axes[0, 1].legend()
    
    # 3. Detrending
    from scipy.signal import detrend
    
    # Polynomial detrending
    poly_coef = np.polyfit(t_obs, flux, 2)
    trend_fit = np.polyval(poly_coef, t_obs)
    flux_detrended = flux - trend_fit + np.mean(flux)
    
    axes[0, 2].scatter(t_obs, flux_detrended, s=1, alpha=0.5)
    axes[0, 2].set_xlabel('Time [days]')
    axes[0, 2].set_ylabel('Flux')
    axes[0, 2].set_title('Detrended Light Curve')
    
    # 4. Filtering
    from scipy.signal import savgol_filter
    
    # Need regular sampling for filtering
    t_regular = np.linspace(t_obs.min(), t_obs.max(), 2000)
    flux_interp = np.interp(t_regular, t_obs, flux_detrended)
    
    # Savitzky-Golay filter
    flux_smooth = savgol_filter(flux_interp, 51, 3)
    
    axes[1, 0].plot(t_regular, flux_interp, 'k-', alpha=0.3, linewidth=0.5, label='Interpolated')
    axes[1, 0].plot(t_regular, flux_smooth, 'r-', linewidth=2, label='Smoothed')
    axes[1, 0].set_xlabel('Time [days]')
    axes[1, 0].set_ylabel('Flux')
    axes[1, 0].set_title('Savitzky-Golay Filter')
    axes[1, 0].set_xlim(50, 60)  # Zoom in
    axes[1, 0].legend()
    
    # 5. Wavelet analysis
    from scipy.signal import cwt, ricker
    
    # Continuous wavelet transform
    widths = np.arange(1, 100)
    cwt_matrix = cwt(flux_interp, ricker, widths)
    
    im = axes[1, 1].imshow(np.abs(cwt_matrix), extent=[t_regular.min(), t_regular.max(), 
                                                        widths[-1], widths[0]], 
                          cmap='viridis', aspect='auto')
    axes[1, 1].set_xlabel('Time [days]')
    axes[1, 1].set_ylabel('Scale')
    axes[1, 1].set_title('Wavelet Transform')
    plt.colorbar(im, ax=axes[1, 1])
    
    # 6. Phase folding
    best_period = P1  # Use known period
    phase = (t_obs % best_period) / best_period
    
    axes[1, 2].scatter(phase, flux_detrended, s=1, alpha=0.3, c='k')
    axes[1, 2].scatter(phase + 1, flux_detrended, s=1, alpha=0.3, c='k')  # Repeat
    
    # Binned curve
    phase_bins = np.linspace(0, 1, 20)
    binned_flux = []
    for i in range(len(phase_bins)-1):
        mask = (phase > phase_bins[i]) & (phase < phase_bins[i+1])
        if mask.sum() > 0:
            binned_flux.append(np.median(flux_detrended[mask]))
        else:
            binned_flux.append(np.nan)
    
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    axes[1, 2].plot(bin_centers, binned_flux, 'ro-', linewidth=2, markersize=5)
    axes[1, 2].set_xlabel('Phase')
    axes[1, 2].set_ylabel('Flux')
    axes[1, 2].set_title(f'Phase-folded (P={best_period:.4f} days)')
    axes[1, 2].set_xlim(0, 2)
    
    plt.tight_layout()
    plt.show()

signal_processing_astronomy()
```

## Statistical Analysis

### Distribution Fitting and Testing

```python
def statistical_analysis():
    """Statistical tests for astronomical data."""
    
    # Generate galaxy cluster data
    np.random.seed(42)
    
    # Two clusters with different properties
    n1, n2 = 150, 100
    
    # Cluster 1: nearby, rich
    velocities1 = np.random.normal(5000, 800, n1)  # km/s
    luminosities1 = np.random.lognormal(10, 0.5, n1)  # Solar luminosities
    
    # Cluster 2: distant, poor
    velocities2 = np.random.normal(15000, 600, n2)
    luminosities2 = np.random.lognormal(9.5, 0.7, n2)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # 1. Histogram and distribution fitting
    from scipy.stats import norm, lognorm
    
    # Velocity distribution
    axes[0, 0].hist(velocities1, bins=20, alpha=0.5, density=True, label='Cluster 1')
    axes[0, 0].hist(velocities2, bins=20, alpha=0.5, density=True, label='Cluster 2')
    
    # Fit normal distributions
    mu1, std1 = norm.fit(velocities1)
    mu2, std2 = norm.fit(velocities2)
    
    v_range = np.linspace(0, 20000, 100)
    axes[0, 0].plot(v_range, norm.pdf(v_range, mu1, std1), 'b-', linewidth=2)
    axes[0, 0].plot(v_range, norm.pdf(v_range, mu2, std2), 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Velocity [km/s]')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title('Velocity Distributions')
    axes[0, 0].legend()
    
    # 2. Q-Q plot for normality test
    from scipy.stats import probplot
    
    probplot(velocities1, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Cluster 1 Velocities)')
    
    # 3. Two-sample tests
    from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = ks_2samp(velocities1, velocities2)
    
    # Mann-Whitney U test (non-parametric)
    mw_stat, mw_pvalue = mannwhitneyu(velocities1, velocities2)
    
    # Student's t-test (parametric)
    t_stat, t_pvalue = ttest_ind(velocities1, velocities2)
    
    axes[0, 2].text(0.1, 0.8, 'Two-Sample Tests:', fontsize=12, fontweight='bold',
                   transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.6, f'K-S test: p = {ks_pvalue:.3e}',
                   transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.5, f'Mann-Whitney: p = {mw_pvalue:.3e}',
                   transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.4, f't-test: p = {t_pvalue:.3e}',
                   transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.2, 'p < 0.05 suggests different distributions',
                   transform=axes[0, 2].transAxes, fontsize=10, style='italic')
    axes[0, 2].axis('off')
    
    # 4. Correlation analysis
    from scipy.stats import spearmanr, pearsonr
    
    # Add some correlation
    masses1 = luminosities1 + np.random.normal(0, 1, n1)
    masses2 = luminosities2 + np.random.normal(0, 1, n2)
    
    # Combine data
    all_lum = np.concatenate([luminosities1, luminosities2])
    all_mass = np.concatenate([masses1, masses2])
    
    axes[1, 0].scatter(np.log10(luminosities1), np.log10(masses1), 
                      alpha=0.5, label='Cluster 1')
    axes[1, 0].scatter(np.log10(luminosities2), np.log10(masses2), 
                      alpha=0.5, label='Cluster 2')
    axes[1, 0].set_xlabel('log(Luminosity) [L☉]')
    axes[1, 0].set_ylabel('log(Mass) [M☉]')
    axes[1, 0].set_title('Mass-Luminosity Relation')
    axes[1, 0].legend()
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(np.log10(all_lum), np.log10(all_mass))
    spearman_r, spearman_p = spearmanr(all_lum, all_mass)
    
    axes[1, 0].text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}',
                   transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.05, 0.90, f'Spearman ρ = {spearman_r:.3f}',
                   transform=axes[1, 0].transAxes)
    
    # 5. Bootstrap confidence intervals
    from scipy.stats import bootstrap
    
    def median_diff(x, y):
        """Difference in medians."""
        return np.median(x) - np.median(y)
    
    # Bootstrap
    rng = np.random.default_rng(42)
    res = bootstrap((velocities1, velocities2), 
                   lambda x, y: np.median(x) - np.median(y),
                   n_resamples=10000,
                   confidence_level=0.95,
                   random_state=rng,
                   method='percentile')
    
    axes[1, 1].hist(res.bootstrap_distribution, bins=50, alpha=0.7)
    axes[1, 1].axvline(res.confidence_interval.low, color='red', 
                      linestyle='--', label='95% CI')
    axes[1, 1].axvline(res.confidence_interval.high, color='red', 
                      linestyle='--')
    axes[1, 1].set_xlabel('Median Velocity Difference [km/s]')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Bootstrap Distribution')
    axes[1, 1].legend()
    
    # 6. Survival analysis (for truncated data)
    from scipy.stats import kaplan_meier_estimator
    
    # Simulate detection limits
    detection_limit = 11.0
    detected1 = luminosities1 > detection_limit
    detected2 = luminosities2 > detection_limit
    
    # This is simplified - real survival analysis needs more care
    axes[1, 2].hist(luminosities1[detected1], bins=20, alpha=0.5, 
                   label=f'Cluster 1 ({detected1.sum()}/{n1} detected)')
    axes[1, 2].hist(luminosities2[detected2], bins=20, alpha=0.5,
                   label=f'Cluster 2 ({detected2.sum()}/{n2} detected)')
    axes[1, 2].axvline(detection_limit, color='red', linestyle='--', 
                      label='Detection limit')
    axes[1, 2].set_xlabel('Luminosity [L☉]')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Truncated Data')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()

statistical_analysis()
```

## Try It Yourself

### Exercise 1: Build a Complete Spectral Analysis Pipeline

```python
def spectral_analysis_pipeline(wavelength, flux, error):
    """
    Complete spectral analysis pipeline.
    
    Tasks:
    1. Interpolate to common wavelength grid
    2. Identify and mask cosmic rays
    3. Fit and subtract continuum
    4. Find emission/absorption lines
    5. Measure line properties (EW, FWHM, flux)
    6. Estimate radial velocity
    7. Calculate S/N ratio
    """
    # Your code here
    pass
```

### Exercise 2: Orbital Fitting with MCMC

```python
def fit_exoplanet_orbit(times, radial_velocities, errors):
    """
    Fit Keplerian orbit to radial velocity data.
    
    Parameters to fit:
    - Period
    - Eccentricity
    - Semi-amplitude
    - Argument of periastron
    - Time of periastron
    - Systemic velocity
    
    Use MCMC for proper uncertainty estimation.
    """
    # Your code here
    pass
```

### Exercise 3: Image Deconvolution

```python
def deconvolve_image(image, psf, method='richardson-lucy'):
    """
    Deconvolve astronomical image.
    
    Methods:
    - Richardson-Lucy
    - Wiener filter
    - Blind deconvolution
    
    Handle noise and artifacts properly.
    """
    # Your code here
    pass
```

## Key Takeaways

✅ **SciPy provides optimized algorithms** - Don't reinvent the wheel  
✅ **Choose interpolation carefully** - Cubic can oscillate, consider Akima  
✅ **Never blindly extrapolate** - Use physical models instead  
✅ **Integration has many methods** - Adaptive quadrature, Monte Carlo for high-D  
✅ **ODEs need appropriate solvers** - RK45 for general, DOP853 for long integration  
✅ **Optimization requires good initial guesses** - Consider global methods  
✅ **Signal processing handles real data** - Uneven sampling, noise, trends  
✅ **Statistical tests have assumptions** - Check them before applying  

## Next Steps

You now have the foundation for scientific computing in Python:
- **NumPy** for array operations and linear algebra
- **Matplotlib** for publication-quality visualizations
- **SciPy** for numerical algorithms and analysis

Combined with your Python fundamentals and optimization techniques, you're ready to tackle complex astronomical problems. The next section on Pandas will add powerful data manipulation capabilities for working with catalogs and time series data.

Remember: These libraries work best together. Use NumPy for computation, SciPy for algorithms, and Matplotlib for visualization - all integrated in your astronomical workflows.