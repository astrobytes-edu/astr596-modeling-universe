# Starter Scripts for Project 1

## `zams.py`

```python

# zams.py
"""
ZAMS mass-luminosity and mass-radius relations from Tout et al. (1996)
Self-contained functions with coefficients from the paper
Valid for masses 0.1 - 100 M_sun and Z = 0.0001 - 0.03
"""
import numpy as np

def luminosity(M, Z=0.02, solar_Z_only=True):
    """
    Calculate ZAMS luminosity using Tout et al. (1996) Eq. 1
    
    L/L_sun = (α M^2.5 + β M^11) / (M^3 + γ + δ M^5 + ε M^7 + ζ M^8 + η M^9.5)
    
    Parameters
    ----------
    M : float or np.ndarray
        Stellar mass in solar masses
        Valid range: 0.1 - 100 M_sun
    Z : float, optional
        Metallicity (default: 0.02 for solar)
        Valid range from Tout et al.: 0.0001 - 0.03
    solar_Z_only : bool, optional
        If True (default), only Z=0.02 is implemented
        Set to False for extension with Z-dependence
    
    Returns
    -------
    L : float or np.ndarray
        Luminosity in solar luminosities
    
    Raises
    ------
    AssertionError
        If mass or metallicity outside valid range
    
    References
    ----------
    Tout et al. (1996) MNRAS 281, 257
    See equations (1) for luminosity formula
    See equations (3)-(4) for metallicity dependence
    """
    # Input validation for mass
    # isinstance(M, np.ndarray) checks if M is a numpy array object
    # This lets our function handle both single values and arrays!
    if isinstance(M, np.ndarray):
        # np.all() returns True only if ALL elements satisfy the condition
        assert np.all(M >= 0.1) and np.all(M <= 100), \
            "Mass must be between 0.1 and 100 M_sun (Tout et al. 1996 validity range)"
    else:
        assert 0.1 <= M <= 100, \
            "Mass must be between 0.1 and 100 M_sun (Tout et al. 1996 validity range)"
    
    # TODO: Assert Z is in Tout et al. valid range (0.0001 to 0.03)
    # TODO: If solar_Z_only is True, also assert Z == 0.02
    
    # Coefficients from Table 1 (check equations 3-4 to understand the table structure)
    coeffs = {
        'alpha': 0.39704170,  # α coefficient
        # TODO: Complete the dictionary with remaining coefficients
        # 'beta': ,
        # 'gamma': ,
        # 'delta': ,
        # 'epsilon': ,
        # 'zeta': ,
        # 'eta': 
    }
    
    # TODO: Implement equation (1) from Tout et al. (1996)
    # IMPORTANT: Note the fractional exponents (M^2.5, M^9.5, etc.)!
    pass

# TODO: Implement radius() function following the same pattern
# Use Equation (2) and coefficients from Table 2
# Check equations (3)-(4) to understand which column corresponds to Z=0.02
# Remember: the equation has fractional exponents just like luminosity!

```

## `star.py`

```python

# star.py
"""
Star class representing individual stars with ZAMS properties
"""
import numpy as np
from zams import luminosity, radius
# TODO: Import necessary constants from your constants module

class Star:
    """
    Represents a single star with ZAMS properties
    """
    
    def __init__(self, mass, name=None):
        """
        Initialize a star with given mass
        
        Parameters
        ----------
        mass : float
            Stellar mass in solar masses (0.1 - 100 M_sun)
        name : str, optional
            Identifier for the star
        """
        # TODO: Validate mass range (0.1 - 100 M_sun)
        # TODO: Set all attributes:
        #   - mass
        #   - name (use provided or create default like f"Star_{mass:.2f}Msun")
        #   - luminosity (use zams.luminosity)
        #   - radius (use zams.radius)
        #   - effective_temperature (call method below)
        #   - t_ms (main sequence lifetime)
        #   - t_kh (Kelvin-Helmholtz timescale)
        #   - lambda_peak (Wien's peak wavelength)
        pass
    
    # TODO: Add methods to calculate:
    
    # 1. Effective temperature
    #    Formula: T/T_sun = (L/L_sun)^0.25 * (R/R_sun)^-0.5
    #    Remember: T_sun = 5777 K (present day, not ZAMS!)
    
    # 2. Main sequence lifetime
    #    Formula: t_MS ∝ M/L
    #    Normalize so 1 M_sun gives ~10 Gyr for present-day Sun
    #    But YOUR 1 M_sun star has L=0.698, so t_MS will be ~14 Gyr
    
    # 3. Kelvin-Helmholtz timescale
    #    Formula: t_KH = GM²/(RL) in CGS units
    #    Convert mass, radius, luminosity to CGS before calculation
    #    Return result in years
    
    # 4. Wien's peak wavelength
    #    Formula: λ_max = b/T where b = 0.2898 cm·K
    #    Return in nanometers (multiply cm by 10^7)
    
    # 5. String representation using f-strings
    #    Use f-strings with units!
    #    Example: f"Mass: {self.mass:.2f} M_sun"
    #    Include all properties with appropriate precision

```

## astro_plot.py

```python

"""
Reusable plotting utilities for astrophysical data
This module will be used throughout the course - make it good!
"""
import matplotlib.pyplot as plt
import numpy as np

def setup_plot(figsize=(10, 8), dpi=100):
    """
    Create figure with consistent style settings
    
    Parameters
    ----------
    figsize : tuple
        Figure size in inches
    dpi : int
        Resolution for display
    
    Returns
    -------
    fig, ax : matplotlib objects
        Figure and single axis for plotting
    
    Examples
    --------
    >>> fig, ax = setup_plot()
    >>> ax.plot(x, y)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # TODO: Set consistent style settings, for example:
    # ax.tick_params(labelsize=12)
    # ax.grid(True, alpha=0.3)
    # Consider setting default font sizes, line widths, etc.
    
    return fig, ax

def plot_hr_diagram(temp, lum, color_by=None, ax=None, **kwargs):
    """
    Create a properly formatted HR diagram
    
    Parameters
    ----------
    temp : array-like
        Effective temperatures in Kelvin
    lum : array-like
        Luminosities in solar units
    color_by : array-like, optional
        Values to color-code points (e.g., mass)
    ax : matplotlib axis, optional
        If provided, plot on this axis. If None, create new figure
    **kwargs : dict
        Additional plotting parameters. Common options:
        - cmap : str, colormap name (default: 'viridis')
        - s : float, marker size (default: 20)
        - alpha : float, transparency (default: 0.7)
        - figsize : tuple, figure dimensions (only used if ax is None)
        
    Returns
    -------
    fig, ax : matplotlib objects
        Figure and axis used for plotting
        
    Notes
    -----
    What happens if you pass in an axis? Plots on that axis.
    What happens if you don't? Creates a new figure with one panel.
    
    Examples
    --------
    >>> # Single plot
    >>> fig, ax = plot_hr_diagram(T, L, color_by=M)
    >>> 
    >>> # Multi-panel
    >>> fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    >>> plot_hr_diagram(T1, L1, ax=axes[0])
    >>> plot_hr_diagram(T2, L2, ax=axes[1])
    """
    # Handle ax parameter - create new figure if None
    if ax is None:
        figsize = kwargs.pop('figsize', (10, 8))  # Remove figsize from kwargs
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Extract common kwargs with defaults
    cmap = kwargs.get('cmap', 'viridis')
    s = kwargs.get('s', 20)
    alpha = kwargs.get('alpha', 0.7)
    
    # TODO: Create the scatter plot
    # TODO: Remember HR diagrams have inverted temperature axis!
    # TODO: Use log scales for both axes
    # TODO: Add labels with units
    # TODO: If color_by is provided, add a colorbar
    
    return fig, ax

# TODO: Add more plotting functions as needed
# Consider: plot_mass_luminosity, plot_performance, plot_distributions
# Each should follow the same pattern:
#   - Accept optional ax parameter
#   - Extract kwargs with sensible defaults
#   - Always return (fig, ax)


```python