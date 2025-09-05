"""
ASTR 596: Modeling the Universe - Plotting Utilities Module
Provides consistent styling, colors, and utilities for all course visualizations
Author: Anna Rosen
Date: January 2025

Usage:
    from astr596_plotting_utils import ASTR596Plotter
    
    plotter = ASTR596Plotter()
    fig, ax = plotter.create_figure(figsize=(10, 6))
    plotter.apply_style(ax)
    colors = plotter.get_color_sequence(5)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import scipy.stats as stats
from typing import Tuple, List, Optional, Union, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ASTR596Plotter:
    """
    Master plotting class for ASTR 596 course visualizations.
    Ensures consistency across all modules.
    """
    
    def __init__(self, style: str = 'default', save_dir: str = './figures/'):
        """
        Initialize the plotter with consistent settings.
        
        Parameters:
        -----------
        style : str
            Plotting style ('default', 'presentation', 'print')
        save_dir : str
            Directory for saving figures
        """
        self.save_dir = save_dir
        self.style = style
        
        # Create directory if it doesn't exist
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        
        # Physical constants (CGS units)
        self.constants = {
            'k_B': 1.38e-16,     # erg/K (Boltzmann constant)
            'm_H': 1.67e-24,     # g (hydrogen mass)
            'm_e': 9.11e-28,     # g (electron mass)
            'c': 2.998e10,       # cm/s (speed of light)
            'G': 6.674e-8,       # cm³/g/s² (gravitational constant)
            'M_sun': 1.989e33,   # g (solar mass)
            'R_sun': 6.96e10,    # cm (solar radius)
            'L_sun': 3.828e33,   # erg/s (solar luminosity)
            'pc': 3.086e18,      # cm (parsec)
            'AU': 1.496e13,      # cm (astronomical unit)
            'yr': 3.156e7,       # s (year)
        }
        
        # Color palettes
        self._define_color_palettes()
        
        # Apply style settings
        self._apply_style_settings()
    
    def _define_color_palettes(self):
        """Define comprehensive color palettes for different uses."""
        
        # Main scientific palette (Nord-inspired + astrophysics)
        self.colors_main = [
            '#2E3440',  # 0: Dark space gray
            '#5E81AC',  # 1: Deep space blue
            '#81A1C1',  # 2: Stellar blue
            '#88C0D0',  # 3: Nebula cyan
            '#8FBCBB',  # 4: Quantum teal
            '#A3BE8C',  # 5: Aurora green
            '#EBCB8B',  # 6: Solar yellow
            '#D08770',  # 7: Plasma orange
            '#BF616A',  # 8: Stellar red
            '#B48EAD',  # 9: Cosmic purple
        ]
        
        # Temperature palette (cold to hot)
        self.colors_temperature = [
            '#5E81AC',  # Cold blue
            '#88C0D0',  # Cool cyan
            '#8FBCBB',  # Neutral teal
            '#A3BE8C',  # Warm green
            '#EBCB8B',  # Hot yellow
            '#D08770',  # Very hot orange
            '#BF616A',  # Extreme red
        ]
        
        # Diverging palette (for positive/negative)
        self.colors_diverging = [
            '#5E81AC',  # Negative strong
            '#81A1C1',  # Negative medium
            '#88C0D0',  # Negative weak
            '#ECEFF4',  # Neutral
            '#EBCB8B',  # Positive weak
            '#D08770',  # Positive medium
            '#BF616A',  # Positive strong
        ]
        
        # Sequential palette (for magnitude)
        self.colors_sequential = [
            '#ECEFF4',  # Lightest
            '#D8DEE9',
            '#88C0D0',
            '#5E81AC',
            '#3B4252',  # Darkest
        ]
        
        # Categorical palette (for discrete categories)
        self.colors_categorical = [
            '#5E81AC',  # Blue
            '#A3BE8C',  # Green
            '#D08770',  # Orange
            '#B48EAD',  # Purple
            '#BF616A',  # Red
            '#EBCB8B',  # Yellow
            '#8FBCBB',  # Teal
            '#81A1C1',  # Light blue
        ]
        
        # Special colors
        self.color_accept = '#A3BE8C'  # Green for acceptance/success
        self.color_reject = '#BF616A'  # Red for rejection/failure
        self.color_neutral = '#4C566A' # Gray for neutral
        self.color_highlight = '#EBCB8B' # Yellow for highlighting
        
        # Background colors
        self.color_background = '#FAFAFA'
        self.color_grid = '#E5E9F0'
    
    def _apply_style_settings(self):
        """Apply consistent matplotlib style settings."""
        
        base_settings = {
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'figure.facecolor': 'white',
            'axes.facecolor': self.color_background,
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': self.color_grid,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.edgecolor': self.color_neutral,
            'axes.linewidth': 1.0,
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            'patch.linewidth': 1.0,
            'text.usetex': False,  # Set to True if LaTeX is available
        }
        
        # Style-specific modifications
        if self.style == 'presentation':
            base_settings.update({
                'font.size': 14,
                'axes.labelsize': 16,
                'axes.titlesize': 18,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'legend.fontsize': 14,
                'figure.titlesize': 20,
                'lines.linewidth': 3.0,
                'lines.markersize': 8,
            })
        elif self.style == 'print':
            base_settings.update({
                'savefig.dpi': 300,
                'figure.dpi': 150,
            })
        
        # Apply settings
        for key, value in base_settings.items():
            plt.rcParams[key] = value
    
    def create_figure(self, 
                     figsize: Tuple[float, float] = (10, 6),
                     nrows: int = 1,
                     ncols: int = 1,
                     **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create a figure with consistent styling.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size in inches
        nrows, ncols : int
            Number of subplot rows and columns
        **kwargs : dict
            Additional arguments for plt.subplots()
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        return fig, ax
    
    def create_figure_with_gridspec(self, 
                                   figsize: Tuple[float, float] = (12, 8),
                                   nrows: int = 2,
                                   ncols: int = 2,
                                   **kwargs) -> Tuple[plt.Figure, gridspec.GridSpec]:
        """
        Create a figure with GridSpec for complex layouts.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size in inches
        nrows, ncols : int
            Number of grid rows and columns
        **kwargs : dict
            Additional arguments for GridSpec
        
        Returns:
        --------
        fig, gs : matplotlib figure and GridSpec object
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig, **kwargs)
        return fig, gs
    
    def get_color_sequence(self, n: int, palette: str = 'main') -> List[str]:
        """
        Get a sequence of colors from a palette.
        
        Parameters:
        -----------
        n : int
            Number of colors needed
        palette : str
            Which palette to use ('main', 'temperature', 'categorical', etc.)
        
        Returns:
        --------
        colors : list of str
            List of hex color codes
        """
        palettes = {
            'main': self.colors_main,
            'temperature': self.colors_temperature,
            'diverging': self.colors_diverging,
            'sequential': self.colors_sequential,
            'categorical': self.colors_categorical,
        }
        
        selected_palette = palettes.get(palette, self.colors_main)
        
        if n <= len(selected_palette):
            return selected_palette[:n]
        else:
            # Cycle through palette if more colors needed
            return [selected_palette[i % len(selected_palette)] for i in range(n)]
    
    def create_colormap(self, colors: List[str], name: str = 'custom') -> LinearSegmentedColormap:
        """
        Create a custom colormap from a list of colors.
        
        Parameters:
        -----------
        colors : list of str
            List of hex color codes
        name : str
            Name for the colormap
        
        Returns:
        --------
        cmap : LinearSegmentedColormap
            Custom colormap
        """
        return LinearSegmentedColormap.from_list(name, colors)
    
    def apply_style(self, ax: plt.Axes, 
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   title: Optional[str] = None,
                   legend: bool = False,
                   grid: bool = True):
        """
        Apply consistent styling to an axes object.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to style
        xlabel, ylabel, title : str
            Labels and title
        legend : bool
            Whether to add legend
        grid : bool
            Whether to show grid
        """
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontweight='bold')
        if legend:
            ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.95)
        if grid:
            ax.grid(True, alpha=0.3)
    
    def save_figure(self, fig: plt.Figure, 
                   filename: str,
                   formats: List[str] = ['png', 'svg'],
                   dpi: Optional[int] = None,
                   transparent: bool = False):
        """
        Save figure in multiple formats with consistent settings.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str
            Base filename (without extension)
        formats : list of str
            File formats to save
        dpi : int
            DPI for raster formats (uses style default if None)
        transparent : bool
            Whether to save with transparent background
        """
        for fmt in formats:
            filepath = os.path.join(self.save_dir, f"{filename}.{fmt}")
            save_dpi = dpi if dpi else plt.rcParams['savefig.dpi']
            
            fig.savefig(filepath,
                       format=fmt,
                       dpi=save_dpi if fmt in ['png', 'jpg'] else None,
                       bbox_inches='tight',
                       facecolor='white' if not transparent else 'none',
                       edgecolor='none',
                       transparent=transparent)
            print(f"  ✓ Saved {fmt.upper()}: {filepath}")
    
    def add_text_box(self, ax: plt.Axes,
                    text: str,
                    location: str = 'upper right',
                    style: str = 'info') -> None:
        """
        Add a styled text box to axes.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add text box to
        text : str
            Text content
        location : str
            Location of text box
        style : str
            Style of box ('info', 'warning', 'success', 'equation')
        """
        styles = {
            'info': {'boxstyle': 'round,pad=0.5', 'facecolor': '#D8DEE9', 'alpha': 0.8},
            'warning': {'boxstyle': 'round,pad=0.5', 'facecolor': '#EBCB8B', 'alpha': 0.8},
            'success': {'boxstyle': 'round,pad=0.5', 'facecolor': '#A3BE8C', 'alpha': 0.8},
            'equation': {'boxstyle': 'square,pad=0.3', 'facecolor': 'white', 'edgecolor': 'black'},
        }
        
        props = styles.get(style, styles['info'])
        
        # Convert location string to coordinates
        loc_dict = {
            'upper right': (0.95, 0.95),
            'upper left': (0.05, 0.95),
            'lower right': (0.95, 0.05),
            'lower left': (0.05, 0.05),
            'center': (0.5, 0.5),
        }
        
        x, y = loc_dict.get(location, (0.95, 0.95))
        ha = 'right' if 'right' in location else ('left' if 'left' in location else 'center')
        va = 'top' if 'upper' in location else ('bottom' if 'lower' in location else 'center')
        
        ax.text(x, y, text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment=va,
               horizontalalignment=ha,
               bbox=props)
    
    def plot_with_errors(self, ax: plt.Axes,
                        x: np.ndarray,
                        y: np.ndarray,
                        yerr: Optional[np.ndarray] = None,
                        xerr: Optional[np.ndarray] = None,
                        label: Optional[str] = None,
                        color: Optional[str] = None,
                        marker: str = 'o',
                        linestyle: str = '-',
                        alpha: float = 0.8,
                        fill_between: bool = False):
        """
        Plot data with error bars or bands.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        x, y : array-like
            Data to plot
        yerr, xerr : array-like
            Error values
        label : str
            Label for legend
        color : str
            Color for plot
        marker : str
            Marker style
        linestyle : str
            Line style
        alpha : float
            Transparency
        fill_between : bool
            Whether to use fill_between for errors instead of error bars
        """
        if color is None:
            color = self.colors_main[1]
        
        # Main line
        line = ax.plot(x, y, color=color, marker=marker, linestyle=linestyle,
                      label=label, alpha=alpha, markersize=6, linewidth=2)[0]
        
        # Error representation
        if yerr is not None:
            if fill_between:
                ax.fill_between(x, y - yerr, y + yerr, 
                              color=color, alpha=0.2)
            else:
                ax.errorbar(x, y, yerr=yerr, xerr=xerr,
                          color=color, alpha=alpha,
                          fmt='none', capsize=3, capthick=1)
        elif xerr is not None:
            ax.errorbar(x, y, xerr=xerr, 
                       color=color, alpha=alpha,
                       fmt='none', capsize=3, capthick=1)
        
        return line
    
    def add_scalebar(self, ax: plt.Axes,
                    length: float,
                    label: str,
                    location: str = 'lower right',
                    color: str = 'black'):
        """
        Add a scale bar to the plot.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add scalebar to
        length : float
            Length of scalebar in data units
        label : str
            Label for the scalebar
        location : str
            Location of scalebar
        color : str
            Color of scalebar
        """
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Calculate position
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        positions = {
            'lower right': (xlim[1] - 0.15*x_range, ylim[0] + 0.1*y_range),
            'lower left': (xlim[0] + 0.05*x_range, ylim[0] + 0.1*y_range),
            'upper right': (xlim[1] - 0.15*x_range, ylim[1] - 0.1*y_range),
            'upper left': (xlim[0] + 0.05*x_range, ylim[1] - 0.1*y_range),
        }
        
        x_pos, y_pos = positions.get(location, positions['lower right'])
        
        # Draw scalebar
        ax.plot([x_pos, x_pos + length], [y_pos, y_pos],
               color=color, linewidth=3)
        
        # Add label
        ax.text(x_pos + length/2, y_pos - 0.03*y_range, label,
               ha='center', va='top', color=color, fontsize=10)
    
    def create_inset_axis(self, ax: plt.Axes,
                         bounds: List[float] = [0.6, 0.6, 0.35, 0.35]) -> plt.Axes:
        """
        Create an inset axis within the main axis.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Parent axes
        bounds : list of float
            [x, y, width, height] in axes fraction
        
        Returns:
        --------
        ax_inset : matplotlib.axes.Axes
            Inset axes
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        ax_inset = inset_axes(ax, width="100%", height="100%",
                             bbox_to_anchor=bounds,
                             bbox_transform=ax.transAxes,
                             loc='center')
        return ax_inset
    
    def format_axis_scientific(self, ax: plt.Axes, 
                              axis: str = 'y',
                              decimals: int = 1):
        """
        Format axis labels in scientific notation.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to format
        axis : str
            Which axis to format ('x', 'y', or 'both')
        decimals : int
            Number of decimal places
        """
        from matplotlib.ticker import ScalarFormatter
        
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        
        if axis in ['x', 'both']:
            ax.xaxis.set_major_formatter(formatter)
        if axis in ['y', 'both']:
            ax.yaxis.set_major_formatter(formatter)
    
    def add_arrow_annotation(self, ax: plt.Axes,
                           text: str,
                           xy: Tuple[float, float],
                           xytext: Tuple[float, float],
                           color: Optional[str] = None,
                           arrowstyle: str = '->'):
        """
        Add an annotated arrow to highlight features.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to annotate
        text : str
            Annotation text
        xy : tuple
            Point to annotate (data coordinates)
        xytext : tuple
            Text position (data coordinates)
        color : str
            Arrow and text color
        arrowstyle : str
            Arrow style string
        """
        if color is None:
            color = self.color_neutral
        
        ax.annotate(text, xy=xy, xytext=xytext,
                   arrowprops=dict(arrowstyle=arrowstyle, color=color, lw=1.5),
                   fontsize=10, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, alpha=0.8))
    
    def add_shaded_region(self, ax: plt.Axes,
                         x_start: float,
                         x_end: float,
                         color: Optional[str] = None,
                         alpha: float = 0.2,
                         label: Optional[str] = None):
        """
        Add a shaded vertical region to highlight areas.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to add shading to
        x_start, x_end : float
            Boundaries of shaded region
        color : str
            Color of shading
        alpha : float
            Transparency
        label : str
            Label for legend
        """
        if color is None:
            color = self.color_highlight
        
        ax.axvspan(x_start, x_end, alpha=alpha, color=color, label=label)
    
    @staticmethod
    def generate_caption(figure_number: int,
                        title: str,
                        description: str,
                        details: Optional[str] = None) -> Dict[str, str]:
        """
        Generate consistent figure captions.
        
        Parameters:
        -----------
        figure_number : int
            Figure number
        title : str
            Short title
        description : str
            Main description
        details : str
            Additional technical details
        
        Returns:
        --------
        dict : Dictionary with 'full' and 'myst' caption formats
        """
        full_caption = f"Figure {figure_number}: {title}. {description}"
        if details:
            full_caption += f" {details}"
        
        myst_caption = f"""```{{figure}} ./figures/fig{figure_number}_{title.lower().replace(' ', '_')}.svg
:name: fig-{title.lower().replace(' ', '-')}
:width: 100%
:alt: {title}

{description}{' ' + details if details else ''}
```"""
        
        return {
            'full': full_caption,
            'myst': myst_caption
        }
    
    @staticmethod
    def validate_data(data: np.ndarray, 
                     name: str = 'data',
                     allow_negative: bool = True,
                     allow_zero: bool = True,
                     allow_inf: bool = False) -> np.ndarray:
        """
        Validate data array for common issues.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to validate
        name : str
            Name for error messages
        allow_negative : bool
            Whether negative values are allowed
        allow_zero : bool
            Whether zero values are allowed
        allow_inf : bool
            Whether infinite values are allowed
        
        Returns:
        --------
        data : np.ndarray
            Validated data
        
        Raises:
        -------
        ValueError : If data fails validation
        """
        data = np.asarray(data)
        
        # Check for NaN
        if np.any(np.isnan(data)):
            raise ValueError(f"{name} contains NaN values")
        
        # Check for Inf
        if not allow_inf and np.any(np.isinf(data)):
            raise ValueError(f"{name} contains infinite values")
        
        # Check for negative
        if not allow_negative and np.any(data < 0):
            raise ValueError(f"{name} contains negative values")
        
        # Check for zero
        if not allow_zero and np.any(data == 0):
            raise ValueError(f"{name} contains zero values")
        
        return data
    
    def plot_comparison(self, ax: plt.Axes,
                       data_sets: List[Dict[str, Any]],
                       plot_type: str = 'line'):
        """
        Plot multiple datasets for comparison.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        data_sets : list of dict
            Each dict should have 'x', 'y', 'label', and optionally 'color', 'style'
        plot_type : str
            Type of plot ('line', 'scatter', 'bar')
        """
        colors = self.get_color_sequence(len(data_sets), 'categorical')
        
        for i, data in enumerate(data_sets):
            x = data['x']
            y = data['y']
            label = data.get('label', f'Dataset {i+1}')
            color = data.get('color', colors[i])
            
            if plot_type == 'line':
                style = data.get('style', '-')
                ax.plot(x, y, label=label, color=color, linestyle=style,
                       linewidth=2, alpha=0.8)
            elif plot_type == 'scatter':
                ax.scatter(x, y, label=label, color=color, alpha=0.6, s=30)
            elif plot_type == 'bar':
                width = data.get('width', 0.8 / len(data_sets))
                offset = (i - len(data_sets)/2 + 0.5) * width
                ax.bar(x + offset, y, width, label=label, color=color, alpha=0.8)


# Specialized plot functions for common astrophysics visualizations

def plot_spectrum(plotter: ASTR596Plotter,
                 wavelength: np.ndarray,
                 flux: np.ndarray,
                 flux_err: Optional[np.ndarray] = None,
                 title: str = "Spectrum",
                 xlabel: str = r"Wavelength (Å)",
                 ylabel: str = r"Flux (erg/s/cm$^2$/Å)",
                 **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a standardized spectrum plot.
    """
    fig, ax = plotter.create_figure(**kwargs)
    
    plotter.plot_with_errors(ax, wavelength, flux, yerr=flux_err,
                            color=plotter.colors_main[1],
                            marker='', linestyle='-',
                            fill_between=True)
    
    plotter.apply_style(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_xlim(wavelength.min(), wavelength.max())
    
    return fig, ax


def plot_hr_diagram(plotter: ASTR596Plotter,
                   temp: np.ndarray,
                   lum: np.ndarray,
                   masses: Optional[np.ndarray] = None,
                   title: str = "Hertzsprung-Russell Diagram",
                   **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a standardized H-R diagram.
    """
    fig, ax = plotter.create_figure(**kwargs)
    
    if masses is not None:
        # Color by mass
        scatter = ax.scatter(np.log10(temp), np.log10(lum),
                           c=masses, cmap=plotter.create_colormap(plotter.colors_temperature),
                           s=20, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label=r'Mass ($M_\odot$)')
    else:
        ax.scatter(np.log10(temp), np.log10(lum),
                  color=plotter.colors_main[1], s=20, alpha=0.6)
    
    ax.invert_xaxis()  # Temperature decreases to the right
    plotter.apply_style(ax,
                       xlabel=r'$\log_{10}(T_{\rm eff}/{\rm K})$',
                       ylabel=r'$\log_{10}(L/L_\odot)$',
                       title=title)
    
    return fig, ax


def plot_power_spectrum(plotter: ASTR596Plotter,
                       k: np.ndarray,
                       power: np.ndarray,
                       theory: Optional[np.ndarray] = None,
                       title: str = "Power Spectrum",
                       **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a standardized power spectrum plot.
    """
    fig, ax = plotter.create_figure(**kwargs)
    
    ax.loglog(k, power, 'o', color=plotter.colors_main[1],
             markersize=4, alpha=0.6, label='Data')
    
    if theory is not None:
        ax.loglog(k, theory, '-', color=plotter.colors_main[8],
                 linewidth=2, label='Theory')
    
    plotter.apply_style(ax,
                       xlabel=r'$k$ (h/Mpc)',
                       ylabel=r'$P(k)$ (Mpc/h)$^3$',
                       title=title,
                       legend=True if theory is not None else False)
    
    return fig, ax


# Example usage function
def example_usage():
    """Demonstrate the usage of the plotting utilities."""
    
    # Initialize plotter
    plotter = ASTR596Plotter(style='default', save_dir='./figures/')
    
    # Example 1: Simple plot
    fig, ax = plotter.create_figure(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    colors = plotter.get_color_sequence(3, 'categorical')
    
    for i, color in enumerate(colors):
        y = np.sin(x + i*np.pi/3) * np.exp(-x/10)
        plotter.plot_with_errors(ax, x, y,
                                yerr=0.1*np.abs(np.random.randn(len(x))),
                                label=f'Dataset {i+1}',
                                color=color,
                                fill_between=True)
    
    plotter.apply_style(ax,
                       xlabel='Time (s)',
                       ylabel='Amplitude',
                       title='Example Time Series',
                       legend=True)
    
    plotter.save_figure(fig, 'example_timeseries')
    
    # Example 2: Grid layout
    fig, gs = plotter.create_figure_with_gridspec(figsize=(12, 8),
                                                  nrows=2, ncols=2,
                                                  hspace=0.3, wspace=0.3)
    
    for i in range(4):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        x = np.random.randn(100)
        y = np.random.randn(100)
        ax.scatter(x, y, color=plotter.colors_categorical[i], alpha=0.5)
        plotter.apply_style(ax, 
                          xlabel='X', 
                          ylabel='Y',
                          title=f'Panel {i+1}')
    
    plt.suptitle('Multi-Panel Figure', fontsize=16, fontweight='bold')
    plotter.save_figure(fig, 'example_multipanel')
    
    print("Example figures created successfully!")


if __name__ == "__main__":
    example_usage()
