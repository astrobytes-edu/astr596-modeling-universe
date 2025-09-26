#!/usr/bin/env python3
"""
Enhanced Electromagnetic Spectrum Physics Ladder Figure
Advanced pedagogical visualization with customization options
for astronomy education.

Usage: python enhanced_em_spectrum.py [options]
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import argparse
import json

class EMSpectrumVisualizer:
    def __init__(self, style='graduate', size='large'):
        """
        Initialize the EM Spectrum visualizer
        
        Parameters:
        style: 'graduate', 'undergraduate', 'public'
        size: 'large', 'medium', 'small', 'presentation'
        """
        self.style = style
        self.size = size
        self.setup_figure_params()
        
    def setup_figure_params(self):
        """Set up figure parameters based on style and size"""
        size_params = {
            'large': {'figsize': (16, 10), 'dpi': 300, 'fontsize_base': 12},
            'medium': {'figsize': (12, 8), 'dpi': 200, 'fontsize_base': 10},
            'small': {'figsize': (8, 6), 'dpi': 150, 'fontsize_base': 8},
            'presentation': {'figsize': (20, 12), 'dpi': 200, 'fontsize_base': 14}
        }
        
        self.fig_params = size_params[self.size]
        
        # Style-specific content depth
        if self.style == 'graduate':
            self.detail_level = 'high'
            self.show_equations = True
            self.show_telescopes = True
        elif self.style == 'undergraduate':
            self.detail_level = 'medium'
            self.show_equations = True
            self.show_telescopes = False
        else:  # public
            self.detail_level = 'low'
            self.show_equations = False
            self.show_telescopes = False

    def get_spectrum_data(self):
        """Return spectrum band data with appropriate detail level"""
        base_data = {
            'Radio': {
                'energy_range': (1e-9, 1e-5),
                'wavelength_range': (1e-1, 1e3),
                'color': '#FF4444',
                'objects': {
                    'high': ['Galaxy spiral arms (synchrotron)', '21-cm hydrogen line', 'Pulsar wind nebulae', 'Molecular clouds'],
                    'medium': ['Galaxy spiral arms', '21-cm hydrogen', 'Pulsars'],
                    'low': ['Galaxies', 'Cold gas']
                },
                'physics': {
                    'high': 'Synchrotron emission, magnetic fields, cold gas',
                    'medium': 'Cold gas, magnetic fields',
                    'low': 'Cold space objects'
                },
                'temp_range': '10-100 K',
                'telescopes': ['VLA', 'Arecibo', 'ALMA', 'LOFAR']
            },
            'Infrared': {
                'energy_range': (1e-5, 1),
                'wavelength_range': (1e-6, 1e-1),
                'color': '#FF8800',
                'objects': {
                    'high': ['Star formation regions', 'Protoplanetary disks', 'Cool stars (M, L, T dwarfs)', 'Dust emission'],
                    'medium': ['Star forming regions', 'Cool stars', 'Dust clouds'],
                    'low': ['New stars forming', 'Cool objects']
                },
                'physics': {
                    'high': 'Thermal emission, warm dust, stellar nurseries',
                    'medium': 'Warm dust, stellar nurseries',
                    'low': 'Warm cosmic dust'
                },
                'temp_range': '100-1000 K',
                'telescopes': ['JWST', 'Spitzer', 'Herschel', 'WISE']
            },
            'Visible': {
                'energy_range': (1, 3),
                'wavelength_range': (4e-7, 7e-7),
                'color': '#FFFF44',
                'objects': {
                    'high': ['Stellar photospheres', 'H II regions (656nm)', 'Planetary nebulae', 'Galaxy disks'],
                    'medium': ['Stars', 'Colorful nebulae', 'Galaxies'],
                    'low': ['Stars', 'Colorful space clouds']
                },
                'physics': {
                    'high': 'Atomic transitions, stellar atmospheres',
                    'medium': 'Atomic transitions, starlight',
                    'low': 'Light we can see'
                },
                'temp_range': '3000-10,000 K',
                'telescopes': ['HST', 'Ground telescopes', 'Kepler', 'TESS']
            },
            'Ultraviolet': {
                'energy_range': (3, 100),
                'wavelength_range': (1e-8, 4e-7),
                'color': '#8888FF',
                'objects': {
                    'high': ['Hot O,B stars', 'Starburst galaxies', 'AGN accretion disks', 'Superbubbles'],
                    'medium': ['Hot young stars', 'Star forming galaxies'],
                    'low': ['Very hot stars']
                },
                'physics': {
                    'high': 'High-energy transitions, ionization',
                    'medium': 'High-energy atomic processes',
                    'low': 'High-energy starlight'
                },
                'temp_range': '10,000+ K',
                'telescopes': ['GALEX', 'Swift', 'FUSE']
            },
            'X-ray': {
                'energy_range': (100, 1e5),
                'wavelength_range': (1e-11, 1e-8),
                'color': '#FF88FF',
                'objects': {
                    'high': ['Black hole accretion disks', 'SNR shock fronts', 'Cluster intracluster medium', 'Stellar coronae'],
                    'medium': ['Black hole disks', 'Supernova remnants', 'Galaxy clusters'],
                    'low': ['Black holes', 'Exploded stars']
                },
                'physics': {
                    'high': 'Thermal bremsstrahlung, million-K plasma',
                    'medium': 'Million-degree gas, violent processes',
                    'low': 'Extremely hot gas'
                },
                'temp_range': '10⁶-10⁸ K',
                'telescopes': ['Chandra', 'XMM-Newton', 'Swift', 'NuSTAR']
            },
            'Gamma': {
                'energy_range': (1e5, 1e9),
                'wavelength_range': (1e-15, 1e-11),
                'color': '#BB88FF',
                'objects': {
                    'high': ['Gamma-ray bursts', 'Pulsar magnetospheres', 'AGN jets', 'Cosmic ray interactions'],
                    'medium': ['Gamma-ray bursts', 'Pulsars', 'Active galaxies'],
                    'low': ['Cosmic explosions', 'Extreme objects']
                },
                'physics': {
                    'high': 'Nuclear processes, pair production, inverse Compton',
                    'medium': 'Nuclear processes, extreme physics',
                    'low': 'Most violent cosmic events'
                },
                'temp_range': '10⁸+ K',
                'telescopes': ['Fermi', 'HESS', 'MAGIC', 'VERITAS']
            }
        }
        
        # Extract appropriate detail level
        for band in base_data.values():
            band['objects'] = band['objects'][self.detail_level]
            band['physics'] = band['physics'][self.detail_level]
            
        return base_data

    def create_figure(self, save_path=None, show_figure=True):
        """Create the main EM spectrum figure"""
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=self.fig_params['figsize'], 
                              dpi=self.fig_params['dpi'])
        
        spectrum_data = self.get_spectrum_data()
        fontsize_base = self.fig_params['fontsize_base']
        
        # Create spectrum bar
        spectrum_height = 1.0
        spectrum_y = 4.0
        bar_positions = np.linspace(0, 12, 7)
        
        # Draw spectrum bands
        for i, (band_name, props) in enumerate(spectrum_data.items()):
            x_start = bar_positions[i]
            x_end = bar_positions[i+1]
            
            # Main spectrum rectangle with gradient effect
            rect = patches.Rectangle((x_start, spectrum_y), x_end - x_start, spectrum_height,
                                   facecolor=props['color'], alpha=0.8, 
                                   edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Band labels
            ax.text(x_start + (x_end - x_start)/2, spectrum_y + spectrum_height + 0.2,
                   band_name, ha='center', va='bottom', 
                   fontsize=fontsize_base + 2, fontweight='bold')
        
        # Energy scale
        if self.show_equations:
            energy_positions = [1e-9, 1e-6, 1e-3, 1, 1e3, 1e6, 1e9]
            energy_labels = ['1 neV', '1 μeV', '1 meV', '1 eV', '1 keV', '1 MeV', '1 GeV']
        else:
            energy_positions = [1e-6, 1e-3, 1, 1e3, 1e6]
            energy_labels = ['Low Energy', 'Medium Energy', 'Visible Energy', 'High Energy', 'Extreme Energy']
        
        for i, (energy, label) in enumerate(zip(energy_positions[:6], energy_labels[:6])):
            x_pos = 2 * i
            ax.annotate(label, xy=(x_pos, spectrum_y + spectrum_height + 0.8),
                       ha='center', va='bottom', fontsize=fontsize_base - 1,
                       arrowprops=dict(arrowstyle='->', lw=1, color='black'))
        
        # Physics ladder arrow
        arrow = patches.FancyArrowPatch((1, 6.5), (11, 6.5),
                                      connectionstyle="arc3",
                                      arrowstyle='->', mutation_scale=20,
                                      color='red', linewidth=3)
        ax.add_patch(arrow)
        
        ax.text(6, 7, 'PHYSICS LADDER', ha='center', va='bottom',
               fontsize=fontsize_base + 4, fontweight='bold', color='red',
               path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])
        
        # Add objects and physics for each band
        y_positions = [3.2, 2.7, 2.2, 1.7]
        
        for i, (band_name, props) in enumerate(spectrum_data.items()):
            x_center = bar_positions[i] + (bar_positions[i+1] - bar_positions[i])/2
            
            # Objects (limit based on detail level)
            max_objects = {'high': 4, 'medium': 3, 'low': 2}[self.detail_level]
            for j, obj in enumerate(props['objects'][:max_objects]):
                if j < len(y_positions):
                    ax.text(x_center, y_positions[j], f"• {obj}", ha='center', va='center',
                           fontsize=fontsize_base - 2, 
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor=props['color'], alpha=0.3))
            
            # Physics description
            ax.text(x_center, 1.2, props['physics'], ha='center', va='center',
                   fontsize=fontsize_base - 1, fontweight='bold', style='italic')
            
            # Temperature
            ax.text(x_center, 0.7, f"T ~ {props['temp_range']}", ha='center', va='center',
                   fontsize=fontsize_base - 2, color='darkred')
        
        # Add telescopes if requested
        if self.show_telescopes:
            telescope_info = [
                ('VLA\nALMA', 1, 0.2),
                ('JWST\nSpitzer', 3, 0.2),
                ('HST\nGround', 5, 0.2),
                ('GALEX\nSpace', 7, 0.2),
                ('Chandra\nXMM', 9, 0.2),
                ('Fermi\nHESS', 11, 0.2)
            ]
            
            for name, x, y in telescope_info:
                ax.text(x, y, name, ha='center', va='center', 
                       fontsize=fontsize_base - 3,
                       bbox=dict(boxstyle="round,pad=0.15", 
                               facecolor='lightgray', alpha=0.8))
        
        # Educational sidebar
        if self.detail_level == 'high':
            sidebar_x = 12.5
            ax.text(sidebar_x, 4.5, 'Graduate-Level Insights:',
                   ha='left', va='top', fontsize=fontsize_base, fontweight='bold')
            
            insights = [
                '• Different wavelengths = different physics',
                '• Energy determines accessible processes',
                '• Multi-wavelength observations essential',
                '• Instrument design follows physics'
            ]
            
            for i, insight in enumerate(insights):
                ax.text(sidebar_x, 4.1 - i*0.3, insight, ha='left', va='top',
                       fontsize=fontsize_base - 1)
        
        # Wien's law connection
        if self.show_equations:
            ax.text(12.5, 2.5, "Wien's Law:", ha='left', va='top',
                   fontsize=fontsize_base, fontweight='bold')
            ax.text(12.5, 2.2, r'$\lambda_{max} = \frac{0.29 \mathrm{~cm·K}}{T}$',
                   ha='left', va='top', fontsize=fontsize_base - 1)
        
        # Formatting
        ax.set_xlim(-0.5, 16)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Title
        titles = {
            'graduate': 'The Electromagnetic Spectrum: A Physics Ladder for Professional Astronomy',
            'undergraduate': 'The Electromagnetic Spectrum: Different Wavelengths, Different Physics',
            'public': 'The Rainbow of Space: How Astronomers See the Invisible Universe'
        }
        
        fig.suptitle(titles[self.style], fontsize=fontsize_base + 6, 
                    fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(f'{save_path}.png', dpi=self.fig_params['dpi'], 
                       bbox_inches='tight', facecolor='white')
            plt.savefig(f'{save_path}.pdf', bbox_inches='tight', facecolor='white')
            print(f"Figure saved as {save_path}.png and {save_path}.pdf")
        
        if show_figure:
            plt.show()
        
        return fig, ax

def main():
    parser = argparse.ArgumentParser(description='Generate EM Spectrum Physics Ladder Figure')
    parser.add_argument('--style', choices=['graduate', 'undergraduate', 'public'], 
                       default='graduate', help='Detail level and style')
    parser.add_argument('--size', choices=['large', 'medium', 'small', 'presentation'], 
                       default='large', help='Figure size')
    parser.add_argument('--output', default='/mnt/user-data/outputs/enhanced_em_spectrum', 
                       help='Output file path (without extension)')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display figure')
    
    args = parser.parse_args()
    
    visualizer = EMSpectrumVisualizer(style=args.style, size=args.size)
    fig, ax = visualizer.create_figure(save_path=args.output, 
                                      show_figure=not args.no_show)

if __name__ == "__main__":
    main()
