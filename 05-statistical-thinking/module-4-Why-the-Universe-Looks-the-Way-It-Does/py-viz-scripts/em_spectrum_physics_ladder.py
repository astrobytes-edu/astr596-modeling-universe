#!/usr/bin/env python3
"""
Electromagnetic Spectrum Physics Ladder Figure
Creates a pedagogical visualization showing the EM spectrum with energy scales
and familiar astronomical objects for astronomy education.

Author: Module 4 Educational Materials
Purpose: Graduate-level astronomy pedagogy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

# Set up the figure with high DPI for publication quality
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=300)

# Define the electromagnetic spectrum bands with their properties
spectrum_bands = {
    'Radio': {
        'energy_range': (1e-9, 1e-5),  # eV
        'wavelength_range': (1e-1, 1e3),  # meters  
        'color': '#FF4444',
        'objects': ['Galaxy spiral arms', '21-cm hydrogen line', 'Pulsars'],
        'physics': 'Cold gas, magnetic fields',
        'temp_range': '10-100 K'
    },
    'Infrared': {
        'energy_range': (1e-5, 1),  # eV
        'wavelength_range': (1e-6, 1e-1),  # meters
        'color': '#FF8800', 
        'objects': ['Star formation regions', 'Cool stars & planets', 'Dust emission'],
        'physics': 'Warm dust, stellar nurseries',
        'temp_range': '100-1000 K'
    },
    'Visible': {
        'energy_range': (1, 3),  # eV
        'wavelength_range': (4e-7, 7e-7),  # meters
        'color': '#FFFF00',
        'objects': ['Stellar photospheres', 'Nebular emission', 'Galaxies'],
        'physics': 'Atomic transitions',
        'temp_range': '3000-10,000 K'  
    },
    'Ultraviolet': {
        'energy_range': (3, 100),  # eV
        'wavelength_range': (1e-8, 4e-7),  # meters
        'color': '#8888FF',
        'objects': ['Hot young stars', 'Star-forming regions', 'AGN'],
        'physics': 'High-energy transitions',
        'temp_range': '10,000+ K'
    },
    'X-ray': {
        'energy_range': (100, 1e5),  # eV  
        'wavelength_range': (1e-11, 1e-8),  # meters
        'color': '#FF88FF',
        'objects': ['Black hole accretion', 'Supernova remnants', 'Galaxy clusters'],
        'physics': 'Million-degree plasma',
        'temp_range': '10⁶-10⁸ K'
    },
    'Gamma': {
        'energy_range': (1e5, 1e9),  # eV
        'wavelength_range': (1e-15, 1e-11),  # meters  
        'color': '#BB88FF',
        'objects': ['Gamma-ray bursts', 'Pulsars', 'Cosmic ray interactions'],
        'physics': 'Nuclear processes',
        'temp_range': '10⁸+ K'
    }
}

# Create the main spectrum bar
spectrum_height = 1.0
spectrum_y = 4.0
bar_positions = np.linspace(0, 12, 7)  # 6 bands + end point

# Draw spectrum bands with smooth transitions
for i, (band_name, props) in enumerate(spectrum_bands.items()):
    x_start = bar_positions[i] 
    x_end = bar_positions[i+1]
    
    # Main spectrum rectangle
    rect = patches.Rectangle((x_start, spectrum_y), x_end - x_start, spectrum_height,
                           facecolor=props['color'], alpha=0.8, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Add band labels
    ax.text(x_start + (x_end - x_start)/2, spectrum_y + spectrum_height + 0.2, 
            band_name, ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add energy scale (logarithmic)
energy_positions = [1e-9, 1e-6, 1e-3, 1, 1e3, 1e6, 1e9]
energy_labels = ['1 neV', '1 μeV', '1 meV', '1 eV', '1 keV', '1 MeV', '1 GeV']

for i, (energy, label) in enumerate(zip(energy_positions, energy_labels)):
    x_pos = 2 * i  # Spread across spectrum
    ax.annotate(label, xy=(x_pos, spectrum_y + spectrum_height + 0.8), 
                ha='center', va='bottom', fontsize=10,
                arrowprops=dict(arrowstyle='->', lw=1, color='black'))

# Add wavelength scale  
wavelength_labels = ['km', 'm', 'mm', 'μm', 'nm', 'pm', 'fm']
for i, label in enumerate(wavelength_labels):
    x_pos = 2 * i
    ax.text(x_pos, spectrum_y - 0.3, label, ha='center', va='top', fontsize=10, 
            style='italic', color='blue')

# Add "Physics Ladder" arrow and concept
arrow = patches.FancyArrowPatch((1, 6.5), (11, 6.5),
                              connectionstyle="arc3", 
                              arrowstyle='->', mutation_scale=25,
                              color='red', linewidth=3)
ax.add_patch(arrow)

ax.text(6, 7, 'PHYSICS LADDER', ha='center', va='bottom', 
        fontsize=16, fontweight='bold', color='red',
        path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])
ax.text(6, 6.7, 'Increasing Energy → More Violent Physics', ha='center', va='bottom', 
        fontsize=12, style='italic', color='red')

# Add astronomical objects for each band
y_positions = [3, 2.5, 2, 1.5]  # Multiple levels for objects

for i, (band_name, props) in enumerate(spectrum_bands.items()):
    x_center = bar_positions[i] + (bar_positions[i+1] - bar_positions[i])/2
    
    # Add representative objects
    for j, obj in enumerate(props['objects'][:3]):  # Max 3 objects per band
        if j < len(y_positions):
            ax.text(x_center, y_positions[j], f"• {obj}", ha='center', va='top',
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor=props['color'], alpha=0.3))
    
    # Add physics description
    ax.text(x_center, 1, props['physics'], ha='center', va='top',
           fontsize=10, fontweight='bold', style='italic')
    
    # Add temperature range
    ax.text(x_center, 0.5, f"T ~ {props['temp_range']}", ha='center', va='top',
           fontsize=9, color='darkred')

# Add telescopes/instruments
telescope_info = [
    ('VLA\nArecibo', 1, 0.2),
    ('JWST\nSpitzer', 3, 0.2), 
    ('HST\nGround', 5, 0.2),
    ('GALEX\nSpace', 7, 0.2),
    ('Chandra\nXMM', 9, 0.2),
    ('Fermi\nCGRO', 11, 0.2)
]

for name, x, y in telescope_info:
    ax.text(x, y, name, ha='center', va='center', fontsize=8,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.8))

# Add educational annotations
ax.text(12.5, 4.5, 'Why Multi-Wavelength\nAstronomy Matters:', 
        ha='left', va='top', fontsize=12, fontweight='bold')

educational_points = [
    '• Each wavelength reveals different physics',
    '• Complete picture needs full spectrum', 
    '• Different instruments required',
    '• Energy determines accessible processes'
]

for i, point in enumerate(educational_points):
    ax.text(12.5, 4 - i*0.4, point, ha='left', va='top', fontsize=10)

# Add Wien's Law connection
ax.text(12.5, 2, 'Wien\'s Law Connection:', ha='left', va='top', 
        fontsize=12, fontweight='bold')
ax.text(12.5, 1.7, r'$\lambda_{max} = \frac{0.29 \mathrm{~cm·K}}{T}$', 
        ha='left', va='top', fontsize=11)
ax.text(12.5, 1.4, 'Peak wavelength ∝ 1/Temperature', ha='left', va='top', 
        fontsize=10, style='italic')

# Formatting and styling
ax.set_xlim(-0.5, 16)
ax.set_ylim(0, 8)
ax.set_aspect('equal', adjustable='box')

# Remove axes for clean appearance
ax.axis('off')

# Add title and subtitle
fig.suptitle('The Electromagnetic Spectrum: A Physics Ladder for Astronomy', 
             fontsize=20, fontweight='bold', y=0.95)

subtitle = ('Each wavelength band reveals different astronomical phenomena and physical processes.\n'
           'Higher photon energies probe increasingly violent and exotic cosmic physics.')
ax.text(6, 7.7, subtitle, ha='center', va='top', fontsize=12, style='italic',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

# Add scale indicators
ax.text(0, spectrum_y - 0.8, 'Wavelength:', ha='left', va='top', 
        fontsize=11, fontweight='bold', color='blue')
ax.text(0, spectrum_y + spectrum_height + 1.2, 'Photon Energy:', ha='left', va='top', 
        fontsize=11, fontweight='bold', color='black')

plt.tight_layout()

# Save the figure
plt.savefig('/mnt/user-data/outputs/em_spectrum_physics_ladder.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/mnt/user-data/outputs/em_spectrum_physics_ladder.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("EM Spectrum Physics Ladder figure saved as:")
print("- em_spectrum_physics_ladder.png (high-resolution)")
print("- em_spectrum_physics_ladder.pdf (vector format)")
print("\nFigure includes:")
print("✓ Complete electromagnetic spectrum with accurate energy scales")
print("✓ Representative astronomical objects for each band")
print("✓ Physics processes and temperature ranges")
print("✓ Telescope/instrument examples")
print("✓ Educational annotations and Wien's law connection")
print("✓ Professional formatting suitable for graduate education")

# Display the figure
plt.show()
