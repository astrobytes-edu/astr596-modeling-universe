
"""
physics_cgs.py - Physics calculations in CGS units.

All calculations use CGS (centimeter-gram-second) units.
"""

# Physical constants in CGS
SPEED_OF_LIGHT = 2.998e10      # cm/s
PLANCK_CONSTANT = 6.626e-27    # erg⋅s  
BOLTZMANN = 1.381e-16          # erg/K
ELECTRON_MASS = 9.109e-28      # g
PROTON_MASS = 1.673e-24        # g
GRAVITATIONAL_CONSTANT = 6.674e-8  # cm³/(g⋅s²)

def kinetic_energy(mass_g, velocity_cms):
    """KE = ½mv² in ergs."""
    return 0.5 * mass_g * velocity_cms**2

def momentum(mass_g, velocity_cms):
    """p = mv in g⋅cm/s."""
    return mass_g * velocity_cms

def de_broglie_wavelength(mass_g, velocity_cms):
    """λ = h/p in cm."""
    p = momentum(mass_g, velocity_cms)
    return PLANCK_CONSTANT / p if p != 0 else float('inf')

def thermal_velocity(temp_k, mass_g):
    """RMS thermal velocity in cm/s."""
    import math
    return math.sqrt(3 * BOLTZMANN * temp_k / mass_g)

def photon_energy(wavelength_cm):
    """E = hc/λ in ergs."""
    return PLANCK_CONSTANT * SPEED_OF_LIGHT / wavelength_cm

def gravitational_force(m1_g, m2_g, distance_cm):
    """F = Gm₁m₂/r² in dynes."""
    if distance_cm == 0:
        return float('inf')
    return GRAVITATIONAL_CONSTANT * m1_g * m2_g / distance_cm**2

class Particle:
    """Simple particle with physics methods."""

    def __init__(self, mass_g, velocity_cms):
        self.mass = mass_g
        self.velocity = velocity_cms

    def kinetic_energy(self):
        return kinetic_energy(self.mass, self.velocity)

    def wavelength(self):
        return de_broglie_wavelength(self.mass, self.velocity)
