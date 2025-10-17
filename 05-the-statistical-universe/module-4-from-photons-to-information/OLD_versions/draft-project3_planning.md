# Project 3: Monte Carlo Radiative Transfer Through Dusty Star Clusters

## Observing Your Simulated Universe Through Cosmic Dust

---

## Project Overview

In Projects 1 and 2, you created stellar populations and evolved them dynamically through gravitational N-body simulations. Now you'll discover what these clusters would actually look like when observed through realistic interstellar dust. This project bridges the gap between theoretical models and observable reality by implementing Monte Carlo radiative transfer to create synthetic observations of your star clusters.

The profound realization you'll gain from this project is that what we observe is not what exists. A cluster with 10,000 stars might show only 100 in optical observations if embedded in dust. The cluster's apparent structure, color-magnitude diagram, and even inferred properties like age and mass depend critically on the dust distribution along our line of sight. By the end of this project, you'll never look at an astronomical image the same way again—you'll always wonder what's hidden behind the dust.

---

## Learning Objectives

By completing this project, you will:

1. **Implement** a gray Monte Carlo radiative transfer code to simulate photon propagation through dusty media
2. **Apply** your code to realistic star clusters from Projects 1 and 2, creating synthetic observations
3. **Analyze** how dust affects observed cluster properties across multiple evolutionary snapshots
4. **Understand** the fundamental difference between intrinsic and observed astronomical properties
5. **Develop** debugging strategies for stochastic simulations where errors manifest statistically
6. **Create** synthetic color-magnitude diagrams showing reddening and extinction effects
7. **Quantify** completeness limits and selection effects in dusty environments

---

## Scientific Motivation

Every astronomical observation beyond our solar system involves light that has traveled through the interstellar medium. The Milky Way contains approximately one dust grain per million cubic meters—sparse by terrestrial standards but significant over astronomical distances. These dust grains preferentially absorb and scatter blue light, making objects appear both fainter and redder than they intrinsically are. 

Consider the striking case of Westerlund 2, one of the Milky Way's most massive young clusters. In optical images, we see perhaps 50 bright blue stars. Switch to near-infrared observations, and suddenly over 5,000 stars appear! The cluster didn't change—our ability to peer through dust did. Your Project 3 will quantify exactly how this transformation occurs.

This project connects directly to current research. Every JWST observation of star-forming regions involves correcting for dust extinction. Every attempt to measure the stellar initial mass function must account for dust hiding low-mass stars. Every distance measurement to a star cluster requires disentangling the effects of distance and dust. By implementing MCRT yourself, you'll understand both the power and limitations of these corrections.

---

## Project Structure

### Part A: Core Implementation (Required, 70% of grade)

You will implement a gray (wavelength-independent) Monte Carlo radiative transfer code with the following requirements:

**1. Photon Packet Initialization**
- Sample photon packets from your stellar population luminosities
- Implement isotropic emission from point sources
- Correctly weight packets by stellar luminosity

**2. Optical Depth Integration**
- Support two density distributions:
  - Uniform density: $\rho = \rho_0$
  - Power-law profile: $\rho = \rho_0 (r/r_0)^{-\alpha}$ where $\alpha \in [0, 2]$
- Calculate optical depth along arbitrary rays
- Implement optical depth sampling: $\tau = -\ln(\xi)$

**3. Photon Propagation**
- Transport photons through 3D space
- Implement absorption and scattering (albedo $\omega$)
- Handle boundary conditions (escape or absorb)

**4. Image Generation**
- Create 2D projected images at specified viewing angles
- Implement proper flux collection in detector pixels
- Handle both face-on and edge-on viewing geometries

**5. Verification Tests**
- Energy conservation: absorbed + escaped = emitted
- Comparison with analytical solutions for uniform slab
- Convergence tests showing $\sigma \propto 1/\sqrt{N}$

### Part B: Cluster Evolution Analysis (Required, 20% of grade)

Apply your MCRT code to at least 5 snapshots from your Project 2 N-body simulation:

**1. Time Evolution Visualization**
- Create images at t = 0, 25, 50, 75, 100 Myr
- Show both "no dust" and "with dust" versions
- Document how cluster appearance changes with dynamics

**2. Observable Properties**
- Calculate total observed luminosity vs time
- Measure half-light radius in dusty vs dust-free cases  
- Quantify how many stars are "lost" to extinction

**3. Color-Magnitude Diagrams**
- Construct CMDs with and without dust
- Show reddening vectors for different dust amounts
- Identify which stellar types are most affected

### Part C: Scientific Analysis (Required, 10% of grade)

**1. Completeness Analysis**
- Determine what fraction of stars remain detectable
- How does completeness depend on stellar mass?
- What is your effective limiting magnitude with dust?

**2. Structural Changes**
- Does dust change the apparent cluster morphology?
- Is the cluster center shifted by differential extinction?
- How does the surface brightness profile change?

**3. Written Report**
- 3-4 page analysis of your results
- Discussion of how dust affects cluster studies
- Implications for interpreting real observations

---

## Optional Extensions (Extra Credit)

### Extension 1: Multi-Wavelength Observations (+15%)
Extend your code to handle different wavelengths:
- Implement wavelength-dependent opacity: $\kappa_\lambda \propto \lambda^{-\beta}$
- Create images in optical (V) and near-IR (K) bands
- Show how cluster appearance changes with wavelength
- Calculate color maps showing spatial extinction variations

### Extension 2: Complex Dust Geometries (+15%)
Move beyond spherical distributions:
- Implement clumpy dust with multiple dense cores
- Add a dust disk or shell around the cluster
- Include foreground screen + embedded dust
- Show how geometry affects observed properties

### Extension 3: Scattering Physics (+10%)
Replace isotropic scattering with realistic phase functions:
- Implement Henyey-Greenstein scattering
- Add polarization tracking
- Create scattered light images
- Show how forward scattering affects results

### Extension 4: Adaptive Techniques (+10%)
Improve computational efficiency:
- Implement variance reduction techniques
- Add photon splitting in high-opacity regions
- Use importance sampling for source emission
- Demonstrate improved convergence rates

---

## Implementation Strategy

### Phase 1: Foundation (Week 1)
Start with the simplest possible case that you can verify:
1. Single star at origin
2. Uniform density sphere
3. Pure absorption (no scattering)
4. Compare with analytical $e^{-\tau}$ attenuation

### Phase 2: Scattering (Week 2)
Add complexity incrementally:
1. Add isotropic scattering with albedo $\omega$
2. Verify energy conservation precisely
3. Test mean free path statistics
4. Check photon path length distributions

### Phase 3: Realistic Sources (Week 3)
Incorporate your stellar populations:
1. Read StellarPopulation object from Project 1
2. Sample photons according to luminosities
3. Handle multiple point sources
4. Create first cluster images

### Phase 4: Dynamics Integration (Week 4)
Connect to your N-body simulations:
1. Read multiple snapshots from Project 2
2. Update stellar positions and luminosities
3. Generate time sequence of images
4. Analyze evolution of observables

---

## Debugging Strategies

Monte Carlo codes present unique debugging challenges because errors often manifest statistically rather than as obvious failures. Here's a systematic approach:

### 1. Physical Validation Tests

**Energy Conservation**
Your code must conserve energy to machine precision:
```
Energy_in = Sum over all stellar luminosities × time
Energy_out = Energy_escaped + Energy_absorbed
Fractional_error = |Energy_in - Energy_out| / Energy_in < 1e-10
```

**Optical Depth Tests**
For uniform medium with total optical depth $\tau_{total}$:
- Fraction transmitted (no scattering): $e^{-\tau_{total}}$
- Fraction absorbed: $1 - e^{-\tau_{total}}$
- Your code should match these within statistical error

**Mean Free Path**
Track path lengths between interactions:
- Should follow exponential distribution
- Mean should equal $1/(\kappa \rho)$
- Verify with histogram of path lengths

### 2. Statistical Validation

**Convergence Behavior**
Run with N = 10³, 10⁴, 10⁵, 10⁶ photons:
- Measure standard deviation σ of results
- Plot log(σ) vs log(N)
- Should see slope of -0.5

**Isotropy Tests**
For isotropic point source in empty space:
- Flux should follow inverse square law
- Angular distribution should be uniform
- No preferred directions should appear

### 3. Visualization Debugging

**Path Visualization**
For small photon numbers (~100):
- Plot 3D paths of all photons
- Color by fate (escaped vs absorbed)
- Check for obvious geometric errors

**Density Cross-Sections**
Plot 2D slices through your density grid:
- Verify correct implementation of ρ(r)
- Check boundary conditions
- Ensure no discontinuities

---

## Physical Parameters

### Dust Properties (Gray Approximation)
- Mass absorption coefficient: $\kappa = 100$ cm²/g (at V-band)
- Albedo: $\omega = 0.6$ (60% scatter, 40% absorb)
- Mean density for $A_V = 1$ mag at 1 kpc: $\rho_0 = 10^{-23}$ g/cm³

### Cluster Environment Options
Choose one for base implementation:

**Low Extinction (Field Cluster)**
- Uniform density: $\rho = 10^{-24}$ g/cm³
- Typical $A_V \sim 0.5$ mag through cluster
- Most stars remain visible

**Moderate Extinction (Spiral Arm)**  
- Power law: $\rho = 10^{-23} (r/100 \text{ pc})^{-1}$ g/cm³
- Central $A_V \sim 3$ mag
- Significant reddening and incompleteness

**High Extinction (Star-Forming Region)**
- Power law: $\rho = 10^{-22} (r/10 \text{ pc})^{-1.5}$ g/cm³  
- Central $A_V \sim 10$ mag
- Only brightest stars visible in optical

### Computational Parameters
- Minimum photons for testing: $N_{phot} = 10^4$
- Minimum for analysis: $N_{phot} = 10^6$
- Recommended for final images: $N_{phot} = 10^7$
- Image resolution: 256×256 pixels minimum
- Field of view: 2× cluster half-mass radius

---

## Expected Outcomes

Your successfully completed project will demonstrate:

### Visual Results
1. **Image Gallery**: Side-by-side comparison of cluster with/without dust at multiple times
2. **Evolution Movie**: Animation showing cluster evolution through dust
3. **Multi-Angle Views**: Same cluster from different viewing angles through dust

### Quantitative Results
1. **Extinction Curves**: Plot of observed vs intrinsic luminosity for different dust amounts
2. **CMD Transformation**: Showing reddening vectors and magnitude limits
3. **Completeness Functions**: Fraction of stars detected vs magnitude
4. **Structure Analysis**: Half-light radius and concentration changes

### Scientific Insights
1. Understanding of selection effects in cluster observations
2. Appreciation for why infrared observations are crucial
3. Recognition that observed ≠ intrinsic properties
4. Experience with stochastic simulation methods

---

## Assessment Rubric

### Core Implementation (70%)
- **Correct Physics** (25%): Accurate optical depth calculation, proper scattering implementation
- **Energy Conservation** (15%): Demonstrated to < 0.1% error
- **Code Structure** (15%): Clear, documented, modular design
- **Verification Tests** (15%): Comparison with analytical solutions

### Cluster Analysis (20%)
- **Time Evolution** (10%): Multiple snapshots processed correctly
- **Observable Properties** (10%): Proper calculation of luminosities, radii, completeness

### Scientific Analysis (10%)
- **Written Report** (5%): Clear presentation of results
- **Physical Interpretation** (5%): Correct understanding of dust effects

### Code Quality (Applies to all parts)
- Clear variable names and code organization
- Appropriate comments explaining physics
- Efficient algorithm implementation
- Proper error handling

---

## Resources and References

### Key Papers
- Steinacker et al. (2013): "Three-Dimensional Dust Radiative Transfer" - Comprehensive review
- Whitney (2011): "Monte Carlo Radiative Transfer" - Practical implementation guide
- Robitaille (2011): "HYPERION: An open-source parallelized 3D dust continuum radiative transfer code"

### Validation Data
- Ivezić et al. (1997): Benchmark problems for dust radiative transfer
- Pascucci et al. (2004): 2D benchmark test cases
- Gordon et al. (2017): TRUST benchmark for 3D geometries

### Useful Algorithms
- Photon path sampling: Wood & Reynolds (1999)
- Variance reduction: Yusef-Zadeh et al. (1984)
- Grid traversal: Amanatides & Woo (1987)

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Incorrect Random Number Usage
**Problem**: Using same random seed repeatedly gives identical photon paths
**Solution**: Initialize RNG once, then use sequentially

### Pitfall 2: Energy Non-Conservation
**Problem**: Lost photons at boundaries or numerical precision issues
**Solution**: Track every photon explicitly, use double precision

### Pitfall 3: Biased Sampling
**Problem**: Not weighting photons by stellar luminosity correctly
**Solution**: Use cumulative distribution function for source sampling

### Pitfall 4: Wrong Units
**Problem**: Mixing CGS and SI, or forgetting conversion factors
**Solution**: Use consistent CGS throughout, document all units

### Pitfall 5: Infinite Loops
**Problem**: Photon stuck in high scattering region
**Solution**: Implement maximum scattering limit or Russian roulette

---

## Timeline and Milestones

### Week 1: Foundation
- [ ] Implement basic photon transport
- [ ] Verify against uniform slab
- [ ] Achieve energy conservation

### Week 2: Scattering and Sources
- [ ] Add scattering with albedo
- [ ] Implement stellar population sources
- [ ] Create first images

### Week 3: Dynamics Integration
- [ ] Read N-body snapshots
- [ ] Generate time evolution
- [ ] Produce CMDs

### Week 4: Analysis and Report
- [ ] Complete statistical analysis
- [ ] Generate all figures
- [ ] Write final report

---

## Code Examples and Snippets

### 1. Basic Photon Packet Structure
```python
import numpy as np

class PhotonPacket:
    """
    Represents a Monte Carlo photon packet traveling through dusty medium
    """
    def __init__(self, position, direction, energy):
        """
        Initialize a photon packet
        
        Parameters:
        -----------
        position : array-like, shape (3,)
            Initial position [x, y, z] in cm
        direction : array-like, shape (3,)  
            Unit vector for propagation direction
        energy : float
            Energy carried by packet in erg
        """
        self.pos = np.array(position, dtype=np.float64)
        self.dir = np.array(direction, dtype=np.float64)
        self.dir /= np.linalg.norm(self.dir)  # Ensure unit vector
        self.energy = energy
        self.tau_traveled = 0.0  # Optical depth traveled
        self.n_scatters = 0      # Number of scattering events
        self.absorbed = False    # Has packet been absorbed?
        self.escaped = False     # Has packet escaped domain?
```

### 2. Optical Depth Sampling
```python
def sample_optical_depth():
    """
    Sample optical depth to next interaction using inverse transform method
    
    The probability of traveling optical depth tau without interaction is exp(-tau).
    Using inverse transform: tau = -ln(xi) where xi is uniform random [0,1]
    
    Returns:
    --------
    tau : float
        Optical depth to next interaction
    """
    xi = np.random.random()
    # Avoid log(0) by using 1-xi if xi is exactly 0
    if xi == 0:
        xi = 1.0 - xi
    tau = -np.log(xi)
    return tau
```

### 3. Isotropic Scattering Implementation
```python
def scatter_isotropic(packet):
    """
    Scatter photon packet into random isotropic direction
    
    Parameters:
    -----------
    packet : PhotonPacket
        Packet to scatter
        
    Returns:
    --------
    None (modifies packet in place)
    """
    # Sample random direction on unit sphere
    # Use Marsaglia (1972) method for uniform sphere point picking
    
    # Sample z uniformly in [-1, 1]
    z = 2.0 * np.random.random() - 1.0
    
    # Sample phi uniformly in [0, 2π]
    phi = 2.0 * np.pi * np.random.random()
    
    # Calculate x, y from z and phi
    r_perp = np.sqrt(1.0 - z**2)
    x = r_perp * np.cos(phi)
    y = r_perp * np.sin(phi)
    
    # Update packet direction
    packet.dir = np.array([x, y, z])
    packet.n_scatters += 1
```

### 4. Ray-Sphere Intersection for Escape
```python
def ray_sphere_intersect(pos, dir, radius):
    """
    Find intersection of ray with sphere boundary
    
    Parameters:
    -----------
    pos : array-like, shape (3,)
        Current position
    dir : array-like, shape (3,)
        Direction (unit vector)
    radius : float
        Sphere radius
        
    Returns:
    --------
    distance : float
        Distance to sphere boundary (inf if no intersection)
    """
    # Solve quadratic equation for ray-sphere intersection
    # Ray: r(t) = pos + t*dir
    # Sphere: |r|^2 = radius^2
    
    a = np.dot(dir, dir)  # Should be 1 for unit vector
    b = 2.0 * np.dot(pos, dir)
    c = np.dot(pos, pos) - radius**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return np.inf  # No intersection
    
    # Two solutions, we want the positive one (forward along ray)
    t1 = (-b - np.sqrt(discriminant)) / (2*a)
    t2 = (-b + np.sqrt(discriminant)) / (2*a)
    
    if t2 < 0:
        return np.inf  # Both intersections behind ray
    elif t1 < 0:
        return t2  # Inside sphere, use exit point
    else:
        return t1  # Outside sphere, use entry point
```

### 5. Density Profiles
```python
def density_uniform(pos, rho0):
    """
    Uniform density distribution
    
    Parameters:
    -----------
    pos : array-like, shape (3,)
        Position in cm
    rho0 : float
        Density in g/cm^3
        
    Returns:
    --------
    rho : float
        Density at position
    """
    return rho0

def density_power_law(pos, rho0, r0, alpha):
    """
    Power-law density profile: rho = rho0 * (r/r0)^(-alpha)
    
    Parameters:
    -----------
    pos : array-like, shape (3,)
        Position in cm
    rho0 : float
        Reference density in g/cm^3
    r0 : float
        Reference radius in cm
    alpha : float
        Power law index (positive for decreasing density)
        
    Returns:
    --------
    rho : float
        Density at position
    """
    r = np.linalg.norm(pos)
    if r < r0 * 0.01:  # Avoid singularity at origin
        r = r0 * 0.01
    return rho0 * (r / r0)**(-alpha)
```

### 6. Main Monte Carlo Loop Structure
```python
def run_monte_carlo(sources, n_photons, dust_params, domain_size):
    """
    Main Monte Carlo radiative transfer loop
    
    Parameters:
    -----------
    sources : list of StellarSource objects
        Stellar sources with positions and luminosities
    n_photons : int
        Number of photon packets to simulate
    dust_params : dict
        Contains 'kappa' (opacity), 'albedo', 'density_func'
    domain_size : float
        Radius of computational domain in cm
        
    Returns:
    --------
    results : dict
        Contains escaped energy, absorbed energy, images, etc.
    """
    # Initialize counters
    energy_escaped = 0.0
    energy_absorbed = 0.0
    
    # Initialize image array (example: 256x256 pixels)
    image = np.zeros((256, 256))
    
    for i in range(n_photons):
        # Sample source according to luminosity
        packet = sample_source(sources)
        
        while not (packet.absorbed or packet.escaped):
            # Sample optical depth to next interaction
            tau_to_event = sample_optical_depth()
            
            # Find physical distance for this optical depth
            distance = optical_depth_to_distance(packet, tau_to_event, dust_params)
            
            # Check if packet escapes domain
            dist_to_boundary = ray_sphere_intersect(packet.pos, packet.dir, domain_size)
            
            if distance > dist_to_boundary:
                # Packet escapes
                packet.escaped = True
                energy_escaped += packet.energy
                # Add to image based on escape direction
                add_to_image(image, packet)
            else:
                # Packet interacts with dust
                packet.pos += distance * packet.dir
                
                # Decide: scatter or absorb?
                if np.random.random() < dust_params['albedo']:
                    # Scatter
                    scatter_isotropic(packet)
                else:
                    # Absorb
                    packet.absorbed = True
                    energy_absorbed += packet.energy
        
        # Progress indicator
        if (i + 1) % (n_photons // 10) == 0:
            print(f"Progress: {100*(i+1)/n_photons:.0f}%")
    
    # Verify energy conservation
    energy_total = energy_escaped + energy_absorbed
    energy_input = sum(source.luminosity for source in sources) * n_photons
    conservation_error = abs(energy_total - energy_input) / energy_input
    print(f"Energy conservation error: {conservation_error:.2e}")
    
    return {
        'image': image,
        'energy_escaped': energy_escaped,
        'energy_absorbed': energy_absorbed,
        'conservation_error': conservation_error
    }
```

### 7. Integration with Previous Projects
```python
def setup_from_previous_projects(snapshot_file, stellar_pop_file):
    """
    Load data from Projects 1 and 2
    
    Parameters:
    -----------
    snapshot_file : str
        Path to N-body snapshot from Project 2
    stellar_pop_file : str
        Path to StellarPopulation from Project 1
        
    Returns:
    --------
    sources : list
        List of stellar sources with positions and luminosities
    """
    # Load N-body positions (Project 2)
    nbody_data = np.load(snapshot_file)
    positions = nbody_data['positions']  # Shape: (n_stars, 3)
    star_ids = nbody_data['star_ids']
    
    # Load stellar properties (Project 1)
    import pickle
    with open(stellar_pop_file, 'rb') as f:
        stellar_pop = pickle.load(f)
    
    # Match stars and create sources
    sources = []
    for i, star_id in enumerate(star_ids):
        star = stellar_pop.get_star(star_id)
        if star.luminosity > 0:  # Only include luminous stars
            source = StellarSource(
                position=positions[i],
                luminosity=star.luminosity,
                temperature=star.temperature
            )
            sources.append(source)
    
    return sources
```

### 8. Creating Synthetic Observations
```python
def create_synthetic_cmd(sources, dust_params, n_photons=1e6):
    """
    Create color-magnitude diagram with dust effects
    
    Parameters:
    -----------
    sources : list
        Stellar sources
    dust_params : dict
        Dust parameters including extinction
    n_photons : int
        Photons per band
        
    Returns:
    --------
    cmd_data : dict
        Contains intrinsic and observed magnitudes
    """
    # Intrinsic properties (no dust)
    v_intrinsic = np.array([-2.5 * np.log10(s.luminosity_v) for s in sources])
    b_intrinsic = np.array([-2.5 * np.log10(s.luminosity_b) for s in sources])
    
    # Run MCRT for V and B bands
    v_observed = run_band_observation(sources, 'V', dust_params, n_photons)
    b_observed = run_band_observation(sources, 'B', dust_params, n_photons)
    
    # Calculate colors
    color_intrinsic = b_intrinsic - v_intrinsic
    color_observed = b_observed - v_observed
    
    # Reddening vector
    E_BV = color_observed - color_intrinsic
    A_V = v_observed - v_intrinsic
    
    return {
        'v_intrinsic': v_intrinsic,
        'color_intrinsic': color_intrinsic,
        'v_observed': v_observed,
        'color_observed': color_observed,
        'E_BV': E_BV,
        'A_V': A_V
    }
```

---

## Final Thoughts

This project bridges the gap between theoretical models and observable reality. Your star clusters from Projects 1 and 2 exist in a perfect, dust-free computational universe. Project 3 adds the messy reality of interstellar dust, transforming your pristine models into something that could actually be observed with a telescope.

The Monte Carlo method you'll implement is the same technique used to interpret observations from the world's most sophisticated telescopes. When astronomers analyze JWST images of star-forming regions or measure distances to star clusters, they're using codes conceptually identical to what you'll write. The difference is only in complexity, not in fundamental approach.

Remember that debugging Monte Carlo codes requires patience and systematic testing. Trust your verification tests more than your intuition. If energy is conserved and your code matches analytical solutions in simple cases, it's probably working correctly even if the results surprise you. Dust can dramatically transform how clusters appear—that's the physics, not a bug in your code!