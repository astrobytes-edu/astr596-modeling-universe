## Chapter 5 Practice Exercises

### Exercise 5.1: Build Statistical Analysis Functions

Create a suite of analysis functions for experimental data:

```{code-cell} ipython3
def calculate_statistics(data):
    """
    Calculate comprehensive statistics.
    
    TODO: Your implementation should:
    1. Handle empty lists (return None or raise ValueError)
    2. Check for None values in the data
    3. Return a dictionary with: mean, std, min, max, median
    
    Example return value:
    {'mean': 5.2, 'std': 1.3, 'min': 3.1, 'max': 7.8, 'median': 5.0}
    """
    # Your implementation here
    # Start with: if not data: return None
    pass

def remove_outliers(data, n_sigma=3):
    """
    Remove points more than n_sigma standard deviations from mean.
    
    TODO: Your implementation should:
    1. Calculate mean and standard deviation
    2. Keep only values within mean ± n_sigma*std
    3. Return filtered list
    
    Hint: Use the statistics from calculate_statistics()
    """
    # Your implementation here
    pass

def bootstrap_error(data, statistic_func=None, n_samples=1000):
    """
    Estimate error using bootstrap resampling.
    
    TODO (Advanced challenge!):
    1. Default to mean if no statistic_func provided
    2. Resample data with replacement n_samples times
    3. Calculate statistic for each resample
    4. Return standard deviation of the statistics
    
    Hint: Use random.choices(data, k=len(data)) for resampling
    """
    # Your implementation here
    pass

# Test with sample data
test_data = [9.8, 9.7, 10.1, 9.9, 50.0, 9.8, 10.0, 9.9]  # Note outlier!
print(f"Original data: {test_data}")
print("Implement the functions above to analyze this data!")

# Once implemented, you should be able to:
# stats = calculate_statistics(test_data)
# clean_data = remove_outliers(test_data, n_sigma=2)
# error = bootstrap_error(clean_data)
```

### Exercise 5.2: Create a Scientific Module

Build `analysis_tools.py` module:

```python
"""
analysis_tools.py - Data analysis utilities

TODO: Create this module with:
1. Constants (confidence levels, etc.)
2. Statistical functions
3. Data cleaning functions
4. Plotting helpers
5. Module testing in __main__
"""

# Your module here
```

### Exercise 5.3: Variable Star Analysis Functions

Continue building our variable star analysis toolkit (building on Chapter 3's loop exercises and Chapter 4's data structures):

```{code-cell} ipython3
def generate_cepheid_data(period_days=5.4, amplitude_mag=0.3, n_points=50):
    """
    Generate simulated Cepheid variable star data.
    
    Parameters:
        period_days: Period in days
        amplitude_mag: Amplitude in magnitudes
        n_points: Number of observations
    
    Returns:
        times, magnitudes, errors (all as lists)
    """
    import random
    import math
    
    times = []
    mags = []
    errors = []
    
    for i in range(n_points):
        # Irregular sampling
        t = i * period_days / 10 + random.uniform(-0.1, 0.1)
        
        # Cepheid light curve (asymmetric - rises quickly, falls slowly)
        phase = (t % period_days) / period_days
        
        if phase < 0.3:
            # Rising branch (quick brightening)
            mag = 12.0 - amplitude_mag * (phase / 0.3)
        else:
            # Falling branch (slow dimming)
            mag = 12.0 - amplitude_mag * math.exp(-(phase - 0.3) / 0.4)
        
        # Add realistic noise
        mag += random.gauss(0, 0.02)
        error = 0.01 + 0.01 * random.random()
        
        times.append(t)
        mags.append(mag)
        errors.append(error)
    
    return times, mags, errors

# Create analysis functions (for you to implement)
def find_period_simple(times, mags):
    """
    TODO: Estimate period from data.
    Hint: Look for repeating patterns in brightness!
    
    One approach:
    1. Find time between brightness minima
    2. Average these intervals
    3. Return estimated period
    """
    # Your implementation here
    pass

def phase_fold(times, mags, period):
    """
    TODO: Fold data on given period.
    
    Algorithm:
    1. Calculate phase for each time: phase = (time % period) / period
    2. Sort by phase
    3. Return phases and corresponding magnitudes
    """
    # Your implementation here
    pass

# Generate and analyze data
t, m, e = generate_cepheid_data()
print(f"Generated {len(t)} observations")
print(f"Time range: {min(t):.1f} to {max(t):.1f} days")
print(f"Magnitude range: {min(m):.2f} to {max(m):.2f}")
print("\nImplement the analysis functions to find the period!")
print("(Building on the lightcurve data patterns from Chapter 3's exercises)")
```

## Chapter 6 Practice Exercises

### Exercise 1: Build a DataPoint Class (3-Part Scaffolded)

#### Part A: Basic DataPoint Implementation (5-10 minutes)

```{code-cell} ipython3
"""
Part A: Basic DataPoint class
Create a class that stores a value with timestamp
"""

import time

class DataPoint:
    """Single measurement with timestamp."""
    
    def __init__(self, value, label=""):
        """Initialize with value and optional label."""
        self.value = value
        self.label = label
        self.timestamp = time.time()
    
    def age_seconds(self):
        """How old is this data point?"""
        return time.time() - self.timestamp

# Test it
dp = DataPoint(42.5, "temperature")
print(f"Value: {dp.value}")
print(f"Label: {dp.label}")
# Note: Brief pause to demonstrate timestamp difference
# In real applications, timestamps naturally differ
time.sleep(0.01)  
print(f"Age: {dp.age_seconds():.3f} seconds")
```

#### Part B: Add Validation and Properties (10-15 minutes)

```{code-cell} ipython3
"""
Part B: Add validation and properties
Enhance with properties for safety
"""

class DataPoint:
    """Measurement with validation."""
    
    def __init__(self, value, error=0, label=""):
        """Initialize with value, error, and label."""
        self.value = value
        self.error = abs(error)  # Ensure positive
        self.label = label
        self.timestamp = time.time()
    
    @property
    def relative_error(self):
        """Relative error as percentage."""
        if self.value == 0:
            return float('inf')
        return (self.error / abs(self.value)) * 100
    
    @property
    def is_significant(self):
        """Check if measurement is significant."""
        return self.relative_error < 5.0  # Less than 5% error
    
    def __str__(self):
        """Nice string representation."""
        return f"{self.value} ± {self.error} ({self.label})"

# Test enhanced version
dp = DataPoint(100, 2, "voltage")
print(dp)
print(f"Relative error: {dp.relative_error:.1f}%")
print(f"Significant?: {dp.is_significant}")
```

#### Part C: Complete DataSeries Class with Analysis (15-20 minutes)

```{code-cell} ipython3
"""
Part C: Complete DataSeries class
Container for multiple DataPoints with statistical analysis
Synthesizes concepts from Chapters 3-5
"""

class DataSeries:
    """Collection of DataPoints with statistics."""
    
    def __init__(self, name):
        """Initialize empty series with name."""
        self.name = name
        self._points = []
    
    def add_point(self, value, error=0):
        """Add new data point."""
        dp = DataPoint(value, error, self.name)
        self._points.append(dp)
    
    def __len__(self):
        """Number of points."""
        return len(self._points)
    
    def __getitem__(self, index):
        """Access points by index."""
        return self._points[index]
    
    @property
    def values(self):
        """Array of values."""
        return [p.value for p in self._points]
    
    @property
    def mean(self):
        """Calculate mean value (Chapter 5 function concept)."""
        if not self._points:
            return 0
        return sum(self.values) / len(self._points)
    
    @property
    def std_dev(self):
        """Calculate standard deviation (Chapter 3 loop pattern)."""
        if len(self._points) < 2:
            return 0
        m = self.mean
        variance = sum((x - m)**2 for x in self.values) / (len(self._points) - 1)
        return variance**0.5
    
    def remove_outliers(self, n_sigma=3):
        """Remove outliers (Chapter 4 list operations)."""
        if len(self._points) < 3:
            return
        m = self.mean
        s = self.std_dev
        self._points = [p for p in self._points 
                       if abs(p.value - m) <= n_sigma * s]
    
    def __str__(self):
        """String representation with statistics."""
        return f"DataSeries '{self.name}': {len(self)} points, mean={self.mean:.2f}±{self.std_dev:.2f}"

# Test complete system
series = DataSeries("Temperature")
for temp in [20.1, 20.5, 19.8, 20.2, 20.0, 35.0]:  # Note outlier!
    series.add_point(temp, error=0.1)

print(f"Before cleaning: {series}")
series.remove_outliers(n_sigma=2)
print(f"After removing outliers: {series}")
print(f"All values: {series.values}")
```

### Exercise 2: Variable Star Class (Continuing Our Thread)

Building on Chapter 5's lightcurve analysis functions, now we organize our variable star code using OOP:

#### Part A: Basic VariableStar Class (5-10 minutes)

```{code-cell} ipython3
"""
Part A: Transform Chapter 5's functions into a class
"""

class VariableStar:
    """A variable star with photometric properties."""
    
    def __init__(self, name, observations):
        """
        Initialize with observations.
        observations: list of (time, magnitude, error) tuples
        """
        self.name = name
        self.observations = sorted(observations)  # Sort by time
    
    @property
    def mean_magnitude(self):
        """Average magnitude."""
        mags = [obs[1] for obs in self.observations]
        return sum(mags) / len(mags) if mags else 0
    
    @property
    def amplitude(self):
        """Peak-to-peak amplitude."""
        if not self.observations:
            return 0
        mags = [obs[1] for obs in self.observations]
        return max(mags) - min(mags)

# Test basic class
obs = [(0, 12.0, 0.01), (1, 11.8, 0.01), (2, 12.0, 0.01)]
star = VariableStar("Test Star", obs)
print(f"Star: {star.name}")
print(f"Mean mag: {star.mean_magnitude:.2f}")
print(f"Amplitude: {star.amplitude:.2f}")
```

#### Part B: Add Analysis Methods (10-15 minutes)

```{code-cell} ipython3
"""
Part B: Add time-series analysis methods
"""

class VariableStar:
    """Enhanced variable star class with analysis."""
    
    def __init__(self, name, observations):
        """Initialize with sorted observations."""
        self.name = name
        self.observations = sorted(observations)
    
    @property
    def mean_magnitude(self):
        """Average magnitude."""
        mags = [obs[1] for obs in self.observations]
        return sum(mags) / len(mags) if mags else 0
    
    @property
    def amplitude(self):
        """Peak-to-peak amplitude."""
        if not self.observations:
            return 0
        mags = [obs[1] for obs in self.observations]
        return max(mags) - min(mags)
    
    @property
    def time_span(self):
        """Total observation time span in days."""
        if len(self.observations) < 2:
            return 0
        times = [obs[0] for obs in self.observations]
        return max(times) - min(times)
    
    def phase_fold(self, period):
        """
        Fold lightcurve at given period (from Chapter 5).
        
        Phase folding: Maps all times to [0,1] based on period.
        Like overlaying multiple periods to see the repeating pattern.
        """
        result = []
        for time, mag, err in self.observations:
            phase = (time % period) / period
            result.append((phase, mag, err))
        return sorted(result)
    
    def find_period_simple(self):
        """
        Estimate period from peak-to-peak intervals.
        Simplified version of Chapter 5's algorithm.
        """
        if len(self.observations) < 3:
            return None
        
        mags = [obs[1] for obs in self.observations]
        times = [obs[0] for obs in self.observations]
        
        # Find brightness peaks
        peak_times = []
        for i in range(1, len(mags)-1):
            if mags[i] < mags[i-1] and mags[i] < mags[i+1]:  # Local minimum (brightest)
                peak_times.append(times[i])
        
        if len(peak_times) < 2:
            return None
        
        # Average intervals between peaks
        intervals = [peak_times[i+1] - peak_times[i] for i in range(len(peak_times)-1)]
        return sum(intervals) / len(intervals) if intervals else None
    
    def __str__(self):
        """String representation with key statistics."""
        return (f"VariableStar({self.name}): "
                f"{len(self.observations)} obs, "
                f"<m>={self.mean_magnitude:.2f}, "
                f"amp={self.amplitude:.2f}")
    
    def __len__(self):
        """Number of observations."""
        return len(self.observations)

# Test with more realistic data
import math
observations = []
for day in range(20):
    time = day + 0.1 * day  # Non-uniform sampling
    phase = 2 * math.pi * time / 5.366  # Delta Cephei period
    magnitude = 4.0 + 0.5 * math.sin(phase)
    error = 0.01
    observations.append((time, magnitude, error))

delta_cep = VariableStar("Delta Cephei", observations)
print(delta_cep)
print(f"Time span: {delta_cep.time_span:.1f} days")
```

#### Part C: Complete Analysis System (15-20 minutes)

```{code-cell} ipython3
"""
Part C: Full variable star analysis system
Synthesizes OOP with concepts from Chapters 3-5
"""

import random
import math

class VariableStarCatalog:
    """
    Manages collection of variable stars.
    Builds on Chapter 4's data structures and Chapter 5's modules.
    """
    
    def __init__(self, name="Catalog"):
        """Initialize empty catalog."""
        self.name = name
        self.stars = {}  # Dictionary for fast lookup
    
    def add_star(self, star):
        """Add star to catalog."""
        self.stars[star.name] = star
    
    def get_star(self, name):
        """Retrieve star by name."""
        return self.stars.get(name, None)
    
    def find_by_amplitude(self, min_amp, max_amp):
        """Find stars with amplitude in range."""
        results = []
        for star in self.stars.values():
            if min_amp <= star.amplitude <= max_amp:
                results.append(star)
        return results
    
    def generate_cepheid(self, name, period=5.4, amplitude=0.3, n_points=50):
        """
        Generate simulated Cepheid variable.
        Uses pattern from Chapter 5's generate_cepheid_data function.
        """
        observations = []
        
        for i in range(n_points):
            # Irregular sampling (Chapter 3 random concepts)
            t = i * period / 10 + random.uniform(-0.1, 0.1)
            
            # Cepheid light curve (asymmetric)
            phase = (t % period) / period
            
            if phase < 0.3:
                # Rising branch (quick brightening)
                mag = 12.0 - amplitude * (phase / 0.3)
            else:
                # Falling branch (slow dimming)
                mag = 12.0 - amplitude * math.exp(-(phase - 0.3) / 0.4)
            
            # Add realistic noise
            mag += random.gauss(0, 0.02)
            error = 0.01 + 0.01 * random.random()
            
            observations.append((t, mag, error))
        
        star = VariableStar(name, observations)
        self.add_star(star)
        return star
    
    def __len__(self):
        """Number of stars in catalog."""
        return len(self.stars)
    
    def __str__(self):
        """Catalog summary."""
        return f"{self.name}: {len(self)} variable stars"
    
    def statistics_summary(self):
        """
        Statistical summary of catalog.
        Uses functional concepts from Chapter 5.
        """
        if not self.stars:
            return "Empty catalog"
        
        amplitudes = [star.amplitude for star in self.stars.values()]
        mean_amp = sum(amplitudes) / len(amplitudes)
        
        # Standard deviation (Chapter 5 pattern)
        variance = sum((a - mean_amp)**2 for a in amplitudes) / len(amplitudes)
        std_amp = variance ** 0.5
        
        return (f"Catalog statistics:\n"
                f"  Stars: {len(self)}\n"
                f"  Mean amplitude: {mean_amp:.3f} mag\n"
                f"  Std amplitude: {std_amp:.3f} mag\n"
                f"  Range: [{min(amplitudes):.3f}, {max(amplitudes):.3f}] mag")

# Create and populate catalog
catalog = VariableStarCatalog("Survey A")

# Generate different types of variables
catalog.generate_cepheid("Cepheid-1", period=5.4, amplitude=0.3)
catalog.generate_cepheid("Cepheid-2", period=10.2, amplitude=0.5)
catalog.generate_cepheid("RR Lyrae-1", period=0.5, amplitude=0.8)

print(catalog)
print("\n" + catalog.statistics_summary())

# Find specific amplitude range
high_amp = catalog.find_by_amplitude(0.4, 1.0)
print(f"\nHigh amplitude stars: {[s.name for s in high_amp]}")

# Demonstrate phase folding (continuing from Chapter 5)
star = catalog.get_star("Cepheid-1")
if star:
    folded = star.phase_fold(5.4)  # Fold at known period
    print(f"\n{star.name} folded: {len(folded)} phase points")
```

### Exercise 3: Performance Comparison

```{code-cell} ipython3
"""
Compare OOP vs functional approaches for performance
Understanding the tradeoffs helps you choose wisely
"""

import time

# OOP Approach
class ParticleOOP:
    def __init__(self, x, y, vx, vy):
        """Initialize particle with position and velocity."""
        self.x = x    # cm
        self.y = y    # cm
        self.vx = vx  # cm/s
        self.vy = vy  # cm/s
    
    def update(self, dt):
        """Update position based on velocity."""
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def energy(self):
        """Calculate kinetic energy in ergs (mass = 1g)."""
        return 0.5 * (self.vx**2 + self.vy**2)

# Functional Approach
def create_particle(x, y, vx, vy):
    """Create particle dictionary."""
    return {'x': x, 'y': y, 'vx': vx, 'vy': vy}

def update_particle(p, dt):
    """Update particle position."""
    p['x'] += p['vx'] * dt
    p['y'] += p['vy'] * dt
    return p

def particle_energy(p):
    """Calculate particle energy."""
    return 0.5 * (p['vx']**2 + p['vy']**2)

# Performance test with moderate sample for demonstration
n = 5000  # Moderate size for notebook execution
dt = 0.01

# OOP timing
start = time.perf_counter()
particles_oop = [ParticleOOP(i, i, 1, 1) for i in range(n)]
for p in particles_oop:
    p.update(dt)
    e = p.energy()
oop_time = time.perf_counter() - start

# Functional timing
start = time.perf_counter()
particles_func = [create_particle(i, i, 1, 1) for i in range(n)]
for p in particles_func:
    update_particle(p, dt)
    e = particle_energy(p)
func_time = time.perf_counter() - start

print(f"OOP approach: {oop_time*1000:.2f} ms")
print(f"Functional approach: {func_time*1000:.2f} ms")
print(f"Ratio: {oop_time/func_time:.2f}x")

print("\nNote: OOP is often slightly slower due to:")
print("- Attribute lookup overhead")
print("- Object creation and memory allocation costs")
print("\nHowever, this performance difference only matters when")
print("creating millions of objects in tight loops. For most")
print("scientific applications, code clarity and maintainability")
print("far outweigh these microsecond differences.")
print("\nFor N-body simulations with millions of particles,")
print("the performance difference DOES matter. That's when")
print("you'll use NumPy arrays (Chapter 7) which combine")
print("OOP interfaces with C-speed operations!")
```

## 7.10 Practice Exercises

### Exercise 1: Photometry Analysis Pipeline

Build a complete photometry analysis system:

**Part A: Basic photometry (5 minutes)**

```{code-cell} ipython3
# Convert instrumental magnitudes to calibrated values
instrumental_mags = np.array([14.2, 15.1, 13.8, 16.3, 14.7])

# Known standard star (for calibration)
standard_instrumental = 14.0
standard_true = 12.5  # Known true magnitude

# Calculate zero point
zero_point = standard_true - standard_instrumental
print(f"Zero point: {zero_point}")

# Calibrate all magnitudes
calibrated_mags = instrumental_mags + zero_point
print(f"Calibrated magnitudes: {calibrated_mags}")

# Convert to flux (relative to mag 0)
fluxes = 10**(-calibrated_mags / 2.5)
print(f"Relative fluxes: {fluxes}")

# Calculate signal-to-noise (assuming Poisson noise)
exposure_time = 300  # seconds
counts = fluxes * 1e6 * exposure_time  # Scale to realistic counts
snr = np.sqrt(counts)
print(f"SNR: {snr}")
```

**Part B: Aperture photometry (10 minutes)**

```{code-cell} ipython3
# Create synthetic star image
x = np.arange(50)
y = np.arange(50)
X, Y = np.meshgrid(x, y)

# Add three stars at different positions
star_x = [15, 35, 25]
star_y = [20, 30, 40]
star_flux = [1000, 500, 750]

image = np.zeros((50, 50))
for sx, sy, sf in zip(star_x, star_y, star_flux):
    # Gaussian PSF for each star
    r_squared = (X - sx)**2 + (Y - sy)**2
    image += sf * np.exp(-r_squared / (2 * 2**2))

# Add noise
np.random.seed(42)
image += np.random.normal(0, 5, image.shape)
```

```{code-cell} ipython3
# Aperture photometry on first star
cx, cy = star_x[0], star_y[0]
aperture_radius = 5

# Create circular mask
distances = np.sqrt((X - cx)**2 + (Y - cy)**2)
aperture_mask = distances <= aperture_radius

# Measure flux
aperture_flux = image[aperture_mask].sum()
aperture_area = aperture_mask.sum()
background = np.median(image[distances > 20])  # Sky background

# Correct for background
corrected_flux = aperture_flux - background * aperture_area
print(f"Aperture flux: {aperture_flux:.1f}")
print(f"Background: {background:.2f} per pixel")
print(f"Corrected flux: {corrected_flux:.1f}")
print(f"True flux: {star_flux[0]}")
print(f"Recovery: {corrected_flux/star_flux[0]*100:.1f}%")
```

**Part C: Light curve analysis (15 minutes)**

```{code-cell} ipython3
# Generate variable star light curve
time = np.linspace(0, 10, 500)  # 10 days, 500 observations
period = 1.7  # days
amplitude = 0.5  # magnitudes

# True light curve
phase = 2 * np.pi * time / period
true_mag = 12.0 + amplitude * np.sin(phase)

# Add realistic noise
np.random.seed(42)
noise = np.random.normal(0, 0.05, len(time))
observed_mag = true_mag + noise

# Period finding using simplified Lomb-Scargle
test_periods = np.linspace(0.5, 5, 1000)
chi_squared = []
```

```{code-cell} ipython3
# Test each period (simplified algorithm)
for test_period in test_periods:
    # Fold data at test period
    test_phase = (time % test_period) / test_period
    
    # Bin the folded light curve
    n_bins = 10
    binned_mags = []
    for i in range(n_bins):
        bin_mask = (test_phase >= i/n_bins) & (test_phase < (i+1)/n_bins)
        if bin_mask.sum() > 0:
            binned_mags.append(observed_mag[bin_mask].mean())
    
    # Calculate scatter (simplified chi-squared)
    if len(binned_mags) > 1:
        chi_squared.append(np.std(binned_mags))
    else:
        chi_squared.append(np.inf)
```

```{code-cell} ipython3
# Find best period
chi_squared = np.array(chi_squared)
best_period = test_periods[chi_squared.argmax()]

print(f"True period: {period} days")
print(f"Found period: {best_period:.3f} days")

# Verify by folding at found period
folded_phase = (time % best_period) / best_period
phase_order = np.argsort(folded_phase)

print(f"Period recovery error: {abs(best_period - period)/period * 100:.1f}%")
```

### Exercise 2: Spectral Analysis

Analyze stellar spectra with NumPy:

```{code-cell} ipython3
"""
Part 1: Generate synthetic spectrum
"""
# Generate wavelength grid
wavelength = np.linspace(4000, 7000, 1000)  # Angstroms

# Continuum (blackbody approximation)
T = 5800  # Solar temperature in K
h = 6.626e-27  # erg*s
c = 2.998e10  # cm/s
k = 1.381e-16  # erg/K
wave_cm = wavelength * 1e-8

# Planck function (simplified)
continuum = 2 * h * c**2 / wave_cm**5
continuum /= np.exp(h * c / (wave_cm * k * T)) - 1
continuum /= continuum.max()  # Normalize

# Add absorption lines
lines = [4861, 6563]  # H-beta, H-alpha
for line_center in lines:
    line_depth = 0.3
    line_width = 5  # Angstroms
    profile = 1 - line_depth * np.exp(-((wavelength - line_center) / line_width)**2)
    continuum *= profile

# Add noise
np.random.seed(42)
spectrum = continuum + np.random.normal(0, 0.02, len(wavelength))
```

```{code-cell} ipython3
"""
Part 2: Measure equivalent widths
"""
for line_center in lines:
    # Select region around line
    region_mask = np.abs(wavelength - line_center) < 20
    wave_region = wavelength[region_mask]
    flux_region = spectrum[region_mask]
    
    # Local continuum (linear fit to edges)
    edge_mask = (np.abs(wave_region - line_center) > 10)
    continuum_fit = np.polyfit(wave_region[edge_mask], 
                               flux_region[edge_mask], 1)
    local_continuum = np.polyval(continuum_fit, wave_region)
    
    # Equivalent width
    normalized = flux_region / local_continuum
    ew = np.trapz(1 - normalized, wave_region)
    
    print(f"Line at {line_center} Å: EW = {ew:.2f} Å")
```

```{code-cell} ipython3
"""
Part 3: Measure radial velocity
"""
# Find line shift
reference_line = 6563  # nm; H-alpha rest wavelength
line_region = np.abs(wavelength - reference_line) < 10
line_flux = spectrum[line_region]
line_wave = wavelength[line_region]

# Find minimum (line center)
observed_center = line_wave[line_flux.argmin()]
shift = observed_center - reference_line
velocity = (shift / reference_line) * 3e5  # km/s

print(f"Observed line center: {observed_center:.2f} Å")
print(f"Radial velocity: {velocity:.1f} km/s")
```

### Exercise 3: Debug This

```{code-cell} ipython3
"""
Debug This! Fixed version with explanations
"""

def analyze_galaxy_redshifts(distances_mpc, velocities_km_s):
    """Calculate Hubble constant from galaxy data."""
    # Original bugs and fixes:
    
    # BUG 1 FIXED: Ensure proper operation order
    # Was: hubble = velocities_km_s / distances_mpc (wrong comment)
    # Fixed: This is actually correct!
    
    # BUG 2 FIXED: Using correct numpy function
    # Was: distances_log = np.log(distances_mpc)  # Wrong base
    # Fixed: Use log10 for astronomical calculations
    distances_log = np.log10(distances_mpc)
    
    # BUG 3 FIXED: Ensure float division
    # Was: normalized = velocities_km_s / velocities_km_s.max()
    # Fixed: Convert to float to avoid integer division issues
    normalized = velocities_km_s.astype(float) / velocities_km_s.max()
    
    # BUG 4 FIXED: Correct axis for 1D array
    # Was: residuals = velocities_km_s - np.mean(velocities_km_s, axis=1)
    # Fixed: No axis needed for 1D array
    residuals = velocities_km_s - np.mean(velocities_km_s)
    
    # Remove outliers for robust estimate
    mask = (velocities_km_s > 0) & (distances_mpc > 0)
    clean_vel = velocities_km_s[mask]
    clean_dist = distances_mpc[mask]
    
    # Calculate Hubble constant
    hubble = clean_vel / clean_dist
    hubble_mean = np.mean(hubble)
    
    return hubble_mean, residuals.std()

# Test data
distances = np.array([10, 20, 30, 40, 50], dtype=float)  # Mpc
velocities = np.array([700, 1400, 2200, 2800, 3500], dtype=float)  # km/s

H0, scatter = analyze_galaxy_redshifts(distances, velocities)
print(f"Hubble constant: {H0:.1f} km/s/Mpc")
print(f"Velocity scatter: {scatter:.1f} km/s")
print(f"(Expected H0 ≈ 70 km/s/Mpc for this simplified data)")
```

:::{admonition} 🌟 Why This Matters: The Hubble Tension
:class: info, important

The code above calculates the Hubble constant, one of cosmology's most important parameters. Different measurement methods give different values (67 vs 73 km/s/Mpc), creating the "Hubble tension" – one of modern cosmology's biggest mysteries. Your NumPy skills are the foundation for analyzing the data that might resolve this cosmic puzzle!
:::

## Chapter 8 Practice Exercises

## 8.9 Practice Exercises

### Exercise 1: Creating a Complete Astronomical Figure

Build a publication-quality multi-wavelength light curve:

**Part A: Generate realistic data (5 minutes)**

```{code-cell} ipython3
# Simulate multi-wavelength observations of a flaring star
np.random.seed(42)

# Time array (days)
time = np.linspace(0, 30, 300)

# Base stellar brightness (different in each band)
base_optical = 12.0  # magnitude
base_xray = 1e-12    # erg/s/cm^2
base_radio = 10.0    # mJy

# Add periodic variation (rotation)
period = 5.3  # days
phase = 2 * np.pi * time / period

optical = base_optical - 0.1 * np.sin(phase)
xray = base_xray * (1 + 0.2 * np.sin(phase + 0.5))
radio = base_radio * (1 + 0.15 * np.sin(phase - 0.3))

# Add flares at specific times
flare_times = [8, 15, 22]
flare_widths = [0.5, 0.3, 0.7]

for ft, fw in zip(flare_times, flare_widths):
    flare_profile = np.exp(-0.5 * ((time - ft) / fw)**2)
    optical -= 0.5 * flare_profile  # Brightening (lower magnitude)
    xray *= (1 + 10 * flare_profile)  # X-ray enhancement
    radio *= (1 + 3 * flare_profile)  # Radio enhancement

# Add realistic noise
optical += np.random.normal(0, 0.02, len(time))
xray *= (1 + np.random.normal(0, 0.1, len(time)))
radio *= (1 + np.random.normal(0, 0.05, len(time)))

print(f"Generated {len(time)} observations over {time.max():.1f} days")
print(f"Detected {len(flare_times)} flares")
```

**Part B: Create the multi-panel figure (10 minutes)**

```{code-cell} python
# Create publication-quality figure
fig = plt.figure(figsize=(10, 8))

# Use GridSpec for custom layout
from matplotlib.gridspec import GridSpec
gs = GridSpec(4, 1, height_ratios=[2, 2, 2, 1], hspace=0)

# Optical light curve
ax1 = fig.add_subplot(gs[0])
ax1.scatter(time, optical, s=10, alpha=0.6, color='blue')
ax1.set_ylabel('V Magnitude', fontsize=11)
ax1.invert_yaxis()  # Astronomical convention
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)
ax1.tick_params(labelbottom=False)

# Mark flares
for ft in flare_times:
    ax1.axvline(x=ft, color='red', linestyle='--', alpha=0.5)
ax1.text(0.02, 0.98, 'Optical', transform=ax1.transAxes,
         fontweight='bold', va='top')

# X-ray light curve (log scale)
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.semilogy(time, xray, 'r-', linewidth=0.8, alpha=0.8)
ax2.set_ylabel('X-ray Flux\n(erg/s/cm²)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelbottom=False)
ax2.text(0.02, 0.98, 'X-ray', transform=ax2.transAxes,
         fontweight='bold', va='top')

# Radio light curve
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(time, radio, 'g-', linewidth=1)
ax3.set_ylabel('Radio Flux\n(mJy)', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelbottom=False)
ax3.text(0.02, 0.98, 'Radio', transform=ax3.transAxes,
         fontweight='bold', va='top')

# Hardness ratio
ax4 = fig.add_subplot(gs[3], sharex=ax1)
hardness = xray / (xray.mean())  # Normalized X-ray
ax4.plot(time, hardness, 'k-', linewidth=0.8)
ax4.set_ylabel('Hardness\nRatio', fontsize=11)
ax4.set_xlabel('Time (days)', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 30)

# Overall title
fig.suptitle('Multi-wavelength Observations of Flare Star', 
             fontsize=14, fontweight='bold', y=0.995)

plt.show()
```

**Part C: Export and document (5 minutes)**

```{code-cell} python
# Create a function to save figures properly
def save_publication_figure(fig, basename, formats=['pdf', 'png']):
    """
    Save figure in multiple formats with proper settings.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    basename : str
        Base filename without extension
    formats : list
        List of formats to save
    """
    for fmt in formats:
        filename = f"{basename}.{fmt}"
        
        if fmt == 'pdf':
            fig.savefig(filename, format='pdf', dpi=300,
                       bbox_inches='tight', transparent=True)
        elif fmt == 'png':
            fig.savefig(filename, format='png', dpi=300,
                       bbox_inches='tight', transparent=False,
                       facecolor='white')
        elif fmt == 'svg':
            fig.savefig(filename, format='svg',
                       bbox_inches='tight', transparent=True)
        
        print(f"Saved: {filename}")
    
    # Also save the data for reproducibility
    data_file = f"{basename}_data.npz"
    np.savez(data_file, time=time, optical=optical, 
             xray=xray, radio=radio)
    print(f"Saved data: {data_file}")

# Example usage (commented out to avoid creating files)
# save_publication_figure(fig, 'flare_star_multiwave')

print("Figure ready for publication!")
print("\nChecklist:")
print("✓ All axes labeled with units")
print("✓ Multi-wavelength data aligned")
print("✓ Flares marked consistently")
print("✓ Professional styling applied")
print("✓ Ready for ApJ submission (7-inch width)")
```

### Exercise 2: Exploring Scaling Effects

Understand how different scales reveal different features:

```{code-cell} python
"""
Part 1: Generate power-law distributed data
"""
np.random.seed(42)
# Simulate galaxy cluster masses (power law distribution)
n_galaxies = 5000
masses = np.random.pareto(1.5, n_galaxies) + 1  # M ∝ N^(-2.5)
masses *= 1e11  # Solar masses

# Add measurement uncertainty
masses_observed = masses * np.random.lognormal(0, 0.1, n_galaxies)

print(f"Generated {n_galaxies} galaxy masses")
print(f"Mass range: {masses_observed.min():.2e} to {masses_observed.max():.2e} M_☉")
```

```{code-cell} python
"""
Part 2: Try different visualizations
"""
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Linear histogram - useless for power law
axes[0, 0].hist(masses_observed, bins=50, alpha=0.7, color='blue')
axes[0, 0].set_xlabel('Mass [M_☉]')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Linear Scale: Useless for Power Law')

# Log-scale x-axis
axes[0, 1].hist(masses_observed, bins=np.logspace(11, 15, 50), 
                alpha=0.7, color='green')
axes[0, 1].set_xscale('log')
axes[0, 1].set_xlabel('Mass [M_☉]')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Log X-axis: Better but not ideal')

# Log-log scale reveals power law
axes[0, 2].hist(masses_observed, bins=np.logspace(11, 15, 50), 
                alpha=0.7, color='red')
axes[0, 2].set_xscale('log')
axes[0, 2].set_yscale('log')
axes[0, 2].set_xlabel('Mass [M_☉]')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_title('Log-Log: Power Law Revealed!')

# Cumulative distribution
sorted_masses = np.sort(masses_observed)
cumulative = np.arange(1, len(sorted_masses) + 1) / len(sorted_masses)

axes[1, 0].plot(sorted_masses, cumulative, 'b-', linewidth=2)
axes[1, 0].set_xlabel('Mass [M_☉]')
axes[1, 0].set_ylabel('Cumulative Fraction')
axes[1, 0].set_title('Linear CDF')

axes[1, 1].loglog(sorted_masses, 1 - cumulative, 'r-', linewidth=2)
axes[1, 1].set_xlabel('Mass [M_☉]')
axes[1, 1].set_ylabel('Fraction > M')
axes[1, 1].set_title('Log-Log Complementary CDF: Slope = Power Law Index')

# Rank-frequency plot
ranks = np.arange(1, len(sorted_masses) + 1)
axes[1, 2].loglog(ranks, sorted_masses[::-1], 'g.', markersize=1)
axes[1, 2].set_xlabel('Rank')
axes[1, 2].set_ylabel('Mass [M_☉]')
axes[1, 2].set_title('Rank-Frequency: Alternative Power Law View')

plt.suptitle('Same Data, Different Scales, Different Insights', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Exercise 3: Debug This!

```{code-cell} python
"""
Debug This! Find and fix the visualization problems.
"""

def plot_hr_diagram_broken(b_v, abs_mag):
    """This function has several plotting issues. Can you fix them?"""
    # BUG 1: Figure too small for publication
    # fig, ax = plt.subplots(figsize=(4, 3))  
    fig, ax = plt.subplots(figsize=(8, 10))  # FIXED: Appropriate size
    
    # BUG 2: Wrong plot type for scattered data
    # ax.plot(b_v, abs_mag, 'b-')  
    ax.scatter(b_v, abs_mag, s=20, alpha=0.6, c=b_v, cmap='RdYlBu_r')  # FIXED
    
    # BUG 3: Y-axis not inverted (astronomical convention)
    # ax.set_ylim(-10, 15)  
    ax.set_ylim(15, -10)  # FIXED: Inverted for magnitudes
    
    # BUG 4: No axis labels or units
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.set_xlabel('B - V Color Index', fontsize=12)  # FIXED
    ax.set_ylabel('Absolute Magnitude (M$_V$)', fontsize=12)  # FIXED
    
    # BUG 5: No title
    ax.set_title('Hertzsprung-Russell Diagram', fontsize=14, fontweight='bold')
    
    # BUG 6: No grid for reference
    ax.grid(True, alpha=0.3)  # FIXED
    
    # BUG 7: Using jet colormap
    # (Already fixed above with RdYlBu_r)
    
    # Add colorbar
    plt.colorbar(ax.collections[0], ax=ax, label='B - V Color')
    
    return fig, ax

# Test with synthetic data
np.random.seed(42)
n_stars = 1000

# Main sequence
b_v_ms = np.random.uniform(-0.3, 2.0, n_stars)
abs_mag_ms = 4.5 * b_v_ms + np.random.normal(0, 1, n_stars) + 2

# Giants branch
b_v_gb = np.random.uniform(0.5, 2.0, 200)
abs_mag_gb = np.random.normal(-1, 1, 200)

# Combine
b_v_all = np.concatenate([b_v_ms, b_v_gb])
abs_mag_all = np.concatenate([abs_mag_ms, abs_mag_gb])

# Create fixed plot
fig, ax = plot_hr_diagram_broken(b_v_all, abs_mag_all)
plt.show()

print("Fixed issues:")
print("✓ Increased figure size for readability")
print("✓ Changed from line to scatter plot")
print("✓ Inverted y-axis (astronomical convention)")
print("✓ Added proper axis labels with units")
print("✓ Added descriptive title")
print("✓ Added grid for reference")
print("✓ Used perceptually uniform colormap")
```

## Chapter 9