# ⚠️ Chapter 11: Performance and Optimization

## Learning Objectives
By the end of this chapter, you will:
- Profile code to identify bottlenecks scientifically
- Vectorize computations for 10-100x speedups
- Use Numba JIT compilation for near-C performance
- Implement parallel processing for multi-core systems
- Optimize memory usage for large astronomical datasets
- Know when to optimize and when to use existing solutions

## 7.1 Profiling: Measure Before Optimizing

### The Golden Rule of Optimization

```python
import time
import numpy as np
import cProfile
import pstats
from line_profiler import LineProfiler

def demonstrate_premature_optimization():
    """
    Knuth's famous quote: "Premature optimization is the root of all evil"
    
    Profile FIRST, optimize SECOND.
    """
    
    # Version 1: Readable but "inefficient"?
    def calculate_distances_readable(coords):
        n = len(coords)
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((coords[i][0] - coords[j][0])**2 + 
                              (coords[i][1] - coords[j][1])**2)
                distances.append(dist)
        return distances
    
    # Version 2: "Optimized" but actually slower!
    def calculate_distances_clever(coords):
        n = len(coords)
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                # "Optimize" by avoiding sqrt for comparison
                dist_sq = (coords[i][0] - coords[j][0])**2 + \
                         (coords[i][1] - coords[j][1])**2
                # But then we need sqrt anyway...
                distances.append(np.sqrt(dist_sq))
        return distances
    
    # Version 3: Actually optimized
    def calculate_distances_vectorized(coords):
        coords = np.array(coords)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        return dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    
    # Test with realistic data
    n_stars = 100
    coords = [(np.random.uniform(0, 100), np.random.uniform(0, 100)) 
               for _ in range(n_stars)]
    
    # Time each version
    start = time.perf_counter()
    d1 = calculate_distances_readable(coords)
    t1 = time.perf_counter() - start
    
    start = time.perf_counter()
    d2 = calculate_distances_clever(coords)
    t2 = time.perf_counter() - start
    
    start = time.perf_counter()
    d3 = calculate_distances_vectorized(coords)
    t3 = time.perf_counter() - start
    
    print(f"Readable version:   {t1*1000:.2f} ms")
    print(f"'Clever' version:   {t2*1000:.2f} ms (no improvement!)")
    print(f"Vectorized version: {t3*1000:.2f} ms ({t1/t3:.1f}x faster)")

demonstrate_premature_optimization()
```

### Using cProfile for Function-Level Profiling

```python
def profile_spectrum_analysis():
    """Profile a realistic spectrum analysis pipeline."""
    
    def load_spectrum(n_points=10000):
        """Simulate loading spectrum data."""
        wavelengths = np.linspace(4000, 7000, n_points)
        fluxes = np.random.randn(n_points) + 100
        # Add some spectral lines
        for line_center in [4861, 6563]:  # H-beta, H-alpha
            fluxes += 50 * np.exp(-(wavelengths - line_center)**2 / 25)
        return wavelengths, fluxes
    
    def remove_continuum(wavelengths, fluxes, window=100):
        """Remove continuum using median filter."""
        from scipy.ndimage import median_filter
        continuum = median_filter(fluxes, size=window)
        return fluxes - continuum
    
    def find_lines(wavelengths, fluxes, threshold=3):
        """Find emission lines."""
        from scipy.signal import find_peaks
        std = np.std(fluxes)
        peaks, properties = find_peaks(fluxes, height=threshold*std)
        return wavelengths[peaks], fluxes[peaks]
    
    def measure_redshift(line_wavelengths, rest_wavelength=6563):
        """Calculate redshift from lines."""
        if len(line_wavelengths) == 0:
            return 0
        # Find closest to expected H-alpha
        closest_idx = np.argmin(np.abs(line_wavelengths - rest_wavelength))
        observed = line_wavelengths[closest_idx]
        return (observed - rest_wavelength) / rest_wavelength
    
    # Profile the complete pipeline
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run analysis
    wavelengths, fluxes = load_spectrum(50000)
    normalized = remove_continuum(wavelengths, fluxes)
    line_waves, line_fluxes = find_lines(wavelengths, normalized)
    redshift = measure_redshift(line_waves)
    
    profiler.disable()
    
    # Print statistics
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\nTop 10 functions by cumulative time:")
    stats.print_stats(10)
    
    return redshift

# z = profile_spectrum_analysis()  # Uncomment to see profiling
```

### Line-by-Line Profiling

```python
# Use line_profiler for detailed analysis
# Install: pip install line_profiler

def detailed_profile_example():
    """Example of line-by-line profiling."""
    
    # Decorate function with @profile (added by line_profiler)
    # Run with: kernprof -l -v script.py
    
    def process_image(image):  # Add @profile decorator
        """Process CCD image - which lines are slow?"""
        # Line 1: Median filter for cosmic ray removal
        from scipy.ndimage import median_filter
        cleaned = median_filter(image, size=3)
        
        # Line 2: Calculate statistics
        mean = np.mean(cleaned)
        std = np.std(cleaned)
        
        # Line 3: Find sources above threshold
        threshold = mean + 5 * std
        sources = cleaned > threshold
        
        # Line 4: Label connected regions
        from scipy.ndimage import label
        labeled, num_sources = label(sources)
        
        # Line 5: Calculate properties for each source
        source_properties = []
        for i in range(1, num_sources + 1):
            mask = labeled == i
            flux = np.sum(cleaned[mask])
            centroid = np.mean(np.argwhere(mask), axis=0)
            source_properties.append({'flux': flux, 'centroid': centroid})
        
        return source_properties
    
    # Simulate CCD image
    image = np.random.randn(1024, 1024) + 100
    # Add some "stars"
    for _ in range(50):
        x, y = np.random.randint(10, 1014, 2)
        image[x-2:x+3, y-2:y+3] += 1000
    
    # This would show time spent on each line
    # sources = process_image(image)
    
    print("Line profiler example ready - add @profile decorator and run with kernprof")

detailed_profile_example()
```

## 7.2 Vectorization: The NumPy Way

### Understanding Why Vectorization is Fast

```python
def vectorization_explanation():
    """Why is NumPy so much faster than pure Python loops?"""
    
    print("Why Vectorization Works:\n")
    print("1. Python loops have overhead:")
    print("   - Type checking each iteration")
    print("   - Function calls for each operation")
    print("   - Memory allocation for intermediate results")
    
    print("\n2. NumPy operations:")
    print("   - Implemented in optimized C")
    print("   - Use CPU vector instructions (SIMD)")
    print("   - Better memory access patterns")
    print("   - No Python overhead in inner loop")
    
    # Demonstration
    n = 1_000_000
    
    # Python list operations
    python_list = list(range(n))
    start = time.perf_counter()
    python_result = [x**2 for x in python_list]
    python_time = time.perf_counter() - start
    
    # NumPy operations
    numpy_array = np.arange(n)
    start = time.perf_counter()
    numpy_result = numpy_array**2
    numpy_time = time.perf_counter() - start
    
    print(f"\nSquaring {n:,} numbers:")
    print(f"Python list: {python_time*1000:.1f} ms")
    print(f"NumPy array: {numpy_time*1000:.1f} ms")
    print(f"Speedup: {python_time/numpy_time:.1f}x")

vectorization_explanation()
```

### Vectorization Patterns for Astronomy

```python
class VectorizedAstronomyCalculations:
    """Common vectorization patterns in astronomy."""
    
    @staticmethod
    def angular_distance_matrix(ra, dec):
        """
        Calculate all pairwise angular distances.
        Vectorized haversine formula.
        """
        # Convert to radians
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        # Use broadcasting to compute all pairs
        # Shape: (n, 1) - (1, n) = (n, n)
        dra = ra_rad[:, np.newaxis] - ra_rad[np.newaxis, :]
        ddec = dec_rad[:, np.newaxis] - dec_rad[np.newaxis, :]
        
        # Haversine formula
        a = np.sin(ddec/2)**2 + \
            np.cos(dec_rad[:, np.newaxis]) * np.cos(dec_rad[np.newaxis, :]) * \
            np.sin(dra/2)**2
        
        c = 2 * np.arcsin(np.sqrt(a))
        return np.degrees(c)
    
    @staticmethod
    def extinction_correction_vectorized(magnitudes, colors, R_v=3.1):
        """
        Apply extinction correction to many stars at once.
        
        Instead of looping over stars, process all simultaneously.
        """
        # Ensure arrays
        magnitudes = np.asarray(magnitudes)
        colors = np.asarray(colors)
        
        # Cardelli extinction law coefficients (simplified)
        a_v = 0.574 * colors  # Simplified color-extinction relation
        
        # Apply to all magnitudes at once
        corrected = magnitudes - R_v * a_v
        
        # Handle edge cases with numpy
        corrected = np.where(colors < 0, magnitudes, corrected)  # No correction for blue colors
        
        return corrected
    
    @staticmethod
    def phase_fold_vectorized(times, fluxes, period):
        """
        Phase fold a light curve - vectorized version.
        """
        # Calculate phase for all times at once
        phases = (times % period) / period
        
        # Sort by phase (vectorized)
        sort_idx = np.argsort(phases)
        
        return phases[sort_idx], fluxes[sort_idx]
    
    @staticmethod
    def match_catalogs_vectorized(ra1, dec1, ra2, dec2, max_sep=1.0):
        """
        Cross-match catalogs using vectorized operations.
        
        More memory intensive but much faster than loops.
        """
        # Convert max separation to radians
        max_sep_rad = np.radians(max_sep / 3600)  # arcsec to radians
        
        # Use KDTree for efficient spatial matching
        from scipy.spatial import cKDTree
        
        # Convert to Cartesian for KDTree
        def radec_to_xyz(ra, dec):
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            x = np.cos(dec_rad) * np.cos(ra_rad)
            y = np.cos(dec_rad) * np.sin(ra_rad)
            z = np.sin(dec_rad)
            return np.column_stack([x, y, z])
        
        xyz1 = radec_to_xyz(ra1, dec1)
        xyz2 = radec_to_xyz(ra2, dec2)
        
        # Build tree and query
        tree = cKDTree(xyz2)
        
        # Vectorized query for all points at once
        # Convert angular separation to 3D distance
        max_3d_dist = 2 * np.sin(max_sep_rad / 2)
        
        distances, indices = tree.query(xyz1, distance_upper_bound=max_3d_dist)
        
        # Valid matches where distance is within bound
        valid = distances < max_3d_dist
        
        matches = []
        for i, (is_valid, idx) in enumerate(zip(valid, indices)):
            if is_valid:
                matches.append((i, idx))
        
        return matches

# Test vectorized calculations
calc = VectorizedAstronomyCalculations()

# Generate test data
n_stars = 1000
ra = np.random.uniform(0, 360, n_stars)
dec = np.random.uniform(-30, 30, n_stars)

# Time angular distance calculation
start = time.perf_counter()
dist_matrix = calc.angular_distance_matrix(ra, dec)
print(f"Angular distance matrix for {n_stars} stars: {time.perf_counter() - start:.3f}s")

# Test extinction correction
magnitudes = np.random.uniform(10, 15, n_stars)
colors = np.random.uniform(-0.5, 2.0, n_stars)
corrected = calc.extinction_correction_vectorized(magnitudes, colors)
print(f"Corrected {n_stars} magnitudes (vectorized)")
```

## 7.3 Numba: JIT Compilation for Python

### Basic Numba Usage

```python
from numba import jit, njit, prange, vectorize
import numba

@njit  # No Python mode, faster
def orbital_integration_numba(x0, v0, mass, dt, n_steps):
    """
    Orbital integration with Numba acceleration.
    
    Compare this to pure Python version!
    """
    G = 6.67430e-11
    
    # Pre-allocate arrays
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    
    positions[0] = x0
    velocities[0] = v0
    
    for i in range(1, n_steps):
        # Current state
        r = positions[i-1]
        v = velocities[i-1]
        
        # Calculate acceleration
        r_mag = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        a = -G * mass * r / r_mag**3
        
        # Leapfrog integration
        v_half = v + 0.5 * dt * a
        positions[i] = r + dt * v_half
        
        # Update acceleration at new position
        r_new = positions[i]
        r_mag_new = np.sqrt(r_new[0]**2 + r_new[1]**2 + r_new[2]**2)
        a_new = -G * mass * r_new / r_mag_new**3
        
        velocities[i] = v_half + 0.5 * dt * a_new
    
    return positions, velocities

def compare_integration_performance():
    """Compare Python vs Numba performance."""
    
    # Pure Python version (simplified)
    def orbital_integration_python(x0, v0, mass, dt, n_steps):
        G = 6.67430e-11
        positions = []
        velocities = []
        
        r = np.array(x0)
        v = np.array(v0)
        
        for _ in range(n_steps):
            positions.append(r.copy())
            velocities.append(v.copy())
            
            r_mag = np.linalg.norm(r)
            a = -G * mass * r / r_mag**3
            
            v = v + dt * a
            r = r + dt * v
        
        return np.array(positions), np.array(velocities)
    
    # Test parameters
    x0 = np.array([1.496e11, 0.0, 0.0])  # 1 AU
    v0 = np.array([0.0, 29780.0, 0.0])   # Earth orbital velocity
    mass = 1.989e30  # Solar mass
    dt = 3600.0  # 1 hour
    n_steps = 10000
    
    # Time Python version
    start = time.perf_counter()
    pos_py, vel_py = orbital_integration_python(x0, v0, mass, dt, n_steps)
    python_time = time.perf_counter() - start
    
    # Time Numba version (first call includes compilation)
    start = time.perf_counter()
    pos_nb, vel_nb = orbital_integration_numba(x0, v0, mass, dt, n_steps)
    numba_time_with_compile = time.perf_counter() - start
    
    # Time Numba version again (already compiled)
    start = time.perf_counter()
    pos_nb, vel_nb = orbital_integration_numba(x0, v0, mass, dt, n_steps)
    numba_time = time.perf_counter() - start
    
    print(f"Integration of {n_steps} steps:")
    print(f"  Pure Python: {python_time:.3f}s")
    print(f"  Numba (with compilation): {numba_time_with_compile:.3f}s")
    print(f"  Numba (compiled): {numba_time:.3f}s")
    print(f"  Speedup: {python_time/numba_time:.1f}x")

compare_integration_performance()
```

### Parallel Computing with Numba

```python
@njit(parallel=True)
def monte_carlo_pi_parallel(n_samples):
    """
    Parallel Monte Carlo calculation of π.
    
    Demonstrates Numba's automatic parallelization.
    """
    count = 0
    
    # prange automatically parallelizes this loop
    for i in prange(n_samples):
        x = np.random.random()
        y = np.random.random()
        
        if x*x + y*y <= 1.0:
            count += 1
    
    return 4.0 * count / n_samples

@njit(parallel=True)
def process_many_spectra(spectra, noise_threshold=3.0):
    """
    Process multiple spectra in parallel.
    
    Each spectrum is processed independently - perfect for parallelization.
    """
    n_spectra, n_points = spectra.shape
    results = np.zeros(n_spectra)
    
    # Process each spectrum in parallel
    for i in prange(n_spectra):
        spectrum = spectra[i]
        
        # Calculate statistics
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        
        # Count significant peaks
        threshold = mean + noise_threshold * std
        n_peaks = 0
        for j in range(1, n_points - 1):
            if spectrum[j] > threshold:
                if spectrum[j] > spectrum[j-1] and spectrum[j] > spectrum[j+1]:
                    n_peaks += 1
        
        results[i] = n_peaks
    
    return results

# Test parallel performance
def test_parallel_speedup():
    """Demonstrate parallel speedup with Numba."""
    
    # Monte Carlo test
    n_samples = 10_000_000
    
    start = time.perf_counter()
    pi_estimate = monte_carlo_pi_parallel(n_samples)
    parallel_time = time.perf_counter() - start
    
    print(f"π estimate: {pi_estimate:.6f}")
    print(f"Time with {numba.config.NUMBA_NUM_THREADS} threads: {parallel_time:.3f}s")
    
    # Spectra processing test
    n_spectra = 1000
    n_points = 2048
    spectra = np.random.randn(n_spectra, n_points) + 100
    
    start = time.perf_counter()
    peak_counts = process_many_spectra(spectra)
    spec_time = time.perf_counter() - start
    
    print(f"\nProcessed {n_spectra} spectra in {spec_time:.3f}s")
    print(f"Average peaks per spectrum: {np.mean(peak_counts):.1f}")

test_parallel_speedup()
```

### Custom NumPy UFuncs with Numba

```python
@vectorize(['float64(float64, float64)'], nopython=True)
def magnitude_addition(mag1, mag2):
    """
    Vectorized function to add astronomical magnitudes.
    
    Works element-wise on arrays, just like NumPy functions.
    """
    flux1 = 10**(-0.4 * mag1)
    flux2 = 10**(-0.4 * mag2)
    total_flux = flux1 + flux2
    return -2.5 * np.log10(total_flux)

@vectorize(['float64(float64, float64, float64)'], nopython=True)
def planck_function_fast(wavelength, temperature, scale):
    """
    Fast Planck function evaluation.
    """
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    
    # Wavelength in meters
    lam = wavelength * 1e-9
    
    # Planck function
    numerator = 2 * h * c**2 / lam**5
    x = h * c / (lam * k * temperature)
    
    # Avoid overflow
    if x > 700:
        return 0.0
    
    denominator = np.exp(x) - 1
    return scale * numerator / denominator

# Test custom ufuncs
def test_custom_ufuncs():
    """Test our custom vectorized functions."""
    
    # Test magnitude addition
    mag_array1 = np.array([10.0, 11.0, 12.0, 13.0])
    mag_array2 = np.array([10.5, 10.5, 10.5, 10.5])
    
    combined = magnitude_addition(mag_array1, mag_array2)
    print("Combined magnitudes:")
    for m1, m2, mc in zip(mag_array1, mag_array2, combined):
        print(f"  {m1:.1f} + {m2:.1f} = {mc:.2f}")
    
    # Test Planck function
    wavelengths = np.linspace(100, 3000, 1000)  # nm
    temps = np.array([3000, 5778, 10000])  # K
    
    print(f"\nPlanck function evaluated at {len(wavelengths)} wavelengths")
    
    for temp in temps:
        start = time.perf_counter()
        spectrum = planck_function_fast(wavelengths, temp, 1.0)
        elapsed = time.perf_counter() - start
        peak_idx = np.argmax(spectrum)
        print(f"  T={temp}K: Peak at {wavelengths[peak_idx]:.0f}nm, "
              f"computed in {elapsed*1000:.2f}ms")

test_custom_ufuncs()
```

## 7.4 Multiprocessing for CPU-Bound Tasks

### Process Pool for Parallel Data Processing

```python
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

def process_single_observation(args):
    """
    Process a single observation file.
    
    This function runs in a separate process.
    """
    filename, params = args
    
    # Simulate loading and processing
    np.random.seed(hash(filename) % 2**32)  # Reproducible randomness
    
    # "Load" data
    data = np.random.randn(1024, 1024) + 1000
    
    # Apply calibration
    dark = np.random.randn(1024, 1024) * 10
    flat = np.ones((1024, 1024)) + np.random.randn(1024, 1024) * 0.1
    
    calibrated = (data - dark) / flat
    
    # Extract photometry
    sources = []
    threshold = np.mean(calibrated) + 5 * np.std(calibrated)
    
    # Find bright pixels (simplified source detection)
    bright_pixels = np.argwhere(calibrated > threshold)
    
    if len(bright_pixels) > 0:
        # Group into sources (simplified)
        n_sources = min(10, len(bright_pixels))
        for i in range(n_sources):
            y, x = bright_pixels[i]
            flux = calibrated[y, x]
            sources.append({
                'file': filename,
                'x': x,
                'y': y,
                'flux': flux
            })
    
    return sources

def parallel_pipeline(filenames, n_processes=None):
    """
    Process multiple observations in parallel.
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"Processing {len(filenames)} files with {n_processes} processes")
    
    # Prepare arguments
    args = [(f, {'threshold': 5.0}) for f in filenames]
    
    # Process in parallel
    start = time.perf_counter()
    
    with Pool(n_processes) as pool:
        # map applies function to each item in parallel
        results = pool.map(process_single_observation, args)
    
    elapsed = time.perf_counter() - start
    
    # Flatten results
    all_sources = []
    for source_list in results:
        all_sources.extend(source_list)
    
    print(f"Found {len(all_sources)} sources in {elapsed:.2f}s")
    print(f"Processing rate: {len(filenames)/elapsed:.1f} files/second")
    
    return all_sources

# Test parallel processing
def test_multiprocessing():
    """Compare serial vs parallel processing."""
    
    # Generate fake filenames
    n_files = 50
    filenames = [f"observation_{i:04d}.fits" for i in range(n_files)]
    
    # Serial processing
    print("Serial processing:")
    start = time.perf_counter()
    serial_results = []
    for filename in filenames:
        sources = process_single_observation((filename, {}))
        serial_results.extend(sources)
    serial_time = time.perf_counter() - start
    print(f"  Time: {serial_time:.2f}s")
    
    # Parallel processing
    print("\nParallel processing:")
    parallel_results = parallel_pipeline(filenames)
    
    # Note: Parallel might be slower for small tasks due to overhead!
    # But scales better for real work

test_multiprocessing()
```

### Shared Memory for Large Arrays

```python
def shared_memory_example():
    """
    Use shared memory to avoid copying large arrays between processes.
    """
    from multiprocessing import shared_memory
    
    def process_shared_chunk(args):
        """Process a chunk of shared array."""
        shm_name, shape, dtype, start_idx, end_idx = args
        
        # Attach to existing shared memory
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        
        # Create numpy array from shared memory
        array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        # Process the chunk (example: apply median filter)
        from scipy.ndimage import median_filter
        chunk = array[start_idx:end_idx]
        filtered = median_filter(chunk, size=3)
        
        # Write back to shared memory
        array[start_idx:end_idx] = filtered
        
        # Clean up
        existing_shm.close()
        
        return f"Processed rows {start_idx}-{end_idx}"
    
    # Create large array in shared memory
    size = (4096, 4096)
    dtype = np.float64
    
    # Create shared memory
    shm = shared_memory.SharedMemory(
        create=True, 
        size=np.prod(size) * np.dtype(dtype).itemsize
    )
    
    # Create numpy array backed by shared memory
    shared_array = np.ndarray(size, dtype=dtype, buffer=shm.buf)
    
    # Initialize with data
    shared_array[:] = np.random.randn(*size) + 1000
    
    print(f"Created shared array: {shared_array.shape}, "
          f"size: {shared_array.nbytes / 1e6:.1f} MB")
    
    # Process in parallel without copying
    n_processes = 4
    chunk_size = size[0] // n_processes
    
    args = []
    for i in range(n_processes):
        start = i * chunk_size
        end = start + chunk_size if i < n_processes - 1 else size[0]
        args.append((shm.name, size, dtype, start, end))
    
    with Pool(n_processes) as pool:
        results = pool.map(process_shared_chunk, args)
    
    print("Processing complete:", results)
    
    # Clean up shared memory
    shm.close()
    shm.unlink()

# shared_memory_example()  # Uncomment to run
print("Shared memory example ready")
```

## 7.5 Memory Optimization

### Memory Profiling and Optimization

```python
def memory_optimization_techniques():
    """Demonstrate memory optimization strategies."""
    
    print("Memory Optimization Techniques:\n")
    
    # 1. Use appropriate dtypes
    print("1. Choose appropriate data types:")
    
    # Bad: Using float64 when not needed
    large_array_64 = np.random.randn(10000, 10000)  # 800 MB
    
    # Good: Use float32 if precision allows
    large_array_32 = np.random.randn(10000, 10000).astype(np.float32)  # 400 MB
    
    # Better: Use int16 for counts
    count_array = np.random.randint(0, 1000, (10000, 10000), dtype=np.int16)  # 200 MB
    
    print(f"  float64: {large_array_64.nbytes / 1e6:.1f} MB")
    print(f"  float32: {large_array_32.nbytes / 1e6:.1f} MB")
    print(f"  int16:   {count_array.nbytes / 1e6:.1f} MB")
    
    # Clean up
    del large_array_64, large_array_32, count_array
    
    # 2. Use views instead of copies
    print("\n2. Use views instead of copies:")
    
    spectrum = np.random.randn(100000)
    
    # Bad: Creates a copy
    blue_region_copy = spectrum[10000:20000].copy()
    
    # Good: Creates a view (no memory duplication)
    blue_region_view = spectrum[10000:20000]
    
    print(f"  Original: {spectrum.nbytes / 1e3:.1f} KB")
    print(f"  Copy uses additional: {blue_region_copy.nbytes / 1e3:.1f} KB")
    print(f"  View uses: 0 KB additional")
    
    # 3. Generator expressions for large datasets
    print("\n3. Use generators for streaming data:")
    
    def load_observations_generator(n_files):
        """Generator: loads one at a time."""
        for i in range(n_files):
            # Simulate loading
            yield np.random.randn(1024, 1024)
    
    def load_observations_list(n_files):
        """List: loads all into memory."""
        return [np.random.randn(1024, 1024) for i in range(n_files)]
    
    # Generator uses constant memory
    gen = load_observations_generator(100)
    print(f"  Generator object size: {sys.getsizeof(gen)} bytes")
    
    # List would use ~800 MB!
    # observations = load_observations_list(100)  # Don't run this!
    
    # 4. Memory mapping for large files
    print("\n4. Memory-mapped files for large datasets:")
    
    # Create a memory-mapped array
    filename = 'large_data.dat'
    shape = (10000, 10000)
    
    # Write
    fp = np.memmap(filename, dtype='float32', mode='w+', shape=shape)
    fp[:] = np.random.randn(*shape)
    del fp  # Flush to disk
    
    # Read without loading into RAM
    data = np.memmap(filename, dtype='float32', mode='r', shape=shape)
    print(f"  Memory-mapped array: {shape}, accessing without loading all")
    
    # Only accessed parts are loaded
    small_section = data[0:100, 0:100]
    print(f"  Accessed section: {small_section.shape}")
    
    # Clean up
    del data
    import os
    os.remove(filename)

memory_optimization_techniques()
```

### Chunked Processing for Large Datasets

```python
class ChunkedProcessor:
    """Process large datasets in manageable chunks."""
    
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
    
    def process_large_catalog(self, filename, n_total):
        """
        Process a large catalog without loading it all.
        
        Simulates reading from a huge file.
        """
        results = []
        
        for chunk_start in range(0, n_total, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, n_total)
            
            # "Load" just this chunk
            chunk_data = self._load_chunk(filename, chunk_start, chunk_end)
            
            # Process the chunk
            chunk_results = self._process_chunk(chunk_data)
            
            # Store results (or write to disk)
            results.extend(chunk_results)
            
            # Clear chunk from memory
            del chunk_data
            
            if chunk_start % (self.chunk_size * 10) == 0:
                print(f"  Processed {chunk_start}/{n_total} objects")
        
        return results
    
    def _load_chunk(self, filename, start, end):
        """Simulate loading a chunk of data."""
        n_objects = end - start
        return {
            'ra': np.random.uniform(0, 360, n_objects),
            'dec': np.random.uniform(-90, 90, n_objects),
            'mag': np.random.uniform(10, 20, n_objects)
        }
    
    def _process_chunk(self, chunk):
        """Process a chunk of objects."""
        # Example: Select bright objects
        mask = chunk['mag'] < 15
        
        bright_objects = []
        for i in np.where(mask)[0]:
            bright_objects.append({
                'ra': chunk['ra'][i],
                'dec': chunk['dec'][i],
                'mag': chunk['mag'][i]
            })
        
        return bright_objects
    
    def streaming_statistics(self, data_generator):
        """
        Calculate statistics on streaming data without storing it all.
        
        Uses Welford's algorithm for numerical stability.
        """
        n = 0
        mean = 0
        M2 = 0
        min_val = float('inf')
        max_val = float('-inf')
        
        for chunk in data_generator:
            for value in chunk:
                n += 1
                delta = value - mean
                mean += delta / n
                delta2 = value - mean
                M2 += delta * delta2
                
                min_val = min(min_val, value)
                max_val = max(max_val, value)
        
        variance = M2 / (n - 1) if n > 1 else 0
        std = np.sqrt(variance)
        
        return {
            'count': n,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val
        }

# Test chunked processing
processor = ChunkedProcessor(chunk_size=5000)

print("Processing large catalog in chunks:")
results = processor.process_large_catalog("huge_catalog.dat", n_total=100000)
print(f"Found {len(results)} bright objects")

# Test streaming statistics
def data_stream():
    """Generate data chunks."""
    for _ in range(100):
        yield np.random.randn(1000) * 10 + 50

print("\nCalculating statistics on streaming data:")
stats = processor.streaming_statistics(data_stream())
print(f"Statistics: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
```

## Try It Yourself

### Exercise 7.1: Optimize Light Curve Analysis
Profile and optimize this light curve analysis code.

```python
def analyze_light_curves_slow(times, fluxes, periods_to_test):
    """
    Slow light curve period analysis.
    
    Your task:
    1. Profile to find bottlenecks
    2. Vectorize the period folding
    3. Add Numba acceleration
    4. Parallelize across multiple light curves
    """
    best_periods = []
    
    for time, flux in zip(times, fluxes):
        chi2_values = []
        
        for period in periods_to_test:
            # Phase fold
            phases = []
            for t in time:
                phase = (t % period) / period
                phases.append(phase)
            
            # Sort by phase
            sorted_indices = sorted(range(len(phases)), key=lambda i: phases[i])
            sorted_flux = [flux[i] for i in sorted_indices]
            
            # Calculate chi-squared (simplified)
            mean_flux = sum(sorted_flux) / len(sorted_flux)
            chi2 = 0
            for f in sorted_flux:
                chi2 += (f - mean_flux) ** 2
            
            chi2_values.append(chi2)
        
        # Find best period (minimum chi2)
        best_idx = chi2_values.index(min(chi2_values))
        best_periods.append(periods_to_test[best_idx])
    
    return best_periods

# Your optimized version:
def analyze_light_curves_fast(times, fluxes, periods_to_test):
    """Your optimized implementation."""
    # Your code here
    pass

# Test data
n_curves = 100
n_points = 1000
n_periods = 100

times = [np.sort(np.random.uniform(0, 100, n_points)) for _ in range(n_curves)]
fluxes = [np.random.randn(n_points) + 100 for _ in range(n_curves)]
periods = np.linspace(0.5, 10, n_periods)

# Compare performance
# slow_results = analyze_light_curves_slow(times[:5], fluxes[:5], periods)
# fast_results = analyze_light_curves_fast(times, fluxes, periods)
```

### Exercise 7.2: Memory-Efficient Spectrum Stack
Implement memory-efficient processing of many spectra.

```python
class SpectrumStack:
    """
    Handle thousands of spectra efficiently.
    
    Requirements:
    - Load spectra on demand, not all at once
    - Compute median spectrum without loading all data
    - Find outlier spectra using streaming statistics
    - Support both in-memory and memory-mapped modes
    """
    
    def __init__(self, mode='memory-mapped'):
        self.mode = mode
        # Your code here
        pass
    
    def add_spectrum(self, wavelengths, flux):
        """Add a spectrum to the stack."""
        # Your code here
        pass
    
    def compute_median_spectrum(self):
        """
        Compute median spectrum across all spectra.
        
        Challenge: Do this without loading all spectra at once!
        """
        # Your code here
        pass
    
    def find_outliers(self, threshold=5):
        """
        Find spectra that deviate significantly from median.
        
        Use streaming algorithm to avoid loading all data.
        """
        # Your code here
        pass
    
    def coadd_spectra(self, weights=None):
        """Coadd all spectra with optional weights."""
        # Your code here
        pass

# Test your implementation
stack = SpectrumStack()

# Add many spectra
for i in range(1000):
    wavelengths = np.linspace(4000, 7000, 3000)
    flux = np.random.randn(3000) + 100
    if i % 100 == 0:  # Add some outliers
        flux += 50
    stack.add_spectrum(wavelengths, flux)

# median = stack.compute_median_spectrum()
# outliers = stack.find_outliers()
# print(f"Found {len(outliers)} outlier spectra")
```

### Exercise 7.3: Parallel Monte Carlo Simulation
Implement parallel Monte Carlo for stellar population synthesis.

```python
def stellar_population_monte_carlo(n_stars, n_realizations, imf_params):
    """
    Monte Carlo simulation of stellar populations.
    
    Your task:
    1. Implement IMF sampling
    2. Add stellar evolution tracks
    3. Parallelize across realizations
    4. Use Numba for the inner loops
    5. Optimize memory usage
    
    Should return statistics about the population.
    """
    # Your code here
    pass

# Target performance:
# - 10,000 stars per population
# - 1,000 realizations
# - Complete in < 10 seconds

# imf_params = {'alpha': -2.35, 'min_mass': 0.08, 'max_mass': 120}
# results = stellar_population_monte_carlo(10000, 1000, imf_params)
```

## Key Takeaways

✅ **Profile before optimizing** - Use cProfile and line_profiler to find real bottlenecks  
✅ **Vectorization first** - NumPy operations are usually fast enough  
✅ **Numba for loops you can't vectorize** - Near C-speed with minimal changes  
✅ **Parallelize embarrassingly parallel problems** - Multiple files, independent calculations  
✅ **Choose the right dtype** - float32 vs float64 can halve memory usage  
✅ **Stream large datasets** - Process in chunks, use generators  
✅ **Memory-map huge files** - Access TB-sized files without loading to RAM  
✅ **Know when to stop** - "Fast enough" is often better than "fastest possible"  

## Next Chapter Preview
Chapter 8 will provide an optional sampler of advanced Python topics including async programming, metaclasses, descriptors, and advanced decorators - choose what's relevant for your projects!