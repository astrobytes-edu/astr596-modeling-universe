# NumPy: The Foundation of Scientific Computing

## Learning Objectives
By the end of this chapter, you will:
- Understand why NumPy is fast and when to use it
- Master array creation, manipulation, and broadcasting
- Use advanced indexing for astronomical data selection
- Apply linear algebra operations to scientific problems
- Optimize memory layout and array operations
- Build complex astronomical calculations efficiently

## Why NumPy? The Foundation of Everything

### The Speed Secret: Contiguous Memory and Vectorization

```python
import numpy as np
import time
import sys

def why_numpy_is_fast():
    """Demonstrate NumPy's performance advantages."""
    
    n = 1_000_000
    
    # Python list - scattered in memory
    python_list = list(range(n))
    start = time.perf_counter()
    python_result = [x**2 + 2*x + 1 for x in python_list]
    python_time = time.perf_counter() - start
    
    # NumPy array - contiguous memory
    numpy_array = np.arange(n)
    start = time.perf_counter()
    numpy_result = numpy_array**2 + 2*numpy_array + 1
    numpy_time = time.perf_counter() - start
    
    print(f"Python list: {python_time*1000:.1f} ms")
    print(f"NumPy array: {numpy_time*1000:.1f} ms")
    print(f"Speedup: {python_time/numpy_time:.1f}x")
    
    # Memory layout matters
    print(f"\nMemory usage:")
    print(f"Python list: {sys.getsizeof(python_list) / 1e6:.1f} MB (+ objects)")
    print(f"NumPy array: {numpy_array.nbytes / 1e6:.1f} MB (contiguous)")

why_numpy_is_fast()
```

### NumPy's Architecture

```python
# NumPy arrays are:
# 1. Homogeneous (all same type)
# 2. Contiguous in memory
# 3. Support vectorized operations
# 4. Interface to optimized C/Fortran libraries (BLAS, LAPACK)

# Understanding array structure
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

print(f"Data type: {arr.dtype}")
print(f"Shape: {arr.shape}")
print(f"Strides: {arr.strides}")  # Bytes to next element
print(f"Contiguous: {arr.flags['C_CONTIGUOUS']}")
print(f"Size: {arr.size} elements")
print(f"Memory: {arr.nbytes} bytes")
```

## Array Creation for Astronomy

### Creating Arrays from Astronomical Data

```python
# Common patterns in astronomy
def create_astronomical_arrays():
    """Common array creation patterns for astronomy."""
    
    # 1. Wavelength grids for spectra
    wavelengths = np.linspace(3000, 8000, 5000)  # Å
    log_wavelengths = np.logspace(np.log10(3000), np.log10(8000), 5000)
    
    # 2. Coordinate grids for images
    nx, ny = 512, 512
    x = np.arange(nx) - nx/2  # Centered coordinates
    y = np.arange(ny) - ny/2
    X, Y = np.meshgrid(x, y)  # 2D coordinate arrays
    R = np.sqrt(X**2 + Y**2)  # Radial distance
    
    # 3. Time series for observations
    mjd_start = 59000.0
    mjd_end = 59365.0
    times = np.arange(mjd_start, mjd_end, 1.0/24)  # Hourly observations
    
    # 4. Empty arrays for results
    photometry = np.zeros((len(times), 5))  # 5 filters
    photometry[:] = np.nan  # Initialize with NaN for missing data
    
    # 5. Structured arrays for catalogs
    dtype = [('id', 'i4'), ('ra', 'f8'), ('dec', 'f8'), 
             ('mag', 'f4'), ('class', 'U10')]
    catalog = np.zeros(1000, dtype=dtype)
    catalog['ra'] = np.random.uniform(0, 360, 1000)
    catalog['dec'] = np.random.uniform(-90, 90, 1000)
    
    return wavelengths, X, Y, R, times, photometry, catalog

waves, X, Y, R, times, phot, cat = create_astronomical_arrays()
print(f"Created {len(times)} time points")
print(f"Catalog sample: {cat[:3]}")
```

### Special Array Initializations

```python
# Arrays with specific patterns
def special_arrays():
    """Create arrays with special patterns."""
    
    # Identity matrix for transformations
    transform = np.eye(3)
    
    # Diagonal matrix for error propagation
    errors = [0.1, 0.2, 0.15]
    covariance = np.diag(errors)**2
    
    # Vandermonde matrix for polynomial fitting
    x = np.array([1, 2, 3, 4, 5])
    vander = np.vander(x, 3)  # For quadratic fit
    
    # Block matrices for multi-component models
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    block = np.block([[A, np.zeros((2, 2))],
                      [np.zeros((2, 2)), B]])
    
    print(f"Vandermonde for polynomial fitting:\n{vander}")
    print(f"\nBlock matrix:\n{block}")

special_arrays()
```

## Broadcasting: The Heart of NumPy

### Understanding Broadcasting Rules

```python
def broadcasting_examples():
    """Demonstrate broadcasting in astronomical contexts."""
    
    # Rule 1: Dimensions are aligned right to left
    # Rule 2: Dimensions of size 1 stretch to match
    
    # Example 1: Subtracting sky background from image
    image = np.random.randn(100, 100) + 100
    sky_per_column = np.median(image, axis=0)  # Shape: (100,)
    
    # Broadcasting: (100, 100) - (100,) → (100, 100) - (1, 100) → (100, 100)
    sky_subtracted = image - sky_per_column
    
    print(f"Image shape: {image.shape}")
    print(f"Sky shape: {sky_per_column.shape}")
    print(f"Result shape: {sky_subtracted.shape}")
    
    # Example 2: Distance matrix between stars
    n_stars = 5
    ra = np.random.uniform(0, 10, n_stars)
    dec = np.random.uniform(-5, 5, n_stars)
    
    # Broadcasting to get all pairwise differences
    # (n, 1) - (1, n) → (n, n)
    dra = ra[:, np.newaxis] - ra[np.newaxis, :]
    ddec = dec[:, np.newaxis] - dec[np.newaxis, :]
    distances = np.sqrt(dra**2 + ddec**2)
    
    print(f"\nDistance matrix shape: {distances.shape}")
    print(f"Distance from star 0 to others: {distances[0, :]}")
    
    # Example 3: Applying extinction to multiple filters
    wavelengths = np.array([4400, 5500, 6500, 7900])  # BVRI
    A_V = np.array([0.1, 0.5, 1.0, 2.0])  # Different extinctions
    
    # Cardelli extinction law (simplified)
    k_lambda = 1.0 + 0.17699 * (1 - 5500/wavelengths)
    
    # Broadcasting: (4,) × (4, 1) → (4, 4)
    extinction_matrix = k_lambda * A_V[:, np.newaxis]
    
    print(f"\nExtinction matrix (A_V × filters):\n{extinction_matrix}")

broadcasting_examples()
```

### Advanced Broadcasting Patterns

```python
def advanced_broadcasting():
    """Complex broadcasting for scientific calculations."""
    
    # PSF convolution across multiple images
    n_images, ny, nx = 10, 64, 64
    images = np.random.randn(n_images, ny, nx)
    
    # Create Gaussian PSF
    x = np.arange(nx) - nx/2
    y = np.arange(ny) - ny/2
    X, Y = np.meshgrid(x, y)
    sigma = 2.0
    psf = np.exp(-(X**2 + Y**2) / (2*sigma**2))
    psf /= psf.sum()
    
    # Apply PSF to all images using FFT (broadcasting)
    images_fft = np.fft.fft2(images)  # Shape: (10, 64, 64)
    psf_fft = np.fft.fft2(psf)  # Shape: (64, 64)
    
    # Broadcasting in Fourier space
    convolved_fft = images_fft * psf_fft  # (10, 64, 64) * (64, 64)
    convolved = np.fft.ifft2(convolved_fft).real
    
    print(f"Convolved {n_images} images with PSF")
    print(f"Original shape: {images.shape}")
    print(f"PSF shape: {psf.shape}")
    print(f"Result shape: {convolved.shape}")

advanced_broadcasting()
```

## Advanced Indexing and Selection

### Boolean and Fancy Indexing

```python
def advanced_indexing():
    """Advanced indexing for astronomical data selection."""
    
    # Create sample catalog
    n_objects = 1000
    catalog = {
        'ra': np.random.uniform(0, 360, n_objects),
        'dec': np.random.uniform(-30, 30, n_objects),
        'mag': np.random.uniform(10, 20, n_objects),
        'color': np.random.normal(0.5, 0.3, n_objects),
        'type': np.random.choice(['star', 'galaxy', 'qso'], n_objects)
    }
    
    # Boolean indexing - select bright blue objects
    is_bright = catalog['mag'] < 15
    is_blue = catalog['color'] < 0.3
    blue_bright = is_bright & is_blue
    
    selected_ra = catalog['ra'][blue_bright]
    selected_dec = catalog['dec'][blue_bright]
    
    print(f"Found {blue_bright.sum()} bright blue objects")
    
    # Fancy indexing - select specific indices
    interesting_indices = [10, 50, 100, 150, 200]
    subset = catalog['mag'][interesting_indices]
    
    # Advanced: select objects in specific sky region
    def in_region(ra, dec, ra_center, dec_center, radius):
        """Check if objects are within radius of center."""
        # Simple approximation for small fields
        cos_dec = np.cos(np.radians(dec_center))
        dra = (ra - ra_center) * cos_dec
        ddec = dec - dec_center
        return np.sqrt(dra**2 + ddec**2) < radius
    
    in_field = in_region(catalog['ra'], catalog['dec'], 150, 0, 5)
    field_objects = {k: v[in_field] for k, v in catalog.items()}
    
    print(f"Found {in_field.sum()} objects in 5° field")
    
    # Combined conditions with where
    magnitude_bins = [10, 12, 14, 16, 18, 20]
    bin_indices = np.digitize(catalog['mag'], magnitude_bins)
    
    # Count objects in each magnitude bin
    for i in range(len(magnitude_bins)-1):
        count = np.sum(bin_indices == i+1)
        print(f"  Mag {magnitude_bins[i]}-{magnitude_bins[i+1]}: {count} objects")

advanced_indexing()
```

### Structured Arrays for Catalogs

```python
def structured_arrays():
    """Use structured arrays for astronomical catalogs."""
    
    # Define complex catalog structure
    dtype = np.dtype([
        ('id', 'i8'),
        ('coords', [('ra', 'f8'), ('dec', 'f8')]),
        ('photometry', [('u', 'f4'), ('g', 'f4'), ('r', 'f4'), 
                        ('i', 'f4'), ('z', 'f4')]),
        ('morphology', [('sersic_n', 'f4'), ('r_eff', 'f4')]),
        ('redshift', 'f4'),
        ('classification', 'U10')
    ])
    
    # Create catalog
    n_galaxies = 100
    galaxies = np.zeros(n_galaxies, dtype=dtype)
    
    # Fill with data
    galaxies['id'] = np.arange(n_galaxies)
    galaxies['coords']['ra'] = np.random.uniform(0, 360, n_galaxies)
    galaxies['coords']['dec'] = np.random.uniform(-30, 30, n_galaxies)
    
    # Simulate photometry
    for band in ['u', 'g', 'r', 'i', 'z']:
        galaxies['photometry'][band] = np.random.uniform(18, 25, n_galaxies)
    
    galaxies['morphology']['sersic_n'] = np.random.uniform(0.5, 4, n_galaxies)
    galaxies['morphology']['r_eff'] = np.random.lognormal(1, 0.5, n_galaxies)
    galaxies['redshift'] = np.random.uniform(0, 2, n_galaxies)
    
    # Access nested fields
    colors_gr = galaxies['photometry']['g'] - galaxies['photometry']['r']
    high_z = galaxies[galaxies['redshift'] > 1.0]
    
    print(f"Catalog with {len(galaxies)} galaxies")
    print(f"Mean g-r color: {colors_gr.mean():.2f}")
    print(f"High-z galaxies: {len(high_z)}")
    print(f"\nFirst galaxy:\n{galaxies[0]}")

structured_arrays()
```

## Linear Algebra for Astronomy

### Essential Linear Algebra Operations

```python
def linear_algebra_astronomy():
    """Linear algebra operations in astronomical contexts."""
    
    # 1. Coordinate transformations
    def rotation_matrix_3d(angle, axis='z'):
        """Create 3D rotation matrix."""
        c, s = np.cos(angle), np.sin(angle)
        if axis == 'z':
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        elif axis == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:  # x
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    # Transform coordinates
    coords = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    angle = np.radians(30)
    R = rotation_matrix_3d(angle, 'z')
    rotated = R @ coords  # Matrix multiplication
    
    print(f"Rotation matrix:\n{R}")
    print(f"Rotated coordinates:\n{rotated}")
    
    # 2. Solving linear systems (spectral unmixing)
    # Ax = b where A is mixing matrix, x is abundances, b is observed
    
    # Stellar spectra templates (simplified)
    n_wavelengths = 100
    A_template = np.random.randn(n_wavelengths) + 10  # A-type star
    G_template = np.random.randn(n_wavelengths) + 8   # G-type star
    M_template = np.random.randn(n_wavelengths) + 6   # M-type star
    
    # Mixing matrix
    A = np.column_stack([A_template, G_template, M_template])
    
    # Observed composite spectrum
    true_fractions = np.array([0.2, 0.5, 0.3])
    observed = A @ true_fractions + np.random.randn(n_wavelengths) * 0.1
    
    # Solve for fractions
    fractions, residuals, rank, s = np.linalg.lstsq(A, observed, rcond=None)
    
    print(f"\nTrue fractions: {true_fractions}")
    print(f"Recovered fractions: {fractions}")
    print(f"Residual: {residuals[0]:.4f}")
    
    # 3. Eigenvalue problems (PCA for spectra)
    n_spectra = 50
    spectra = np.random.randn(n_spectra, n_wavelengths) + 10
    
    # Center the data
    mean_spectrum = spectra.mean(axis=0)
    centered = spectra - mean_spectrum
    
    # Covariance matrix
    cov = centered.T @ centered / (n_spectra - 1)
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Sort by eigenvalue
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Variance explained
    var_explained = eigenvals / eigenvals.sum()
    print(f"\nPCA variance explained by first 3 components: "
          f"{var_explained[:3] * 100}")
    
    # 4. SVD for image compression
    image = np.random.randn(64, 64) + 100
    U, s, Vt = np.linalg.svd(image)
    
    # Reconstruct with fewer components
    n_components = 10
    reconstructed = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
    
    compression_ratio = image.size / (n_components * (U.shape[0] + Vt.shape[1] + 1))
    print(f"\nImage compression: {compression_ratio:.1f}x")
    print(f"Reconstruction error: {np.mean((image - reconstructed)**2):.4f}")

linear_algebra_astronomy()
```

## Memory and Performance Optimization

### Memory Layout and Views

```python
def memory_optimization():
    """Optimize memory usage and access patterns."""
    
    # C vs Fortran order
    shape = (1000, 1000)
    c_array = np.zeros(shape, order='C')  # Row-major (default)
    f_array = np.zeros(shape, order='F')  # Column-major
    
    # Check memory layout
    print(f"C-order strides: {c_array.strides}")
    print(f"F-order strides: {f_array.strides}")
    
    # Performance difference for different access patterns
    import time
    
    # Row-wise access (faster for C-order)
    start = time.perf_counter()
    for i in range(shape[0]):
        _ = c_array[i, :].sum()
    c_row_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for i in range(shape[0]):
        _ = f_array[i, :].sum()
    f_row_time = time.perf_counter() - start
    
    print(f"\nRow-wise access:")
    print(f"  C-order: {c_row_time*1000:.2f} ms")
    print(f"  F-order: {f_row_time*1000:.2f} ms")
    
    # Views vs copies
    large_array = np.random.randn(10000, 10000)
    
    # View - no memory copy
    view = large_array[::2, ::2]  # Every other element
    print(f"\nView shares memory: {np.shares_memory(large_array, view)}")
    
    # Copy - new memory
    copy = large_array[::2, ::2].copy()
    print(f"Copy shares memory: {np.shares_memory(large_array, copy)}")
    
    # Memory usage
    print(f"Original size: {large_array.nbytes / 1e9:.2f} GB")
    print(f"View adds: 0 GB")
    print(f"Copy adds: {copy.nbytes / 1e9:.2f} GB")
    
    # In-place operations
    data = np.random.randn(1000000)
    
    # Not in-place (creates temporary)
    result1 = data * 2 + 3
    
    # In-place (memory efficient)
    result2 = data.copy()
    result2 *= 2
    result2 += 3
    
    # Using out parameter
    result3 = np.empty_like(data)
    np.multiply(data, 2, out=result3)
    np.add(result3, 3, out=result3)
    
    print(f"\nResults identical: {np.allclose(result1, result2) and np.allclose(result2, result3)}")

memory_optimization()
```

## Random Number Generation for Simulations

### Astronomical Simulations with Random Numbers

```python
def random_astronomy():
    """Random number generation for astronomical simulations."""
    
    # Create random generator with seed for reproducibility
    rng = np.random.default_rng(42)
    
    # 1. Initial Mass Function (IMF) sampling
    def sample_imf(n_stars, alpha=-2.35, m_min=0.08, m_max=120):
        """Sample from Salpeter/Kroupa IMF using inverse transform."""
        if alpha == -1:
            # Special case
            norm = np.log(m_max/m_min)
        else:
            norm = (m_max**(alpha+1) - m_min**(alpha+1)) / (alpha+1)
        
        u = rng.uniform(0, 1, n_stars)
        if alpha == -1:
            masses = m_min * (m_max/m_min)**u
        else:
            masses = (m_min**(alpha+1) + u * norm * (alpha+1))**(1/(alpha+1))
        
        return masses
    
    masses = sample_imf(10000)
    print(f"IMF: {len(masses)} stars, mean mass = {masses.mean():.2f} M☉")
    
    # 2. Poisson noise for photon counting
    expected_counts = np.array([100, 1000, 10000])
    observed_counts = rng.poisson(expected_counts)
    snr = expected_counts / np.sqrt(expected_counts)
    
    print(f"\nPhoton noise:")
    for exp, obs, s in zip(expected_counts, observed_counts, snr):
        print(f"  Expected: {exp}, Observed: {obs}, SNR: {s:.1f}")
    
    # 3. Gaussian noise for measurements
    true_magnitude = 15.0
    n_observations = 100
    photometric_error = 0.05
    
    observed_mags = rng.normal(true_magnitude, photometric_error, n_observations)
    mean_mag = observed_mags.mean()
    error_on_mean = observed_mags.std() / np.sqrt(n_observations)
    
    print(f"\nPhotometry: {mean_mag:.3f} ± {error_on_mean:.3f} (true: {true_magnitude})")
    
    # 4. Monte Carlo for error propagation
    def distance_from_parallax(parallax_mas, parallax_error_mas, n_samples=10000):
        """Calculate distance with uncertainty using Monte Carlo."""
        # Sample parallax measurements
        parallax_samples = rng.normal(parallax_mas, parallax_error_mas, n_samples)
        
        # Remove negative parallaxes (unphysical)
        parallax_samples = parallax_samples[parallax_samples > 0]
        
        # Calculate distances
        distances = 1000 / parallax_samples  # parsecs
        
        # Statistics
        d_median = np.median(distances)
        d_16, d_84 = np.percentile(distances, [16, 84])
        
        return d_median, d_median - d_16, d_84 - d_median
    
    d, d_minus, d_plus = distance_from_parallax(2.0, 0.3)  # 2±0.3 mas
    print(f"\nDistance: {d:.1f} +{d_plus:.1f} -{d_minus:.1f} pc")

random_astronomy()
```

## Try It Yourself

### Exercise 1: Build an Efficient Image Processing Pipeline

```python
def process_astronomical_images(images, dark, flat, cosmic_ray_threshold=5):
    """
    Process CCD images with calibration and cosmic ray removal.
    
    Parameters
    ----------
    images : ndarray, shape (n_images, ny, nx)
        Raw images
    dark : ndarray, shape (ny, nx)
        Dark frame
    flat : ndarray, shape (ny, nx)
        Flat field
    cosmic_ray_threshold : float
        Sigma threshold for cosmic ray detection
    
    Returns
    -------
    processed : ndarray
        Calibrated images
    
    Your task:
    1. Subtract dark frame
    2. Divide by normalized flat
    3. Detect and remove cosmic rays using median filtering
    4. Use broadcasting efficiently
    5. Minimize memory usage
    """
    # Your code here
    pass

# Test data
n_images, ny, nx = 10, 512, 512
images = np.random.randn(n_images, ny, nx) + 1000
dark = np.random.randn(ny, nx) * 10 + 100
flat = np.ones((ny, nx)) + np.random.randn(ny, nx) * 0.1

# processed = process_astronomical_images(images, dark, flat)
```

### Exercise 2: Spectral Cross-Correlation

```python
def cross_correlate_spectra(spectrum, template, velocity_range=(-500, 500)):
    """
    Cross-correlate spectrum with template to find radial velocity.
    
    Parameters
    ----------
    spectrum : ndarray
        Observed spectrum (wavelength, flux)
    template : ndarray
        Template spectrum (same wavelength grid)
    velocity_range : tuple
        Min and max velocity to search (km/s)
    
    Returns
    -------
    velocity : float
        Best-fit radial velocity
    correlation : ndarray
        Cross-correlation function
    
    Your task:
    1. Implement cross-correlation using FFT
    2. Convert pixel shifts to velocity
    3. Find peak in correlation function
    4. Estimate uncertainty from peak width
    """
    # Your code here
    pass
```

### Exercise 3: Efficient Catalog Matching

```python
def match_catalogs_kdtree(cat1_ra, cat1_dec, cat2_ra, cat2_dec, max_sep_arcsec=1.0):
    """
    Match two catalogs using KD-tree for efficiency.
    
    Your task:
    1. Convert RA/Dec to Cartesian coordinates
    2. Build KD-tree for catalog 2
    3. Query for nearest neighbors
    4. Return matched indices and separations
    5. Handle edge cases (no matches, multiple matches)
    """
    # Your code here
    pass
```

## Key Takeaways

✅ **NumPy is fast because** of contiguous memory and vectorized operations  
✅ **Broadcasting** eliminates explicit loops - align dimensions from right  
✅ **Advanced indexing** enables complex data selection without copies  
✅ **Structured arrays** perfect for astronomical catalogs  
✅ **Linear algebra** operations are highly optimized  
✅ **Memory layout matters** - use views, not copies when possible  
✅ **Random numbers** essential for simulations and error analysis  
✅ **Always vectorize** - if you're writing loops, think again!  

## Next Chapter Preview
Matplotlib: Creating publication-quality astronomical visualizations, moving beyond simple plots to professional figures.