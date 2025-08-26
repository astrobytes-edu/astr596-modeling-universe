# ⚠️ Chapter 13: Pandas - Python Data Analysis Library

## Learning Objectives

By the end of this chapter, you will:

- Master DataFrame operations for astronomical catalogs
- Perform efficient time series analysis on light curves
- Execute complex GroupBy operations for population studies
- Merge and join datasets from multiple surveys
- Prepare data for machine learning pipelines
- Handle missing data and outliers systematically
- Scale to large astronomical surveys with optimization techniques

## Introduction: Why Pandas for Astronomy?

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
sns.set_style('whitegrid')

def why_pandas_for_astronomy():
    """Demonstrate Pandas' value for astronomical data science."""
    
    print("Pandas bridges astronomy and data science:")
    print("\n1. CATALOG MANAGEMENT")
    print("   - Millions of sources from surveys (SDSS, Gaia, LSST)")
    print("   - Heterogeneous data types (positions, magnitudes, spectra)")
    print("   - Missing values and quality flags")
    
    print("\n2. TIME SERIES ANALYSIS")
    print("   - Irregular sampling from ground-based telescopes")
    print("   - Multi-band light curves")
    print("   - Automated classification of variables")
    
    print("\n3. DATA SCIENCE WORKFLOW")
    print("   - Exploratory data analysis (EDA)")
    print("   - Feature engineering for ML")
    print("   - Statistical summaries by groups")
    
    print("\n4. MACHINE LEARNING PREPARATION")
    print("   - Data cleaning and normalization")
    print("   - Feature extraction")
    print("   - Train/test splitting while preserving groups")
    
    # Quick example: Load and explore a galaxy catalog
    np.random.seed(42)
    n_galaxies = 10000
    
    catalog = pd.DataFrame({
        'galaxy_id': np.arange(n_galaxies),
        'ra': np.random.uniform(0, 360, n_galaxies),
        'dec': np.random.uniform(-90, 90, n_galaxies),
        'redshift': np.random.gamma(2, 0.5, n_galaxies),
        'magnitude_r': np.random.normal(20, 2, n_galaxies),
        'magnitude_g': np.random.normal(21, 2, n_galaxies),
        'stellar_mass': 10**np.random.normal(10, 0.5, n_galaxies),
        'morphology': np.random.choice(['E', 'S0', 'Sa', 'Sb', 'Sc', 'Irr'], n_galaxies),
        'survey': np.random.choice(['SDSS', 'DECALS', 'HSC'], n_galaxies, p=[0.5, 0.3, 0.2])
    })
    
    print(f"\n5. EXAMPLE CATALOG OPERATIONS:")
    print(f"   Shape: {catalog.shape}")
    print(f"   Memory usage: {catalog.memory_usage().sum() / 1e6:.1f} MB")
    print(f"   Surveys: {catalog['survey'].value_counts().to_dict()}")
    print(f"   Redshift range: {catalog['redshift'].min():.2f} - {catalog['redshift'].max():.2f}")
    
    return catalog

catalog = why_pandas_for_astronomy()
```

## DataFrames for Astronomical Catalogs

### Creating and Loading Catalogs

```python
def catalog_operations():
    """Essential DataFrame operations for astronomical catalogs."""
    
    # Create a realistic stellar catalog
    n_stars = 5000
    
    # Generate synthetic Gaia-like data
    stars = pd.DataFrame({
        'source_id': np.arange(1000000, 1000000 + n_stars),
        'ra': np.random.uniform(120, 140, n_stars),  # Limited sky region
        'dec': np.random.uniform(-30, -10, n_stars),
        'parallax': np.random.gamma(2, 0.5, n_stars),  # mas
        'parallax_error': np.random.gamma(1, 0.05, n_stars),
        'pmra': np.random.normal(0, 5, n_stars),  # mas/yr
        'pmdec': np.random.normal(0, 5, n_stars),  # mas/yr
        'phot_g_mean_mag': np.random.normal(15, 2, n_stars),
        'phot_bp_mean_mag': np.random.normal(15.5, 2, n_stars),
        'phot_rp_mean_mag': np.random.normal(14.5, 2, n_stars),
        'radial_velocity': np.random.normal(0, 30, n_stars),
        'teff_val': np.random.normal(5500, 1000, n_stars),
        'logg_val': np.random.normal(4.5, 0.5, n_stars),
        'fe_h': np.random.normal(0, 0.3, n_stars),
    })
    
    # Add some missing values (realistic)
    rv_missing = np.random.random(n_stars) > 0.3  # 70% have RV
    stars.loc[rv_missing, 'radial_velocity'] = np.nan
    
    print("1. BASIC CATALOG INFO:")
    print(stars.info())
    
    print("\n2. STATISTICAL SUMMARY:")
    print(stars[['parallax', 'phot_g_mean_mag', 'teff_val']].describe())
    
    # Add derived columns
    print("\n3. ADDING DERIVED QUANTITIES:")
    
    # Distance from parallax
    stars['distance_pc'] = 1000 / stars['parallax']
    
    # Absolute magnitude
    stars['abs_mag_g'] = stars['phot_g_mean_mag'] - 5 * np.log10(stars['distance_pc']) + 5
    
    # Color indices
    stars['bp_rp'] = stars['phot_bp_mean_mag'] - stars['phot_rp_mean_mag']
    stars['g_rp'] = stars['phot_g_mean_mag'] - stars['phot_rp_mean_mag']
    
    # Quality flags
    stars['high_quality'] = (
        (stars['parallax_error'] / stars['parallax'] < 0.1) &  # Good parallax
        (stars['parallax'] > 0) &  # Positive parallax
        (stars['phot_g_mean_mag'] < 18)  # Bright enough
    )
    
    print(f"High quality stars: {stars['high_quality'].sum()} / {len(stars)}")
    
    # Efficient selection
    print("\n4. EFFICIENT DATA SELECTION:")
    
    # Method 1: Boolean indexing
    nearby = stars[stars['distance_pc'] < 100]
    print(f"Stars within 100 pc: {len(nearby)}")
    
    # Method 2: Query method (more readable)
    bright_nearby = stars.query('distance_pc < 100 & phot_g_mean_mag < 10')
    print(f"Bright nearby stars: {len(bright_nearby)}")
    
    # Method 3: loc for complex conditions
    solar_type = stars.loc[
        (stars['teff_val'].between(5300, 6000)) &
        (stars['logg_val'].between(4.0, 4.6)) &
        (stars['fe_h'].between(-0.1, 0.1))
    ]
    print(f"Solar-type stars: {len(solar_type)}")
    
    return stars

stars_df = catalog_operations()
```

### Advanced Indexing and MultiIndex

```python
def advanced_indexing():
    """Advanced indexing techniques for hierarchical astronomical data."""
    
    # Create multi-band photometry catalog
    n_objects = 1000
    n_epochs = 5
    
    # Generate hierarchical data
    data = []
    for obj_id in range(n_objects):
        for epoch in range(n_epochs):
            for band in ['u', 'g', 'r', 'i', 'z']:
                mjd = 58000 + epoch * 30 + np.random.uniform(-1, 1)
                mag = np.random.normal(20 + ord(band) * 0.01, 0.1)
                err = np.random.uniform(0.01, 0.05)
                
                data.append({
                    'object_id': obj_id,
                    'mjd': mjd,
                    'band': band,
                    'magnitude': mag,
                    'error': err
                })
    
    photometry = pd.DataFrame(data)
    
    # Create MultiIndex
    photometry_indexed = photometry.set_index(['object_id', 'mjd', 'band'])
    
    print("1. MULTIINDEX STRUCTURE:")
    print(photometry_indexed.head(10))
    
    # Access specific levels
    print("\n2. ACCESSING DATA:")
    
    # Single object, all epochs
    obj_0 = photometry_indexed.loc[0]
    print(f"Object 0 measurements: {len(obj_0)}")
    
    # Cross-section: all g-band measurements
    g_band = photometry_indexed.xs('g', level='band')
    print(f"G-band measurements: {len(g_band)}")
    
    # Pivot for analysis
    print("\n3. PIVOTING FOR ANALYSIS:")
    
    # Wide format for color analysis
    colors = photometry.pivot_table(
        values='magnitude',
        index=['object_id', 'mjd'],
        columns='band',
        aggfunc='mean'
    )
    
    # Calculate colors
    colors['g-r'] = colors['g'] - colors['r']
    colors['r-i'] = colors['r'] - colors['i']
    
    print(colors.head())
    
    # Hierarchical grouping
    print("\n4. HIERARCHICAL GROUPING:")
    
    # Mean magnitude per object per band
    mean_mags = photometry.groupby(['object_id', 'band'])['magnitude'].agg(['mean', 'std'])
    print(mean_mags.head(10))
    
    return photometry, colors

photometry_df, colors_df = advanced_indexing()
```

## Time Series Analysis for Astronomy

### Light Curve Analysis

```python
def time_series_astronomy():
    """Time series analysis for variable stars and transients."""
    
    # Generate realistic variable star light curve
    np.random.seed(42)
    
    # RR Lyrae-like variable
    true_period = 0.5427  # days
    amplitude = 0.8
    
    # Irregular sampling (ground-based reality)
    n_nights = 150
    observations = []
    
    for night in range(n_nights):
        # Weather: 30% chance of no observation
        if np.random.random() < 0.3:
            continue
            
        # Multiple observations per night
        n_obs = np.random.poisson(3)
        for _ in range(n_obs):
            mjd = 58000 + night + np.random.uniform(0, 0.3)
            phase = (mjd % true_period) / true_period
            
            # RR Lyrae-like light curve shape
            mag = 14.5 - amplitude * (0.5 * np.sin(2*np.pi*phase) + 
                                      0.3 * np.sin(4*np.pi*phase) +
                                      0.1 * np.sin(6*np.pi*phase))
            
            # Add noise
            mag += np.random.normal(0, 0.02)
            error = np.random.uniform(0.015, 0.025)
            
            observations.append({
                'mjd': mjd,
                'magnitude': mag,
                'error': error,
                'band': 'V'
            })
    
    # Create DataFrame
    lightcurve = pd.DataFrame(observations)
    lightcurve['datetime'] = pd.to_datetime(lightcurve['mjd'] - 40587, unit='D', origin='unix')
    lightcurve = lightcurve.sort_values('mjd')
    
    print("1. LIGHT CURVE DATA:")
    print(lightcurve.info())
    print(f"Time span: {lightcurve['mjd'].max() - lightcurve['mjd'].min():.1f} days")
    print(f"Number of observations: {len(lightcurve)}")
    
    # Time series features for ML
    print("\n2. FEATURE EXTRACTION FOR ML:")
    
    features = {}
    
    # Basic statistics
    features['mean_mag'] = lightcurve['magnitude'].mean()
    features['std_mag'] = lightcurve['magnitude'].std()
    features['amplitude'] = lightcurve['magnitude'].max() - lightcurve['magnitude'].min()
    
    # Percentiles
    features['mag_5'] = lightcurve['magnitude'].quantile(0.05)
    features['mag_95'] = lightcurve['magnitude'].quantile(0.95)
    
    # Time series specific
    features['n_observations'] = len(lightcurve)
    features['timespan'] = lightcurve['mjd'].max() - lightcurve['mjd'].min()
    features['mean_sampling'] = features['timespan'] / features['n_observations']
    
    # Changes between consecutive observations
    lightcurve['mag_diff'] = lightcurve['magnitude'].diff()
    lightcurve['time_diff'] = lightcurve['mjd'].diff()
    lightcurve['rate_of_change'] = lightcurve['mag_diff'] / lightcurve['time_diff']
    
    features['mean_rate_change'] = lightcurve['rate_of_change'].abs().mean()
    features['max_rate_change'] = lightcurve['rate_of_change'].abs().max()
    
    # Skewness and kurtosis (shape indicators)
    features['skewness'] = lightcurve['magnitude'].skew()
    features['kurtosis'] = lightcurve['magnitude'].kurtosis()
    
    # Beyond features (for period finding)
    from scipy.stats import skew, kurtosis
    features['beyond1std'] = ((lightcurve['magnitude'] - features['mean_mag']).abs() > 
                              features['std_mag']).mean()
    
    print("Extracted features for ML:")
    for key, value in features.items():
        print(f"  {key}: {value:.3f}")
    
    # Resampling for regular time series
    print("\n3. RESAMPLING FOR ANALYSIS:")
    
    lightcurve_indexed = lightcurve.set_index('datetime')
    
    # Daily binning
    daily_lc = lightcurve_indexed.resample('1D').agg({
        'magnitude': ['mean', 'std', 'count'],
        'error': 'mean'
    })
    daily_lc.columns = ['_'.join(col).strip() for col in daily_lc.columns]
    daily_lc = daily_lc[daily_lc['magnitude_count'] > 0]
    
    print(f"Daily binned: {len(daily_lc)} days with data")
    
    # Rolling statistics (for trend detection)
    print("\n4. ROLLING WINDOW ANALYSIS:")
    
    window_size = 10  # days
    lightcurve_indexed['rolling_mean'] = lightcurve_indexed['magnitude'].rolling(
        window=f'{window_size}D', min_periods=5
    ).mean()
    
    lightcurve_indexed['rolling_std'] = lightcurve_indexed['magnitude'].rolling(
        window=f'{window_size}D', min_periods=5
    ).std()
    
    # Detect outliers
    lightcurve_indexed['outlier'] = (
        np.abs(lightcurve_indexed['magnitude'] - lightcurve_indexed['rolling_mean']) > 
        3 * lightcurve_indexed['rolling_std']
    )
    
    print(f"Outliers detected: {lightcurve_indexed['outlier'].sum()}")
    
    # Period analysis preparation
    print("\n5. PERIOD ANALYSIS PREPARATION:")
    
    # Create phase-folded DataFrame for different trial periods
    def phase_fold(lc_df, period):
        """Phase fold light curve at given period."""
        df = lc_df.copy()
        df['phase'] = (df['mjd'] % period) / period
        return df
    
    # Try multiple periods
    trial_periods = np.linspace(0.5, 0.6, 100)
    chi2_values = []
    
    for period in trial_periods:
        folded = phase_fold(lightcurve, period)
        
        # Bin in phase
        phase_bins = np.linspace(0, 1, 20)
        binned = folded.groupby(pd.cut(folded['phase'], phase_bins))['magnitude'].agg(['mean', 'std'])
        
        # Simple chi-squared
        chi2 = binned['std'].mean() if not binned['std'].isna().all() else np.inf
        chi2_values.append(chi2)
    
    best_period_idx = np.argmin(chi2_values)
    best_period = trial_periods[best_period_idx]
    
    print(f"Best period found: {best_period:.4f} days (true: {true_period:.4f})")
    
    return lightcurve, features

lightcurve_df, ml_features = time_series_astronomy()
```

## GroupBy Operations for Population Studies

### Analyzing Stellar Populations

```python
def population_analysis():
    """GroupBy operations for stellar population studies."""
    
    # Create a large stellar survey dataset
    n_stars = 50000
    
    # Multiple stellar populations
    populations = []
    
    # Thin disk
    n_thin = int(0.7 * n_stars)
    thin_disk = pd.DataFrame({
        'population': 'thin_disk',
        'age_gyr': np.random.gamma(3, 1.5, n_thin),
        'metallicity': np.random.normal(0, 0.2, n_thin),
        'velocity_dispersion': np.random.gamma(20, 2, n_thin),
        'scale_height_pc': np.random.normal(300, 50, n_thin),
    })
    
    # Thick disk
    n_thick = int(0.2 * n_stars)
    thick_disk = pd.DataFrame({
        'population': 'thick_disk',
        'age_gyr': np.random.gamma(10, 2, n_thick),
        'metallicity': np.random.normal(-0.5, 0.3, n_thick),
        'velocity_dispersion': np.random.gamma(40, 5, n_thick),
        'scale_height_pc': np.random.normal(900, 100, n_thick),
    })
    
    # Halo
    n_halo = n_stars - n_thin - n_thick
    halo = pd.DataFrame({
        'population': 'halo',
        'age_gyr': np.random.gamma(12, 1, n_halo),
        'metallicity': np.random.normal(-1.5, 0.5, n_halo),
        'velocity_dispersion': np.random.gamma(100, 20, n_halo),
        'scale_height_pc': np.random.exponential(3000, n_halo),
    })
    
    # Combine populations
    survey = pd.concat([thin_disk, thick_disk, halo], ignore_index=True)
    
    # Add observational properties
    survey['apparent_mag'] = 10 + 5 * np.log10(survey['scale_height_pc']) + np.random.normal(0, 0.5, n_stars)
    survey['color_index'] = 0.5 + 0.1 * survey['metallicity'] + np.random.normal(0, 0.1, n_stars)
    
    print("1. POPULATION STATISTICS:")
    population_stats = survey.groupby('population').agg({
        'age_gyr': ['mean', 'std', 'median'],
        'metallicity': ['mean', 'std'],
        'velocity_dispersion': ['mean', 'std'],
        'scale_height_pc': ['mean', 'median']
    })
    print(population_stats)
    
    print("\n2. CUSTOM AGGREGATIONS:")
    
    def weighted_mean(values, weights):
        """Calculate weighted mean."""
        return np.average(values, weights=weights)
    
    # Custom aggregation functions
    agg_funcs = {
        'age_gyr': ['mean', 'std', lambda x: np.percentile(x, 90)],
        'metallicity': ['mean', 'median', lambda x: x.quantile(0.25)],
        'velocity_dispersion': ['mean', 'max']
    }
    
    custom_stats = survey.groupby('population').agg(agg_funcs)
    custom_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                            for col in custom_stats.columns]
    print(custom_stats)
    
    print("\n3. TRANSFORM OPERATIONS:")
    
    # Normalize within groups
    survey['age_normalized'] = survey.groupby('population')['age_gyr'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Rank within groups
    survey['metallicity_rank'] = survey.groupby('population')['metallicity'].rank(
        method='dense', ascending=False
    )
    
    # Cumulative statistics
    survey_sorted = survey.sort_values(['population', 'age_gyr'])
    survey_sorted['cumulative_fraction'] = survey_sorted.groupby('population').cumcount() / \
                                           survey_sorted.groupby('population')['population'].transform('count')
    
    print("Sample of transformed data:")
    print(survey[['population', 'age_gyr', 'age_normalized', 'metallicity', 'metallicity_rank']].head(10))
    
    print("\n4. GROUPED FILTERING:")
    
    # Keep only populations with enough statistics
    min_pop_size = 100
    large_pops = survey.groupby('population').filter(lambda x: len(x) >= min_pop_size)
    print(f"Stars in large populations: {len(large_pops)} / {len(survey)}")
    
    # Select extreme objects in each population
    def select_extremes(group, n=100):
        """Select n most extreme objects by metallicity."""
        return group.nlargest(n, 'metallicity', keep='all')
    
    metal_rich = survey.groupby('population').apply(select_extremes, n=50)
    print(f"Metal-rich selection: {len(metal_rich)} stars")
    
    print("\n5. BINNING AND CATEGORICAL ANALYSIS:")
    
    # Bin ages for analysis
    survey['age_bin'] = pd.cut(survey['age_gyr'], 
                               bins=[0, 2, 5, 10, 15],
                               labels=['young', 'intermediate', 'old', 'ancient'])
    
    # Cross-tabulation
    cross_tab = pd.crosstab(survey['population'], survey['age_bin'], 
                            normalize='index') * 100
    print("Age distribution by population (%):")
    print(cross_tab.round(1))
    
    return survey

stellar_survey = population_analysis()
```

## Data Preparation for Machine Learning

### Feature Engineering Pipeline

```python
def ml_data_preparation():
    """Prepare astronomical data for machine learning."""
    
    # Create a galaxy classification dataset
    n_galaxies = 10000
    
    # Generate features
    galaxies = pd.DataFrame({
        # Photometry
        'mag_u': np.random.normal(22, 1.5, n_galaxies),
        'mag_g': np.random.normal(21, 1.5, n_galaxies),
        'mag_r': np.random.normal(20, 1.5, n_galaxies),
        'mag_i': np.random.normal(19.5, 1.5, n_galaxies),
        'mag_z': np.random.normal(19, 1.5, n_galaxies),
        
        # Morphology
        'petrosian_radius': np.random.lognormal(1, 0.5, n_galaxies),
        'concentration': np.random.uniform(1.5, 5, n_galaxies),
        'asymmetry': np.random.beta(2, 5, n_galaxies),
        'smoothness': np.random.beta(2, 8, n_galaxies),
        
        # Spectroscopy
        'redshift': np.random.gamma(2, 0.3, n_galaxies),
        'velocity_dispersion': np.random.lognormal(5, 0.5, n_galaxies),
        'h_alpha_flux': np.random.lognormal(-15, 1, n_galaxies),
        'd4000_break': np.random.normal(1.5, 0.3, n_galaxies),
        
        # Environment
        'n_neighbors_1mpc': np.random.poisson(5, n_galaxies),
        'distance_to_nearest': np.random.exponential(0.5, n_galaxies),
    })
    
    # Add classification labels (simplified)
    def classify_galaxy(row):
        if row['d4000_break'] > 1.6 and row['h_alpha_flux'] < -15:
            return 'elliptical'
        elif row['concentration'] > 3.5:
            return 'spiral_early'
        elif row['asymmetry'] > 0.3:
            return 'irregular'
        else:
            return 'spiral_late'
    
    galaxies['morphology_class'] = galaxies.apply(classify_galaxy, axis=1)
    
    print("1. INITIAL DATA EXPLORATION:")
    print(galaxies.info())
    print(f"\nClass distribution:\n{galaxies['morphology_class'].value_counts()}")
    
    print("\n2. FEATURE ENGINEERING:")
    
    # Color indices
    galaxies['u_g'] = galaxies['mag_u'] - galaxies['mag_g']
    galaxies['g_r'] = galaxies['mag_g'] - galaxies['mag_r']
    galaxies['r_i'] = galaxies['mag_r'] - galaxies['mag_i']
    galaxies['i_z'] = galaxies['mag_i'] - galaxies['mag_z']
    
    # Composite features
    galaxies['CAS_score'] = (galaxies['concentration'] + 
                            10 * galaxies['asymmetry'] + 
                            5 * galaxies['smoothness'])
    
    # Logarithmic transforms for skewed features
    galaxies['log_vdisp'] = np.log10(galaxies['velocity_dispersion'])
    galaxies['log_halpha'] = np.log10(galaxies['h_alpha_flux'] - galaxies['h_alpha_flux'].min() + 1)
    
    # Interaction features
    galaxies['density_indicator'] = galaxies['n_neighbors_1mpc'] / (galaxies['distance_to_nearest'] + 0.1)
    
    print("Added features:", [col for col in galaxies.columns if col not in 
                              ['mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 
                               'morphology_class']])
    
    print("\n3. HANDLING MISSING DATA:")
    
    # Introduce missing data (realistic scenario)
    missing_fraction = 0.1
    for col in ['velocity_dispersion', 'h_alpha_flux', 'd4000_break']:
        missing_idx = np.random.choice(galaxies.index, 
                                      size=int(len(galaxies) * missing_fraction),
                                      replace=False)
        galaxies.loc[missing_idx, col] = np.nan
    
    print(f"Missing data summary:\n{galaxies.isnull().sum()}")
    
    # Imputation strategies
    from sklearn.impute import SimpleImputer, KNNImputer
    
    # Simple imputation
    simple_imputer = SimpleImputer(strategy='median')
    
    # KNN imputation (better for correlated features)
    knn_imputer = KNNImputer(n_neighbors=5)
    
    # Apply to numerical features
    numerical_features = galaxies.select_dtypes(include=[np.number]).columns.drop('morphology_class', errors='ignore')
    
    galaxies_simple = galaxies.copy()
    galaxies_simple[numerical_features] = simple_imputer.fit_transform(galaxies[numerical_features])
    
    print("\n4. OUTLIER DETECTION:")
    
    from sklearn.ensemble import IsolationForest
    
    # Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso_forest.fit_predict(galaxies_simple[numerical_features])
    
    galaxies_simple['is_outlier'] = outlier_labels == -1
    print(f"Outliers detected: {galaxies_simple['is_outlier'].sum()}")
    
    print("\n5. FEATURE SCALING:")
    
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
    
    # Different scaling methods
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),  # Better with outliers
        'quantile': QuantileTransformer(output_distribution='normal')  # Gaussianize
    }
    
    # Apply scaling
    scaled_features = {}
    for name, scaler in scalers.items():
        scaled_features[name] = pd.DataFrame(
            scaler.fit_transform(galaxies_simple[numerical_features]),
            columns=numerical_features,
            index=galaxies_simple.index
        )
    
    print("Scaling comparison (first 5 features):")
    for name in scalers.keys():
        print(f"\n{name.capitalize()} scaling:")
        print(scaled_features[name].iloc[:, :5].describe().loc[['mean', 'std']])
    
    print("\n6. TRAIN-TEST SPLIT WITH STRATIFICATION:")
    
    from sklearn.model_selection import train_test_split
    
    # Stratified split to maintain class balance
    X = scaled_features['robust']
    y = galaxies_simple['morphology_class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"\nClass distribution in train:\n{y_train.value_counts(normalize=True)}")
    print(f"\nClass distribution in test:\n{y_test.value_counts(normalize=True)}")
    
    return galaxies_simple, X_train, X_test, y_train, y_test

galaxies_ml, X_train, X_test, y_train, y_test = ml_data_preparation()
```

## Merging and Joining Astronomical Catalogs

### Cross-Matching Surveys

```python
def catalog_merging():
    """Merge and join operations for multi-survey astronomy."""
    
    # Simulate different surveys
    n_objects = 5000
    
    # Optical survey (like SDSS)
    optical = pd.DataFrame({
        'objid': np.arange(1000, 1000 + n_objects),
        'ra': np.random.uniform(150, 160, n_objects),
        'dec': np.random.uniform(-5, 5, n_objects),
        'mag_g': np.random.normal(20, 2, n_objects),
        'mag_r': np.random.normal(19.5, 2, n_objects),
        'mag_i': np.random.normal(19, 2, n_objects),
        'photoz': np.random.gamma(2, 0.3, n_objects),
        'survey': 'SDSS'
    })
    
    # X-ray survey (like Chandra)
    n_xray = 500
    xray_idx = np.random.choice(n_objects, n_xray, replace=False)
    xray = pd.DataFrame({
        'source_id': np.arange(2000, 2000 + n_xray),
        'ra': optical.iloc[xray_idx]['ra'].values + np.random.normal(0, 0.0003, n_xray),  # Small offset
        'dec': optical.iloc[xray_idx]['dec'].values + np.random.normal(0, 0.0003, n_xray),
        'flux_soft': np.random.lognormal(-13, 1, n_xray),
        'flux_hard': np.random.lognormal(-13.5, 1, n_xray),
        'hardness_ratio': np.random.uniform(-1, 1, n_xray),
        'survey': 'Chandra'
    })
    
    # Radio survey (like FIRST)
    n_radio = 300
    radio_idx = np.random.choice(n_objects, n_radio, replace=False)
    radio = pd.DataFrame({
        'name': [f'FIRST_J{i:06d}' for i in range(n_radio)],
        'ra': optical.iloc[radio_idx]['ra'].values + np.random.normal(0, 0.0005, n_radio),
        'dec': optical.iloc[radio_idx]['dec'].values + np.random.normal(0, 0.0005, n_radio),
        'flux_1400mhz': np.random.lognormal(0, 2, n_radio),
        'spectral_index': np.random.normal(-0.7, 0.3, n_radio),
        'survey': 'FIRST'
    })
    
    print("1. SURVEY SUMMARIES:")
    print(f"Optical: {len(optical)} objects")
    print(f"X-ray: {len(xray)} objects")
    print(f"Radio: {len(radio)} objects")
    
    print("\n2. POSITIONAL CROSS-MATCHING:")
    
    # Function for angular separation
    def angular_separation(ra1, dec1, ra2, dec2):
        """Calculate angular separation in arcseconds."""
        # Simplified for small angles
        cos_dec = np.cos(np.radians(dec1))
        dra = (ra2 - ra1) * cos_dec
        ddec = dec2 - dec1
        return 3600 * np.sqrt(dra**2 + ddec**2)
    
    # Cross-match optical with X-ray
    from sklearn.neighbors import BallTree
    
    # Convert to radians for BallTree
    optical_coords = np.radians(optical[['ra', 'dec']].values)
    xray_coords = np.radians(xray[['ra', 'dec']].values)
    
    # Build tree and query
    tree = BallTree(optical_coords, metric='haversine')
    max_sep_rad = np.radians(3/3600)  # 3 arcsec
    
    indices, distances = tree.query_radius(xray_coords, r=max_sep_rad, return_distance=True)
    
    # Create matched catalog
    matches = []
    for i, (idx_list, dist_list) in enumerate(zip(indices, distances)):
        if len(idx_list) > 0:
            # Take closest match
            best_idx = idx_list[np.argmin(dist_list)]
            matches.append({
                'optical_idx': best_idx,
                'xray_idx': i,
                'separation_arcsec': np.degrees(dist_list[np.argmin(dist_list)]) * 3600
            })
    
    match_df = pd.DataFrame(matches)
    print(f"Optical-Xray matches: {len(match_df)} / {len(xray)}")
    
    # Merge matched catalogs
    optical_xray = optical.iloc[match_df['optical_idx'].values].reset_index(drop=True)
    xray_matched = xray.iloc[match_df['xray_idx'].values].reset_index(drop=True)
    
    combined = pd.concat([optical_xray, xray_matched.drop(['ra', 'dec', 'survey'], axis=1)], axis=1)
    combined['separation'] = match_df['separation_arcsec'].values
    
    print(f"Combined catalog shape: {combined.shape}")
    
    print("\n3. DIFFERENT JOIN TYPES:")
    
    # Create simplified catalogs for demonstration
    cat1 = optical[['objid', 'ra', 'dec', 'mag_g']].head(100)
    cat2 = pd.DataFrame({
        'objid': optical['objid'].iloc[50:150].values,  # Partial overlap
        'specz': np.random.gamma(2, 0.3, 100),
        'quality': np.random.choice(['good', 'bad'], 100)
    })
    
    # Inner join - only matched objects
    inner = pd.merge(cat1, cat2, on='objid', how='inner')
    print(f"Inner join: {len(inner)} objects (both catalogs)")
    
    # Left join - all from cat1
    left = pd.merge(cat1, cat2, on='objid', how='left')
    print(f"Left join: {len(left)} objects (all from catalog 1)")
    print(f"  With specz: {left['specz'].notna().sum()}")
    
    # Outer join - all objects
    outer = pd.merge(cat1, cat2, on='objid', how='outer')
    print(f"Outer join: {len(outer)} objects (union)")
    
    print("\n4. CONCATENATING SURVEYS:")
    
    # Standardize column names
    optical_std = optical[['ra', 'dec', 'survey']].copy()
    optical_std['flux'] = 10**(-0.4 * optical['mag_r'])
    
    xray_std = xray[['ra', 'dec', 'survey']].copy()
    xray_std['flux'] = xray['flux_soft']
    
    radio_std = radio[['ra', 'dec', 'survey']].copy()
    radio_std['flux'] = radio['flux_1400mhz']
    
    # Concatenate all surveys
    all_surveys = pd.concat([optical_std, xray_std, radio_std], 
                           ignore_index=True, sort=False)
    
    print(f"Combined all surveys: {len(all_surveys)} total detections")
    print(all_surveys.groupby('survey')['flux'].describe())
    
    return combined, all_surveys

matched_catalog, all_surveys = catalog_merging()
```

## Performance Optimization for Large Catalogs

### Scaling to Big Data

```python
def performance_optimization():
    """Optimize Pandas for large astronomical catalogs."""
    
    print("1. MEMORY OPTIMIZATION:")
    
    # Create large catalog
    n = 1_000_000
    
    # Bad: default dtypes
    catalog_bad = pd.DataFrame({
        'id': np.arange(n),  # int64 (8 bytes)
        'ra': np.random.uniform(0, 360, n),  # float64 (8 bytes)
        'dec': np.random.uniform(-90, 90, n),  # float64
        'mag': np.random.uniform(10, 25, n),  # float64
        'flag': np.random.choice([0, 1], n),  # int64
        'survey': np.random.choice(['SDSS', 'GAIA', 'WISE'], n)  # object
    })
    
    memory_bad = catalog_bad.memory_usage(deep=True).sum() / 1e6
    print(f"Default dtypes: {memory_bad:.1f} MB")
    
    # Good: optimized dtypes
    catalog_good = pd.DataFrame({
        'id': pd.array(np.arange(n), dtype='uint32'),  # 4 bytes
        'ra': pd.array(np.random.uniform(0, 360, n), dtype='float32'),  # 4 bytes
        'dec': pd.array(np.random.uniform(-90, 90, n), dtype='float32'),
        'mag': pd.array(np.random.uniform(10, 25, n), dtype='float32'),
        'flag': pd.array(np.random.choice([0, 1], n), dtype='bool'),  # 1 byte
        'survey': pd.Categorical(np.random.choice(['SDSS', 'GAIA', 'WISE'], n))  # Categorical
    })
    
    memory_good = catalog_good.memory_usage(deep=True).sum() / 1e6
    print(f"Optimized dtypes: {memory_good:.1f} MB")
    print(f"Memory saved: {(1 - memory_good/memory_bad)*100:.1f}%")
    
    print("\n2. CHUNKING FOR LARGE FILES:")
    
    def process_large_catalog(filename, chunksize=10000):
        """Process large catalog in chunks."""
        
        results = []
        
        # Simulate reading chunks
        for chunk_id in range(3):  # Normally: pd.read_csv(filename, chunksize=chunksize)
            # Simulate chunk
            chunk = pd.DataFrame({
                'ra': np.random.uniform(0, 360, chunksize),
                'dec': np.random.uniform(-90, 90, chunksize),
                'mag': np.random.uniform(15, 22, chunksize)
            })
            
            # Process chunk
            chunk_stats = {
                'chunk_id': chunk_id,
                'mean_mag': chunk['mag'].mean(),
                'bright_count': (chunk['mag'] < 18).sum(),
                'area_coverage': (chunk['ra'].max() - chunk['ra'].min()) * 
                                (chunk['dec'].max() - chunk['dec'].min())
            }
            
            results.append(chunk_stats)
        
        return pd.DataFrame(results)
    
    chunk_results = process_large_catalog('large_catalog.csv')
    print("Chunk processing results:")
    print(chunk_results)
    
    print("\n3. QUERY OPTIMIZATION:")
    
    # Create indexed DataFrame
    catalog_indexed = catalog_good.set_index('id')
    
    # Slow: Python loop
    import time
    
    start = time.time()
    bright_loop = []
    for idx, row in catalog_good.iterrows():
        if row['mag'] < 15 and row['dec'] > 0:
            bright_loop.append(row['id'])
        if len(bright_loop) >= 100:
            break
    time_loop = time.time() - start
    
    # Fast: Vectorized query
    start = time.time()
    bright_vector = catalog_good.query('mag < 15 & dec > 0')['id'].head(100).tolist()
    time_vector = time.time() - start
    
    print(f"Loop time: {time_loop:.4f}s")
    print(f"Vectorized time: {time_vector:.4f}s")
    print(f"Speedup: {time_loop/time_vector:.1f}x")
    
    print("\n4. PARALLEL PROCESSING:")
    
    # For CPU-bound operations
    import multiprocessing as mp
    from functools import partial
    
    def calculate_color(group):
        """Calculate color for group of objects."""
        return group['mag'].mean() - group['mag'].median()
    
    # Split data for parallel processing
    n_cores = mp.cpu_count()
    chunks = np.array_split(catalog_good, n_cores)
    
    # Parallel apply (conceptual - actual implementation would use Dask)
    print(f"Would process on {n_cores} cores")
    
    print("\n5. USING DASK FOR SCALING:")
    
    # Conceptual Dask usage
    print("""
    import dask.dataframe as dd
    
    # Read large catalog
    ddf = dd.read_csv('huge_catalog.csv', blocksize='100MB')
    
    # Operations are lazy
    result = ddf[ddf.mag < 20].groupby('survey').mag.mean()
    
    # Compute when needed
    result.compute()
    """)
    
    return catalog_good

optimized_catalog = performance_optimization()
```

## Try It Yourself

### Exercise 1: Complete Variable Star Classification Pipeline

```python
def variable_star_pipeline(lightcurves_df):
    """
    Build a complete pipeline for variable star classification.
    
    Tasks:
    1. Extract time series features (mean, std, skewness, etc.)
    2. Find periods using Lomb-Scargle
    3. Extract phase-folded features
    4. Handle missing data and outliers
    5. Prepare features for ML classification
    6. Split into train/test maintaining class balance
    
    Returns feature matrix ready for sklearn classifiers.
    """
    # Your code here
    pass
```

### Exercise 2: Multi-Survey Catalog Integration

```python
def integrate_surveys(optical_df, xray_df, radio_df, ir_df):
    """
    Integrate multiple astronomical surveys.
    
    Requirements:
    1. Cross-match by position (handle different coordinate precisions)
    2. Handle different column names and units
    3. Create unified photometry columns
    4. Calculate multi-wavelength colors
    5. Flag objects detected in multiple surveys
    6. Handle missing data appropriately
    
    Return integrated catalog with source classification.
    """
    # Your code here
    pass
```

### Exercise 3: Galaxy Cluster Analysis

```python
def analyze_galaxy_cluster(galaxies_df):
    """
    Comprehensive galaxy cluster analysis.
    
    Tasks:
    1. Identify cluster members using redshift
    2. Calculate velocity dispersion
    3. Estimate cluster mass (virial theorem)
    4. Find red sequence in color-magnitude diagram
    5. Identify BCG (brightest cluster galaxy)
    6. Calculate radial profiles
    7. Prepare data for ML substructure detection
    """
    # Your code here
    pass
```

## Key Takeaways

✅ **Pandas bridges astronomy and data science** - From catalogs to ML pipelines  
✅ **DataFrames handle heterogeneous data** - Mixed types, missing values, metadata  
✅ **GroupBy enables population studies** - Statistics by galaxy type, stellar population  
✅ **Time series tools for light curves** - Resampling, rolling windows, feature extraction  
✅ **Efficient merging for multi-survey science** - Cross-matching, joining, concatenating  
✅ **Feature engineering for ML** - From raw observations to ML-ready features  
✅ **Memory optimization crucial** - Use appropriate dtypes, chunking, Dask for scaling  
✅ **Vectorization is key** - Avoid loops, use query(), apply(), transform()  

## Connecting to Your Course

This Pandas knowledge directly enables:
- **Project 3**: Statistical analysis of simulation outputs
- **Project 4**: Managing Monte Carlo results
- **Project 5**: Preparing data for neural networks
- **Final Project**: Professional data science workflows

Combined with NumPy, SciPy, and Matplotlib, you now have the complete toolkit for modern astronomical data science, preparing you for both research and industry applications.