# ASTR596: Modeling the Universe - Complete Scaffolded Project Progression

## Project 1: Stellar Structure & Astrophysical Foundations
**Duration**: 2 weeks (Aug 25 - Sept 8) | **Skills Focus**: Python/OOP foundations, fundamental astronomy

### Science Challenge
Build a comprehensive `Star` class that calculates stellar properties with metallicity dependence and implements fundamental astronomical relations for synthetic data generation.

### Core Physics Components
- **Stellar Structure**: Implement full Tout et al. (1996) metallicity-dependent ZAMS relations for L(M,Z) and R(M,Z)
- **Stellar Evolution**: Basic stellar lifetime calculations, main sequence evolution tracks
- **Fundamental Astronomy**: Wien's law, blackbody function, parallax-distance relation, angular size
- **Color Systems**: B-V colors, color-magnitude diagrams, surface brightness calculations
- **H-R Diagram**: Multi-metallicity stellar evolutionary tracks and zero-age main sequence

### Technical Implementation
- **Object-Oriented Design**: 
  - `Star` class with properties (mass, radius, temperature, luminosity, metallicity, age)
  - `StellarPopulation` class managing multiple stars with different metallicities
- **Astrophysical Toolkit Functions**:
  ```python
  def wien_displacement_law(T): # Peak wavelength of blackbody
  def blackbody_flux(T, wavelength): # Planck function
  def parallax_distance(parallax_mas): # Distance from parallax
  def angular_size(physical_size, distance): # Angular size in arcseconds
  def luminosity_distance(z, H0=70): # Cosmological distances
  def surface_brightness(luminosity, angular_area): # Extended object brightness
  ```
- **Python Fundamentals**: NumPy arrays, matplotlib visualization, proper documentation
- **Software Practices**: Git workflow, modular code organization, unit testing

### Expected Deliverables
- Working Star and StellarPopulation classes with full Tout metallicity dependence
- Complete astrophysical toolkit for synthetic observations
- H-R diagrams across metallicity range (Z = 0.0001 to 0.03)
- Color-magnitude diagrams showing metallicity effects
- Mass-lifetime relationship analysis for different stellar populations

### Exploration Opportunities
- **Creative Experiments**: "How would H-R diagram look for stars with Z = 0.1?" "What if stellar lifetimes scaled differently with mass?"
- **Parameter Exploration**: Compare low-metallicity (Population II) vs high-metallicity (Population I) stellar populations
- **Connections to Observations**: Calculate properties of nearby stars (Vega, Sirius, etc.)

### Learning Outcomes
- Master object-oriented programming through stellar physics
- Understand stellar structure, evolution, and metallicity effects
- Build foundational astronomical calculation toolkit
- Develop professional software development practices

---

## Project 2: N-Body Dynamics & Statistical Stellar Systems
**Duration**: 2 weeks (Sept 8 - Sept 22) | **Skills Focus**: Numerical integration, Monte Carlo sampling, stellar clusters

### Science Challenge
Simulate realistic gravitational N-body stellar systems by sampling from Initial Mass Functions (IMF) and spatial distributions, enabling exploration of diverse cluster configurations.

### Core Physics Components
- **Gravitational Dynamics**: N-body Newton's laws with softened gravity for close encounters
- **Stellar Cluster Physics**: King profiles, virial equilibrium, relaxation timescales, escape velocities
- **IMF Sampling**: Monte Carlo sampling from Salpeter, Kroupa, and custom/top-heavy IMFs
- **Spatial Distributions**: King profiles, Plummer spheres, uniform distributions
- **Cluster Evolution**: Mass segregation, evaporation, core collapse timescales

### Technical Implementation
- **Extensions from Project 1**: Use realistic stellar masses and properties from Star class
- **Monte Carlo Integration**: 
  - Sample stellar masses from IMF (provided functions: `sample_salpeter_imf()`, `sample_kroupa_imf()`)
  - Sample positions from King profile (provided: `sample_king_profile()`)
  - Include binary fraction as adjustable parameter
- **Numerical Integration**: Euler, RK4, Leapfrog integrators with stability analysis
- **Energy Tracking**: Kinetic, potential, and total energy conservation
- **Performance Optimization**: Efficient force calculations, adaptive timesteps

### Expected Deliverables
- N-body integrator with multiple algorithms (20-100 star systems)
- IMF sampling and cluster initialization tools
- Solar system simulation demonstrating long-term stability
- Stellar cluster evolution with realistic mass spectrum and spatial structure
- Energy conservation analysis and algorithmic performance comparison

### Creative Experimental Opportunities
- **"Design a cluster that evaporates in 1 Myr"**: Explore cluster binding energy
- **"What if all stars were 10 M☉?"**: Test equipartition and stellar interactions
- **"Super dense globular cluster"**: Push density limits, explore core collapse
- **"Primordial star formation"**: Top-heavy IMF with zero metallicity stars
- **"Binary-dominated cluster"**: High binary fraction effects on dynamics

### Parameter Exploration Suggestions
- Compare different IMF slopes and mass ranges
- Vary metallicity across the cluster (metallicity gradients)
- Experiment with extreme density configurations
- Test different binary fractions and orbital distributions
- Explore clusters with initial rotation or turbulence

### Learning Outcomes
- Understand gravitational dynamics and stellar cluster physics
- Master Monte Carlo sampling and statistical methods
- Learn numerical integration theory and algorithm comparison
- Develop intuition for stellar system evolution through experimentation

---

## Project 3: Monte Carlo Radiative Transfer & Synthetic Observations
**Duration**: 2 weeks (Sept 22 - Oct 6) | **Skills Focus**: Complex algorithms, radiative physics, data generation

### Science Challenge
Implement Monte Carlo Radiative Transfer (MCRT) for stellar atmospheres and dusty environments, generating realistic synthetic observations for statistical analysis.

### Core Physics Components
- **Radiative Transfer Physics**: Photon transport through absorbing and scattering media
- **Dust Properties**: Wavelength-dependent opacity, scattering albedo, phase functions
- **Stellar Atmosphere Models**: Plane-parallel atmospheres, limb darkening, temperature gradients
- **Scattering Physics**: Isotropic vs anisotropic scattering, single vs multiple scattering
- **Observational Effects**: Line-of-sight variations, dust geometry effects

### Technical Implementation
- **Builds on Projects 1-2**: Use stellar properties and Monte Carlo experience
- **MCRT Algorithm**: 
  - Photon packet tracking with statistical weights
  - Absorption and scattering event handling
  - Optical depth calculations and Beer's law
  - Multiple scattering implementations
- **Advanced Features**:
  - Start with isotropic scattering, add anisotropic as extension
  - Include both absorption and scattering components
  - Variable dust grain size distributions
  - Optional: polarization for ambitious students
- **Data Generation**: Create comprehensive synthetic datasets for Projects 4-5

### Expected Deliverables
- Working MCRT code for plane-parallel stellar atmospheres
- Synthetic multi-wavelength spectral energy distributions
- Parameter space exploration (dust density, optical depth, geometry)
- Comparison of different scattering assumptions
- High-quality synthetic datasets with realistic noise for statistical analysis

### Science Case Connections
- **"Why do stars appear redder through dust?"**: Explore wavelength-dependent extinction
- **"How does dust geometry affect observations?"**: Compare different dust distributions
- **"What creates the 2175 Å bump?"**: Investigate specific dust features

### Creative Experiments
- **"Dust only in outer atmosphere"**: Explore layered dust distributions
- **"Extreme optical depths"**: Push into optically thick regime
- **"Binary star with circumbinary dust"**: Complex geometric configurations
- **"Variable dust properties"**: Time-dependent or spatially varying extinction

### Learning Outcomes
- Master advanced Monte Carlo algorithm implementation
- Understand radiative transfer physics and stellar atmospheres
- Develop complex scientific algorithm debugging and validation
- Generate realistic synthetic datasets for statistical analysis

---

## Project 4: Linear Regression & Frequentist Parameter Estimation
**Duration**: 2 weeks (Oct 6 - Oct 20) | **Skills Focus**: ML from scratch, statistical modeling

### Science Challenge
Implement linear regression from mathematical foundations and apply to recovering physical parameters from Project 3's MCRT synthetic observations.

### Core Physics & Data Science Components
- **Parameter Recovery**: Extract dust properties (optical depth, grain size, geometry) from synthetic spectra
- **Degeneracy Analysis**: Understand parameter correlations and measurement limitations
- **Model Selection**: Compare linear vs polynomial fits, regularization techniques
- **Error Propagation**: Handle observational uncertainties and systematic effects
- **Astrophysical Applications**: Dust-to-gas ratios, extinction curve fitting, stellar parameter recovery

### Technical Implementation
- **ML from Mathematical Foundations**: 
  - Derive normal equations: (X^T X)β = X^T y
  - Implement gradient descent optimization from scratch
  - No scikit-learn - build complete understanding
- **Data Integration**: Use identical MCRT synthetic observations for direct comparison with Project 5
- **Statistical Analysis**: 
  - Confidence intervals, cross-validation, model comparison
  - Outlier detection and robust fitting techniques
  - Feature engineering for astrophysical problems

### Expected Deliverables
- Complete linear regression implementation without external ML libraries
- Parameter recovery pipeline for MCRT synthetic observations
- Analysis of parameter degeneracies and measurement uncertainties
- Model validation framework with statistical diagnostics
- Confidence interval calculations and model comparison metrics

### Exploration Components
- **"Which parameters are most degenerate?"**: Explore parameter correlation matrices
- **"How much noise can the method handle?"**: Test robustness against observational errors
- **"Can we recover non-linear relationships?"**: Experiment with polynomial features

### Connection to Real Astronomy
- Compare recovered parameters with known input values
- Investigate which observational setups break degeneracies
- Explore how wavelength coverage affects parameter recovery

### Learning Outcomes
- Master machine learning fundamentals and optimization theory
- Understand statistical model validation and comparison
- Learn scientific data analysis and uncertainty quantification
- Develop intuition for parameter estimation challenges in astronomy

---

## Project 5: Bayesian Inference & MCMC - Statistical Method Comparison
**Duration**: 2 weeks (Oct 20 - Nov 3) | **Skills Focus**: Bayesian methods, advanced statistical inference

### Science Challenge
Implement MCMC from scratch and apply Bayesian inference to the identical MCRT data from Projects 3-4, enabling direct comparison of frequentist vs Bayesian approaches.

### Core Physics & Statistical Components
- **Bayesian Framework**: Prior specification using astrophysical knowledge of dust and stellar properties
- **MCMC Implementation**: Metropolis-Hastings algorithm with adaptive step sizing
- **Physical Priors**: Incorporate realistic constraints (dust properties, stellar physics, IMF)
- **Posterior Analysis**: Full probability distributions vs point estimates
- **Method Comparison**: Direct statistical comparison with Project 4 results on identical data

### Technical Implementation
- **Same Dataset**: Apply to identical MCRT synthetic observations from Project 3
- **MCMC from Scratch**: 
  - Metropolis-Hastings with proposal distributions
  - Adaptive step size algorithms
  - Multiple chain implementation
  - Convergence diagnostics (Gelman-Rubin, autocorrelation)
- **Prior Engineering**: 
  - Physical priors on dust-to-gas ratios
  - Stellar parameter priors from Project 1
  - Hierarchical priors for population studies
- **Posterior Analysis**: Corner plots, credible intervals, model comparison

### Expected Deliverables
- Complete MCMC sampler implementation from mathematical principles
- Bayesian parameter estimation pipeline for dust and stellar properties
- Direct statistical comparison: Bayesian posteriors vs frequentist confidence intervals
- Analysis of prior sensitivity and robustness
- Convergence diagnostics and computational efficiency analysis

### Statistical Method Comparison Focus
- **Plot both confidence intervals and credible intervals on same plots**
- **Model selection comparison**: Which approach handles degeneracies better?
- **Outlier robustness**: Compare method performance with contaminated data
- **Computational efficiency**: Runtime and convergence comparison

### Advanced Extensions
- **Hierarchical Bayesian modeling**: Population-level parameters
- **Model selection**: Bayesian evidence vs frequentist model comparison
- **Robust likelihood functions**: Handle outliers and systematic errors

### Learning Outcomes
- Master Bayesian statistical inference and MCMC implementation
- Understand the fundamental differences between statistical paradigms
- Learn advanced uncertainty quantification and model comparison
- Develop intuition for when to use Bayesian vs frequentist approaches

---

## Project 6: Multi-Wavelength Stellar Extinction with Gaussian Processes
**Duration**: 2 weeks (Nov 3 - Nov 17) | **Skills Focus**: Advanced ML, multi-wavelength astrophysics

### Science Challenge
Implement Gaussian Processes from scratch for multi-wavelength stellar extinction modeling, enabling rapid parameter recovery and uncertainty quantification.

### Core Physics & ML Components
- **Advanced Stellar Physics**: Full Tout et al. metallicity-dependent stellar populations
- **Multi-Wavelength Extinction**: Frequency-dependent dust opacities (Weingartner & Draine 2001)
- **Realistic Observations**: Include photometric uncertainties, systematic effects, filter responses
- **GP Applications**: Spectral interpolation, uncertainty quantification, active learning
- **Surrogate Modeling**: Fast prediction vs full stellar atmosphere calculations

### Technical Implementation
- **Building from Project 1**: Extend Tout relations and astrophysical toolkit
- **Radiative Transfer Connection**: Apply Project 3 concepts to realistic dust extinction
- **GP from Scratch**: 
  - Implement kernel functions (RBF, Matérn, periodic combinations)
  - Hyperparameter optimization via marginal likelihood
  - Uncertainty quantification and prediction intervals
- **Multi-Wavelength Physics**: 
  - Model I_observed(ν) = I_intrinsic(ν) × e^(-τ_ν) across UV/optical/IR
  - Binned opacity approach (6-10 frequency bins)
  - Realistic filter convolutions

### Expected Deliverables
- Complete GP implementation for multi-wavelength stellar data
- Stellar population synthesis with realistic extinction modeling
- Fast SED prediction pipeline with uncertainty quantification
- Performance comparison: GP interpolation vs traditional stellar atmosphere grids
- Analysis of GP performance across parameter space (mass, metallicity, extinction)

### Advanced Extensions & Experiments
- **Kernel Combinations**: Experiment with different kernel combinations for spectral features
- **Active Learning**: GP chooses which observations would be most informative
- **Time-Domain Applications**: Variable star light curves, stellar pulsations
- **GP Emulators**: Train GP to emulate expensive MCRT calculations from Project 3

### Creative Exploration Opportunities
- **"Design optimal filter sets"**: Use GP uncertainty to choose best wavelength coverage
- **"Extreme stellar populations"**: Very metal-poor or super-metal-rich stars
- **"Non-standard extinction laws"**: Explore deviations from standard dust properties

### Connection to Modern Astronomy
- Applications to large astronomical surveys (Gaia, LSST, Euclid)
- Real-time parameter estimation for massive datasets
- Uncertainty-aware predictions for follow-up observations

### Learning Outcomes
- Master advanced machine learning (non-parametric methods)
- Understand multi-wavelength stellar astrophysics and realistic observations
- Learn scientific surrogate modeling and computational efficiency
- Connect modern ML techniques to practical astronomical applications

---

## Final Project: Neural Networks with JAX - Modern Computational Astrophysics
**Duration**: 4 weeks (Nov 17 - Dec 18) | **Skills Focus**: Modern frameworks, neural networks, research integration

### Science Challenge
Build neural networks from mathematical foundations using JAX, then apply to a research-level astronomical problem integrating multiple course techniques.

### Phase 1: Neural Networks from JAX Fundamentals (Week 1)
**Technical Focus**: Implement NN mathematics from scratch using JAX arrays
- **Glass Box Philosophy**: Build feedforward networks, backpropagation, gradient descent using only JAX arrays
- **Structural Support**: Possibly use Equinox for PyTree organization while implementing all mathematics by hand
- **Mathematical Mastery**: Complete understanding of neural network algorithms before using frameworks
- **Simple Validation**: Function approximation or basic astronomical classification problems

### Phase 2: JAX Ecosystem Integration (Week 2)
**Technical Focus**: Transition to modern frameworks with full understanding
- **Framework Transition**: Convert hand-built JAX implementation to Flax/Optax ecosystem
- **Algorithm Translation**: Convert previous algorithm (N-body, MCRT, or regression) to JAX
- **Performance Analysis**: Benchmark hand-built vs framework implementations
- **JAX Transformations**: Master jit, grad, vmap through practical applications

### Phase 3: Research-Level Application (Weeks 3-4)
**Science Focus**: Choose ONE application integrating multiple course concepts

**Option A: Multi-Method Stellar Parameter Estimation**
- **Data Integration**: Combine stellar models (Project 1), extinction (Project 6), synthetic observations
- **Method Comparison**: Neural networks vs GP (Project 6) vs Bayesian (Project 5) approaches
- **Scientific Validation**: Test on realistic stellar survey data, quantify systematic differences
- **Advanced Features**: Bayesian neural networks, uncertainty quantification, ensemble methods

**Option B: Neural Surrogate for Complex Simulations**
- **Physics Integration**: Use MCRT (Project 3) or N-body (Project 2) as expensive "truth" calculations
- **Surrogate Development**: Train neural networks to emulate full physics simulations
- **Performance Demonstration**: Orders-of-magnitude speedup with quantified accuracy
- **Active Learning**: Neural network chooses which simulations to run for optimal training

**Option C: Hierarchical Bayesian Neural Networks**
- **Advanced Integration**: Combine Bayesian inference (Project 5) with neural network flexibility
- **Population Studies**: Apply to stellar populations or cluster properties
- **Uncertainty Quantification**: Full Bayesian treatment of neural network parameters
- **Scientific Application**: Real astronomical datasets with hierarchical structure

### Expected Deliverables
- Complete neural network implementation from mathematical foundations
- JAX ecosystem integration with performance benchmarking
- Research-level final application with scientific validation
- Professional presentation comparing modern vs traditional methods
- Computational portfolio demonstrating technical and scientific growth

### Research Preparation Components
- **Literature Integration**: Reference relevant astronomical applications and recent papers
- **Method Validation**: Compare with established techniques, quantify advantages/limitations
- **Computational Efficiency**: Analyze scaling, memory usage, and optimization
- **Scientific Impact**: Discuss applications to real astronomical problems

### Learning Outcomes
- Master neural network theory and modern computational frameworks
- Integrate multiple course techniques into research-level applications
- Develop skills for computational research careers in academia and industry
- Understand the progression from classical methods to cutting-edge approaches

---

## Course Integration & Pedagogical Framework

### Three-Phase Learning Progression

**Phase 1: Computational Physics Foundations (Projects 1-3)**
- **Character**: Classical computational astrophysics with modern software practices
- **Focus**: Physics understanding through code implementation
- **Skills**: Object-oriented programming, numerical methods, complex algorithms
- **Mindset**: "How do I translate physics equations into working code?"

**Phase 2: Statistical Analysis & Machine Learning (Projects 4-6)**
- **Character**: Modern data analysis applied to astronomical problems
- **Focus**: Statistical inference and machine learning from mathematical foundations
- **Skills**: Parameter estimation, uncertainty quantification, advanced ML
- **Mindset**: "How do I extract knowledge from complex datasets?"

**Phase 3: Modern Framework Integration (Final Project)**
- **Character**: Cutting-edge computational tools for research applications
- **Focus**: Integration of physics understanding with modern ML frameworks
- **Skills**: JAX ecosystem, neural networks, research-level problem solving
- **Mindset**: "How do I solve research problems using state-of-the-art methods?"

### Cross-Project Integration Strategy

**Data Flow & Reuse**:
- **Projects 1→2**: Stellar properties and masses flow into cluster simulations
- **Projects 3→4→5**: Identical MCRT data enables direct statistical method comparison
- **Projects 1→6**: Stellar physics extended to multi-wavelength applications
- **All→Final**: Students can build on any previous project for final application

**Skill Accumulation**:
1. **Programming Fundamentals** (Projects 1-2) → **Advanced Algorithms** (Project 3)
2. **Statistical Foundations** (Projects 4-5) → **Advanced ML** (Project 6)
3. **Classical Methods** (Projects 1-6) → **Modern Frameworks** (Final Project)

**Physics Understanding Development**:
- **Stellar Structure** → **Stellar Systems** → **Stellar Observations** → **Data Analysis** → **Modern Applications**
- Each project adds complexity while reinforcing previous concepts

### Assessment Philosophy & Expectations

**Understanding Over Performance**:
- Students must explain every line of code they submit
- Emphasis on physical intuition and mathematical foundations
- Creativity and experimentation valued over "correct" results

**Research Preparation Focus**:
- Professional software development practices throughout
- Literature connections and scientific context for each project
- Final presentations mirror research conference talks

**Computational Thinking Development**:
- Explicit debugging challenges and optimization exercises
- Peer code review sessions for collaborative learning
- Portfolio development for career preparation

### Creative Experimentation Guidelines

**Encourage Scientific Curiosity**:
- "What happens if..." questions drive exploration
- Parameter space exploration over rigid problem sets
- Students design their own scientific experiments

**Support Creative Risk-Taking**:
- Weird parameter choices and extreme cases welcomed
- Failed experiments are learning opportunities
- Peer sharing of surprising results and discoveries

**Research Skills Integration**:
- Literature connections for each project
- Scientific communication through presentations
- Professional portfolio development

---

## Detailed Implementation Feedback & Extensions

### Project 1 Enhancement Details
**Additional Toolkit Functions**:
```python
def magnitude_system(luminosity, distance, filter_band):
    """Convert to astronomical magnitude system"""

def extinction_correction(observed_mag, A_v, extinction_law):
    """Apply dust extinction corrections"""

def stellar_density_profile(radius, stellar_type):
    """Radial density profiles for different stellar types"""
```

**Creative Extension Ideas**:
- **"Alien star systems"**: Non-solar metallicity relationships
- **"Failed stars"**: Brown dwarf parameter exploration
- **"Extreme environments"**: High-radiation or low-metallicity galaxies

### Project 2 Advanced Features
**Cluster Physics Extensions**:
- **Tidal disruption**: External gravitational fields
- **Stellar evolution effects**: Supernovae kicks, stellar winds
- **Primordial binaries**: Formation and evolution in cluster environment

**Experimental Suggestions**:
- **"Impossible clusters"**: Violate virial theorem, explore consequences
- **"Time-reversed evolution"**: Start with evolved cluster, run backward
- **"Multi-component systems"**: Different stellar populations with different IMFs

### Project 3 Advanced Radiative Transfer
**Physical Extensions**:
- **Polarization**: Track Stokes parameters through scattering
- **Time-dependent sources**: Variable stars, binary eclipse modeling
- **Complex geometries**: Disk systems, outflow cavities, clumpy media

**Computational Challenges**:
- **Variance reduction**: Importance sampling, Russian roulette
- **Parallel implementation**: Domain decomposition strategies
- **Adaptive sampling**: Focus photons in regions of interest

### Projects 4-5 Statistical Method Deep Dive
**Advanced Comparison Studies**:
- **Hierarchical modeling**: Population vs individual parameter inference
- **Model selection**: Information criteria vs Bayesian evidence
- **Computational scaling**: Performance with increasing dataset size

**Real Data Applications**:
- **Gaia stellar parameters**: Apply methods to real survey data
- **Systematic error modeling**: Handle calibration uncertainties
- **Missing data treatment**: Partial observations and selection effects

### Project 6 GP Advanced Applications
**Modern Astronomical Applications**:
- **Survey optimization**: Design optimal observing strategies
- **Real-time analysis**: Process large datasets efficiently
- **Multi-fidelity modeling**: Combine different simulation accuracies

**Research Connections**:
- **Exoplanet detection**: GP for stellar activity modeling
- **Supernova classification**: Spectroscopic analysis with uncertainties
- **Galaxy evolution**: Multi-wavelength SED fitting with GP emulators

### Final Project Research Directions
**Cutting-Edge Applications**:
- **Physics-informed neural networks**: Incorporate stellar structure equations
- **Differentiable simulations**: End-to-end optimization of simulation parameters
- **Federated learning**: Combine datasets across different observatories
- **Graph neural networks**: Analyze astronomical survey catalog structures

**Industry Preparation**:
- **MLOps practices**: Model deployment and monitoring
- **Distributed computing**: Large-scale data processing
- **Software engineering**: Production-quality code development
- **Technical communication**: Present to non-technical stakeholders

### Course Outcome Assessment

**Technical Skills Mastery**:
- Implement complex algorithms from mathematical foundations
- Understand and apply modern computational frameworks
- Develop professional software engineering practices
- Master statistical inference and machine learning techniques

**Scientific Thinking Development**:
- Translate physical understanding into computational implementations
- Design and execute scientific computational experiments
- Interpret and communicate complex technical results
- Connect computational methods to real astronomical applications

**Research Preparation**:
- Read and implement methods from research literature
- Develop original research questions and computational approaches
- Present technical work to scientific audiences
- Build professional portfolio for academic or industry careers

This comprehensive framework prepares students for the modern landscape of computational astrophysics, where deep physics understanding meets cutting-edge computational techniques. The emphasis on experimentation and creativity ensures students develop both technical skills and scientific intuition essential for research careers.