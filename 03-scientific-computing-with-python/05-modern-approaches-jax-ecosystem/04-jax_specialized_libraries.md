# JAX Specialized Libraries: BlackJAX, Numpyro, and Domain-Specific Tools

## Learning Objectives
By the end of this chapter, you will:
- Implement MCMC samplers with BlackJAX
- Build probabilistic models with Numpyro
- Use JAXopt for constrained optimization
- Apply ChemJAX for molecular dynamics
- Leverage domain-specific JAX libraries for astronomy
- Build custom JAX-based tools for your research

## BlackJAX: Modern MCMC Sampling

### Introduction to BlackJAX

```python
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import blackjax
import blackjax.smc as smc
import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple, Callable

def blackjax_fundamentals():
    """Learn BlackJAX for MCMC sampling in astronomy."""
    
    print("BLACKJAX: MODERN MCMC SAMPLING")
    print("=" * 50)
    
    # 1. Basic HMC sampling
    print("\n1. HAMILTONIAN MONTE CARLO:")
    
    # Define a cosmological likelihood
    def log_likelihood(theta):
        """Log likelihood for cosmological parameters."""
        omega_m, h0 = theta
        
        # Mock SNe Ia data
        z_data = jnp.array([0.01, 0.05, 0.1, 0.5, 1.0])
        mu_obs = jnp.array([33.0, 36.0, 38.0, 42.0, 44.0])
        mu_err = jnp.array([0.1, 0.15, 0.2, 0.25, 0.3])
        
        # Theoretical distance modulus (simplified)
        def luminosity_distance(z, om, h):
            # Simplified for flat universe
            c = 3e5  # km/s
            dL = c * z * (1 + z/2 * (1 - om))  # Taylor expansion
            return dL / h
        
        dL = vmap(lambda z: luminosity_distance(z, omega_m, h0))(z_data)
        mu_theory = 5 * jnp.log10(dL) + 25
        
        # Chi-squared
        chi2 = jnp.sum(((mu_obs - mu_theory) / mu_err) ** 2)
        return -0.5 * chi2
    
    def log_prior(theta):
        """Log prior for cosmological parameters."""
        omega_m, h0 = theta
        
        # Uniform priors
        if 0 < omega_m < 1 and 50 < h0 < 100:
            return 0.0
        return -jnp.inf
    
    def log_prob(theta):
        """Log posterior probability."""
        lp = log_prior(theta)
        if jnp.isfinite(lp):
            return lp + log_likelihood(theta)
        return lp
    
    # Initialize HMC
    key = random.PRNGKey(0)
    initial_position = jnp.array([0.3, 70.0])
    
    # Build HMC kernel
    inv_mass_matrix = jnp.array([0.01, 1.0])  # Diagonal mass matrix
    num_integration_steps = 10
    step_size = 0.01
    
    hmc = blackjax.hmc(
        log_prob,
        step_size=step_size,
        inverse_mass_matrix=inv_mass_matrix,
        num_integration_steps=num_integration_steps
    )
    
    # Initialize state
    state = hmc.init(initial_position)
    
    # Run sampling
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        """MCMC sampling loop."""
        
        @jit
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)
        
        keys = random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
        
        return states, infos
    
    key, sample_key = random.split(key)
    states, infos = inference_loop(sample_key, hmc.step, state, 1000)
    
    samples = states.position
    print(f"  Sampled {len(samples)} points")
    print(f"  Mean Ωₘ = {jnp.mean(samples[:, 0]):.3f} ± {jnp.std(samples[:, 0]):.3f}")
    print(f"  Mean H₀ = {jnp.mean(samples[:, 1]):.1f} ± {jnp.std(samples[:, 1]):.1f}")
    print(f"  Acceptance rate: {jnp.mean(infos.is_accepted):.2%}")
    
    # 2. NUTS (No U-Turn Sampler)
    print("\n2. NUTS SAMPLER:")
    
    nuts = blackjax.nuts(log_prob, step_size=step_size)
    state = nuts.init(initial_position)
    
    key, sample_key = random.split(key)
    states, infos = inference_loop(sample_key, nuts.step, state, 500)
    
    print(f"  NUTS: {len(states.position)} samples")
    print(f"  Mean tree depth: {jnp.mean(infos.num_trajectory_expansions):.1f}")
    
    # 3. Adaptive sampling
    print("\n3. ADAPTIVE SAMPLING:")
    
    # Window adaptation for step size and mass matrix
    def run_adaptive_sampling(kernel_factory, num_chains=4):
        """Run parallel chains with adaptation."""
        
        key = random.PRNGKey(42)
        keys = random.split(key, num_chains + 1)
        
        # Initialize multiple chains
        initial_positions = jnp.array([
            [0.25 + 0.1 * random.normal(keys[i], ()), 
             65.0 + 5.0 * random.normal(keys[i+1], ())]
            for i in range(num_chains)
        ])
        
        # Warmup with window adaptation
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_prob,
            num_steps=500,
            initial_step_size=0.01,
            target_acceptance_rate=0.8
        )
        
        # Run warmup
        (last_states, parameters), _ = warmup.run(keys[-1], initial_positions[0])
        
        print(f"  Adapted step size: {parameters['step_size']:.4f}")
        
        return last_states, parameters
    
    adapted_state, adapted_params = run_adaptive_sampling(blackjax.nuts)
    
    # 4. Sequential Monte Carlo
    print("\n4. SEQUENTIAL MONTE CARLO:")
    
    # Temperature schedule for tempering
    def tempered_log_prob(theta, beta):
        """Tempered posterior for SMC."""
        return beta * log_prob(theta)
    
    # SMC sampler
    def run_smc(num_particles=100):
        """Run SMC sampler."""
        key = random.PRNGKey(123)
        
        # Initial particles from prior
        init_key, smc_key = random.split(key)
        initial_particles = jnp.array([
            [random.uniform(init_key, (), minval=0.1, maxval=0.5),
             random.uniform(init_key, (), minval=60, maxval=80)]
            for _ in range(num_particles)
        ])
        
        # Temperature schedule
        betas = jnp.linspace(0, 1, 10)
        
        print(f"  SMC with {num_particles} particles, {len(betas)} temperatures")
        
        # Would run full SMC here
        return initial_particles
    
    smc_particles = run_smc()

blackjax_fundamentals()
```

### Advanced Sampling Techniques

```python
def advanced_blackjax():
    """Advanced sampling techniques with BlackJAX."""
    
    print("\nADVANCED BLACKJAX TECHNIQUES")
    print("=" * 50)
    
    # 1. Riemannian HMC
    print("\n1. RIEMANNIAN HMC:")
    
    def log_prob_with_metric(theta):
        """Log probability with position-dependent metric."""
        # Stellar population synthesis parameters
        age, metallicity = theta
        
        # Mock observables
        color_obs = 0.8
        magnitude_obs = -2.0
        
        # Model predictions (simplified)
        color_model = 0.5 + 0.1 * age - 0.2 * metallicity
        mag_model = -3.0 + 0.2 * age + 0.3 * metallicity
        
        # Position-dependent uncertainties
        sigma_color = 0.1 * (1 + 0.1 * jnp.abs(age))
        sigma_mag = 0.2 * (1 + 0.05 * jnp.abs(metallicity))
        
        log_like = -0.5 * (
            ((color_obs - color_model) / sigma_color) ** 2 +
            ((magnitude_obs - mag_model) / sigma_mag) ** 2
        )
        
        # Priors
        log_prior = -0.5 * (age ** 2 / 100 + metallicity ** 2 / 4)
        
        return log_like + log_prior
    
    # Metric tensor (Fisher information)
    def metric_fn(theta):
        """Compute metric tensor at position."""
        hess = jax.hessian(log_prob_with_metric)(theta)
        return -hess  # Fisher information
    
    # Would implement full Riemannian HMC here
    print("  Riemannian HMC configured for curved parameter space")
    
    # 2. Parallel tempering
    print("\n2. PARALLEL TEMPERING:")
    
    def parallel_tempering(log_prob, num_chains=4, num_samples=1000):
        """Parallel tempering MCMC."""
        
        # Temperature ladder
        betas = jnp.array([1.0, 0.5, 0.25, 0.1])
        
        def tempered_log_prob(theta, beta):
            return beta * log_prob(theta)
        
        # Initialize chains at different temperatures
        key = random.PRNGKey(42)
        keys = random.split(key, num_chains + 1)
        
        initial_positions = jnp.array([
            [0.3 + 0.05 * i, 70.0 + 2.0 * i] 
            for i in range(num_chains)
        ])
        
        # Build kernels for each temperature
        kernels = []
        states = []
        for i, beta in enumerate(betas):
            kernel = blackjax.hmc(
                lambda theta: tempered_log_prob(theta, beta),
                step_size=0.01 / jnp.sqrt(beta),  # Adjust step size
                inverse_mass_matrix=jnp.ones(2),
                num_integration_steps=10
            )
            kernels.append(kernel)
            states.append(kernel.init(initial_positions[i]))
        
        # Sampling with swaps
        @jit
        def swap_step(states, key):
            """Propose and accept/reject swaps."""
            swap_key, accept_key = random.split(key)
            
            # Random pair to swap
            pair = random.choice(swap_key, num_chains - 1)
            
            # Compute swap acceptance probability
            theta_i = states[pair].position
            theta_j = states[pair + 1].position
            
            log_prob_i = tempered_log_prob(theta_i, betas[pair])
            log_prob_j = tempered_log_prob(theta_j, betas[pair + 1])
            
            log_prob_i_swap = tempered_log_prob(theta_j, betas[pair])
            log_prob_j_swap = tempered_log_prob(theta_i, betas[pair + 1])
            
            log_alpha = (log_prob_i_swap + log_prob_j_swap - 
                        log_prob_i - log_prob_j)
            
            # Accept/reject
            accept = random.uniform(accept_key) < jnp.exp(log_alpha)
            
            # Swap if accepted (simplified - would use lax.cond)
            return states, accept
        
        print(f"  Parallel tempering with {num_chains} chains")
        print(f"  Temperatures: {1/betas}")
        
        return states
    
    # Example cosmology log prob
    def cosmo_log_prob(theta):
        omega_m, h0 = theta
        if 0 < omega_m < 1 and 50 < h0 < 100:
            return -0.5 * ((omega_m - 0.3)**2 / 0.01 + (h0 - 70)**2 / 25)
        return -jnp.inf
    
    pt_states = parallel_tempering(cosmo_log_prob)
    
    # 3. Ensemble samplers
    print("\n3. ENSEMBLE SAMPLERS:")
    
    def affine_invariant_ensemble_sampler(log_prob, num_walkers=32):
        """Affine invariant ensemble sampler (like emcee)."""
        
        key = random.PRNGKey(99)
        ndim = 2
        
        # Initialize walkers
        initial_positions = random.normal(key, (num_walkers, ndim))
        
        @jit
        def stretch_move(positions, key):
            """Stretch move for ensemble sampler."""
            n_walkers = len(positions)
            keys = random.split(key, n_walkers + 2)
            
            # Random pairs
            pairs = random.choice(keys[0], n_walkers, (n_walkers,))
            
            # Stretch factors
            a = 2.0  # Stretch scale
            z = ((a - 1) * random.uniform(keys[1], (n_walkers,)) + 1) ** 2 / a
            
            # Propose new positions
            proposals = positions[pairs] + z[:, None] * (positions - positions[pairs])
            
            # Compute acceptance
            log_probs_old = vmap(log_prob)(positions)
            log_probs_new = vmap(log_prob)(proposals)
            
            log_alpha = (ndim - 1) * jnp.log(z) + log_probs_new - log_probs_old
            
            # Accept/reject
            accept = random.uniform(keys[2], (n_walkers,)) < jnp.exp(log_alpha)
            
            new_positions = jnp.where(accept[:, None], proposals, positions)
            
            return new_positions, accept
        
        # Run ensemble
        positions = initial_positions
        for i in range(100):
            key, step_key = random.split(key)
            positions, accept = stretch_move(positions, step_key)
        
        print(f"  Ensemble sampler with {num_walkers} walkers")
        print(f"  Mean position: {jnp.mean(positions, axis=0)}")
        
        return positions
    
    ensemble_samples = affine_invariant_ensemble_sampler(cosmo_log_prob)

advanced_blackjax()
```

## Numpyro: Probabilistic Programming

### Building Probabilistic Models

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

def numpyro_fundamentals():
    """Probabilistic programming with Numpyro."""
    
    print("\nNUMPYRO: PROBABILISTIC PROGRAMMING")
    print("=" * 50)
    
    # 1. Basic hierarchical model
    print("\n1. HIERARCHICAL MODEL - CEPHEID CALIBRATION:")
    
    def cepheid_model(periods, magnitudes=None):
        """
        Hierarchical model for Cepheid period-luminosity relation.
        Different galaxies have different zero points.
        """
        n_obs = len(periods)
        
        # Hyperpriors for P-L relation
        alpha = numpyro.sample('alpha', dist.Normal(-2.5, 0.5))
        beta = numpyro.sample('beta', dist.Normal(-3.0, 0.5))
        
        # Intrinsic scatter
        sigma_int = numpyro.sample('sigma_int', dist.HalfNormal(0.1))
        
        # Galaxy-specific zero points (distance moduli)
        # Assume we have galaxy IDs
        n_galaxies = 5
        galaxy_ids = jnp.array([i % n_galaxies for i in range(n_obs)])
        
        with numpyro.plate('galaxies', n_galaxies):
            mu_gal = numpyro.sample('mu_gal', dist.Normal(30.0, 2.0))
        
        # Expected magnitude
        log_P = jnp.log10(periods)
        M_expected = alpha * log_P + beta  # Absolute magnitude
        m_expected = M_expected + mu_gal[galaxy_ids]  # Apparent magnitude
        
        # Observational uncertainty
        sigma_obs = 0.05
        sigma_total = jnp.sqrt(sigma_int**2 + sigma_obs**2)
        
        # Likelihood
        with numpyro.plate('observations', n_obs):
            numpyro.sample('magnitudes', 
                          dist.Normal(m_expected, sigma_total),
                          obs=magnitudes)
    
    # Generate synthetic data
    key = random.PRNGKey(0)
    n_cepheids = 100
    true_alpha = -2.43
    true_beta = -3.05
    
    periods = jnp.exp(random.uniform(key, (n_cepheids,), minval=0, maxval=3))
    galaxy_ids = jnp.array([i % 5 for i in range(n_cepheids)])
    true_mu = jnp.array([29.5, 30.0, 30.5, 31.0, 31.5])
    
    log_P = jnp.log10(periods)
    M_true = true_alpha * log_P + true_beta
    m_true = M_true + true_mu[galaxy_ids]
    
    key, noise_key = random.split(key)
    observed_mags = m_true + 0.05 * random.normal(noise_key, (n_cepheids,))
    
    # Run MCMC
    nuts_kernel = NUTS(cepheid_model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    
    key, run_key = random.split(key)
    mcmc.run(run_key, periods, magnitudes=observed_mags)
    
    samples = mcmc.get_samples()
    print(f"  α = {jnp.mean(samples['alpha']):.3f} ± {jnp.std(samples['alpha']):.3f}")
    print(f"  β = {jnp.mean(samples['beta']):.3f} ± {jnp.std(samples['beta']):.3f}")
    
    # 2. Mixture models
    print("\n2. MIXTURE MODEL - STELLAR POPULATIONS:")
    
    def stellar_population_mixture(colors=None, magnitudes=None):
        """
        Mixture model for stellar populations.
        Main sequence, giants, and white dwarfs.
        """
        n_obs = len(colors) if colors is not None else 100
        
        # Mixture weights
        weights = numpyro.sample('weights', dist.Dirichlet(jnp.ones(3)))
        
        # Component parameters
        with numpyro.plate('components', 3):
            mu_color = numpyro.sample('mu_color', dist.Normal(0.0, 2.0))
            mu_mag = numpyro.sample('mu_mag', dist.Normal(0.0, 5.0))
            sigma_color = numpyro.sample('sigma_color', dist.HalfNormal(0.5))
            sigma_mag = numpyro.sample('sigma_mag', dist.HalfNormal(1.0))
        
        # Mixture assignment
        with numpyro.plate('stars', n_obs):
            assignment = numpyro.sample('assignment', dist.Categorical(weights))
            
            # Observed color and magnitude
            numpyro.sample('colors',
                          dist.Normal(mu_color[assignment], sigma_color[assignment]),
                          obs=colors)
            numpyro.sample('magnitudes',
                          dist.Normal(mu_mag[assignment], sigma_mag[assignment]),
                          obs=magnitudes)
    
    # Generate synthetic CMD data
    key, data_key = random.split(key)
    n_stars = 200
    
    # True mixture
    true_weights = jnp.array([0.7, 0.2, 0.1])  # MS, Giants, WDs
    true_mu_color = jnp.array([0.5, 1.5, -0.5])
    true_mu_mag = jnp.array([5.0, 0.0, 10.0])
    
    assignments = random.categorical(data_key, jnp.log(true_weights), shape=(n_stars,))
    
    keys = random.split(data_key, 3)
    obs_colors = true_mu_color[assignments] + 0.2 * random.normal(keys[0], (n_stars,))
    obs_mags = true_mu_mag[assignments] + 0.5 * random.normal(keys[1], (n_stars,))
    
    print("  Mixture model for CMD analysis configured")
    
    # 3. Time series model
    print("\n3. TIME SERIES - QUASAR VARIABILITY:")
    
    def quasar_drw_model(times, fluxes=None):
        """
        Damped Random Walk model for quasar variability.
        """
        n_obs = len(times)
        dt = jnp.diff(times)
        
        # DRW parameters
        tau = numpyro.sample('tau', dist.LogNormal(jnp.log(100), 0.5))
        sigma = numpyro.sample('sigma', dist.HalfNormal(0.1))
        
        # Mean flux
        mean_flux = numpyro.sample('mean_flux', dist.Normal(0, 1))
        
        # Initial state
        flux_0 = numpyro.sample('flux_0', dist.Normal(mean_flux, sigma))
        
        # Evolution
        def transition(carry, dt):
            flux_prev = carry
            
            # DRW transition
            decay = jnp.exp(-dt / tau)
            noise_var = sigma**2 * (1 - decay**2)
            
            flux_next = numpyro.sample('flux_next',
                                       dist.Normal(mean_flux + decay * (flux_prev - mean_flux),
                                                  jnp.sqrt(noise_var)))
            return flux_next, flux_next
        
        # Scan through time series
        _, flux_trajectory = jax.lax.scan(transition, flux_0, dt)
        flux_trajectory = jnp.concatenate([jnp.array([flux_0]), flux_trajectory])
        
        # Observational uncertainty
        obs_error = 0.01
        
        with numpyro.plate('observations', n_obs):
            numpyro.sample('fluxes',
                          dist.Normal(flux_trajectory, obs_error),
                          obs=fluxes)
    
    print("  DRW model for quasar variability created")

numpyro_fundamentals()
```

### Variational Inference with Numpyro

```python
def numpyro_variational_inference():
    """Variational inference for large-scale problems."""
    
    print("\nVARIATIONAL INFERENCE WITH NUMPYRO")
    print("=" * 50)
    
    # 1. SVI for galaxy clustering
    print("\n1. SVI FOR GALAXY CLUSTERING:")
    
    def galaxy_clustering_model(positions, redshifts=None):
        """
        Hierarchical model for galaxy clustering.
        """
        n_galaxies = len(positions)
        
        # Cosmological parameters
        omega_m = numpyro.sample('omega_m', dist.Uniform(0.1, 0.5))
        sigma_8 = numpyro.sample('sigma_8', dist.Uniform(0.5, 1.0))
        
        # Bias parameters for different galaxy types
        n_types = 3
        with numpyro.plate('galaxy_types', n_types):
            bias = numpyro.sample('bias', dist.LogNormal(0, 0.5))
        
        # Galaxy type assignments (latent)
        with numpyro.plate('galaxies', n_galaxies):
            galaxy_type = numpyro.sample('galaxy_type',
                                         dist.Categorical(jnp.ones(n_types) / n_types))
            
            # Simplified clustering likelihood
            # In reality, would use correlation function
            clustering_strength = bias[galaxy_type] * sigma_8
            
            # Observed redshift (with peculiar velocities)
            if redshifts is not None:
                z_cosmo = jnp.linalg.norm(positions, axis=1) / 3000  # Hubble flow
                z_pec = clustering_strength * random.normal(random.PRNGKey(0), (n_galaxies,)) * 0.001
                
                numpyro.sample('redshifts',
                              dist.Normal(z_cosmo + z_pec, 0.0001),
                              obs=redshifts)
    
    # Generate mock data
    key = random.PRNGKey(42)
    n_gal = 1000
    positions = random.normal(key, (n_gal, 3)) * 100
    redshifts = jnp.linalg.norm(positions, axis=1) / 3000 + \
                0.001 * random.normal(key, (n_gal,))
    
    # Setup SVI
    guide = AutoNormal(galaxy_clustering_model)
    optimizer = numpyro.optim.Adam(step_size=0.01)
    svi = SVI(galaxy_clustering_model, guide, optimizer, loss=Trace_ELBO())
    
    # Run optimization
    svi_result = svi.run(random.PRNGKey(0), 1000, positions, redshifts)
    
    params = svi_result.params
    print(f"  SVI converged with loss: {svi_result.losses[-1]:.2f}")
    
    # 2. Custom guides
    print("\n2. CUSTOM VARIATIONAL GUIDES:")
    
    def custom_guide(positions, redshifts=None):
        """Custom guide for more control."""
        n_galaxies = len(positions)
        
        # Cosmological parameters with custom distributions
        omega_m_loc = numpyro.param('omega_m_loc', 0.3)
        omega_m_scale = numpyro.param('omega_m_scale', 0.01,
                                      constraint=dist.constraints.positive)
        omega_m = numpyro.sample('omega_m', dist.Normal(omega_m_loc, omega_m_scale))
        
        sigma_8_loc = numpyro.param('sigma_8_loc', 0.8)
        sigma_8_scale = numpyro.param('sigma_8_scale', 0.01,
                                      constraint=dist.constraints.positive)
        sigma_8 = numpyro.sample('sigma_8', dist.Normal(sigma_8_loc, sigma_8_scale))
        
        # Galaxy bias parameters
        with numpyro.plate('galaxy_types', 3):
            bias_loc = numpyro.param('bias_loc', jnp.ones(3))
            bias_scale = numpyro.param('bias_scale', 0.1 * jnp.ones(3),
                                       constraint=dist.constraints.positive)
            numpyro.sample('bias', dist.LogNormal(bias_loc, bias_scale))
        
        # Galaxy type assignments (discrete - use Gumbel-softmax)
        with numpyro.plate('galaxies', n_galaxies):
            type_logits = numpyro.param('type_logits', jnp.zeros((n_galaxies, 3)))
            numpyro.sample('galaxy_type', dist.CategoricalLogits(type_logits))
    
    print("  Custom variational guide configured")
    
    # 3. Normalizing flows
    print("\n3. NORMALIZING FLOWS FOR COMPLEX POSTERIORS:")
    
    from numpyro.infer.autoguide import AutoIAFNormal
    
    def complex_posterior_model(data=None):
        """Model with complex, multimodal posterior."""
        # Bimodal distribution
        mode = numpyro.sample('mode', dist.Bernoulli(0.5))
        
        # Different parameters for each mode
        with numpyro.handlers.mask(mask=mode):
            theta1 = numpyro.sample('theta1', dist.Normal(2.0, 0.5))
        
        with numpyro.handlers.mask(mask=~mode):
            theta2 = numpyro.sample('theta2', dist.Normal(-2.0, 0.5))
        
        # Combine
        theta = jnp.where(mode, theta1, theta2)
        
        # Likelihood
        numpyro.sample('data', dist.Normal(theta, 0.1), obs=data)
    
    # Use IAF (Inverse Autoregressive Flow)
    flow_guide = AutoIAFNormal(complex_posterior_model, num_flows=3)
    
    print("  Normalizing flow guide created for complex posteriors")

numpyro_variational_inference()
```

## JAXopt: Constrained Optimization

### Optimization with Constraints

```python
import jaxopt

def jaxopt_optimization():
    """Constrained optimization for astronomical problems."""
    
    print("\nJAXOPT: CONSTRAINED OPTIMIZATION")
    print("=" * 50)
    
    # 1. Constrained least squares
    print("\n1. CONSTRAINED ORBIT FITTING:")
    
    def orbit_objective(params, observations):
        """Objective for orbit fitting."""
        a, e, i, omega = params
        times, ra_obs, dec_obs = observations
        
        # Kepler orbit (simplified)
        M = 2 * jnp.pi * times / (a ** 1.5)
        E = M  # Small eccentricity approximation
        
        # Projected positions
        x = a * (jnp.cos(E) - e)
        y = a * jnp.sqrt(1 - e**2) * jnp.sin(E)
        
        # Rotate by inclination and argument
        ra_model = x * jnp.cos(omega) - y * jnp.sin(omega) * jnp.cos(i)
        dec_model = x * jnp.sin(omega) + y * jnp.cos(omega) * jnp.cos(i)
        
        # Residuals
        residuals = jnp.sum((ra_obs - ra_model)**2 + (dec_obs - dec_model)**2)
        return residuals
    
    # Constraints: physical orbital elements
    def orbit_constraints(params):
        a, e, i, omega = params
        return jnp.array([
            a - 0.1,      # a > 0.1 AU
            10.0 - a,     # a < 10 AU
            e,            # e >= 0
            1.0 - e,      # e < 1
            i,            # i >= 0
            jnp.pi - i,   # i <= pi
        ])
    
    # Generate mock observations
    key = random.PRNGKey(123)
    times = jnp.linspace(0, 10, 50)
    true_params = jnp.array([2.0, 0.3, 0.5, 1.0])
    
    # Mock data (using true model)
    M_true = 2 * jnp.pi * times / (true_params[0] ** 1.5)
    x_true = true_params[0] * jnp.cos(M_true)
    y_true = true_params[0] * jnp.sin(M_true) * jnp.sqrt(1 - true_params[1]**2)
    
    ra_obs = x_true + 0.01 * random.normal(key, x_true.shape)
    dec_obs = y_true + 0.01 * random.normal(key, y_true.shape)
    observations = (times, ra_obs, dec_obs)
    
    # Solve with constraints
    initial_params = jnp.array([1.5, 0.2, 0.4, 0.8])
    
    solver = jaxopt.ProjectedGradient(
        fun=lambda p: orbit_objective(p, observations),
        projection=jaxopt.projection.projection_box,
        maxiter=100
    )
    
    # Box constraints
    lower_bounds = jnp.array([0.1, 0.0, 0.0, 0.0])
    upper_bounds = jnp.array([10.0, 0.99, jnp.pi, 2*jnp.pi])
    
    result = solver.run(initial_params, 
                        hyperparams_proj=(lower_bounds, upper_bounds))
    
    print(f"  Optimization converged: {result.state.error < 1e-3}")
    print(f"  True params: {true_params}")
    print(f"  Fitted params: {result.params}")
    
    # 2. Quadratic programming
    print("\n2. QUADRATIC PROGRAMMING - TELESCOPE SCHEDULING:")
    
    def telescope_scheduling():
        """Optimize telescope observation schedule."""
        
        # Problem: maximize scientific value subject to constraints
        # Variables: time allocated to each target
        n_targets = 10
        
        # Scientific value (priority) of each target
        values = jnp.array([9, 8, 7, 6, 5, 5, 4, 3, 2, 1])
        
        # Observability windows (hours)
        max_observable = jnp.array([3, 4, 2, 5, 3, 2, 4, 3, 2, 1])
        
        # Quadratic objective (negative for maximization)
        # Include diversity bonus for observing multiple targets
        Q = -jnp.eye(n_targets) * 0.1  # Diversity penalty
        c = -values  # Linear term (negative for maximization)
        
        # Constraints: Ax <= b
        # Total time constraint
        A_time = jnp.ones((1, n_targets))
        b_time = jnp.array([8.0])  # 8 hours total
        
        # Combine with observability constraints
        A_obs = jnp.eye(n_targets)
        b_obs = max_observable
        
        A = jnp.vstack([A_time, A_obs])
        b = jnp.concatenate([b_time, b_obs])
        
        # Also need non-negativity: x >= 0
        
        # Solve QP
        qp_solver = jaxopt.QuadraticProgramming()
        
        # Initial guess
        x_init = jnp.ones(n_targets) * 0.5
        
        solution = qp_solver.run(x_init, params_obj=(Q, c),
                                 params_ineq=(A, b))
        
        schedule = solution.params
        
        print(f"  Telescope schedule optimized")
        print(f"  Total time used: {jnp.sum(schedule):.1f} hours")
        print(f"  Targets observed: {jnp.sum(schedule > 0.01)}")
        print(f"  Scientific value: {jnp.dot(values, schedule):.1f}")
        
        return schedule
    
    schedule = telescope_scheduling()
    
    # 3. Proximal gradient methods
    print("\n3. PROXIMAL METHODS - SPARSE DECONVOLUTION:")
    
    def sparse_deconvolution():
        """Deconvolve image with sparsity constraint."""
        
        # Create blurred image
        key = random.PRNGKey(456)
        true_image = jnp.zeros((50, 50))
        
        # Add point sources
        n_sources = 10
        positions = random.choice(key, 50*50, (n_sources,), replace=False)
        true_image = true_image.flatten()
        true_image = true_image.at[positions].set(
            random.uniform(key, (n_sources,), minval=0.5, maxval=2.0)
        )
        true_image = true_image.reshape(50, 50)
        
        # Blur with PSF
        def create_psf(size=5):
            x = jnp.arange(size) - size // 2
            X, Y = jnp.meshgrid(x, x)
            psf = jnp.exp(-(X**2 + Y**2) / 2)
            return psf / jnp.sum(psf)
        
        psf = create_psf()
        
        # Convolve (simplified - using scipy in practice)
        blurred = jax.scipy.signal.convolve2d(true_image, psf, mode='same')
        
        # Add noise
        observed = blurred + 0.01 * random.normal(key, blurred.shape)
        
        # Deconvolution with L1 regularization for sparsity
        def objective(x):
            # Data fidelity
            convolved = jax.scipy.signal.convolve2d(x.reshape(50, 50), psf, mode='same')
            fidelity = 0.5 * jnp.sum((convolved - observed) ** 2)
            
            # L1 regularization
            sparsity = 0.01 * jnp.sum(jnp.abs(x))
            
            return fidelity + sparsity
        
        # Use proximal gradient
        prox_solver = jaxopt.ProximalGradient(
            fun=objective,
            prox=jaxopt.prox.prox_lasso,
            maxiter=100
        )
        
        x_init = observed.flatten()
        result = prox_solver.run(x_init)
        
        deconvolved = result.params.reshape(50, 50)
        
        print(f"  Sparse deconvolution completed")
        print(f"  Sparsity: {jnp.sum(jnp.abs(deconvolved) > 0.01)} non-zero pixels")
        
        return deconvolved
    
    deconvolved_image = sparse_deconvolution()

jaxopt_optimization()
```

## Domain-Specific Libraries

### Astronomical JAX Tools

```python
def astronomical_jax_libraries():
    """Specialized JAX libraries for astronomy."""
    
    print("\nASTRONOMICAL JAX LIBRARIES")
    print("=" * 50)
    
    # 1. s2fft - Spherical harmonic transforms
    print("\n1. S2FFT - SPHERICAL HARMONICS FOR CMB:")
    
    # Note: This is conceptual - s2fft would need to be installed
    def cmb_analysis():
        """Analyze CMB maps using spherical harmonics."""
        
        # Generate mock CMB map
        nside = 64  # HEALPix resolution
        npix = 12 * nside**2
        
        key = random.PRNGKey(789)
        
        # Generate random alm coefficients (simplified)
        lmax = 100
        n_alm = (lmax + 1) * (lmax + 2) // 2
        
        # Power spectrum (simplified)
        ell = jnp.arange(lmax + 1)
        Cl = 100 / (ell + 10) ** 2  # Simplified CMB spectrum
        
        # Random realization
        alm_real = jnp.sqrt(Cl[None, :]) * random.normal(key, (1, lmax + 1))
        
        print(f"  CMB analysis with lmax={lmax}")
        print(f"  Power spectrum computed")
        
        # Would use s2fft for actual transforms
        # import s2fft
        # map_data = s2fft.inverse_transform(alm_real, L=lmax)
        
        return alm_real
    
    cmb_alm = cmb_analysis()
    
    # 2. jax-cosmo - Cosmological calculations
    print("\n2. JAX-COSMO - COSMOLOGICAL CALCULATIONS:")
    
    def cosmological_calculations():
        """Differentiable cosmology calculations."""
        
        # Cosmological parameters
        cosmo_params = {
            'omega_m': 0.3,
            'omega_de': 0.7,
            'h': 0.7,
            'omega_b': 0.05,
            'n_s': 0.96,
            'sigma_8': 0.8
        }
        
        # Redshifts
        z = jnp.linspace(0, 2, 100)
        
        # Comoving distance (simplified Friedmann equation)
        def comoving_distance(z, omega_m, omega_de):
            def integrand(z):
                return 1 / jnp.sqrt(omega_m * (1 + z)**3 + omega_de)
            
            # Integrate (simplified - use quadrature in practice)
            dz = z[1] - z[0]
            integral = jnp.cumsum(vmap(integrand)(z)) * dz
            
            c_over_H0 = 3000 / 0.7  # Mpc
            return c_over_H0 * integral
        
        d_c = comoving_distance(z, cosmo_params['omega_m'], cosmo_params['omega_de'])
        
        # Angular diameter distance
        d_a = d_c / (1 + z)
        
        # Luminosity distance
        d_l = d_c * (1 + z)
        
        print(f"  Computed distances for {len(z)} redshifts")
        print(f"  Max comoving distance: {d_c[-1]:.0f} Mpc")
        
        # Growth function (simplified)
        def growth_factor(z, omega_m):
            a = 1 / (1 + z)
            # Approximate growth factor
            omega_m_z = omega_m / (omega_m + (1 - omega_m) * a**3)
            return a * omega_m_z**0.55
        
        D_growth = vmap(lambda zi: growth_factor(zi, cosmo_params['omega_m']))(z)
        
        print(f"  Growth factor at z=0: {D_growth[0]:.3f}")
        
        return d_c, d_a, d_l, D_growth
    
    distances = cosmological_calculations()
    
    # 3. Neural density estimation
    print("\n3. NEURAL DENSITY ESTIMATION FOR INFERENCE:")
    
    def neural_posterior_estimation():
        """Use neural networks for likelihood-free inference."""
        
        import flax.linen as nn
        
        class PosteriorNetwork(nn.Module):
            """Neural network to approximate posterior."""
            
            @nn.compact
            def __call__(self, data, params):
                # Concatenate data and parameters
                x = jnp.concatenate([data, params])
                
                # Deep network
                x = nn.Dense(128)(x)
                x = nn.relu(x)
                x = nn.Dense(128)(x)
                x = nn.relu(x)
                x = nn.Dense(64)(x)
                x = nn.relu(x)
                
                # Output: parameters of posterior distribution
                mean = nn.Dense(len(params))(x)
                log_std = nn.Dense(len(params))(x)
                
                return mean, log_std
        
        print("  Neural posterior estimator configured")
        
        # Would train on simulated data
        # This enables likelihood-free inference for complex simulators
        
        return PosteriorNetwork()
    
    npe_model = neural_posterior_estimation()

astronomical_jax_libraries()
```

## Complete Example: Gravitational Wave Analysis

### End-to-End GW Pipeline

```python
def gravitational_wave_pipeline():
    """
    Complete gravitational wave analysis pipeline.
    Combines multiple JAX libraries for production analysis.
    """
    
    print("\nGRAVITATIONAL WAVE ANALYSIS PIPELINE")
    print("=" * 50)
    
    # 1. Generate GW signal
    def generate_gw_signal(params, times):
        """Generate gravitational wave signal."""
        m1, m2, d_l, t_c, phi_c = params
        
        # Chirp mass
        M_chirp = (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)
        
        # Frequency evolution (simplified)
        t_to_merger = t_c - times
        f_gw = jnp.where(
            t_to_merger > 0,
            (M_chirp / t_to_merger) ** (3/8),
            0.0
        )
        
        # Amplitude
        amplitude = (M_chirp ** (5/3) * f_gw ** (2/3)) / d_l
        
        # Phase
        phase = 2 * jnp.pi * jnp.cumsum(f_gw) * 0.001 + phi_c
        
        # Strain
        h_plus = amplitude * jnp.cos(phase)
        h_cross = amplitude * jnp.sin(phase)
        
        return h_plus, h_cross
    
    # 2. Detector response
    def detector_response(h_plus, h_cross, detector_params):
        """Compute detector response to GW signal."""
        F_plus, F_cross, psi = detector_params
        
        strain = F_plus * h_plus + F_cross * h_cross
        return strain
    
    # 3. Likelihood function
    def log_likelihood(params, data, times, noise_psd):
        """Log likelihood for GW parameters."""
        h_plus, h_cross = generate_gw_signal(params, times)
        
        # Detector response (simplified - single detector)
        detector_params = (0.5, 0.5, 0.0)  # Antenna patterns
        h_model = detector_response(h_plus, h_cross, detector_params)
        
        # Matched filter in frequency domain
        h_data_fft = jnp.fft.rfft(data)
        h_model_fft = jnp.fft.rfft(h_model)
        
        # Inner product weighted by noise
        inner_product = jnp.sum(
            jnp.conj(h_data_fft) * h_model_fft / noise_psd
        ).real
        
        # Normalization
        norm = jnp.sum(jnp.abs(h_model_fft) ** 2 / noise_psd)
        
        return inner_product - 0.5 * norm
    
    # 4. Generate mock data
    key = random.PRNGKey(2024)
    n_samples = 4096
    times = jnp.linspace(0, 1, n_samples)
    
    # True parameters
    true_params = jnp.array([
        30.0,   # m1 (solar masses)
        25.0,   # m2
        100.0,  # luminosity distance (Mpc)
        0.5,    # coalescence time
        0.0     # phase
    ])
    
    # Generate signal
    h_plus_true, h_cross_true = generate_gw_signal(true_params, times)
    signal = detector_response(h_plus_true, h_cross_true, (0.5, 0.5, 0.0))
    
    # Add noise
    noise_psd = jnp.ones(n_samples // 2 + 1)  # White noise (simplified)
    noise = 1e-21 * random.normal(key, (n_samples,))
    data = signal + noise
    
    print(f"  Generated GW signal with SNR ≈ {jnp.max(jnp.abs(signal)) / jnp.std(noise):.1f}")
    
    # 5. Parameter estimation with Numpyro
    def gw_model(data, times):
        """Numpyro model for GW parameter estimation."""
        
        # Priors
        m1 = numpyro.sample('m1', dist.Uniform(5, 50))
        m2 = numpyro.sample('m2', dist.Uniform(5, 50))
        d_l = numpyro.sample('d_l', dist.Uniform(10, 500))
        t_c = numpyro.sample('t_c', dist.Uniform(0.4, 0.6))
        phi_c = numpyro.sample('phi_c', dist.Uniform(0, 2 * jnp.pi))
        
        params = jnp.array([m1, m2, d_l, t_c, phi_c])
        
        # Likelihood (simplified)
        h_plus, h_cross = generate_gw_signal(params, times)
        h_model = detector_response(h_plus, h_cross, (0.5, 0.5, 0.0))
        
        # Gaussian likelihood
        sigma = 1e-21
        numpyro.sample('data', dist.Normal(h_model, sigma), obs=data)
    
    # Run MCMC
    print("\n  Running parameter estimation...")
    
    nuts_kernel = NUTS(gw_model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    
    key, run_key = random.split(key)
    mcmc.run(run_key, data, times)
    
    samples = mcmc.get_samples()
    
    print(f"\n  Parameter estimates:")
    for param in ['m1', 'm2', 'd_l', 't_c']:
        mean = jnp.mean(samples[param])
        std = jnp.std(samples[param])
        true = true_params[['m1', 'm2', 'd_l', 't_c'].index(param)]
        print(f"    {param}: {mean:.2f} ± {std:.2f} (true: {true:.2f})")
    
    # 6. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Signal
    axes[0, 0].plot(times, data, 'gray', alpha=0.5, label='Data')
    axes[0, 0].plot(times, signal, 'b', label='True signal')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Strain')
    axes[0, 0].set_title('Gravitational Wave Signal')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0.45, 0.55)
    
    # Mass posterior
    axes[0, 1].scatter(samples['m1'], samples['m2'], alpha=0.1, s=1)
    axes[0, 1].scatter(true_params[0], true_params[1], c='r', s=100, marker='*')
    axes[0, 1].set_xlabel('m₁ (M☉)')
    axes[0, 1].set_ylabel('m₂ (M☉)')
    axes[0, 1].set_title('Mass Posterior')
    
    # Distance posterior
    axes[1, 0].hist(samples['d_l'], bins=30, alpha=0.7)
    axes[1, 0].axvline(true_params[2], c='r', ls='--', label='True')
    axes[1, 0].set_xlabel('Luminosity Distance (Mpc)')
    axes[1, 0].set_ylabel('Samples')
    axes[1, 0].set_title('Distance Posterior')
    axes[1, 0].legend()
    
    # Coalescence time
    axes[1, 1].hist(samples['t_c'], bins=30, alpha=0.7)
    axes[1, 1].axvline(true_params[3], c='r', ls='--', label='True')
    axes[1, 1].set_xlabel('Coalescence Time (s)')
    axes[1, 1].set_ylabel('Samples')
    axes[1, 1].set_title('Merger Time Posterior')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return samples

# Run complete pipeline
gw_samples = gravitational_wave_pipeline()
```

## Key Takeaways

✅ **BlackJAX** - State-of-the-art MCMC samplers with HMC, NUTS, and more  
✅ **Numpyro** - Probabilistic programming with automatic inference  
✅ **JAXopt** - Constrained optimization and proximal methods  
✅ **Domain-specific** - Growing ecosystem of astronomical JAX tools  
✅ **Integration** - Libraries work together for complete pipelines  
✅ **Performance** - Orders of magnitude faster than traditional tools  
✅ **Differentiable** - Gradients everywhere enable new methods  

## Next Steps

With these specialized libraries, you can:
1. Replace traditional MCMC codes with faster JAX versions
2. Build differentiable simulators for likelihood-free inference  
3. Implement neural posterior estimation
4. Create custom domain-specific tools
5. Scale to massive datasets with GPU acceleration

The JAX ecosystem continues to grow rapidly, with new libraries appearing regularly for specialized scientific applications!