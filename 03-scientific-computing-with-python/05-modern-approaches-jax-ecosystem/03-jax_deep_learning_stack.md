# ⚠️ JAX Deep Learning Stack: Flax, Optax, and Orbax

## Learning Objectives
By the end of this chapter, you will:
- Build large-scale neural networks with Flax
- Implement advanced optimization strategies with Optax
- Manage checkpoints and experiment tracking with Orbax
- Train transformer models for astronomical applications
- Implement distributed training across multiple GPUs
- Build production ML pipelines for astronomy

## Flax: Scalable Neural Networks

### Introduction to Flax

```python
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
from flax.training import train_state
from flax.core import freeze, unfreeze
import optax
import orbax.checkpoint as ocp
from typing import Any, Callable, Sequence, Optional
import matplotlib.pyplot as plt
import numpy as np

def flax_fundamentals():
    """Learn Flax's approach to neural network design."""
    
    print("FLAX: SCALABLE NEURAL NETWORKS")
    print("=" * 50)
    
    # 1. Basic Flax module
    print("\n1. BASIC FLAX MODULE:")
    
    class SpectralClassifier(nn.Module):
        """Classify astronomical spectra."""
        
        features: Sequence[int]
        dropout_rate: float = 0.1
        
        @nn.compact
        def __call__(self, x, training: bool = False):
            # Input: (batch, wavelengths)
            
            for i, feat in enumerate(self.features):
                x = nn.Dense(feat)(x)
                
                # Batch normalization
                x = nn.BatchNorm(use_running_average=not training)(x)
                
                # Activation
                x = nn.relu(x)
                
                # Dropout
                if training:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            
            # Output layer (7 stellar classes)
            x = nn.Dense(7)(x)
            return x
    
    # Initialize model
    model = SpectralClassifier(features=[128, 64, 32])
    
    # Create dummy input
    key = random.PRNGKey(0)
    dummy_input = jnp.ones((1, 1000))  # (batch, wavelengths)
    
    # Initialize parameters
    params = model.init(key, dummy_input)
    
    # Forward pass
    output = model.apply(params, dummy_input, training=False)
    
    print(f"  Model initialized")
    print(f"  Parameter tree structure: {jax.tree_map(lambda x: x.shape, params)}")
    print(f"  Output shape: {output.shape}")
    
    # 2. Advanced architectures
    print("\n2. CONVOLUTIONAL NETWORK FOR IMAGES:")
    
    class GalaxyMorphologyNet(nn.Module):
        """Classify galaxy morphology from images."""
        
        @nn.compact
        def __call__(self, x, training: bool = False):
            # Input: (batch, height, width, channels)
            
            # Convolutional blocks
            x = nn.Conv(features=32, kernel_size=(3, 3))(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            
            x = nn.Conv(features=64, kernel_size=(3, 3))(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            
            x = nn.Conv(features=128, kernel_size=(3, 3))(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            
            # Global average pooling
            x = jnp.mean(x, axis=(1, 2))
            
            # Dense layers
            x = nn.Dense(64)(x)
            x = nn.relu(x)
            
            if training:
                x = nn.Dropout(rate=0.3)(x, deterministic=False)
            
            # Output: Hubble types (E, S0, Sa, Sb, Sc, Irr)
            x = nn.Dense(6)(x)
            
            return x
    
    # Initialize CNN
    cnn_model = GalaxyMorphologyNet()
    dummy_image = jnp.ones((1, 128, 128, 3))
    cnn_params = cnn_model.init(key, dummy_image)
    
    print(f"  CNN initialized for 128x128 RGB images")
    
    # 3. Attention mechanisms
    print("\n3. ATTENTION FOR TIME SERIES:")
    
    class LightCurveTransformer(nn.Module):
        """Transformer for variable star classification."""
        
        embed_dim: int = 256
        num_heads: int = 8
        num_layers: int = 4
        mlp_dim: int = 512
        dropout: float = 0.1
        
        @nn.compact
        def __call__(self, times, fluxes, training: bool = False):
            # Input: (batch, sequence_length)
            batch_size, seq_len = fluxes.shape
            
            # Positional encoding using observation times
            time_embed = nn.Dense(self.embed_dim)(times[:, :, None])
            flux_embed = nn.Dense(self.embed_dim)(fluxes[:, :, None])
            
            x = time_embed + flux_embed
            
            # Transformer blocks
            for _ in range(self.num_layers):
                # Multi-head attention
                attn = nn.MultiHeadDotProductAttention(
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout if training else 0.0
                )
                
                x_norm = nn.LayerNorm()(x)
                attn_out = attn(x_norm, x_norm)
                x = x + attn_out
                
                # MLP block
                x_norm = nn.LayerNorm()(x)
                mlp_out = nn.Sequential([
                    nn.Dense(self.mlp_dim),
                    nn.relu,
                    nn.Dropout(rate=self.dropout, deterministic=not training),
                    nn.Dense(self.embed_dim)
                ])(x_norm)
                x = x + mlp_out
            
            # Global pooling
            x = jnp.mean(x, axis=1)
            
            # Classification head
            x = nn.Dense(10)(x)  # 10 variable star types
            
            return x
    
    # Initialize transformer
    transformer = LightCurveTransformer()
    dummy_times = jnp.linspace(0, 100, 200)[None, :]  # (1, 200)
    dummy_fluxes = jnp.ones((1, 200))
    transformer_params = transformer.init(key, dummy_times, dummy_fluxes)
    
    print(f"  Transformer initialized for light curves")
    
    # 4. Custom layers
    print("\n4. CUSTOM LAYERS:")
    
    class SpectralConvolution(nn.Module):
        """1D convolution with physical constraints."""
        
        features: int
        kernel_size: int
        use_wavelength_weighting: bool = True
        
        @nn.compact
        def __call__(self, x, wavelengths=None):
            # Standard convolution
            conv_out = nn.Conv(
                features=self.features,
                kernel_size=(self.kernel_size,),
                padding='SAME'
            )(x)
            
            # Wavelength-dependent weighting
            if self.use_wavelength_weighting and wavelengths is not None:
                # Weight by inverse wavelength (blue more important)
                weights = 1.0 / wavelengths
                weights = weights / jnp.mean(weights)
                conv_out = conv_out * weights[None, :, None]
            
            return conv_out
    
    print("  Custom spectral convolution layer defined")

flax_fundamentals()
```

### Training with Flax

```python
def flax_training():
    """Complete training pipeline with Flax."""
    
    print("\nFLAX TRAINING PIPELINE")
    print("=" * 50)
    
    # 1. Create training state
    print("\n1. TRAINING STATE MANAGEMENT:")
    
    class PhotometricRedshiftNet(nn.Module):
        """Estimate redshift from photometry."""
        
        @nn.compact
        def __call__(self, x, training: bool = False):
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = nn.Dropout(0.2, deterministic=not training)(x)
            
            x = nn.Dense(64)(x)
            x = nn.relu(x)
            x = nn.Dropout(0.2, deterministic=not training)(x)
            
            x = nn.Dense(32)(x)
            x = nn.relu(x)
            
            # Output: redshift and uncertainty
            mean = nn.Dense(1)(x)
            log_std = nn.Dense(1)(x)
            
            return mean, log_std
    
    # Initialize
    model = PhotometricRedshiftNet()
    key = random.PRNGKey(42)
    
    # Create optimizer
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=100,
        decay_steps=1000,
        end_value=1e-5
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate_schedule)
    )
    
    # Initialize training state
    def create_train_state(rng, model, optimizer, input_shape):
        """Create initial training state."""
        dummy_input = jnp.ones(input_shape)
        params = model.init(rng, dummy_input)
        
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
    
    state = create_train_state(key, model, optimizer, (1, 5))  # 5 photometric bands
    
    print(f"  Training state created with {optimizer}")
    
    # 2. Loss functions
    print("\n2. LOSS FUNCTIONS:")
    
    def gaussian_nll_loss(params, batch, training=True):
        """Gaussian negative log likelihood."""
        inputs, targets = batch
        mean, log_std = state.apply_fn(params, inputs, training=training)
        
        # Negative log likelihood
        std = jnp.exp(log_std)
        nll = 0.5 * jnp.log(2 * jnp.pi) + log_std + \
              0.5 * ((targets - mean) / std) ** 2
        
        return jnp.mean(nll)
    
    def robust_loss(params, batch, training=True):
        """Robust loss using Huber."""
        inputs, targets = batch
        mean, _ = state.apply_fn(params, inputs, training=training)
        
        delta = 0.1  # Huber delta
        residuals = jnp.abs(targets - mean)
        
        loss = jnp.where(
            residuals < delta,
            0.5 * residuals ** 2,
            delta * (residuals - 0.5 * delta)
        )
        
        return jnp.mean(loss)
    
    # 3. Training step
    print("\n3. TRAINING STEP:")
    
    @jit
    def train_step(state, batch, rng):
        """Single training step."""
        dropout_rng = rng
        
        def loss_fn(params):
            return gaussian_nll_loss(params, batch, training=True)
        
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        
        # Update parameters
        state = state.apply_gradients(grads=grads)
        
        return state, loss
    
    @jit
    def eval_step(state, batch):
        """Evaluation step."""
        loss = gaussian_nll_loss(state.params, batch, training=False)
        
        inputs, targets = batch
        mean, log_std = state.apply_fn(state.params, inputs, training=False)
        
        # Compute metrics
        mse = jnp.mean((mean - targets) ** 2)
        mae = jnp.mean(jnp.abs(mean - targets))
        
        # Calibration: fraction within 1-sigma
        std = jnp.exp(log_std)
        within_1sigma = jnp.mean(jnp.abs(mean - targets) < std)
        
        return {
            'loss': loss,
            'mse': mse,
            'mae': mae,
            'calibration': within_1sigma
        }
    
    # 4. Data loading
    print("\n4. DATA LOADING:")
    
    def create_dataset(key, n_samples=10000):
        """Create synthetic photometric redshift dataset."""
        keys = random.split(key, 6)
        
        # Generate redshifts
        z_true = random.uniform(keys[0], (n_samples, 1), minval=0, maxval=3)
        
        # Generate photometry (simplified SED model)
        wavelengths = jnp.array([3500, 4500, 5500, 6500, 7500])  # ugriz
        
        # Rest-frame SED (simplified)
        rest_wavelengths = wavelengths[None, :] / (1 + z_true)
        
        # Blackbody approximation
        T_eff = 5000  # K
        h, c, k = 6.626e-34, 3e8, 1.38e-23
        
        planck = lambda lam, T: (
            2 * h * c**2 / lam**5 / 
            (jnp.exp(h * c / (lam * k * T)) - 1)
        )
        
        # Add noise
        photometry = jnp.log10(planck(rest_wavelengths * 1e-10, T_eff))
        photometry += 0.1 * random.normal(keys[1], photometry.shape)
        
        return photometry, z_true
    
    # Create datasets
    train_data = create_dataset(random.PRNGKey(0), n_samples=5000)
    val_data = create_dataset(random.PRNGKey(1), n_samples=1000)
    
    print(f"  Created training set: {train_data[0].shape}")
    print(f"  Created validation set: {val_data[0].shape}")
    
    # 5. Training loop
    print("\n5. TRAINING LOOP:")
    
    def train_epoch(state, train_data, batch_size, rng):
        """Train for one epoch."""
        X_train, y_train = train_data
        n_samples = len(X_train)
        n_batches = n_samples // batch_size
        
        # Shuffle data
        rng, shuffle_rng = random.split(rng)
        perm = random.permutation(shuffle_rng, n_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        epoch_loss = 0.0
        
        for i in range(n_batches):
            rng, step_rng = random.split(rng)
            
            start = i * batch_size
            end = start + batch_size
            batch = (X_train[start:end], y_train[start:end])
            
            state, loss = train_step(state, batch, step_rng)
            epoch_loss += loss
        
        return state, epoch_loss / n_batches
    
    # Train
    n_epochs = 10
    batch_size = 32
    
    train_losses = []
    val_metrics = []
    
    for epoch in range(n_epochs):
        key, epoch_rng = random.split(key)
        
        # Training
        state, train_loss = train_epoch(state, train_data, batch_size, epoch_rng)
        train_losses.append(train_loss)
        
        # Validation
        val_batch = (val_data[0][:100], val_data[1][:100])  # Sample
        metrics = eval_step(state, val_batch)
        val_metrics.append(metrics)
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Train Loss = {train_loss:.4f}, " +
                  f"Val MAE = {metrics['mae']:.4f}, " +
                  f"Calibration = {metrics['calibration']:.2%}")

flax_training()
```

## Optax: Advanced Optimization

### Optimization Algorithms

```python
def optax_optimizers():
    """Advanced optimization strategies with Optax."""
    
    print("\nOPTAX: ADVANCED OPTIMIZATION")
    print("=" * 50)
    
    # 1. Basic optimizers
    print("\n1. OPTIMIZER COMPARISON:")
    
    def create_loss_landscape():
        """Create a challenging loss landscape."""
        def loss(params):
            x, y = params
            # Rosenbrock function (challenging optimization)
            return (1 - x)**2 + 100 * (y - x**2)**2
        
        return loss
    
    loss_fn = create_loss_landscape()
    
    # Compare optimizers
    optimizers = {
        'SGD': optax.sgd(learning_rate=1e-3),
        'Adam': optax.adam(learning_rate=1e-3),
        'RMSprop': optax.rmsprop(learning_rate=1e-3),
        'AdamW': optax.adamw(learning_rate=1e-3, weight_decay=1e-4),
        'LAMB': optax.lamb(learning_rate=1e-3),
    }
    
    initial_params = jnp.array([-1.0, 1.0])
    
    for name, optimizer in optimizers.items():
        params = initial_params.copy()
        opt_state = optimizer.init(params)
        
        for step in range(100):
            grads = grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        
        final_loss = loss_fn(params)
        print(f"  {name}: Final loss = {final_loss:.6f}, Final params = {params}")
    
    # 2. Learning rate schedules
    print("\n2. LEARNING RATE SCHEDULES:")
    
    # Different schedules
    schedules = {
        'Constant': optax.constant_schedule(1e-3),
        'Exponential': optax.exponential_decay(
            init_value=1e-3,
            transition_steps=100,
            decay_rate=0.9
        ),
        'Cosine': optax.cosine_decay_schedule(
            init_value=1e-3,
            decay_steps=1000
        ),
        'Warmup-Cosine': optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=100,
            decay_steps=1000
        ),
        'Piecewise': optax.piecewise_constant_schedule(
            init_value=1e-3,
            boundaries_and_scales={200: 0.1, 400: 0.1}
        ),
    }
    
    # Visualize schedules
    steps = jnp.arange(1000)
    
    for name, schedule in schedules.items():
        lrs = [schedule(step) for step in steps]
        print(f"  {name}: LR range [{min(lrs):.6f}, {max(lrs):.6f}]")
    
    # 3. Gradient transformations
    print("\n3. GRADIENT TRANSFORMATIONS:")
    
    # Chain multiple transformations
    optimizer_chain = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.scale_by_adam(),            # Adam scaling
        optax.add_decayed_weights(1e-4),  # Weight decay
        optax.scale(-1e-3)                # Learning rate
    )
    
    print("  Chained optimizer created with:")
    print("    - Global norm clipping (1.0)")
    print("    - Adam scaling")
    print("    - Weight decay (1e-4)")
    print("    - Learning rate scaling")
    
    # 4. Advanced techniques
    print("\n4. ADVANCED OPTIMIZATION TECHNIQUES:")
    
    # Lookahead optimizer
    base_optimizer = optax.adam(1e-3)
    lookahead_optimizer = optax.lookahead(base_optimizer, slow_step_size=0.5, period=5)
    
    print("  Lookahead optimizer configured")
    
    # Gradient accumulation
    def gradient_accumulation_optimizer(base_opt, accumulation_steps=4):
        """Accumulate gradients over multiple steps."""
        
        def init_fn(params):
            return {
                'base_state': base_opt.init(params),
                'accumulated_grads': jax.tree_map(jnp.zeros_like, params),
                'step': 0
            }
        
        def update_fn(grads, state, params=None):
            accumulated_grads = jax.tree_map(
                lambda a, g: a + g / accumulation_steps,
                state['accumulated_grads'], grads
            )
            
            step = state['step'] + 1
            
            if step % accumulation_steps == 0:
                # Apply accumulated gradients
                updates, base_state = base_opt.update(
                    accumulated_grads, state['base_state'], params
                )
                # Reset accumulation
                accumulated_grads = jax.tree_map(jnp.zeros_like, accumulated_grads)
            else:
                # Just accumulate
                updates = jax.tree_map(jnp.zeros_like, grads)
                base_state = state['base_state']
            
            new_state = {
                'base_state': base_state,
                'accumulated_grads': accumulated_grads,
                'step': step
            }
            
            return updates, new_state
        
        return optax.GradientTransformation(init_fn, update_fn)
    
    acc_optimizer = gradient_accumulation_optimizer(optax.adam(1e-3))
    print("  Gradient accumulation optimizer created")
    
    # 5. Per-parameter learning rates
    print("\n5. PER-PARAMETER LEARNING RATES:")
    
    def create_model_with_different_lrs():
        """Different learning rates for different layers."""
        
        class Model(nn.Module):
            @nn.compact
            def __call__(self, x):
                # Feature extractor (lower LR)
                x = nn.Dense(128, name='feature_extractor')(x)
                x = nn.relu(x)
                
                # Classifier (higher LR)
                x = nn.Dense(10, name='classifier')(x)
                return x
        
        model = Model()
        dummy_input = jnp.ones((1, 100))
        params = model.init(random.PRNGKey(0), dummy_input)
        
        # Create optimizer with different LRs
        def partition_params(params):
            """Partition parameters by layer."""
            feature_params = {'feature_extractor': params['params']['feature_extractor']}
            classifier_params = {'classifier': params['params']['classifier']}
            return feature_params, classifier_params
        
        # Different optimizers for different parts
        feature_opt = optax.adam(1e-4)  # Lower LR
        classifier_opt = optax.adam(1e-2)  # Higher LR
        
        return model, params, (feature_opt, classifier_opt)
    
    model, params, opts = create_model_with_different_lrs()
    print("  Model with per-layer learning rates created")

optax_optimizers()
```

### Advanced Training Strategies

```python
def advanced_training_strategies():
    """Advanced training techniques with Optax."""
    
    print("\nADVANCED TRAINING STRATEGIES")
    print("=" * 50)
    
    # 1. Mixed precision training
    print("\n1. MIXED PRECISION TRAINING:")
    
    class MixedPrecisionModel(nn.Module):
        """Model with mixed precision computation."""
        
        use_mixed_precision: bool = True
        
        @nn.compact
        def __call__(self, x):
            dtype = jnp.float16 if self.use_mixed_precision else jnp.float32
            
            # Cast to lower precision
            x = x.astype(dtype)
            
            # Compute in lower precision
            x = nn.Dense(128, dtype=dtype)(x)
            x = nn.relu(x)
            x = nn.Dense(64, dtype=dtype)(x)
            x = nn.relu(x)
            
            # Cast back to float32 for loss
            x = nn.Dense(10, dtype=jnp.float32)(x.astype(jnp.float32))
            
            return x
    
    # Loss scaling for mixed precision
    def create_loss_scaled_optimizer(optimizer, loss_scale=1024):
        """Add loss scaling for mixed precision."""
        return optax.chain(
            optax.scale(1 / loss_scale),  # Unscale gradients
            optimizer
        )
    
    base_opt = optax.adam(1e-3)
    scaled_opt = create_loss_scaled_optimizer(base_opt)
    
    print("  Mixed precision optimizer with loss scaling created")
    
    # 2. Differential learning rates
    print("\n2. DIFFERENTIAL LEARNING RATES:")
    
    def layer_wise_lr_decay(base_lr, decay_factor, num_layers):
        """Exponentially decay LR for earlier layers."""
        lrs = []
        for i in range(num_layers):
            lr = base_lr * (decay_factor ** (num_layers - i - 1))
            lrs.append(lr)
        return lrs
    
    layer_lrs = layer_wise_lr_decay(1e-3, 0.5, 4)
    print(f"  Layer-wise LRs: {layer_lrs}")
    
    # 3. Gradient penalty
    print("\n3. GRADIENT PENALTIES:")
    
    def add_gradient_penalty(loss_fn, penalty_weight=0.1):
        """Add gradient penalty for regularization."""
        
        def penalized_loss(params, inputs):
            # Original loss
            loss = loss_fn(params, inputs)
            
            # Gradient penalty
            grads = grad(loss_fn)(params, inputs)
            grad_norm = jnp.sqrt(
                sum(jnp.sum(g**2) for g in jax.tree_leaves(grads))
            )
            
            penalty = penalty_weight * grad_norm
            
            return loss + penalty
        
        return penalized_loss
    
    print("  Gradient penalty wrapper created")
    
    # 4. EMA (Exponential Moving Average)
    print("\n4. EXPONENTIAL MOVING AVERAGE:")
    
    def create_ema():
        """Create EMA of model parameters."""
        
        def init_fn(params):
            return params  # Initialize with current params
        
        def update_fn(params, ema_params, decay=0.999):
            """Update EMA parameters."""
            return jax.tree_map(
                lambda e, p: decay * e + (1 - decay) * p,
                ema_params, params
            )
        
        return init_fn, update_fn
    
    ema_init, ema_update = create_ema()
    
    print("  EMA functions created for model averaging")
    
    # 5. Stochastic Weight Averaging
    print("\n5. STOCHASTIC WEIGHT AVERAGING (SWA):")
    
    class SWAState:
        """State for SWA training."""
        
        def __init__(self, params):
            self.params = params
            self.swa_params = jax.tree_map(jnp.zeros_like, params)
            self.n_averaged = 0
        
        def update(self, new_params, start_averaging_step, current_step):
            """Update SWA parameters."""
            if current_step >= start_averaging_step:
                # Update running average
                self.n_averaged += 1
                self.swa_params = jax.tree_map(
                    lambda swa, p: swa + (p - swa) / self.n_averaged,
                    self.swa_params, new_params
                )
            
            self.params = new_params
            return self
    
    print("  SWA state management created")

advanced_training_strategies()
```

## Orbax: Checkpointing and Experiment Management

### Checkpoint Management

```python
def orbax_checkpointing():
    """Checkpoint management with Orbax."""
    
    print("\nORBAX: CHECKPOINT MANAGEMENT")
    print("=" * 50)
    
    import orbax.checkpoint as ocp
    import tempfile
    import os
    
    # 1. Basic checkpointing
    print("\n1. BASIC CHECKPOINTING:")
    
    # Create temporary directory for checkpoints
    checkpoint_dir = tempfile.mkdtemp()
    
    # Create a simple model and optimizer
    class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = nn.Dense(10)(x)
            return x
    
    model = SimpleModel()
    key = random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 100)))
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    # Create checkpoint manager
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        keep_period=5,
        create=True
    )
    
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
        item_names=('params', 'opt_state', 'metadata')
    )
    
    # Save checkpoint
    step = 0
    metadata = {'epoch': 0, 'loss': 0.5, 'accuracy': 0.85}
    
    checkpoint_manager.save(
        step,
        args=ocp.args.Composite(
            params=ocp.args.StandardSave(params),
            opt_state=ocp.args.StandardSave(opt_state),
            metadata=ocp.args.JsonSave(metadata)
        )
    )
    
    print(f"  Checkpoint saved at step {step}")
    
    # 2. Restoring checkpoints
    print("\n2. RESTORING CHECKPOINTS:")
    
    # Restore latest checkpoint
    restored = checkpoint_manager.restore(
        checkpoint_manager.latest_step(),
        args=ocp.args.Composite(
            params=ocp.args.StandardRestore(params),
            opt_state=ocp.args.StandardRestore(opt_state),
            metadata=ocp.args.JsonRestore()
        )
    )
    
    print(f"  Restored from step {checkpoint_manager.latest_step()}")
    print(f"  Metadata: {restored['metadata']}")
    
    # 3. Async checkpointing
    print("\n3. ASYNCHRONOUS CHECKPOINTING:")
    
    async_checkpoint_manager = ocp.CheckpointManager(
        os.path.join(checkpoint_dir, 'async'),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=2,
            save_interval_steps=10,
            enable_async_checkpointing=True
        )
    )
    
    print("  Async checkpoint manager created")
    
    # 4. Best model tracking
    print("\n4. BEST MODEL TRACKING:")
    
    class BestModelTracker:
        """Track and save best model based on metric."""
        
        def __init__(self, checkpoint_dir, metric='loss', mode='min'):
            self.checkpoint_dir = checkpoint_dir
            self.metric = metric
            self.mode = mode
            self.best_value = float('inf') if mode == 'min' else float('-inf')
            self.checkpointer = ocp.Checkpointer(
                ocp.StandardCheckpointHandler()
            )
        
        def update(self, params, metrics, step):
            """Update best model if improved."""
            current_value = metrics[self.metric]
            
            is_better = (
                (self.mode == 'min' and current_value < self.best_value) or
                (self.mode == 'max' and current_value > self.best_value)
            )
            
            if is_better:
                self.best_value = current_value
                
                # Save best model
                best_path = os.path.join(self.checkpoint_dir, 'best_model')
                self.checkpointer.save(
                    best_path,
                    args=ocp.args.Composite(
                        params=ocp.args.StandardSave(params),
                        metrics=ocp.args.JsonSave(metrics),
                        step=ocp.args.JsonSave({'step': step})
                    )
                )
                
                print(f"    New best model saved: {self.metric}={current_value:.4f}")
                return True
            
            return False
    
    tracker = BestModelTracker(checkpoint_dir, metric='loss', mode='min')
    
    # Simulate training with metric tracking
    for step in range(5):
        fake_loss = 1.0 / (step + 1)  # Decreasing loss
        metrics = {'loss': fake_loss, 'accuracy': 1 - fake_loss}
        tracker.update(params, metrics, step)
    
    # 5. Multi-host checkpointing
    print("\n5. DISTRIBUTED CHECKPOINTING:")
    
    # For multi-host training (conceptual)
    def create_distributed_checkpoint_manager():
        """Create checkpoint manager for distributed training."""
        
        # Would use in multi-host setting
        # checkpoint_manager = ocp.CheckpointManager(
        #     directory,
        #     options=ocp.CheckpointManagerOptions(
        #         save_interval_steps=100,
        #         max_to_keep=3
        #     ),
        #     metadata={'host_count': jax.process_count()}
        # )
        
        print("  Distributed checkpointing configured (requires multi-host setup)")
    
    create_distributed_checkpoint_manager()
    
    # Cleanup
    import shutil
    shutil.rmtree(checkpoint_dir)

orbax_checkpointing()
```

## Complete Training Pipeline

### Production ML Pipeline

```python
def complete_ml_pipeline():
    """Complete ML pipeline for astronomical applications."""
    
    print("\nCOMPLETE ML PIPELINE")
    print("=" * 50)
    
    # 1. Model definition
    class TransientClassifier(nn.Module):
        """Classify astronomical transients from light curves."""
        
        num_classes: int = 5  # SN Ia, SN II, SN Ibc, SLSN, Kilonova
        hidden_dim: int = 256
        num_heads: int = 8
        num_layers: int = 4
        dropout: float = 0.1
        
        @nn.compact
        def __call__(self, times, fluxes, errors, filters, training: bool = False):
            batch_size, seq_len = fluxes.shape
            
            # Embed observations
            obs_features = jnp.stack([
                fluxes,
                errors,
                times,
                filters
            ], axis=-1)  # (batch, seq, 4)
            
            # Initial projection
            x = nn.Dense(self.hidden_dim)(obs_features)
            
            # Positional encoding from times
            pos_encoding = self.param(
                'pos_encoding',
                nn.initializers.normal(stddev=0.02),
                (1, seq_len, self.hidden_dim)
            )
            x = x + pos_encoding
            
            # Transformer layers
            for i in range(self.num_layers):
                # Self-attention
                attn_out = nn.MultiHeadDotProductAttention(
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout if training else 0.0,
                    deterministic=not training
                )(x, x)
                x = nn.LayerNorm()(x + attn_out)
                
                # FFN
                ffn_out = nn.Sequential([
                    nn.Dense(self.hidden_dim * 4),
                    nn.gelu,
                    nn.Dropout(self.dropout, deterministic=not training),
                    nn.Dense(self.hidden_dim)
                ])(x)
                x = nn.LayerNorm()(x + ffn_out)
            
            # Global pooling with attention
            attention_weights = nn.Dense(1)(x)
            attention_weights = nn.softmax(attention_weights, axis=1)
            x = jnp.sum(x * attention_weights, axis=1)
            
            # Classification head
            x = nn.Dense(self.hidden_dim // 2)(x)
            x = nn.relu(x)
            x = nn.Dropout(self.dropout, deterministic=not training)(x)
            logits = nn.Dense(self.num_classes)(x)
            
            return logits
    
    # 2. Data generation
    def generate_transient_data(key, n_samples=1000):
        """Generate synthetic transient light curves."""
        keys = random.split(key, 5)
        
        all_times = []
        all_fluxes = []
        all_errors = []
        all_filters = []
        all_labels = []
        
        for i in range(n_samples):
            # Random transient type
            label = random.choice(keys[0], 5)
            
            # Generate light curve based on type
            n_obs = random.choice(keys[1], 1, minval=20, maxval=100)[0]
            times = jnp.sort(random.uniform(keys[2], (n_obs,), minval=0, maxval=100))
            
            # Different templates for different types
            if label == 0:  # SN Ia
                peak_time = 20.0
                rise_time = 15.0
                decay_time = 30.0
            elif label == 1:  # SN II
                peak_time = 30.0
                rise_time = 20.0
                decay_time = 60.0
            else:
                peak_time = 25.0
                rise_time = 10.0
                decay_time = 40.0
            
            # Generate flux (simplified)
            fluxes = jnp.where(
                times < peak_time,
                jnp.exp(-(times - peak_time)**2 / (2 * rise_time**2)),
                jnp.exp(-(times - peak_time)**2 / (2 * decay_time**2))
            )
            
            # Add noise
            errors = 0.05 + 0.05 * random.uniform(keys[3], (n_obs,))
            fluxes += errors * random.normal(keys[4], (n_obs,))
            
            # Random filters (ugriz)
            filters = random.choice(keys[0], 5, shape=(n_obs,))
            
            # Pad to fixed length
            max_len = 100
            padded_times = jnp.pad(times, (0, max_len - len(times)))
            padded_fluxes = jnp.pad(fluxes, (0, max_len - len(fluxes)))
            padded_errors = jnp.pad(errors, (0, max_len - len(errors)))
            padded_filters = jnp.pad(filters, (0, max_len - len(filters)))
            
            all_times.append(padded_times)
            all_fluxes.append(padded_fluxes)
            all_errors.append(padded_errors)
            all_filters.append(padded_filters)
            all_labels.append(label)
        
        return (
            jnp.stack(all_times),
            jnp.stack(all_fluxes),
            jnp.stack(all_errors),
            jnp.stack(all_filters),
            jnp.array(all_labels)
        )
    
    # Generate data
    key = random.PRNGKey(42)
    train_data = generate_transient_data(key, n_samples=500)
    val_data = generate_transient_data(random.PRNGKey(43), n_samples=100)
    
    print(f"  Generated {len(train_data[0])} training samples")
    
    # 3. Training setup
    model = TransientClassifier()
    
    # Initialize
    dummy_batch = tuple(x[:1] for x in train_data[:-1])
    params = model.init(key, *dummy_batch)
    
    # Optimizer with schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=50,
        decay_steps=500
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    )
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # 4. Training functions
    @jit
    def train_step(state, batch, dropout_key):
        """Training step."""
        times, fluxes, errors, filters, labels = batch
        
        def loss_fn(params):
            logits = state.apply_fn(
                params, times, fluxes, errors, filters,
                training=True, rngs={'dropout': dropout_key}
            )
            
            # Cross-entropy loss
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            loss = jnp.mean(loss)
            
            # L2 regularization (handled by adamw)
            
            return loss, logits
        
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # Update
        state = state.apply_gradients(grads=grads)
        
        # Metrics
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        
        return state, {'loss': loss, 'accuracy': accuracy}
    
    @jit
    def eval_step(state, batch):
        """Evaluation step."""
        times, fluxes, errors, filters, labels = batch
        
        logits = state.apply_fn(
            state.params, times, fluxes, errors, filters,
            training=False
        )
        
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(loss)
        
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == labels)
        
        # Per-class accuracy
        per_class_acc = []
        for c in range(5):
            mask = labels == c
            if jnp.sum(mask) > 0:
                class_acc = jnp.mean(predictions[mask] == c)
                per_class_acc.append(class_acc)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc
        }
    
    # 5. Training loop
    print("\n  Training transient classifier...")
    
    n_epochs = 5
    batch_size = 32
    
    for epoch in range(n_epochs):
        # Training
        key, dropout_key = random.split(key)
        
        n_batches = len(train_data[0]) // batch_size
        epoch_metrics = {'loss': 0.0, 'accuracy': 0.0}
        
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            
            batch = tuple(x[start:end] for x in train_data)
            
            dropout_key, step_key = random.split(dropout_key)
            state, metrics = train_step(state, batch, step_key)
            
            epoch_metrics['loss'] += metrics['loss']
            epoch_metrics['accuracy'] += metrics['accuracy']
        
        epoch_metrics['loss'] /= n_batches
        epoch_metrics['accuracy'] /= n_batches
        
        # Validation
        val_batch = tuple(x[:batch_size] for x in val_data)
        val_metrics = eval_step(state, val_batch)
        
        print(f"  Epoch {epoch}: "
              f"Train Loss={epoch_metrics['loss']:.4f}, "
              f"Train Acc={epoch_metrics['accuracy']:.2%}, "
              f"Val Loss={val_metrics['loss']:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.2%}")
    
    print("\n  Training complete!")
    
    return state

# Run pipeline
final_state = complete_ml_pipeline()
```

## Key Takeaways

✅ **Flax** - Production-ready neural networks with clean module system  
✅ **Optax** - Composable optimization with advanced schedules and techniques  
✅ **Orbax** - Robust checkpointing and experiment management  
✅ **Integration** - Seamless workflow from model definition to production  
✅ **Scalability** - Ready for multi-GPU and large-scale training  
✅ **Best Practices** - Type safety, mixed precision, and monitoring built-in  

## Next Chapter Preview
Specialized Libraries: BlackJAX for MCMC, NetKet for quantum systems, and more domain-specific JAX tools.