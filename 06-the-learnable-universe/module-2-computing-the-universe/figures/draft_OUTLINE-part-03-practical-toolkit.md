# OUTLINE: Part 3 - The JAX Toolkit for Scientific Computing

## Design Principles

**Scope**: Comprehensive JAX toolkit for general scientific computing
**NOT**: N-body migration guide (that's Project 5 - students' work)
**Approach**: Tool-by-tool with small, generic physics examples
**Structure**: Similar to Parts 1 and 2 (no deliverables, understanding checklist)

---

## Title & Metadata

```markdown
---
title: "Module 6 Part 3: The JAX Toolkit for Scientific Computing"
subtitle: "Control Flow, Data Structures, and Essential Patterns | Computing the Universe | ASTR 596"
---

**Prerequisites**: [Part 2: Core Transformations](part-02-core-transformations.md) completed

> *"Programs must be written for people to read, and only incidentally for machines to execute."* — Abelson & Sussman
>
> *"Premature optimization is the root of all evil."* — Donald Knuth
```

---

## Learning Outcomes

By the end of Part 3, you will be able to:

- [ ] **Implement** loops using `lax.scan`, `lax.fori_loop`, and `lax.while_loop`
- [ ] **Apply** `lax.cond` for conditional logic that works with JIT
- [ ] **Manage** state updates through functional patterns (carry state)
- [ ] **Work with** PyTrees for nested data structures
- [ ] **Generate** reproducible random numbers using the PRNGKey system
- [ ] **Update** arrays functionally using `.at[]` syntax
- [ ] **Choose** appropriate data types and precision for your problem
- [ ] **Debug** JAX code using `jax.debug.print` and selective JIT disabling
- [ ] **Design** time-stepping algorithms compatible with JAX transformations
- [ ] **Benchmark** different JAX patterns and select the most efficient

---

## Roadmap

**Priority: 🔴 Essential**

**Parts 1 & 2 gave you foundations and transformations.**

You now understand:
- ✅ Functional programming and pure functions (Part 1)
- ✅ How grad, jit, vmap work (Part 2)
- ✅ Composing transformations (Part 2)

**Part 3 teaches the practical toolkit**:

This is your JAX "toolbox" for writing real scientific computing code. Every tool here solves a specific problem that arises when writing physics simulations, optimization algorithms, or data analysis pipelines.

**Structure**: We'll cover 8 essential topics:

1. **Control Flow: Loops** — `lax.scan`, `lax.fori_loop`, `lax.while_loop`
2. **Control Flow: Conditionals** — `lax.cond`, `lax.switch`
3. **Data Structures** — PyTrees, tree_map, tree_flatten/unflatten
4. **Random Numbers** — PRNGKey system, splitting, jax.random functions
5. **Array Operations** — `.at[]` syntax, functional updates
6. **Data Types & Precision** — dtypes, type promotion, precision tradeoffs
7. **Debugging** — `jax.debug.print`, disabling JIT, NaN checking
8. **Design Patterns** — State management, time-stepping, performance patterns

**Each section follows this pattern**:
- **The Problem**: What coding challenge does this solve?
- **The JAX Solution**: How does JAX handle it?
- **When to Use**: Decision criteria
- **Hands-On Examples**: Small, generic physics demos
- **Common Errors**: What goes wrong and how to fix it

---

## Section 3.1: Control Flow — Loops with lax.scan

**Priority: 🔴 Essential**

### The Problem

You have a time-stepping loop (e.g., integrating an ODE):

```python
# NumPy version - does NOT work with JIT
def integrate_forward_euler(state, dt, n_steps):
    states = [state]  # ❌ Growing Python list
    for i in range(n_steps):
        state = state + dt * derivative(state)
        states.append(state)  # ❌ Side effect
    return np.array(states)
```

**Issues**:
- Growing Python list → can't JIT compile
- Appending in loop → side effect
- Can't use `vmap` over time dimension

### The JAX Solution: lax.scan

```python
def integrate_forward_euler(state_init, dt, n_steps):
    def step_fn(state, i):
        state_new = state + dt * derivative(state)
        return state_new, state_new  # (carry, output)

    final_state, all_states = jax.lax.scan(step_fn, state_init, jnp.arange(n_steps))
    return all_states
```

**Key insight**: `scan(f, init, xs)` applies `f` repeatedly while "carrying" state forward.

**Pattern**: `(carry_out, output) = f(carry_in, x_i)`

### When to Use lax.scan

✅ **Use when**:
- Time-stepping loops (ODE/PDE integration)
- Iterative algorithms (optimization, root-finding)
- You need ALL intermediate states (not just final)
- Loop body is pure function

❌ **Don't use when**:
- Just need final state (use `lax.fori_loop` instead)
- Loop body has complex conditionals (harder to debug)
- Number of iterations is tiny (<10) and speed doesn't matter

### Hands-On Example: Damped Harmonic Oscillator

**Physics**: $\ddot{x} + 2\gamma\dot{x} + \omega_0^2 x = 0$

Convert to first-order system: $\vec{y} = (x, v)$

```python
import jax
import jax.numpy as jnp

def damped_oscillator_derivative(state, gamma, omega0):
    """Compute dy/dt = (v, -2*gamma*v - omega0^2*x)"""
    x, v = state
    dxdt = v
    dvdt = -2 * gamma * v - omega0**2 * x
    return jnp.array([dxdt, dvdt])

def integrate_oscillator(y0, dt, n_steps, gamma, omega0):
    """Forward Euler integration using lax.scan."""

    def step(state, i):
        # Compute derivative at current state
        dydt = damped_oscillator_derivative(state, gamma, omega0)

        # Forward Euler step
        state_new = state + dt * dydt

        # Return (carry, output)
        return state_new, state_new

    # Run scan
    final_state, trajectory = jax.lax.scan(step, y0, jnp.arange(n_steps))

    return trajectory  # Shape: (n_steps, 2)

# Example usage
y0 = jnp.array([1.0, 0.0])  # Start at x=1, v=0
trajectory = integrate_oscillator(y0, dt=0.01, n_steps=1000, gamma=0.1, omega0=2.0)

print(f"Final position: {trajectory[-1, 0]:.4f}")
print(f"Final velocity: {trajectory[-1, 1]:.4f}")
```

**Why this works**:
- `step` is pure (no side effects)
- State is explicitly carried forward
- All intermediate states automatically collected
- Can JIT compile: `integrate_jit = jax.jit(integrate_oscillator)`
- Can vectorize: `integrate_batch = jax.vmap(integrate_oscillator, in_axes=(0, None, None, None, None))`

### lax.fori_loop: When You Only Need Final State

If you DON'T need intermediate states, `lax.fori_loop` is simpler:

```python
def integrate_to_final(y0, dt, n_steps, gamma, omega0):
    """Only return final state, not trajectory."""

    def body_fn(i, state):
        dydt = damped_oscillator_derivative(state, gamma, omega0)
        return state + dt * dydt

    final_state = jax.lax.fori_loop(0, n_steps, body_fn, y0)
    return final_state
```

**Pattern**: `state_final = lax.fori_loop(start, stop, body_fn, state_init)`

**Difference from scan**:
- `fori_loop`: Only returns final carry (more memory efficient)
- `scan`: Returns all outputs (needed for trajectories)

### lax.while_loop: Adaptive Termination

When iterations depend on a condition (not fixed count):

```python
def integrate_until_threshold(y0, dt, threshold, gamma, omega0):
    """Integrate until |x| < threshold."""

    def cond_fn(state_and_count):
        state, count = state_and_count
        x, v = state
        return (jnp.abs(x) > threshold) & (count < 10000)  # Safety limit

    def body_fn(state_and_count):
        state, count = state_and_count
        dydt = damped_oscillator_derivative(state, gamma, omega0)
        state_new = state + dt * dydt
        return (state_new, count + 1)

    final_state, n_steps = jax.lax.while_loop(cond_fn, body_fn, (y0, 0))
    return final_state, n_steps
```

**Warning**: `while_loop` is harder to vectorize (different trajectories may have different lengths).

### Common Errors

**Error 1: Forgetting to return carry**
```python
def step(state, i):
    state_new = state + dt * f(state)
    return state_new  # ❌ Missing output!
# Fix: return (state_new, state_new)
```

**Error 2: Wrong shapes**
```python
# If state is (2,) array, scan returns (n_steps, 2)
# If you want (2, n_steps), need to transpose or use different carry structure
```

**Error 3: Using Python control flow inside scan**
```python
def step(state, i):
    if state[0] > 0:  # ❌ Python if doesn't work with JIT
        ...
# Fix: Use lax.cond (see Section 3.2)
```

### Key Takeaways

- `lax.scan`: Use for loops where you collect all intermediate values (trajectories)
- `lax.fori_loop`: Use when you only need final result (more efficient)
- `lax.while_loop`: Use when iteration count depends on condition
- Always return `(carry, output)` from scan body
- State must be JAX arrays, not Python types

---

## Section 3.2: Control Flow — Conditionals with lax.cond

**Priority: 🔴 Essential**

### The Problem

You have conditional logic that needs to work with JIT:

```python
def apply_boundary_condition(position, boundary_type):
    if boundary_type == "periodic":  # ❌ Python if
        return position % box_size
    elif boundary_type == "reflective":
        return jnp.where(position > box_size, 2*box_size - position, position)
    else:
        return position  # Open boundaries
```

**Issue**: Python `if` statements don't work with JIT (tracer doesn't know which branch to take).

### The JAX Solution: lax.cond

```python
def apply_boundary_periodic(position):
    return position % box_size

def apply_boundary_reflective(position):
    return jnp.where(position > box_size, 2*box_size - position, position)

def apply_boundary_open(position):
    return position

def apply_boundary_condition(position, boundary_code):
    """boundary_code: 0=periodic, 1=reflective, 2=open"""
    return jax.lax.cond(
        boundary_code == 0,
        apply_boundary_periodic,
        lambda pos: jax.lax.cond(
            boundary_code == 1,
            apply_boundary_reflective,
            apply_boundary_open,
            pos
        ),
        position
    )
```

**Pattern**: `lax.cond(pred, true_fn, false_fn, operand)`

**Key constraint**: BOTH branches are traced (but only one executed at runtime).

### When to Use lax.cond

✅ **Use when**:
- Conditional logic inside JIT-compiled functions
- Branches have similar computational cost
- Condition depends on traced array values

❌ **Use jnp.where instead when**:
- Simple element-wise conditionals
- `jnp.where(condition, x, y)` is simpler and often faster

```python
# For element-wise: prefer jnp.where
velocity_clamped = jnp.where(jnp.abs(velocity) > v_max,
                              jnp.sign(velocity) * v_max,
                              velocity)

# For structural: use lax.cond
result = jax.lax.cond(use_complex_model,
                       run_full_physics,
                       run_simplified_physics,
                       state)
```

### Hands-On Example: Adaptive Time-Stepping

**Physics**: Use small timestep if velocity is high, large timestep otherwise.

```python
def compute_timestep_small(state):
    """Small timestep for high velocities."""
    return 0.001

def compute_timestep_large(state):
    """Large timestep for low velocities."""
    return 0.01

def adaptive_step(state):
    """Choose timestep based on velocity magnitude."""
    x, v = state
    v_mag = jnp.sqrt(jnp.sum(v**2))

    # Threshold velocity
    v_threshold = 1.0

    # Choose timestep adaptively
    dt = jax.lax.cond(
        v_mag > v_threshold,
        compute_timestep_small,
        compute_timestep_large,
        state
    )

    return dt
```

### lax.switch: Multiple Branches

For more than 2 branches, use `lax.switch`:

```python
def integrator_euler(state, dt):
    return state + dt * derivative(state)

def integrator_rk2(state, dt):
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    return state + dt * k2

def integrator_rk4(state, dt):
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def integrate_adaptive(state, dt, method_code):
    """method_code: 0=Euler, 1=RK2, 2=RK4"""
    branches = [integrator_euler, integrator_rk2, integrator_rk4]
    return jax.lax.switch(method_code, branches, state, dt)
```

**Pattern**: `lax.switch(index, branches, operand)`

### Common Errors

**Error 1: Branches have side effects**
```python
results = []
def true_fn(x):
    results.append(x)  # ❌ Side effect
    return x
# Both branches are traced, so this behaves unexpectedly
```

**Error 2: Branches return different shapes**
```python
def true_fn(x):
    return jnp.array([1.0, 2.0])  # Shape (2,)

def false_fn(x):
    return jnp.array([1.0])  # Shape (1,) ❌

# Fix: Ensure both branches return same shape/dtype
```

**Error 3: Using Python bool**
```python
use_model_A = True  # Python bool
result = jax.lax.cond(use_model_A, ...)  # ❌ Must be JAX array
# Fix: use_model_A = jnp.array(True) or pass as traced value
```

### Key Takeaways

- Use `lax.cond` for structural conditionals in JIT functions
- Use `jnp.where` for element-wise conditionals (simpler, often faster)
- Use `lax.switch` for multiple branches
- Both/all branches are TRACED (even if only one executes)
- Ensure branches return same shape/dtype

---

## Section 3.3: Data Structures — PyTrees

**Priority: 🟡 Important**

### The Problem

You have complex state (positions, velocities, masses, etc.):

```python
# As separate arrays - gets messy
positions = jnp.array(...)
velocities = jnp.array(...)
masses = jnp.array(...)
times = jnp.array(...)

# How to pass all of these through functions?
# How to apply transformations uniformly?
```

### The JAX Solution: PyTrees

**PyTree** = Any nested structure of containers (dicts, lists, tuples, namedtuples, dataclasses) with JAX arrays at the leaves.

```python
# Option 1: Dictionary
state = {
    'positions': jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    'velocities': jnp.array([[0.1, 0.2], [-0.1, 0.3]]),
    'masses': jnp.array([1.0, 2.0]),
    'time': 0.0
}

# Option 2: Nested structure
state = {
    'particles': {
        'positions': jnp.array(...),
        'velocities': jnp.array(...)
    },
    'system': {
        'time': 0.0,
        'energy': 1.5
    }
}
```

### Working with PyTrees

**tree_map**: Apply function to all leaves

```python
import jax.tree_util as jtu

# Scale all arrays by 2
state_scaled = jtu.tree_map(lambda x: 2*x, state)

# Add two states
state_sum = jtu.tree_map(lambda x, y: x + y, state1, state2)
```

**tree_flatten / tree_unflatten**: Convert to/from flat representation

```python
# Flatten
leaves, treedef = jtu.tree_flatten(state)
print(f"Leaves: {leaves}")  # List of arrays
print(f"Tree structure: {treedef}")  # Structure info

# Unflatten
state_reconstructed = jtu.tree_unflatten(treedef, leaves)
```

### Hands-On Example: Two-Body Problem

```python
from typing import NamedTuple

class TwoBodyState(NamedTuple):
    """State for two-body gravitational problem."""
    r1: jnp.ndarray  # Position of body 1 (3D)
    v1: jnp.ndarray  # Velocity of body 1
    r2: jnp.ndarray  # Position of body 2
    v2: jnp.ndarray  # Velocity of body 2
    m1: float
    m2: float
    time: float

# Create initial state
state = TwoBodyState(
    r1=jnp.array([1.0, 0.0, 0.0]),
    v1=jnp.array([0.0, 0.5, 0.0]),
    r2=jnp.array([-1.0, 0.0, 0.0]),
    v2=jnp.array([0.0, -0.5, 0.0]),
    m1=1.0,
    m2=1.0,
    time=0.0
)

def compute_derivative(state):
    """Compute time derivatives (this is physics - NOT the focus here)."""
    # Relative position
    r = state.r1 - state.r2
    r_norm = jnp.sqrt(jnp.sum(r**2))

    # Gravitational force (G = 1)
    F_mag = state.m1 * state.m2 / r_norm**2
    F_dir = r / r_norm

    # Accelerations
    a1 = -F_mag * F_dir / state.m1
    a2 = F_mag * F_dir / state.m2

    # Return derivatives as PyTree with same structure
    return TwoBodyState(
        r1=state.v1,
        v1=a1,
        r2=state.v2,
        v2=a2,
        m1=0.0,  # Constants don't change
        m2=0.0,
        time=1.0  # dt/dt = 1
    )

def integrate_step(state, dt):
    """Forward Euler step using PyTree operations."""
    deriv = compute_derivative(state)

    # Add: state + dt * deriv
    # tree_map applies this to ALL fields
    state_new = jtu.tree_map(lambda s, d: s + dt * d, state, deriv)

    return state_new

# Works with vmap, jit, grad automatically!
integrate_jit = jax.jit(integrate_step)
```

### When to Use PyTrees

✅ **Use when**:
- Complex state with multiple components
- Want clean, readable code
- Need to apply operations uniformly
- Building differentiable simulations

❌ **Don't use when**:
- Simple state (single array) - unnecessary overhead
- Need maximum performance (flat arrays can be faster)
- Debugging complex shape issues (harder to track)

### Common Errors

**Error 1: Mixing PyTree structures**
```python
state1 = {'pos': x, 'vel': v}
state2 = {'position': x, 'velocity': v}  # Different keys!

# This fails:
jtu.tree_map(lambda a, b: a + b, state1, state2)  # ❌ Structure mismatch
```

**Error 2: Including non-arrays in tree**
```python
state = {
    'data': jnp.array([1, 2, 3]),
    'metadata': "some string"  # ❌ Not a leaf type
}
# Fix: Register custom types or keep metadata separate
```

**Error 3: Expecting tree_map to preserve non-leaf structure**
```python
state = {'a': 1, 'b': 2}  # Python ints, not arrays
result = jtu.tree_map(lambda x: x * 2, state)
# This works, but you usually want JAX arrays as leaves
```

### Key Takeaways

- PyTrees = nested containers with arrays at leaves
- `tree_map` applies function to all leaves uniformly
- Great for organizing complex simulation state
- Works seamlessly with jit, vmap, grad
- Use NamedTuple or dataclass for type hints
- Ensure consistent structure when combining trees

---

## Section 3.4: Random Numbers — The PRNGKey System

**Priority: 🔴 Essential**

### The Problem

NumPy's random numbers are STATEFUL (breaks JAX functional paradigm):

```python
import numpy as np

np.random.seed(42)  # Global state ❌
x = np.random.normal(0, 1, size=10)
y = np.random.normal(0, 1, size=10)  # Different from x

# Problems:
# 1. Can't JIT (hidden state)
# 2. Not reproducible across transformations
# 3. Parallel execution gives wrong results
```

### The JAX Solution: Explicit PRNGKey

```python
import jax.random as jrd

# Create initial key
key = jrd.PRNGKey(42)

# Split key for each random operation
key, subkey = jrd.split(key)
x = jrd.normal(subkey, shape=(10,))

key, subkey = jrd.split(key)
y = jrd.normal(subkey, shape=(10,))

# x and y are different (different subkeys)
# But both reproducible from original key!
```

**Key insight**: Random state is EXPLICIT and passed as function argument.

### How PRNGKey Works

**Counter-based PRNG** (Threefry algorithm):
- Key is just a pair of uint32 values
- Splitting is deterministic and cheap
- No hidden global state

```python
key = jrd.PRNGKey(0)
print(key)  # DeviceArray([0, 0], dtype=uint32)

key1, key2 = jrd.split(key)
print(key1)  # Different from key2
print(key2)  # But both deterministic from original key

# Can split into many keys at once
keys = jrd.split(key, num=10)  # Array of 10 independent keys
```

### Hands-On Example: Monte Carlo Integration

**Physics**: Estimate $\pi$ by random sampling in unit square.

```python
def estimate_pi(key, n_samples):
    """Estimate pi using Monte Carlo."""
    # Split key for x and y coordinates
    key, subkey_x, subkey_y = jrd.split(key, 3)

    # Random points in [0, 1] × [0, 1]
    x = jrd.uniform(subkey_x, shape=(n_samples,))
    y = jrd.uniform(subkey_y, shape=(n_samples,))

    # Check if inside unit circle
    inside = (x**2 + y**2) <= 1.0

    # Estimate: (Area of circle) / (Area of square) = π/4
    pi_estimate = 4 * jnp.mean(inside)

    return pi_estimate

# Usage
key = jrd.PRNGKey(42)
pi_est = estimate_pi(key, n_samples=100000)
print(f"π ≈ {pi_est:.6f}")  # Should be close to 3.14159...

# Reproducible: same key gives same result
pi_est2 = estimate_pi(jrd.PRNGKey(42), n_samples=100000)
assert pi_est == pi_est2
```

### Common Random Functions

```python
key = jrd.PRNGKey(0)

# Uniform [0, 1)
key, subkey = jrd.split(key)
uniform_samples = jrd.uniform(subkey, shape=(100,))

# Normal (Gaussian)
key, subkey = jrd.split(key)
normal_samples = jrd.normal(subkey, shape=(100,))

# Uniform in range [min, max)
key, subkey = jrd.split(key)
uniform_range = jrd.uniform(subkey, shape=(100,), minval=-1.0, maxval=1.0)

# Integer sampling
key, subkey = jrd.split(key)
random_ints = jrd.randint(subkey, shape=(100,), minval=0, maxval=10)

# Choice from array
key, subkey = jrd.split(key)
indices = jrd.choice(subkey, 100, shape=(10,), replace=False)  # 10 unique samples
```

### Random Numbers in Loops

**Pattern**: Carry key through scan

```python
def random_walk_1d(key, n_steps, step_size):
    """1D random walk using lax.scan."""

    def step(carry, i):
        position, key = carry

        # Split key for this step
        key, subkey = jrd.split(key)

        # Random step: -1 or +1
        direction = jrd.choice(subkey, jnp.array([-1.0, 1.0]))
        position_new = position + step_size * direction

        return (position_new, key), position_new

    # Initial state: (position, key)
    init_carry = (0.0, key)

    # Run walk
    (final_pos, _), trajectory = jax.lax.scan(step, init_carry, jnp.arange(n_steps))

    return trajectory

# Usage
key = jrd.PRNGKey(123)
walk = random_walk_1d(key, n_steps=1000, step_size=0.1)
print(f"Final position: {walk[-1]:.3f}")
```

### Random Numbers with vmap

```python
def single_walk(key):
    """One random walk."""
    return random_walk_1d(key, n_steps=100, step_size=1.0)

# Generate 1000 walks in parallel
key = jrd.PRNGKey(0)
keys = jrd.split(key, num=1000)  # 1000 independent keys

# vmap over keys
many_walks = jax.vmap(single_walk)(keys)
print(f"Shape: {many_walks.shape}")  # (1000, 100)
```

### Common Errors

**Error 1: Reusing keys**
```python
key = jrd.PRNGKey(0)
x = jrd.normal(key, shape=(10,))
y = jrd.normal(key, shape=(10,))  # ❌ Reused key - x and y are IDENTICAL!

# Fix: Split key
key, subkey1 = jrd.split(key)
x = jrd.normal(subkey1, shape=(10,))
key, subkey2 = jrd.split(key)
y = jrd.normal(subkey2, shape=(10,))  # Different from x
```

**Error 2: Forgetting to thread key through loop**
```python
def broken_random_loop(key, n):
    def step(carry, i):
        x = jrd.normal(key, shape=())  # ❌ Uses SAME key every iteration!
        return carry, x
    _, samples = jax.lax.scan(step, None, jnp.arange(n))
    return samples  # All samples are IDENTICAL!

# Fix: Carry key through loop (see random_walk_1d example above)
```

**Error 3: Using NumPy random in JAX code**
```python
@jax.jit
def broken_function(x):
    noise = np.random.normal(0, 1)  # ❌ NumPy random doesn't work with JIT!
    return x + noise
```

### Key Takeaways

- JAX random numbers use explicit PRNGKey (no global state)
- Always split keys before use: `key, subkey = jrd.split(key)`
- Thread keys through loops as part of carry state
- Generate many keys at once: `jrd.split(key, num=N)`
- Never reuse the same key for multiple random calls
- Works seamlessly with jit, vmap, grad

---

## Section 3.5: Array Operations — Functional Updates with .at[]

**Priority: 🔴 Essential**

### The Problem

NumPy in-place updates don't work with JAX (arrays are immutable):

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
x[2] = 99  # ✓ Works in NumPy

import jax.numpy as jnp

x = jnp.array([1, 2, 3, 4, 5])
x[2] = 99  # ❌ Error: JAX arrays are immutable!
```

### The JAX Solution: .at[] Syntax

```python
x = jnp.array([1, 2, 3, 4, 5])

# Functional update: returns NEW array
x_new = x.at[2].set(99)

print(x)      # [1, 2, 3, 4, 5]  (unchanged)
print(x_new)  # [1, 2, 99, 4, 5]  (new array)
```

**Key insight**: `.at[]` returns a NEW array; original is unchanged (functional programming).

### .at[] Operations

```python
x = jnp.array([1, 2, 3, 4, 5])

# Set value
x.at[2].set(99)              # [1, 2, 99, 4, 5]

# Add to existing value
x.at[2].add(10)              # [1, 2, 13, 4, 5]

# Multiply
x.at[2].mul(2)               # [1, 2, 6, 4, 5]

# Min/max
x.at[2].min(1)               # [1, 2, 1, 4, 5]
x.at[2].max(10)              # [1, 2, 10, 4, 5]

# Multiple indices
x.at[jnp.array([1, 3])].set(0)  # [1, 0, 3, 0, 5]

# Slices
x.at[1:4].set(0)             # [1, 0, 0, 0, 5]

# Boolean mask
mask = x > 2
x.at[mask].set(0)            # [0, 0, 0, 4, 5]
```

### Multidimensional Arrays

```python
A = jnp.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

# Update single element
A.at[1, 2].set(99)           # [1, 2, 3], [4, 5, 99], [7, 8, 9]

# Update row
A.at[1, :].set(0)            # [1, 2, 3], [0, 0, 0], [7, 8, 9]

# Update column
A.at[:, 1].set(0)            # [1, 0, 3], [4, 0, 6], [7, 0, 9]

# Update submatrix
A.at[0:2, 1:3].add(10)       # [1, 12, 13], [4, 15, 16], [7, 8, 9]
```

### Hands-On Example: Particle Collisions

**Physics**: Detect particle-wall collisions and reverse velocities.

```python
def apply_wall_collisions(positions, velocities, box_size):
    """Reverse velocity when particle hits wall."""

    # Find particles outside box
    outside_left = positions < 0
    outside_right = positions > box_size

    # Reverse velocities for collisions
    velocities_new = velocities
    velocities_new = velocities_new.at[outside_left].mul(-1)
    velocities_new = velocities_new.at[outside_right].mul(-1)

    # Clamp positions to box
    positions_new = jnp.clip(positions, 0, box_size)

    return positions_new, velocities_new

# Example
positions = jnp.array([-0.5, 0.3, 1.2])  # One particle outside left, one right
velocities = jnp.array([-0.1, 0.2, 0.3])
box_size = 1.0

pos_new, vel_new = apply_wall_collisions(positions, velocities, box_size)
print(f"Velocities: {vel_new}")  # [0.1, 0.2, -0.3]  (reversed for collisions)
```

### When to Use .at[]

✅ **Use when**:
- Need to update specific array elements
- Conditionally modify array values
- Building arrays incrementally (e.g., histograms)

❌ **Use other operations when**:
- Updating ALL elements → use `jnp.where` or element-wise ops
- Creating new array from scratch → use `jnp.array`, `jnp.zeros`, etc.

```python
# For element-wise conditionals, jnp.where is cleaner
velocities_clamped = jnp.where(velocities > v_max, v_max, velocities)

# vs .at[] version (more verbose)
mask = velocities > v_max
velocities_clamped = velocities.at[mask].set(v_max)
```

### Common Errors

**Error 1: Forgetting .at[] returns NEW array**
```python
x = jnp.array([1, 2, 3])
x.at[1].set(99)  # ❌ Doesn't modify x!
print(x)  # Still [1, 2, 3]

# Fix: Assign result
x = x.at[1].set(99)  # or x_new = x.at[1].set(99)
```

**Error 2: Repeated updates**
```python
x = jnp.array([1, 2, 3, 4, 5])

# This doesn't work as expected
x = x.at[2].set(10)
x = x.at[2].add(5)   # Result: [1, 2, 15, 4, 5] ✓

# But this is inefficient (creates intermediate arrays)
# Better: Chain operations if possible, or just compute final value
x = x.at[2].set(15)
```

**Error 3: Out-of-bounds updates**
```python
x = jnp.array([1, 2, 3])
x_new = x.at[10].set(99)  # No error! Just ignores out-of-bounds

# This is BY DESIGN for JIT (can't raise dynamic errors)
# Check bounds manually if needed
```

### Key Takeaways

- JAX arrays are immutable; use `.at[]` for functional updates
- `.at[idx].set(value)` returns NEW array (original unchanged)
- Operations: `.set()`, `.add()`, `.mul()`, `.min()`, `.max()`
- Works with indices, slices, boolean masks
- Out-of-bounds updates are silently ignored (for JIT compatibility)

---

## Section 3.6: Data Types & Precision

**Priority: 🟡 Important**

### The Problem

Type mismatches and precision issues:

```python
x_float32 = jnp.array([1.0], dtype=jnp.float32)
y_float64 = jnp.array([2.0], dtype=jnp.float64)

z = x_float32 + y_float64  # What dtype is z?
```

### JAX Type Promotion Rules

JAX follows NumPy's type promotion:

```python
jnp.float32 + jnp.float32 → jnp.float32
jnp.float32 + jnp.float64 → jnp.float64
jnp.int32 + jnp.float32   → jnp.float32
```

**But**: JAX defaults to `float32` (unlike NumPy's `float64`).

### Controlling Precision

```python
# Explicit dtype
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

# Global precision setting
from jax import config
config.update("jax_enable_x64", True)  # Enable float64

# Now defaults to float64
x = jnp.array([1.0])  # dtype=float64
```

### When to Use float32 vs float64

**float32 (default)**:
✅ Faster on GPUs
✅ Uses half the memory
✅ Sufficient for most ML and graphics

❌ Accumulated error in long simulations
❌ Issues with ill-conditioned problems

**float64**:
✅ Better numerical stability
✅ Accurate for sensitive physics calculations
✅ Match scientific computing conventions

❌ Slower (especially on GPUs)
❌ Uses more memory

### Hands-On Example: Precision Matters

```python
def test_precision():
    """Show where float32 breaks down."""

    # Sum many small numbers
    n = 10_000_000

    # float32
    x32 = jnp.ones(n, dtype=jnp.float32) * 1e-8
    sum32 = jnp.sum(x32)
    print(f"float32: {sum32:.10f} (should be 100)")

    # float64
    jax.config.update("jax_enable_x64", True)
    x64 = jnp.ones(n, dtype=jnp.float64) * 1e-8
    sum64 = jnp.sum(x64)
    print(f"float64: {sum64:.10f}")

test_precision()
# float32: 99.9999694824 (significant error)
# float64: 100.0000000000 (accurate)
```

### Common dtypes

```python
# Floating point
jnp.float16  # Half precision (rarely used)
jnp.float32  # Single precision (default)
jnp.float64  # Double precision

# Integer
jnp.int8, jnp.int16, jnp.int32, jnp.int64
jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64

# Boolean
jnp.bool_

# Complex
jnp.complex64, jnp.complex128
```

### Type Checking

```python
x = jnp.array([1.0, 2.0])

print(x.dtype)           # float32 (by default)
print(x.dtype == jnp.float32)  # True

# Convert
x64 = x.astype(jnp.float64)
```

### Common Errors

**Error 1: Forgetting JAX defaults to float32**
```python
# NumPy
x_np = np.array([1.0])  # float64

# JAX
x_jax = jnp.array([1.0])  # float32 (different!)

# Fix: Enable x64 or specify dtype explicitly
```

**Error 2: Type mismatches in function arguments**
```python
@jax.jit
def compute(x):  # Traced with float32
    return x + 1.0

x32 = jnp.array([1.0], dtype=jnp.float32)
x64 = jnp.array([1.0], dtype=jnp.float64)

compute(x32)  # ✓ Works
compute(x64)  # ❌ Recompiles (different dtype)
```

### Key Takeaways

- JAX defaults to float32 (unlike NumPy's float64)
- Enable float64: `jax.config.update("jax_enable_x64", True)`
- float32 faster but less accurate; float64 stabler but slower
- Be explicit about dtypes when precision matters
- Type promotion follows NumPy rules

---

## Section 3.7: Debugging JAX Code

**Priority: 🔴 Essential**

### The Problem

Standard Python debugging doesn't work with JIT:

```python
@jax.jit
def buggy_function(x):
    print(f"x = {x}")  # ❌ Prints tracer, not value
    breakpoint()        # ❌ Doesn't work inside JIT
    return x + 1
```

### JAX Debugging Tools

#### 1. jax.debug.print

```python
@jax.jit
def compute(x):
    jax.debug.print("x = {}", x)  # ✓ Prints actual value
    y = x + 1
    jax.debug.print("y = {}", y)
    return y

compute(jnp.array([1.0, 2.0]))
# x = [1. 2.]
# y = [2. 3.]
```

#### 2. Disable JIT Temporarily

```python
# Option 1: Comment out @jax.jit
# @jax.jit
def compute(x):
    print(f"x = {x}")  # Now works (no JIT)
    return x + 1

# Option 2: with jax.disable_jit()
with jax.disable_jit():
    result = compute(x)  # JIT disabled in this block
```

#### 3. Check for NaNs

```python
# Option 1: Manual checks
jax.debug.print("Has NaN: {}", jnp.any(jnp.isnan(x)))

# Option 2: Enable NaN checking (slow but helpful)
from jax import config
config.update("jax_debug_nans", True)

# Now raises error immediately when NaN is produced
result = jnp.sqrt(jnp.array([-1.0]))  # Error!
```

#### 4. Inspect Shapes and Dtypes

```python
@jax.jit
def compute(x):
    jax.debug.print("x.shape = {s}, dtype = {d}", s=x.shape, d=x.dtype)
    return x + 1
```

### Common Debugging Patterns

**Pattern 1: Print intermediate values**
```python
def physics_step(state):
    # Compute forces
    forces = compute_forces(state)
    jax.debug.print("Max force: {}", jnp.max(jnp.abs(forces)))

    # Update velocities
    velocities = state.velocities + dt * forces
    jax.debug.print("Max velocity: {}", jnp.max(jnp.abs(velocities)))

    return state._replace(velocities=velocities)
```

**Pattern 2: Check conservation laws**
```python
def integrate_step(state):
    energy_before = compute_energy(state)

    # Physics update
    state_new = update(state)

    energy_after = compute_energy(state_new)
    energy_error = jnp.abs(energy_after - energy_before) / energy_before
    jax.debug.print("Energy error: {}", energy_error)

    return state_new
```

**Pattern 3: Assert shapes**
```python
def compute(x, y):
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    # Note: This only checks at trace time, not runtime!
    return x + y
```

### Common Errors and How to Debug

**Error 1: ConcretizationError**
```
ConcretizationError: Abstract tracer value encountered where concrete value is expected
```

**Cause**: Using traced value in Python control flow

**Debug**:
```python
# Before (broken)
def f(x):
    if x > 0:  # ❌ Can't compare tracer to 0
        return x
    return -x

# After (fixed)
def f(x):
    return jax.lax.cond(x > 0, lambda x: x, lambda x: -x, x)
```

**Error 2: Shape mismatch**
```python
# Add debug prints
def compute(x, y):
    jax.debug.print("x.shape = {}, y.shape = {}", x.shape, y.shape)
    return x + y  # Now you can see what's wrong
```

**Error 3: NaN/Inf appearing**
```python
# Enable NaN checking
jax.config.update("jax_debug_nans", True)

# Add intermediate checks
def update(state):
    force = compute_force(state)
    jax.debug.print("force has NaN: {}", jnp.any(jnp.isnan(force)))

    velocity = state.velocity + dt * force
    jax.debug.print("velocity has NaN: {}", jnp.any(jnp.isnan(velocity)))

    return velocity
```

### Performance Debugging

**Pattern: Compare implementations**
```python
import time

# Version 1: NumPy
start = time.time()
result_np = numpy_version(data)
time_np = time.time() - start

# Version 2: JAX
start = time.time()
result_jax = jax_version(data)
jax_version(data)  # Run twice (first time includes compilation)
time_jax = time.time() - start

print(f"NumPy: {time_np:.3f}s")
print(f"JAX:   {time_jax:.3f}s")
print(f"Speedup: {time_np/time_jax:.1f}x")
```

### Key Takeaways

- Use `jax.debug.print()` inside JIT functions (not `print()`)
- Disable JIT temporarily for interactive debugging
- Enable NaN checking: `jax.config.update("jax_debug_nans", True)`
- Check shapes, dtypes at intermediate steps
- Use assertions at trace time to catch errors early
- Compare NumPy vs JAX for performance validation

---

## Section 3.8: Design Patterns for Scientific Computing

**Priority: 🟡 Important**

### Pattern 1: Functional State Updates

**Principle**: Never mutate state; always return new state.

```python
# BAD: Mutating state
class Simulation:
    def __init__(self):
        self.positions = jnp.array([1.0, 2.0])
        self.velocities = jnp.array([0.1, 0.2])

    def step(self):
        self.velocities += dt * self.forces()  # ❌ Mutation
        self.positions += dt * self.velocities

# GOOD: Functional state
from typing import NamedTuple

class SimState(NamedTuple):
    positions: jnp.ndarray
    velocities: jnp.ndarray

def step(state: SimState) -> SimState:
    forces = compute_forces(state.positions)
    velocities_new = state.velocities + dt * forces
    positions_new = state.positions + dt * velocities_new
    return SimState(positions_new, velocities_new)
```

### Pattern 2: Separating Physics from Numerics

**Principle**: Separate "what to compute" (physics) from "how to compute" (integration).

```python
# Physics function (takes state, returns derivative)
def compute_derivative(state):
    """Physical laws: dstate/dt = f(state)"""
    positions, velocities = state.positions, state.velocities
    forces = compute_forces(positions)
    masses = state.masses

    # Return time derivatives
    return SimState(
        positions=velocities,  # dx/dt = v
        velocities=forces / masses  # dv/dt = F/m
    )

# Integrator (how to step forward in time)
def forward_euler_step(state, derivative_fn, dt):
    """Generic forward Euler: state(t+dt) = state(t) + dt * f(state)"""
    deriv = derivative_fn(state)
    return jtu.tree_map(lambda s, d: s + dt * d, state, deriv)

def rk4_step(state, derivative_fn, dt):
    """Generic RK4 integrator."""
    k1 = derivative_fn(state)
    k2 = derivative_fn(jtu.tree_map(lambda s, d: s + 0.5*dt*d, state, k1))
    k3 = derivative_fn(jtu.tree_map(lambda s, d: s + 0.5*dt*d, state, k2))
    k4 = derivative_fn(jtu.tree_map(lambda s, d: s + dt*d, state, k3))

    deriv_avg = jtu.tree_map(lambda d1, d2, d3, d4: (d1 + 2*d2 + 2*d3 + d4)/6,
                              k1, k2, k3, k4)
    return jtu.tree_map(lambda s, d: s + dt * d, state, deriv_avg)

# Now can swap integrators easily!
state_new = forward_euler_step(state, compute_derivative, dt)
# or
state_new = rk4_step(state, compute_derivative, dt)
```

### Pattern 3: Vectorization Over Ensembles

**Principle**: Use `vmap` to run many simulations in parallel.

```python
# Single simulation
def run_single_simulation(initial_state, key):
    """Run one simulation with specific initial conditions."""
    # ... integration loop ...
    return final_state

# Many simulations (Monte Carlo, sensitivity analysis, etc.)
def run_ensemble(initial_states, keys):
    """
    initial_states: (n_simulations, ...) shaped state
    keys: (n_simulations,) PRNGKeys
    """
    # vmap over first dimension
    ensemble_fn = jax.vmap(run_single_simulation, in_axes=(0, 0))
    final_states = ensemble_fn(initial_states, keys)
    return final_states

# Usage
n_ensemble = 1000
keys = jrd.split(jrd.PRNGKey(0), num=n_ensemble)

# Create different initial conditions
initial_positions = jrd.uniform(keys[0], shape=(n_ensemble, n_particles, 3))
initial_velocities = jrd.normal(keys[1], shape=(n_ensemble, n_particles, 3))
initial_states = SimState(initial_positions, initial_velocities, ...)

# Run all at once
final_states = run_ensemble(initial_states, keys)
```

### Pattern 4: Checkpointing Long Simulations

**Principle**: Save state periodically to avoid recomputing from scratch.

```python
def simulate_with_checkpoints(state_init, n_steps, checkpoint_every=100):
    """Integrate with periodic checkpoints."""
    checkpoints = []
    state = state_init

    n_checkpoints = n_steps // checkpoint_every

    def run_segment(state, i):
        # Integrate for checkpoint_every steps
        final_state, trajectory = integrate_segment(state, checkpoint_every)
        return final_state, (final_state, trajectory)

    # Run segments with scan
    final_state, (checkpoint_states, trajectories) = jax.lax.scan(
        run_segment, state_init, jnp.arange(n_checkpoints)
    )

    return final_state, checkpoint_states, trajectories

def integrate_segment(state, n_steps):
    """Integrate for n_steps, return final state and trajectory."""
    def step(s, i):
        s_new = integrate_step(s)
        return s_new, s_new

    final, traj = jax.lax.scan(step, state, jnp.arange(n_steps))
    return final, traj
```

### Pattern 5: Benchmarking

**Principle**: Always measure before optimizing.

```python
import time

def benchmark_function(fn, *args, n_runs=10):
    """Benchmark JAX function with multiple runs."""
    # Compile
    fn_jit = jax.jit(fn)

    # Warmup (triggers compilation)
    _ = fn_jit(*args)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = fn_jit(*args).block_until_ready()  # Wait for GPU
        times.append(time.time() - start)

    mean_time = jnp.mean(jnp.array(times))
    std_time = jnp.std(jnp.array(times))

    return mean_time, std_time

# Usage
time_mean, time_std = benchmark_function(my_function, arg1, arg2)
print(f"Time: {time_mean*1000:.2f} ± {time_std*1000:.2f} ms")
```

### Pattern 6: Choosing the Right Loop

**Decision tree**:

```
Need intermediate values?
├─ Yes → Use lax.scan
│   └─ Example: Time-stepping with trajectory output
│
└─ No → Need adaptive termination?
    ├─ Yes → Use lax.while_loop
    │   └─ Example: Iterative solver (continue until convergence)
    │
    └─ No → Use lax.fori_loop
        └─ Example: Fixed iterations, only need final result
```

**Concrete examples**:

```python
# 1. lax.scan: Need trajectory
def compute_trajectory(state, n_steps):
    def step(s, i):
        s_new = update(s)
        return s_new, s_new  # Save intermediate
    final, trajectory = jax.lax.scan(step, state, jnp.arange(n_steps))
    return trajectory

# 2. lax.fori_loop: Only need final
def evolve_to_time(state, n_steps):
    def step(i, s):
        return update(s)  # Don't save intermediate
    final = jax.lax.fori_loop(0, n_steps, step, state)
    return final

# 3. lax.while_loop: Adaptive
def iterate_until_converged(state, tol=1e-6):
    def cond_fn(s):
        residual = compute_residual(s)
        return jnp.linalg.norm(residual) > tol

    def body_fn(s):
        return update(s)

    converged_state = jax.lax.while_loop(cond_fn, body_fn, state)
    return converged_state
```

### Key Takeaways

- Always use functional state updates (no mutation)
- Separate physics (derivatives) from numerics (integrators)
- Use `vmap` for ensemble simulations
- Checkpoint long simulations
- Benchmark before optimizing
- Choose the right loop primitive for your use case

---

## Understanding Checklist

Before applying these tools to your projects (like Project 5: N-body migration), ensure you can:

- [ ] **Implement** time-stepping loops using `lax.scan` (with intermediate state collection)
- [ ] **Choose** between `lax.scan`, `lax.fori_loop`, and `lax.while_loop` based on requirements
- [ ] **Apply** `lax.cond` for conditional logic in JIT functions
- [ ] **Organize** complex state using PyTrees (dicts/NamedTuples)
- [ ] **Use** `tree_map` to apply operations uniformly across state
- [ ] **Generate** reproducible random numbers with PRNGKey splitting
- [ ] **Thread** random keys through loops correctly
- [ ] **Update** arrays functionally using `.at[]` syntax
- [ ] **Explain** when to use float32 vs float64
- [ ] **Debug** JAX code using `jax.debug.print` and JIT disabling
- [ ] **Design** functional state updates (no mutation)
- [ ] **Separate** physics functions from integration schemes
- [ ] **Vectorize** simulations with `vmap` for ensemble runs
- [ ] **Benchmark** code and identify performance bottlenecks

If you answered "yes" to all → **Ready to migrate your N-body code to JAX! (Project 5)**

---

## Resources

**Official JAX Documentation**:
- [Control Flow Primitives](https://jax.readthedocs.io/en/latest/jax.lax.html)
- [Working with PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html)
- [Random Numbers](https://jax.readthedocs.io/en/latest/jax.random.html)
- [Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)

**Advanced Topics**:
- [JAX Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [ODE Integration in JAX (Diffrax)](https://github.com/patrick-kidger/diffrax)
- [Neural ODEs with Equinox](https://github.com/patrick-kidger/equinox)

---

## Connection to Projects

**This module prepares you for Project 5** (N-body migration to JAX):

- ✅ `lax.scan` → Time-stepping your N-body integrator
- ✅ `lax.cond` → Adaptive timesteps, boundary conditions
- ✅ PyTrees → Organizing particle state (positions, velocities, masses)
- ✅ PRNGKey → Generating initial conditions
- ✅ `.at[]` → Updating particle properties
- ✅ `vmap` → Batching force calculations, running ensembles

**You now have ALL the tools to write high-performance, differentiable physics simulations in JAX.**

---

**You're ready to migrate your N-body code and unlock the full power of autodiff + JIT + GPU acceleration!**
