---
title: "Part 1: Conceptual Foundations - Why JAX Requires Functional Programming"
subtitle: "Pure Functions, Immutability, and Computational Graphs | Computing the Universe | ASTR 596"
---

**Prerequisites**: [Overview](overview-module-6-jax-revolution.md) completed

---

(learning-outcomes)=
## Learning Outcomes

By the end of Part 1, you will be able to:

- [ ] **Contrast** object-oriented and functional programming paradigms
- [ ] **Identify** pure vs impure functions in Python code
- [ ] **Explain** why mutation breaks JAX transformations
- [ ] **Recognize** data-dependent control flow that prevents JIT compilation
- [ ] **Describe** computational graphs conceptually (not just "magic")
- [ ] **Articulate** why JAX's constraints enable its transformations

---

(roadmap)=
## Roadmap: The Conceptual Foundation

**Priority: 🔴 Essential**

**Core Question**: *Why does JAX require such strange constraints?* (No mutation, explicit control flow, pure functions)

**The Answer**: These constraints aren't arbitrary — they're what make transformations possible.

This part builds your mental model **before** you write JAX code. Understanding WHY functional programming matters enables you to:

- Debug when things break (and they will)
- Recognize patterns that JAX can't handle
- Write efficient, composable JAX code naturally

**Structure**:

1. **OOP vs Functional** — Two ways to organize code (you know OOP from this semester)
2. **Pure Functions** — The foundation everything builds on
3. **Control Flow** — Why Python `if`/`for` break, what JAX provides instead
4. **Computational Graphs** — How JAX "sees" your code
5. **Transformations** — What JAX actually does
6. **Synthesis** — Why constraints = capabilities

---

(oop-vs-functional)=
## 1.1: Programming Paradigms — OOP vs Functional

**Priority: 🔴 Essential**

### You Already Know Object-Oriented Programming

Earlier this semester, you built classes like `Star` in your projects:

```python
class Star:
    """Object-oriented design (what you've been learning)."""

    def __init__(self, mass):
        self.mass = mass          # State: data stored in object
        self.age = 0.0            # State: changes over time
        self.radius = None        # State: computed and stored

    def evolve(self, dt):
        """Evolve star forward in time (mutation!)."""
        self.age += dt            # Mutation: change internal state
        self.radius = self._compute_radius()  # Mutation: update state

    def _compute_radius(self):
        """Compute radius based on current mass and age (ZAMS approximation)."""
        # Mass-radius relation for main sequence stars (solar units)
        if self.mass < 1.0:
            radius = self.mass**0.8  # Low-mass stars
        else:
            radius = self.mass**0.57  # High-mass stars
        return radius

    def compute_luminosity(self):
        """Method: operates on internal state."""
        return 4 * np.pi * self.radius**2 * self.effective_temperature**4
```

**Key characteristics of OOP**:
- **Encapsulation**: Data (`mass`, `age`, `radius`) bundled with methods
- **State**: Objects hold mutable state that changes over time
- **Methods**: Functions that operate on and modify `self`
- **Mutation**: `self.age += dt` changes the object in-place

**This is intuitive!** Objects model real-world entities (stars) that change over time. You call methods, the object's state updates, everything makes sense.

### What Is Functional Programming?

Functional programming is a **completely different paradigm**:

```python
def evolve_star(star_state, dt):
    """
    Functional design (what JAX requires).

    Takes: immutable star_state dict, time step
    Returns: NEW star_state dict (doesn't modify input)
    """
    new_age = star_state['age'] + dt
    new_radius = compute_radius(star_state['mass'], new_age)

    # Return NEW dict, original star_state unchanged
    return {
        'mass': star_state['mass'],      # Unchanged
        'age': new_age,                   # New value
        'radius': new_radius              # New value
    }

def compute_radius(mass, age):
    """
    Pure function: same inputs → always same output.

    Simplified main-sequence stellar radius evolution (CGS units).
    Based on R ∝ M^0.8 scaling with slow contraction over lifetime.

    Args:
        mass: stellar mass [M☉]
        age: stellar age [years]

    Returns:
        radius: stellar radius [cm]
    """
    # Main-sequence radius (R ∝ M^0.8 for low mass, M^0.57 for high mass)
    R_sun = 6.96e10  # Solar radius [cm]
    if mass < 1.0:
        R_MS = R_sun * mass**0.8   # Low-mass stars
    else:
        R_MS = R_sun * mass**0.57  # High-mass stars

    # Main-sequence lifetime (τ_MS ∝ M^-2.5)
    tau_MS = 1e10 * (mass)**(-2.5)  # years

    # Simple contraction with age: R(t) = R_MS * (1 - 0.05 * t/τ_MS)
    # Stars slowly contract on main sequence (5% over lifetime)
    radius = R_MS * (1.0 - 0.05 * age / tau_MS)
    return radius

def compute_luminosity(radius, temperature):
    """Pure function: no hidden state, no side effects."""
    return 4 * np.pi * radius**2 * temperature**4
```

:::{admonition} 🔗 Connection to Project 1
:class: note

These are the **same ZAMS relations** you implemented in Project 1! The difference:
- **Project 1**: OOP design with `Star` class and mutable state
- **Here**: Functional design with pure functions and immutable data

Same physics, different programming paradigm. By the end of Module 6, you'll appreciate why the functional approach enables autodiff, JIT, and vectorization.
:::

**Key characteristics of functional programming**:

- **Immutability**: Data never changes, only transformed
- **Pure functions**: Same inputs → always same outputs
- **No side effects**: Functions don't modify anything outside their scope
- **Transformations**: Create new data instead of mutating existing data

### Side-by-Side Comparison

| **Aspect** | **OOP (Your `Star` Class)** | **Functional (JAX Style)** |
|------------|------------------------------|----------------------------|
| **Data** | Mutable state in object | Immutable values |
| **Time evolution** | `self.age += dt` | `new_state = evolve(old_state, dt)` |
| **Functions** | Methods modify `self` | Pure functions return new values |
| **Calling** | `star.evolve(dt)` changes star | `new_star = evolve(star, dt)` preserves star |
| **Philosophy** | Objects **change** over time | Data **transforms** through functions |

### Why This Feels Weird

If you're thinking "functional programming seems unnecessarily complicated," **you're not alone**. Your brain has been trained on:

- **Imperative style**: "Do this, then do that, then modify this variable"
- **OOP**: Objects that encapsulate state and behavior
- **Mutation**: Updating variables in-place feels natural

Functional programming requires a different mental model:

- Don't think: "The star ages"
- Think: "I create a new star-state from the old star-state + time step"

**This will feel awkward for a few weeks. Then it will click. Trust the process.**

### Example: Simulating 10 Time Steps

**OOP way** (mutation):

```python
star = Star(mass=1.0)  # Create object with state

for _ in range(10):
    star.evolve(dt)    # Mutate star's internal state each step
    print(star.age, star.radius)  # State keeps changing
```

**Functional way** (transformation):
```python
star_state = {'mass': 1.0, 'age': 0.0, 'radius': None}  # Initial state

for _ in range(10):
    star_state = evolve_star(star_state, dt)  # Transform → new state
    print(star_state['age'], star_state['radius'])
```

In the functional version:

- Each iteration creates a **new** `star_state`
- The old state isn't modified (though Python may reuse memory under the hood)
- We could keep a history: `states = [evolve(s, dt) for s in ...]`

### Why JAX Cares About This

**JAX transformations require functional style.** Here's why:

1. **JIT compilation** needs to know data flow at compile time
   - Mutation makes data flow implicit (`star.age` could change anywhere)
   - Pure functions make data flow explicit (inputs → outputs)

2. **Automatic differentiation** needs to track dependencies
   - If `star.radius` can be modified from anywhere, how do we track gradients?
   - Pure functions: clear dependency graph

3. **Vectorization (vmap)** needs independent computations
   - If functions modify shared state, they can't run in parallel
   - Pure functions are independent by definition

**The paradigm shift**:

- OOP: "I have a star object, I evolve it"
- Functional: "I have star data, I transform it through functions"

### Practical JAX Example

Here's actual JAX code for N-body simulation:

```python
import jax.numpy as jnp

# OOP style (would break JAX!)
class Particle:
    def __init__(self, position, velocity):
        self.position = position  # ❌ Mutable state
        self.velocity = velocity  # ❌ Mutable state

    def update(self, force, dt):
        self.velocity += force * dt  # ❌ In-place mutation
        self.position += self.velocity * dt  # ❌ Breaks JAX

# Functional style (JAX-compatible!)
def update_particle(state, force, dt):
    """
    Pure function: (state, force, dt) → new_state
    No mutation, clear data flow.
    """
    pos, vel = state
    new_vel = vel + force * dt       # ✅ Create new array
    new_pos = pos + new_vel * dt     # ✅ Create new array
    return (new_pos, new_vel)        # ✅ Return new state
```

**Why the functional version works with JAX**:

- Clear inputs: `state`, `force`, `dt`
- Clear output: `new_state`
- No hidden dependencies
- JAX can JIT compile, autodiff, and vectorize this trivially

### Connection to Your Learning This Semester

You learned OOP because it's:

- **Intuitive** for modeling real-world entities
- **Widely used** in scientific Python (NumPy arrays are objects!)
- **Good software engineering** for organizing complex systems

You're now learning functional programming because it's:

- **Required** for JAX transformations
- **Enables** automatic differentiation and compilation
- **The paradigm** of modern ML frameworks

**Both are valuable.** You'll use OOP for structuring your overall project (Package class hierarchy) and functional programming for the computational core (JAX-transformed functions).

### Key Takeaway

**Functional programming isn't "better" than OOP** — they're tools for different jobs.

- **OOP**: Great for modeling stateful systems, organizing large codebases
- **Functional**: Required for JAX transformations, mathematical clarity

By Project 5, you'll naturally write:

- Functional JAX code for performance-critical computation
- OOP Python code for project structure and I/O

The mental shift happens in the next few weeks. Be patient with yourself.

---

(pure-functions)=
## 1.2: Pure Functions — The Foundation of JAX

**Priority: 🔴 Essential**

:::{margin}
**Pure Function**
A function where (1) same inputs always produce same outputs (deterministic), and (2) no side effects occur (no mutation, no I/O, no hidden state).
:::

### What Is a Pure Function?

A **pure function** has two properties:

1. **Deterministic**: Same inputs → always same outputs (no randomness, no hidden state)
2. **No side effects**: Doesn't modify anything outside its scope (no mutation, no I/O, no global state changes)

### Examples: Pure vs Impure

#### ✅ Pure Functions

```python
def add(x, y):
    """Pure: same inputs → same output, no side effects."""
    return x + y

def gravitational_force(m1, m2, r, G=6.67e-8):
    """Pure: deterministic physics calculation (CGS units)."""
    return G * m1 * m2 / r**2

def compute_energy(positions, velocities, masses):
    """Pure: all inputs explicit, output deterministic."""
    kinetic = 0.5 * jnp.sum(masses * jnp.sum(velocities**2, axis=1))
    potential = compute_potential(positions, masses)
    return kinetic + potential
```

**Why these are pure**:

- All inputs are function parameters (no hidden dependencies)
- Same `(m1, m2, r)` → always same force
- No modifications to anything outside the function
- You could call `add(3, 5)` a million times, always get `8`

#### ❌ Impure Functions

```python
# Impure: Global state (hidden dependency)
counter = 0
def impure_add(x, y):
    global counter
    counter += 1  # Side effect: modifies global state
    return x + y  # Output depends on when you call it (if you print counter)

# Impure: Non-deterministic
import random
def impure_random():
    return random.random()  # Different output every call!

# Impure: I/O (side effect)
def impure_log(x):
    print(f"Computing for x={x}")  # Side effect: prints to console
    with open('log.txt', 'a') as f:  # Side effect: modifies file
        f.write(f"{x}\n")
    return x ** 2

# Impure: Mutation
def impure_accumulate(array, value):
    array.append(value)  # Side effect: modifies input array
    return array  # Violates immutability

# Impure: Time-dependent
import time
def impure_timestamp():
    return time.time()  # Different output every call!

# Impure: Hidden state
class StatefulCalculator:
    def __init__(self):
        self.history = []  # Hidden state

    def add(self, x, y):
        result = x + y
        self.history.append(result)  # Side effect: modifies self
        return result
```

**Why these are impure**:

- **Global state**: `counter` changes → same inputs, different behavior over time
- **Non-deterministic**: Random numbers, timestamps → different outputs each call
- **I/O**: Printing, file writing → side effects beyond return value
- **Mutation**: Modifying input arrays or object state → breaks referential transparency
- **Hidden state**: Behavior depends on `self.history` → not visible in function signature

### Why Purity Matters for JAX

JAX transformations **require** pure functions. Here's why:

#### 1. JIT Compilation Needs Determinism

When JAX JIT-compiles a function, it traces it **once** with abstract values:

```python
@jax.jit
def f(x):
    return x ** 2
```

JAX's compilation:

1. Call `f` with "tracer" (abstract value, knows shape/dtype, not actual value)
2. Record all operations: `square(x)`
3. Compile graph to optimized machine code
4. **Reuse** compiled code for all future calls

**If your function is impure**, the compiled graph won't match reality:

```python
counter = 0

@jax.jit
def impure_f(x):
    global counter
    counter += 1  # Side effect!
    return x + counter

# First call: counter=0, compiles x + 0
result1 = impure_f(5.0)  # Returns 5.0

# Second call: uses cached compiled code (x + 0), but counter=1 now!
result2 = impure_f(5.0)  # Still returns 5.0 (not 6.0!)
```

**The compiled graph is frozen**—it doesn't re-execute Python, just the compiled operations.

#### 2. Automatic Differentiation Needs Clear Dependencies

Autodiff builds a computational graph by tracking dependencies:

```python
def f(x):
    y = x ** 2
    z = jnp.sin(y)
    return z

# JAX builds graph:
# x → [square] → y → [sin] → z
# Backward pass: dz/dx = dz/dy * dy/dx
```

**If you have hidden dependencies**, JAX can't track them:

```python
global_constant = 5.0

def impure_f(x):
    return x * global_constant  # Hidden dependency!

# JAX can't differentiate w.r.t. global_constant
# It only sees: f(x) = x * [some constant]
grad_f = jax.grad(impure_f)  # Only d/dx, missing d/d(global_constant)
```

#### 3. Vectorization (vmap) Needs Independence

`vmap` parallelizes computations over a batch dimension:

```python
def f(x):
    return x ** 2

# Vectorize: apply f to each element independently
batched_f = jax.vmap(f)
batched_f(jnp.array([1, 2, 3]))  # [1, 4, 9]
```

**If functions share mutable state**, they can't be parallelized:

```python
accumulator = 0.0

def impure_f(x):
    global accumulator
    accumulator += x  # Race condition in parallel execution!
    return accumulator

# vmap would have race conditions—undefined behavior
```

:::{admonition} 💡 Special Case: Random Numbers in JAX
:class: important

**NumPy/Python random** is impure (non-deterministic global state):

```python
import numpy as np

def monte_carlo_step_impure():
    return np.random.normal()  # Different every call!

# Multiple calls give different outputs
print(monte_carlo_step_impure())  # 0.47143906
print(monte_carlo_step_impure())  # -1.23982791
print(monte_carlo_step_impure())  # 0.89276421

# JAX problem: I compiled this function once, but it gives
# different outputs with SAME (empty) input?
# Cannot trace! Cannot compile reliably!
```

**JAX solution**: Explicit PRNG keys (pure randomness):

```python
import jax

def monte_carlo_step_pure(key):
    """Pure: same key → same output."""
    return jax.random.normal(key)

# Create initial key
key = jax.random.PRNGKey(42)

# Split key for multiple samples
key, subkey1 = jax.random.split(key)
print(monte_carlo_step_pure(subkey1))  # Always: 0.47143906 (given key 42)

key, subkey2 = jax.random.split(key)
print(monte_carlo_step_pure(subkey2))  # Always: -1.23982791 (given split)

# Same key → same result (deterministic!)
print(monte_carlo_step_pure(subkey1))  # 0.47143906 again!
```

**Why JAX requires this**:
- **Determinism**: Same key → same random number (reproducible)
- **JIT-compilable**: Key is just data (array), no hidden global state
- **vmap-able**: Each batch element gets its own key → no race conditions
- **Autodiff-able**: Can differentiate through random sampling (reparameterization trick)

**You'll use this extensively in Module 7** (Monte Carlo methods, Bayesian inference, stochastic simulations).
:::

### The Mutation Problem in Detail

This is the #1 stumbling block for newcomers:

```python
# NumPy style (mutation)
def update_positions_numpy(positions, velocities, dt):
    positions += velocities * dt  # ❌ In-place mutation
    return positions

# This breaks JAX! positions array is modified in-place
# JAX can't track this properly for autodiff/jit
```

**JAX solution**: Explicit updates that return new arrays:

```python
# JAX style (functional)
def update_positions_jax(positions, velocities, dt):
    return positions + velocities * dt  # ✅ New array created

# For more complex updates, JAX provides .at[] syntax:
def update_element_jax(array, index, value):
    return array.at[index].set(value)  # ✅ Returns new array

# Example:
x = jnp.array([1, 2, 3, 4, 5])
y = x.at[2].set(99)  # y = [1, 2, 99, 4, 5], x unchanged
```

### Common "Impure" Patterns and JAX Equivalents

| **Impure Pattern (breaks JAX)** | **Pure Pattern (JAX-compatible)** |
|----------------------------------|-----------------------------------|
| `x[i] = new_value` | `x = x.at[i].set(new_value)` |
| `x += delta` | `x = x + delta` |
| `list.append(item)` | `list = list + [item]` (or use `jnp.concatenate`) |
| `if random.random() > 0.5:` | `key, subkey = jax.random.split(key); if jax.random.uniform(subkey) > 0.5:` |
| `print(x)` | Use `jax.debug.print(x)` sparingly |

### Testing for Purity: The Substitution Test

A function is pure if you can **replace the function call with its return value** without changing behavior:

```python
# Pure: this substitution is always valid
def f(x):
    return x ** 2

y = f(3) + f(3)  # Can be optimized to:
y = 9 + 9

# Impure: substitution changes meaning
counter = 0
def g(x):
    global counter
    counter += 1
    return x

y = g(3) + g(3)  # counter increments twice
y = 3 + 3        # counter doesn't increment—different behavior!
```

This is called **referential transparency**: pure functions can be replaced by their values.

### Practical Exercise: Identify Purity

```python
# Which of these are pure?

def func1(x, y):
    return x + y

def func2(array):
    return jnp.sum(array)

def func3(x):
    print(f"Computing {x}")
    return x ** 2

def func4(x, external_param=5):
    return x * external_param

params = {'scale': 5}
def func5(x):
    return x * params['scale']

def func6(x):
    return jnp.linalg.norm(x)

cache = {}
def func7(x):
    if x in cache:
        return cache[x]
    result = expensive_computation(x)
    cache[x] = result
    return result
```

**Answers**:
- ✅ `func1`: Pure (deterministic, no side effects)
- ✅ `func2`: Pure (depends only on input)
- ❌ `func3`: Impure (side effect: prints)
- ✅ `func4`: Pure (default params are fine, part of function signature)
- ❌ `func5`: Impure (depends on external mutable dict)
- ✅ `func6`: Pure (deterministic JAX operation)
- ❌ `func7`: Impure (side effect: modifies cache)

### Key Takeaway

**Purity is JAX's foundation.**

- Same inputs → same outputs (always)
- No side effects (no mutation, no I/O, no hidden state)
- Enables JIT, autodiff, vmap to work correctly

**By Project 5**, checking for purity will be second nature. For now, when in doubt:
- Can I replace `f(x)` with its return value? → Pure
- Does the function modify anything outside its scope? → Impure
- Does the function depend on anything not in its parameters? → Impure

:::{important} 💡 What We Just Learned

**Pure functions are the foundation of JAX transformations:**

- **Deterministic**: Same inputs always produce same outputs (no randomness, no hidden state)
- **No side effects**: Don't modify anything outside their scope
- **Enable transformations**: JIT compilation, automatic differentiation, and vectorization all require purity
- **Practical test**: If you can replace `f(x)` with its return value everywhere and the program behaves identically, it's pure

This constraint isn't arbitrary—it's what makes JAX's magic possible.
:::

---

(control-flow)=
## 1.3: Control Flow Constraints — Why `if` and `for` Break JAX

**Priority: 🔴 Essential**

### The Problem: Data-Dependent Control Flow

Consider this innocent-looking function:

```python
def f(x):
    if x > 0:
        return x ** 2
    else:
        return -x

# Works fine in NumPy:
print(f(5))   # 25
print(f(-5))  # 5
```

**Try to JIT-compile it:**

```python
@jax.jit
def f_jit(x):
    if x > 0:  # ❌ This will fail!
        return x ** 2
    else:
        return -x

f_jit(5.0)  # ConcretizationError!
```

**Error**:
```
ConcretizationError: Abstract tracer value encountered where concrete value expected.
The error occurred while tracing the function f_jit for jit compilation.
```

**What happened?**

### Understanding JIT Compilation and Tracing

When JAX JIT-compiles a function:

1. **Tracing phase**: JAX calls your function with **abstract values** (tracers)
   - Tracers have shape and dtype: `ShapedArray(shape=(), dtype=float32)`
   - Tracers don't have concrete values: Can't evaluate `x > 0`

2. **Graph building**: JAX records operations to build computational graph

3. **Compilation**: Graph → optimized machine code (XLA)

4. **Execution**: Compiled code runs with actual values

**The problem**: `if x > 0:` requires knowing `x`'s value, but during tracing we only have its shape/dtype!

### What Is Data-Dependent Control Flow?

**Data-dependent**: Control flow decision depends on **runtime values**

```python
# Data-dependent (value of x determines path)
if x > 0:  # Depends on x's value at runtime
    ...

# Data-dependent (loop count depends on value)
for i in range(n):  # If n is a traced variable, this fails
    ...

# Data-dependent (while loop with data condition)
while x > tolerance:  # Condition depends on runtime value
    ...
```

**Not data-dependent**: Control flow known at compile time

```python
# Static (shape-dependent, okay)
if x.shape[0] > 10:  # Shape is known during tracing
    ...

# Static (hardcoded, okay)
for i in range(100):  # Loop count is constant
    ...

# Static (shape-based, okay)
for i in range(x.shape[0]):  # Shape known at tracing
    ...
```

### Why This Matters for JIT

**JAX needs to compile ONE graph that works for ALL possible input values**:

```python
def f(x):
    if x > 0:
        return x ** 2  # Graph A
    else:
        return -x      # Graph B

# JAX can't know which graph to compile!
# x could be positive or negative at runtime
```

If JAX picked one branch during tracing:
- Trace with `x > 0` → compiles Graph A
- Run with `x < 0` → wrong graph! (executes `x ** 2` even though x is negative)

**JAX refuses to guess**—throws ConcretizationError instead.

### JAX's Solution: `jax.lax.cond`

JAX provides **functional control flow primitives** that compile **both branches**:

```python
import jax
import jax.lax as lax

def f_jax(x):
    # lax.cond compiles both branches, selects at runtime
    return lax.cond(
        x > 0,              # Predicate (boolean)
        lambda x: x ** 2,   # True branch (function)
        lambda x: -x,       # False branch (function)
        x                   # Operand (passed to selected branch)
    )

# Now JIT works!
f_jit = jax.jit(f_jax)
print(f_jit(5.0))   # 25.0
print(f_jit(-5.0))  # 5.0
```

**How `lax.cond` works**:
1. During tracing: Compiles **both** branches
2. At runtime: Evaluates predicate, selects which branch output to return
3. Both branches are in the compiled graph—no runtime Python interpretation

**Syntax**:
```python
lax.cond(predicate, true_fun, false_fun, operand)
#        ↑          ↑          ↑          ↑
#        bool       callable   callable   argument(s)
```

### Loops: Why `for` and `while` Are Tricky

Similar problem with loops:

```python
@jax.jit
def sum_until_large(n):
    total = 0
    for i in range(n):  # ❌ n is traced variable
        total += i
    return total

# ConcretizationError: can't determine loop count at compile time
```

**JAX solutions**:

#### 1. `jax.lax.fori_loop` — Simple fixed-count loops

```python
def sum_until_large_jax(n):
    def body_fun(i, total):
        return total + i

    return lax.fori_loop(
        0,          # Start index
        n,          # End index (exclusive)
        body_fun,   # Body: (index, carry) → new_carry
        0           # Initial carry value
    )

sum_jit = jax.jit(sum_until_large_jax)
print(sum_jit(10))  # 45 = 0+1+2+...+9
```

#### 2. `jax.lax.scan` — Loops with accumulation

More powerful: accumulate values + optionally collect outputs:

```python
def cumulative_sum_jax(array):
    def scan_fun(carry, x):
        new_carry = carry + x
        output = new_carry  # What to collect
        return new_carry, output

    final_carry, outputs = lax.scan(
        scan_fun,        # (carry, x) → (new_carry, output)
        0,               # Initial carry
        array            # Sequence to scan over
    )
    return final_carry, outputs

# Example:
result, intermediate = cumulative_sum_jax(jnp.array([1, 2, 3, 4, 5]))
# result = 15 (final sum)
# intermediate = [1, 3, 6, 10, 15] (cumulative sums)
```

#### 3. `jax.lax.while_loop` — Conditional loops

```python
def newton_method_jax(f, df, x0, tolerance=1e-6):
    def cond_fun(state):
        x, error = state
        return error > tolerance  # Continue while error > tolerance

    def body_fun(state):
        x, _ = state
        x_new = x - f(x) / df(x)
        error = jnp.abs(x_new - x)
        return (x_new, error)

    init_state = (x0, 1.0)  # (initial x, large initial error)
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    return final_state[0]
```

### Quick Reference: Control Flow Patterns

| **Pattern** | **NumPy / Python** | **JAX Equivalent** |
|-------------|-------------------|-------------------|
| If-else | `if cond: ... else: ...` | `lax.cond(predicate, true_fn, false_fn, operand)` |
| Element-wise conditional | `np.where(cond, x, y)` | `jnp.where(cond, x, y)` ✅ (works fine!) |
| Fixed loop | `for i in range(n):` | `lax.fori_loop(start, end, body, init)` |
| Accumulating loop | `for x in array:` | `lax.scan(fun, init, xs)` |
| While loop | `while cond:` | `lax.while_loop(cond_fun, body_fun, init)` |

### When Do You Actually Need `lax` Functions?

**Good news**: Many cases work without `lax`!

#### ✅ These work fine (no `lax` needed):

```python
# Element-wise conditionals (jnp.where)
result = jnp.where(x > 0, x**2, -x)  # ✅ Works!

# Shape-based control flow
if x.ndim == 2:  # ✅ Shape known at tracing
    ...

# Static loops (hardcoded range)
for i in range(10):  # ✅ Constant loop count
    ...

# Shape-based loops
for i in range(x.shape[0]):  # ✅ Shape known
    ...
```

#### ❌ These need `lax`:

```python
# Value-dependent conditionals
if x > threshold:  # ❌ Use lax.cond

# Value-dependent loop counts
for i in range(n):  # ❌ Use lax.fori_loop (if n is traced)

# While loops with data-dependent conditions
while error > tolerance:  # ❌ Use lax.while_loop
```

### Practical Example: N-body Timesteps

```python
# NumPy style (won't JIT compile if n_steps is dynamic)
def integrate_numpy(state, forces_fn, dt, n_steps):
    for step in range(n_steps):  # Breaks if n_steps is traced
        forces = forces_fn(state)
        state = update_state(state, forces, dt)
    return state

# JAX style (JIT-compilable)
def integrate_jax(state, forces_fn, dt, n_steps):
    def step_fn(carry, _):
        state = carry
        forces = forces_fn(state)
        new_state = update_state(state, forces, dt)
        return new_state, None  # (new_carry, output)

    final_state, _ = lax.scan(step_fn, state, jnp.arange(n_steps))
    return final_state
```

### Why These Constraints?

**It's all about compilation**:

- Python `if`/`for`/`while` are **interpreted** → can't be compiled ahead of time
- `lax.cond`/`fori_loop`/`scan` are **declarative** → JAX knows the structure
- JAX compiles the entire control flow → runs at machine code speed

**Trade-off**:
- **Cost**: More verbose, requires functional thinking
- **Benefit**: Entire program compiles → 10-100× faster execution

### Key Takeaway (Conceptual Understanding)

**JAX can't trace through Python control flow that depends on data values.**

**Why?**
- Tracing uses abstract values (shapes/types, not actual numbers)
- `if x > 0` requires knowing `x`'s value
- JAX doesn't know which branch to compile

**Solution?**
- `lax.cond`, `lax.scan`, `lax.fori_loop`, `lax.while_loop`
- These compile ALL possible paths, select at runtime
- Works because JAX knows the structure (not the values)

**In Part 2**, you'll learn the technical details and practical patterns. For now, understand the **why**: JAX's constraints enable compilation, and compilation enables speed.

---

(computational-graphs)=
## 1.4: Computational Graphs — The Mental Model

**Priority: 🟡 Important**

:::{margin}
**Computational Graph**
A directed acyclic graph (DAG) representing operations (nodes) and data flow (edges). JAX builds these graphs to enable transformations like JIT compilation and automatic differentiation.
:::

### How Does JAX "See" Your Code?

When you write a function, you see Python code. When JAX sees your function, it sees a **computational graph**—a directed acyclic graph (DAG) of operations.

Understanding this mental model is crucial for:
- Debugging JAX errors
- Understanding autodiff
- Writing efficient JAX code

### What Is a Computational Graph?

A computational graph is a data structure representing a computation:

- **Nodes**: Operations (add, multiply, sin, exp, etc.)
- **Edges**: Data flow (outputs of one op feed inputs of next)

**Simple example**:

```python
def f(x):
    y = x ** 2        # Operation 1: square
    z = jnp.sin(y)    # Operation 2: sin
    return z
```

**Computational graph**:
```
x → [square] → y → [sin] → z
```

Each arrow carries a value (tensors/arrays), each box is an operation.

### Slightly More Complex Example

```python
def f(x):
    a = x + 1          # Op 1: add
    b = x * 2          # Op 2: multiply
    c = a * b          # Op 3: multiply
    return c
```

**Graph**:
```
    x
   / \
  /   \
[+1]  [*2]
  |    |
  a    b
   \  /
    \/
   [*]
    |
    c
```

Notice:
- Multiple operations can use the same input (`x` feeds both `a` and `b`)
- Operations can combine results from different branches (`c = a * b`)
- The graph is a DAG: no cycles (important for autodiff!)

### Real Physics Example: Gravitational Potential Energy

```python
def gravitational_potential(positions, masses, G=6.67e-8):
    """
    U = -G * sum_{i<j} (m_i * m_j / r_ij)
    """
    n = len(masses)
    U_total = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r_vec = positions[j] - positions[i]     # Op: vector subtraction
            r = jnp.linalg.norm(r_vec)               # Op: norm (sqrt of sum of squares)
            U_ij = -G * masses[i] * masses[j] / r    # Ops: multiply, divide
            U_total += U_ij                           # Op: accumulate
    return U_total
```

**Simplified graph** (for 2 particles):
```
positions[0], positions[1]
         |
      [subtract]
         |
       r_vec
         |
       [norm]
         |
         r
         |    masses[0], masses[1], G
         |         |         |        |
         └─────────[multiply]─────────┘
                   |
                [divide]
                   |
                 U_ij
```

JAX builds this graph **automatically** by tracing your Python code.

### Why Computational Graphs Matter

#### 1. **Automatic Differentiation**

Once you have a graph, you can compute derivatives **automatically** via chain rule:

**Forward pass**: Compute values from inputs to outputs
**Backward pass** (autodiff): Compute gradients from outputs to inputs

```python
def f(x):
    y = x ** 2       # y = x²
    z = jnp.sin(y)   # z = sin(y) = sin(x²)
    return z

# Graph:
# x → [square] → y → [sin] → z

# Gradients (chain rule):
# dz/dx = dz/dy * dy/dx
#       = cos(y) * 2x
#       = cos(x²) * 2x
```

JAX computes this **automatically** by traversing the graph backward.

**More in Part 2!** This is just the conceptual foundation.

:::{admonition} 🔬 What Tracing Actually Looks Like
:class: tip

**During tracing, JAX doesn't use real values—it uses "tracers" (abstract representations).**

Here's what your function "sees" during JIT compilation:

```python
import jax
import jax.numpy as jnp

def f(x):
    print(f"Type of x: {type(x)}")
    print(f"Value of x: {x}")
    y = x + 1
    print(f"Type of y: {type(y)}")
    z = y * 2
    return z

# First call: tracing
print("=== First call (tracing) ===")
f_jit = jax.jit(f)
result = f_jit(3.0)

print("\n=== Second call (cached) ===")
result = f_jit(5.0)  # Doesn't print—uses cached code!
```

**Output**:
```
=== First call (tracing) ===
Type of x: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>
Value of x: Traced<ShapedArray(float32[])>with<DynamicJaxprTrace(level=1/0)>
Type of y: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>

=== Second call (cached) ===
(no output—compiled code runs directly, no Python execution)
```

**Key insight**: `x` is NOT the number `3.0` during tracing! It's an abstract `ShapedArray` that knows:
- **Shape**: `()` (scalar)
- **Type**: `float32`
- But **NOT the actual value**!

**This is why `if x > 0` fails**—JAX doesn't know if `x` is positive during tracing! The value is abstract.

**Second call doesn't trace**—JAX reuses the compiled code. Your print statements don't execute. The function runs as optimized machine code.
:::

#### 2. **JIT Compilation Optimization**

Having the graph lets XLA compiler optimize:

**Operation fusion**:
```python
# Your code:
y = x + 1
z = y * 2
w = z ** 2

# Naive execution: 3 separate kernels
# Optimized: XLA fuses into 1 kernel: w = ((x + 1) * 2) ** 2
```

**Dead code elimination**:
```python
# Your code:
y = expensive_computation(x)
z = x ** 2
return z  # y is never used!

# XLA removes expensive_computation from graph
```

**Memory optimization**:
```python
# XLA can reuse buffers when intermediate values aren't needed
```

#### 3. **Vectorization (vmap)**

With the graph, `vmap` can automatically broadcast operations over batch dimensions.

### Tracing: How JAX Builds the Graph

When you call a JIT-compiled function:

```python
@jax.jit
def f(x):
    y = x ** 2
    z = jnp.sin(y)
    return z

result = f(3.0)
```

**What happens**:

1. **First call**: Tracing phase
   - JAX calls `f` with a "tracer" (abstract value)
   - Tracer: `ShapedArray(shape=(), dtype=float32)`
   - Records operations: `square`, then `sin`
   - Builds graph: `x → [square] → y → [sin] → z`
   - Compiles graph to machine code

2. **Subsequent calls**: Use cached compiled code
   - No Python execution!
   - Just run compiled graph with new values

**This is why side effects break JIT**:

```python
@jax.jit
def f(x):
    print(f"x = {x}")  # Side effect
    return x ** 2

f(3.0)  # Prints: "x = Traced<ShapedArray(...)>" (tracer, not 3.0!)
f(5.0)  # Doesn't print! (cached compiled code)
```

The `print` happens during **tracing** (first call), not during **execution** (subsequent calls).

### Pure Functions → Clean Graphs

**Pure functions** produce clean, predictable graphs:

```python
# Pure function
def f(x, y):
    return x ** 2 + y ** 2

# Graph is simple:
# x, y → [square, square, add] → result
```

**Impure functions** produce unpredictable graphs:

```python
# Impure function
counter = 0
def g(x):
    global counter
    counter += 1
    return x + counter

# Graph doesn't capture `counter` mutation!
# Compiled graph might compute x + 1 (from first trace)
# But counter keeps incrementing → mismatch
```

### Visualizing Graphs with `jax.make_jaxpr`

JAX provides a tool to inspect computational graphs:

```python
import jax

def f(x):
    y = x + 1
    z = y * 2
    return z

# See the graph (JAX intermediate representation)
print(jax.make_jaxpr(f)(3.0))
```

**Output**:
```
{ lambda ; a:f32[]. let
    b:f32[] = add a 1.0
    c:f32[] = mul b 2.0
  in (c,) }
```

This shows:
- `a` is the input (f32 scalar)
- `b = a + 1.0`
- `c = b * 2.0`
- Returns `c`

**Useful for debugging!** If your graph doesn't look like you expect, you've found a bug.

:::{admonition} Stellar Physics Example: Visualizing the Luminosity Computation Graph
:class: note

Let's see what JAX's graph looks like for **computing stellar luminosity** (Stefan-Boltzmann law):

```python
import jax
import jax.numpy as jnp

def stellar_luminosity(radius, temperature):
    """
    Compute luminosity using Stefan-Boltzmann law (CGS units).

    L = 4π R² σ T⁴
    """
    sigma = 5.67e-5  # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
    area = 4 * jnp.pi * radius**2
    flux = sigma * temperature**4
    luminosity = area * flux
    return luminosity

# Visualize the computational graph
print(jax.make_jaxpr(stellar_luminosity)(6.96e10, 5778.0))
```

**Output** (actual jaxpr):
```
{ lambda ; a:f32[] b:f32[]. let
    c:f32[] = integer_pow[y=2] a
    d:f32[] = mul 12.566370964050293 c
    e:f32[] = integer_pow[y=4] b
    f:f32[] = mul 5.67e-05 e
    g:f32[] = mul d f
  in (g,) }
```

**Reading the graph**:
- `a` = radius (R☉), `b` = temperature (K)
- `c = a²` (radius squared)
- `d = 4π × c` (surface area, note 4π ≈ 12.566)
- `e = b⁴` (temperature to 4th power)
- `f = σ × e` (flux per unit area)
- `g = d × f` (total luminosity)
- Returns `g`

**Key insights**:
1. **JAX optimized the computation**: Combined constants (4π became 12.566)
2. **Graph is pure data flow**: No loops, no conditionals, just operations
3. **Each line is differentiable**: `grad` can reverse this graph for autodiff
4. **This graph compiles to fast machine code**: XLA optimizes further

**Try this yourself**: What does the graph look like if you add $L = L_0 (M/M_☉)^{3.5}$?
:::


### Key Insights

1. **JAX sees graphs, not Python code**
   - Your for-loops become `scan` operations in the graph
   - Your if-statements become `cond` nodes
   - Your arithmetic becomes primitive ops

2. **Graphs enable transformations**
   - `jit`: Optimize and compile the graph
   - `grad`: Traverse graph backward computing derivatives
   - `vmap`: Add batch dimensions to graph nodes

3. **Pure functions → predictable graphs**
   - No hidden dependencies
   - No mutations
   - Graph fully describes computation

:::{admonition} 🎯 Conceptual Checkpoint
:class: tip

Before moving on, make sure you can:

- **Draw** a simple computational graph for `y = (x + 1) * (x - 1)`
- **Explain** why `print()` inside a JIT function only executes once
- **Recognize** that JAX builds graphs during tracing, executes them later

**You don't need to memorize graph structures**—JAX handles that automatically. You need to **think in terms of data flow**: inputs → operations → outputs.
:::

---

(jax-transformations)=
## 1.5: What Are JAX Transformations?

**Priority: 🔴 Essential**

### Transformations Are Functions That Transform Functions

This sounds abstract, but it's powerful:

```python
# A normal function
def f(x):
    return x ** 2

# A transformation that takes a function and returns a new function
fast_f = jax.jit(f)        # Returns a compiled version of f
grad_f = jax.grad(f)       # Returns a function that computes df/dx
batched_f = jax.vmap(f)    # Returns a vectorized version of f
```

**You're not just calling functions—you're transforming them.**

### The Four Core Transformations

:::{margin}
**JIT (Just-In-Time) Compilation**
Compiles Python functions to optimized machine code at runtime using XLA (Accelerated Linear Algebra). Typically provides 10-100× speedups.
:::

#### 1. `jit` — Just-In-Time Compilation

**What it does**: Compile your function to machine code for speed

```python
import jax
import jax.numpy as jnp

def slow_function(x):
    return jnp.sum(x ** 2)

fast_function = jax.jit(slow_function)

# Or use decorator:
@jax.jit
def fast_function(x):
    return jnp.sum(x ** 2)
```

**Effect**: Typically 10-100× faster (hardware-dependent)

**When to use**: Almost always! JIT compilation is JAX's superpower.

**Part 2 dives deep**: How XLA works, when JIT overhead dominates, debugging tricks.

#### 2. `grad` — Automatic Differentiation

**What it does**: Take derivatives automatically

```python
def f(x):
    return x ** 2 + 3 * x + 5

# Derivative function
df_dx = jax.grad(f)

print(f(2.0))        # 15.0
print(df_dx(2.0))    # 7.0 = df/dx at x=2 = 2*2 + 3
```

**Effect**: Numerically exact derivatives to machine precision (not finite-difference approximations!)

**When to use**: Optimization, MCMC (HMC), physics-informed ML, anywhere you need gradients

**Part 2 dives deep**: Chain rule mechanics, Jacobians, higher-order derivatives.

#### 3. `vmap` — Automatic Vectorization

**What it does**: Batch computations over arrays automatically

```python
# Function for single input
def f(x):
    return x ** 2

# Vectorized version
batched_f = jax.vmap(f)

# Apply to batch
x_batch = jnp.array([1, 2, 3, 4, 5])
print(batched_f(x_batch))  # [1, 4, 9, 16, 25]
```

**Effect**: Automatic parallelization, GPU utilization

**When to use**: Ensemble simulations, batch processing, anytime you have independent computations

**Part 2 dives deep**: `in_axes`, `out_axes`, nested vmap, combining with grad.

#### 4. `pmap` — Multi-Device Parallelization

**What it does**: Parallelize across GPUs/TPUs

```python
# Run same function on multiple devices
parallel_f = jax.pmap(f)

# Data sharded across devices
x_sharded = jnp.array([...])  # Automatically distributed
result = parallel_f(x_sharded)  # Runs on all devices in parallel
```

**Effect**: Scale to multi-GPU/TPU systems

**When to use**: Large-scale training, ensemble simulations beyond single device

**Part 2 preview**: This is advanced—you'll likely use it less often than `jit`/`grad`/`vmap`.

### The Magic: Transformations Compose

**This is where JAX becomes powerful:**

```python
def loss(params, data):
    predictions = model(params, data)
    return jnp.mean((predictions - targets) ** 2)

# Compose transformations!
fast_batched_gradients = jax.jit(jax.vmap(jax.grad(loss)))

# This is ONE transformation that:
# 1. Computes gradients (grad)
# 2. Over a batch of data (vmap)
# 3. Compiled for speed (jit)
```

**Order matters sometimes**:

```python
jax.vmap(jax.grad(f))  # Batched gradients: gradient for each example
jax.grad(jax.vmap(f))  # Gradient of batched function: different!
```

But you can experiment! JAX transformations are functions—you can combine them any way that's mathematically valid.

### Contrast with NumPy

| **Operation** | **NumPy** | **JAX** |
|---------------|-----------|---------|
| **Compute** | `result = f(x)` | Same: `result = f(x)` |
| **Speed up** | Write C extension | `jax.jit(f)` |
| **Derivatives** | Write `df/dx` by hand | `jax.grad(f)` |
| **Vectorize** | Manual broadcasting | `jax.vmap(f)` |
| **Multi-GPU** | MPI, CUDA kernels | `jax.pmap(f)` |

NumPy requires **doing** optimization/parallelization/differentiation manually. JAX lets you **describe** what you want, then handles the how.

### Example: Training Loop (Preview)

This won't make complete sense yet—that's fine! This shows how transformations combine in practice:

```python
# Loss function
def loss_fn(params, data, targets):
    predictions = model(params, data)
    return jnp.mean((predictions - targets) ** 2)

# Gradient function (w.r.t. params)
grad_fn = jax.grad(loss_fn, argnums=0)

# Vectorized gradient (over batch)
batched_grad_fn = jax.vmap(grad_fn, in_axes=(None, 0, 0))

# JIT-compiled for speed
fast_batched_grad_fn = jax.jit(batched_grad_fn)

# Use in training loop
for epoch in range(num_epochs):
    for batch_data, batch_targets in dataloader:
        # Compute gradients for batch
        grads = fast_batched_grad_fn(params, batch_data, batch_targets)
        # Update parameters
        params = params - learning_rate * jnp.mean(grads, axis=0)
```

**What's happening**:
- `grad`: Compute derivatives w.r.t. params
- `vmap`: Apply to each example in batch
- `jit`: Compile the whole thing
- Result: Fast, batched gradient computation

### Why This Is Different from TensorFlow/PyTorch

**TensorFlow 1.x**: Define static graph, then execute
- Inflexible (had to know structure ahead of time)
- Debugging was hard (no Python introspection)

**PyTorch**: Dynamic graphs, but tightly coupled with autodiff
- Can't easily separate "compile" from "differentiate"
- Harder to compose transformations flexibly

**JAX**: Transformations are **decoupled** and **composable**
- `jit`, `grad`, `vmap` are independent operations
- Combine them in any order that makes sense
- More flexible than static graphs, more composable than PyTorch

### Key Takeaway

**JAX transformations are function transforms**:
- Take a function, return a transformed function
- `jit`: function → compiled function
- `grad`: function → gradient function
- `vmap`: function → batched function
- `pmap`: function → parallel function

**They compose freely**:
- `jit(vmap(grad(f)))` is valid!
- Build complex pipelines from simple transforms

**Part 2 teaches you**:
- How each transformation works internally
- When to use which transformation
- How to combine them effectively
- Common patterns and pitfalls

For now, understand the **concept**: JAX gives you building blocks (transformations) that you compose to build fast, differentiable, parallel scientific code.

---

(why-functional-synthesis)=
## 1.6: Why JAX Requires Functional Programming (Synthesis)

**Priority: 🔴 Essential**

### Connecting All the Pieces

You've learned:
1. **OOP vs Functional**: Different paradigms for organizing code
2. **Pure Functions**: Same inputs → same outputs, no side effects
3. **Control Flow**: Why Python `if`/`for` break, what `lax` provides
4. **Computational Graphs**: How JAX "sees" your code
5. **Transformations**: What `jit`, `grad`, `vmap`, `pmap` actually do

**Now**: Why do these all fit together?

### The Constraint → Capability Table

| **Constraint** | **Why It's Required** | **What It Enables** |
|----------------|----------------------|---------------------|
| **Pure functions** | Deterministic behavior → predictable graphs | JIT compilation, caching, reproducibility |
| **No mutation** | Clear data flow → graph structure preserved | Autodiff, composability, parallelization |
| **Explicit control flow** (`lax`) | Traceable at compile time → structure known | JIT optimization, compile-time loop unrolling |
| **Explicit randomness** (PRNG keys) | Reproducible RNG state → pure sampling | vmap over random processes, deterministic debugging |
| **Functional updates** (`.at[]`) | Immutable arrays → safe parallelization | vmap, pmap work correctly |

**Each constraint enables multiple capabilities.**

### Why Purity Enables JIT

**JIT compilation requires**:
- Tracing code once with abstract values
- Building a computational graph
- Caching compiled code for reuse

**If your function is impure**:
- Hidden dependencies → incomplete graph
- Non-determinism → cached code gives wrong results
- Side effects → not captured in graph

**Example**:
```python
cache = {}  # Global state

@jax.jit
def f(x):
    if x in cache:
        return cache[x]  # Hidden dependency!
    result = expensive_computation(x)
    cache[x] = result  # Side effect!
    return result

# First call (x=3): traces expensive_computation, caches result
# Second call (x=3): uses cached compiled code, but cache[3] exists!
# Result: graph doesn't match reality
```

**Pure version**:
```python
# No global state, explicit dependencies
@jax.jit
def f(x, cache_dict):
    # ... functional caching logic ...
    return result, updated_cache
```

### Why Immutability Enables Autodiff

**Autodiff works by**:
- Building computational graph during forward pass
- Storing intermediate values
- Traversing graph backward, computing gradients via chain rule

**If you mutate data**:
```python
def f(x):
    y = x ** 2
    x = x + 1  # ❌ Mutation! x is now different
    z = x * y  # Which x? Original or mutated?
    return z

# Gradient computation becomes ambiguous
# Which path through the graph did we take?
```

**Immutable version**:
```python
def f(x):
    y = x ** 2
    x_new = x + 1  # ✅ Create new variable
    z = x_new * y  # Clear which values are used
    return z

# Graph is unambiguous:
# x → [square] → y
# x → [+1] → x_new
# (x_new, y) → [multiply] → z
```

### Why Explicit Control Flow Enables Compilation

**Python `if`/`for` are dynamic**:
- Interpreter evaluates conditions at runtime
- Can't compile ahead of time (don't know which branch)

**`lax.cond`/`scan` are declarative**:
- Describe control flow structure
- JAX compiles **all branches**
- Selects at runtime (fast!)

**Example**:
```python
# Python if (can't compile)
def f(x):
    if x > 0:
        return expensive_branch_A(x)
    else:
        return expensive_branch_B(x)

# lax.cond (compiles both branches)
def f_jax(x):
    return lax.cond(
        x > 0,
        expensive_branch_A,
        expensive_branch_B,
        x
    )

# At runtime: evaluates x > 0, selects which output to return
# Both branches compiled → fast execution
```

### Why Functional Style Enables vmap

**vmap requires**:
- Independent computations (no shared state)
- Clear batch dimensions
- No side effects

**If you mutate shared state**:
```python
accumulator = 0.0

def f(x):
    global accumulator
    accumulator += x  # ❌ Race condition!
    return accumulator

# vmap would execute f on multiple x values in parallel
# Race condition: undefined order of accumulator updates
```

**Functional version**:
```python
def f(x, accumulator):
    new_accumulator = accumulator + x  # ✅ Pure transformation
    return new_accumulator, new_accumulator

# vmap works correctly:
batched_f = jax.vmap(f, in_axes=(0, None))
# Each batch element gets its own computation, no interference
```

### The Profound Unity

**All of JAX's constraints serve one purpose: Enable automatic, composable transformations.**

```
Pure Functions
    ↓
Clear Data Flow
    ↓
Traceable Computational Graphs
    ↓
JIT, Grad, Vmap, Pmap
    ↓
Fast, Differentiable, Parallel Scientific Code
```

**You give up**:
- Convenient mutation (`x += 1`)
- Implicit control flow (`if x > 0:`)
- Hidden state (`self.age`)

**You gain**:
- 10-100× speedups (JIT)
- Exact automatic gradients (grad)
- Effortless parallelization (vmap, pmap)
- Composable transformations

### The Glass-Box Connection

**Remember from Module 1**: Glass-box methodology means understanding WHY, not just HOW.

**Black-box JAX user**:
- "JAX is like NumPy but add `@jax.jit` for speed"
- Confused when code breaks
- Cargo-cult programming (copy patterns without understanding)

**Glass-box JAX user** (you, after Part 1):
- "JAX requires purity because JIT compiles graphs"
- Debugs by checking: "Is this function pure? Does it have data-dependent control flow?"
- Writes efficient code naturally because you understand transformations

**By understanding the constraints → you understand the capabilities.**

### Practical Mental Checklist

When writing JAX code, ask:

1. **Is my function pure?**
   - Same inputs → always same outputs?
   - No side effects (mutation, I/O, global state)?

2. **Is my data flow explicit?**
   - All inputs in function parameters?
   - All outputs in return values?
   - No hidden dependencies?

3. **Is my control flow traceable?**
   - Loops with static ranges, or using `lax.scan`?
   - Conditionals using `jnp.where` or `lax.cond`?

4. **Am I creating new data, not mutating?**
   - Using `x = x + 1`, not `x += 1`?
   - Using `.at[].set()`, not `[i] = value`?

**If yes to all → JAX will work smoothly. If no → expect errors.**

:::{important} 💡 What We Just Learned

**JAX's constraints enable its capabilities—this is the key insight:**

- **Functional programming** → enables JAX to build predictable computational graphs
- **Pure functions** → enable JIT compilation, automatic differentiation, and vectorization
- **Explicit control flow** → enables tracing with abstract values (shapes/types, not data)
- **Immutability** → enables safe parallelization and composable transformations

**The paradigm shift**: From "writing scripts that compute values" to "composing mathematical transformations that JAX can optimize, differentiate, and parallelize automatically."

This isn't just faster code—it's a fundamentally different way of thinking about scientific computation.
:::

### Looking Ahead to Part 2

**Part 1** built your conceptual foundation. You understand:
- WHY functional programming
- WHY pure functions
- WHY explicit control flow
- HOW computational graphs work (conceptually)
- WHAT transformations do (conceptually)

**Part 2** teaches technical mastery:
- HOW to use `jit`, `grad`, `vmap`, `pmap` in practice
- WHEN each transformation is appropriate
- Detailed autodiff mathematics (chain rule, Jacobians)
- Composing transformations effectively
- Debugging common errors
- Performance optimization

**You're ready.** The conceptual understanding you've built makes Part 2's technical details make sense.

---


(summary)=
## Summary: What You've Learned

**Priority: 🔴 Essential**

**TL;DR**: JAX's functional constraints (purity, immutability, explicit control) aren't arbitrary—they're precisely what enable automatic transformations (JIT, grad, vmap). Understanding WHY these constraints exist is the key to using JAX effectively.

**Part 1 built the conceptual foundation for JAX**:

✅ **Programming paradigms**: OOP models objects with state; functional models transformations of immutable data

✅ **Pure functions**: Same inputs → same outputs, no side effects. Required for JIT, autodiff, vmap.

✅ **Control flow**: Python `if`/`for` require values at compile time (can't trace). Use `lax.cond`, `lax.scan`, `lax.fori_loop`.

✅ **Computational graphs**: JAX sees your code as DAG of operations. Enables transformations.

✅ **Transformations**: `jit` (compile), `grad` (differentiate), `vmap` (vectorize), `pmap` (parallelize). Compose freely.

✅ **Constraints → Capabilities**: Functional constraints aren't arbitrary—they enable automatic, composable transformations.

**You now understand WHY JAX works the way it does.**

**Next**: [Part 2: Core Transformations](part-02-core-transformations.md) — Learn the technical details, practice using transformations, understand autodiff mathematics, build real JAX code.

**Remember**: Functional programming will feel awkward for a few weeks. Then it will click. By Project 5, you'll prefer it. Trust the process.

---

## Understanding Checklist

Before proceeding to Part 2, ensure you can:

- [ ] **Contrast** OOP and functional programming paradigms with concrete examples
- [ ] **Identify** whether a function is pure or impure by checking determinism and side effects
- [ ] **Explain** why mutation breaks JAX transformations (JIT, grad, vmap)
- [ ] **Recognize** data-dependent control flow that prevents JIT compilation
- [ ] **Describe** what a computational graph is and how JAX builds them
- [ ] **Articulate** why functional programming constraints enable JAX's capabilities

If you answered "yes" to all → **Ready for Part 2: Core Transformations**

---

*Continue to **[Part 2: Core Transformations — From Concepts to Code](part-02-core-transformations.md)** →*

