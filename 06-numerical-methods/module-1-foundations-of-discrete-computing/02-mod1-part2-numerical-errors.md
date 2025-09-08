---
title: "Part 2: Numbers Aren't Real - Computer Arithmetic & Cosmic Consequences"
subtitle: "Module 1: Foundations of Discrete Computing | ASTR 596"
---

## Learning Outcomes

By the end of this section, you will be able to:

- [ ] **Calculate** machine epsilon for different data types and **predict** when round-off errors dominate
- [ ] **Identify** and **classify** the three types of numerical error (round-off, truncation, propagation) in code
- [ ] **Diagnose** numerical issues in calculations and **reformulate** expressions to avoid catastrophic cancellation
- [ ] **Predict** error scaling in iterative calculations and **design** stable algorithms
- [ ] **Apply** appropriate scaling and units to maintain precision across astronomical scales

---

## Why This Matters for Astronomy

Astronomical simulations face unique numerical challenges from extreme dynamic ranges within individual problems:

**Scale contrasts in typical simulations:**

- **Planetary systems**: Tracking Mercury's orbit (0.39 AU) and Neptune (30 AU) simultaneously requires representing positions across 2 orders of magnitude while computing forces that vary by 4 orders of magnitude
- **Star formation**: Modeling a molecular cloud (10 pc) down to protostellar cores (100 AU) spans 6 orders of magnitude in length and 11 in density
- **Star clusters**: Resolving both tight binaries (0.01 AU) and cluster dynamics (10-100 pc) means handling force variations of 16-18 orders of magnitude
- **Galaxy simulations**: Must maintain accuracy from star-forming regions (~1 pc) to dark matter halos (~100 kpc) - 5 orders of magnitude in length

**The precision problem:**
Double precision provides only ~16 decimal digits. This means:

- We cannot track a planet's position to millimeter precision while also representing its distance from the Sun
- Subtracting Earth's position (1 AU) from Mars' position (1.5 AU) loses several digits of precision
- After a million timesteps, even tiny round-off errors can accumulate to destroy energy conservation

**Practical consequences:**
We must carefully:

- Choose units that keep numbers near unity (e.g., AU for solar system, pc for clusters)
- Use hierarchical coordinate systems (e.g., barycentric for systems, heliocentric for planets)
- Monitor error accumulation in long integrations
- Reformulate algorithms to minimize precision loss

## Finding Machine Epsilon

Machine epsilon $\epsilon$ is the smallest number such that $1 + \epsilon \neq 1$ in floating-point arithmetic:

```python
import numpy as np

def find_machine_epsilon():
    """
    Find the smallest power of 2 such that 1 + eps != 1
    Note: This finds 2^(-52) for double precision
    """
    eps = 1.0
    while (1.0 + eps/2.0) != 1.0:
        eps = eps / 2.0
    return eps

# Test and compare with NumPy's value
eps_computed = find_machine_epsilon()
eps_numpy = np.finfo(float).eps
print(f"Computed ε: {eps_computed:.3e}")
print(f"NumPy's ε:  {eps_numpy:.3e}")
assert abs(eps_computed - eps_numpy) < 1e-20, "Mismatch in epsilon"
```

This value determines the fundamental limits of numerical accuracy.

## The Three Types of Error

### 1. Round-off Error (from finite precision)

:::{margin}
**Round-off error**: Error introduced by representing real numbers with finite precision. For example, 1/3 = 0.333... cannot be stored exactly, so it's rounded to 0.33333333333333331 in double precision, introducing an error of ~10^-17.
:::

**Round-off error** occurs when numbers cannot be represented exactly. The most dangerous situation is catastrophic cancellation when subtracting nearly equal numbers.

**Why this matters**: In astrophysics, we often compute small differences between large quantities - like parallax angles, proper motions, or energy changes in orbits. Understanding how to reformulate these calculations preserves precision in your results.

```python
import numpy as np

def demonstrate_catastrophic_cancellation():
    """
    Show how subtracting nearly equal numbers destroys precision
    Example: Computing parallax from distance measurements
    """
    # Earth's orbit baseline: small change in distance to star
    d1_au = 1.0000000  # Distance in AU (winter)
    d2_au = 1.0000067  # Distance in AU (summer) - 1000 km difference
    
    def parallax_bad(d1, d2):
        """Naive: loses precision when d1 ≈ d2"""
        if d1 == 0 or d2 == 0:
            raise ValueError("Distance cannot be zero")
        return 1.0/d1 - 1.0/d2
    
    def parallax_good(d1, d2):
        """Reformulated: preserves precision"""
        if d1 == 0 or d2 == 0:
            raise ValueError("Distance cannot be zero")
        return (d2 - d1)/(d1 * d2)
    
    bad_result = parallax_bad(d1_au, d2_au)
    good_result = parallax_good(d1_au, d2_au)
    relative_error = abs(bad_result-good_result)/good_result*100
    
    print(f"Bad method:  {bad_result:.10e}")
    print(f"Good method: {good_result:.10e}")
    print(f"Relative difference: {relative_error:.2f}%")
    
    return bad_result, good_result, relative_error
```

### 2. Truncation Error (from approximations)

:::{margin}
**Truncation error**: Error from approximating an infinite process with finite terms. For example, stopping a Taylor series after $n$ terms, or approximating a derivative with finite differences instead of taking the limit $h→0$.
:::

**Truncation error** arises from approximating infinite processes with finite ones. For example, truncating the Taylor series for $e^x$:

$$e^x \approx 1 + x + \frac{x^2}{2!} + ... + \frac{x^n}{n!}$$

The error is all the neglected terms beyond $n$.

### 3. Propagation Error (accumulation over iterations)

:::{margin}
**Propagation error**: The accumulation and amplification of errors through repeated calculations.
:::

**Propagation error** is how small errors compound through repeated calculations. This is particularly dangerous in astronomy where we often integrate orbits over millions of timesteps:

```python
import numpy as np

def demonstrate_error_propagation():
    """
    Show how tiny errors grow through repeated operations
    This models what happens in long-term orbital integration
    """
    value = 1.0
    tiny_error = 1e-15  # Below machine precision
    
    # Track error growth
    steps = [1, 10, 100, 1000, 10000, 100000, 1000000]
    results = []
    
    print("Error Propagation in Iterative Calculations:")
    print("Steps    | Total Error | Growth Factor")
    print("-" * 40)
    
    for n in steps:
        result = 1.0
        for _ in range(n):
            result *= (1 + tiny_error)
        
        actual_error = result - 1.0
        growth_factor = actual_error / tiny_error
        results.append((n, actual_error, growth_factor))
        
        print(f"{n:7d} | {actual_error:.3e} | {growth_factor:.1e}")
    
    return results
```

This is why long-term orbital integrations require special care.

:::{admonition} Check Your Understanding
:class: question

1. Why does $0.1 + 0.2 \neq 0.3$ in binary floating-point arithmetic?
2. How would you reformulate $(a^2 - b^2)/(a - b)$ when $a \approx b$?
3. If error grows by factor of 2 each step, how many steps until error > 1?
:::

## Measuring Errors: Absolute vs. Relative

:::{margin}
**Absolute error**: The magnitude of the difference between computed and true values: $|x_{\text{computed}} - x_{\text{true}}|$

**Relative error**: The absolute error normalized by the true value: $\frac{|x_{\text{computed}} - x_{\text{true}}|}{|x_{\text{true}}|}$
:::

### Why Both Matter

For any computed value $x_{\text{computed}}$ and true value $x_{\text{true}}$, we define:

$$\boxed{E_{\text{abs}} = |x_{\text{computed}} - x_{\text{true}}|}$$

$$\boxed{E_{\text{rel}} = \frac{|x_{\text{computed}} - x_{\text{true}}|}{|x_{\text{true}}|} = \frac{E_{\text{abs}}}{|x_{\text{true}}|}}$$

**Key insight**: Relative error gives us a scale-independent measure - 1% error means the same thing whether measuring galaxies or atoms. But it breaks down near zero.

### Worked Example: Computing Jupiter's Mass

Let's calculate both error types step-by-step:

- True value: $M_{\text{true}} = 1.898 \times 10^{30}$ g
- Computed: $M_{\text{computed}} = 1.897 \times 10^{30}$ g

**Step 1: Absolute error**
$$E_{\text{abs}} = |1.897 \times 10^{30} \text{ g} - 1.898 \times 10^{30} \text{ g}| = 1 \times 10^{27} \text{ g}$$

**Step 2: Relative error**
$$E_{\text{rel}} = \frac{1 \times 10^{27} \text{ g}}{1.898 \times 10^{30} \text{ g}} = 5.3 \times 10^{-4} = 0.053\%$$

**Conclusion**: The 0.05% relative error tells us our accuracy regardless of Jupiter's enormous mass. For most astrophysical applications, this is excellent precision.

### When Each Error Type Matters

**Example 1 - Large scales:** Computing Earth-Sun distance

- True value: $x_{\text{true}} = 1.496 \times 10^{13}$ cm  
- Computed: $x_{\text{computed}} = 1.496 \times 10^{13} + 1000$ cm
- $E_{\text{abs}} = 1000$ cm (seems huge!)
- $E_{\text{rel}} = \frac{1000}{1.496 \times 10^{13}} \approx 6.7 \times 10^{-11}$ (actually negligible!)
- **Use relative error** - absolute doesn't capture the insignificance

**Example 2 - Small values:** Computing stellar parallax

- True value: $x_{\text{true}} = 0.001$ arcsec
- Computed: $x_{\text{computed}} = 0.002$ arcsec  
- $E_{\text{abs}} = 0.001$ arcsec (seems tiny!)
- $E_{\text{rel}} = \frac{0.001}{0.001} = 1.0$ (100% error - catastrophic!)
- **Use relative error** - reveals the true magnitude of error

**Example 3 - Near zero:** Computing velocity at aphelion

- True value: $x_{\text{true}} = 10^{-12}$ km/s (essentially zero)
- Computed: $x_{\text{computed}} = 10^{-11}$ km/s
- $E_{\text{abs}} = 9 \times 10^{-12}$ km/s  
- $E_{\text{rel}} = 9.0$ (900% error - but both values are negligible!)
- **Use absolute error** - relative error is misleading near zero

**Decision rule**: Use relative error when $|x_{\text{true}}| > 10^{-10} \times x_{\text{scale}}$, where $x_{\text{scale}}$ is the typical scale of your problem. Otherwise, use absolute error.

```python
def compute_errors(computed, true, threshold=1e-10):
    """Compute both errors, intelligently choosing which to trust"""
    abs_error = abs(computed - true)
    
    if abs(true) > threshold:
        rel_error = abs_error / abs(true)
        primary = ('relative', rel_error)
    else:
        rel_error = float('inf')  # Undefined but we record it
        primary = ('absolute', abs_error)
    
    return abs_error, rel_error, primary
```

### Tracking Accumulated Error

In iterative algorithms, tiny errors compound. After $n$ steps with error $\epsilon$ per step:

$$E_{\text{accumulated}} \approx n \cdot \epsilon \quad \text{(if errors are random)}$$
$$E_{\text{accumulated}} \approx \epsilon \cdot (1 + \epsilon)^n - 1 \quad \text{(if errors are systematic)}$$

```python
# Systematic error growth (worst case)
value = 1.0
epsilon = 1e-15  # Machine precision error
for n in [10, 1000, 100000, 1000000]:
    accumulated_error = epsilon * ((1 + epsilon)**n - 1)
    print(f"Steps: {n:7d} | Error: {accumulated_error:.3e}")
```

This is why simulating a galaxy for 10 Gyr requires tolerances of $10^{-12}$ or better!

### Adaptive Error Control

:::{margin}
**Tolerance**: The maximum acceptable error for a calculation. Written as $\text{tol}$ or $\epsilon_{\text{tol}}$, it sets the threshold for when to refine calculations or accept results.
:::

Algorithms adjust parameters to keep errors below tolerance:

$$\boxed{\text{If } E_{\text{estimated}} > \text{tol} \Rightarrow \text{reduce } h \text{ by factor } \sqrt{\text{tol}/E}}$$
$$\boxed{\text{If } E_{\text{estimated}} < \text{tol}/10 \Rightarrow \text{increase } h \text{ by factor 1.5}}$$
$$\boxed{\text{If } \text{tol}/10 \leq E_{\text{estimated}} \leq \text{tol} \Rightarrow \text{maintain } h}$$

```python
def adaptive_h(error_estimate, h_current, tolerance):
    """Adjust step size to maintain target tolerance"""
    if error_estimate > tolerance:
        # Reduce h to bring error under control
        return h_current * 0.5 * np.sqrt(tolerance/error_estimate)
    elif error_estimate < tolerance/10:
        # Safe to increase h for efficiency
        return h_current * 1.5
    else:
        # Error is acceptable
        return h_current
```

### Error Tolerances in Practice

| Application | Typical Tolerance | Physical Justification |
|------------|------------------|------------------------|
| Planetary orbits (< 1 yr) | $10^{-10}$ | Must conserve energy to 10 digits |
| Planetary orbits (Gyr) | $10^{-14}$ | Errors accumulate over $10^{15}$ timesteps |
| Stellar evolution | $10^{-6}$ | Opacity uncertainties dominate |
| Galaxy mergers | $10^{-4}$ | Statistical properties matter, not individual stars |
| MCMC sampling | $10^{-8}$ | Must satisfy detailed balance exactly |
| Gravitational waves | $10^{-20}$ | Strain measurements at detection limit |

### Convergence Testing Without True Values

In research, you rarely know the true answer. Test convergence by comparing results at different resolutions:

$$\text{Richardson extrapolation: } E_{\text{estimated}} = \frac{|f_h - f_{h/2}|}{2^p - 1}$$

where $p$ is the order of your method (2 for central difference).

```python
# Verify your implementation is correct
h_values = [0.1, 0.05, 0.025, 0.0125]
for i in range(len(h_values)-1):
    ratio = (result[i] - result[i+1]) / (result[i+1] - result[i+2])
    print(f"Convergence ratio: {ratio:.2f} (expect {2**p:.1f})")
```

:::{admonition} Common Misconceptions
:class: warning

1. **"Smaller tolerance is always better"** - No! Below machine precision it's meaningless
2. **"Relative error is superior"** - No! It fails near zero
3. **"We can achieve any tolerance"** - No! Limited by $\epsilon_{\text{machine}} \approx 10^{-16}$
:::

:::{admonition} Connection to Your Projects
:class: tip

- **Project 2**: Energy conservation check: $E_{\text{rel}} = |E(t) - E(0)|/|E(0)| < 10^{-10}$
- **Project 3**: Photon conservation: $N_{\text{abs}} = |N_{\text{in}} - N_{\text{out}} - N_{\text{absorbed}}| < \sqrt{N}$
- **Project 4**: MCMC convergence: Gelman-Rubin $\hat{R} < 1.01$
- **Final**: Neural network: Stop when $|\mathcal{L}_{n} - \mathcal{L}_{n-1}|/|\mathcal{L}_{n-1}| < 10^{-6}$
:::

:::{admonition} Check Your Understanding
:class: question

1. Calculate the absolute and relative errors for: true = 3.5 AU, computed = 3.51 AU
2. At what value does relative error become unreliable for distances measured in parsecs?
3. If you need 6 digits of accuracy after $10^9$ steps, what per-step tolerance do you need?
:::

---

## Bridge to Part 3: From Machine Arithmetic to Taylor Series

You now understand the three types of numerical error and how they arise from finite precision arithmetic. But how do we systematically analyze and predict these errors in our algorithms?

In Part 3, we'll explore Taylor series - the mathematical bridge that connects continuous calculus to discrete numerical methods. You'll discover:

- How Taylor expansions reveal the exact error in finite difference formulas
- Why symmetric methods achieve higher accuracy "for free"
- How to derive numerical methods of any order
- When to use finite differences vs. modern automatic differentiation

The error analysis techniques you've just learned will combine with Taylor series to give you a complete toolkit for understanding and controlling numerical accuracy.

*Next: Part 3 - Taylor Series*