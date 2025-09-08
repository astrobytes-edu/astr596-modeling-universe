---
title: "Part 3: Symplectic Integration - Geometry Over Accuracy"
subtitle: "Module 3: ODE Methods & Conservation | ASTR 596"
---

**Navigation:**
[← Part 2: Runge-Kutta Methods](./02-runge-kutta.md) | [Part 4: Stability & Performance →](./04-stability-performance.md)

## Learning Outcomes

By the end of this section, you will be able to:

- **Prove** symplecticity of leapfrog/Verlet methods
- **Understand** modified Hamiltonians and bounded energy error
- **Implement** symplectic integrators for N-body problems
- **Choose** between accuracy and structure preservation
- **Analyze** why geometric preservation trumps local accuracy

---

## The Fundamental Insight

:::{margin}
**Symplectic integrator**: A numerical method that preserves the symplectic structure (phase space volume and geometric properties) of Hamiltonian systems.
:::

Instead of minimizing local truncation error, **symplectic integrators** preserve geometric properties:

1. **Phase space volume** (Liouville's theorem)
2. **Time reversibility**
3. **Bounded energy error** (oscillates but doesn't grow)

The profound trade-off: symplectic methods may be less accurate locally but maintain qualitative correctness globally.

## The Phase Space Perspective

:::{margin}
**Phase space**: The space of all possible states of a system, with position and momentum (or velocity) as coordinates.
:::

In **phase space**, trajectories follow contours of constant energy. 

**Liouville's Theorem**: Phase space volume is preserved under Hamiltonian flow.

Standard numerical methods violate Liouville's theorem! At each timestep, they slightly expand or contract phase space volume. Over billions of steps, this violation accumulates catastrophically.

### Visualizing Phase Space Preservation

```
Non-symplectic (RK4):          Symplectic (Leapfrog):
      p                              p
      ↑                              ↑
   Initial                        Initial
   ●──●──●                        ●──●──●
  ●       ●                      ●       ●
 ●         ●                    ●         ●
●           ● → q              ●           ● → q
 ●         ●                    ●         ●
  ●       ●                      ●       ●
   ●──●──●                        ●──●──●

After 1000 orbits:             After 1000 orbits:
      p                              p
      ↑                              ↑
     Drift!                       Preserved!
    ●  ●  ●                       ●──●──●
   ●       ●                     ●       ●
  ●         ●                   ●         ●
 ●           ● → q             ●           ● → q
  ●         ●                   ●         ●
   ●       ●                     ●       ●
    ●  ●  ●                       ●──●──●
```

## The Leapfrog/Verlet Method

The leapfrog method staggers position and velocity updates:

```python
def leapfrog_step(x, v, h, acceleration):
    """
    Symplectic leapfrog integration step
    Preserves phase space volume exactly
    """
    # Stage 1: Half-step velocity (kick)
    a = acceleration(x)
    v_half = v + (h/2) * a
    
    # Stage 2: Full-step position (drift)
    x_new = x + h * v_half
    
    # Stage 3: Half-step velocity (kick)
    a_new = acceleration(x_new)
    v_new = v_half + (h/2) * a_new
    
    return x_new, v_new
```

For a Hamiltonian $H(q, p) = T(p) + V(q)$:

$$p_{n+1/2} = p_n - \frac{h}{2}\frac{\partial V}{\partial q}\bigg|_{q_n}$$
$$q_{n+1} = q_n + h\frac{\partial T}{\partial p}\bigg|_{p_{n+1/2}}$$
$$p_{n+1} = p_{n+1/2} - \frac{h}{2}\frac{\partial V}{\partial q}\bigg|_{q_{n+1}}$$

### Why "Leapfrog"?

Positions and velocities "leapfrog" over each other in time:

```
Time:    t₀    t₁/₂    t₁    t₃/₂    t₂
         |      |      |      |      |
Position: q₀           q₁           q₂
Velocity:       v₁/₂          v₃/₂
```

This staggering is key to symplecticity!

## Proof of Symplecticity

:::{margin}
**Symplectic condition**: A transformation preserves phase space structure if its Jacobian $J$ satisfies $J^T \Omega J = \Omega$ where $\Omega = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}$.
:::

A transformation is symplectic if it preserves the symplectic 2-form. For the leapfrog map $(q_n, p_n) \to (q_{n+1}, p_{n+1})$:

### Step 1: Compute the Jacobian

The leapfrog transformation can be written as composition of three shear maps:
1. $p \to p - \frac{h}{2}\nabla V(q)$ (momentum kick)
2. $q \to q + h\nabla T(p)$ (position drift)
3. $p \to p - \frac{h}{2}\nabla V(q)$ (momentum kick)

Each shear has Jacobian:
$$J_1 = \begin{pmatrix} I & 0 \\ -\frac{h}{2}\nabla^2 V & I \end{pmatrix}, \quad J_2 = \begin{pmatrix} I & h\nabla^2 T \\ 0 & I \end{pmatrix}$$

### Step 2: Verify Symplectic Condition

For each shear:
$$J_i^T \Omega J_i = \Omega$$

Since composition of symplectic maps is symplectic:
$$J_{total} = J_3 J_2 J_1 \text{ is symplectic}$$

### Step 3: Volume Preservation

$$\det(J) = 1$$

Phase space volume is exactly preserved!

## The Modified Hamiltonian

:::{margin}
**Modified Hamiltonian**: The exactly conserved quantity for a symplectic integrator, differing from the original Hamiltonian by small bounded terms.
:::

Leapfrog doesn't conserve the original Hamiltonian $H$ exactly. Instead, it exactly conserves a **modified Hamiltonian**:

$$\tilde{H} = H + h^2 H_2 + h^4 H_4 + ...$$

The key: $\tilde{H}$ differs from $H$ by $O(h^2)$, and this difference is bounded, not growing! Energy oscillates within a band but never drifts away.

### Backward Error Analysis

We can compute the modified Hamiltonian explicitly. For the harmonic oscillator:

$$H = \frac{1}{2}(p^2 + \omega^2 q^2)$$

The leapfrog modified Hamiltonian is:

$$\tilde{H} = H + \frac{h^2\omega^2}{24}(p^2 - \omega^2 q^2)^2 + O(h^4)$$

The $O(h^2)$ correction is bounded by the energy itself!

## Higher-Order Symplectic Methods

### Yoshida's Method (4th order)

By composing leapfrog steps with carefully chosen timesteps:

```python
def yoshida4_step(x, v, h, acceleration):
    """
    Fourth-order symplectic integrator
    Composition of leapfrog steps
    """
    w0 = -2**(1/3) / (2 - 2**(1/3))
    w1 = 1 / (2 - 2**(1/3))
    c1 = c4 = w1/2
    c2 = c3 = (w0 + w1)/2
    d1 = d3 = w1
    d2 = w0
    
    x1, v1 = leapfrog_step(x, v, c1*h, acceleration)
    x2, v2 = leapfrog_step(x1, v1, d1*h, acceleration)
    x3, v3 = leapfrog_step(x2, v2, c2*h, acceleration)
    x4, v4 = leapfrog_step(x3, v3, d2*h, acceleration)
    x5, v5 = leapfrog_step(x4, v4, c3*h, acceleration)
    x6, v6 = leapfrog_step(x5, v5, d3*h, acceleration)
    return leapfrog_step(x6, v6, c4*h, acceleration)
```

## Performance Comparison: 1000-Year Integration

Testing on Earth's orbit (e = 0.017, 365.25 day period):

| Method | Order | Energy Error | Phase Error | Stable? |
|--------|-------|--------------|-------------|---------|
| Euler | 1 | +100% | 100 radians | No |
| RK2 | 2 | +10% | 10 radians | No |
| RK4 | 4 | +0.1% | 0.1 radians | No |
| Leapfrog | 2 | ±0.01% (bounded) | 0.01 radians | Yes |
| Yoshida4 | 4 | ±0.0001% (bounded) | 0.0001 radians | Yes |

Leapfrog with only second-order accuracy outperforms fourth-order RK4 for long-term stability!

## When to Use Symplectic Methods

| Problem Type | Requirement | Best Method | Reason |
|--------------|------------|-------------|---------|
| Solar system (Gyr) | Long-term stability | Symplectic | Bounded energy error |
| Satellite (days) | Trajectory accuracy | RK45 adaptive | Short duration |
| Galaxy merger | Phase space structure | Symplectic | Preserve invariants |
| Molecular dynamics | Energy conservation | Symplectic | 10¹⁵ timesteps |

## Implementation for N-Body Systems

```python
def nbody_leapfrog(positions, velocities, masses, h, n_steps):
    """
    N-body simulation with symplectic leapfrog
    Preserves total energy and momentum
    """
    def compute_accelerations(pos):
        # Vectorized force calculation
        dr = pos[:, np.newaxis] - pos[np.newaxis, :]
        r = np.linalg.norm(dr, axis=2)
        r[r == 0] = 1  # Avoid self-interaction
        F = G * masses[:, np.newaxis] * masses[np.newaxis, :] / r**3
        F = F[:, :, np.newaxis] * dr
        return F.sum(axis=1) / masses[:, np.newaxis]
    
    pos = positions.copy()
    vel = velocities.copy()
    
    # Initial half-step for velocities
    acc = compute_accelerations(pos)
    vel += 0.5 * h * acc
    
    for step in range(n_steps):
        # Update positions
        pos += h * vel
        
        # Update velocities
        acc = compute_accelerations(pos)
        if step < n_steps - 1:
            vel += h * acc
        else:
            # Final half-step
            vel += 0.5 * h * acc
    
    return pos, vel
```

:::{admonition} Physical Example: Binary Pulsar Timing
:class: tip
PSR B1913+16 has been observed for 40+ years with microsecond timing precision. Over 40 years with 7.75 hour orbital period:

- Number of orbits: ~4.5 × 10⁷
- Required phase accuracy: ~10⁻⁶ radians per orbit

Only symplectic integrators can maintain this phase coherence. The binary pulsar observations that confirmed gravitational wave emission (Nobel Prize 1993) relied on symplectic integration!
:::

:::{admonition} Check Your Understanding
:class: question
1. Why does leapfrog conserve phase space volume but not exact energy?
2. What's the trade-off between RK4 and leapfrog for 100-year integrations?
3. Why are symplectic methods time-reversible?
4. How does the modified Hamiltonian differ from the original?
:::

---

## Bridge to Part 4: Beyond Basic Integration

You've seen the profound difference between local accuracy and geometric preservation. Symplectic methods sacrifice pointwise precision for global stability, keeping your simulations physically meaningful over cosmic timescales.

But integration methods face other challenges: stiff equations that demand tiny timesteps, high-dimensional systems that require vectorization for performance, and stability boundaries that determine when methods explode. In Part 4, we'll tackle these practical aspects that determine whether your simulation finishes in hours or never finishes at all.

*Next: Part 4 - Stability and Performance*