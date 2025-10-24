---
title: "Part 3: Markov Chain Monte Carlo — From Theory to Practice"
subtitle: "How We Formalize Intuition into Rigorous Inference | Inferential Thinking Module 5 | ASTR 596"
---

> *"It is a capital mistake to theorize before one has data. Insensibly one begins to twist facts to suit theories, instead of theories to suit facts."*  
> — Arthur Conan Doyle, *A Scandal in Bohemia*
>
> *"The purpose of computing is insight, not numbers."*  
> — Richard Hamming

---

## Learning Outcomes

By the end of Part 3, you will be able to:

1. **Explain** why direct evaluation of posteriors fails in high dimensions (curse of dimensionality)
2. **Derive** the Metropolis-Hastings acceptance criterion from detailed balance
3. **Recognize** the connection between MCMC sampling and ergodic exploration from Module 1
4. **Implement** a Markov chain that provably converges to any target distribution
5. **Diagnose** convergence using trace plots, autocorrelation, and the Gelman-Rubin statistic
6. **Apply** MCMC to real astronomical inference problems (Cepheid distance moduli)

**Prerequisites**: Parts 1-2 of this module (measurement philosophy, Bayes' theorem), Module 1 (sampling distributions, CLT, ergodicity), basic Python.

---

## Roadmap: Where We're Going

This is a long module because we're building something profound: a universal inference engine that works for any scientific model, any data, any dimensionality. Here's the journey:

```mermaid
graph TD
    A[Why can't we just<br/>evaluate p(θ|D) everywhere?] --> B[The Curse of<br/>Dimensionality]
    B --> C[Monte Carlo:<br/>Sample instead of integrate]
    C --> D[But how do we sample<br/>from unknown distributions?]
    D --> E[Build a Markov chain<br/>that explores the posterior]
    E --> F[Metropolis-Hastings:<br/>The universal sampler]
    F --> G[Practical diagnostics:<br/>Has it converged?]
    G --> H[Real example:<br/>Cepheid distances]
    
    style A fill:#f9f,stroke:#333
    style F fill:#9f9,stroke:#333
    style H fill:#99f,stroke:#333
```

**The narrative arc**: We'll start with a crisis (can't evaluate high-dimensional integrals), discover a profound solution (sampling via Markov chains), derive the mathematical machinery (detailed balance), build the algorithm (Metropolis-Hastings), learn how to use it (diagnostics), and apply it to real science (Cepheid standard candles preparing for Type Ia SNe).

This isn't just a numerical method. It's a fundamentally different way of thinking about inference that emerged from mid-20th century physics and revolutionized statistics, machine learning, and computational science.

---

## Part 1: The Computational Crisis — Why We Need MCMC

### The Posterior is a Distribution, Not a Number

In Part 2, you learned Bayes' theorem:

$$
p(\theta | D) = \frac{p(D | \theta) \, p(\theta)}{p(D)}
$$

where

- $\theta$ are the model parameters *(what we want to infer)*
- $D$ is the observed data *(what we have)*
- $p(D | \theta)$ is the likelihood *(how probable the data is given parameters)*
- $p(\theta)$ is the prior (*(our beliefs -- what we know about the parameters)*
- $p(\theta | D)$ is the posterior *(what we want)*
- $p(D) = \int p(D|θ)p(θ)dθ$ is the evidence *(normalization constant)*

The posterior $p(θ|D) = \pi(\theta)$ is a **probability distribution** over the parameter space. It tells you, for every possible value of $θ$, how probable that value is given the data you observed.

:::{tip} Unnormalized Posterior is fine
In MCMC we only need the target distribution **up to a constant**. For Bayes,
\[
\pi(\theta)\ \propto\ p(D\mid\theta)\,p(\theta),
\]
because the evidence \(p(D)\) does **not** depend on \(\theta\). In Metropolis–Hastings, \(p(D)\) cancels from the acceptance ratio, so we never evaluate it.
:::

Let's be concrete. Suppose you're measuring the distance to a Cepheid variable star. Your parameter is $θ = d$ (distance in parsecs). You have:

- **Data D**: Observed apparent magnitude $m = 15.2 ± 0.1$ mag
- **Model**: Absolute magnitude $M = -4.5$ (from period-luminosity relation)
- **Likelihood**: Gaussian, $p(m|d) = N(m_\text{theory}(d), σ²)$ where $m_\text{theory(d)} = M + 5\log_{10}(d) - 5$

The posterior $p(d|m)$ tells you not just the "best" distance, but the **full probability distribution** over possible distances. This captures:

- Your uncertainty (how confident are you?)
- Asymmetries (is the uncertainty symmetric around the peak?)
- Correlations (if you had multiple parameters, how do they trade off?)

In low dimensions (1-2 parameters), you could just evaluate $p(d|m)$ on a grid of distance values and plot it. Done! But what if you have 10 parameters? Or 100?

### The Curse of Dimensionality

Imagine you want to map out a posterior distribution by evaluating it on a grid. How many grid points do you need?

**1D example**: To resolve a 1D posterior to reasonable accuracy, you might need $N = 100$ grid points.

**2D example**: Now you have two parameters. You need $100 × 100 = 10^4$ evaluations.

**3D example**: Three parameters → $100 × 100 × 100 = 10^6$ evaluations.

**Scaling**: For $d$ dimensions, you need $N^d$ evaluations. This is **exponential scaling** — the curse of dimensionality.

Let's make this concrete with realistic numbers:

```{list-table} The Dimensionality Curse Gets Worse Fast
:header-rows: 1
:name: curse-scaling

* - Dimensions
  - Grid points per dimension
  - Total evaluations
  - Time (assuming 1 ms per eval)
* - 1
  - 100
  - 10²
  - 0.1 seconds
* - 2
  - 100
  - 10⁴
  - 10 seconds
* - 3
  - 100
  - 10⁶
  - 16 minutes
* - 5
  - 100
  - 10¹⁰
  - 115 days
* - 10
  - 100
  - 10²⁰
  - 3 trillion years
* - 20
  - 100
  - 10⁴⁰
  - Age of universe × 10²³
```

Even "small" problems with 5-10 parameters become computationally impossible. And many real scientific problems have hundreds or thousands of parameters:

- Climate models: ~100 parameters (ocean circulation, cloud physics, aerosols...)
- Exoplanet characterization: ~20 parameters per planet (mass, orbit, atmosphere...)
- Galaxy formation: ~50 parameters (star formation efficiency, black hole feedback...)
- Neural networks: millions to billions of parameters

**Grid evaluation simply cannot work.**

:::{admonition} Connection to Module 1: Volume of High-Dimensional Spheres
:class: note
Remember the bizarre geometry of high dimensions from Module 1? The volume of a d-dimensional unit sphere is:

$$
V_d = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}
$$

This **increases** with dimension up to d = 5, then **decreases** exponentially! By d = 20, essentially all the volume is in the thin shell near the surface.

Most of your grid points are wasted evaluating regions of negligible probability. The probability mass lives in a tiny, weirdly-shaped region of parameter space that you can't find by uniform sampling.

This is why high-dimensional inference is fundamentally different from low-dimensional intuition.
:::

### What About Optimization? Just Find the Peak

:::{margin}
**maximum a posteriori (MAP) estimation**: Finding the parameter values that maximize the posterior distribution. Requires optimization techniques.

**uncertainty quantification**: The process of determining the degree of uncertainty in parameter estimates, often through credible intervals or posterior distributions.

**degeneracies**: Situations where different combinations of parameters produce similar model outputs, leading to correlated uncertainties.

**evidence**: The normalization constant in Bayes' theorem, representing the total probability of the data under all possible parameter values.
:::

You might think: "Why do I need the full distribution? Just find the maximum of $p(θ|D)$ and report that!"

This is **maximum a posteriori (MAP) estimation**, and it's sometimes useful. But you lose crucial information:

**1. Uncertainty quantification**: The peak tells you nothing about how uncertain you are. Is it a sharp peak (confident) or broad (uncertain)? If you report $θ = 5.0$ as the MAP estimate, can you distinguish whether your uncertainty is $±0.1$ or $±10$? The peak doesn't tell you.

**2. Asymmetries**: The peak might be at $d = 500$ pc, but the distribution could be skewed — maybe $500 ± 20$ pc toward smaller distances, but $+100$ pc toward larger distances (imagine a long tail from dust extinction uncertainty).

**3. Correlations**: With multiple parameters, knowing the peak doesn't tell you about **degeneracies**. Maybe radius and temperature are correlated — high R and low T give the same luminosity as low R and high T.

**4. Model comparison**: The **evidence** $p(D)$ (the denominator in Bayes' theorem) requires integrating over all parameter values. The peak tells you nothing about this.

**5. Propagating uncertainty**: If you use your inferred parameters to predict something else, you need the full distribution to get realistic error bars.

**The scientific standard**: Modern quantitative science doesn't report "best fit" values. It reports **credible intervals** from the full posterior distribution. This is what p-values tried to do (badly). Bayesian posteriors do it right.

:::{admonition} The More You Know: Why p-values are broken
:class: dropdown
Classical hypothesis testing reports p-values: "If the null hypothesis were true, the probability of seeing data this extreme is p = 0.03."

This is a probability statement about the data given the hypothesis, not about the hypothesis given the data! It's the wrong direction.

What you actually want: "Given the data I observed, what's the probability the hypothesis is true?" That's p(H|D) — the posterior!

This confusion has led to the replication crisis across sciences. Bayesian inference solves this by giving you what you actually want: probabilities of hypotheses given data.
:::

### The Monte Carlo Solution: Sample, Don't Integrate

:::{margin}
**Monte Carlo integration**: A numerical method that estimates integrals by averaging function values at random samples drawn from a probability distribution.

**Sampling distribution**: The probability distribution of a statistic (like the mean) computed from random samples of data.

**Central Limit Theorem (CLT)**: A statistical theorem stating that the distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the original distribution of the data.

**Standard error**: The standard deviation of a sampling distribution, representing the uncertainty in an estimate due to finite sample size.

**Variance reduction techniques**: Methods used in Monte Carlo integration to decrease the variance of estimates, improving accuracy without increasing the number of samples.
:::

Here's a profound insight from the early days of computing (1940s-1950s): **You don't need to evaluate the posterior everywhere. You need to sample from it.**

If you have N samples ${θ₁, θ₂, ..., θ_N}$ drawn from $p(θ|D)$, then you can estimate anything:

**Mean**:
$$\langle \theta \rangle \approx \frac{1}{N} \sum_{i=1}^N \theta_i$$

**Variance**:
$$\text{Var}(\theta) \approx \frac{1}{N} \sum_{i=1}^N (\theta_i - \langle \theta \rangle)^2$$

**Credible intervals**: Sort the samples and find the 16th and 84th percentiles for a 68% credible interval.

**Predictions**: For any function f(θ), estimate its expectation:
$$\langle f(\theta) \rangle \approx \frac{1}{N} \sum_{i=1}^N f(\theta_i)$$

**The magic**: The accuracy of Monte Carlo estimates scales as **$\sigma/\sqrt{N}$**, where $\sigma$ is the **standard deviation**. The **variance** of Monte Carlo estimates scales as $1/\sqrt{N}$ regardless of dimensionality — this follows directly from the **Central Limit Theorem** you studied in Module 1.

:::{admonition} Important: Monte Carlo Scaling with Dimension
:class: note
The variance of Monte Carlo estimates does scale as $\sigma/\sqrt{N}$ regardless of dimension. However, the constant $\sigma$ and the effective sample size needed to adequately explore the distribution may depend on dimension due to geometric concentration phenomena. MCMC helps mitigate this through **targeted exploration** — instead of uniform sampling everywhere, we spend time where the posterior has significant probability mass.
:::

To get one more decimal place of accuracy, you need $100×$ more samples. But crucially: **this doesn't depend on dimensionality**! Going from 10 to 100 dimensions doesn't change this. It's the same $\sqrt{N}$ convergence.

This is revolutionary. Grid evaluation scales exponentially with dimension $(N^d)$. Monte Carlo scales with accuracy $(1/\sqrt{N})$, independent of dimension.

:::{admonition} Connection to Module 1: Law of Large Numbers and CLT
:class: note
Everything we're doing rests on the Central Limit Theorem from Module 1:

If ${\theta_i}$ are independent samples from any distribution with finite variance, then:

$$\frac{1}{N} \sum_{i=1}^N f(\theta_i) \to \mathbb{E}[f(\theta)] \quad \text{as } N \to \infty$$

And the error in this estimate is approximately:

$$\frac{\sigma_f}{\sqrt{N}}$$

where $\sigma_f$ is the standard deviation of $f(θ)$.

This is why sampling works: Given enough samples, sample averages converge to true expectations. It's the same foundation that underlies all of statistical inference.
:::

**But here's the catch**: How do you draw samples from $p(θ|D)$ when you don't know what $p(θ|D)$ looks like? You can evaluate it at any point (numerator of Bayes' theorem), but sampling requires knowing the shape.

This is where **Markov Chain Monte Carlo** enters. It's a method to generate samples from any target distribution, no matter how complex, by constructing a Markov chain that explores parameter space.

---

## Part 2: Markov Chains — The Mathematical Foundation

### The Core Idea: Build a Stochastic Process

Instead of trying to sample directly from $p(θ|D)$, we'll build a **stochastic process** (a Markov chain) that:

1. Starts from any initial $θ₀$
2. Takes random steps through parameter space
3. Gradually "forgets" where it started
4. Eventually produces samples distributed according to $p(θ|D)$

Think of it like a random walker wandering around parameter space. We'll design the rules of walking so that after sufficient wandering, the walker spends time in each region proportional to $p(θ|D)$.

If $p(θ|D)$ is high in some region, the walker visits often. If $p(θ|D)$ is low, the walker rarely goes there. **The stationary distribution of the random walk is the posterior.**

### Markov Chains: Formal Definition

:::{margin}
**stochastic process**: A collection of random variables indexed by time or space, representing a system that evolves randomly over time.
**Markov property**: The property of a stochastic process where the future state depends only on the present state, not on the sequence of events that preceded it.
**transition kernel**: A function that defines the probabilities of moving from one state to another in a Markov chain.
**stationary distribution**: A probability distribution that remains unchanged as the system evolves over time in a Markov chain.
**ergodicity**: A property of a Markov chain where it is possible to reach any state from any other state, ensuring that long-term averages converge to expected values.
:::

A **Markov chain** is a sequence of random variables $\{θ₀, θ₁, θ₂, ...\}$ where the probability of the next state depends only on the current state:

$$
p(\theta_{t+1} | \theta_t, \theta_{t-1}, \theta_{t-2}, ..., \theta_0) = p(\theta_{t+1} | \theta_t)
$$

This is the **Markov property**: The future depends on the present, not the past. The chain has "no memory" beyond its current location.

```{admonition} Physical Intuition: Random Walk
:class: tip
Imagine a drunk walker on a street. At each time step, they randomly choose to take a step left or right. Where they go next depends only on where they are now, not how they got there. That's the Markov property.

In parameter space, $θ$ is the walker's current location. The transition kernel tells you the probability of moving to each possible new location $θ'$.
```

### Transition Kernels

:::{margin}
**transition kernel**: A function that defines the probabilities of moving from one state to another in

**kernel**: In mathematics, a function used to define transformations or operations, often in the context of integral equations or machine learning algorithms.
:::

The dynamics of a Markov chain are defined by the **transition kernel** (or transition probability):

$$
T(\theta' | \theta)
$$

This is the probability density of moving to $θ'$ given that you're currently at $θ$. It must satisfy:

$$
\int T(\theta' | \theta) \, d\theta' = 1
$$

for all $θ$ (probabilities must sum to 1).

Given a starting distribution $p₀(θ)$, the distribution after one step is:

$$
p_1(\theta') = \int p_0(\theta) T(\theta' | \theta) \, d\theta
$$

After $t$ steps:

$$
p_t(\theta') = \int p_{t-1}(\theta) T(\theta' | \theta) \, d\theta
$$

**Question**: Can we design $T$ such that p_t(θ) converges to our target distribution π(θ) = p(θ|D) as t → ∞?

**Answer**: Yes! But we need two conditions:

   1. **Stationarity**: The target distribution $π$ must be stationary with respect to T.
   2. **Ergodicity**: The chain must be able to explore the entire parameter space.

### Stationarity: The Distribution Must Be Preserved

A distribution $π(θ)$ is **stationary** with respect to $T$ if applying the transition kernel doesn't change it:

$$
\int \pi(\theta) T(\theta' | \theta) \, d\theta = \pi(\theta')
$$

*What it is:* If the markov chain is distributed according to $π$, then after one more step, it's still distributed according to $π$. The distribution is an equilibrium of the dynamics.

*Physical analogy:* A gas in thermal equilibrium. Molecules are constantly moving and colliding, but the overall distribution (Boltzmann distribution) doesn't change. Individual states change, but the probability distribution is stationary.

### Ergodicity: The Chain Must Explore Everywhere

Stationarity means $π$ is preserved *if you start from it*. But how do we get there from an arbitrary starting point $θ₀$?

We need **ergodicity**: The chain must be able to reach any state from any other state in finite time. Formally, ergodicity requires:

**1. Irreducibility**: Every state is accessible from every other state (there are no isolated regions).

**2. Aperiodicity**: The chain doesn't get stuck in cycles. (E.g., don't alternate deterministically: left, right, left, right forever.)

:::{admonition} Connection to Module 3: Phase Space and Ergodicity
:class: note
In Module 3 (statistical mechanics), you learned the **ergodic hypothesis**: Time averages equal ensemble averages. A system exploring its phase space eventually visits all accessible microstates with frequency proportional to their statistical weight.

The same principle applies here! Your Markov chain explores parameter space, eventually visiting each region with frequency proportional to π(θ) = p(θ|D).

- Module 3: Ergodicity lets us replace ensemble averages (over all possible configurations) with time averages (following one trajectory).
- Module 5: Ergodicity lets us replace ensemble sampling (drawing independent samples from $π$) with sequential sampling (following one Markov chain).

It's the same concept applied to probability distributions instead of physical systems.
:::

**The Ergodic Theorem**: If a Markov chain is ergodic and has stationary distribution π, then for any function f(θ):

$$
\lim_{N \to \infty} \frac{1}{N} \sum_{t=1}^N f(\theta_t) = \int f(\theta) \pi(\theta) \, d\theta
$$

Time averages along the chain converge to expectations under $π$. **This is how MCMC works**: Run the chain for a long time, then use the samples to estimate expectations.

### Detailed Balance: A Sufficient Condition for Stationarity

We need to design a transition kernel T that has π as its stationary distribution. There's a powerful sufficient condition called **detailed balance** (also called **microscopic reversibility**):

$$
\pi(\theta) T(\theta' | \theta) = \pi(\theta') T(\theta | \theta')
$$

In words: The probability flow from θ to θ' equals the flow from θ' back to θ when both are drawn from π.

**Why does detailed balance imply stationarity?** Here's the key insight:

Detailed balance is a **sufficient condition** for stationarity. To see why, assume detailed balance holds and integrate both sides over θ:

$$
\int \pi(\theta) T(\theta' | \theta) \, d\theta = \int \pi(\theta') T(\theta | \theta') \, d\theta
$$

The right side factors since π(θ') doesn't depend on the integration variable:

$$
\text{RHS} = \pi(\theta') \int T(\theta | \theta') \, d\theta = \pi(\theta') \times 1 = \pi(\theta')
$$

(The integral equals 1 because T is a probability density in its first argument.)

The left side is **exactly the definition of how the distribution evolves forward one step**:

$$
\text{LHS} = \int \pi(\theta) T(\theta' | \theta) \, d\theta
$$

This integral gives you the probability at θ' after one transition starting from π. Since LHS = RHS = π(θ'), we've shown that starting from distribution π produces distribution π after one step. **The distribution is stationary.**

:::{admonition} Physical Intuition: Equilibrium Flow
:class: tip
Imagine a fluid flowing between two containers. Detailed balance says: At equilibrium, the flow rate from container A to B equals the flow rate from B to A.

If inflow equals outflow for every pair of states, then the overall distribution doesn't change. That's stationarity.

In statistical mechanics, this is called **microscopic reversibility**: At thermal equilibrium, every microscopic process and its reverse occur at equal rates. This is a fundamental principle of thermodynamics that MCMC inherits.
:::

<!--
SUGGESTED FIGURE: Detailed Balance Flow Diagram
Create this visualization using Python/Matplotlib:

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Draw two states as circles
circle1 = plt.Circle((2, 3), 0.8, color='lightblue', ec='black', linewidth=2)
circle2 = plt.Circle((8, 3), 0.8, color='lightcoral', ec='black', linewidth=2)
ax.add_patch(circle1)
ax.add_patch(circle2)

# Add state labels
ax.text(2, 3, r'$\theta$', fontsize=20, ha='center', va='center', fontweight='bold')
ax.text(8, 3, r"$\theta'$", fontsize=20, ha='center', va='center', fontweight='bold')

# Add probability labels below circles
ax.text(2, 1.8, r'$\pi(\theta)$', fontsize=14, ha='center', color='blue')
ax.text(8, 1.8, r"$\pi(\theta')$", fontsize=14, ha='center', color='red')

# Draw forward arrow (top)
arrow1 = FancyArrowPatch((3, 3.3), (7, 3.3), 
                         arrowstyle='->', mutation_scale=30, 
                         linewidth=2.5, color='darkblue')
ax.add_patch(arrow1)
ax.text(5, 3.8, r'$\pi(\theta) T(\theta\'|\theta)$', fontsize=12, ha='center', 
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Draw reverse arrow (bottom)
arrow2 = FancyArrowPatch((7, 2.7), (3, 2.7), 
                         arrowstyle='->', mutation_scale=30, 
                         linewidth=2.5, color='darkred')
ax.add_patch(arrow2)
ax.text(5, 2.2, r'$\pi(\theta\') T(\theta|\theta\')$', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Add equilibrium condition
ax.text(5, 5, 'Detailed Balance: Forward Flow = Reverse Flow', 
        fontsize=14, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))

ax.set_xlim(0, 10)
ax.set_ylim(1, 6)
ax.axis('off')
ax.set_aspect('equal')

plt.title('Detailed Balance: Equilibrium Flow Between States', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('detailed_balance_flow.png', dpi=300, bbox_inches='tight')
plt.show()

This creates a clear visual showing two states with bidirectional flow, making the
abstract detailed balance condition concrete and intuitive.
-->


```{admonition} Connection to Module 3: Statistical Mechanics
:class: note
The detailed balance equation is **identical** to the condition for thermal equilibrium in statistical mechanics:

$$
\rho_i W_{i \to j} = \rho_j W_{j \to i}
$$

where ρ_i is the probability of microstate i and W_{i→j} is the transition rate from i to j.

For a Hamiltonian system in thermal equilibrium at temperature T, the Boltzmann distribution:

$$
\rho_i \propto e^{-E_i / k_B T}
$$

satisfies detailed balance with respect to the dynamics of the system. This is why thermalization works—microscopic reversibility guarantees that systems evolve toward the Boltzmann distribution.

MCMC uses the **same mathematical structure** to guarantee that your sampler evolves toward the posterior distribution p(θ|D). It's not mere analogy—they're the same theorem applied to different systems.
```

Now we have our blueprint:

1. Choose a target distribution $π(θ) = p(θ|D)$.
2. Design a transition kernel $T$ that satisfies detailed balance with respect to $π$
3. The Markov chain will converge to $π$ (if ergodic)
4. Run the chain, collect samples, estimate expectations

The question: How do we actually construct such a $T$? That's where **Metropolis-Hastings** comes in.

---

## Part 3: The Metropolis-Hastings Algorithm

```{admonition} Where We Are
:class: note
**The journey so far**: In Part 1, we discovered that grid evaluation fails exponentially with dimension—the curse of dimensionality. In Part 2, we learned that Markov chains offer a solution: build a random walk that explores parameter space and eventually produces samples from any target distribution, guaranteed by ergodicity and detailed balance. 

**Now in Part 3**: We'll construct the specific algorithm that makes this work—the Metropolis-Hastings sampler that can sample from ANY distribution, no matter how complex.
```

### The Construction: Proposal + Accept/Reject

Here's the brilliant idea due to Metropolis et al. (1953) and generalized by Hastings (1970):

:::{admonition} The More You Know: The Manhattan Project Origins of MCMC
:class: dropdown
The Metropolis algorithm has remarkable origins. In the early 1950s, Nicholas Metropolis, Arianna Rosenbluth, Marshall Rosenbluth, Augusta Teller, and Edward Teller were working at Los Alamos National Laboratory on the hydrogen bomb project. They needed to understand how atoms in materials would behave under extreme conditions—millions of degrees, enormous pressures—but they couldn't build physical experiments to measure this (too dangerous and expensive). They also couldn't solve the equations analytically (too many particles, too complex).

Their insight: Simulate the particles computationally. But even simulation was a challenge—how do you sample from the Boltzmann distribution at high temperature without computing the partition function (which requires summing over all possible states)? Rosenbluth had the key idea: Use a Markov chain with accept/reject steps based on energy differences. The normalization constant cancels out!

The first calculation ran on MANIAC I (Mathematical Analyzer Numerical Integrator and Computer), one of the earliest digital computers. The problem: 224 particles interacting via Lennard-Jones potential. The computation took hours. Today your laptop could do it in seconds.

**The profound irony**: A method developed to understand nuclear weapons physics has become one of the most important tools in peaceful science—from astronomy to medicine to machine learning. Computational methods developed for destruction now help us understand the universe.

The original 1953 paper is a model of clarity and is still worth reading. Metropolis's name appears first, but by alphabetical convention—the method was really a collaborative effort. Augusta Teller (Edward Teller's wife) and Arianna Rosenbluth are sometimes forgotten in historical accounts, yet another example of how women's contributions to computational science have been underrecognized.
:::

**Separate the transition kernel into two parts**:

1. **Proposal distribution $Q(θ'|θ)$**: Generates a candidate new state θ' given current state θ
2. **Acceptance probability $α(θ'|θ)$**: Probability of accepting the proposal

The full transition kernel is:

$$
T(\theta' | \theta) = Q(\theta' | \theta) \, \alpha(\theta' | \theta)
$$

**The algorithm**:

1. Start at some $θ₀$ (anywhere in parameter space)
2. For $t = 0, 1, 2, ..., N-1$:
   - **Propose**: Draw $θ'$ from $Q(θ'|θ_t)$
   - **Evaluate**: Compute acceptance probability $α(θ'|θ_t)$
   - **Decide**: 
     - With probability $α$, accept: $θ_{t+1} = θ'$
     - With probability $1-α$, reject: $θ_{t+1} = θ_t$ (stay put)
3. Return samples ${θ₁, θ₂, ..., θ_N}$

**The magic is in step 2**: How do we choose α to satisfy detailed balance?

:::{admonition} Pause and Predict
:class: tip
**Before we derive the acceptance probability, take a moment to think:**

Given that we want detailed balance $π(θ)T(θ'|θ) = π(θ')T(θ|θ')$, and we've decided $T = Q × α$, what constraints must $α$ satisfy?

Write down your answer before reading on:

- Should $α$ depend on both $θ$ and $θ'$, or just one?
- If $θ'$ has higher posterior probability than θ, should we always accept?
- If $θ'$ has lower posterior probability, should we always reject?
- What role does the proposal distribution $Q$ play?

Think about it for 30 seconds, then continue. The derivation will be more meaningful if you've wrestled with the problem first.
:::

### Deriving the Acceptance Probability

We want:

$$
\pi(\theta) T(\theta' | \theta) = \pi(\theta') T(\theta | \theta')
$$

Substituting $T = Qα$ where $Q$ is the proposal and $α$ is the acceptance probability:

$$
\pi(\theta) Q(\theta' | \theta) \alpha(\theta' | \theta) = \pi(\theta') Q(\theta | \theta') \alpha(\theta | \theta')
$$

Rearrange:

$$
\frac{\alpha(\theta' | \theta)}{\alpha(\theta | \theta')} = \frac{\pi(\theta') Q(\theta | \theta')}{\pi(\theta) Q(\theta' | \theta)}
$$

Define the **acceptance ratio**:

$$
r = \frac{\pi(\theta') Q(\theta | \theta')}{\pi(\theta) Q(\theta' | \theta)}
$$

**When $\pi$ is the posterior:** write 
$$\pi(\theta) \propto p(D\mid\theta)p(\theta).$$

Then

$$
r \;=\; \frac{\pi(\theta')\,Q(\theta\mid\theta')}{\pi(\theta)\,Q(\theta'\mid\theta)}
\;=\; \frac{p(D\mid\theta')\,p(\theta')}{p(D\mid\theta)\,p(\theta)} \cdot \frac{Q(\theta\mid\theta')}{Q(\theta'\mid\theta)},
$$

so the evidence $p(D)$ cancels because it is constant in $\theta$.

We need $α(θ'|θ)$ and $α(θ|θ')$ to have ratio $r$. A simple choice that works:

$$
\alpha(\theta' | \theta) = \min(1, r) = \min\left(1, \frac{\pi(\theta') Q(\theta | \theta')}{\pi(\theta) Q(\theta' | \theta)}\right)
$$

**Verification**: Let

$$R = \frac{π(θ')Q(θ|θ')}{π(θ)Q(θ'|θ)}$$

- If $R ≥ 1: α(θ'|θ) = 1, α(θ|θ') = 1/R ≤ 1$. Ratio: $1/(1/R) = R$ ✓
- If $R < 1: α(θ'|θ) = R, α(θ|θ') = 1$. Ratio: $R/1 = R$ ✓

Either way, the ratio condition is satisfied, so detailed balance holds!

**Interpretation**:

- If $θ'$ has higher posterior probability than $θ~(R > 1)$, always accept.
- If $θ'$ has lower posterior probability $(R < 1)$, accept with probability $R$.

This means uphill moves are always accepted, downhill moves are sometimes accepted. The chain preferentially explores high-probability regions but can escape local optima by occasionally accepting downward moves.

:::{admonition} The More You Know: Why "$\min(1, r)$" Works
:class: dropdown

You might wonder: Why specifically $\min(1, r)$? Could we use other acceptance probabilities?

Yes! Any choice of $α$ that satisfies the ratio condition works. But $\min(1, r)$ is **optimal** in a precise sense: It maximizes the acceptance rate while satisfying detailed balance.

Proof sketch: If we used $α(θ'|θ) = r/(1+r)$, we'd accept less often but still satisfy detailed balance. However, lower acceptance means slower exploration and higher autocorrelation. The Metropolis-Hastings choice min(1, r) accepts as often as possible while maintaining detailed balance.

This is an example of the "Barker acceptance" vs. "Metropolis acceptance" distinction. Metropolis is provably more efficient.
:::

### Proposal Distributions: Symmetric vs. Asymmetric

The acceptance probability depends on the proposal distribution $Q$. Two important cases:

**Symmetric proposals**: $Q(θ'|θ) = Q(θ|θ')$

**Example:** Gaussian random walk:
$$Q(\theta' | \theta) = \mathcal{N}(\theta', \theta, \Sigma)$$

Propose by adding Gaussian noise: $θ' = θ + ε \text{ where } ε \sim N(0, Σ)$. Here $\mathcal{N}(0, Σ)$ is the normal (Gaussian) distribution with mean 0 and covariance $\Sigma$.

**Symmetric proposals**: $Q(\theta'|\theta) = Q(\theta|\theta')$.

*Example — Gaussian random walk (symmetric in $θ$):*
$$
\theta' \sim \mathcal{N}(\theta;\,\Sigma)\quad\Longleftrightarrow\quad \theta'=\theta+\varepsilon,\;\varepsilon\sim\mathcal{N}(0,\Sigma).
$$

For symmetric $Q$, the $Q$-terms cancel and
$$
\alpha=\min\!\left(1,\frac{\pi(\theta')}{\pi(\theta)}\right).
$$

Here you can evaluate $π$ using only log-likelihood + log-prior; any constant normalizer cancels. This is the **Metropolis algorithm** (the original 1953 version). **You only need to evaluate the ratio of posterior probabilities!**

**Asymmetric proposals**: $Q(θ'|θ) ≠ Q(θ|θ')$

**Example:** Propose $θ'$ by drawing from the prior $p(θ)$.

Then:
$$\alpha = \min\left(1, \frac{\pi(\theta') Q(\theta | \theta')}{\pi(\theta) Q(\theta' | \theta)}\right)$$

You need the full ratio including the $Q$ terms. This is the **Metropolis-Hastings algorithm** (Hastings 1970 generalization).

:::{admonition} Practical Tip: Start Simple
:class: tip
For most problems, start with a symmetric Gaussian random walk proposal:
$$\theta' = \theta_t + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

This is simple to implement and works well if you tune σ appropriately. Once you have that working, you can try fancier proposals if needed.
:::

### Tuning the Proposal: The Art of Step Size

The proposal distribution Q determines how the chain explores. A crucial parameter is the **step size** (scale of proposals).

**Too small**: Proposals are always accepted $(α ≈ 1)$, but you take tiny steps. The chain explores very slowly. High autocorrelation between successive samples.

**Too large**: Proposals often land in low-probability regions. High rejection rate $(α ≈ 0)$. The chain stays stuck in one place most of the time. Again, slow exploration.

**Just right**: Moderate acceptance rate (typically 20-50%). The chain explores efficiently, taking substantial steps while still accepting frequently enough.

:::{admonition} Optimal Acceptance Rates
:class: dropdown
For high-dimensional problems with Gaussian targets and Gaussian proposals, there's a beautiful theory (Roberts & Rosenthal 2001):

**Optimal acceptance rate**: ~23.4% as dimension $d → ∞$

This balances:

- Acceptance rate (how often you move)
- Step size (how far you move when accepted)

In practice, aim for:

- **1D problems**: 40-50% acceptance
- **Moderate dimensions (2-20)**: 25-40% acceptance  
- **High dimensions (>20)**: 20-30% acceptance

If your acceptance rate is outside these ranges, adjust your proposal scale σ:

- Too high acceptance (>60%)? Increase $σ$
- Too low acceptance (<15%)? Decrease $σ$
:::

**Adaptive tuning**: In practice, you might run the chain for a **burn-in** period, monitor the acceptance rate, and adjust $\sigma$. Common strategy:

- Run 1000 steps (burn-in)
- If acceptance rate > 50%, multiply $σ$ by 1.2
- If acceptance rate < 20%, divide $σ$ by 1.2
- Repeat until acceptance rate is in target range

Once tuned, **fix** the proposal and run your production chain. (Don't keep adapting during production — this violates the Markov property and detailed balance.)

### Pseudocode: The Complete Algorithm

```python
import numpy as np  # for np.log, np.random.uniform, array handling

def metropolis_hastings(log_posterior, proposal, theta_init, n_samples):
    """
    Generic Metropolis-Hastings MCMC sampler.
    
    Parameters:
    -----------
    log_posterior : function
        Computes log π(θ) (log posterior probability). Return the unnormalized log posterior: log p(D|θ) + log p(θ). Do not include log p(D); it is a constant and cancels in MH.
    proposal : function  
        Generates θ' given θ. Returns θ’ and log_Q_ratio = log Q(θ|θ') − log Q(θ'|θ) (order matters)
    theta_init : array
        Initial parameter values
    n_samples : int
        Number of MCMC samples to generate
    
    Returns:
    --------
    chain : array (n_samples, n_params)
        MCMC samples
    acceptance_rate : float
        Fraction of accepted proposals
    """
    
    theta = theta_init.copy()
    chain = np.zeros((n_samples, len(theta)))
    n_accepted = 0
    
    log_pi_current = log_posterior(theta)
    
    for i in range(n_samples):
        # 1. Propose new state
        theta_proposed, log_Q_ratio = proposal(theta)
        
        # 2. Evaluate posterior at proposed state
        log_pi_proposed = log_posterior(theta_proposed)
        
        # 3. Compute acceptance probability (in log space!)
        log_alpha = log_pi_proposed - log_pi_current + log_Q_ratio
        
        # 4. Accept or reject
        if np.log(np.random.uniform()) < log_alpha:
            # Accept
            theta = theta_proposed
            log_pi_current = log_pi_proposed
            n_accepted += 1
        # If rejected, theta stays the same (automatic in this code structure)
        
        # 5. Store sample
        chain[i] = theta
    
    acceptance_rate = n_accepted / n_samples
    return chain, acceptance_rate
```

**Key implementation details**:

1. **Work in log space**: Posteriors often involve tiny numbers (like 10⁻¹⁰⁰⁰). Directly computing exp(-1000) causes underflow — it becomes exactly 0 in floating point arithmetic. But log-space arithmetic handles this gracefully:

```python
# Direct space: numerical disaster
p1 = 1e-1000  # Underflows to 0!
p2 = 1e-1001  # Also underflows to 0!
ratio = p2 / p1  # 0/0 = NaN. Game over.

# Log space: works perfectly  
log_p1 = -1000 * log(10)  # ~-2302.6
log_p2 = -1001 * log(10)  # ~-2305.0  
log_ratio = log_p2 - log_p1  # -2.3 (a perfectly fine number!)
# If log_p = -1000, then p = exp(-1000) = 0 (underflow!)
# But log_p = -1000 is a perfectly fine number to work with.
```

2. **Proposal functions**: For symmetric Gaussian:

```python
def gaussian_proposal(theta, sigma=1.0):
    theta_new = theta + np.random.normal(0, sigma, size=len(theta))
    log_Q_ratio = 0.0  # Symmetric, so Q(θ|θ')/Q(θ'|θ) = 1
    return theta_new, log_Q_ratio
```

3. **Acceptance decision**: Compare log(uniform random) to log $α$ instead of comparing uniform random to $α$. This avoids exponentiating large negative numbers. (Use unnormalized log posterior: `loglike + logprior`; do not add `log p(D)`.)

### What About the Normalization Constant?

**Recall:** we only need $\pi(\theta)$ up to proportionality; this section shows explicitly how $p(D)$ cancels from the MH acceptance ratio.

Remember Bayes' theorem:

$$
p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)}
$$

The denominator $p(D)$ is often intractable (that's why we're doing MCMC in the first place!). But look at the acceptance probability:

$$
\alpha = \min\left(1, \frac{p(\theta' | D)}{p(\theta | D)}\right) = \min\left(1, \frac{p(D | \theta') p(\theta') / p(D)}{p(D | \theta) p(\theta) / p(D)}\right)
$$

The $p(D)$ terms **cancel**! You only need:

$$
\alpha = \min\left(1, \frac{p(D | \theta') p(\theta')}{p(D | \theta) p(\theta)}\right)
$$

**Profound implication**: MCMC only requires evaluating the **unnormalized** posterior (likelihood × prior). You never need the evidence p(D)!

This is why MCMC works for arbitrarily complex models. As long as you can evaluate $p(D|θ)$ (your likelihood/model) and $p(θ)$ (your prior), you can sample from the posterior.

---

## Part 4: Practical Diagnostics — Has Your Chain Converged?

:::{admonition} Where We Are
:class: note
**The journey so far**: Part 1 showed us the crisis (grid evaluation fails). Part 2 gave us the theory (Markov chains, detailed balance, ergodicity). Part 3 delivered the algorithm (Metropolis-Hastings acceptance criterion).

**Now in Part 4**: We face the practitioner's challenge—you've run your MCMC code and it produced numbers, but are they trustworthy? How do you know the chain has converged? This part gives you the diagnostic tools to answer that critical question.
:::

You've run your MCMC algorithm for N steps. You have N samples. Now the crucial question: **Are these samples from the posterior?**

The ergodic theorem says the chain *eventually* converges to π. But how long is "eventually"? If you haven't run long enough, your samples don't represent the posterior—they're still biased toward your starting point.

This is the **burn-in problem**: Early samples are not from π. You need to discard them.

### Visual Diagnostic: Trace Plots

The first and most important diagnostic: **Plot your samples over time.**

A trace plot shows $θ_t$ vs. $t$. What to look for:

**Good signs**:

- Looks like random noise around a stable mean
- No trends (increasing or decreasing)
- No periodic oscillations  
- Mixes well (explores the full range rapidly)
- Multiple chains from different starting points overlap

**Bad signs**:

- Systematic trend (chain hasn't equilibrated)
- Stuck in one region for a long time (mixing poorly)
- Periodic oscillations (not ergodic, perhaps?)
- Multiple chains give different distributions (not converged)

:::{admonition} Example: What Trace Plots Tell You
:class: note
Imagine three scenarios:

**Scenario 1: Not burned in**

```markdown
Iteration:  0 -------- 5000 -------- 10000
Parameter:  10         ↓               5
            |          ↓               ↓
            |          ↓               ⟷
            |          ↓               ⟷
```

Clear downward trend. The chain started at $θ=10$ and is still drifting toward the high-probability region near $θ=5$. Solution: Discard first ~7000 samples.

**Scenario 2: Well-mixed, converged**

``` markdown
Iteration:  0 -------- 5000 -------- 10000
Parameter:  ⟷          ⟷               ⟷
            ⟷          ⟷               ⟷
            5          5               5
```

Rapid fluctuations around stable mean. Looks like white noise. This is what you want!

**Scenario: 3 Stuck (poor mixing)**

```markdown
Iteration:  0 -------- 5000 -------- 10000  
Parameter:  5555555555 4444444444     6666666
```
Chain stays at one value for thousands of iterations, then jumps. The proposals are being rejected too often (step size too large?) or the posterior is multimodal (chain is trapped in one mode). Bad!
:::

<!--
SUGGESTED FIGURES: Trace Plot Examples
Create these three trace plots using Python to replace the ASCII art above:

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_iter = 10000

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Scenario 1: Not burned in (drifting toward equilibrium)
# Start at 10, exponentially decay toward 5 with noise
theta1 = np.zeros(n_iter)
theta1[0] = 10.0
for i in range(1, n_iter):
    # Exponential approach to 5 with decreasing rate
    drift = (theta1[i-1] - 5.0) * np.exp(-i/3000)
    noise = np.random.normal(0, 0.3)
    theta1[i] = theta1[i-1] - drift * 0.3 + noise

axes[0].plot(theta1, linewidth=0.5, alpha=0.7, color='darkblue')
axes[0].axvline(7000, color='red', linestyle='--', linewidth=2, label='Suggested burn-in')
axes[0].axhline(5, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='True value')
axes[0].set_ylabel('Parameter θ', fontsize=12)
axes[0].set_title('Scenario 1: Not Burned In (Systematic Drift)', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, n_iter)

# Scenario 2: Well-mixed, converged
# Stationary around 5 with white noise
theta2 = 5.0 + np.random.normal(0, 0.5, n_iter)
axes[1].plot(theta2, linewidth=0.5, alpha=0.7, color='darkgreen')
axes[1].axhline(5, color='red', linestyle='--', linewidth=1.5, label='Mean')
axes[1].fill_between(range(n_iter), 4.5, 5.5, alpha=0.2, color='green', label='±1σ region')
axes[1].set_ylabel('Parameter θ', fontsize=12)
axes[1].set_title('Scenario 2: Well-Mixed and Converged (White Noise)', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, n_iter)

# Scenario 3: Stuck (poor mixing, high autocorrelation)
# Chain gets stuck in regions for long periods
theta3 = np.zeros(n_iter)
theta3[0] = 5.0
values = [4.0, 5.0, 6.0]  # Three sticky values
current_val = 1
sticky_time = 0
for i in range(1, n_iter):
    sticky_time += 1
    # Stay stuck for ~2000 iterations, then jump
    if sticky_time > 2000 + np.random.randint(-500, 500):
        current_val = (current_val + 1) % 3
        sticky_time = 0
    # Add small noise around the sticky value
    theta3[i] = values[current_val] + np.random.normal(0, 0.05)

axes[2].plot(theta3, linewidth=0.5, alpha=0.7, color='darkred')
axes[2].axhline(5, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='True value')
axes[2].set_ylabel('Parameter θ', fontsize=12)
axes[2].set_xlabel('Iteration', fontsize=12)
axes[2].set_title('Scenario 3: Stuck (Poor Mixing, High Autocorrelation)', fontsize=14, fontweight='bold')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, n_iter)

plt.tight_layout()
plt.savefig('trace_plots_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Add text annotations explaining what to look for
print("Key diagnostic features:")
print("Scenario 1: Clear trend - needs more burn-in")
print("Scenario 2: Rapid exploration - good convergence")
print("Scenario 3: Long sticky periods - poor proposal tuning or multimodality")

This creates publication-quality trace plots showing the three diagnostic scenarios.
Students can see exactly what "good" and "bad" mixing looks like.
-->

### Quantitative Diagnostic 1: Autocorrelation Function

The **autocorrelation function (ACF)** measures how correlated θ_t is with θ_{t+k}:

$$
\rho_k = \frac{\text{Cov}(\theta_t, \theta_{t+k})}{\text{Var}(\theta_t)} = \frac{\mathbb{E}[(\theta_t - \mu)(\theta_{t+k} - \mu)]}{\mathbb{E}[(\theta_t - \mu)^2]}
$$

where $μ = ⟨θ⟩$ is the chain mean.

**Interpretation**:

- $ρ₀ = 1$ (perfect correlation with itself)
- $ρ_k → 0$ as $k → ∞$ for an ergodic chain
- Larger $k$ needed for $ρ_k ≈ 0$ means higher correlation (worse mixing)

**Autocorrelation time τ**: The characteristic lag at which samples become independent:

$$
\tau = 1 + 2 \sum_{k=1}^{\infty} \rho_k
$$

In practice, truncate the infinite sum when $ρ_k$ becomes negligible (say, when $ρ_k < 0.05$ or when $ρ_k$ is no longer statistically significant).

**Effective sample size (ESS)**:

$$
N_{\text{eff}} = \frac{N}{\tau}
$$

If $τ = 10$, then your $N=10000$ samples are only as informative as $N_\text{eff} = 1000$ independent samples. Your error bars should be computed using $N_\text{eff}$, not $N$.

:::{admonition} Practical Tip: What's a Good Autocorrelation Time?
:class: tip
**Target**: τ < 100 for most problems.

- $τ = 1$: Every sample is independent (ideal, rarely achieved)
- $τ = 10-50$: Good mixing, typical for well-tuned samplers
- $τ = 100-500$: Acceptable, but you'll need a long chain
- $τ > 500$: Poor mixing. Fix your proposal or increase step size.

If $τ$ is very large, you have two options:

1. Run the chain much longer ($N = 100τ$ or more for reliable statistics)
2. Improve your sampler (tune proposals, try HMC, etc.)
:::

### Quantitative Diagnostic 2: Gelman-Rubin Statistic

:::{margin}
**Gelman-Rubin diagnostic**:
:::

The **Gelman-Rubin diagnostic** (also called R-hat, $\hat{R}$) tests convergence by comparing multiple chains.

**Idea**: Run M chains from different starting points. If they've all converged to the same distribution, they should have:

- Similar means
- Similar variances
- Samples that are indistinguishable when pooled

**The R-hat statistic** compares variance *between* chains to variance *within* chains:

$$
\hat{R} = \sqrt{\frac{\text{Var}_{\text{pooled}}}{\text{Var}_{\text{within}}}}
$$

More precisely:

1. Run $M$ chains of length $N$ each (after burn-in)
2. Compute within-chain variance W (average variance across chains)
3. Compute between-chain variance B (variance of chain means)
4. Estimate total variance: $\hat{\text{Var}} = \frac{N-1}{N} W + \frac{1}{N} B$
5. Compute: $\hat{R} = \sqrt{\hat{\text{Var}} / W}$

**Interpretation**:

- $\hat{R} \approx 1$: Chains have converged (within-chain and between-chain variances agree)
- $\hat{R} > 1.1$: Chains have not converged (they give different answers)
- $\hat{R} > 1.2$: Serious convergence problem

**Rule of thumb**: $\hat{R} < 1.01$ for all parameters before trusting your results.

:::{admonition} Why Multiple Chains?
:class: note
You might wonder: Why run multiple chains when one chain (if run long enough) should converge to π?

**Reasons**:

1. **Convergence diagnosis**: If chains from different starting points give different results, you haven't converged.
2. **Multimodal distributions**: One chain might get stuck in one mode. Multiple chains with diverse initializations are more likely to find all modes.
3. **Parallelization**: Modern computers have multiple cores. Running M chains in parallel uses your hardware efficiently.

In practice, run at least $M=4$ chains. Some recommend $M=8$ for safety.
:::

### Burn-In: How Much to Discard?

The first part of each chain is **burn-in**: Samples before the chain has reached equilibrium. You must discard these before computing statistics.

**How long is burn-in?**

No universal answer. It depends on:

- How far your starting point is from the typical set
- How well your proposals are tuned  
- The geometry of your posterior (simple unimodal vs. complicated multimodal)
- Dimensionality

**Conservative approach**:

1. Run the chain for $N$ total steps
2. Plot trace plots for several parameters
3. Identify where the chain stabilizes (stops drifting)
4. Discard everything before that point (typically 10-50% of samples)
5. Check that R-hat < 1.01 for the remaining samples

**Practical tip**: It's safer to discard too much than too little. If you have 100,000 samples and discard the first 20,000, you still have plenty. But if you keep burn-in samples, your posterior estimates will be wrong.

### Putting It All Together: A Convergence Checklist

Before trusting your MCMC results, verify:

**Visual inspection**:

- [ ] Trace plots look like stationary noise (no trends or drift)
- [ ] Trace plots from multiple chains overlap
- [ ] Posterior distributions from each chain look similar

**Quantitative metrics**:

- [ ] R-hat < 1.01 for all parameters
- [ ] Effective sample size ESS $> 400$ per parameter (for 95% credible intervals)
- [ ] Autocorrelation time $τ < 100$ (or you've run long enough that $N/τ > 400$)

**Acceptance rate** (for Metropolis-Hastings):

- [ ] Acceptance rate between 20% and 50%
- [ ] Not too high (>60% means proposals too timid)  
- [ ] Not too low (<15% means proposals too aggressive)

**Sensitivity checks**:

- [ ] Results don't change if you run longer
- [ ] Results don't change if you discard more burn-in
- [ ] Results don't change if you change starting points

If all these pass, you can trust that your samples approximate the true posterior.

---

## Part 5: Real Example — Cepheid Variable Distance

:::{admonition} Where We Are
:class: note

**The journey so far**: We've built the complete MCMC framework—the crisis that motivates it (Part 1), the mathematical theory (Part 2), the algorithm itself (Part 3), and the diagnostic tools to validate it (Part 4).

**Now in Part 5**: Time to see it all come together. We'll apply MCMC to a real astrophysical inference problem: measuring the distance to a Cepheid variable star. This is not a toy problem—it's the same methodology that anchors the cosmic distance ladder and connects to the 2011 Nobel Prize-winning discovery of dark energy.
:::

Let's apply everything you've learned to a concrete astrophysical problem: Inferring the distance to a Cepheid variable star.

### The Science: Cepheids as Standard Candles

**Cepheid variables** are pulsating stars discovered by Henrietta Leavitt in 1908. She found a remarkable relationship: **The period of pulsation correlates with intrinsic luminosity.**

:::{admonition} The More You Know: Henrietta Leavitt and the Women Computers of Harvard
:class: dropdown

Henrietta Swan Leavitt's discovery of the period-luminosity relation was one of the most important breakthroughs in 20th century astronomy — it gave us our first "standard candle" for measuring cosmic distances. Yet her story reveals the barriers women faced in science.

In the early 1900s, Harvard College Observatory employed women as "computers" — human calculators who analyzed astronomical photographs. They were paid 25-50 cents per hour (about $7-14 in today's money), roughly half what male assistants earned, and were explicitly barred from using telescopes or proposing their own research. The director, Edward Pickering, hired them because they were cheaper than men and could do "tedious" computational work.

Despite these constraints, Leavitt was brilliant. Working with thousands of photographic plates of the Small Magellanic Cloud (SMC), she noticed that brighter Cepheid variables had longer periods. She published this in 1908, but her 1912 paper made it quantitative: there's a precise mathematical relationship between period and luminosity. This insight transformed astronomy.

Leavitt's law enabled Edwin Hubble to measure distances to other galaxies (1920s), discover the expanding universe, and overthrow the prevailing belief that the Milky Way was the entire cosmos. Hubble is famous; Leavitt is less known. She never received recognition from the major astronomy institutions during her lifetime — women weren't admitted to their societies. She died of cancer in 1921 at age 53.

Swedish mathematician Gösta Mittag-Leffler tried to nominate her for the Nobel Prize in 1925, unaware she had died. Nobel Prizes aren't awarded posthumously. Some historians believe that if she had lived longer, she would have received it — before Hubble.

**The lesson**: Scientific progress depends not just on brilliant individuals, but on systems that allow talent to flourish regardless of gender, race, or socioeconomic background. When we exclude people from science, we lose discoveries. Leavitt succeeded despite the system, not because of it — imagine what she could have achieved with equal resources and recognition.

Today when you use Cepheids as distance indicators, you're using Leavitt's Law. Remember the person behind it.
:::

This makes Cepheids **standard candles**: If you measure the period $(P)$, you know the absolute magnitude (M). Compare to the observed apparent magnitude $(m)$, and you can infer the distance.

**The distance modulus relation**:

$$
m - M = 5 \log_{10}(d / 10 \text{ pc})
$$

where d is the distance in parsecs.

**The Period-Luminosity relation** (Leavitt Law):

For Classical Cepheids in the V-band, the absolute magnitude depends on the logarithm of the period:

$$
M_V = -2.43 \log_{10}(P / 10 \text{ days}) - 4.05
$$

This gives $M_V$ for a reference metallicity. In practice, there are corrections for:

- Metallicity [Fe/H]
- Reddening/extinction
- Period-color-luminosity relations

For our example, we'll use the idealized case: A Cepheid with well-measured period, negligible extinction, and solar metallicity.

### The Problem: Inferring Distance from Photometry

**What you observe**:

- **Period**: P = 10.0 days (measured from light curve)
- **Apparent magnitude**: m_V = 18.50 ± 0.15 mag (from photometry)

**What you want to infer**:

- **Distance**: $d$ in parsecs (or kiloparsecs)
- **Uncertainty**: Full posterior $p(d | m, P)$

**The forward model**:

From the Leavitt Law with $P = 10.0$ days:
$$
M_V = -2.43 \log_{10}(10.0 / 10) - 4.05 = -2.43 \times 0 - 4.05 = -4.05
$$

The distance modulus is:
$$
\mu = m - M = 18.50 - (-4.05) = 22.55
$$

So:
$$
\mu = 5 \log_{10}(d / 10) = 22.55
$$

Solving for $d$:
$$
d = 10^{(\mu + 5)/5} = 10^{(22.55 + 5)/5} = 10^{5.51} \approx 323,000 \text{ pc} \approx 323 \text{ kpc}
$$

But this is just a point estimate. What's the uncertainty? Let's use Bayesian inference.

### Bayesian Setup

**Parameters**: $θ =$ distance $d$ (in kpc for numerical convenience)

**Data**: Observed apparent magnitude$m_\text{obs} = 18.50$ mag

**Likelihood**: Assuming Gaussian measurement errors:

$$
p(m_{\text{obs}} | d) = \mathcal{N}(m_{\text{obs}}; m_{\text{model}}(d), \sigma_m^2)
$$

where:

- $m_{\text{model}}(d) = M_V + 5 \log_{10}(d \times 1000 / 10)$ (d in kpc, distance in pc for the log term)
- $\sigma_m = 0.15$ mag (measurement uncertainty)

In log form:
$$
\log p(m_{\text{obs}} | d) = -\frac{1}{2} \left[ \frac{(m_{\text{obs}} - m_{\text{model}}(d))^2}{\sigma_m^2} + \log(2\pi\sigma_m^2) \right]
$$

**Prior**: Uniform in log-space (Jeffrey's prior for a scale parameter):

$$
p(d) \propto \frac{1}{d}
$$

We'll actually sample in log(d) to make this uniform:
$$
p(\log d) = \text{Uniform}[\log(50), \log(10,000)]
$$

This corresponds to distances between 50 kpc and 10 Mpc, a reasonable range for extragalactic Cepheids (e.g., in M31, M33, or Local Group galaxies).

**Posterior**: By Bayes' theorem:

$$
p(d | m_{\text{obs}}) \propto p(m_{\text{obs}} | d) \times p(d)
$$

<!--
### Implementation

Here's the complete MCMC implementation:

```python
import numpy as np
import matplotlib.pyplot as plt

# Data
P_days = 10.0  # period in days
m_obs = 18.50  # observed apparent magnitude
sigma_m = 0.15  # magnitude uncertainty

# Leavitt Law: Absolute magnitude from period
M_V = -2.43 * np.log10(P_days / 10.0) - 4.05
print(f"Absolute magnitude M_V = {M_V:.2f}")

# Forward model: Predicted apparent magnitude given distance
def magnitude_model(d_kpc):
    """
    Compute apparent magnitude from distance.
    
    Parameters:
    -----------
    d_kpc : float
        Distance in kiloparsecs
    
    Returns:
    --------
    m_pred : float
        Predicted apparent magnitude
    """
    d_pc = d_kpc * 1000  # Convert to parsecs
    distance_modulus = 5 * np.log10(d_pc / 10)
    m_pred = M_V + distance_modulus
    return m_pred

# Log-likelihood
def log_likelihood(d_kpc):
    m_pred = magnitude_model(d_kpc)
    chi_sq = ((m_obs - m_pred) / sigma_m)**2
    return -0.5 * chi_sq

# Log-prior (uniform in log(d), which gives p(d) ∝ 1/d)
def log_prior(d_kpc):
    if 50 <= d_kpc <= 10000:  # 50 kpc to 10 Mpc
        return -np.log(d_kpc)  # Jeffreys prior: p(d) ∝ 1/d
    else:
        return -np.inf  # Zero probability outside range

# Log-posterior
def log_posterior(d_kpc):
    lp = log_prior(d_kpc)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(d_kpc)

# MCMC sampler
def metropolis_hastings_cepheid(log_posterior, d_init, proposal_sigma, n_samples):
    """
    Run Metropolis-Hastings for Cepheid distance inference.
    """
    d_current = d_init
    chain = np.zeros(n_samples)
    n_accepted = 0
    
    log_post_current = log_posterior(d_current)
    
    for i in range(n_samples):
        # Propose new distance (Gaussian random walk in log-space)
        log_d_proposed = np.log(d_current) + np.random.normal(0, proposal_sigma)
        d_proposed = np.exp(log_d_proposed)
        
        # Evaluate posterior
        log_post_proposed = log_posterior(d_proposed)
        
        # Metropolis acceptance
        log_alpha = log_post_proposed - log_post_current
        
        if np.log(np.random.uniform()) < log_alpha:
            d_current = d_proposed
            log_post_current = log_post_proposed
            n_accepted += 1
        
        chain[i] = d_current
    
    acceptance_rate = n_accepted / n_samples
    return chain, acceptance_rate

# Run MCMC
np.random.seed(42)
n_samples = 20000
proposal_sigma = 0.05  # Step size in log(d)
d_init = 200.0  # Initial guess (200 kpc)

print("Running MCMC...")
chain, acc_rate = metropolis_hastings_cepheid(
    log_posterior, d_init, proposal_sigma, n_samples
)

print(f"Acceptance rate: {acc_rate:.1%}")
print(f"Expected distance: {10**((m_obs - M_V + 5) / 5):.1f} kpc")

# Discard burn-in
burn_in = 5000
chain_burned = chain[burn_in:]

# Posterior statistics
d_mean = np.mean(chain_burned)
d_std = np.std(chain_burned)
d_median = np.median(chain_burned)
d_16, d_84 = np.percentile(chain_burned, [16, 84])

print(f"\nPosterior summary (after {burn_in} burn-in):")
print(f"  Mean: {d_mean:.1f} kpc")
print(f"  Median: {d_median:.1f} kpc")
print(f"  Std dev: {d_std:.1f} kpc")
print(f"  68% credible interval: [{d_16:.1f}, {d_84:.1f}] kpc")

# Trace plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(chain, alpha=0.7)
plt.axvline(burn_in, color='r', linestyle='--', label='Burn-in')
plt.xlabel('Iteration')
plt.ylabel('Distance (kpc)')
plt.title('Trace Plot')
plt.legend()

# Posterior histogram
plt.subplot(1, 2, 2)
plt.hist(chain_burned, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.axvline(d_median, color='r', linestyle='--', label=f'Median = {d_median:.1f} kpc')
plt.axvline(d_16, color='orange', linestyle=':', label=f'16th %ile = {d_16:.1f} kpc')
plt.axvline(d_84, color='orange', linestyle=':', label=f'84th %ile = {d_84:.1f} kpc')
plt.xlabel('Distance (kpc)')
plt.ylabel('Posterior probability density')
plt.title('Posterior Distribution')
plt.legend()
plt.tight_layout()
plt.show()

# Compute autocorrelation
def autocorr(x, max_lag=500):
    x_centered = x - np.mean(x)
    c0 = np.dot(x_centered, x_centered) / len(x)
    acf = np.zeros(max_lag)
    for k in range(max_lag):
        ck = np.dot(x_centered[:-k or None], x_centered[k:]) / len(x)
        acf[k] = ck / c0
    return acf

acf = autocorr(chain_burned, max_lag=200)

plt.figure(figsize=(8, 4))
plt.plot(acf, 'o-', alpha=0.7, markersize=3)
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.axhline(0.05, color='r', linestyle=':', label='5% threshold')
plt.axhline(-0.05, color='r', linestyle=':')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Estimate autocorrelation time
acf_sum = 1 + 2 * np.sum(acf[1:np.where(np.abs(acf) < 0.05)[0][0]])
print(f"\nAutocorrelation time τ ≈ {acf_sum:.1f}")
print(f"Effective sample size N_eff ≈ {len(chain_burned) / acf_sum:.0f}")
```
--->

<!--
SUGGESTED FIGURES: Cepheid MCMC Diagnostics
Create comprehensive diagnostic plots for the Cepheid example:

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# After running the MCMC code above, create a comprehensive 4-panel figure

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Panel 1: Full trace plot with burn-in marker
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(chain, linewidth=0.5, alpha=0.7, color='steelblue')
ax1.axvline(burn_in, color='red', linestyle='--', linewidth=2, label=f'Burn-in cutoff ({burn_in})')
ax1.axhline(d_median, color='orange', linestyle=':', linewidth=1.5, label=f'Posterior median ({d_median:.1f} kpc)')
ax1.fill_between(range(len(chain)), d_16, d_84, alpha=0.2, color='green', label='68% credible region')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Distance (kpc)', fontsize=12)
ax1.set_title('MCMC Trace Plot: Cepheid Distance Inference', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel 2: Posterior distribution with credible intervals
ax2 = fig.add_subplot(gs[1, 0])
counts, bins, _ = ax2.hist(chain_burned, bins=50, density=True, alpha=0.7, 
                           edgecolor='black', color='skyblue', label='MCMC samples')
# Overlay a KDE for smoothness
from scipy.stats import gaussian_kde
kde = gaussian_kde(chain_burned)
x_range = np.linspace(chain_burned.min(), chain_burned.max(), 200)
ax2.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
ax2.axvline(d_median, color='darkred', linestyle='--', linewidth=2, label=f'Median: {d_median:.1f} kpc')
ax2.axvline(d_16, color='orange', linestyle=':', linewidth=1.5, label=f'16th %ile: {d_16:.1f} kpc')
ax2.axvline(d_84, color='orange', linestyle=':', linewidth=1.5, label=f'84th %ile: {d_84:.1f} kpc')
ax2.set_xlabel('Distance (kpc)', fontsize=12)
ax2.set_ylabel('Posterior density', fontsize=12)
ax2.set_title('Posterior Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Autocorrelation function
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(acf[:100], 'o-', markersize=3, alpha=0.7, color='darkblue')
ax3.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax3.axhline(0.05, color='red', linestyle=':', linewidth=1.5, label='5% threshold')
ax3.axhline(-0.05, color='red', linestyle=':', linewidth=1.5)
ax3.fill_between(range(100), -0.05, 0.05, alpha=0.2, color='gray', label='Negligible correlation')
ax3.set_xlabel('Lag', fontsize=12)
ax3.set_ylabel('Autocorrelation', fontsize=12)
ax3.set_title(f'ACF (τ ≈ {acf_sum:.1f})', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 100)

# Panel 4: Cumulative distribution for credible intervals
ax4 = fig.add_subplot(gs[2, :])
sorted_chain = np.sort(chain_burned)
cumulative = np.arange(1, len(sorted_chain) + 1) / len(sorted_chain)
ax4.plot(sorted_chain, cumulative * 100, linewidth=2, color='darkgreen')
ax4.axhline(50, color='red', linestyle='--', alpha=0.5, label='Median (50th percentile)')
ax4.axhline(16, color='orange', linestyle=':', alpha=0.5, label='16th percentile')
ax4.axhline(84, color='orange', linestyle=':', alpha=0.5, label='84th percentile')
ax4.axvline(d_median, color='red', linestyle='--', alpha=0.5)
ax4.axvline(d_16, color='orange', linestyle=':', alpha=0.5)
ax4.axvline(d_84, color='orange', linestyle=':', alpha=0.5)
ax4.fill_betweenx([0, 100], d_16, d_84, alpha=0.2, color='green', label='68% credible region')
ax4.set_xlabel('Distance (kpc)', fontsize=12)
ax4.set_ylabel('Cumulative probability (%)', fontsize=12)
ax4.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(sorted_chain.min(), sorted_chain.max())
ax4.set_ylim(0, 100)

plt.savefig('cepheid_mcmc_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()

This comprehensive figure shows all key diagnostics in one view, making it easy
to assess convergence, mixing, and posterior uncertainty at a glance.
-->
<!---
:::{admonition} Pause and Predict
:class: tip
**Before looking at the results below, make predictions:**

Given our setup ($m_\text{obs} = 18.50 ± 0.15$ mag, $M_V = -4.05$ mag):

1. Roughly what distance do you expect? (Hint: Use the distance modulus formula)
2. What uncertainty range seems reasonable given σ_m = 0.15 mag?
3. Will the acceptance rate be in the optimal range (20-50%)?
4. Should the autocorrelation time be small (<20) or large (>100)?

Write down your predictions, then check them against the actual results below. This active prediction significantly improves learning and retention.
:::

### Results and Interpretation

Running this code yields:

```bash
Absolute magnitude M_V = -4.05
Running MCMC...
Acceptance rate: 25.3%
Expected distance: 323.0 kpc

Posterior summary (after 5000 burn-in):
  Mean: 324.5 kpc
  Median: 323.8 kpc
  Std dev: 22.8 kpc
  68% credible interval: [304.5, 345.2] kpc

Autocorrelation time τ ≈ 9.2
Effective sample size N_eff ≈ 1630
```

**What we learned**:

1. **Distance**: The Cepheid is at d = 324 ± 23 kpc (68% credible interval: 305-345 kpc). This is consistent with the point estimate of 323 kpc but now we have proper uncertainty quantification.

2. **Acceptance rate**: 25% is in the optimal range for a 1D problem. The sampler is well-tuned.

3. **Convergence**: The trace plot shows good mixing after ~2000 iterations. By discarding 5000 as burn-in, we're being conservative.

4. **Autocorrelation**: $τ ≈ 9$ means samples are nearly independent. Our 15,000 post-burn-in samples are equivalent to ~1630 truly independent samples — more than enough for robust statistics.

5. **Physical context**: A distance of ~324 kpc places this Cepheid well beyond the Milky Way. Possible host galaxies at this distance include:

   - **Andromeda (M31)**: ~780 kpc (too close)
   - **Triangulum (M33)**: ~950 kpc (too close)
   - **Local Group dwarf galaxies**: 100-1000 kpc (possible!)

   Our Cepheid might be in a dwarf galaxy in the Local Group, or in the extended halo of M31. With this distance and the Leavitt Law, we can place the Cepheid on the cosmic distance ladder — the same ladder that extends to Type Ia supernovae (your Project 4) and ultimately measures cosmological parameters.

:::{admonition} Connection to Project 4
:class: note
You've just inferred a distance to a Cepheid using MCMC. In Project 4, you'll do the same thing for Type Ia supernovae — but now inferring cosmological parameters $(Ω_m, h)$ from hundreds of supernovae at $z ~ 0.01 to 1.0$.

The method is identical: Write down a forward model (distance-redshift relation in cosmology), define a likelihood (Gaussian magnitude errors), specify priors, and sample the posterior with MCMC.

Cepheids calibrate the local distance scale. Type Ia SNe extend it to cosmological distances. Together, they map the geometry of the universe. The 2011 Nobel Prize in Physics went to Saul Perlmutter, Brian Schmidt, and Adam Riess for using this technique to discover dark energy—the universe's accelerating expansion.

You're learning the same methods that earned a Nobel Prize.
:::
--->
---

## Synthesis — The Conceptual Architecture

:::{admonition} Where We Are
:class: note
**The complete journey**: You've traveled from the computational crisis that makes MCMC necessary (Part 1), through the elegant mathematical theory of Markov chains (Part 2), to the ingenious Metropolis-Hastings algorithm (Part 3), learned how to diagnose whether your sampler actually works (Part 4), and applied it all to real astrophysics (Part 5).

Now we zoom out to see the big picture. How do all these pieces fit together? How does MCMC connect to the broader themes of this course — statistical thinking, physical intuition, computational methods? This synthesis reveals the deep unity underlying everything you've learned.
:::

You've learned a lot. Let's step back and see the structure of what you now know.

### The Four Pillars of MCMC

:::{admonition} Conceptual Architecture of MCMC
:class: note

**1. Statistical Foundation (Module 1):**

- Law of Large Numbers: Sample averages converge to expectations
- Central Limit Theorem: Estimation errors scale as 1/√N
- Sampling distributions: Characterize uncertainty via distributions

**2. Stochastic Process Theory (Module 5, Part 2):**

- Markov chains: Memoryless random walks through state space
- Stationary distributions: Equilibrium distributions preserved by dynamics
- Ergodicity: Time averages equal ensemble averages
- Detailed balance: Sufficient condition for stationarity

**3. Algorithmic Implementation (Module 5, Part 3):**

- Proposal distribution $Q$: How to generate candidates
- Acceptance probability $α$: When to accept, based on posterior ratio
- Metropolis-Hastings: The universal recipe for MCMC sampling

**4. Practical Application (Module 5, Parts 4-5):**

- Diagnostics: Trace plots, R-hat, autocorrelation, ESS
- Burn-in: Discarding pre-equilibrium samples
- Tuning: Adjusting proposals for efficient exploration
- Real inference: From theory to astrophysical discoveries
:::

Each pillar rests on the previous one. You can't appreciate MCMC diagnostics without understanding Markov chains. You can't design good samplers without knowing detailed balance. And you can't trust your results without proper statistical foundations.

### Universal Patterns Across Modules

One of the profound lessons of this course: **The same mathematical structures appear everywhere.** Let's make this explicit:

| **Concept** | **Module 1: Statistics** | **Modules 3: Statistical Mechanics** | **Module 5: Bayesian Inference** |
|-------------|-------------------------|--------------------------------|--------------------------------|
| **What we're sampling** | Random variables | Microstates in phase space | Parameters in posterior |
| **Target distribution** | Population distribution | Boltzmann distribution | Posterior p(θ\|D) |
| **How we sample** | Independent and Identically Distributed (IID) random draws | Molecular dynamics | Markov chain (MCMC) |
| **Why it works** | Law of Large Numbers | Ergodic hypothesis | Ergodic theorem |
| **Convergence condition** | CLT (independent samples) | Thermal equilibration | Detailed balance |
| **What we compute** | Sample statistics | Thermodynamic quantities | Parameter estimates |
| **Time average = ?** | Ensemble average (Module 1) | Ensemble average (Module 3) | Expectation under π (Module 5) |

**The profound unity**: Whether you're:

- Drawing random samples to estimate a population mean (Module 1)
- Simulating molecules reaching thermal equilibrium (Module 3)  
- Sampling parameters to quantify uncertainty (Module 5)

You're using the **same mathematical structure**: stochastic processes converging to equilibrium distributions, with time averages equaling ensemble averages.

This is why physics is so powerful *and predictive*. The same principles work everywhere, from atoms to stars to probability distributions.

:::{admonition} Reflection Question
:class: tip
**Before moving on, pause and think**:

How is MCMC sampling from $p(θ|D)$ analogous to molecular dynamics sampling from the Boltzmann distribution $p(E) ∝ \exp(-E/kT)$?

Write down the correspondence:

- Temperature T ↔ ?
- Energy E ↔ ?
- Particle position x ↔ ?
- Boltzmann distribution ↔ ?

**Answer**: T = 1 (fixed), E = -log p(θ|D), x = θ, Boltzmann ↔ Posterior.

They're mathematically identical! Replace energy with negative log-probability, and MCMC *is* molecular dynamics in parameter space.
:::

### From Theory to Practice: Your Toolbox

You now have:

**1. Conceptual understanding**:

- Why MCMC is necessary (curse of dimensionality)
- How MCMC works (Markov chains, detailed balance)
- When MCMC converges (ergodicity, stationarity)

**2. Practical skills**:

- Implementing Metropolis-Hastings
- Tuning proposal distributions
- Diagnosing convergence (trace plots, R-hat, ACF)
- Interpreting results (corner plots, credible intervals)

**3. Scientific applications**:

- Setting up Bayesian models (likelihood + prior)
- Running inference on real data (Cepheids)
- Quantifying uncertainty properly

**4. Connections**:

- To Module 1 (sampling, CLT, ergodicity)
- To Module 3 (phase space, thermalization, detailed balance)
- To Project 4 (Type Ia SNe, cosmological parameters)

This is a **complete inference framework** you can apply to any scientific problem.

---
<!---
## Conceptual Checkpoints

Before Project 4, verify your understanding:

1. **Why doesn't grid evaluation work in 10 dimensions?**  
   *Answer*: You need $100^{10}$ = 10^{20} evaluations. Even at 1 microsecond each, that's 3 million years.

2. **Explain detailed balance in your own words. Why does it guarantee that π(θ) is stationary?**  
   *Answer*: Detailed balance says forward and reverse transition rates balance when both states are drawn from π. If you integrate this condition, you find that starting from distribution π produces π again after one step. This means π is unchanged by the dynamics — it's stationary.

3. **Your acceptance rate is 95%. Is this good or bad? Why?**  
   *Answer*: Bad! It means your proposals are tiny. You're taking baby steps. Lower the acceptance rate by increasing the proposal step size.

4. **You run two chains from different starting points. One gives $θ = 5.0 ± 0.2$, the other gives $θ = 5.8 ± 0.3$. R-hat = 1.25. What's wrong?**  
   *Answer*: The chains haven't converged to the same distribution yet. Run longer. The burn-in may be insufficient, or the proposals are poorly tuned.

5. **Your chain has N = 20,000 samples and $τ = 100$. How many "effectively independent" samples do you have?**  
   *Answer*: N_eff = N/τ = 20,000/100 = 200. Your statistical uncertainties are as if you had only 200 independent samples.

6. **True or false: "High acceptance rate means my sampler is working well."**  
   *Answer*: False! High acceptance (> 60%) usually means your proposals are too timid. Optimal is 20-40%.

7. **You want to infer parameters (mass, radius) of a star. The posterior has a strong correlation: high mass correlates with high radius. Will a diagonal proposal covariance work well?**  
   *Answer*: No. Independent proposals don't respect the correlation structure. You'll propose (low mass, high radius) frequently, which has low probability. Use a proposal covariance that matches the posterior correlation.

8. **[Advanced Integration Problem]** You're inferring the orbital parameters of an exoplanet from radial velocity measurements. Your model has five parameters: orbital period P, eccentricity $e~(0 ≤ e < 1)$, inclination $i$, argument of periastron $ω$, and time of periastron passage $τ$. Your likelihood requires numerically solving Kepler's equation at each evaluation (computationally expensive, ~50ms per call). The posterior is known to have a degeneracy: certain combinations of i and ω produce similar radial velocity signals, creating a multimodal posterior.

   **Design your MCMC strategy. Address each of these questions:**

    a) What proposal distribution would you use? Would you propose all five parameters jointly, or use some other strategy?

    b) How would you handle the bounded parameter e ∈ [0, 1)? (Hint: Consider transformation)

    c) Given the expensive likelihood (50ms per call), approximately how long would it take to generate 100,000 MCMC samples on one CPU core? Is this practical?

    d) How many independent chains would you run, and from what starting points?

    e) What specific diagnostics would you use to detect the multimodality? (R-hat might look fine even if chains are stuck in different modes!)

    f) The degeneracy between i and ω means the posterior has multiple peaks. How would you ensure your chains explore all modes? What happens if you miss one?

   **Suggested answer approach**: There's no single "right" answer, but good responses should demonstrate: (1) awareness of the computational budget (expensive likelihood), (2) strategies for bounded parameters (transform e), (3) recognition that standard diagnostics may miss multimodality (need careful initialization and visual inspection of 2D marginals), (4) understanding that missing modes leads to incorrect uncertainty quantification. Advanced responses might suggest parallel tempering, adaptive proposals, or using cheap surrogate models for initial exploration.
--->

## Common Misconceptions About MCMC

Before moving to Project 4, let's address some frequent misunderstandings that trip up even experienced practitioners:

**Misconception 1: "High acceptance rate means good sampling"**

**Reality**: High acceptance (>60%) usually means your proposals are too timid—you're taking baby steps and exploring slowly. The chain mixes poorly and has high autocorrelation. Conversely, very low acceptance (<15%) means proposals are too aggressive. Optimal acceptance depends on dimension: aim for 20-40% in most cases.

**Why it's tempting**: It feels good when proposals are accepted! But acceptance rate is about efficiency, not validity.

**Misconception 2: "R-hat < 1.1 guarantees convergence"**

**Reality**: R-hat < 1.1 is *necessary* but not *sufficient* for convergence. If your posterior has multiple well-separated modes and each chain gets stuck in one mode, R-hat will look fine but you haven't converged to the full posterior. You've converged to individual modes.

**How to catch it**: Always visually inspect trace plots and 2D marginal distributions. Look for chains that never overlap. Run chains from diverse starting points spanning the prior range.

**Misconception 3: "MCMC samples are independent"**

**Reality**: MCMC samples are *correlated*. The effective sample size N_eff = N/τ accounts for this. If τ = 100, your 10,000 samples carry as much information as 100 independent samples. Always report N_eff, not just N.

**Practical impact**: If you compute credible intervals assuming independence, they'll be too narrow (overconfident). Use N_eff for uncertainty quantification.

**Misconception 4: "Longer chains are always better"**

**Reality**: Running a poorly-tuned chain longer doesn't fix the problem—it just wastes time. If your proposals are bad, adding more samples doesn't help. Fix the sampler first (tune step size, use better proposals, add constraints), then run long chains.

**The exception**: If your chain is well-tuned but exploring a complex posterior, then yes, longer is better. But "longer" means N_eff > 400 per parameter, not just more samples.

**Misconception 5: "I can start MCMC anywhere"**

**Reality**: While theory says chains converge from any starting point, starting very far from the typical set (the region of high probability) means long burn-in. Starting outside your prior bounds or at parameter values where the likelihood is undefined will cause immediate failure.

**Best practice**: Start from reasonable values. For well-specified priors, sample from the prior. For complex problems, run optimization first to find the posterior mode, then start MCMC nearby.

**Misconception 6: "Burn-in is waste"**

**Reality**: Burn-in is not waste—it's how the chain finds the typical set! It's like a spacecraft's trajectory correction burns to reach the right orbit. The real waste is keeping burn-in samples and contaminating your posterior estimates.

**How much to discard**: Be conservative. If in doubt, discard more. Look at trace plots—when do they stabilize?

**Misconception 7: "MCMC always works"**

**Reality**: MCMC can fail in many ways:

- Multimodal posteriors where chains get trapped in one mode
- Extremely high-dimensional problems where mixing is glacially slow  
- Posteriors with complex geometry (funnel shapes, banana-shaped ridges)
- Highly correlated parameters that random walk proposals can't handle
- Likelihood functions with discontinuities or numerical instabilities

**Recognition**: That's why diagnostics exist! If R-hat is bad, acceptance is extreme, or chains look weird, **don't trust the results**. Fix the sampler or use a different method.

**Misconception 8: "The posterior mean is the best parameter estimate"**

**Reality**: Depends on your loss function! The posterior mean minimizes squared-error loss. The posterior median minimizes absolute-error loss. The posterior mode (MAP) minimizes 0-1 loss. Which one you report depends on what you're trying to do. For asymmetric posteriors, they can differ substantially.

**Best practice**: Report the full posterior (credible intervals) rather than a single point estimate. Let readers make their own decisions about point estimates.

---

## Looking Ahead: Project 4 and Beyond

### What's Next

In **Project 4**, you'll implement everything from this module to measure cosmological parameters from Type Ia supernovae:

**Your tasks**:

1. Build the forward model (cosmological distance-redshift relation)
2. Implement Metropolis-Hastings MCMC
3. Apply diagnostics (trace plots, R-hat, ACF, corner plots)
4. Run inference on real SNe data (JLA sample)
5. Measure $Ωₘ$ and $h$ (the contents of the universe!)
6. Extend to Hamiltonian Monte Carlo (using your leapfrog integrator from Project 2)

This is the **same data and methods** that earned the 2011 Nobel Prize in Physics for the discovery of dark energy.

### Advanced Topics (Module 5.4 Preview)

After mastering Metropolis-Hastings, you'll learn:

**Hamiltonian Monte Carlo (HMC)**:

- Uses gradient information for efficient exploration
- Based on Hamiltonian dynamics (Module 3!)
- Much lower autocorrelation than random walk
- The leapfrog integrator from Project 2 is exactly what HMC uses

**Affine-invariant ensemble samplers** (like `emcee`):

- Multiple "walkers" evolve together
- Self-tuning, no manual proposal tuning needed
- Handles strong correlations automatically

**No-U-Turn Sampler (NUTS)**:

- Adaptive HMC that automatically tunes step size and trajectory length
- The algorithm behind Stan and PyMC
- State of the art for general-purpose MCMC

But these are all variations on the core principles you learned here. Metropolis-Hastings is the foundation.

### Where is MCMC Heading? The Frontier of Computational Inference

MCMC is not a settled field — it's actively evolving. Here are some exciting research directions that may transform how we do inference in the next decade:

**Neural network-assisted MCMC**: Recent work uses machine learning to learn optimal proposal distributions from data. Instead of hand-tuning proposals, train a neural network to predict good proposals based on the posterior landscape. Early results show dramatic speedups for complex problems. This combines classical statistics with modern deep learning.

**Adaptive and self-tuning methods**: Modern samplers can adapt their behavior *during* sampling without violating detailed balance (which traditional theory said was impossible!). Methods like adaptive Metropolis (Haario et al. 2001) and delayed rejection algorithms show that we can be smarter about exploration while maintaining theoretical guarantees.

**Non-reversible MCMC**: Traditional MCMC satisfies detailed balance (microscopic reversibility). But what if we intentionally break reversibility to explore faster? Non-reversible Langevin samplers and lifted particle filters can have much lower autocorrelation than reversible methods. This is cutting-edge research connecting to non-equilibrium statistical mechanics.

**Variational inference and hybrid methods**: Not everything requires sampling. Variational methods approximate posteriors with simpler distributions (like Gaussians) by optimization rather than sampling. Modern hybrid approaches combine MCMC's accuracy with variational inference's speed. The future may be algorithms that switch between sampling and optimization adaptively.

**Quantum MCMC**: As quantum computers mature, researchers are developing quantum algorithms for Bayesian inference. Quantum walks could explore posterior landscapes exponentially faster than classical MCMC—though practical implementations are years away.

**Amortized inference**: Instead of running MCMC separately for each dataset, train a model once that can instantly produce posteriors for new data. This "simulation-based inference" approach (also called likelihood-free inference or neural posterior estimation) is revolutionizing fields where likelihood evaluation is expensive.

**The common thread**: All these advances build on the foundations you learned—Markov chains, stationarity, convergence theory. Understanding Metropolis-Hastings deeply prepares you to understand and contribute to these cutting-edge methods. The principles don't change; the implementations get ever more sophisticated.

MCMC has come a long way from simulating nuclear reactions on MANIAC I in 1953. Where will it go next? Perhaps you'll help decide.

### The Professional Path

After building MCMC from scratch, you'll be able to use professional tools **intelligently**:

**emcee** (Python): Affine-invariant ensemble sampler  
**PyMC** (Python): Full Bayesian modeling framework with HMC/NUTS  
**Stan** (C++): Gold-standard Bayesian inference (autodiff, NUTS)  
**JAX/numpyro** (Python): HMC with Just-In-Time (JIT) compilation and GPU acceleration

You won't be a black-box user clicking buttons. You'll understand:

- Why these tools work (detailed balance, ergodicity)
- How to diagnose problems (R-hat, trace plots, divergences)
- When methods fail (multimodal posteriors, stiff likelihoods)
- What the algorithms are actually doing under the hood

**That's the glass-box philosophy in action.**

---
<!---
## Further Reading and Resources

**Essential**:

- **Hogg & Foreman-Mackey (2018)**: "Data analysis recipes: Using Markov Chain Monte Carlo" (arXiv:1710.06068) — Best practical guide to MCMC in astronomy

**Historical**:

- **Metropolis et al. (1953)**: "Equation of State Calculations by Fast Computing Machines" — The original paper that started it all
- **Hastings (1970)**: "Monte Carlo Sampling Methods Using Markov Chains..." — The generalization that gives us Metropolis-Hastings

**Textbooks**:

- **Gelman et al.**: *Bayesian Data Analysis* (3rd ed.) — The bible of Bayesian inference
- **MacKay**: *Information Theory, Inference, and Learning Algorithms* — Beautiful connections between physics, information theory, and inference
- **Brooks et al.**: *Handbook of Markov Chain Monte Carlo* — Advanced theoretical treatment

**Code**:

- **emcee**: https://emcee.readthedocs.io — Excellent documentation and examples
- **corner.py**: https://corner.readthedocs.io — For making corner plots

--->

**Congratulations!** You've learned one of the most important computational techniques in modern science. MCMC underpins:

- Bayesian inference across all sciences
- Machine learning (training deep networks is MCMC-like)
- Computational physics (lattice QCD, molecular dynamics)
- Statistics (everywhere!)

More importantly, you've seen how **measurement, probability, physics, and computation** all connect into a unified framework for learning from data.

Now go build your inference engine. The universe is waiting to be measured.

---

## Self-Assessment Rubric

Before moving forward, honestly assess your mastery of this module. This rubric helps you identify areas for review and confirms readiness for Project 4.

### Level 1: Conceptual Understanding

*Can you explain the ideas to someone else?*

- [ ] **Basic** (Developing): I can explain why MCMC is needed in general terms but struggle with the curse of dimensionality details.
- [ ] **Proficient**: I can clearly explain the curse of dimensionality with concrete examples (e.g., 10^20 evaluations for d=10).
- [ ] **Advanced**: I can explain the conceptual progression: crisis (curse) → solution (Monte Carlo) → problem (how to sample?) → answer (Markov chains) → implementation (Metropolis-Hastings), and justify each step.

### Level 2: Mathematical Foundations

*Can you derive key results?*

- [ ] **Basic** (Developing): I can state the detailed balance equation but cannot derive how it implies stationarity.
- [ ] **Proficient**: I can derive that detailed balance implies stationarity by integrating both sides, and I understand why the normalization constant cancels in the acceptance probability.
- [ ] **Advanced**: I can derive the Metropolis-Hastings acceptance criterion from detailed balance, verify it satisfies the ratio condition for both R > 1 and R < 1 cases, and explain why min(1, r) is optimal.

### Level 3: Implementation Skills

*Can you code MCMC from scratch?*

- [ ] **Basic** (Developing): I can follow the pseudocode but need significant help implementing it in Python.
- [ ] **Proficient**: I can implement Metropolis-Hastings for a simple 1D problem (like the Cepheid example) with working proposal and acceptance logic, using log-space arithmetic correctly.
- [ ] **Advanced**: I can implement M-H for multidimensional problems, tune proposals adaptively, handle constraints (bounded parameters), and debug common issues (underflow, stuck chains, poor mixing).

### Level 4: Diagnostic Expertise

*Can you tell if your sampler is working?*

- [ ] **Basic** (Developing): I know I should check trace plots but I'm not sure what "good" looks like.
- [ ] **Proficient**: I can create and interpret trace plots, compute R-hat and ACF, determine when burn-in is sufficient, and recognize clear convergence failures (stuck chains, high R-hat).
- [ ] **Advanced**: I can diagnose subtle problems (multimodal posteriors where chains are in different modes but R-hat looks okay, systematic biases in proposals), adjust sampling strategy accordingly, and justify my burn-in and thinning choices quantitatively.

### Level 5: Scientific Application

*Can you apply MCMC to real problems?*

- [ ] **Basic** (Developing): I can run provided MCMC code but struggle to set up likelihoods and priors for new problems.
- [ ] **Proficient**: I can formulate a Bayesian inference problem from scratch (define likelihood from measurement model, choose appropriate priors, implement log-posterior), run MCMC, and interpret credible intervals correctly.
- [ ] **Advanced**: I can handle complex scientific problems (multiple parameters with constraints, multimodal posteriors, expensive likelihoods requiring optimization), choose appropriate diagnostics for the specific problem, and present results in publication-ready format with proper uncertainty quantification.

### Level 6: Connections and Transfer

*Can you see how MCMC fits into the bigger picture?*

- [ ] **Basic** (Developing): I see MCMC as an isolated technique for Bayesian inference.
- [ ] **Proficient**: I understand how MCMC connects to Module 1 (CLT, LLN, ergodicity) and Module 3 (detailed balance = microscopic reversibility, Boltzmann distribution analogy).
- [ ] **Advanced**: I can articulate the deep mathematical unity across modules (same stochastic process theory governs sampling, molecular dynamics, and MCMC), explain how different MCMC variants (HMC, ensemble samplers) relate to the core principles, and identify when MCMC is the right tool versus alternatives (optimization, variational inference, grid methods).

**Remember**: This is self-assessment for learning, not grading. Be honest with yourself. Identifying gaps now makes Project 4 more productive.

---

**Next Module**: Hamiltonian Monte Carlo (HMC) and advanced sampling methods  
