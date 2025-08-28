---
title: Understanding Your Learning Journey in ASTR 596
subtitle: "ASTR 596: Modeling the Universe"
exports:
  - format: pdf
---

:::{tip} TL;DR: ASTR 596 Course Overview
:class: dropdown

**What**: Build algorithms from scratch (stellar populations/HR diagrams → N-body dynamics → radiative transfer → MCMC → Gaussian Processes → neural networks) to understand how they work, not just how to use them.

**How**: Glass-box philosophy - construct every algorithm yourself before using libraries. Three-phase AI scaffolding (struggle first → strategic use → professional integration).

**Why**: Transform from package-user to tool-builder. When everyone has AI, those who understand fundamentals win.

**Projects**: Each builds on previous work. Your N-body simulations become training data for machine learning. You'll literally measure the Universe's composition and discover hidden physics.

**Reality check:** This is challenging but achievable. Projects build incrementally with clear guidance. Mathematical complexity increases gradually, with just-in-time tutorials when you need them.

**Outcome:** Implement algorithms from foundational principles → understand deeply how they work → modify and extend them confidently. Gain the ability to tackle any computational method.
:::

## Why ASTR 596 is designed this way

This course follows a specific progression: **Fundamentals → Classical Methods → Statistical Methods → Modern ML**. This mirrors how the field itself evolved, but more importantly, each topic builds essential skills for the next. You'll essentially recreate the historical development of computational astrophysics, but in a compressed, logical sequence that maximizes your learning.

At the heart of this course is the **"glass-box" philosophy"** – you'll build every algorithm from scratch before using advanced libraries. This isn't masochism; it's pedagogy. When you implement backpropagation by hand, you understand why neural networks fail. When you code your own MCMC sampler, you recognize convergence problems, understand why certain priors constrain parameters effectively, and grasp how MCMC methods asymptotically sample from the true posterior with proper convergence diagnostics. This deep understanding – knowing why algorithms work mathematically while also being able to implement them from scratch – distinguishes computational scientists from package users who just call functions. You're learning to think, not just code.

Understanding why this course is structured this way helps you see the forest through the trees and appreciate how each assignment builds toward your growth as a computational scientist. Throughout this journey, you'll also develop AI literacy through our **three-phase AI scaffolding approach**:

- **AI Phase 1 (Projects 1-3)**: Struggle first, AI for debugging only after 30 minutes
- **AI Phase 2 (Projects 4-5)**: Strategic AI use after baseline implementation works  
- **AI Phase 3 (Final Project)**: Professional AI integration as a research amplifier

**Why does AI literacy matter?** Because AI isn't going away – it's becoming the standard research accelerator across academia and industry. The difference between those who thrive and those who struggle in the next decade won't be whether they use AI, but HOW they use it. By building algorithms from scratch first, you'll develop the expertise to verify AI outputs, catch subtle errors, and push AI beyond its training to solve novel problems. You'll learn to recognize when AI is confidently wrong (it often is!), how to fact-check its mathematical derivations, spot nonsensical code that "looks right," and debug AI-generated algorithms that almost-but-don't-quite work. You'll understand when AI genuinely helps (automating boilerplate, exploring variations, catching bugs) versus when it hinders (replacing understanding with copy-paste, generating plausible-but-wrong code).

Most importantly, you'll understand why AI makes human expertise MORE valuable, not less. When everyone has access to the same AI tools, the differentiator becomes who can verify outputs, catch errors, ask better questions, and push beyond what AI already knows. Critical thinking and deep understanding are no longer optional – they're mandatory for doing anything meaningful. You'll develop the ability to use AI as a sophisticated assistant rather than a crutch – the difference between someone who asks "write me an MCMC sampler" and someone who says "help me optimize my Metropolis-Hastings acceptance ratio for this specific posterior geometry."

## What This Course Isn't

This isn't a typical survey course where you learn to use pre-packaged libraries like scikit-learn, TensorFlow, astropy, or MESA. You won't spend the semester calling `model.fit()` and `model.predict()` without understanding what happens under the hood, or using `scipy.integrate` without knowing how the algorithms work. While these tools are powerful and you'll use them professionally, this course takes a different approach: you'll build your own versions from scratch.

Every algorithm you implement will solve real astrophysical problems. By constructing these tools yourself, you'll understand their strengths, limitations, and failure modes through direct experience. When you eventually use professional libraries, you'll know exactly what's happening under the hood – and more importantly, you'll know when and why they might fail.

But here's the real transformation: after building neural networks from scratch, you'll have no fear diving into PyTorch's source code. After implementing your own MCMC sampler, you'll confidently modify emcee for your research needs. You'll be able to read papers, understand the algorithms, and implement them yourself. When a package doesn't quite do what you need, you won't search for alternatives or abandon your research question – you'll extend it or write your own.

**You're becoming a tool-builder, not just a tool-user.** This is the difference between a computational scientist and someone who depends on existing packages. Tool-users are limited by what exists; tool-builders create what's needed. In real research, the problems you'll tackle won't have pre-built solutions. The ability to implement algorithms from first principles – to translate physics into code – is what enables breakthrough science.

This approach is rare in modern curricula, where the pressure to "cover more material" often means sacrificing depth for breadth. But consider this: the scientists who revolutionized our field didn't just use existing tools – they built new ones. When Metropolis needed to understand neutron diffusion, he invented Monte Carlo methods. When astronomers needed to analyze LIGO data, they developed new statistical frameworks. The Rubin Observatory team is creating entirely new algorithms to process 20 terabytes of data per night and detect millions of transients in real-time – tasks no existing software could handle. Your future breakthrough might not come from applying existing methods more cleverly, but from building something entirely new. This course is strategically designed to give you that capability – not just for today's challenges, but for problems we haven't even imagined yet.

We won't cover every ML algorithm, statistical method, or astrophysical process – that's impossible and, frankly, pointless. Instead, you'll develop something far more valuable: the ability to teach yourself any computational method you encounter and translate physics into code. By implementing basic stellar properties (mass-luminosity relations), N-body dynamics, and radiative transfer from first principles, you'll learn the universal pattern: identify the governing equations, choose appropriate numerical methods, handle edge cases, and validate results. Once you can transform $F=ma$ into stable orbital integrators and radiative transfer equations into Monte Carlo simulations, you can tackle any physical system. This first-principles understanding, combined with modern AI tools, transforms how you learn: AI becomes your knowledgeable companion for exploring new territories, not a crutch to avoid thinking.

The goal isn't to make you memorize algorithms; it's to ignite your curiosity and remove your fear of the unknown. When you encounter an exciting new method in a paper, you won't think "that looksway too complicated" – you'll think "I want to try that!" And you'll have the skills to do it: understanding the mathematics, implementing the core ideas yourself, and using AI strategically to handle boilerplate while you focus on the novel concepts. This course plants the seeds for a lifetime of fearless, joyful learning in computational science and astrophysics.

## The Learning Journey: Three Parts, Three AI Phases

### Part 1: Foundation Building (Projects 1-3)

*AI Scaffolding Phase 1: Struggle first, AI for debugging only after 30 minutes.*

You start with stellar physics because it's conceptually accessible – everyone intuitively understands that hot things glow and massive things attract. Implementing a `Star` class teaches object-oriented thinking naturally. A star has properties (mass, temperature, luminosity) and methods (evolve, radiate, calculate_lifetime). This makes OOP concrete rather than abstract.

You'll then build a `StellarPopulation` class that manages hundreds to thousands of stars simultaneously. Here's where you'll discover the power of vectorization – a fundamental concept in scientific computing. Instead of writing loops like:

```python
for star in stars:
    star.luminosity = calculate_luminosity(star.mass)
```

You'll learn to think in arrays:

```python
luminosities = stellar_constant * masses**3.5  # Main sequence relation, all stars at once!
```

This single line replaces thousands of function calls. Your code will run *much* faster using NumPy's vectorized operations. This isn't just about speed—vectorized thinking changes how you approach problems. Instead of "for each particle, calculate force," you'll think "calculate all forces simultaneously as matrix operations." This mental shift is essential for everything that follows: Monte Carlo simulations, neural network operations, and JAX's array programming paradigm all require this vectorized mindset.

N-body dynamics becomes your introduction to numerical methods (Project 2). The physics is simple ($F = GM m/r^2$) but you can't solve it analytically for $N>2$. You'll discover firsthand why algorithm choices matter when your solar system flies apart using Euler integration but remains stable with Verlet and Leapfrog. You'll also implement Monte Carlo sampling methods to generate initial mass distributions (sampling from the Initial Mass Function), positions, and velocities for your star clusters.

**Monte Carlo Radiative Transfer** (Project 3) introduces probabilistic thinking at scale. Radiative transfer – the physics of how light propagates through and interacts with matter via absorption, emission, and scattering – is fundamental to astrophysics because everything we observe has been processed through these interactions. It determines how stellar light escapes atmospheres, how dust clouds obscure and redden starlight, and how the early universe's radiation reached us across cosmic time. Monte Carlo makes this problem tractable by simulating individual photon packets as they scatter, absorb, and re-emit through dusty media. By tracking millions of random photon paths, you'll predict what telescopes observe when looking through cosmic dust. You'll explore how dust affects observations of your stellar populations from Project 1, learning why astronomers must account for extinction and reddening when interpreting any astronomical data.

Monte Carlo methods serve as the perfect bridge between deterministic and statistical paradigms. You're using random sampling to solve problems that would be intractable otherwise – computing complex integrals, simulating photon transport, or calculating optical depths through inhomogeneous media. You're still solving physics problems, but now through statistical approximation rather than exact calculation.

**Your first *"aha!"* moment**: When you see the Central Limit Theorem emerge naturally from your Monte Carlo simulations – no matter what distribution you sample from, the mean converges to a Gaussian. This isn't just theory; you'll watch it happen in real-time through your own code.

### Part 2: Statistical Methods & Advanced Techniques (Projects 4-5)

*AI Scaffolding Phase 2: Strategic AI use after baseline implementation works.*

Each project includes just-in-time mathematical tutorials covering exactly the concepts needed – you don't need to be a mathematician before starting, just willing to learn as you build.

**Bayesian Inference and MCMC** (Project 4) represents a conceptual turning point in the course – the bridge from classical methods to modern machine learning. **Bayesian inference** flips traditional statistics: instead of asking *"what's the probability of this data given my model?"* you ask *"what's the probability of my model given this data?"* This is formalized in **Bayes' theorem**:

$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$

where $P(\theta|D)$ is the **posterior** (*what we want* – probability of our model parameters given the data), $P(D|\theta)$ is the **likelihood** (probability of observing this data if our model is true), $P(\theta)$ is the **prior** (our beliefs before seeing the data), and $P(D)$ is the **evidence** (normalization constant).

This is a **parametric method** – you assume a specific functional form for your model (e.g., linear, exponential, power law) with fixed parameters to estimate. You're learning the values of these parameters, not the function itself. In this project, you'll literally model the Universe itself – specifically its expansion history – using Type Ia supernovae to measure cosmological parameters ($Ω_m,~ h$) that determine the Universe's composition and expansion rate. The revolutionary insight is the prior – you can mathematically encode what you believe is (physically) reasonable before seeing any data. These beliefs might be wrong! Often that's fine since strong data will override weak priors. But when data is sparse (like it always is in astronomy), priors help constrain solutions to physically plausible regions.

You'll implement **MCMC (Markov Chain Monte Carlo)** to explore these probability distributions. Imagine a random walker that spends more time in high-probability regions, eventually mapping out the parameter landscape once properly converged. You'll also implement **Hamiltonian Monte Carlo (HMC)**, cleverly refactoring your **Leapfrog integrator** from Project 2 – the same algorithm that kept your star clusters stable now helps you sample parameter space more efficiently for many high-dimensional problems.

While powerful for parameter estimation, parametric methods require you to know your model's functional form in advance. What if you don't know whether your data follows a linear, exponential, or power-law relationship? This limitation motivates our transition to non-parametric methods.

**Gaussian Processes** (Project 5) bridge classical statistics and modern ML through **non-parametric methods**. Unlike parametric approaches where you assume a fixed functional form (linear, exponential, etc.), GPs learn the function itself directly from data. They're still Bayesian but now you're *learning* functions, not parameters. Think of GPs as defining a probability distribution over all functions consistent with your chosen **kernel** (covariance function). The **kernel** encodes your assumptions about function smoothness – a **squared exponential kernel** assumes infinitely differentiable functions, while a **Matérn kernel** allows rougher functions. Instead of fitting a single curve to your data, you're modeling an infinite-dimensional space of smooth functions that could explain it, weighted by their probability. The model complexity grows with your data – more data points mean more flexibility in the function space. This lets you quantify uncertainty everywhere: you'll know not just the predicted value but also how confident you should be in that prediction based on data density and kernel properties.

But GPs have a critical limitation: they scale with O(n³) computational complexity and O(n²) memory requirements due to matrix operations, making them impractical for large datasets (even a few thousand points can overwhelm memory). This computational barrier motivates neural networks in your final project – another non-parametric approach that can handle millions of data points efficiently.

As part of Project 5, you'll implement GPs as the main component, applying them to emulate your N-body simulations – *learning* to predict gravitational dynamics without running expensive simulations. During class, we'll also guide you through a JAX conversion of your N-body code – a straightforward educational exercise to learn JAX syntax, JIT compilation, and vmap. This hands-on introduction to JAX prepares you for the final project without overwhelming you with too many new concepts at once.

Each project now requires extensions where you ask *"what if?"* – what if we vary the stellar initial mass function (IMF)? What if we use different sampling strategies? This mirrors real research where the interesting discoveries come from exploring beyond the minimum requirements.

### Part 3: Modern Machine Learning (Final Project)

*AI Scaffolding Phase 3: Professional AI integration as productivity multiplier (begins after Project 5 submission).*

The culmination is your **final project**: building a **neural network** from scratch using JAX. Where MCMC gave us parametric Bayesian inference and GPs showed us non-parametric function learning, neural networks represent the ultimate flexibility – **deep learning** through multiple **layers** of non-linear transformations. They're called "neural" networks because they're loosely inspired by biological neurons: each artificial **neuron** receives inputs, applies **weights** and a non-linear **activation function**, and passes signals forward, mimicking how brain neurons fire when sufficiently stimulated.

A **layer** is a collection of neurons that process information in parallel – the **input layer** receives your data, **hidden layers** learn increasingly abstract representations, and the **output layer** produces predictions. Deep learning means using neural networks with multiple hidden layers, where each layer learns increasingly abstract representations of your data. The "deep" refers to this layered architecture that can automatically discover complex patterns without manual feature engineering. For instance, in analyzing your N-body simulations, the first layer might learn basic position-velocity relationships, middle layers could discover conservation laws, and deeper layers might uncover complex dynamical patterns.

Neural networks share GPs' non-parametric nature but overcome their computational limitations through **stochastic gradient descent** – instead of inverting massive matrices, you update parameters iteratively using small batches of data. This makes them scalable to millions of data points and parameters.

You'll receive the project description on *November 17* (while finishing Project 5) so you can start planning or even begin work if you choose. Phase 3 officially begins once you submit Project 5, giving you full access to professional AI integration capabilities. You'll implement forward propagation and training loops manually, define **loss functions** and optimization steps explicitly, while leveraging JAX's **autodiff** for gradient computation. This removes the black box mystique – neural networks are clever applications of calculus and linear algebra you already understand, and you'll see exactly how **gradients** flow backward through your network even though JAX handles the differentiation.

**Why JAX?** Developed by Google, it's the framework of choice for cutting-edge numerical computing and ML research. JAX offers three superpowers: (1) JIT (just-in-time) compilation makes your code run orders of magnitude faster, (2) vmap lets you parallelize operations effortlessly – perfect for running hundreds of N-body simulations simultaneously in Project 5, and (3) functional automatic differentiation with XLA compilation – essential for neural network training in your final project. You'll first use JAX for speed and parallelization (Project 5), then harness autodiff for deep learning (Final Project). This staged approach lets you learn JAX incrementally rather than all at once. Learning JAX makes you industry-ready while keeping you at the research frontier.

You'll apply neural networks to your N-body simulations for model discovery – finding hidden relationships and patterns in gravitational dynamics that you didn't explicitly program. Your simulations generate training data, neural networks learn the underlying physics, and suddenly you can discover governing equations or predict long-term evolution without running expensive simulations.

**Your final *"aha!"* moment**: When your neural network learns patterns you didn't explicitly program – perhaps discovering a relationship in your N-body simulations you hadn't noticed, or finding an optimal sampling strategy for your MCRT code. You'll realize you've built something that can discover things you don't know.

By the end of Part 3, that initial feeling of *"this is actually fun"* has evolved into genuine research capability. You're ready to implement any algorithm from a paper, combine classical and modern methods creatively, and contribute to the field.

## Why This Progression Works

1. **Each topic motivates the next**: Numerical integration struggles motivate Monte Carlo. Monte Carlo motivates statistics. Statistics motivates ML. Your frustrations become the seeds of insight.

2. **Complexity ramps gradually**: Start with $F=ma$, end with neural networks, but each step is manageable. You're never asked to take multiple conceptual leaps simultaneously.

3. **Real astrophysics throughout**: Every algorithm solves actual astronomy problems. You're not learning abstract methods – you're building tools astronomers use daily.

4. **Modern skills emerge from fundamentals**: By the end, you understand what JAX and modern tools do under the hood because you've built their components yourself.

5. **AI scaffolding matches skill development**: As you become more competent, AI transitions from emergency help (Phase 1) to strategic tool (Phase 2) to productivity multiplier (Phase 3).

## What You'll Gain

By the end of this course:

- You'll have built a portfolio of working astronomical software that solves real problems.
- You'll understand methods from classical mechanics to neural networks – not just how to use them, but why they work.
- You'll be fluent in modern tools used at Google, DeepMind, and leading research institutions.
- You'll be able to read research papers and implement new methods.
- You'll think computationally about physical problems while maintaining physical intuition about computational results.
- You'll know when and how to effectively leverage AI as a research tool.

## Assessment Philosophy

Your learning is evaluated through:

- **Projects (50%)**: Five core implementations demonstrating understanding
- **Growth Memos (10%)**: Reflection on learning and challenges with each project
- **Final Project (25%)**: ~4.5-week capstone synthesizing all course concepts  
- **Technical Growth Synthesis (5%)**: Reflective synthesis demonstrating your evolution as a computational scientist
- **Participation (10%)**: Engagement in pair programming and discussions

Students receive the Final Project description November 17, and the one-week overlap with Project 5 lets you see connections between the JAX refactoring and neural network implementation.

This course teaches that computational astrophysics isn't only about computers and astrophysics – it's about *thinking*. How do we translate physical understanding into algorithms? How do we diagnose when those algorithms fail? How do we improve them? How do we know when to trust our results?

The same mathematical structures appear everywhere: calculus and optimization, linear algebra, probability theory. A small set of fundamental ideas powers everything from stellar dynamics to radiative transfer to deep learning. This unity reveals the deep elegance underlying computational science and modern astronomy.

---

:::{admonition} Why This Matters: The AI Revolution Connection
:class: important

**_Why learn to build these from scratch when I can just use ChatGPT?_**

Because understanding the internals makes you irreplaceable in an AI-saturated world.

The neural networks you'll build in the final project share the same fundamental architecture as ChatGPT, Claude, and other large language models (LLMs) – just scaled up by factors of millions.

**What's the same:**

- Forward propagation through layers (you'll implement this)
- Backpropagation for learning (you'll use JAX's autodiff)
- Gradient descent optimization (you'll code this)
- The core mathematical operations (matrix multiplication, non-linear activations)

**What's different:**

- Scale: Your network might have ~1000 parameters; GPT-4 reportedly has ~1.7 trillion
- Data: You'll use N-body simulations; LLMs train on internet-scale text
- Architecture: You'll build feedforward networks; LLMs use transformers

**The critical insight**: Once you understand how a 3-layer network learns to predict gravitational dynamics, you understand the core principles behind AI systems worth billions of dollars. The mathematics is identical – matrix multiplications, non-linear activations, gradient updates. The revolution isn't in fundamentally new math; it's in scale, data, and clever architectures.

**This is why the glass-box approach matters**: When you've built neural networks from scratch, you'll understand why LLMs hallucinate (they're doing interpolation in high-dimensional space without physical constraints), why they can't truly reason about novel problems (they lack the causal models you'll implement in your physics simulations), and why domain expertise remains crucial (knowing what outputs are physically possible vs. merely plausible).

**The bottom line**: In a world where anyone can prompt an AI, the scarce resource becomes people who know when the AI is wrong, why it failed, and how to fix it. This course makes you one of those people. You're not just learning to use AI tools – you're learning to understand, evaluate, and eventually improve them.
:::

---

## Glossary of Key Terms

**Activation function**: Non-linear function applied to a neuron's weighted inputs (e.g., ReLU, sigmoid, tanh) that enables neural networks to learn non-linear patterns.

**Autodiff (Automatic differentiation)**: Computational technique to calculate exact derivatives of functions programmatically, essential for training neural networks efficiently.

**Bayesian inference**: Statistical approach that updates probability beliefs about parameters based on observed data, combining prior knowledge with new evidence.

**Deep learning**: Machine learning using neural networks with multiple hidden layers, capable of learning hierarchical representations automatically.

**Evidence**: Normalization constant P(D) in Bayes' theorem, ensures posterior probabilities sum to 1.

**Gradients**: Partial derivatives indicating the direction and rate of change of a function, used to optimize neural network parameters.

**Hamiltonian Monte Carlo (HMC)**: Advanced MCMC method using Hamiltonian dynamics to explore parameter space more efficiently than random walk methods.

**Hidden layers**: Intermediate layers between input and output in a neural network where abstract feature learning occurs.

**Input layer**: First layer of a neural network that receives the raw data features.

**Kernel**: Function defining the covariance between points in Gaussian Processes, encoding assumptions about function smoothness and structure.

**Layer**: Collection of neurons processing information in parallel within a neural network.

**Leapfrog integrator**: Symplectic integration method that conserves energy in dynamical systems, used in both N-body simulations and HMC.

**Likelihood**: P(D|θ) - probability of observing data given specific parameter values in a model.

**Loss function**: Mathematical function measuring prediction error that neural networks minimize during training (e.g., mean squared error, cross-entropy).

**Matérn kernel**: GP kernel allowing control over function smoothness, more flexible than squared exponential for modeling rougher functions.

**MCMC (Markov Chain Monte Carlo)**: Class of algorithms for sampling from probability distributions by constructing a Markov chain with the desired distribution as its equilibrium.

**Neural network**: Computational model inspired by biological neurons, composed of interconnected layers that learn to map inputs to outputs.

**Neuron**: Basic computational unit receiving weighted inputs, applying an activation function, and producing an output signal.

**Non-parametric methods**: Statistical approaches where model complexity grows with data size, not limited to fixed functional forms.

**Output layer**: Final layer of a neural network producing the predictions or classifications.

**Parametric methods**: Statistical approaches assuming a specific functional form with fixed number of parameters to estimate.

**Posterior**: P(θ|D) - probability distribution of parameters after observing data, combining prior beliefs with evidence.

**Prior**: P(θ) - probability distribution encoding beliefs about parameters before observing data.

**Squared exponential kernel**: Smooth GP kernel assuming infinitely differentiable functions, produces very smooth interpolations.

**Stochastic gradient descent**: Optimization algorithm updating parameters using gradients computed on random subsets (batches) of data.

**Weights**: Learnable parameters in neural networks determining the strength of connections between neurons.

---

**Ready to begin this journey? Let's build the universe, one algorithm at a time.**