# ASTR 596: Final Project Guide

## **Neural Networks for Astrophysical Discovery**

### **Overview**

The final project is your opportunity to synthesize everything you've learned by tackling a novel scientific question using neural networks. You'll extend one of your previous projects (P1-P6), refactor it to JAX, and apply neural network methods to solve a problem that would be difficult or impossible with classical approaches alone.

**Key Dates:**

- **Nov 17:** Project assigned, begin planning
- **Nov 21:** Proposal due (2 pages)
- **Dec 5:** Progress report due (1 page + preliminary results)
- **Dec 11:** Technical Growth Synthesis due
- **Dec 17/18:** Final presentations (10 minutes)
- **Dec 18:** Final submission (code + 8-12 page report)

## Project Structure

### Phase 1: Selection & Refactoring (Week 1)

Choose one of your previous projects and identify a scientific question that:

- Extends beyond the original project scope
- Benefits from neural network approaches
- Cannot be easily solved with classical methods alone

Then refactor your existing code to JAX:

- Convert NumPy operations to JAX
- Implement automatic differentiation where beneficial
- Prepare for GPU acceleration
- Maintain modular structure

### Phase 2: Neural Network Implementation (Week 2)

Implement your neural network solution with two components:

1. **From Scratch:** Build the core NN architecture manually (forward pass, backprop)
2. **JAX Ecosystem:** Use Equinox/Flax for production implementation

### Phase 3: Science & Analysis (Week 3)

- Run experiments and generate results
- Compare NN approach to classical methods
- Analyze what the network learned
- Prepare visualizations and presentation

## Project Ideas by Previous Project

### Extending Project 1: Stellar Physics

**Classical Approach:** HR diagram analysis, stellar classification

**NN Extension Ideas:**

- **Stellar Parameter Prediction:** Train NN to predict Teff, log(g), [Fe/H] from spectra
- **Evolutionary Track Interpolation:** NN to predict stellar evolution between computed models
- **Variable Star Classification:** Time-series classification of light curves
- **Spectral Synthesis:** Generate synthetic spectra from stellar parameters

**Why NNs?** Non-linear relationships in high-dimensional spectral data, pattern recognition in time series

### Extending Project 2: N-Body Dynamics

**Classical Approach:** Direct integration of gravitational forces

**NN Extension Ideas:**

- **Chaos Prediction:** Predict long-term stability of multi-body systems
- **Fast Force Approximation:** NN to approximate expensive force calculations
- **Orbit Classification:** Classify orbital families in galactic potentials
- **Missing Mass Inference:** Infer dark matter distribution from visible orbits

**Why NNs?** Speed up expensive calculations, find patterns in chaotic dynamics, inverse problems

### Extending Project 3: Regression Fundamentals

**Classical Approach:** Linear/polynomial regression, basic optimization

**NN Extension Ideas:**

- **Deep Regression Networks:** Multi-layer networks for complex relationships
- **Uncertainty Quantification:** Bayesian neural networks for error estimates
- **Feature Learning:** Automatic feature extraction from raw data
- **Transfer Learning:** Pre-train on simulations, fine-tune on observations

**Why NNs?** Capture non-linear relationships, automatic feature engineering, uncertainty estimation

### Extending Project 4: Monte Carlo Radiative Transfer

**Classical Approach:** Photon packet propagation through medium

**NN Extension Ideas:**

- **Emulator Networks:** NN to approximate expensive MCRT calculations
- **Inverse RT:** Infer medium properties from observed spectra
- **Acceleration Schemes:** NN to importance sample photon paths
- **Image-to-Image Translation:** Map observations to physical parameters

**Why NNs?** Orders of magnitude speedup, solve inverse problems, learn from simulations

### Extending Project 5: Bayesian/MCMC

**Classical Approach:** Metropolis-Hastings, parameter estimation

**NN Extension Ideas:**

- **Normalizing Flows:** Learn complex posterior distributions
- **Likelihood-Free Inference:** Neural posterior estimation
- **Proposal Networks:** Learn optimal MCMC proposal distributions
- **Variational Inference:** Approximate posteriors with NNs

**Why NNs?** Handle high-dimensional posteriors, accelerate inference, avoid likelihood calculations

### Extending Project 6: Gaussian Processes

**Classical Approach:** Kernel-based regression, hyperparameter optimization
**NN Extension Ideas:**

- **Deep Kernel Learning:** Learn kernel functions with NNs
- **Neural Process Models:** Combine GP flexibility with NN scalability
- **Attention Mechanisms:** Self-attention for irregular time series
- **Meta-Learning:** Learn to learn from few examples

**Why NNs?** Scale to large datasets, learn complex kernels, handle irregular sampling

## Technical Requirements

### Core Implementation Requirements

1. **JAX Refactoring**
   - Convert core functions to JAX
   - Use `jit` compilation where appropriate
   - Implement vectorized operations
   - Demonstrate speedup over NumPy version

2. **Neural Network From Scratch**

   ```python
   class NeuralNetwork:
       def __init__(self, layers):
           self.weights = self.initialize_weights(layers)
       
       def forward(self, x):
           # Implement forward pass
           
       def backward(self, x, y, learning_rate):
           # Implement backpropagation
           
       def train(self, X, y, epochs):
           # Training loop
   ```

3. **JAX Ecosystem Implementation**
   - Use Equinox or Flax for model definition
   - Optax for optimization
   - Proper train/validation/test splits
   - Implement early stopping and regularization

### Scientific Requirements

1. **Hypothesis:** Clear statement of what you're investigating
2. **Baseline:** Compare to non-NN approach from original project
3. **Validation:** Demonstrate correctness on known solutions
4. **Analysis:** What did the network learn? Interpretability attempts
5. **Limitations:** Where does your approach fail?

### Code Structure

```
final_project/
├── src/
│   ├── __init__.py
│   ├── jax_refactor/       # JAX version of original project
│   │   ├── physics.py
│   │   └── numerics.py
│   ├── nn_from_scratch/    # Manual implementation
│   │   ├── network.py
│   │   ├── layers.py
│   │   └── optimizers.py
│   ├── nn_production/      # Equinox/Flax implementation
│   │   ├── models.py
│   │   ├── training.py
│   │   └── evaluation.py
│   └── analysis/           # Results analysis
│       ├── visualize.py
│       └── interpret.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── notebooks/              # Exploration only (not submission)
├── outputs/
│   ├── figures/
│   ├── models/            # Saved model checkpoints
│   └── metrics/           # Training histories
├── tests/
├── README.md
├── requirements.txt
├── proposal.pdf
├── progress_report.pdf
└── final_report.pdf
```

## Deliverables

### 1. Project Proposal (Nov 21) - 2 pages

**Format:** PDF submitted to Canvas

**Required Sections:**
1. **Scientific Question** (0.5 page)
   - What new question will you address?
   - Why can't classical methods solve this?
2. **Methodology** (0.75 page)
   - Which previous project are you extending?
   - What NN architecture will you use?
   - How will you validate results?
3. **Timeline** (0.25 page)
   - Week-by-week plan
4. **Success Metrics** (0.5 page)
   - How will you know if it worked?
   - What's your baseline comparison?

### 2. Progress Report (Dec 5) - 1 page + figures

**Required Elements:**

- JAX refactoring complete (show timing comparisons)
- NN from scratch implemented (show it learns something)
- Preliminary results from JAX ecosystem version
- Any blocking issues identified

### 3. Final Report (Dec 18) - 8-12 pages

**Format:** Research paper style (abstract, intro, methods, results, discussion)

**Sections:**

1. **Abstract** (200 words)

2. **Introduction** (1-2 pages)
   - Scientific motivation
   - Previous work (cite your original project)
   - Why neural networks?

3. **Methods** (3-4 pages)
   - JAX refactoring approach
   - Network architecture choices
   - Training procedure
   - Validation strategy

4. **Results** (2-3 pages)
   - Performance comparisons (classical vs NN)
   - Scientific findings
   - Computational benchmarks

5. **Discussion** (2-3 pages)
   - Interpretation of what network learned
   - Limitations and failure modes
   - Future improvements
   - Broader implications

6. **Conclusion** (0.5 page)

7. **References** (not counted in page limit)

**Figure Requirements:**

- Minimum 5 figures
- Architecture diagram
- Training curves
- Results comparison
- Scientific interpretation plots

### 4. Final Presentation (Dec 17/18) - 10 minutes

**Structure:**

- 2 min: Problem setup and motivation
- 3 min: Methods (focus on NN approach)
- 3 min: Results and comparison
- 1 min: What the network learned
- 1 min: Conclusions and future work

**Slides:** 10-12 slides maximum, emphasize visuals

## Grading Rubric (100 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **JAX Refactoring** | 15 | Correct implementation, performance gains |
| **NN From Scratch** | 20 | Working implementation, clear understanding |
| **JAX Ecosystem** | 20 | Proper use of tools, advanced features |
| **Scientific Merit** | 15 | Novel question, appropriate methods |
| **Results & Analysis** | 15 | Thorough comparison, interpretation |
| **Report Quality** | 10 | Clear writing, good figures, proper citations |
| **Presentation** | 5 | Clear, engaging, on time |

### Detailed Rubric Descriptions

#### JAX Refactoring (15 points)

- Excellent (14-15): Full refactor, significant speedup, uses advanced JAX features
- Good (11-13): Most code refactored, some speedup, basic JAX usage
- Satisfactory (8-10): Partial refactor, works but minimal optimization
- Needs Improvement (0-7): Minimal refactoring or doesn't work

#### NN From Scratch (20 points)

- Excellent (18-20): Full backprop, multiple layers, advanced features (dropout, batch norm)
- Good (14-17): Working backprop, 2+ layers, trains successfully
- Satisfactory (10-13): Basic working network, may have limitations
- Needs Improvement (0-9): Doesn't train or major implementation errors

#### JAX Ecosystem (20 points)

- Excellent (18-20): Advanced architectures, proper training pipeline, uses multiple libraries
- Good (14-17): Standard implementation, works well, uses Equinox/Flax properly
- Satisfactory (10-13): Basic implementation, works but not optimized
- Needs Improvement (0-9): Minimal use of ecosystem or doesn't work

## Scientific Merit (15 points)

- Excellent (14-15): Novel question, clear hypothesis, appropriate for NNs
- Good (11-13): Good question, reasonable approach
- Satisfactory (8-10): Adequate question but could use classical methods
- Needs Improvement (0-7): Unclear question or inappropriate methods

## Tips for Success

### Choosing Your Project

#### Good Final Projects:

- Address a clear scientific question
- Show when/why NNs outperform classical methods
- Build naturally on your previous work
- Are ambitious but achievable in 3 weeks

#### Avoid:

- Applying NNs just because you can
- Problems with analytical solutions
- Datasets too small for NNs to be beneficial
- Overly complex architectures you don't understand

### Time Management

#### Week 1 Focus:

- Days 1-2: Project selection and proposal writing
- Days 3-4: JAX refactoring
- Days 5-7: NN from scratch implementation

#### Week 2 Focus:

- Days 8-10: JAX ecosystem implementation
- Days 11-12: Training and hyperparameter tuning
- Days 13-14: Progress report and debugging

#### Week 3 Focus:

- Days 15-17: Final experiments and analysis
- Days 18-19: Report writing
- Days 20-21: Presentation preparation

### Common Pitfalls

1. **Overcomplicating:** Start simple, add complexity if time permits
2. **Poor Baselines:** Always compare to your classical implementation
3. **No Validation:** Must demonstrate correctness on known cases
4. **Black Box Syndrome:** Understand what your network is doing
5. **Last-Minute JAX:** Start refactoring early, JAX has a learning curve

### Resources

#### JAX Tutorials:

- [Official JAX Documentation](https://jax.readthedocs.io/)
- [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Thinking in JAX](https://jax.readthedocs.io/en/latest/thinking_in_jax.html)

#### Neural Network Theory:

- Deep Learning book (Goodfellow et al.) - free online
- Neural Networks and Deep Learning (Michael Nielsen) - free online
- Fast.ai course materials

#### Astrophysics ML Papers:

- "Machine Learning in Astronomy" (Baron 2019)
- "Deep Learning for Observational Cosmology" (Ntampaka+ 2019)
- Recent papers using NNs in your subfield

## Getting Help

### Technical Support:

- JAX issues: Check documentation first, then ask on Slack
- NN architecture questions: Discuss in Friday sessions
- Debugging: Use AI tutors for concept help, not code generation

### Scientific Guidance: 

- Unsure if your question is appropriate? Discuss in office hours
- Need literature references? Ask on Slack
- Want feedback on approach? Submit optional draft to instructor

**Remember:** This project should showcase your growth as a computational astrophysicist. It's not about building the most complex network—it's about demonstrating understanding of when, why, and how neural networks can advance astrophysical research.

## Final Thoughts

This project is your opportunity to:

- Demonstrate mastery of course concepts
- Explore a scientific question you're passionate about
- Build something you can show future advisors/employers
- Push yourself beyond your comfort zone

Embrace the challenge, ask for help when needed, and create something you're proud of!
