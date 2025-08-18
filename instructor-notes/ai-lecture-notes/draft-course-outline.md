Of course. I have updated the ASTR 596 lecture plan to integrate specific, relevant chapters and sections from Deisenroth, Faisal, & Ong's *Mathematics for Machine Learning* (MML) and Ting's *Statistical Machine Learning for Astronomy* (SMLA).

Here is the revised plan with detailed explanations for each reading assignment, designed to align with the course's "glass box" pedagogical approach.

***

### **Phase 1: Computational Physics Foundations**

This phase remains focused on foundational programming and physics. The primary readings will be from the course's JupyterBook on Python and numerical methods.

#### **Weeks 1-4**
*No changes to the plan for these weeks.* The focus is on Python, OOP, N-body dynamics, and basic Monte Carlo sampling, which are prerequisites for the statistical and machine learning topics to come.

---

### **Phase 2: The Bridge to Statistical Thinking**

This phase introduces the first machine learning concepts. The readings are designed to build your understanding from pure mathematics (MML) and then connect it to astronomical applications (SMLA).

#### **Week 5 (Sept 26): Monte Carlo Radiative Transfer**
* **Topic:** Simulating photon transport through dusty media.
* **Learning Objectives:**
    * Understand the physics of radiative transfer: absorption, scattering, and optical depth.
    * Implement the core Monte Carlo Radiative Transfer (MCRT) algorithm from scratch.
    * Track individual photon packets as they travel, interact, and escape a medium.
* **Required Pre-Class Reading:**
    * **SMLA:** Chapter on **Forward Modeling & Simulation-Based Inference**. This chapter introduces how complex physical simulations like MCRT are used to generate data for subsequent statistical analysis.
    * **JupyterBook:** *Radiative Transfer, Ch. 1* (Topics: Beer's Law, mean free path, isotropic vs. anisotropic scattering, phase functions).
* **Reading Rationale:**
    The reading from **SMLA** provides the crucial "why" for this project. It frames your MCRT code not just as a physics simulation, but as the engine for **simulation-based inference**—a modern statistical technique. This connects your work to the broader goal of comparing models to data, setting the stage for the following projects. The **JupyterBook** provides the specific physics equations you'll need to implement.

#### **Week 6 (Oct 3): Synthetic Observations from MCRT**
*No changes to readings. The focus is on implementing the concepts from the previous week's readings.*

#### **Week 7 (Oct 10): Linear Regression from First Principles**
* **Topic:** Building a linear regression model from scratch.
* **Learning Objectives:**
    * Derive and implement the Normal Equations `(X^T X)β = X^T y` to solve for model parameters.
    * Implement a gradient descent optimizer from scratch to solve the same problem iteratively.
    * Apply the model to recover physical parameters from the MCRT synthetic data.
* **Required Pre-Class Reading:**
    * **MML:** Part I, Chapters 2 (Linear Algebra) & 5 (Vector Calculus).
    * **MML:** Part II, Chapter 9.1-9.2 (Linear Regression, Problem Formulation).
    * **SMLA:** Chapter on **Regression**.
* **Reading Rationale:**
    This combination is central to the "glass box" philosophy. You'll read the chapters on **Linear Algebra** and **Vector Calculus** from **MML** to understand the fundamental mathematical machinery required to even formulate the problem. Then, you'll read the beginning of the **Linear Regression** chapter in **MML** to see how that math is used to derive the Normal Equations and the gradient descent update rule. Finally, the **SMLA** chapter on **Regression** provides the astronomical context, showing how these models are applied to real data and introducing the types of problems you'll encounter in your research.

#### **Week 8 (Oct 17): Advanced Regression & Model Selection**
* **Topic:** Regularization, cross-validation, and model comparison.
* **Learning Objectives:**
    * Understand and implement regularization (Ridge/L2) to prevent overfitting.
    * Use cross-validation to assess model performance robustly.
    * Analyze parameter degeneracies and calculate confidence intervals for your estimates.
* **Required Pre-Class Reading:**
    * **MML:** Chapter 7 (Continuous Optimization, focusing on Gradient Descent).
    * **SMLA:** Chapter on **Regression** (focusing on sections covering regularization, cross-validation, and the bias-variance tradeoff).
* **Reading Rationale:**
    You'll dive into the **Continuous Optimization** chapter of **MML** to get a deep understanding of *why* gradient descent works and to see other optimization algorithms. This provides the theoretical underpinning for your implementation. The sections on **regularization and cross-validation** in **SMLA** are purely practical, explaining the techniques astronomers use to build robust and trustworthy models that don't just memorize noise.

---

### **Phase 3: Advanced Statistical Inference**

Here, you will tackle the core of modern statistical methods in astrophysics. The readings will guide you through the theory and implementation of Bayesian inference and Gaussian Processes.

#### **Week 9 (Oct 24): Bayesian Inference Foundations**
* **Topic:** The Bayesian framework for statistical inference.
* **Learning Objectives:**
    * Understand the components of Bayes' Theorem: prior, likelihood, posterior, and evidence.
    * Formulate a physical problem in Bayesian terms by defining priors and a likelihood function.
    * Appreciate the philosophical difference between Bayesian and frequentist inference.
* **Required Pre-Class Reading:**
    * **MML:** Chapter 6 (Probability and Distributions).
    * **SMLA:** Chapter on **Bayesian Inference**.
* **Reading Rationale:**
    The **MML** chapter on **Probability** is essential background, ensuring you are comfortable with the mathematical language of distributions (like Gaussians) that form the building blocks of Bayesian models. The **SMLA** chapter on **Bayesian Inference** is the main event. It will introduce you to the paradigm shift of Bayesian thinking and walk you through how to construct the key components—the prior and the likelihood—using astronomical examples.

#### **Week 10 (Oct 31): Markov Chain Monte Carlo (MCMC)**
* **Topic:** Implementing the Metropolis-Hastings MCMC algorithm.
* **Learning Objectives:**
    * Implement the Metropolis-Hastings algorithm from scratch.
    * Understand the roles of the proposal distribution and acceptance criteria.
    * Run multiple chains and assess their convergence.
* **Required Pre-Class Reading:**
    * **SMLA:** Chapter on **Markov Chain Monte Carlo (MCMC)**.
* **Reading Rationale:**
    This week is entirely focused on implementation, so the reading is singular and practical. The **SMLA** chapter on **MCMC** will explain the theory behind *why* MCMC works and provide the specific pseudocode and practical advice needed to implement the Metropolis-Hastings algorithm effectively. You will directly translate the concepts from this chapter into your Python code.

#### **Week 11 (Nov 7): Gaussian Processes & Kernel Methods**
* **Topic:** Introduction to Gaussian Processes (GPs) for non-parametric regression.
* **Learning Objectives:**
    * Understand GPs as a "probability distribution over functions".
    * Implement common kernel functions (e.g., RBF, Matérn) from scratch.
    * Optimize GP hyperparameters by maximizing the marginal likelihood.
* **Required Pre-Class Reading:**
    * **MML:** Chapter 6.3 (Gaussian Distribution) & Appendix B (Multivariate Gaussian).
    * **SMLA:** Chapter on **Gaussian Processes**.
* **Reading Rationale:**
    To understand Gaussian Processes, you must first have a rock-solid understanding of the multivariate Gaussian distribution. The readings from **MML** provide this deep mathematical foundation. With that in hand, the **SMLA** chapter on **Gaussian Processes** will show you how this mathematical object can be cleverly extended to perform regression, providing a powerful and flexible tool for modeling complex, non-linear relationships in astronomical data.

#### **Week 12 (Nov 14): Advanced GP Applications**
*No new readings. The focus is on implementing the concepts from the previous week's readings to model stellar extinction.*

---

### **Phase 4: Modern Frameworks & Research Integration**

The final phase brings you to the research frontier. The readings will cover the fundamental theory of neural networks, preparing you to build them yourself in JAX.

#### **Week 13 (Nov 21): Neural Networks from JAX Fundamentals**
* **Topic:** Building a neural network from scratch using only JAX arrays.
* **Learning Objectives:**
    * Understand the architecture of a feedforward neural network.
    * Implement forward propagation, backpropagation, and gradient descent using JAX.
    * Appreciate JAX's core transformations (`grad`, `jit`, `vmap`) through manual application.
* **Required Pre-Class Reading:**
    * **MML:** Chapter 5 (Vector Calculus, focusing on the Chain Rule).
    * **SMLA:** Chapter on **Neural Networks**.
* **Reading Rationale:**
    Backpropagation is nothing more than a clever, recursive application of the chain rule from calculus. A thorough review of **Vector Calculus** from **MML** is the key to truly understanding what happens "under the hood" when a neural network learns. The **SMLA** chapter on **Neural Networks** will then provide the architectural overview, explaining concepts like layers, activation functions, and loss functions in the context of astronomical problems, such as classifying galaxies or estimating stellar parameters.

#### **Week 14 (Nov 28): THANKSGIVING - NO CLASS**

#### **Week 15 (Dec 5): JAX Ecosystem & Research Integration**
*No new readings. The focus is on transitioning your from-scratch code to professional frameworks (Flax/Optax) and working on the final research project.*

#### **Week 16 & Finals Week**
*No new readings. The focus is on the final project workshop and presentations.*