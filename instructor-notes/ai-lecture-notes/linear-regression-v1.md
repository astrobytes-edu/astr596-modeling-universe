Here are the lecture notes for Week 7, formatted for a Canvas page.

***

# **ASTR 596: Modeling the Universe**
### **Week 7: Linear Regression from First Principles**

---

## 🎯 **This Week's Goal: From Data to Insight**

Welcome to Phase 2 of the course! You've just spent weeks building a sophisticated simulation (MCRT) to *generate* realistic data. Now, we flip the script. How do we take data and work backward to infer the physical parameters that produced it?

This week, we build our first machine learning model, **Linear Regression**, from scratch. This isn't just a programming exercise; it's the foundation of almost all parameter estimation in science. We're not just going to `import sklearn`. Instead, we'll derive every equation from first principles, using linear algebra and calculus. This "glass box" approach ensures you know exactly why it works and, more importantly, how it can fail. Our goal is to take the synthetic spectra you created in Project 3 and recover the "ground truth" dust parameters you fed into the simulation.



---

## **🔬 The Anatomy of a Physical Model**

Before we dive into the math, let's define our terms. A simple model is a mathematical function that maps inputs to outputs.

* **Features ($\mathbf{x}$):** These are our inputs—the variables we believe can predict the outcome. For a simple line, the feature is just the x-coordinate. In our MCRT problem, the features could be the wavelength-dependent properties we use to predict the final flux.
* **Target ($y$):** This is the output we want to predict. For a simple line, it's the y-coordinate. For our project, it's the observed flux in a specific filter.
* **Parameters ($\boldsymbol{\beta}$):** These are the knobs of our model that we "tune" to get the best fit. For a line, $y = \beta_1 x + \beta_0$, the parameters are the slope ($\beta_1$) and the intercept ($\beta_0$). In our astronomical context, these are the physical quantities we want to recover, like dust optical depth ($\tau$) or scattering albedo ($\omega$).

Our goal is to find the optimal set of parameters $\boldsymbol{\beta}$ that makes our model's predictions, $\hat{y}$, as close as possible to the true target values, $y$.

---

## **🧮 Mathematical Formulation: Building the Model**

Let's formalize this. We'll assume a linear relationship between our features and our target. For a single data point $i$ with $D$ features, our model's prediction, $\hat{y}_i$, is a weighted sum of those features:

$$\hat{y}_i = \beta_0 x_{i,0} + \beta_1 x_{i,1} + \dots + \beta_D x_{i,D} = \sum_{j=0}^{D} \beta_j x_{i,j}$$

Here, we've included a "dummy" feature $x_{i,0} = 1$ for all data points, which allows the parameter $\beta_0$ to act as the y-intercept.

This is cumbersome to write for all $N$ data points in our dataset. We can express this entire system elegantly using linear algebra, which is why Chapter 2 of MML was required reading.

We define:
* The **target vector** $\mathbf{y} \in \mathbb{R}^N$. This is a column vector containing all our observed outcomes.
* The **parameter vector** $\boldsymbol{\beta} \in \mathbb{R}^{D+1}$. This is a column vector of the weights we want to find.
* The **feature matrix** (or **Design Matrix**) $\mathbf{X} \in \mathbb{R}^{N \times (D+1)}$. Each *row* is a single data point's features, and each *column* corresponds to a specific feature across all data points.

$$
\mathbf{y} =
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{bmatrix}
, \quad
\mathbf{X} =
\begin{bmatrix}
x_{1,0} & x_{1,1} & \dots & x_{1,D} \\
x_{2,0} & x_{2,1} & \dots & x_{2,D} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N,0} & x_{N,1} & \dots & x_{N,D}
\end{bmatrix}
, \quad
\boldsymbol{\beta} =
\begin{bmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_D
\end{bmatrix}
$$

With these definitions, our entire system of $N$ equations simplifies to a single, beautiful matrix equation:

$$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}$$

This compact notation is the key to everything that follows. It's how you'll implement the solution in `NumPy`.

---

## **📉 The Cost Function: Quantifying "Wrongness"**

How do we find the *best* $\boldsymbol{\beta}$? We need a way to measure how "wrong" our model's predictions are. We'll define a **cost function** (or loss function), $J(\boldsymbol{\beta})$, that calculates this "wrongness." A common and effective choice is the **Sum of Squared Errors (SSE)**, which is the sum of the squared differences between the true values ($y_i$) and our predicted values ($\hat{y}_i$).

$$J(\boldsymbol{\beta}) = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{N} (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2$$

Where $\mathbf{x}_i^\top$ is the $i$-th row of our design matrix $\mathbf{X}$. The goal of "learning" is simply to find the vector $\boldsymbol{\beta}$ that *minimizes* this cost function.

Using our matrix notation, the SSE can be written even more compactly. The term $(y_i - \hat{y}_i)$ is the $i$-th element of the residual vector $(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$. The sum of squares is therefore just the dot product of this vector with itself:

$$J(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Now we have a single equation that defines the "wrongness" of our model for any choice of parameters $\boldsymbol{\beta}$. How do we minimize it?

---

## **Method 1: The Analytical Solution (Normal Equations)**

From calculus, we know that a function's minimum can be found where its derivative (or gradient) is zero. This is where MML Chapter 5 on Vector Calculus becomes critical. We need to compute the gradient of our cost function $J(\boldsymbol{\beta})$ with respect to the entire vector of parameters $\boldsymbol{\beta}$, and set it to zero.

Let's first expand the cost function:
$J(\boldsymbol{\beta}) = (\mathbf{y}^\top - (\mathbf{X}\boldsymbol{\beta})^\top) (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = (\mathbf{y}^\top - \boldsymbol{\beta}^\top\mathbf{X}^\top) (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$
$J(\boldsymbol{\beta}) = \mathbf{y}^\top\mathbf{y} - \mathbf{y}^\top\mathbf{X}\boldsymbol{\beta} - \boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{y} + \boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta}$

Since $\mathbf{y}^\top\mathbf{X}\boldsymbol{\beta}$ is a scalar, its transpose is equal to itself, so $\mathbf{y}^\top\mathbf{X}\boldsymbol{\beta} = (\mathbf{y}^\top\mathbf{X}\boldsymbol{\beta})^\top = \boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{y}$. This simplifies the expression:
$J(\boldsymbol{\beta}) = \mathbf{y}^\top\mathbf{y} - 2\boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{y} + \boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta}$

Now, we take the gradient with respect to $\boldsymbol{\beta}$ (using the rules of matrix calculus from MML Ch. 5):

$$\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = -2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta}$$

To find the minimum, we set this gradient to zero:
$-2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = 0 $$ \mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y} $

This final result is known as the **Normal Equations**. It gives us a direct, analytical recipe for finding the best parameter vector $\boldsymbol{\beta}$:

$$\boldsymbol{\beta} = (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{X}^\top\mathbf{y}$$

This is a profound result. It means we can find the optimal parameters with a single calculation, involving a few matrix multiplications and one matrix inversion.

---

## **Method 2: The Iterative Solution (Gradient Descent)**

The Normal Equations are elegant, but calculating the inverse of $\mathbf{X}^\top\mathbf{X}$ can be computationally expensive or numerically unstable if the matrix is very large or ill-conditioned. An alternative is an iterative approach called **Gradient Descent**.

The idea is simple and intuitive:
1.  Start with a random guess for $\boldsymbol{\beta}$.
2.  Calculate the gradient of the cost function at that point. The gradient is a vector that points in the direction of the *steepest ascent* of the cost function.
3.  Take a small step in the *opposite* direction of the gradient (downhill).
4.  Repeat until you reach the bottom of the "cost valley."



The "small step" is controlled by a **learning rate**, $\alpha$. The update rule for each iteration $k$ is:

$$\boldsymbol{\beta}_{k+1} := \boldsymbol{\beta}_k - \alpha \nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}_k)$$

Substituting our gradient from before:

$$\boldsymbol{\beta}_{k+1} := \boldsymbol{\beta}_k - \alpha (2\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta}_k - 2\mathbf{X}^\top\mathbf{y})$$

You will implement both methods in your project. The Normal Equations will give you the exact right answer, which you can use to check if your Gradient Descent implementation is converging correctly.

---

## **🔭 Connecting to Your Project**

So, how does this relate to recovering dust properties from your MCRT simulation?
* Your **target** $y$ is the synthetic flux you calculated for a given star.
* Your **features** $\mathbf{X}$ could be any number of things you think are predictive. They could be simple, like powers of the wavelength ($\lambda, \lambda^2, \dots$), or more physically motivated quantities derived from the simulation setup.
* Your **parameters** $\boldsymbol{\beta}$ are the weights that, hopefully, correspond to physical quantities you want to recover, like optical depth $\tau$ or properties of the dust grains.

As discussed in SMLA, the art and science of regression in astronomy often involves clever **feature engineering**—choosing the right inputs $\mathbf{X}$ so that the resulting parameters $\boldsymbol{\beta}$ are physically meaningful.

## **💻 Week 7 Lab Goals**

1.  **Load Your Data:** Load the synthetic MCRT dataset you created in Project 3.
2.  **Build the Design Matrix:** Choose your features and construct the matrix $\mathbf{X}$ and target vector $\mathbf{y}$. Remember to add the column of ones to $\mathbf{X}$ for the intercept term!
3.  **Implement the Normal Equations:** Write a Python function that solves for $\boldsymbol{\beta}$ using the analytical solution. Use `numpy.linalg.inv`.
4.  **Implement Gradient Descent:** Write a second function that iteratively solves for $\boldsymbol{\beta}$. You will need to choose a learning rate $\alpha$ and a number of iterations.
5.  **Compare and Validate:** Check that your Gradient Descent implementation converges to the same answer as your Normal Equations solution. Plot your model's predictions against the true data to see how well it fits.