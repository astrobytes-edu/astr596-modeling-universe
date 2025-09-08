---
title: "Part 3: Taylor Series - The Bridge from Continuous to Discrete"
subtitle: "Module 1: Foundations of Discrete Computing | ASTR 596"
---

**Navigation:**
[← Part 2: Numbers Aren't Real](./02-numbers-arent-real.md) | [Synthesis & Summary →](./04-synthesis-summary.md)

## Learning Outcomes

By the end of this section, you will be able to:

- [ ] **Apply** Taylor series to **derive** numerical methods of different orders
- [ ] **Verify** error predictions empirically through numerical experiments
- [ ] **Explain** why symmetric methods achieve higher accuracy through error cancellation
- [ ] **Design** custom finite difference formulas for specific accuracy requirements
- [ ] **Recognize** when to use finite differences vs. automatic differentiation

---

## The Foundation

The Taylor series connects continuous calculus to discrete numerical methods. For a smooth function $f(x)$:

$$f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \frac{f'''(x_0)}{3!}(x-x_0)^3 + ...$$

In numerical methods, we truncate this series, introducing truncation error. The art lies in managing this error within finite precision constraints.

## From Taylor Series to Finite Differences

Let's see how Taylor series creates each finite difference formula.

### Deriving Forward Difference

Starting with Taylor expansion:

$$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + O(h^3)$$

Rearranging:

$$f'(x) = \frac{f(x+h) - f(x)}{h} - \frac{h}{2}f''(x) + O(h^2)$$

The forward difference $f'(x) \approx \frac{f(x+h) - f(x)}{h}$ has error $O(h)$.

### Deriving Central Difference  

Using both directions:

$$f(x+h) - f(x-h) = 2hf'(x) + \frac{2h^3}{6}f'''(x) + O(h^5)$$

Therefore:

$$f'(x) = \frac{f(x+h) - f(x-h)}{2h} + O(h^2)$$

The symmetry cancels all even-order terms, giving us second-order accuracy "for free"!

## Verifying Error Predictions

**Why this matters**: Taylor series isn't just abstract mathematics - it accurately predicts the errors in our numerical methods. This verification shows that our theoretical analysis matches reality, giving us confidence in our error estimates. When you implement numerical methods in your research, you can use similar tests to verify your code is working correctly.

```python
import numpy as np

def verify_taylor_predictions():
    """
    Empirically verify that truncation errors match Taylor predictions
    This builds confidence that our theory accurately describes reality
    """
    # Test function: sin(x) - we know all its derivatives analytically
    f = np.sin
    x, h = 1.0, 0.01
    
    # Forward difference actual error
    fd_approx = (f(x+h) - f(x))/h
    fd_error = fd_approx - np.cos(x)  # cos is the true derivative
    
    # Taylor predicts: error ≈ -(h/2)*f''(x) = (h/2)*sin(x)
    fd_predicted = (h/2) * np.sin(x)
    
    # Central difference actual error  
    cd_approx = (f(x+h) - f(x-h))/(2*h)
    cd_error = cd_approx - np.cos(x)
    
    # Taylor predicts: error ≈ -(h²/6)*f'''(x) = (h²/6)*cos(x)
    cd_predicted = (h**2/6) * np.cos(x)
    
    results = {
        'forward': {'actual': fd_error, 'predicted': fd_predicted},
        'central': {'actual': cd_error, 'predicted': cd_predicted}
    }
    
    print("Taylor Series Predictions vs Reality:")
    for method, errors in results.items():
        ratio = errors['actual']/errors['predicted']
        print(f"{method.capitalize():8} - Actual: {errors['actual']:.6e}, "
              f"Predicted: {errors['predicted']:.6e}, Ratio: {ratio:.3f}")
    
    return results
```

## Why Central Difference Wins

Central difference is superior because:

1. **Higher accuracy**: $O(h^2)$ vs $O(h)$ error
2. **Larger optimal $h$**: $\epsilon^{1/3}$ vs $\epsilon^{1/2}$ 
3. **More robust**: Less susceptible to round-off
4. **Symmetric**: Cancels systematic biases

## When NOT to Use Numerical Derivatives

Before implementing numerical derivatives, consider if they're necessary. **Avoid numerical derivatives when**:

1. **Analytical derivatives are available** - Always use exact derivatives when you can derive them
2. **The function is noisy** - Numerical derivatives amplify noise
3. **You need many derivatives** - Consider automatic differentiation (see below)
4. **The function is expensive** - Each derivative needs multiple evaluations

## Modern Alternative: Automatic Differentiation

For complex functions, especially in machine learning, **automatic differentiation** provides exact derivatives (to machine precision) without the errors of finite differences. Tools like JAX (which you'll use in the final project) compute derivatives by tracking operations, not by approximating with finite differences. This gives the best of both worlds: exact derivatives without manual derivation. However, finite differences remain essential for:

- Verifying automatic differentiation implementations
- Functions only available as black boxes
- Understanding numerical behavior
- Quick derivative estimates

---

## Bridge to Module 2: From Derivatives to Dynamic Systems

You've now mastered the static aspects of numerical computing - how to compute derivatives despite finite precision and how to analyze errors using Taylor series. These are the building blocks, but astrophysics is about motion and evolution.

In the next module, you'll apply these foundations to dynamic problems:

- **Root finding**: Finding equilibrium points where forces balance
- **Integration**: Computing areas, volumes, and accumulated quantities
- **ODEs**: Evolving systems forward in time

The error analysis skills and Taylor series methods you've learned here will be essential as we move from computing instantaneous rates of change to simulating the evolution of entire stellar systems.

*Next: Module 2 - Static Problems & Quadrature*