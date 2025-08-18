# ASTR 596: Scientific Computing with Python - Comprehensive Pedagogical Framework

## CRITICAL INSTRUCTIONS - READ FIRST
This document contains the EXACT specifications for creating and reviewing chapters for ASTR 596. Follow these instructions PRECISELY. Do not deviate from these guidelines. Every chapter must adhere to ALL requirements listed here.

**PLATFORM REQUIREMENT**: All chapters must be written for MyST Markdown/Jupyter Book 2 deployment on GitHub Pages. Use advanced MyST features to create interactive, web-native content.

## Core Mission
Transform astronomy graduate students from superficial coders into computational scientists who can:
- Implement algorithms directly from research papers
- Debug numerical instabilities in complex calculations
- Contribute to major computational projects
- Write robust, maintainable scientific software
- Create materials that serve as both initial learning resources and career-long references

## MANDATORY CHAPTER STRUCTURE

Every chapter MUST be a `.md` file with MyST frontmatter and contain these elements:

```markdown
---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
```

### Required Sections (in order):

1. **Learning Objectives** (5-8 measurable outcomes starting with action verbs)
2. **Prerequisites Check** (interactive checklist with chapter references)
3. **Chapter Overview** (3 paragraphs connecting to course trajectory)
4. **Main Content** (with ALL required active learning elements)
5. **Practice Exercises** (3-4, explicitly scaffolded)
6. **Main Takeaways (Summary)** (comprehensive narrative summary)
7. **Definitions** (alphabetical glossary of ALL technical terms)
8. **Key Takeaways** (bullet points with checkmarks)
9. **Quick Reference Tables** (organized lookup resources)
10. **Next Chapter Preview** (connection to upcoming content)

## MYST MARKDOWN REQUIREMENTS

### Interactive Code Blocks
All code examples must use MyST code-cell directives for execution:

````markdown
```{code-cell} ipython3
# This code executes in the browser
import numpy as np
data = [1, 2, 3, 4, 5]
print(f"Mean: {np.mean(data)}")
```
````

### Advanced MyST Features to Include

#### Admonitions for Active Learning
Use MyST admonitions for all pedagogical elements:

````markdown
```{admonition} 🔍 Check Your Understanding
:class: question
What error would this code produce?
```

```{admonition} Click to see answer
:class: answer, dropdown
The answer with full explanation...
```
````

#### Tabbed Content for Comparisons
Use tabs to show alternative implementations:

````markdown
````{tab-set}
```{tab-item} Naive Implementation
```python
def simple_mean(data):
    return sum(data) / len(data)
```
```

```{tab-item} Robust Implementation
```python
def robust_mean(data):
    if not data:
        raise ValueError("Empty data")
    return sum(data) / len(data)
```
```
````
````

#### Margin Content for Definitions
Place definitions in margins for easy reference:

````markdown
```{margin} **Key Term**
**Exception**: Python's way of signaling that something exceptional has happened preventing normal execution.
```
````

#### Interactive Figures with MyST
Use MyST figure directives with captions:

````markdown
```{figure} ./images/error_propagation.svg
:name: fig-error-prop
:alt: Error propagation through calculations
:align: center
:width: 80%

How a single error cascades through a computational pipeline, corrupting all downstream results.
```
````

#### Cross-References
Use MyST referencing throughout:

````markdown
As we learned in {ref}`chapter-5-functions`, functions should validate inputs.

See {numref}`fig-error-prop` for visualization of error propagation.

{doc}`../chapter-07/numpy-arrays` covers array operations in detail.
````

### Code Output Display
Show code output using MyST output directives:

````markdown
```{code-cell} ipython3
:tags: [hide-output]

result = complex_calculation()
print(result)
```

```{code-cell} ipython3
:tags: [output_scroll]

# Long output that scrolls
for i in range(100):
    print(f"Processing {i}")
```
````

## STRICT CONTENT REQUIREMENTS

### Code Complexity Limits
- **MAXIMUM 30 lines per code example** (most should be 10-20 lines)
- **ONE concept per example** - never mix multiple concepts
- **Build complexity through SEQUENCE, not single blocks**
- **Each example must be complete and runnable**
- **Use code-cell directives for ALL Python code**
- **Violating these limits requires complete restructuring**

### Prose Requirements
- **60% explanation, 40% code** - maintain this ratio throughout
- **Define EVERY technical term** when first introduced (use margin notes)
- **Bold all keywords** on first use and when central to concepts
- **Explain WHY before HOW** - motivation precedes implementation
- **Use full sentences and paragraphs** - bullets only in reference tables
- **Include real-world consequences** (disasters, retracted papers, wasted time)

### Progressive Complexity Model
Each concept must progress through these stages:
1. **Conceptual introduction** (what and why in plain language)
2. **Simplest possible example** (5-10 lines, one idea)
3. **Realistic application** (10-20 lines, practical use)
4. **Robust implementation** (15-30 lines, with validation)
5. **Performance considerations** (if applicable)

## REQUIRED ACTIVE LEARNING ELEMENTS (MYST FORMAT)

### Mandatory Components (MUST include ALL)

#### "Check Your Understanding" Boxes (minimum 3 per chapter)
````markdown
```{admonition} 🔍 Check Your Understanding
:class: question

Present a question that tests comprehension of the just-taught concept.

```{admonition} Click for Answer
:class: answer, dropdown

Complete answer explaining the reasoning with code examples if needed:

```{code-cell} ipython3
# Demonstration code for the answer
```
```
````

#### "Computational Thinking" Boxes (minimum 2 per chapter)
````markdown
```{admonition} 💡 Computational Thinking: Pattern Name
:class: important

Explain universal pattern that applies across domains. Show pattern in multiple contexts and connect to broader principles.

```{code-cell} ipython3
# Code demonstrating the pattern
```
```
````

#### "Common Bug Alert" Sections (minimum 2 per chapter)
````markdown
```{warning}
**Common Bug Alert: The Silent Exception**

Never use bare except clauses:

```{code-cell} ipython3
# BAD - hides all errors
try:
    result = risky_operation()
except:
    result = 0
```

This hides critical errors. Always catch specific exceptions.
```
````

#### "Debug This!" Challenges (minimum 1 per chapter)
````markdown
```{admonition} 🛠️ Debug This!
:class: challenge

This function has a bug. Can you find it?

```{code-cell} ipython3
def buggy_function(data):
    # Buggy implementation
    pass
```

```{admonition} Solution
:class: solution, dropdown

The bug explanation and corrected code:

```{code-cell} ipython3
def fixed_function(data):
    # Corrected implementation
    pass
```
```
```
````

#### "Performance Profile" Sections (where applicable)
````markdown
```{admonition} 📊 Performance Profile: Method Comparison
:class: seealso

```{code-cell} ipython3
import time

# Method 1 timing
start = time.time()
result1 = method1()
time1 = time.time() - start

# Method 2 timing
start = time.time()
result2 = method2()
time2 = time.time() - start

print(f"Method 1: {time1:.4f}s")
print(f"Method 2: {time2:.4f}s")
print(f"Speedup: {time1/time2:.2f}x")
```
```
````

#### "Why This Matters" Connections (minimum 2 per chapter)
````markdown
```{admonition} 🎯 Why This Matters: Real Disaster Example
:class: attention

In 1999, NASA lost the $125 million Mars Climate Orbiter because of missing unit validation. Here's what should have been done:

```{code-cell} ipython3
def combine_measurements(value1, unit1, value2, unit2):
    if unit1 != unit2:
        raise ValueError(f"Unit mismatch: {unit1} vs {unit2}")
    return value1 + value2
```
```
````

## INTERACTIVE ELEMENTS REQUIREMENTS

### Jupyter Widgets (where applicable)
````markdown
```{code-cell} ipython3
import ipywidgets as widgets
from IPython.display import display

def explore_parameters(threshold):
    result = process_data(data, threshold)
    print(f"Result with threshold {threshold}: {result}")

widgets.interact(explore_parameters, 
                threshold=widgets.FloatSlider(min=0, max=10, step=0.1, value=5))
```
````

### Interactive Plots with Plotly
````markdown
```{code-cell} ipython3
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Data'))
fig.update_layout(title='Interactive Plot', 
                  xaxis_title='X', 
                  yaxis_title='Y',
                  hovermode='x unified')
fig.show()
```
````

### Collapsible Sections for Optional Content
````markdown
```{dropdown} Advanced Topic: Numerical Stability
This optional section covers advanced numerical considerations...
```
````

## EXERCISE REQUIREMENTS (MYST FORMAT)

### Structure for EVERY Exercise:
````markdown
```{exercise} Exercise Title
:label: ex-chapter-number

**Part A: Basic Implementation (5-10 lines)**

Implement the simplest version:

```{code-cell} ipython3
# Your code here
def basic_function():
    pass
```

**Part B: Add Validation (10-15 lines)**

Enhance with error checking:

```{code-cell} ipython3
# Build on Part A
def validated_function():
    pass
```

**Part C: Complete Solution (15-25 lines)**

Production-ready implementation:

```{code-cell} ipython3
# Full implementation
def robust_function():
    pass
```
```

```{solution} ex-chapter-number
:class: dropdown

Complete solutions with explanations...
```
````

## EXAMPLE COMPLEXITY GUIDELINES FOR PYTHON TEXTBOOK

### Primary Goal: TEACH PYTHON, not astrophysics

Examples should illuminate Python concepts, not require astronomy knowledge. If students need to understand stellar physics to understand the code, the example is too complex.

### Astrophysics Examples: Keep It Simple
✅ **USE THESE TYPES**:

- Mechanics and conservation laws
- electromagnetic radiation basics
- Kepler's laws (orbital periods, semi-major axes)
- Simple gravitational calculations (F = GMm/r²)
- Basic power-law stellar physics relationships (L ∝ R²T⁴)
- Temperature conversions (Kelvin/Celsius)
- Magnitude calculations (apparent/absolute)
- Simple redshift (z = Δλ/λ)
- Basic stellar properties (mass, radius, temperature)
- Planet/star data as lists/arrays
- Distance calculations (au, parsecs)
- Simple coordinate conversions
- etc.

❌ **AVOID THESE**:

- Complex spectral line analysis
- Detailed photometry pipelines
- CCD reduction workflows
- Telescope pointing corrections
- Atmospheric extinction corrections
- Complex coordinate transformations
- Multi-band color corrections
- PSF fitting
- Image registration/stacking
- Detailed radiative transfer

### Numerical Methods: Start Basic

✅ **APPROPRIATE METHODS** (to illustrate programming concepts):
- **Bisection method**: Finding roots (e.g., when does orbit cross certain radius)
- **Rectangle/Trapezoidal rule**: Simple integration (e.g., area under light curve)
- **Euler's method**: Basic differential equations (e.g., cooling of a planet)
- **Simple Monte Carlo**: Random sampling (e.g., estimating π or stellar collision probability)
- **Linear interpolation**: Between data points
- **Basic statistics**: Mean, median, standard deviation of measurements
- **Simple finite differences**: Numerical derivatives

❌ **TOO ADVANCED** (save for specialized courses):
- Newton-Raphson method
- Runge-Kutta (including Leapfrog)
- Gaussian quadrature
- FFT and signal processing
- Matrix decomposition methods
- Optimization algorithms (gradient descent, etc.)
- Symplectic integrators
- Adaptive timestep methods

### Example Philosophy
Each example should:
1. **Teach a Python concept first** (loops, functions, classes, etc.)
2. **Use astronomy for context**, not complexity
3. **Be solvable in <30 lines of code**
4. **Require minimal physics and astrophysics knowledge** (intro college level)
5. **Build toward later examples** progressively

### Good Example Pattern
```python
# GOOD: Teaching list comprehensions with stellar data
# Simple, clear, focuses on Python syntax
star_temps = [5778, 3500, 9940, 6000, 4900]  # Kelvin
star_colors = ['yellow' if 5000 < T < 6000 else 'other' 
               for T in star_temps]

# BAD: Too much astrophysics obscures the Python
# Requires understanding of blackbody radiation, Wien's law, etc.
def planck_function(wavelength, T):
    return (2*h*c**2/wavelength**5) / (np.exp(h*c/(wavelength*k*T)) - 1)

## REFERENCE SECTIONS (MYST FORMAT)

### Main Takeaways Panel
````markdown
```{panels}
:column: col-12
:card: border-2 shadow

**Main Takeaways**
^^^
Comprehensive narrative summary (300-500 words) synthesizing key concepts into a coherent story, explaining interconnections and practical applications...
```
````

### Definitions Glossary
````markdown
```{glossary}
Exception
  Python's way of signaling that something exceptional has happened preventing normal execution.

Traceback
  Report showing the sequence of function calls leading to an error.

Validation
  Checking that external input meets requirements before processing.
```
````

### Quick Reference Tables
````markdown
```{list-table} Common Exceptions and Solutions
:header-rows: 1
:name: tab-exceptions

* - Exception
  - Meaning
  - Common Fix
* - `TypeError`
  - Wrong type
  - Check data types
* - `ValueError`
  - Invalid value
  - Validate input range
```
````

## MYST CONFIGURATION REQUIREMENTS

### Required Extensions
The `_config.yml` must include:
```yaml
sphinx:
  extra_extensions:
    - sphinx_panels
    - sphinx_tabs.tabs
    - sphinx_exercise
    - sphinx_proof
    - sphinx_togglebutton
  config:
    html_js_files:
    - https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js
```

### Table of Contents Structure
```yaml
format: jb-book
chapters:
- file: chapters/01-introduction
- file: chapters/02-variables
- file: chapters/03-control-flow
# etc.
```

## VISUAL ELEMENTS (MYST ENHANCED)

### Mermaid Diagrams
````markdown
```{mermaid}
flowchart TD
    A[Input Data] --> B{Validate}
    B -->|Valid| C[Process]
    B -->|Invalid| D[Error]
    C --> E[Output]
    D --> F[Log Error]
```
````

### Mathematical Notation
````markdown
```{math}
\text{variance} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
```

Inline math: The mean is $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$.
````

### Code Diffs for Before/After
````markdown
```{code-diff}
--- Before
+++ After
@@ -1,3 +1,5 @@
 def calculate_mean(data):
+    if not data:
+        raise ValueError("Empty data")
     return sum(data) / len(data)
```
````

## QUALITY VERIFICATION CHECKLIST

Before accepting any chapter, verify:

### MyST/Jupyter Book Compliance
- [ ] Valid MyST frontmatter present?
- [ ] All code in code-cell directives?
- [ ] Interactive elements functioning?
- [ ] Cross-references working?
- [ ] Admonitions properly formatted?
- [ ] Glossary terms defined?

### Content Quality
- [ ] Every code example ≤30 lines and demonstrates ONE concept?
- [ ] 60/40 explanation/code ratio maintained?
- [ ] All technical terms defined in glossary and margins?
- [ ] Real disasters/consequences included?
- [ ] Progressive complexity from simple to robust?

### Pedagogical Elements
- [ ] All required admonition types present?
- [ ] Exercises properly scaffolded with solutions?
- [ ] Interactive widgets where appropriate?
- [ ] Tabbed content for comparisons?
- [ ] Performance profiling included?

### Reference Value
- [ ] Comprehensive glossary using MyST glossary directive?
- [ ] Main takeaways in panel format?
- [ ] Quick reference tables properly formatted?
- [ ] Would this help debug real research code?
- [ ] Useful as both learning material and career reference?

## COMMON PITFALLS TO AVOID

### DO NOT:
- Use plain markdown code blocks (must use code-cell)
- Create code examples over 30 lines
- Mix multiple concepts in one example
- Use complex astronomy examples that obscure Python concepts
- Skip MyST directives for structured content
- Forget margin definitions for technical terms
- Omit interactive elements where they'd help understanding
- Use bullet points in explanations (only in reference tables)
- Present code without thorough explanation
- Forget to include all required admonition types

### ALWAYS:
- Use MyST directives for all structured content
- Include interactive code cells for exploration
- Provide collapsible solutions for exercises
- Use tabs to show alternative approaches
- Place definitions in margins for easy reference
- Cross-reference other chapters properly
- Include Mermaid diagrams for complex flows
- Test all code cells for correct execution
- Verify all interactive elements work

## DEPLOYMENT REQUIREMENTS

### GitHub Pages Configuration
Ensure compatibility with GitHub Pages deployment:
- All images in `./images/` subdirectory
- Relative paths for all internal links
- No server-side dependencies
- Static widget fallbacks where needed

### Build Testing
Before submission, verify:
```bash
jupyter-book build chapter_name.md
jupyter-book build . --builder html
```

## REVIEWER INSTRUCTIONS

When reviewing a chapter:

1. **First Pass - MyST Compliance**: Verify proper MyST/JB2 formatting
2. **Second Pass - Interactivity**: Test all code cells and widgets
3. **Third Pass - Structure**: Confirm ALL mandatory sections present
4. **Fourth Pass - Code**: Check complexity limits and concept focus
5. **Fifth Pass - Pedagogy**: Confirm all admonition types used
6. **Sixth Pass - References**: Verify glossary and cross-references
7. **Final Pass - Build**: Ensure chapter builds without errors

If ANY requirement is not met, request specific revisions before acceptance.

## EXAMPLE REVIEW FEEDBACK

"This chapter needs revision:

1. Code blocks not using code-cell directives - convert all to executable cells
2. Missing MyST glossary - add formal glossary directive with all terms
3. No interactive elements in Section 9.3 - add widget for parameter exploration
4. Admonitions using wrong classes - fix question/answer/dropdown classes
5. Cross-references broken - use proper MyST ref syntax
6. Margin definitions missing for 'callback', 'memoization'
7. No Mermaid diagram for algorithm flow - add flowchart"

## FINAL CRITICAL REMINDERS

1. **These are REQUIREMENTS for MyST/JB2 compatibility**
2. **Every chapter must build and deploy on GitHub Pages**
3. **Interactive elements enhance learning - use them**
4. **Cross-references create cohesive course structure**
5. **Glossaries and margins make self-study possible**
6. **Students will access this on web and mobile devices**
7. **Progressive complexity prevents cognitive overload**
8. **MyST features create professional documentation**

## SUCCESS CRITERIA

A chapter is successful when:

- It builds without errors in Jupyter Book 2
- All interactive elements function properly
- Code cells execute correctly in order
- MyST directives enhance understanding
- Cross-references create course cohesion
- It renders beautifully on GitHub Pages
- Students can interact with and explore concepts
- The material serves as both tutorial and reference
- Debugging and numerical issues are addressed
- Students gain transferable computational thinking skills

Remember: We're creating computational scientists using modern web-based pedagogy. Every chapter should leverage MyST/Jupyter Book features to create an engaging, interactive learning experience that works seamlessly on GitHub Pages.