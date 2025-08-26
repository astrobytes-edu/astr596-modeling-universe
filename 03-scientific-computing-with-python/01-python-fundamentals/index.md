---
title: Python Fundamentals
exports:
  - format: pdf
---

# Python Fundamentals

## Your Computational Toolkit for Astrophysics

You've set up your environment and mastered Git—now it's time to build your Python foundation for computational astrophysics. This module provides the essential tools and patterns you'll use throughout the course, from basic numerical operations to object-oriented design for complex simulations.

:::{tip} 📚 How to Use This Module

Think of these chapters as your **working reference**—a toolkit you'll return to repeatedly as you tackle assignments and projects. Rather than memorizing everything, focus on understanding what tools are available and knowing where to find them when you need them. Keep these pages open while working on homework!
:::

## Module Overview

### [📊 Chapter 1: Computational Environments & Scientific Workflows](./01-python-environment.md)

Master IPython, understand how Python finds code, avoid Jupyter's hidden traps, and create reproducible computational environments

### [🔢 Chapter 2: Python as Your Astronomical Calculator](./02-python-calculator.md)

Discover why `0.1 + 0.2 ≠ 0.3`, handle extreme astronomical scales, and learn numerical safety that prevents spacecraft crashes

### [🔀 Chapter 3: Control Flow & Logic](./03-python-control-flow.md)

Design algorithms with pseudocode, implement conditional logic, master loops, and build the patterns that power every simulation

### [🗂️ Chapter 4: Data Structures - Organizing Scientific Data](./04-python-data-structures.md)

Choose between O(1) and O(n) operations, understand memory layout, and architect data organization for million-particle simulations

### [🔧 Chapter 5: Functions & Modules - Building Reusable Scientific Code](./05-python-functions-modules.md)

Create clear functional contracts, understand scope and namespaces, organize code into modules, and build professional scientific libraries

### [🎯 Chapter 6: OOP Fundamentals - Organizing Scientific Code](./06-oop-fundamentals.md)

Transform functions and data into cohesive classes, understand when objects improve code organization, and model scientific concepts naturally

## Learning Strategy

### Build Through Practice

These chapters work best when you:

1. **Skim first** to see what's available
2. **Dive deep** when you hit a specific problem
3. **Return often** as you work on assignments
4. **Run the code** examples in IPython as you read
5. **Modify examples** to test your understanding

### The Variable Star Thread

:::{note} 🌟 Continuous Project
:class: note
Each chapter builds on a continuous project—analyzing Cepheid variable stars. This thread shows how concepts connect in real astronomical applications. You'll start with simple data storage and progressively build a complete analysis pipeline, culminating in a full `VariableStar` class in Chapter 6.
:::

## Quick Navigation Guide

### "I need to..." → "Go to..."

| When you need to... | Check this chapter... | Look for section on... |
|---------------------|----------------------|------------------------|
| Fix `ModuleNotFoundError` | [Ch 1: Environments](./01-python-environment.md) | Import System, Debug Import Problems |
| Compare floating-point numbers | [Ch 2: Calculator](./02-python-calculator.md) | Safe Floating-Point Comparisons |
| Break out of a loop early | [Ch 3: Control Flow](./03-python-control-flow.md) | Loop Control: break, continue |
| Speed up particle lookups | [Ch 4: Data Structures](./04-python-data-structures.md) | Dictionaries: O(1) Lookup Magic |
| Avoid mutable default bug | [Ch 5: Functions](./05-python-functions-modules.md) | The Mutable Default Trap |
| Handle numerical overflow | [Ch 2: Calculator](./02-python-calculator.md) | Overflow and Underflow |
| Design before coding | [Ch 3: Control Flow](./03-python-control-flow.md) | Algorithmic Thinking: Pseudocode |
| Cache expensive calculations | [Ch 4: Data Structures](./04-python-data-structures.md) | Dictionaries for Caching |
| Bundle data with behavior | [Ch 6: OOP](./06-oop-fundamentals.md) | Classes and Objects |
| Validate with properties | [Ch 6: OOP](./06-oop-fundamentals.md) | Properties: Smart Attributes |

## Common Patterns You'll Use Constantly

### Defensive Programming (appears everywhere)

```python
# From Chapter 1: Always validate
assert len(data) > 0, "Cannot process empty data"

# From Chapter 2: Check numerical bounds
if not math.isfinite(value):
    raise ValueError(f"Invalid result: {value}")
```

### Safe Numerical Comparisons (critical for simulations)

```python
# From Chapter 2: Never use == with floats
if math.isclose(calculated, expected, rel_tol=1e-9):
    print("Converged!")
```

### Efficient Lookups (essential for large datasets)

```python
# From Chapter 4: O(1) vs O(n) matters!
# Slow: searching a list
if particle_id in particle_list:  # O(n)
    
# Fast: dictionary lookup  
if particle_id in particle_dict:  # O(1)
```

### Object-Oriented Design (managing complex state)

```python
# From Chapter 6: Bundle data with behavior
class Particle:
    def __init__(self, mass, position):
        self.mass = mass
        self.position = position
    
    def update_position(self, dt):
        self.position += self.velocity * dt
```

## Your Reference Checklist

As you work through the chapter exercises and course assignments, you'll naturally master these concepts:

### Core Competencies

**Numerical Safety**

- [ ] Why `0.1 + 0.2 ≠ 0.3` and how to handle it
- [ ] Preventing overflow in astronomical calculations
- [ ] Using log space for extreme scales

**Algorithm Design**

- [ ] Writing pseudocode before implementation
- [ ] Choosing for vs while loops appropriately
- [ ] Recognizing O(n²) bottlenecks

**Data Organization**

- [ ] When lists become too slow
- [ ] Why dictionaries enable instant lookups
- [ ] How aliasing creates subtle bugs

**Code Structure**

- [ ] Creating reusable functions
- [ ] Organizing code into modules
- [ ] Documenting with clear docstrings
- [ ] Designing classes for complex systems
- [ ] Using properties for validation

## Problem-Solving Flowchart

```markdown
Homework Problem
    ↓
"What kind of problem is this?"
    ├─ Numerical precision issue → Chapter 2
    ├─ Need to repeat operation → Chapter 3 (loops)
    ├─ Organizing many items → Chapter 4 (data structures)
    ├─ Code getting repetitive → Chapter 5 (functions)
    ├─ Managing complex state → Chapter 6 (classes)
    └─ Import not working → Chapter 1 (environments)
```

## Performance Quick Reference

Keep this table handy when choosing data structures:

| Operation | List | Dict | Set | Your Choice When... |
|-----------|------|------|-----|-------------------|
| Find by ID | Slow O(n) | **Fast O(1)** | Fast O(1) | You have unique IDs → Dict |
| Keep order | ✓ Yes | ✗ No | ✗ No | Order matters → List |
| No duplicates | ✗ Manual | ✗ Manual | ✓ Automatic | Unique items → Set |
| By position | **Fast O(1)** | ✗ No | ✗ No | Need indexing → List |

## What's Next?

After building your Python foundation with all six chapters, you'll advance to [**Scientific Computing Core**](../02-scientific-computing-core/index.md) where these fundamentals become powerful tools:

- Your classes from Chapter 6 gain **inheritance and advanced patterns**
- Your lists transform into **NumPy arrays** (100× faster!)
- Your loops become **vectorized operations**
- Your modules grow into **scientific packages**
- Your objects integrate with **Matplotlib, Pandas, and SymPy**

The patterns you learn here—defensive programming, algorithmic thinking, performance awareness — will guide you through increasingly sophisticated computational challenges.

---

**Ready to build your toolkit?** Start exploring [Chapter 1: Computational Environments & Scientific Workflows](./01-python-environment.md) →

*Remember: These chapters are your companions throughout the course. Bookmark them, return often, and use them actively as you solve real problems!*
