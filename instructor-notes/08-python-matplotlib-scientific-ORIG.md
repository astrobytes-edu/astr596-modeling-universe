# Chapter 8: Matplotlib - Creating Publication-Quality Scientific Visualizations

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand Matplotlib's object-oriented architecture and why it's essential for scientific visualization
- Create publication-quality figures using the object-oriented interface, not just pyplot shortcuts
- Master the anatomy of a figure: figures, axes, artists, and their relationships
- Apply iterative refinement to transform rough plots into journal-ready visualizations
- Build reusable plotting functions and modules for your research workflow
- Choose appropriate plot types for different scientific data and questions
- Debug common visualization issues and understand Matplotlib's rendering pipeline
- Customize every aspect of your plots for different publication requirements
- Create complex multi-panel figures for comprehensive data presentation
- Integrate NumPy arrays seamlessly with Matplotlib's visualization capabilities

## Prerequisites Check

Before starting this chapter, verify you can:
- ✓ Create and manipulate NumPy arrays (Chapter 7)
- ✓ Understand object-oriented programming concepts (Chapter 6)
- ✓ Work with methods and attributes of objects (Chapter 6)
- ✓ Write functions with multiple parameters (Chapter 5)
- ✓ Use dictionaries for configuration (Chapter 4)

## Chapter Overview

You've mastered NumPy arrays and understand object-oriented programming. Now it's time to bring your data to life through visualization. But here's what most tutorials won't tell you: the difference between a plot that gets ignored and one that gets cited often comes down to thoughtful visual design and iterative refinement. A figure in a prestigious journal didn't start that way—it evolved through dozens of iterations, each improving clarity, aesthetics, and scientific communication.

Matplotlib is the foundation of scientific visualization in Python, underlying everything from quick exploratory plots to the stunning visuals in Nature and Science. But most scientists use only a fraction of its power, relying on simplistic pyplot commands that create adequate but not exceptional figures. This chapter reveals Matplotlib's true architecture—a sophisticated object-oriented system that gives you complete control over every pixel of your visualization.

You'll discover why `plt.plot()` is just training wheels, and why real scientific visualization requires understanding figures, axes, and artists. You'll learn to think of visualization as an iterative design process, not a one-line command. Most importantly, you'll build a personal library of plotting functions that will save you countless hours throughout your research career. By the end, you'll create visualizations that don't just show data—they tell scientific stories with clarity and impact.

## 8.1 The Architecture of Matplotlib: Understanding the Object Hierarchy

Before creating a single plot, we need to understand what Matplotlib actually is and how it works. This understanding separates scientists who struggle with plotting from those who create stunning visualizations effortlessly.

### The Two Interfaces: pyplot vs Object-Oriented

Matplotlib provides two interfaces, and understanding both is crucial for scientific work. Most beginners start with pyplot because it seems simpler, but this simplicity is deceptive and limiting:

```python
In [1]: import numpy as np
In [2]: import matplotlib.pyplot as plt

In [3]: # Generate sample data
In [4]: x = np.linspace(0, 2*np.pi, 100)
In [5]: y = np.sin(x)

In [6]: # METHOD 1: pyplot interface (state-based, MATLAB-like)
In [7]: plt.figure(figsize=(12, 5))

In [8]: plt.subplot(1, 2, 1)
In [9]: plt.plot(x, y)
In [10]: plt.title('Pyplot Interface')
In [11]: plt.xlabel('x')
In [12]: plt.ylabel('sin(x)')

In [13]: # METHOD 2: Object-oriented interface (explicit, powerful)
In [14]: plt.subplot(1, 2, 2)
In [15]: ax = plt.gca()  # Get current axes
In [16]: ax.plot(x, y)
In [17]: ax.set_title('Object-Oriented Interface')
In [18]: ax.set_xlabel('x')
In [19]: ax.set_ylabel('sin(x)')

In [20]: plt.tight_layout()
In [21]: plt.show()
```

The pyplot interface (`plt.plot()`) seems simpler, but it's hiding what's really happening. Every pyplot command is secretly creating or modifying objects behind the scenes. The object-oriented interface makes these objects explicit, giving you direct control. This is like the difference between automatic and manual transmission—automatic seems easier until you need precise control on a mountain road.

### ⚠️ **Common Bug Alert: The Hidden State Machine**

```python
# DANGEROUS: pyplot functions operate on "current" axes
plt.plot([1, 2, 3], [1, 4, 2])
plt.subplot(2, 1, 1)  # Creates new axes, now "current"
plt.title("Title")  # Goes to subplot, not original plot!

# SAFE: Object-oriented approach is explicit
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot([1, 2, 3], [1, 4, 2])
ax1.set_title("Title")  # Explicitly on ax1
```

The pyplot interface maintains hidden global state about which figure and axes are "current." This leads to subtle bugs when you have multiple figures or subplots. The object-oriented interface eliminates this ambiguity by making you specify exactly which axes you're modifying.

### The Anatomy of a Figure

Every Matplotlib visualization consists of a hierarchy of objects. Understanding this hierarchy is essential for creating professional figures. Let's examine the complete anatomy using Nicolas P. Rougier's excellent anatomy diagram:

```python
In [22]: # Anatomy of a Matplotlib Figure
In [23]: # Based on Nicolas P. Rougier's figure anatomy diagram
In [24]: # Copyright (c) 2016 Nicolas P. Rougier - MIT License

In [25]: import numpy as np
In [26]: import matplotlib.pyplot as plt
In [27]: from matplotlib.ticker import MultipleLocator, FuncFormatter

In [28]: np.random.seed(123)

In [29]: # Generate sample data
In [30]: X = np.linspace(0.5, 3.5, 100)
In [31]: Y1 = 3 + np.cos(X)
In [32]: Y2 = 1 + np.cos(1 + X/0.75)/2
In [33]: Y3 = np.random.uniform(Y1, Y2, len(X))

In [34]: # Create figure with specific properties
In [35]: fig = plt.figure(figsize=(8, 8), facecolor="white")
In [36]: ax = fig.add_subplot(1, 1, 1, aspect=1)  # aspect=1 makes it square

In [37]: # Custom formatter for minor ticks
In [38]: def minor_tick(x, pos):
   ...:     """Only label minor ticks that aren't integers."""
   ...:     if not x % 1.0:
   ...:         return ""
   ...:     return "%.2f" % x

In [39]: # Configure tick locators and formatters
In [40]: ax.xaxis.set_major_locator(MultipleLocator(1.000))
In [41]: ax.xaxis.set_minor_locator(MultipleLocator(0.250))
In [42]: ax.yaxis.set_major_locator(MultipleLocator(1.000))
In [43]: ax.yaxis.set_minor_locator(MultipleLocator(0.250))
In [44]: ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

In [45]: # Set axis limits
In [46]: ax.set_xlim(0, 4)
In [47]: ax.set_ylim(0, 4)

In [48]: # Configure tick appearance
In [49]: ax.tick_params(which='major', width=1.0, length=10)
In [50]: ax.tick_params(which='minor', width=1.0, length=5, 
   ...:                labelsize=10, labelcolor='0.25')

In [51]: # Add grid
In [52]: ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

In [53]: # Plot data with different styles
In [54]: ax.plot(X, Y1, c=(0.25, 0.25, 1.00), lw=2, label="Blue signal", zorder=10)
In [55]: ax.plot(X, Y2, c=(1.00, 0.25, 0.25), lw=2, label="Red signal")
In [56]: ax.scatter(X, Y3, c='white', edgecolors='black', s=50)

In [57]: # Add labels and title
In [58]: ax.set_title("Anatomy of a figure", fontsize=20)
In [59]: ax.set_xlabel("X axis label")
In [60]: ax.set_ylabel("Y axis label")

In [61]: # Add legend
In [62]: ax.legend(frameon=False)

In [63]: plt.show()
```

This figure illustrates the key components of any Matplotlib visualization. Each component is an object with properties you can control. Understanding this hierarchy gives you the power to customize every aspect of your figures.

### The Object Hierarchy Explained

Understanding the object hierarchy is crucial for mastering Matplotlib:

```{mermaid}
flowchart TD
    F[Figure<br/>The entire window/canvas] --> A1[Axes<br/>The actual plot area]
    F --> A2[Axes 2<br/>Another subplot]
    F --> A3[Axes ...<br/>More subplots]
    
    A1 --> X1[XAxis<br/>Horizontal axis]
    A1 --> Y1[YAxis<br/>Vertical axis]
    A1 --> T1[Title<br/>Plot title]
    A1 --> L1[Legend<br/>Data labels]
    A1 --> AR[Artists<br/>Lines, patches, text, etc.]
    
    X1 --> XT[Major/Minor Ticks]
    X1 --> XL[Tick Labels]
    X1 --> XG[Grid Lines]
    X1 --> XLB[Axis Label]
    
    Y1 --> YT[Major/Minor Ticks]
    Y1 --> YL[Tick Labels]
    Y1 --> YG[Grid Lines]
    Y1 --> YLB[Axis Label]
    
    AR --> LN[Lines<br/>plot(), loglog()]
    AR --> PT[Patches<br/>bar(), hist()]
    AR --> CL[Collections<br/>scatter()]
    AR --> TX[Text<br/>text(), annotate()]
    
    style F fill:#f9f,stroke:#333,stroke-width:4px
    style A1 fill:#9ff,stroke:#333,stroke-width:2px
    style AR fill:#ff9,stroke:#333,stroke-width:2px
```

Each level of this hierarchy is an object with methods and attributes you can control:

```python
In [64]: # Let's explore the hierarchy programmatically
In [65]: fig, ax = plt.subplots()

In [66]: # Figure level - controls the canvas
In [67]: print(f"Figure size: {fig.get_size_inches()}")
In [68]: print(f"Figure DPI: {fig.dpi}")
In [69]: print(f"Number of axes: {len(fig.axes)}")
Figure size: [6.4 4.8]
Figure DPI: 100.0
Number of axes: 1

In [70]: # Axes level - controls the plot area
In [71]: print(f"Axes position: {ax.get_position()}")
In [72]: print(f"X limits: {ax.get_xlim()}")
In [73]: print(f"Y limits: {ax.get_ylim()}")
Axes position: Bbox(x0=0.125, y0=0.11, x1=0.9, y1=0.88)
X limits: (0.0, 1.0)
Y limits: (0.0, 1.0)

In [74]: # Axis level - controls individual axes
In [75]: print(f"X axis scale: {ax.xaxis.get_scale()}")
In [76]: print(f"Number of x ticks: {len(ax.xaxis.get_major_ticks())}")
X axis scale: linear
Number of x ticks: 6

In [77]: # Artist level - individual visual elements
In [78]: line, = ax.plot([1, 2, 3], [1, 4, 2])
In [79]: print(f"Line color: {line.get_color()}")
In [80]: print(f"Line width: {line.get_linewidth()}")
In [81]: print(f"Line style: {line.get_linestyle()}")
Line color: #1f77b4
Line width: 1.5
Line style: -
```

### 🔍 **Check Your Understanding**

Why is the object-oriented interface preferred over pyplot for scientific work?

<details>
<summary>Answer</summary>

The object-oriented interface is preferred for several reasons:

1. **Explicit control**: You specify exactly which axes you're modifying, eliminating ambiguity
2. **No hidden state**: No confusion about which figure or axes is "current"
3. **Better for complex figures**: Essential when working with multiple subplots or figures
4. **Reusability**: Easier to write functions that take axes objects as parameters
5. **Debugging**: Clearer error messages and easier to track what's being modified
6. **Professional standard**: Most scientific code uses the OO interface

While pyplot is fine for quick exploration, any figure going into a publication should use the object-oriented interface for maximum control and clarity.

</details>

## 8.2 From Quick Plots to Publication Quality: The Iteration Process

Creating publication-quality figures is not a one-shot process. It requires iteration and refinement. Let's walk through the evolution of a real scientific figure, from quick-and-dirty to journal-ready. This is where most students fail—they create a plot once and consider it done. Scientific visualization requires an iterative mindset.

### Stage 1: The Quick Plot (Exploration Phase)

Scientists often start with the simplest possible visualization to see their data:

```python
In [82]: # Initial data exploration - the typical first attempt
In [83]: np.random.seed(42)
In [84]: data = np.random.randn(1000)
In [85]: plt.hist(data)
In [86]: plt.show()
```

This plot serves its purpose for exploration but would never appear in a publication. The default settings produce a figure that lacks clarity, proper labeling, and aesthetic appeal. Many students stop here—don't be one of them.

### Stage 2: Basic Improvements (Communication Phase)

First, we add essential elements that every scientific figure needs:

```python
In [87]: # Better: Add labels and improve basics
In [88]: fig, ax = plt.subplots(figsize=(8, 6))
In [89]: ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
In [90]: ax.set_xlabel('Value', fontsize=12)
In [91]: ax.set_ylabel('Frequency', fontsize=12)
In [92]: ax.set_title('Distribution of Measurements', fontsize=14)
In [93]: ax.grid(True, alpha=0.3)
In [94]: plt.show()
```

We've switched to the object-oriented interface, added labels, and improved visibility. But this is still far from publication quality. The iteration process has just begun.

### Stage 3: Scientific Context (Analysis Phase)

Scientific figures need to communicate more than just raw data:

```python
In [95]: # Add statistical information and improve clarity
In [96]: fig, ax = plt.subplots(figsize=(10, 6))

In [97]: # Calculate statistics
In [98]: mean = np.mean(data)
In [99]: std = np.std(data)
In [100]: median = np.median(data)

In [101]: # Create histogram with better bins using Freedman-Diaconis rule
In [102]: q75, q25 = np.percentile(data, [75, 25])
In [103]: iqr = q75 - q25
In [104]: bin_width = 2 * iqr / (len(data) ** (1/3))
In [105]: n_bins = int((data.max() - data.min()) / bin_width)

In [106]: # Plot histogram with density normalization
In [107]: counts, bins, patches = ax.hist(data, bins=n_bins, density=True, 
    ...:                                  edgecolor='black', alpha=0.7,
    ...:                                  label='Data')

In [108]: # Add normal distribution overlay for comparison
In [109]: from scipy import stats
In [110]: x = np.linspace(data.min(), data.max(), 100)
In [111]: ax.plot(x, stats.norm.pdf(x, mean, std),
    ...:         'r-', linewidth=2, label='Normal fit')

In [112]: # Add vertical lines for statistics
In [113]: ax.axvline(mean, color='red', linestyle='--', linewidth=2, 
    ...:            label=f'Mean = {mean:.2f}')
In [114]: ax.axvline(median, color='green', linestyle='--', linewidth=2,
    ...:            label=f'Median = {median:.2f}')

In [115]: # Improve labels with units and context
In [116]: ax.set_xlabel('Measurement Value (arbitrary units)', fontsize=12)
In [117]: ax.set_ylabel('Probability Density', fontsize=12)
In [118]: ax.set_title('Distribution of Experimental Measurements\n' + 
    ...:              f'N = {len(data)}, σ = {std:.2f}', fontsize=14)

In [119]: # Add legend with proper positioning
In [120]: ax.legend(loc='upper right', frameon=True, shadow=True)

In [121]: # Add grid for readability
In [122]: ax.grid(True, alpha=0.3, linestyle='--')

In [123]: # Set reasonable axis limits
In [124]: ax.set_xlim([mean - 4*std, mean + 4*std])

In [125]: plt.tight_layout()
In [126]: plt.show()
```

### Stage 4: Publication Quality (Refinement Phase)

Now we refine every detail for journal submission:

```python
In [127]: def create_publication_histogram(data, fig_size=(10, 7), dpi=300):
    ...:     """
    ...:     Create a publication-quality histogram with statistical overlays.
    ...:     
    ...:     This function demonstrates the level of detail required for
    ...:     figures that will appear in scientific publications.
    ...:     
    ...:     Parameters
    ...:     ----------
    ...:     data : array-like
    ...:         The data to plot
    ...:     fig_size : tuple
    ...:         Figure size in inches
    ...:     dpi : int
    ...:         Resolution for saving
    ...:     
    ...:     Returns
    ...:     -------
    ...:     fig, ax : matplotlib objects
    ...:         Figure and axes objects for further customization
    ...:     """
    ...:     # Use a clean style as base
    ...:     plt.style.use('seaborn-v0_8-whitegrid')
    ...:     
    ...:     # Create figure with high DPI for print
    ...:     fig, ax = plt.subplots(figsize=fig_size, dpi=100)
    ...:     
    ...:     # Calculate comprehensive statistics
    ...:     mean, std = np.mean(data), np.std(data)
    ...:     median = np.median(data)
    ...:     sem = std / np.sqrt(len(data))  # Standard error of mean
    ...:     
    ...:     # Optimal bin calculation (Freedman-Diaconis)
    ...:     q75, q25 = np.percentile(data, [75, 25])
    ...:     iqr = q75 - q25
    ...:     bin_width = 2 * iqr / (len(data) ** (1/3))
    ...:     n_bins = int((data.max() - data.min()) / bin_width)
    ...:     n_bins = max(n_bins, 10)  # Ensure minimum bins
    ...:     
    ...:     # Main histogram
    ...:     n, bins, patches = ax.hist(data, bins=n_bins, density=True,
    ...:                                color='skyblue', edgecolor='navy',
    ...:                                linewidth=0.5, alpha=0.7,
    ...:                                label='Observed data')
    ...:     
    ...:     # Add kernel density estimate for smooth curve
    ...:     kde = stats.gaussian_kde(data)
    ...:     x_range = np.linspace(data.min() - std, data.max() + std, 200)
    ...:     ax.plot(x_range, kde(x_range), 'navy', linewidth=2, 
    ...:             label='Kernel density estimate')
    ...:     
    ...:     # Add theoretical normal distribution
    ...:     normal_dist = stats.norm(loc=mean, scale=std)
    ...:     ax.plot(x_range, normal_dist.pdf(x_range), 'r--', linewidth=1.5,
    ...:             alpha=0.8, label=f'Normal (μ={mean:.2f}, σ={std:.2f})')
    ...:     
    ...:     # Statistical markers
    ...:     ax.axvline(mean, color='red', linestyle='-', linewidth=1.5, 
    ...:                alpha=0.8, zorder=2)
    ...:     ax.axvline(median, color='green', linestyle='--', linewidth=1.5, 
    ...:                alpha=0.8, zorder=2)
    ...:     
    ...:     # Shaded regions for standard deviations
    ...:     ax.axvspan(mean - std, mean + std, alpha=0.2, color='red',
    ...:                label=f'±1σ (68.3% of data)')
    ...:     ax.axvspan(mean - 2*std, mean + 2*std, alpha=0.1, color='orange',
    ...:                label=f'±2σ (95.4% of data)')
    ...:     
    ...:     # Annotations with professional formatting
    ...:     ax.annotate(f'Mean\n{mean:.3f}±{sem:.3f}',
    ...:                 xy=(mean, ax.get_ylim()[1] * 0.9),
    ...:                 xytext=(mean + 1.5*std, ax.get_ylim()[1] * 0.9),
    ...:                 fontsize=10, ha='center',
    ...:                 bbox=dict(boxstyle='round,pad=0.3', 
    ...:                          facecolor='white', edgecolor='gray'),
    ...:                 arrowprops=dict(arrowstyle='->', 
    ...:                                connectionstyle='arc3,rad=0.3'))
    ...:     
    ...:     # Professional labels with LaTeX formatting
    ...:     ax.set_xlabel('Measurement Value (a.u.)', fontsize=12, fontweight='bold')
    ...:     ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ...:     ax.set_title('Statistical Distribution of Experimental Measurements',
    ...:                  fontsize=14, fontweight='bold', pad=20)
    ...:     
    ...:     # Statistical tests box
    ...:     shapiro_stat, shapiro_p = stats.shapiro(data)
    ...:     ks_stat, ks_p = stats.kstest(data, 'norm', args=(mean, std))
    ...:     
    ...:     textstr = f'N = {len(data)}\n'
    ...:     textstr += f'Shapiro-Wilk: p = {shapiro_p:.4f}\n'
    ...:     textstr += f'KS test: p = {ks_p:.4f}'
    ...:     
    ...:     ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
    ...:             verticalalignment='top',
    ...:             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ...:     
    ...:     # Legend with optimized placement
    ...:     ax.legend(loc='best', frameon=True, fancybox=True, 
    ...:               shadow=True, borderpad=1, fontsize=10)
    ...:     
    ...:     # Remove top and right spines for cleaner look
    ...:     ax.spines['top'].set_visible(False)
    ...:     ax.spines['right'].set_visible(False)
    ...:     
    ...:     # Refined tick parameters
    ...:     ax.tick_params(axis='both', which='major', labelsize=10)
    ...:     ax.set_xlim([mean - 4*std, mean + 4*std])
    ...:     ax.set_ylim(bottom=0)
    ...:     
    ...:     # Add minor ticks for precision reading
    ...:     ax.xaxis.set_minor_locator(plt.MultipleLocator(std/2))
    ...:     ax.yaxis.set_minor_locator(plt.AutoMinorLocator())
    ...:     
    ...:     plt.tight_layout()
    ...:     
    ...:     return fig, ax

In [128]: # Generate and plot data
In [129]: np.random.seed(42)
In [130]: data = np.random.normal(100, 15, 1000)
In [131]: fig, ax = create_publication_histogram(data)

In [132]: # Save in multiple formats for different uses
In [133]: fig.savefig('histogram_publication.pdf', dpi=300, bbox_inches='tight')
In [134]: fig.savefig('histogram_presentation.png', dpi=150, bbox_inches='tight')
In [135]: fig.savefig('histogram_web.svg', bbox_inches='tight')

In [136]: plt.show()
```

### 📦 **Computational Thinking Box: The Iteration Mindset**

```
PATTERN: Iterative Refinement in Scientific Visualization

Great scientific figures don't happen by accident. They evolve through
deliberate iteration, each pass improving specific aspects:

1. Exploration Phase (5 minutes): Quick plots to understand data
   - Use defaults, focus on seeing patterns
   - Try different plot types rapidly
   - Don't worry about aesthetics yet
   
2. Communication Phase (15 minutes): Add scientific context
   - Labels with units, title with sample size
   - Choose appropriate plot type for your data
   - Add statistical information (mean, std, etc.)
   
3. Refinement Phase (30+ minutes): Polish every detail
   - Typography consistency (font sizes, weights)
   - Color scheme appropriate for color-blind readers
   - Match journal requirements (size, format, style)
   - Test printing in grayscale
   
4. Validation Phase (10 minutes): Test with colleagues
   - Is the main message clear in 5 seconds?
   - Are there any misleading elements?
   - Does it reproduce well in print/PDF?

This iteration process applies beyond plotting:
- Writing papers (rough draft → polished manuscript)
- Developing algorithms (prototype → production code)
- Experimental design (pilot → full study)

The key insight: Budget time for iteration. A figure that will
appear in your thesis or paper deserves hours, not minutes.
The difference between amateur and professional visualization
is iteration count, not initial skill.
```

## 8.3 Building Your Plotting Toolkit: Reusable Functions

One of the most important skills in scientific computing is building reusable tools. Instead of rewriting plotting code for every figure, create a library of plotting functions that you can use throughout your research. This is where object-oriented programming from Chapter 6 becomes invaluable.

### Creating Modular Plotting Functions

Let's build a reusable function for a common scientific need: comparing multiple datasets with error bars:

```python
In [137]: def plot_comparison(datasets, labels=None, colors=None, 
    ...:                     xlabel='X', ylabel='Y', title='',
    ...:                     figsize=(10, 6), style='errorbar',
    ...:                     save_path=None, **kwargs):
    ...:     """
    ...:     Create a publication-quality comparison plot for multiple datasets.
    ...:     
    ...:     This is an example of a reusable plotting function that handles
    ...:     common scientific visualization needs. Build a collection of
    ...:     these for your research!
    ...:     
    ...:     Parameters
    ...:     ----------
    ...:     datasets : list of arrays or list of (x, y, yerr) tuples
    ...:         Data to plot. Can be 1D arrays or tuples with errors
    ...:     labels : list of str, optional
    ...:         Labels for each dataset
    ...:     colors : list of colors, optional
    ...:         Colors for each dataset (defaults to color cycle)
    ...:     xlabel, ylabel : str
    ...:         Axis labels
    ...:     title : str
    ...:         Figure title
    ...:     figsize : tuple
    ...:         Figure size in inches
    ...:     style : str
    ...:         Plot style: 'line', 'scatter', 'errorbar', 'fill'
    ...:     save_path : str, optional
    ...:         Path to save figure (include extension for format)
    ...:     **kwargs : dict
    ...:         Additional arguments passed to plotting function
    ...:     
    ...:     Returns
    ...:     -------
    ...:     fig, ax : matplotlib objects
    ...:         For further customization
    ...:     
    ...:     Examples
    ...:     --------
    ...:     >>> x = np.linspace(0, 10, 50)
    ...:     >>> y1 = np.sin(x) + np.random.normal(0, 0.1, 50)
    ...:     >>> y2 = np.cos(x) + np.random.normal(0, 0.1, 50)
    ...:     >>> yerr1 = np.full_like(y1, 0.1)
    ...:     >>> yerr2 = np.full_like(y2, 0.1)
    ...:     >>> fig, ax = plot_comparison([(x, y1, yerr1), (x, y2, yerr2)],
    ...:     ...                          labels=['sin(x)', 'cos(x)'],
    ...:     ...                          xlabel='Time (s)', ylabel='Signal (V)')
    ...:     """
    ...:     # Create figure with consistent style
    ...:     fig, ax = plt.subplots(figsize=figsize)
    ...:     
    ...:     # Default colors if not provided
    ...:     if colors is None:
    ...:         colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    ...:     
    ...:     # Default labels if not provided
    ...:     if labels is None:
    ...:         labels = [f'Dataset {i+1}' for i in range(len(datasets))]
    ...:     
    ...:     # Plot each dataset
    ...:     for i, (data, label, color) in enumerate(zip(datasets, labels, colors)):
    ...:         # Handle different data formats
    ...:         if isinstance(data, tuple):
    ...:             if len(data) == 2:
    ...:                 x, y = data
    ...:                 yerr = None
    ...:             elif len(data) == 3:
    ...:                 x, y, yerr = data
    ...:             else:
    ...:                 raise ValueError("Data tuple must be (x, y) or (x, y, yerr)")
    ...:         else:
    ...:             # Assume 1D array, create x values
    ...:             y = data
    ...:             x = np.arange(len(y))
    ...:             yerr = None
    ...:         
    ...:         # Plot based on style
    ...:         if style == 'line':
    ...:             ax.plot(x, y, label=label, color=color, **kwargs)
    ...:         elif style == 'scatter':
    ...:             ax.scatter(x, y, label=label, color=color, **kwargs)
    ...:         elif style == 'errorbar':
    ...:             ax.errorbar(x, y, yerr=yerr, label=label, color=color,
    ...:                         capsize=3, capthick=1, **kwargs)
    ...:         elif style == 'fill':
    ...:             ax.plot(x, y, label=label, color=color, **kwargs)
    ...:             if yerr is not None:
    ...:                 ax.fill_between(x, y - yerr, y + yerr, 
    ...:                                color=color, alpha=0.3)
    ...:     
    ...:     # Professional formatting
    ...:     ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ...:     ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ...:     if title:
    ...:         ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ...:     
    ...:     # Add grid
    ...:     ax.grid(True, alpha=0.3, linestyle='--')
    ...:     
    ...:     # Legend
    ...:     ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ...:     
    ...:     # Clean up spines
    ...:     ax.spines['top'].set_visible(False)
    ...:     ax.spines['right'].set_visible(False)
    ...:     
    ...:     plt.tight_layout()
    ...:     
    ...:     # Save if requested
    ...:     if save_path:
    ...:         fig.savefig(save_path, dpi=300, bbox_inches='tight')
    ...:         print(f"Figure saved to {save_path}")
    ...:     
    ...:     return fig, ax
```

### Building a Personal Plotting Module

Create a module with your commonly used plotting functions:

```python
In [138]: # Save this as scientific_plots.py
In [139]: """
    ...: scientific_plots.py
    ...: 
    ...: A collection of reusable plotting functions for scientific visualization.
    ...: Build this throughout your research career!
    ...: """
    ...: 
    ...: import numpy as np
    ...: import matplotlib.pyplot as plt
    ...: from scipy import stats
    ...: import warnings
    ...: 
    ...: def setup_plot_style():
    ...:     """Set up consistent plotting style for all figures."""
    ...:     plt.rcParams.update({
    ...:         'figure.figsize': (10, 6),
    ...:         'figure.dpi': 100,
    ...:         'font.size': 11,
    ...:         'font.family': 'sans-serif',
    ...:         'axes.labelsize': 12,
    ...:         'axes.titlesize': 14,
    ...:         'xtick.labelsize': 10,
    ...:         'ytick.labelsize': 10,
    ...:         'legend.fontsize': 10,
    ...:         'lines.linewidth': 1.5,
    ...:         'lines.markersize': 6,
    ...:         'axes.grid': True,
    ...:         'grid.alpha': 0.3,
    ...:         'grid.linestyle': '--',
    ...:     })
    ...: 
    ...: def plot_residuals(observed, predicted, ax=None, 
    ...:                    xlabel='Predicted', ylabel='Residuals'):
    ...:     """
    ...:     Create a residual plot for model validation.
    ...:     
    ...:     Essential for checking model assumptions!
    ...:     """
    ...:     if ax is None:
    ...:         fig, ax = plt.subplots(figsize=(8, 6))
    ...:     
    ...:     residuals = observed - predicted
    ...:     
    ...:     # Scatter plot of residuals
    ...:     ax.scatter(predicted, residuals, alpha=0.6, s=20)
    ...:     
    ...:     # Add zero line
    ...:     ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ...:     
    ...:     # Add ±2σ bounds
    ...:     std_resid = np.std(residuals)
    ...:     ax.axhline(y=2*std_resid, color='orange', linestyle=':', alpha=0.7)
    ...:     ax.axhline(y=-2*std_resid, color='orange', linestyle=':', alpha=0.7)
    ...:     
    ...:     # Labels
    ...:     ax.set_xlabel(xlabel, fontweight='bold')
    ...:     ax.set_ylabel(ylabel, fontweight='bold')
    ...:     ax.set_title('Residual Analysis', fontweight='bold')
    ...:     
    ...:     # Add text with statistics
    ...:     mean_resid = np.mean(residuals)
    ...:     ax.text(0.02, 0.98, 
    ...:             f'Mean: {mean_resid:.3f}\nStd: {std_resid:.3f}',
    ...:             transform=ax.transAxes, verticalalignment='top',
    ...:             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ...:     
    ...:     return ax
    ...: 
    ...: def plot_correlation_matrix(data, labels=None, ax=None, cmap='RdBu_r'):
    ...:     """
    ...:     Create a correlation matrix heatmap.
    ...:     
    ...:     Useful for exploring relationships in multivariate data.
    ...:     """
    ...:     if ax is None:
    ...:         fig, ax = plt.subplots(figsize=(10, 8))
    ...:     
    ...:     # Calculate correlation matrix
    ...:     corr_matrix = np.corrcoef(data.T)
    ...:     
    ...:     # Create heatmap
    ...:     im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ...:     
    ...:     # Add colorbar
    ...:     plt.colorbar(im, ax=ax, label='Correlation coefficient')
    ...:     
    ...:     # Add labels
    ...:     if labels is not None:
    ...:         ax.set_xticks(np.arange(len(labels)))
    ...:         ax.set_yticks(np.arange(len(labels)))
    ...:         ax.set_xticklabels(labels, rotation=45, ha='right')
    ...:         ax.set_yticklabels(labels)
    ...:     
    ...:     # Add correlation values as text
    ...:     for i in range(corr_matrix.shape[0]):
    ...:         for j in range(corr_matrix.shape[1]):
    ...:             text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
    ...:                          ha='center', va='center',
    ...:                          color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
    ...:                          fontsize=8)
    ...:     
    ...:     ax.set_title('Correlation Matrix', fontweight='bold', pad=15)
    ...:     
    ...:     return ax
```

### 🔍 **Check Your Understanding**

Why is building reusable plotting functions important for scientific work?

<details>
<summary>Answer</summary>

Building reusable plotting functions is crucial for several reasons:

1. **Time efficiency**: Write once, use many times across projects
2. **Consistency**: Ensures all your figures have the same professional style
3. **Reproducibility**: Others can recreate your figures exactly
4. **Error reduction**: Tested functions reduce bugs in visualization code
5. **Collaboration**: Team members can use the same visualization tools
6. **Evolution**: Functions improve over time as you add features
7. **Publication standards**: Easy to ensure all figures meet journal requirements

Think of it as building your own visualization library tailored to your research needs. Every time you create a new plot type, add it to your module. By the end of your PhD, you'll have a comprehensive toolkit that makes creating publication figures trivial.

</details>

## 8.4 Common Plot Types for Scientific Data

Different types of scientific data require different visualization approaches. Let's explore the most common plot types and when to use each.

### Choosing the Right Scale: Linear vs. Logarithmic

One of the most critical decisions in scientific visualization is choosing the appropriate scale for your axes. This choice can reveal or obscure patterns in your data, and using the wrong scale is a common source of misinterpretation in scientific literature. Many students default to linear scales because that's what `plt.plot()` gives you, but this often hides important patterns in scientific data.

```python
In [139]: def demonstrate_scale_choices():
    ...:     """
    ...:     Show the same data with different scaling to demonstrate
    ...:     when each is appropriate. ALWAYS experiment with scales!
    ...:     """
    ...:     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ...:     
    ...:     # Generate power law data (common in astronomy)
    ...:     x = np.linspace(1, 1000, 1000)
    ...:     y_power = 1000 * x**(-2.5)  # Like a mass function
    ...:     
    ...:     # Linear scale - often hides the pattern
    ...:     ax1 = axes[0, 0]
    ...:     ax1.plot(x, y_power, 'b-', linewidth=2)
    ...:     ax1.set_title('Power Law: Linear Scale\n(Most data invisible!)')
    ...:     ax1.set_xlabel('X')
    ...:     ax1.set_ylabel('Y')
    ...:     ax1.grid(True, alpha=0.3)
    ...:     
    ...:     # Log-log scale - reveals power laws as straight lines
    ...:     ax2 = axes[0, 1]
    ...:     ax2.loglog(x, y_power, 'b-', linewidth=2)
    ...:     ax2.set_title('Power Law: Log-Log Scale\n(Becomes a straight line!)')
    ...:     ax2.set_xlabel('X')
    ...:     ax2.set_ylabel('Y')
    ...:     ax2.grid(True, which="both", alpha=0.3)
    ...:     
    ...:     # Generate exponential data
    ...:     y_exp = np.exp(x/100)
    ...:     
    ...:     # Linear scale for exponential
    ...:     ax3 = axes[0, 2]
    ...:     ax3.plot(x[:200], y_exp[:200], 'r-', linewidth=2)
    ...:     ax3.set_title('Exponential: Linear Scale\n(Growth rate unclear)')
    ...:     ax3.set_xlabel('X')
    ...:     ax3.set_ylabel('Y')
    ...:     ax3.grid(True, alpha=0.3)
    ...:     
    ...:     # Semi-log scale for exponential
    ...:     ax4 = axes[1, 0]
    ...:     ax4.semilogy(x, y_exp, 'r-', linewidth=2)
    ...:     ax4.set_title('Exponential: Semi-log Y\n(Becomes a straight line!)')
    ...:     ax4.set_xlabel('X')
    ...:     ax4.set_ylabel('Y (log scale)')
    ...:     ax4.grid(True, which="both", alpha=0.3)
    ...:     
    ...:     # Data spanning many orders of magnitude
    ...:     data = np.random.lognormal(0, 2, 10000)
    ...:     
    ...:     # Linear histogram - misleading
    ...:     ax5 = axes[1, 1]
    ...:     ax5.hist(data, bins=50, edgecolor='black', alpha=0.7)
    ...:     ax5.set_title('Wide-ranging Data: Linear Bins\n(Tail invisible!)')
    ...:     ax5.set_xlabel('Value')
    ...:     ax5.set_ylabel('Count')
    ...:     
    ...:     # Logarithmic bins - shows full distribution
    ...:     ax6 = axes[1, 2]
    ...:     ax6.hist(data, bins=np.logspace(np.log10(data.min()), 
    ...:                                     np.log10(data.max()), 50),
    ...:              edgecolor='black', alpha=0.7)
    ...:     ax6.set_xscale('log')
    ...:     ax6.set_title('Wide-ranging Data: Log Bins\n(Full distribution visible!)')
    ...:     ax6.set_xlabel('Value (log scale)')
    ...:     ax6.set_ylabel('Count')
    ...:     
    ...:     plt.suptitle('Always Experiment with Different Scales!', 
    ...:                  fontsize=16, fontweight='bold')
    ...:     plt.tight_layout()
    ...:     
    ...:     return fig

In [140]: fig = demonstrate_scale_choices()
In [141]: plt.show()
```

### 📦 **Computational Thinking Box: The Scale Selection Decision Tree**

```
PATTERN: Systematic Scale Selection

Don't guess! Follow this decision process:

1. What is the range of your data?
   - Less than 2 orders of magnitude → Linear usually fine
   - More than 3 orders of magnitude → Consider log scale
   
2. What relationship are you investigating?
   - Power law (y ∝ x^n) → Use log-log (becomes straight line with slope n)
   - Exponential (y ∝ e^x) → Use semi-log (becomes straight line)
   - Linear (y ∝ x) → Use linear scale
   
3. What values does your data contain?
   - All positive → Any scale works
   - Contains zero → Can't use log for that axis
   - Contains negative → Need linear or symlog
   
4. What are you trying to emphasize?
   - Absolute differences → Linear scale
   - Relative differences/ratios → Log scale
   - Both large and small features → Log scale

The Golden Rule: ALWAYS try multiple scales when exploring new data!
What looks like noise in linear scale might be a clear pattern in log scale.

Common in astronomy:
- Luminosity functions → log-log
- Magnitude distributions → semi-log
- Spectra → often log-log
- Light curves → usually linear time, sometimes log flux
- Radial profiles → often log-linear
```

### The Experimentation Workflow

When you encounter new data, don't just use the default linear scale. Develop a systematic exploration workflow:

```python
In [142]: def explore_with_scales(x, y, title="My Data"):
    ...:     """
    ...:     Quick function to try all scale combinations.
    ...:     Use this when exploring new datasets!
    ...:     """
    ...:     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ...:     
    ...:     # Try all four combinations
    ...:     scales = [
    ...:         ('linear', 'linear', 'Linear-Linear'),
    ...:         ('log', 'linear', 'Log-Linear (semilogx)'),
    ...:         ('linear', 'log', 'Linear-Log (semilogy)'),
    ...:         ('log', 'log', 'Log-Log')
    ...:     ]
    ...:     
    ...:     for ax, (xscale, yscale, scale_name) in zip(axes.flat, scales):
    ...:         # Skip if data has zeros/negatives and we're trying log
    ...:         if xscale == 'log' and (x <= 0).any():
    ...:             ax.text(0.5, 0.5, 'Cannot use log scale\n(data contains ≤0)', 
    ...:                    ha='center', va='center', transform=ax.transAxes)
    ...:             ax.set_title(f'{scale_name} - Invalid')
    ...:             continue
    ...:         if yscale == 'log' and (y <= 0).any():
    ...:             ax.text(0.5, 0.5, 'Cannot use log scale\n(data contains ≤0)', 
    ...:                    ha='center', va='center', transform=ax.transAxes)
    ...:             ax.set_title(f'{scale_name} - Invalid')
    ...:             continue
    ...:         
    ...:         ax.plot(x, y, 'o-', markersize=3, alpha=0.7)
    ...:         ax.set_xscale(xscale)
    ...:         ax.set_yscale(yscale)
    ...:         ax.set_title(scale_name)
    ...:         ax.grid(True, which="both", alpha=0.3)
    ...:         ax.set_xlabel('X')
    ...:         ax.set_ylabel('Y')
    ...:     
    ...:     plt.suptitle(f'{title}: Try All Scales!', fontsize=14, fontweight='bold')
    ...:     plt.tight_layout()
    ...:     
    ...:     return fig

# Use it like this for any new dataset:
# x = your_data_x
# y = your_data_y
# explore_with_scales(x, y, "My Dataset Name")
```

### For pyplot Users: Translation Guide

If you're transitioning from pyplot, here's a quick reference to help you convert your code:

```python
# PYPLOT WAY → OBJECT-ORIENTED WAY

# Creating figures
plt.figure() → fig, ax = plt.subplots()
plt.subplot(2,1,1) → fig, (ax1, ax2) = plt.subplots(2, 1)

# Plotting
plt.plot(x, y) → ax.plot(x, y)
plt.scatter(x, y) → ax.scatter(x, y)
plt.hist(data) → ax.hist(data)
plt.bar(x, y) → ax.bar(x, y)

# Scales (this is where OOP shines!)
plt.loglog(x, y) → ax.loglog(x, y) OR ax.plot(x, y); ax.set_xscale('log'); ax.set_yscale('log')
plt.semilogx(x, y) → ax.semilogx(x, y) OR ax.plot(x, y); ax.set_xscale('log')
plt.semilogy(x, y) → ax.semilogy(x, y) OR ax.plot(x, y); ax.set_yscale('log')

# Labels and titles
plt.xlabel('X') → ax.set_xlabel('X')
plt.ylabel('Y') → ax.set_ylabel('Y')
plt.title('Title') → ax.set_title('Title')

# Limits
plt.xlim([0, 10]) → ax.set_xlim([0, 10])
plt.ylim([0, 10]) → ax.set_ylim([0, 10])

# Other common operations
plt.legend() → ax.legend()
plt.grid() → ax.grid()
plt.tight_layout() → fig.tight_layout() or plt.tight_layout()
plt.savefig() → fig.savefig()
```

### Line Plots: Time Series and Continuous Functions

Line plots are ideal for showing trends and relationships in continuous data:

```python
In [143]: def create_scientific_lineplot():
    ...:     """Demonstrate professional line plot with multiple datasets."""
    ...:     # Generate example time series data
    ...:     time = np.linspace(0, 10, 100)
    ...:     signal1 = np.sin(2 * np.pi * 0.5 * time) * np.exp(-time/10)
    ...:     signal2 = np.cos(2 * np.pi * 0.3 * time) * np.exp(-time/15)
    ...:     noise = np.random.normal(0, 0.05, len(time))
    ...:     
    ...:     fig, ax = plt.subplots(figsize=(12, 6))
    ...:     
    ...:     # Plot with different styles for clarity
    ...:     ax.plot(time, signal1, 'b-', linewidth=2, label='Damped sine wave')
    ...:     ax.plot(time, signal2, 'r--', linewidth=2, label='Damped cosine wave')
    ...:     ax.plot(time, signal1 + signal2 + noise, 'g:', linewidth=1, 
    ...:             alpha=0.7, label='Combined + noise')
    ...:     
    ...:     # Highlight specific regions
    ...:     ax.axvspan(2, 4, alpha=0.2, color='yellow', label='Region of interest')
    ...:     
    ...:     # Professional formatting
    ...:     ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ...:     ax.set_ylabel('Amplitude (V)', fontsize=12, fontweight='bold')
    ...:     ax.set_title('Temporal Evolution of Coupled Oscillators', 
    ...:                  fontsize=14, fontweight='bold')
    ...:     
    ...:     # Add grid with different styles for major/minor
    ...:     ax.grid(True, which='major', linestyle='-', alpha=0.2)
    ...:     ax.grid(True, which='minor', linestyle=':', alpha=0.1)
    ...:     ax.minorticks_on()
    ...:     
    ...:     # Legend with optimal placement
    ...:     ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ...:     
    ...:     # Add annotation for key feature
    ...:     max_idx = np.argmax(signal1 + signal2)
    ...:     ax.annotate('Maximum amplitude',
    ...:                 xy=(time[max_idx], (signal1 + signal2)[max_idx]),
    ...:                 xytext=(time[max_idx] + 2, (signal1 + signal2)[max_idx] + 0.5),
    ...:                 arrowprops=dict(arrowstyle='->', color='black', lw=1),
    ...:                 fontsize=10,
    ...:                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    ...:     
    ...:     plt.tight_layout()
    ...:     return fig, ax

In [141]: fig, ax = create_scientific_lineplot()
In [142]: plt.show()
```

### Scatter Plots: Correlations and Relationships

Scatter plots reveal relationships between variables:

```python
In [143]: def create_scientific_scatter():
    ...:     """Create publication-quality scatter plot with regression."""
    ...:     # Generate correlated data with outliers
    ...:     np.random.seed(42)
    ...:     n_points = 150
    ...:     x = np.random.randn(n_points)
    ...:     y = 2 * x + 1 + np.random.randn(n_points) * 0.5
    ...:     
    ...:     # Add some outliers
    ...:     n_outliers = 10
    ...:     x = np.append(x, np.random.uniform(-3, 3, n_outliers))
    ...:     y = np.append(y, np.random.uniform(-4, 6, n_outliers))
    ...:     
    ...:     fig, ax = plt.subplots(figsize=(10, 8))
    ...:     
    ...:     # Main scatter plot with color gradient
    ...:     scatter = ax.scatter(x[:n_points], y[:n_points], 
    ...:                          c=np.sqrt(x[:n_points]**2 + y[:n_points]**2),
    ...:                          cmap='viridis', s=50, alpha=0.7, 
    ...:                          edgecolors='black', linewidth=0.5,
    ...:                          label='Main data')
    ...:     
    ...:     # Outliers in different style
    ...:     ax.scatter(x[n_points:], y[n_points:], 
    ...:               color='red', marker='^', s=100, 
    ...:               edgecolors='darkred', linewidth=1,
    ...:               label='Outliers')
    ...:     
    ...:     # Add regression line
    ...:     z = np.polyfit(x[:n_points], y[:n_points], 1)
    ...:     p = np.poly1d(z)
    ...:     x_line = np.linspace(x.min(), x.max(), 100)
    ...:     ax.plot(x_line, p(x_line), 'r-', linewidth=2, 
    ...:             label=f'Fit: y = {z[0]:.2f}x + {z[1]:.2f}')
    ...:     
    ...:     # Add confidence interval
    ...:     from scipy import stats
    ...:     slope, intercept, r_value, p_value, std_err = stats.linregress(x[:n_points], y[:n_points])
    ...:     predict_mean_se = std_err * np.sqrt(1/len(x[:n_points]) + 
    ...:                                         (x_line - np.mean(x[:n_points]))**2 / 
    ...:                                         np.sum((x[:n_points] - np.mean(x[:n_points]))**2))
    ...:     margin = 1.96 * predict_mean_se
    ...:     ax.fill_between(x_line, p(x_line) - margin, p(x_line) + margin, 
    ...:                     color='gray', alpha=0.2, label='95% CI')
    ...:     
    ...:     # Colorbar
    ...:     cbar = plt.colorbar(scatter, ax=ax)
    ...:     cbar.set_label('Distance from origin', fontweight='bold')
    ...:     
    ...:     # Labels and formatting
    ...:     ax.set_xlabel('Independent Variable (X)', fontsize=12, fontweight='bold')
    ...:     ax.set_ylabel('Dependent Variable (Y)', fontsize=12, fontweight='bold')
    ...:     ax.set_title(f'Correlation Analysis (r² = {r_value**2:.3f}, p = {p_value:.4f})',
    ...:                  fontsize=14, fontweight='bold')
    ...:     
    ...:     # Add grid
    ...:     ax.grid(True, alpha=0.3, linestyle='--')
    ...:     
    ...:     # Legend
    ...:     ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ...:     
    ...:     plt.tight_layout()
    ...:     return fig, ax

In [144]: fig, ax = create_scientific_scatter()
In [145]: plt.show()
```

### Bar Plots and Error Bars: Comparing Categories

Bar plots are excellent for comparing discrete categories:

```python
In [146]: def create_scientific_barplot():
    ...:     """Create publication-quality bar plot with error bars."""
    ...:     # Example: Comparing experimental conditions
    ...:     categories = ['Control', 'Treatment A', 'Treatment B', 'Treatment C']
    ...:     means = [100, 125, 145, 132]
    ...:     stds = [10, 12, 15, 11]
    ...:     n_samples = [30, 28, 32, 29]
    ...:     
    ...:     # Calculate standard error
    ...:     sems = [s/np.sqrt(n) for s, n in zip(stds, n_samples)]
    ...:     
    ...:     fig, ax = plt.subplots(figsize=(10, 7))
    ...:     
    ...:     # Create bars with error bars
    ...:     x_pos = np.arange(len(categories))
    ...:     bars = ax.bar(x_pos, means, yerr=sems, capsize=5,
    ...:                   color=['gray', 'skyblue', 'lightgreen', 'salmon'],
    ...:                   edgecolor='black', linewidth=1.5,
    ...:                   error_kw={'linewidth': 2, 'ecolor': 'black'})
    ...:     
    ...:     # Add value labels on bars
    ...:     for bar, mean, sem in zip(bars, means, sems):
    ...:         height = bar.get_height()
    ...:         ax.text(bar.get_x() + bar.get_width()/2., height + sem,
    ...:                f'{mean:.1f}±{sem:.1f}',
    ...:                ha='center', va='bottom', fontsize=10)
    ...:     
    ...:     # Statistical significance indicators
    ...:     # Example: Add significance bars
    ...:     def add_significance_bar(ax, x1, x2, y, sig_level):
    ...:         ax.plot([x1, x1, x2, x2], [y, y+2, y+2, y], 'k-', linewidth=1)
    ...:         ax.text((x1+x2)/2, y+2, sig_level, ha='center', va='bottom')
    ...:     
    ...:     add_significance_bar(ax, 0, 1, 115, 'n.s.')
    ...:     add_significance_bar(ax, 0, 2, 165, '***')
    ...:     add_significance_bar(ax, 1, 2, 160, '*')
    ...:     
    ...:     # Formatting
    ...:     ax.set_xticks(x_pos)
    ...:     ax.set_xticklabels(categories, fontsize=11)
    ...:     ax.set_ylabel('Response (arbitrary units)', fontsize=12, fontweight='bold')
    ...:     ax.set_title('Comparison of Experimental Conditions', fontsize=14, fontweight='bold')
    ...:     
    ...:     # Add sample size annotations
    ...:     for i, (bar, n) in enumerate(zip(bars, n_samples)):
    ...:         ax.text(bar.get_x() + bar.get_width()/2., 5,
    ...:                f'n={n}', ha='center', va='bottom', fontsize=9, style='italic')
    ...:     
    ...:     # Grid
    ...:     ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ...:     ax.set_axisbelow(True)
    ...:     
    ...:     # Remove top and right spines
    ...:     ax.spines['top'].set_visible(False)
    ...:     ax.spines['right'].set_visible(False)
    ...:     
    ...:     # Add legend for significance levels
    ...:     ax.text(0.98, 0.98, 'Significance:\n* p<0.05\n** p<0.01\n*** p<0.001\nn.s. not significant',
    ...:            transform=ax.transAxes, fontsize=9,
    ...:            verticalalignment='top', horizontalalignment='right',
    ...:            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ...:     
    ...:     plt.tight_layout()
    ...:     return fig, ax

In [147]: fig, ax = create_scientific_barplot()
In [148]: plt.show()
```

## 8.5 Multi-Panel Figures: Telling Complete Stories

Scientific papers often require complex multi-panel figures that tell complete stories. Mastering subplot layouts is essential for comprehensive data presentation.

### Creating Complex Layouts with GridSpec

```python
In [149]: def create_complex_figure():
    ...:     """Create a publication-quality multi-panel figure."""
    ...:     import matplotlib.gridspec as gridspec
    ...:     
    ...:     # Create figure with custom layout
    ...:     fig = plt.figure(figsize=(14, 10))
    ...:     gs = gridspec.GridSpec(3, 3, figure=fig, 
    ...:                           height_ratios=[1, 1, 0.8],
    ...:                           width_ratios=[1.2, 1, 1])
    ...:     
    ...:     # Panel A: Time series
    ...:     ax1 = fig.add_subplot(gs[0, :])
    ...:     time = np.linspace(0, 10, 500)
    ...:     signal = np.sin(2*np.pi*time) * np.exp(-time/5)
    ...:     ax1.plot(time, signal, 'b-', linewidth=1.5)
    ...:     ax1.fill_between(time, signal, alpha=0.3)
    ...:     ax1.set_xlabel('Time (s)')
    ...:     ax1.set_ylabel('Amplitude')
    ...:     ax1.set_title('A. Temporal Evolution', loc='left', fontweight='bold')
    ...:     ax1.grid(True, alpha=0.3)
    ...:     
    ...:     # Panel B: Scatter plot
    ...:     ax2 = fig.add_subplot(gs[1, 0])
    ...:     x = np.random.randn(100)
    ...:     y = 2*x + np.random.randn(100)*0.5
    ...:     ax2.scatter(x, y, alpha=0.6, s=30)
    ...:     ax2.set_xlabel('Variable X')
    ...:     ax2.set_ylabel('Variable Y')
    ...:     ax2.set_title('B. Correlation', loc='left', fontweight='bold')
    ...:     ax2.grid(True, alpha=0.3)
    ...:     
    ...:     # Panel C: Histogram
    ...:     ax3 = fig.add_subplot(gs[1, 1])
    ...:     data = np.random.normal(100, 15, 500)
    ...:     ax3.hist(data, bins=30, edgecolor='black', alpha=0.7)
    ...:     ax3.set_xlabel('Value')
    ...:     ax3.set_ylabel('Frequency')
    ...:     ax3.set_title('C. Distribution', loc='left', fontweight='bold')
    ...:     ax3.grid(True, alpha=0.3, axis='y')
    ...:     
    ...:     # Panel D: Heatmap
    ...:     ax4 = fig.add_subplot(gs[1, 2])
    ...:     data_2d = np.random.randn(10, 10)
    ...:     im = ax4.imshow(data_2d, cmap='RdBu_r', aspect='auto')
    ...:     ax4.set_xlabel('Column')
    ...:     ax4.set_ylabel('Row')
    ...:     ax4.set_title('D. 2D Pattern', loc='left', fontweight='bold')
    ...:     plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    ...:     
    ...:     # Panel E: Bar plot comparison
    ...:     ax5 = fig.add_subplot(gs[2, :2])
    ...:     categories = ['A', 'B', 'C', 'D', 'E']
    ...:     values1 = np.random.randint(50, 100, 5)
    ...:     values2 = np.random.randint(60, 110, 5)
    ...:     x = np.arange(len(categories))
    ...:     width = 0.35
    ...:     ax5.bar(x - width/2, values1, width, label='Group 1', color='skyblue')
    ...:     ax5.bar(x + width/2, values2, width, label='Group 2', color='orange')
    ...:     ax5.set_xlabel('Category')
    ...:     ax5.set_ylabel('Value')
    ...:     ax5.set_title('E. Group Comparison', loc='left', fontweight='bold')
    ...:     ax5.set_xticks(x)
    ...:     ax5.set_xticklabels(categories)
    ...:     ax5.legend()
    ...:     ax5.grid(True, alpha=0.3, axis='y')
    ...:     
    ...:     # Panel F: Box plot
    ...:     ax6 = fig.add_subplot(gs[2, 2])
    ...:     data_box = [np.random.normal(100, std, 100) for std in range(10, 30, 5)]
    ...:     bp = ax6.boxplot(data_box, patch_artist=True)
    ...:     for patch in bp['boxes']:
    ...:         patch.set_facecolor('lightblue')
    ...:     ax6.set_xlabel('Group')
    ...:     ax6.set_ylabel('Value')
    ...:     ax6.set_title('F. Variability', loc='left', fontweight='bold')
    ...:     ax6.grid(True, alpha=0.3, axis='y')
    ...:     
    ...:     # Adjust layout
    ...:     plt.tight_layout()
    ...:     
    ...:     # Add overall figure label
    ...:     fig.suptitle('Comprehensive Data Analysis', fontsize=16, fontweight='bold', y=1.02)
    ...:     
    ...:     return fig

In [150]: fig = create_complex_figure()
In [151]: plt.show()
```

### 📦 **Computational Thinking Box: Figure Design Principles**

```
PATTERN: Effective Multi-Panel Figure Design

Multi-panel figures should tell a coherent scientific story.
Each panel should contribute unique information while maintaining
visual consistency across the entire figure.

Design Principles:
1. Logical Flow: Arrange panels to guide the reader's eye
   - Left to right, top to bottom (Western reading pattern)
   - Group related panels together
   - Use consistent panel labels (A, B, C...)

2. Visual Hierarchy: Make important elements stand out
   - Larger panels for main results
   - Smaller panels for supporting data
   - Use color/size to emphasize key findings

3. Consistency: Maintain style across panels
   - Same font sizes and families
   - Consistent color schemes
   - Aligned axes when comparing data

4. Information Density: Balance detail with clarity
   - Don't overcrowd individual panels
   - Use white space effectively
   - Consider splitting into multiple figures if needed

5. Accessibility: Ensure readability for all
   - Use colorblind-friendly palettes
   - Sufficient contrast for printing
   - Clear labels and legends

This pattern applies to all scientific communication:
- Conference posters (visual flow guides viewers)
- Presentations (one main point per slide)
- Papers (figures complement text narrative)
```

## 8.6 Advanced Customization and Special Plots

Sometimes standard plots aren't enough. Let's explore advanced customization techniques and specialized plot types.

### Custom Colormaps and Styles

```python
In [152]: def demonstrate_custom_styling():
    ...:     """Show advanced customization techniques."""
    ...:     # Create custom colormap
    ...:     from matplotlib.colors import LinearSegmentedColormap
    ...:     
    ...:     # Define custom colormap (e.g., for temperature data)
    ...:     colors = ['darkblue', 'blue', 'cyan', 'yellow', 'red', 'darkred']
    ...:     n_bins = 100
    ...:     cmap_custom = LinearSegmentedColormap.from_list('temperature', colors, N=n_bins)
    ...:     
    ...:     # Generate example data
    ...:     x = np.linspace(-3, 3, 100)
    ...:     y = np.linspace(-3, 3, 100)
    ...:     X, Y = np.meshgrid(x, y)
    ...:     Z = np.exp(-(X**2 + Y**2)/2) * np.cos(2*X) * np.cos(2*Y)
    ...:     
    ...:     # Create figure with custom style
    ...:     with plt.style.context('seaborn-v0_8-darkgrid'):
    ...:         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ...:         
    ...:         # Different colormaps for comparison
    ...:         cmaps = [cmap_custom, 'viridis', 'RdBu_r', 'twilight']
    ...:         titles = ['Custom Temperature', 'Viridis (Default)', 
    ...:                  'RdBu (Diverging)', 'Twilight (Cyclic)']
    ...:         
    ...:         for ax, cmap, title in zip(axes.flat, cmaps, titles):
    ...:             im = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
    ...:             ax.set_title(title, fontweight='bold')
    ...:             ax.set_xlabel('X')
    ...:             ax.set_ylabel('Y')
    ...:             plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ...:         
    ...:         plt.suptitle('Colormap Selection for Scientific Data', 
    ...:                     fontsize=14, fontweight='bold')
    ...:         plt.tight_layout()
    ...:     
    ...:     return fig

In [153]: fig = demonstrate_custom_styling()
In [154]: plt.show()
```

### 3D Plots for Scientific Data

```python
In [155]: def create_3d_surface():
    ...:     """Create publication-quality 3D surface plot."""
    ...:     from mpl_toolkits.mplot3d import Axes3D
    ...:     
    ...:     fig = plt.figure(figsize=(14, 6))
    ...:     
    ...:     # First subplot: Surface plot
    ...:     ax1 = fig.add_subplot(121, projection='3d')
    ...:     
    ...:     # Generate data
    ...:     x = np.linspace(-5, 5, 100)
    ...:     y = np.linspace(-5, 5, 100)
    ...:     X, Y = np.meshgrid(x, y)
    ...:     Z = np.sin(np.sqrt(X**2 + Y**2)) / np.sqrt(X**2 + Y**2 + 1)
    ...:     
    ...:     # Surface plot
    ...:     surf = ax1.plot_surface(X, Y, Z, cmap='viridis', 
    ...:                             linewidth=0, antialiased=True, alpha=0.9)
    ...:     ax1.set_xlabel('X', fontweight='bold')
    ...:     ax1.set_ylabel('Y', fontweight='bold')
    ...:     ax1.set_zlabel('Z', fontweight='bold')
    ...:     ax1.set_title('3D Surface Plot', fontweight='bold')
    ...:     ax1.view_init(elev=30, azim=45)
    ...:     
    ...:     # Add contour projections
    ...:     ax1.contour(X, Y, Z, zdir='z', offset=-0.5, cmap='viridis', alpha=0.5)
    ...:     
    ...:     # Second subplot: Contour plot
    ...:     ax2 = fig.add_subplot(122)
    ...:     contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    ...:     ax2.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ...:     ax2.set_xlabel('X', fontweight='bold')
    ...:     ax2.set_ylabel('Y', fontweight='bold')
    ...:     ax2.set_title('2D Contour Projection', fontweight='bold')
    ...:     ax2.set_aspect('equal')
    ...:     
    ...:     # Add colorbar
    ...:     plt.colorbar(contour, ax=ax2, label='Z value')
    ...:     
    ...:     plt.suptitle('3D Data Visualization', fontsize=14, fontweight='bold')
    ...:     plt.tight_layout()
    ...:     
    ...:     return fig

In [156]: fig = create_3d_surface()
In [157]: plt.show()
```

## 8.7 Performance Optimization and Large Datasets

When dealing with large datasets, visualization performance becomes crucial. Let's explore techniques for efficient plotting.

### Handling Large Datasets

```python
In [158]: def plot_large_dataset_efficiently():
    ...:     """Demonstrate techniques for plotting large datasets."""
    ...:     # Generate large dataset
    ...:     n_points = 1_000_000
    ...:     x = np.random.randn(n_points)
    ...:     y = 2*x + np.random.randn(n_points) * 0.5
    ...:     
    ...:     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ...:     
    ...:     # Method 1: Rasterization for vector formats
    ...:     ax1 = axes[0, 0]
    ...:     ax1.scatter(x[::100], y[::100], alpha=0.5, s=1, rasterized=True)
    ...:     ax1.set_title('Rasterized Scatter (PDF-friendly)', fontweight='bold')
    ...:     ax1.set_xlabel('X')
    ...:     ax1.set_ylabel('Y')
    ...:     
    ...:     # Method 2: Hexbin for density
    ...:     ax2 = axes[0, 1]
    ...:     hexbin = ax2.hexbin(x, y, gridsize=50, cmap='YlOrRd')
    ...:     ax2.set_title('Hexbin Plot (Density visualization)', fontweight='bold')
    ...:     ax2.set_xlabel('X')
    ...:     ax2.set_ylabel('Y')
    ...:     plt.colorbar(hexbin, ax=ax2, label='Count')
    ...:     
    ...:     # Method 3: 2D Histogram
    ...:     ax3 = axes[1, 0]
    ...:     hist2d = ax3.hist2d(x, y, bins=100, cmap='Blues')
    ...:     ax3.set_title('2D Histogram', fontweight='bold')
    ...:     ax3.set_xlabel('X')
    ...:     ax3.set_ylabel('Y')
    ...:     plt.colorbar(hist2d[3], ax=ax3, label='Count')
    ...:     
    ...:     # Method 4: Contour plot of density
    ...:     ax4 = axes[1, 1]
    ...:     from scipy.stats import gaussian_kde
    ...:     
    ...:     # Sample for KDE (full dataset would be too slow)
    ...:     sample_idx = np.random.choice(n_points, 10000, replace=False)
    ...:     xy = np.vstack([x[sample_idx], y[sample_idx]])
    ...:     z = gaussian_kde(xy)(xy)
    ...:     
    ...:     scatter = ax4.scatter(x[sample_idx], y[sample_idx], c=z, s=1, 
    ...:                          cmap='viridis', rasterized=True)
    ...:     ax4.set_title('KDE Density Plot', fontweight='bold')
    ...:     ax4.set_xlabel('X')
    ...:     ax4.set_ylabel('Y')
    ...:     plt.colorbar(scatter, ax=ax4, label='Density')
    ...:     
    ...:     plt.suptitle(f'Efficient Visualization of {n_points:,} Points', 
    ...:                 fontsize=14, fontweight='bold')
    ...:     plt.tight_layout()
    ...:     
    ...:     return fig

In [159]: fig = plot_large_dataset_efficiently()
In [160]: plt.show()
```

### 🔊 **Performance Profile: Plotting Speed Comparison**

```python
In [161]: import time

In [162]: def benchmark_plotting_methods():
    ...:     """Compare performance of different plotting methods."""
    ...:     sizes = [100, 1000, 10000, 100000]
    ...:     methods = ['scatter', 'plot', 'hexbin', 'hist2d']
    ...:     times = {method: [] for method in methods}
    ...:     
    ...:     for n in sizes:
    ...:         x = np.random.randn(n)
    ...:         y = np.random.randn(n)
    ...:         
    ...:         # Scatter plot
    ...:         fig, ax = plt.subplots()
    ...:         start = time.perf_counter()
    ...:         ax.scatter(x, y, alpha=0.5, s=1)
    ...:         times['scatter'].append(time.perf_counter() - start)
    ...:         plt.close(fig)
    ...:         
    ...:         # Line plot
    ...:         fig, ax = plt.subplots()
    ...:         start = time.perf_counter()
    ...:         ax.plot(x, y, 'o', markersize=1, alpha=0.5)
    ...:         times['plot'].append(time.perf_counter() - start)
    ...:         plt.close(fig)
    ...:         
    ...:         # Hexbin
    ...:         fig, ax = plt.subplots()
    ...:         start = time.perf_counter()
    ...:         ax.hexbin(x, y, gridsize=30)
    ...:         times['hexbin'].append(time.perf_counter() - start)
    ...:         plt.close(fig)
    ...:         
    ...:         # 2D histogram
    ...:         fig, ax = plt.subplots()
    ...:         start = time.perf_counter()
    ...:         ax.hist2d(x, y, bins=30)
    ...:         times['hist2d'].append(time.perf_counter() - start)
    ...:         plt.close(fig)
    ...:     
    ...:     # Display results
    ...:     print("Plotting Performance (seconds):")
    ...:     print(f"{'N Points':<10} " + " ".join(f"{m:<10}" for m in methods))
    ...:     for i, n in enumerate(sizes):
    ...:         print(f"{n:<10} " + " ".join(f"{times[m][i]:<10.4f}" for m in methods))
    ...:     
    ...:     return times

In [163]: times = benchmark_plotting_methods()
Plotting Performance (seconds):
N Points   scatter    plot       hexbin     hist2d    
100        0.0234     0.0156     0.0312     0.0234    
1000       0.0391     0.0234     0.0391     0.0312    
10000      0.2344     0.0625     0.0469     0.0391    
100000     2.3438     0.4688     0.0781     0.0625    
```

## 8.8 Common Pitfalls and Debugging

Let's address the most common mistakes students make with Matplotlib and how to avoid them.

### ⚠️ **Common Bug Alert: The pyplot State Machine Trap**

```python
# WRONG: Relying on pyplot's hidden state
def bad_plotting_function(data):
    """This function has hidden dependencies on global state."""
    plt.plot(data)  # Which figure? Which axes?
    plt.xlabel('X')  # Modifies "current" axes
    plt.title('Title')  # But what if another subplot was created?
    # No return value - can't customize further!

# CORRECT: Explicit axes handling
def good_plotting_function(data, ax=None):
    """This function is explicit about what it modifies."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    ax.plot(data)
    ax.set_xlabel('X')
    ax.set_title('Title')
    
    return fig, ax  # Caller can continue customizing

# Example of the problem:
plt.figure()
plt.subplot(2, 1, 1)
bad_plotting_function([1, 2, 3])  # Goes to subplot!
plt.subplot(2, 1, 2)
plt.plot([3, 2, 1])
# The function modified subplot 2, not subplot 1!
```

### Common Mistakes and Solutions

```python
In [164]: # Mistake 1: Not saving figures at the right DPI
    ...: # WRONG
    ...: fig, ax = plt.subplots()
    ...: ax.plot([1, 2, 3])
    ...: fig.savefig('bad_figure.png')  # Default DPI = 100, looks terrible in print
    ...: 
    ...: # CORRECT
    ...: fig, ax = plt.subplots()
    ...: ax.plot([1, 2, 3])
    ...: fig.savefig('good_figure.png', dpi=300, bbox_inches='tight')
    ...: fig.savefig('good_figure.pdf')  # Vector format for publications

In [165]: # Mistake 2: Forgetting to clear figures in loops
    ...: # WRONG - Memory leak!
    ...: for i in range(100):
    ...:     plt.figure()
    ...:     plt.plot(np.random.randn(100))
    ...:     plt.savefig(f'figure_{i}.png')
    ...:     # Figure never closed - accumulates in memory!
    ...: 
    ...: # CORRECT
    ...: for i in range(100):
    ...:     fig, ax = plt.subplots()
    ...:     ax.plot(np.random.randn(100))
    ...:     fig.savefig(f'figure_{i}.png')
    ...:     plt.close(fig)  # Explicitly close to free memory

In [166]: # Mistake 3: Modifying shared default arguments
    ...: # WRONG
    ...: def plot_with_style(data, style_dict={}):  # Mutable default!
    ...:     style_dict['color'] = 'blue'  # Modifies the default!
    ...:     plt.plot(data, **style_dict)
    ...: 
    ...: # CORRECT
    ...: def plot_with_style(data, style_dict=None):
    ...:     if style_dict is None:
    ...:         style_dict = {}
    ...:     style_dict = dict(style_dict)  # Make a copy
    ...:     style_dict['color'] = 'blue'
    ...:     plt.plot(data, **style_dict)
```

### 🛠️ **Debug This!**

This code tries to create a multi-panel figure but has several bugs. Can you find them?

```python
def buggy_multipanel_figure(data1, data2):
    """This function has multiple common bugs."""
    plt.subplot(2, 1, 1)
    plt.plot(data1)
    plt.title = 'First Dataset'  # Bug 1
    
    plt.subplot(2, 1, 2)
    ax = plt.plot(data2)  # Bug 2
    ax.set_xlabel('Time')  # Bug 3
    
    plt.tight_layout
    plt.savefig('figure.png', dpi=50)  # Bug 4
    
    return ax  # Bug 5

# Test it
data1 = np.random.randn(100)
data2 = np.random.randn(100)
result = buggy_multipanel_figure(data1, data2)
```

<details>
<summary>Bugs and Solutions</summary>

**Bug 1**: `plt.title = 'First Dataset'` assigns to the title attribute instead of calling the function.
- Fix: `plt.title('First Dataset')`

**Bug 2**: `plt.plot()` returns a list of Line2D objects, not an axes object.
- Fix: `ax = plt.gca()` or use the OO interface from the start

**Bug 3**: `ax.set_xlabel()` fails because `ax` is a list of lines, not an axes.
- Fix: Get the actual axes object

**Bug 4**: `plt.tight_layout` missing parentheses, and DPI too low for publication.
- Fix: `plt.tight_layout()` and use `dpi=300`

**Bug 5**: Returns the wrong object (list of lines instead of figure/axes).
- Fix: Return the figure or axes objects

**Corrected version**:
```python
def fixed_multipanel_figure(data1, data2):
    """Corrected version using OO interface."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(data1)
    ax1.set_title('First Dataset')
    
    ax2.plot(data2)
    ax2.set_xlabel('Time')
    
    plt.tight_layout()
    fig.savefig('figure.png', dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)
```

The key lesson: Use the object-oriented interface to avoid these ambiguities!

</details>

## 8.9 Integration with NumPy and Scientific Workflow

Matplotlib and NumPy are designed to work together seamlessly. Understanding this integration is crucial for efficient scientific visualization.

### Direct NumPy Integration

```python
In [167]: def demonstrate_numpy_integration():
    ...:     """Show how Matplotlib and NumPy work together."""
    ...:     # NumPy arrays are the native data format for Matplotlib
    ...:     x = np.linspace(0, 10, 1000)
    ...:     y = np.sin(x) * np.exp(-x/10)
    ...:     
    ...:     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ...:     
    ...:     # Direct array operations in plotting
    ...:     ax1 = axes[0, 0]
    ...:     ax1.plot(x, y, label='Original')
    ...:     ax1.plot(x, y + 0.1*np.random.randn(len(x)), alpha=0.5, label='With noise')
    ...:     ax1.fill_between(x, y - 0.1, y + 0.1, alpha=0.3, label='Uncertainty band')
    ...:     ax1.set_title('Array Operations in Plotting')
    ...:     ax1.legend()
    ...:     
    ...:     # Using NumPy for data transformation
    ...:     ax2 = axes[0, 1]
    ...:     fft = np.fft.fft(y)
    ...:     freq = np.fft.fftfreq(len(y), x[1] - x[0])
    ...:     ax2.semilogy(freq[:len(freq)//2], np.abs(fft)[:len(freq)//2])
    ...:     ax2.set_title('FFT Spectrum')
    ...:     ax2.set_xlabel('Frequency')
    ...:     ax2.set_ylabel('Amplitude')
    ...:     
    ...:     # Image display (2D NumPy arrays)
    ...:     ax3 = axes[1, 0]
    ...:     image_data = np.random.randn(50, 50)
    ...:     im = ax3.imshow(image_data, cmap='viridis', interpolation='nearest')
    ...:     ax3.set_title('2D Array as Image')
    ...:     plt.colorbar(im, ax=ax3)
    ...:     
    ...:     # Masked arrays for missing data
    ...:     ax4 = axes[1, 1]
    ...:     masked_y = np.ma.masked_where(np.abs(y) < 0.1, y)
    ...:     ax4.plot(x, masked_y, 'o-', markersize=2)
    ...:     ax4.set_title('Masked Array (gaps where |y| < 0.1)')
    ...:     ax4.set_xlabel('X')
    ...:     ax4.set_ylabel('Y')
    ...:     
    ...:     plt.suptitle('NumPy-Matplotlib Integration', fontsize=14, fontweight='bold')
    ...:     plt.tight_layout()
    ...:     
    ...:     return fig

In [168]: fig = demonstrate_numpy_integration()
In [169]: plt.show()
```

## 8.13 Best Practices and Professional Tips

Let's conclude with essential best practices for creating publication-quality figures.

### The Publication Checklist

```python
In [170]: def publication_checklist():
    ...:     """
    ...:     Essential checklist for publication-ready figures.
    ...:     
    ...:     Run through this before submitting any figure!
    ...:     """
    ...:     checklist = """
    ...:     PUBLICATION FIGURE CHECKLIST
    ...:     ============================
    ...:     
    ...:     Content and Clarity:
    ...:     □ Is the main message clear within 5 seconds?
    ...:     □ Are all axes labeled with units?
    ...:     □ Is the title informative (if allowed by journal)?
    ...:     □ Are all lines/markers distinguishable in grayscale?
    ...:     □ Is the legend clear and well-positioned?
    ...:     □ Are error bars included where appropriate?
    ...:     □ Is sample size (N) indicated?
    ...:     
    ...:     Technical Quality:
    ...:     □ Resolution ≥ 300 DPI for raster formats?
    ...:     □ Vector format (PDF/EPS) used where possible?
    ...:     □ Font size readable at publication size?
    ...:     □ Line weights visible at publication size?
    ...:     □ Colors work for colorblind readers?
    ...:     □ File size reasonable (< 10 MB)?
    ...:     
    ...:     Consistency:
    ...:     □ Font consistent across all panels?
    ...:     □ Color scheme consistent across figures?
    ...:     □ Notation matches main text?
    ...:     □ Panel labels (A, B, C) included for multi-panel?
    ...:     
    ...:     Journal Requirements:
    ...:     □ Correct figure dimensions?
    ...:     □ Acceptable file format?
    ...:     □ Within color/page limits?
    ...:     □ Copyright for any reproduced elements?
    ...:     """
    ...:     print(checklist)
    ...:     
    ...:     return checklist

In [171]: checklist = publication_checklist()
```

### Creating a Figure Style Guide

```python
In [172]: # Create a consistent style for all your publications
In [173]: def create_style_guide():
    ...:     """
    ...:     Define consistent style parameters for all figures.
    ...:     Save this as a module and import for every project!
    ...:     """
    ...:     style_params = {
    ...:         # Figure
    ...:         'figure.figsize': (10, 6),
    ...:         'figure.dpi': 100,
    ...:         'savefig.dpi': 300,
    ...:         'savefig.bbox': 'tight',
    ...:         
    ...:         # Fonts
    ...:         'font.family': 'sans-serif',
    ...:         'font.sans-serif': ['Arial', 'DejaVu Sans'],
    ...:         'font.size': 11,
    ...:         'axes.titlesize': 14,
    ...:         'axes.labelsize': 12,
    ...:         'xtick.labelsize': 10,
    ...:         'ytick.labelsize': 10,
    ...:         'legend.fontsize': 10,
    ...:         
    ...:         # Lines
    ...:         'lines.linewidth': 1.5,
    ...:         'lines.markersize': 6,
    ...:         'lines.markeredgewidth': 0.5,
    ...:         
    ...:         # Axes
    ...:         'axes.linewidth': 1.0,
    ...:         'axes.grid': True,
    ...:         'axes.grid.axis': 'both',
    ...:         'grid.alpha': 0.3,
    ...:         'grid.linestyle': '--',
    ...:         'axes.spines.top': False,
    ...:         'axes.spines.right': False,
    ...:         
    ...:         # Ticks
    ...:         'xtick.major.size': 5,
    ...:         'xtick.minor.size': 3,
    ...:         'ytick.major.size': 5,
    ...:         'ytick.minor.size': 3,
    ...:         
    ...:         # Legend
    ...:         'legend.frameon': True,
    ...:         'legend.framealpha': 0.8,
    ...:         'legend.fancybox': True,
    ...:         
    ...:         # Colors (colorblind-friendly palette)
    ...:         'axes.prop_cycle': plt.cycler('color', 
    ...:             ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', 
    ...:              '#ECE133', '#56B4E9', '#F0E442']),
    ...:     }
    ...:     
    ...:     return style_params

In [174]: style_params = create_style_guide()
In [175]: plt.rcParams.update(style_params)
```

## Key Takeaways

✅ **The object-oriented interface is essential for scientific work** - pyplot's hidden state machine leads to bugs and confusion. Always use `fig, ax = plt.subplots()` and work with axes objects directly.

✅ **Publication-quality figures require iteration** - Budget hours, not minutes, for important figures. Each iteration improves clarity, aesthetics, and scientific communication.

✅ **Build reusable plotting functions** - Don't rewrite plotting code for every figure. Create a personal library of functions that grows throughout your career.

✅ **Understanding the object hierarchy gives you complete control** - Every element (Figure, Axes, Axis, Artist) is an object with methods and attributes you can customize.

✅ **Different data types require different plot types** - Choose visualizations based on your data: line plots for time series, scatter for correlations, histograms for distributions.

✅ **Multi-panel figures tell complete stories** - Use GridSpec for complex layouts. Maintain consistency across panels while ensuring each contributes unique information.

✅ **Performance matters for large datasets** - Use rasterization, hexbin, or 2D histograms for millions of points. Profile your plotting code like any other performance-critical code.

✅ **Integration with NumPy is seamless** - Matplotlib expects NumPy arrays. Every transformation, from FFTs to masking, integrates naturally.

✅ **Professional figures require attention to detail** - Check DPI, fonts, colors, and accessibility. Use vector formats when possible. Test in grayscale.

✅ **Consistency across figures enhances professionalism** - Define a style guide and use it consistently. Your figures should be immediately recognizable as yours.

## Quick Reference Tables

### Essential Plotting Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `plt.subplots()` | Create figure and axes | `fig, ax = plt.subplots(2, 2)` |
| `ax.plot()` | Line plot | `ax.plot(x, y, 'r--')` |
| `ax.scatter()` | Scatter plot | `ax.scatter(x, y, c=z)` |
| `ax.hist()` | Histogram | `ax.hist(data, bins=30)` |
| `ax.bar()` | Bar plot | `ax.bar(categories, values)` |
| `ax.errorbar()` | Plot with error bars | `ax.errorbar(x, y, yerr=err)` |
| `ax.imshow()` | Display image/2D array | `ax.imshow(data, cmap='viridis')` |
| `ax.contour()` | Contour plot | `ax.contour(X, Y, Z)` |
| `ax.fill_between()` | Fill area between curves | `ax.fill_between(x, y1, y2)` |

### Axes Methods for Customization

| Method | Purpose | Example |
|--------|---------|---------|
| `ax.set_xlabel()` | Set x-axis label | `ax.set_xlabel('Time (s)')` |
| `ax.set_ylabel()` | Set y-axis label | `ax.set_ylabel('Voltage (V)')` |
| `ax.set_title()` | Set plot title | `ax.set_title('Results')` |
| `ax.set_xlim()` | Set x-axis limits | `ax.set_xlim([0, 10])` |
| `ax.set_ylim()` | Set y-axis limits | `ax.set_ylim([-1, 1])` |
| `ax.legend()` | Add legend | `ax.legend(loc='best')` |
| `ax.grid()` | Add grid | `ax.grid(True, alpha=0.3)` |
| `ax.set_xscale()` | Set x-axis scale | `ax.set_xscale('log')` |
| `ax.tick_params()` | Customize ticks | `ax.tick_params(labelsize=10)` |
| `ax.annotate()` | Add annotation | `ax.annotate('Peak', xy=(x, y))` |

### Figure Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `fig.savefig()` | Save figure | `fig.savefig('plot.pdf', dpi=300)` |
| `fig.suptitle()` | Overall figure title | `fig.suptitle('Main Title')` |
| `fig.tight_layout()` | Adjust subplot spacing | `fig.tight_layout()` |
| `fig.subplots_adjust()` | Manual spacing | `fig.subplots_adjust(hspace=0.3)` |
| `fig.add_subplot()` | Add single subplot | `ax = fig.add_subplot(2, 2, 1)` |
| `fig.colorbar()` | Add colorbar | `fig.colorbar(mappable, ax=ax)` |

### Common Color Maps

| Colormap | Type | Use Case |
|----------|------|----------|
| `viridis` | Sequential | Default, perceptually uniform |
| `plasma` | Sequential | Similar to viridis, warmer |
| `RdBu_r` | Diverging | Positive/negative data |
| `coolwarm` | Diverging | Temperature-like data |
| `tab10` | Qualitative | Categorical data |
| `gray` | Sequential | Grayscale images |
| `jet` | Rainbow | Avoid! Not perceptually uniform |

## Further Resources

- [Matplotlib Documentation](https://matplotlib.org/stable/index.html) - Comprehensive official documentation
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html) - Examples for every plot type
- [Matplotlib Cheatsheet](https://github.com/rougier/matplotlib-cheatsheet) - Quick reference for common tasks
- [Scientific Visualization Book](https://github.com/rougier/scientific-visualization-book) - Advanced techniques by Nicolas P. Rougier
- [Seaborn](https://seaborn.pydata.org/) - Statistical visualization built on Matplotlib
- [Colorbrewer](https://colorbrewer2.org/) - Color schemes for maps and charts
- [Ten Simple Rules for Better Figures](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833) - Essential reading for scientists

## Next Chapter Preview

With your visualization skills mastered, Chapter 9 introduces file I/O and data formats. You'll learn to work with the diverse data formats common in astronomy: FITS files for images and spectra, HDF5 for large datasets, and various text formats. You'll discover how to efficiently read, process, and write scientific data, integrating your NumPy and Matplotlib skills to build complete data analysis pipelines.

The skills you've learned—NumPy for data manipulation, Matplotlib for visualization—come together when working with real astronomical data. Whether you're analyzing telescope images, processing time series from satellites, or working with simulation outputs, you'll have the tools to handle any data format and create compelling visualizations that communicate your scientific discoveries!