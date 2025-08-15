# Chapter 1: Development Environment

## Learning Objectives
By the end of this chapter, you will:
- Understand the tools and environment needed for computational astrophysics
- Set up a reproducible Python environment for scientific computing
- Configure version control for collaborative research
- Establish best practices for project organization

## 1.1 The Modern Astrophysics Toolkit

Computational astrophysics requires a carefully curated development environment. Unlike using pre-built software packages, we'll be building tools from scratch, which demands:

- **Python 3.9+**: Modern language features for clean, efficient code
- **Scientific Libraries**: NumPy, SciPy, Matplotlib, Astropy
- **Development Tools**: Jupyter, VS Code, Git
- **Performance Libraries**: JAX (later in course), Numba
- **Testing Frameworks**: pytest, hypothesis

## 1.2 Environment Management with Conda

### Why Environment Management Matters

Reproducibility is crucial in scientific computing. Your code should produce identical results whether run on your laptop, the campus cluster, or a collaborator's machine three years from now.

```python
# Check your Python version
import sys
print(f"Python version: {sys.version}")
print(f"Version info: {sys.version_info}")

# Verify you have Python 3.9 or higher
assert sys.version_info >= (3, 9), "Python 3.9+ required"
```

### Creating Your Course Environment

```bash
# Create environment with essential packages
conda create -n astr596 python=3.11 numpy scipy matplotlib astropy jupyter pandas

# Activate the environment
conda activate astr596

# Install additional packages
pip install pytest black flake8 mypy
```

## 1.3 Version Control with Git

### Repository Structure for Scientific Projects

```
astr596_project/
├── README.md           # Project overview and setup instructions
├── environment.yml     # Conda environment specification
├── notebooks/         # Jupyter notebooks for exploration
├── src/              # Source code modules
│   ├── __init__.py
│   ├── physics/      # Physics implementations
│   ├── numerics/     # Numerical methods
│   └── utils/        # Utility functions
├── tests/            # Unit tests
├── data/            # Data files (use .gitignore for large files)
└── results/         # Output figures and processed data
```

### Essential Git Commands for Collaboration

```bash
# Initialize repository
git init

# Configure your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@sdsu.edu"

# Create .gitignore for Python projects
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".ipynb_checkpoints/" >> .gitignore
echo "data/large_files/" >> .gitignore
```

## 1.4 Jupyter Notebook Best Practices

### Notebook Organization

```python
# Standard imports at the top of every notebook
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 6),
    'figure.dpi': 100
})

# Set random seed for reproducibility
np.random.seed(42)
```

## 1.5 Campus Cluster Access

*[Instructor will provide specific instructions for SDSU cluster access]*

Key points to remember:
- Use SSH keys for secure access
- Transfer code via Git, not file copying
- Submit batch jobs for long-running simulations
- Use interactive nodes for development

## Try It Yourself

### Exercise 1.1: Environment Verification
Create a Python script that verifies all required packages are installed and reports their versions.

```python
def check_environment():
    """Verify all required packages are installed."""
    required_packages = {
        'numpy': '1.20',
        'scipy': '1.7',
        'matplotlib': '3.3',
        'astropy': '5.0',
        'pandas': '1.3'
    }
    
    for package, min_version in required_packages.items():
        try:
            # Your code here: import package and check version
            pass
        except ImportError:
            print(f"❌ {package} not installed")
    
    return None

# Run the check
check_environment()
```

### Exercise 1.2: Project Setup
Initialize a Git repository for your first project with the proper structure.

```python
import os
from pathlib import Path

def setup_project(project_name):
    """Create project directory structure."""
    base_dir = Path(project_name)
    
    # Directories to create
    directories = [
        'notebooks',
        'src/physics',
        'src/numerics', 
        'src/utils',
        'tests',
        'data',
        'results/figures',
        'results/tables'
    ]
    
    # Your code here: create directories and initial files
    
    return base_dir

# Create your project
project_path = setup_project("stellar_evolution")
print(f"Project created at: {project_path}")
```

### Exercise 1.3: Astropy Units Demo
Verify Astropy's unit system is working correctly by calculating the Schwarzschild radius of the Sun.

```python
from astropy import units as u
from astropy import constants as const

def schwarzschild_radius(mass):
    """
    Calculate Schwarzschild radius for a given mass.
    
    Parameters
    ----------
    mass : astropy.units.Quantity
        Mass of the object
    
    Returns
    -------
    astropy.units.Quantity
        Schwarzschild radius
    """
    # Your code here: r_s = 2GM/c^2
    pass

# Calculate for the Sun
r_sun = schwarzschild_radius(const.M_sun)
print(f"Schwarzschild radius of the Sun: {r_sun.to(u.km):.2f}")

# Verify it's about 3 km
assert 2.5 * u.km < r_sun < 3.5 * u.km, "Check your calculation!"
```

## Key Takeaways

✅ **Environment management** ensures reproducible research  
✅ **Version control** is essential for collaboration and tracking progress  
✅ **Project organization** from the start saves time later  
✅ **Testing your setup** prevents issues during critical work  

## Next Chapter Preview
With our environment configured, we'll review Python fundamentals with an astrophysical focus, ensuring everyone has the foundation for the advanced topics ahead.