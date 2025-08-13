# Software Setup Guide

## Overview

This guide will walk you through setting up your computational environment for ASTR 596. We'll install Python, essential scientific packages, Git for version control, and configure your development environment. 

**Time required**: ~45-60 minutes

**Operating Systems**: Instructions provided for macOS, Linux, and Windows

## Step 1: Install Python via Miniforge

We use Miniforge (community-driven minimal installer for conda) because it:
- Provides conda package manager with conda-forge as default channel
- Ensures reproducible environments across different operating systems
- Handles complex dependencies in scientific packages
- Is completely free and open-source

### Installation Instructions

#### macOS and Linux

1. Open Terminal
2. Download Miniforge:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

3. Run the installer:
```bash
bash Miniforge3-$(uname)-$(uname -m).sh
```

4. Follow prompts:
   - Press ENTER to review license
   - Type `yes` to accept
   - Press ENTER to confirm location or specify different path
   - Type `yes` when asked about conda init

5. Restart terminal or run:
```bash
source ~/.bashrc  # On Linux
source ~/.zshrc   # On macOS with zsh
```

6. Verify installation:
```bash
conda --version
python --version
```

#### Windows

1. Download installer from: https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe

2. Run the `.exe` file

3. Follow installation wizard:
   - Select "Just Me" (recommended)
   - Choose installation path (default is fine)
   - Check "Add Miniforge3 to my PATH environment variable"

4. Open "Miniforge Prompt" from Start Menu

5. Verify installation:
```bash
conda --version
python --version
```

## Step 2: Create Course Environment

We'll create an isolated environment for this course to avoid conflicts with other projects.

### Understanding Conda Environments

**What's an environment?** Think of it as a separate, clean installation of Python with its own packages. This means:
- Your course packages won't conflict with other projects
- You can have different Python versions for different projects
- You can share exact package versions with others (reproducibility!)
- If something breaks, you can delete the environment and start fresh

**Example**: You might need Python 3.11 with JAX for this course, but Python 3.9 with TensorFlow for another project. Environments make this possible!

### Creating Your Course Environment

1. Create environment with Python 3.11:
```bash
conda create -n astr596 python=3.11
```

2. Activate the environment:
```bash
conda activate astr596
```

**Note**: You'll need to activate this environment every time you work on course materials! You'll know it's active when you see `(astr596)` in your terminal prompt.

3. Install essential packages:
```bash
conda install numpy scipy matplotlib pandas jupyter ipython
```

4. Install additional scientific packages:
```bash
conda install scikit-learn astropy h5py
```

5. Install JAX and ecosystem:
```bash
# Install JAX with conda (conda-forge is default with Miniforge!)
conda install jax jaxlib

# Install JAX ecosystem
conda install flax optax
```

### Package Management: Conda vs Pip

**We use conda for this course** because:
- Handles complex dependencies better (especially for scientific packages)
- Manages non-Python dependencies (like CUDA for GPUs)
- Environments are more isolated and reproducible
- Works consistently across all operating systems
- With Miniforge, conda-forge is the default channel (community-maintained, comprehensive)

**When to use pip:**
- Only when a package isn't available on conda-forge
- Always use pip AFTER installing conda packages

**If you must use pip:**
```bash
# First, always check if it's available
conda search package-name

# If not available, then use pip
pip install package-name
```

**Important**: Mixing conda and pip can sometimes cause issues. Best practice:
1. Install everything possible with conda first
2. Use pip only for packages not on conda
3. Document what you installed with pip

## Step 3: Install and Configure VS Code

Visual Studio Code is our recommended editor (free, powerful, cross-platform).

### Installation

1. Download from: https://code.visualstudio.com/
2. Run installer for your operating system
3. Launch VS Code

### Essential Extensions

Install these extensions (click Extensions icon in sidebar or press `Ctrl+Shift+X`):

1. **Python** (by Microsoft) - Python language support
2. **Jupyter** (by Microsoft) - Notebook support
3. **GitLens** (by GitKraken) - Enhanced Git integration
4. **indent-rainbow** - Makes indentation visible

### Configure VS Code for Course

1. Open Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
2. Type "Python: Select Interpreter"
3. Choose the `astr596` environment

### CRITICAL: Disable AI Assistants

Per course policy, ALL AI coding assistants must be disabled:

1. Open Settings (`Ctrl+,` or `Cmd+,`)
2. Search for "github.copilot"
3. Uncheck "Enable" if GitHub Copilot is installed
4. Search for "IntelliCode"
5. Set "Vs › IntelliCode: Enable" to unchecked
6. Search for "AI"
7. Disable any AI-powered suggestions or completions

## Step 4: Test Your Setup

Create a test script to verify everything works:

1. Create a new file called `test_setup.py`

2. Add this code:
```python
"""Test script for ASTR 596 setup"""

# Test imports
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

print("Python packages imported successfully!")
print(f"NumPy version: {np.__version__}")
print(f"JAX version: {jax.__version__}")

# Test computation
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Test plotting
plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title("Setup Test: Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True, alpha=0.3)
plt.savefig("test_plot.png")
print("Plot saved as test_plot.png")

# Test JAX
def f(x):
    return x**2

grad_f = jax.grad(f)
print(f"Derivative of x^2 at x=3.0: {grad_f(3.0)}")

print("\n✅ All tests passed! Your environment is ready.")
```

3. Run the script:
```bash
python test_setup.py
```

You should see success messages and a plot file created.

## Step 5: Terminal Basics

You'll be using the terminal extensively. Here are essential commands:

### Navigation
```bash
pwd              # Print working directory (where am I?)
ls               # List files in current directory
ls -la           # List all files with details
cd folder_name   # Change directory
cd ..            # Go up one directory
cd ~             # Go to home directory
```

### File Operations
```bash
mkdir project1   # Make new directory
touch file.py    # Create empty file
cp file1 file2   # Copy file
mv file1 file2   # Move/rename file
rm file          # Remove file (careful!)
cat file         # Display file contents
```

### Python/Conda
```bash
python script.py         # Run Python script
python                   # Start Python interpreter (exit() to quit)
conda activate astr596   # Activate course environment
conda deactivate        # Deactivate environment
conda list              # List installed packages
conda install package   # Install package with conda
conda search package    # Search for package
```

### Managing Your Environment
```bash
conda env list          # List all environments
conda info              # Show conda information
which python            # Check which Python you're using
```

## Troubleshooting

### "Command not found: conda"
- Make sure you restarted terminal after installation
- On macOS/Linux, check that conda was added to your shell config

### "Import error: No module named..."
- Make sure you activated the astr596 environment: `conda activate astr596`
- Check if package is installed: `conda list | grep package_name`
- Install missing package: `conda install package_name`
- Only as last resort: `pip install package_name`

### VS Code can't find Python interpreter
- Open Command Palette and run "Python: Select Interpreter"
- Choose the astr596 environment
- Restart VS Code if needed

### Permission errors
- On macOS/Linux, you might need `sudo` for some operations
- On Windows, run terminal as Administrator

## Next Steps

Once your environment is set up:
1. Continue to the [Git Introduction Guide](git-intro)
2. Clone the course repository
3. Start Project 1!

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Google the exact error message
3. Ask on course Slack with:
   - Your operating system
   - The command you ran
   - The complete error message
4. Come to office hours

Remember: Setup issues are normal! Don't let technical hurdles discourage you from the actual course content.