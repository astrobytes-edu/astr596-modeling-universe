#!/bin/bash
# .devcontainer/setup.sh
# Setup script for ASTR 596 Codespaces environment

echo "🚀 Setting up ASTR 596 Codespaces environment..."

# Install Python packages
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional packages for the course
pip install \
    black \
    pylint \
    ipykernel \
    jupyterlab-git \
    jupyterlab-github \
    nbgrader \
    pytest \
    pytest-cov

# Install MyST for documentation
echo "📚 Installing MyST..."
npm install -g mystmd

# Install Typst for PDF generation (optional)
echo "📄 Installing Typst..."
curl -fsSL https://typst.app/install.sh | sh

# Create useful directories
mkdir -p workspace/assignments
mkdir -p workspace/projects
mkdir -p workspace/scratch

# Set up Jupyter kernel
python -m ipykernel install --user --name astr596 --display-name "ASTR 596 Python"

# Configure Git (students will need to update this)
git config --global user.name "Student Name"
git config --global user.email "student@example.com"
git config --global init.defaultBranch main

# Create a welcome message
cat > ~/WELCOME.md << 'EOF'
# Welcome to ASTR 596: Modeling the Universe! 🌟

## Quick Start
1. Open any `.ipynb` file to start with Jupyter
2. Use `jupyter lab` in terminal for JupyterLab
3. Run `myst start` to preview the course website
4. Your work auto-saves every few seconds

## Important Commands
- `python script.py` - Run Python scripts
- `jupyter lab` - Start JupyterLab
- `myst start` - Preview MyST documentation
- `pytest` - Run tests

## Directories
- `/workspace/assignments/` - Your assignments
- `/workspace/projects/` - Course projects
- `/workspace/scratch/` - Experimental work

## Need Help?
- Check the course website
- Use GitHub Issues for questions
- Office hours: [See syllabus]

Happy coding! 🚀
EOF

echo "✅ Setup complete! Check ~/WELCOME.md for getting started info."