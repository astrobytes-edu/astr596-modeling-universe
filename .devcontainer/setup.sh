#!/bin/bash
echo "🚀 Setting up ASTR 596 environment..."

# Install Python packages
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional useful packages
pip install black pylint ipykernel jupyterlab-git

# Install MyST
echo "📚 Installing MyST..."
npm install -g mystmd

# Create workspace directories
mkdir -p workspace/scratch

# Configure Git with placeholder (students will update)
git config --global user.name "Student Name"
git config --global user.email "student@example.com"
git config --global init.defaultBranch main

echo "✅ Setup complete! You can now:"
echo "  - Run 'jupyter lab' for JupyterLab"
echo "  - Run 'myst start' to preview the course website"
echo "  - Open any .ipynb file in VS Code"