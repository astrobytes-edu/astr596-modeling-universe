# Complete MyST Build Guide for ASTR 596

## Table of Contents
1. [Installation](#installation)
2. [Local Development](#local-development)
3. [GitHub Deployment](#github-deployment)
4. [Troubleshooting](#troubleshooting)
5. [Important Links](#important-links)

---

## Installation

### Prerequisites

You need Node.js (v18+) and npm installed on your system.

### Install MyST (Choose ONE method)

#### Option A: Via npm (Recommended)

```bash
npm install -g mystmd
```

#### Option B: Via pip (Python users)

```bash
pip install mystmd
```

#### Option C: Via conda

```bash
conda install mystmd -c conda-forge
```

### Verify Installation

```bash
myst --version
# Should show: v1.6.0 or higher
```

---

## Local Development

### 1. Initial Setup (First Time Only)

#### Initialize MyST Project

```bash
# In your repository root (where your content is)
cd /path/to/astr596-modeling-universe

# Initialize MyST configuration
myst init

# Or if you want to set up GitHub Pages immediately:
myst init --gh-pages
```

This creates:
- `myst.yml` - Main configuration file
- `_build/` folder - Where built content goes

### 2. Development Workflow

#### Start Development Server
```bash
# Start with live reload (recommended for development)
myst start

# The site opens at http://localhost:3000
# Changes auto-reload in browser
```

**What happens:**
- MyST processes content into the `_build` folder containing processed content and site assets used by the local web server
- Server watches for file changes
- Browser auto-refreshes on changes

#### Alternative ports (if 3000 is busy):
```bash
# Specify a different port
myst start --port 8080

# Or let MyST find an open port
myst start
# Look for the URL in terminal output
```

### 3. Building for Production

#### Build Everything
```bash
# Build HTML website and PDFs
myst build --all
```

#### Build Specific Outputs
```bash
# Build only HTML website
myst build --html

# Build only PDFs
myst build --pdf

# Build specific file to PDF
myst build path/to/file.md --pdf
```

#### Execute Notebooks During Build
```bash
# Execute all notebooks and build
myst build --execute

# Build without executing notebooks
myst build --html
```

### 4. File Structure After Building

```text
astr596-modeling-universe/
├── _build/
│   ├── site/          # HTML website files
│   ├── exports/       # PDF exports
│   ├── templates/     # Downloaded templates
│   └── temp/          # Temporary files
├── myst.yml           # Configuration
├── index.md           # Your content
└── ...
```

### 5. Clean Build Artifacts

```bash
# Clean build files (prompts for confirmation)
myst clean

# Clean everything including templates and cache
myst clean --all

# Clean without confirmation prompt
myst clean -y

# Clean specific types
myst clean --site      # Only website files
myst clean --exports   # Only PDFs
myst clean --temp      # Only temp files
myst clean --templates # Only templates
```

---

## GitHub Deployment

### Method 1: Automatic Setup (Easiest)

MyST provides a special init function that adds proper configuration for deploying to GitHub Pages with a GitHub Action:

```bash
# In your repository root
myst init --gh-pages
```

This command will:
1. Ask which branch to deploy from (usually `main`)
2. Create `.github/workflows/deploy.yml`
3. Configure BASE_URL automatically

### Method 2: Manual Setup

#### Step 1: Create GitHub Action
Create `.github/workflows/deploy.yml`:

```yaml
name: MyST GitHub Pages Deploy

on:
  push:
    branches: [main]
  workflow_dispatch:  # Allow manual triggers

env:
  BASE_URL: /${{ github.event.repository.name }}

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: 'pages'
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Pages
        uses: actions/configure-pages@v3
      
      - uses: actions/setup-node@v4
        with:
          node-version: 18.x
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install MyST
        run: npm install -g mystmd
      
      - name: Install Python Dependencies
        run: pip install -r requirements.txt
      
      - name: Setup Typst for PDFs
        uses: yusancky/setup-typst@v2
        with:
          version: 'latest'
> **Note (Important):** The `yusancky/setup-typst` action has been migrated to the community-maintained `typst-community/setup-typst`. Update any workflow references to use `typst-community/setup-typst@v4` and rename the input `version` → `typst-version`. Example:

```yaml
      - name: Setup Typst for PDFs
        uses: typst-community/setup-typst@v4
        with:
          typst-version: 'latest'
```

This migration is required to avoid workflow failures (older `yusancky` releases will not receive updates after July 2025). See: https://github.com/typst-community/setup-typst and the announcement in the project README for details.
      
      - name: Build PDFs
        run: myst build --pdf
        continue-on-error: true
      
      - name: Build HTML Site
        run: myst build --html
      
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: './_build/site'
      
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

#### Step 2: Configure GitHub Pages

Navigate to Settings -> Pages in your repository and enable GitHub Pages by choosing GitHub Actions as the source:

1. Go to: `https://github.com/astrobytes-edu/astr596-modeling-universe/settings/pages`
2. Under "Source", select "GitHub Actions"
3. Save

#### Step 3: Push Changes

```bash
git add .
git commit -m "Add MyST GitHub Pages deployment"
git push origin main
```

#### Step 4: Monitor Deployment

1. Go to Actions tab: `https://github.com/astrobytes-edu/astr596-modeling-universe/actions`
2. Watch the workflow run (takes 3-5 minutes)
3. Once complete, visit: `https://astrobytes-edu.github.io/astr596-modeling-universe/`

### BASE_URL Configuration

The MyST CLI needs to know the destination (base URL) of your site during build time:

- **Default repo** (`username.github.io`): No BASE_URL needed
- **Project repo** (`username.github.io/project`): Set `BASE_URL: /${{ github.event.repository.name }}`

---

## Troubleshooting

### Common Issues and Solutions

#### Port Already in Use
```bash
# Error: Port 3000 is already in use

# Solution 1: Use different port
myst start --port 8080

# Solution 2: Kill process on port 3000
lsof -ti:3000 | xargs kill -9  # macOS/Linux
```

#### Build Fails with PDF Errors
```bash
# Solution: Install Typst first
brew install typst  # macOS
# Or download from: https://github.com/typst/typst/releases

# Then rebuild
myst build --pdf
```

#### Changes Not Showing
```bash
# Solution 1: Clear cache and rebuild
myst clean --all
myst build --all

# Solution 2: Hard refresh browser
# Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows/Linux)
```

#### GitHub Pages Not Updating
1. Check Actions tab for errors
2. Verify Pages source is "GitHub Actions"
3. Clear browser cache
4. Wait 5-10 minutes (CDN propagation)

#### MyST Command Not Found
```bash
# If installed via npm but not found:
export PATH="$HOME/.npm-global/bin:$PATH"
echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> ~/.bashrc  # or ~/.zshrc

# If installed via pip:
export PATH="$HOME/.local/bin:$PATH"
```

---

## Important Links

### Official Documentation
- **MyST Documentation**: https://mystmd.org/
- **MyST CLI Reference**: https://mystmd.org/cli/reference
- **MyST GitHub**: https://github.com/jupyter-book/mystmd

### Key Documentation Pages
- **Get Started Guide**: https://mystmd.org/docs/mystjs/quickstart
- **CLI Commands**: https://mystmd.org/cli/reference
- **GitHub Pages Deploy**: https://mystmd.org/guide/deployment-github-pages
- **Deployment Options**: https://mystmd.org/guide/deployment

### Your Project Links
- **Repository**: https://github.com/astrobytes-edu/astr596-modeling-universe
- **Live Site**: https://astrobytes-edu.github.io/astr596-modeling-universe/
- **Actions**: https://github.com/astrobytes-edu/astr596-modeling-universe/actions

### Templates and Themes
- **List PDF templates**: `myst templates list --pdf`
- **List web themes**: `myst templates list --site`
- **Template docs**: https://mystmd.org/guide/templates

---

## Quick Reference Card

### Daily Development Commands
```bash
# Start development
myst start

# Build everything
myst build --all

# Clean and rebuild
myst clean --all
myst build --all

# Check what will be built
myst build --dry-run
```

### Before Pushing to GitHub
```bash
# Test the build locally
myst build --html
myst build --pdf

# Preview the built site
cd _build/site
python -m http.server 8000
# Visit http://localhost:8000

# If everything looks good, push
git add .
git commit -m "Update content"
git push
```

### Debugging Commands
```bash
# Verbose output
myst build --verbose

# Check configuration
cat myst.yml

# Check MyST version
myst --version

# List available templates
myst templates list

# Get help
myst --help
myst build --help
```

---

## Advanced Tips

### Custom Build Scripts
Add to `package.json` in your repo:
```json
{
  "scripts": {
    "dev": "myst start",
    "build": "myst build --all",
    "clean": "myst clean --all",
    "pdf": "myst build --pdf"
  }
}
```

Then use:
```bash
npm run dev    # Start development
npm run build  # Build everything
npm run clean  # Clean build files
npm run pdf    # Build only PDFs
```

### Environment Variables
```bash
# Set BASE_URL for subdirectory deployment
export BASE_URL="/astr596"
myst build --html

# Set custom build directory
export MYST_BUILD_DIR="./custom-build"
myst build --all
```

### Parallel Builds (Faster)
```bash
# Build PDFs and HTML in parallel
myst build --pdf & myst build --html
wait
```

---

## Summary

1. **Local Development**: Use `myst start` for live development
2. **Building**: Use `myst build --all` for production builds
3. **GitHub Deployment**: Use `myst init --gh-pages` for automatic setup
4. **Troubleshooting**: Check Actions tab and use `myst clean --all` when stuck
5. **Documentation**: Always refer to https://mystmd.org/ for latest info