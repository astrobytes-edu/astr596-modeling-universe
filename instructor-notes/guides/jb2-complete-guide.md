# Jupyter Book v2 Deployment Guide

## Quick Setup

### 1. Create and Initialize
```bash
# Create book
jupyter book create mybookname/
cd mybookname/

# Initialize git
git init
git add .
git commit -m "Initial commit"

# Create GitHub repo (on GitHub.com - don't initialize with README)
# Then connect:
git remote add origin https://github.com/[username]/[repo].git
git branch -M main
git push -u origin main
```

### 2. Set up GitHub Pages Deployment
```bash
# Generate GitHub Actions workflow
jupyter book init --gh-pages
# Choose: main branch, deploy.yml

# Commit the workflow
git add .github/workflows/deploy.yml
git commit -m "Add deployment workflow"
git push origin main
```

### 3. Configure GitHub Pages
1. Go to: `https://github.com/[username]/[repo]/settings/pages`
2. Set Source to: **"GitHub Actions"**
3. Save

### 4. Wait and Check
- Go to Actions tab to monitor deployment
- Site will be at: `https://[username].github.io/[repo]/`

---

## Daily Workflow

```bash
# Edit your content
# Test locally
jupyter book start

# Deploy
git add .
git commit -m "Update content"
git push origin main
```

---

## Troubleshooting

### If Things Break
```bash
# Clean build and try again
rm -rf _build/
jupyter book build .
```

### If GitHub Actions Fails
```bash
# Start fresh
rm -rf _build/
rm -rf .github/
jupyter book init --gh-pages
git add .
git commit -m "Regenerate workflow"
git push origin main
```

That's it. The official instructions work fine.