# Complete Jupyter Book v2 Workflow: From Setup to Deployment

## **Current Status:**
- ✅ You have a working Jupyter Book v2 site locally
- ✅ You can preview it with `jupyter book start`
- ❌ GitHub deployment is broken (no HTML files generated)

---

## **The Problem We've Been Fighting:**
Jupyter Book v2 creates dynamic MyST sites (JSON/JS files), not static HTML that GitHub Pages can serve. We need the **official deployment method**.

---

## **STEP 1: Set Up Official GitHub Deployment**

### **Generate the official workflow:**
```bash
# In your local repo directory
jupyter book init --gh-pages
```

**This will prompt you:**
- Branch to deploy from: `main`
- Action name: `deploy.yml` (or whatever you prefer)

### **Commit the generated workflow:**
```bash
git add .github/workflows/deploy.yml
git commit -m "Add official Jupyter Book v2 deployment workflow"
git push origin main
```

---

## **STEP 2: Configure GitHub Pages**

1. **Go to:** `https://github.com/astrobytes-edu/astr596-modeling-universe/settings/pages`
2. **Set Source to:** "GitHub Actions" (NOT "Deploy from a branch")
3. **Save settings**

---

## **STEP 3: Your Daily Workflow (After Initial Setup)**

### **Local Development:**
```bash
# Start local preview server
jupyter book start
# Visit: http://localhost:3000
# Edit your .md files
# Save changes (site auto-refreshes)
# Stop server: Ctrl+C
```

### **Deploy Changes:**
```bash
# Check what you've changed
git status

# Add all changes
git add .

# Commit with descriptive message
git commit -m "Update Project 2 description and add new examples"

# Push to trigger automatic deployment
git push origin main
```

### **Monitor Deployment:**
1. **Go to Actions tab:** `https://github.com/astrobytes-edu/astr596-modeling-universe/actions`
2. **Watch the workflow run** (3-5 minutes)
3. **Check your live site:** `https://astrobytes-edu.github.io/astr596-modeling-universe`

---

## **STEP 4: Key Jupyter Book v2 Commands**

### **Local Development:**
```bash
# Start development server (builds + serves)
jupyter book start

# Clean build files (if needed)
jupyter book clean

# Initialize new project
jupyter book init

# Generate GitHub Pages workflow
jupyter book init --gh-pages
```

### **Manual Building (rarely needed):**
```bash
# Build for development (creates dynamic site)
jupyter book build .

# Check build output
ls _build/site/
```

---

## **What We Learned Today:**

1. **Jupyter Book v2 ≠ Jupyter Book v1**
   - Different config files (`myst.yml` not `_config.yml`)
   - Different build output (dynamic sites, not static HTML)
   - Different deployment methods

2. **The npm issues were real**
   - Jupyter Book v2 requires Node.js/npm
   - Permission conflicts between system and conda npm
   - Fixed by using conda nodejs

3. **GitHub deployment requires the official workflow**
   - Custom workflows don't handle the MyST build properly
   - Must use `jupyter book init --gh-pages`
   - Must use "GitHub Actions" as Pages source

4. **The file structure confusion:**
   - `_build/site/` contains the MyST dynamic site
   - No `index.html` - it's a JavaScript application
   - GitHub Pages needs special handling for this

---

## **Troubleshooting:**

### **If local preview fails:**
```bash
# Check Node.js/npm versions
node --version  # Should be >= 18
npm --version   # Should be >= 8.6

# Clear cache and retry
npm cache clean --force
jupyter book start
```

### **If deployment fails:**
1. **Check Actions tab** for specific error
2. **Verify Pages settings** (GitHub Actions source)
3. **Try re-running the workflow**

### **If site shows 404:**
- **Wait 5-10 minutes** after successful deployment
- **Hard refresh browser** (Ctrl+Shift+R)
- **Check that workflow completed successfully**

---

## **Summary:**
Your course website is built with Jupyter Book v2, which creates dynamic sites that require special deployment handling. The official `jupyter book init --gh-pages` command generates the correct workflow, and GitHub Actions handles the complex build process automatically.