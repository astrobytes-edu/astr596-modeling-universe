# GitHub Classroom + Codespaces Setup for ASTR 596

## Initial Setup (Do Once)

### 1. Create Your GitHub Classroom
1. Go to https://classroom.github.com/
2. Click "New classroom"
3. Select your organization (astrobytes-edu)
4. Name it "ASTR 596 Fall 2025"
5. Add TAs as admins if needed

### 2. Configure Classroom Settings
- ✅ Enable GitHub Codespaces
- ✅ Grant students admin access to their repos
- ✅ Make repos private
- ✅ Enable feedback pull requests

### 3. Add Codespaces to Main Course Repo
```bash
# In your main course repository
mkdir -p .devcontainer
# Add devcontainer.json from artifact
# Add setup.sh and make it executable
chmod +x .devcontainer/setup.sh
```

## For Each Assignment

### Step 1: Create Assignment Template
1. Create new repo: `astr596-assignment-[week]`
2. Add these files:
   - `.devcontainer/devcontainer.json` (for Codespaces)
   - `requirements.txt` (Python packages needed)
   - `assignment.ipynb` (starter code)
   - `tests/test_assignment.py` (autograding tests)
   - `README.md` (instructions)

### Step 2: Create GitHub Classroom Assignment
1. Go to your classroom
2. Click "New assignment"
3. Configure:
   - Title: "Week X: [Topic]"
   - Deadline: [Due date]
   - Template repository: `astr596-assignment-[week]`
   - ✅ Enable Codespaces
   - ✅ Enable autograding
   - Add test cases from your pytest tests

### Step 3: Share with Students
1. Get the assignment URL from GitHub Classroom
2. Post on course website/Canvas/email
3. Students click link → Accept assignment → Open in Codespaces

## Student Workflow

### First Time Setup (5 minutes)
1. Apply for GitHub Student Developer Pack
   - Go to https://education.github.com/pack
   - Verify with school email
   - Get approved (usually instant)

### For Each Assignment (2 minutes)
1. Click assignment link from instructor
2. Accept assignment (creates personal repo)
3. Click "Open in GitHub Codespaces"
4. Start working immediately!
5. Commit and push to submit

## What Students See

### In Codespaces:
- Full VS Code in browser
- All packages pre-installed
- Jupyter notebooks work immediately
- Terminal access
- Git integration
- Autosave enabled

### After Pushing:
- Automatic grading runs
- See results in Actions tab
- Get feedback via pull request

## Instructor Dashboard

### Monitor Progress:
1. Go to GitHub Classroom
2. See all student repos
3. Check who's submitted
4. View autograding results
5. Leave feedback via PR comments

### Bulk Operations:
- Download all submissions
- Clone all repos locally
- Run additional tests
- Export grades to CSV

## Cost Management

### FREE with GitHub Education:
- **You**: Unlimited Codespaces hours
- **Students**: 180 core-hours/month (45 hours on 4-core)
- **Storage**: 20GB per month

### To Maximize Free Hours:
1. Set Codespaces to auto-stop after 30 min idle
2. Remind students to stop Codespaces when done
3. Use 2-core machines for simple assignments
4. Use 4-core for computational work

## Example Timeline for Week 1

### Monday: Post Lecture Materials
- Students read on course website
- "Live Code" button for quick experiments (uses Binder)

### Wednesday: Release Assignment
- Post GitHub Classroom link
- Students accept and start in Codespaces
- Work during class with live help

### Friday: Assignment Due
- Autograding runs on submission
- You review and add feedback
- Grades export to gradebook

## Advanced Features

### GPU Support (for JAX/ML weeks):
```json
// In devcontainer.json
"hostRequirements": {
  "gpu": true
}
```

### Collaborative Coding:
- Students can share Codespace URLs
- Live Share extension for pair programming
- You can join their Codespace to help debug

### Custom Autograding:
- Test notebooks with nbgrader
- Check plot outputs
- Validate numerical results
- Style checking with black/pylint

## Troubleshooting

**Student can't access Codespaces:**
- Check they have GitHub Education Pack
- Verify they accepted the assignment
- Try incognito mode

**Autograding fails:**
- Check timeout settings
- Verify test file paths
- Look at Actions logs

**Codespace won't start:**
- Check devcontainer.json syntax
- Verify image exists
- Reduce resource requirements

## Best Practices

1. **Test everything yourself first**
   - Accept your own assignment
   - Run through in Codespaces
   - Verify autograding works

2. **Provide starter code**
   - Include function signatures
   - Add docstrings
   - Give example test cases

3. **Clear instructions**
   - Step-by-step setup
   - Expected outputs
   - Submission process

4. **Regular check-ins**
   - Monitor GitHub Classroom dashboard
   - Check for struggling students
   - Offer help early

## Integration with Course Website

### On Each Lesson Page:
```markdown
---
title: Week 1: Python Fundamentals
---

## 📚 Reading
[Read the lesson here]

## 💻 Interactive Examples
Click "Live Code" to experiment with examples

## 📝 Assignment
[Accept Week 1 Assignment](https://classroom.github.com/a/YOUR-ASSIGNMENT-ID)

## 🎥 Lecture Recording
[Watch on YouTube](...)
```