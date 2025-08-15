# Introduction to Git and GitHub

## What is Git?

Git is a version control system that tracks changes to your code over time. Think of it as:
- **Time machine** for your code - go back to any previous version
- **Collaboration tool** - work with others without overwriting each other's work  
- **Backup system** - your code lives in multiple places
- **Lab notebook** - document what you changed and why

GitHub is a website that hosts Git repositories online, making it easy to share and collaborate.

## Why Version Control Matters for Scientists

Without version control, you've probably done this:
```
project_final.py
project_final_v2.py
project_final_v2_actually_final.py
project_final_v2_actually_final_FOR_REAL.py
project_old_dont_delete.py
```

With Git, you have one file with complete history of all changes.

## Initial Setup (One Time Only)

### 1. Install Git

**macOS**: Git comes pre-installed. Verify with:
```bash
git --version
```

**Linux**: Install via package manager:
```bash
sudo apt-get install git  # Ubuntu/Debian
sudo yum install git      # Fedora
```

**Windows**: Download from https://git-scm.com/download/win

### 2. Configure Your Identity

Git needs to know who you are for commit messages:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@sdsu.edu"
```

Verify configuration:
```bash
git config --list
```

### 3. Set Up GitHub Account

1. Go to https://github.com
2. Sign up with your SDSU email (gets you free Pro features)
3. Verify your email address
4. Apply for Student Developer Pack: https://education.github.com/pack

## Core Git Concepts

### Repository (Repo)
A folder that Git is tracking. Contains all your project files plus a hidden `.git` folder with the version history.

### Commit
A snapshot of your code at a specific point in time. Like a save point in a video game.

### Branch
An independent line of development. Default branch is usually called `main` or `master`.

### Remote
A version of your repository hosted elsewhere (like GitHub).

## Essential Git Workflow

### 1. Creating a New Repository

```bash
mkdir my_project
cd my_project
git init
```

This creates a new Git repository in the current folder.

### 2. Basic Workflow Cycle

```bash
# 1. Check status (do this often!)
git status

# 2. Add files to staging area
git add filename.py
# Or add all changed files:
git add .

# 3. Commit with descriptive message
git commit -m "Add function to calculate stellar luminosity"

# 4. Push to GitHub (after setting up remote)
git push
```

### 3. Cloning an Existing Repository

For course projects, you'll clone from GitHub Classroom:

```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

## Working with GitHub Classroom

For each project:

1. **Accept Assignment**: Click the link provided on Canvas
2. **Clone Your Repository**:
```bash
git clone https://github.com/sdsu-astr596/project1-yourname.git
cd project1-yourname
```

3. **Work on Your Code**: Make changes, test, debug

4. **Commit Frequently**:
```bash
git add .
git commit -m "Implement Euler integration for N-body"
```

5. **Push to Submit**:
```bash
git push
```

Your submission is whatever is pushed by the deadline!

## Good Commit Messages

**Bad commit messages:**
```bash
git commit -m "fixed stuff"
git commit -m "asdlfkj"
git commit -m "done"
```

**Good commit messages:**
```bash
git commit -m "Fix energy conservation in Verlet integrator"
git commit -m "Add docstrings to Star class methods"
git commit -m "Implement binary star evolution"
```

Rule: Someone (including future you) should understand what changed without looking at the code.

## Common Git Commands

### Status and History
```bash
git status              # What's changed?
git log                 # Show commit history
git log --oneline      # Compact history view
git diff               # Show unstaged changes
git diff --staged      # Show staged changes
```

### Undoing Changes
```bash
git checkout -- file.py          # Discard changes to file
git reset HEAD file.py          # Unstage file
git reset --hard HEAD            # Discard ALL changes (careful!)
git revert <commit-hash>         # Undo specific commit
```

### Branches (Advanced)
```bash
git branch                       # List branches
git branch feature-name          # Create new branch
git checkout feature-name        # Switch to branch
git checkout -b new-feature      # Create and switch
git merge feature-name           # Merge branch into current
```

## .gitignore File

Tell Git which files to never track. Create `.gitignore` in your repository root:

```
# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# Data files (usually too large)
*.fits
*.hdf5
*.npy
large_data/

# Operating system
.DS_Store
Thumbs.db

# Editor
.vscode/
.idea/

# Personal
notes_to_self.txt
scratch/
```

## Git Best Practices for This Course

### 1. Commit Early and Often
- Commit when you get something working
- Don't wait until everything is perfect
- Small commits are easier to understand and revert

### 2. Write Meaningful Messages
- First line: what changed
- Blank line
- Additional details if needed

### 3. Don't Commit Large Files
- No data files > 100MB
- Use `.gitignore` for generated files
- Keep repositories focused on code

### 4. Always Pull Before Push
```bash
git pull   # Get latest changes
git push   # Push your changes
```

### 5. Check Status Frequently
```bash
git status  # Your best friend
```

## Common Issues and Solutions

### "Failed to push some refs"
Someone else pushed before you:
```bash
git pull
# Resolve any conflicts if they exist
git push
```

### Merge Conflicts
When Git can't automatically merge changes:
1. Open conflicted file
2. Look for `<<<<<<<`, `=======`, `>>>>>>>`
3. Edit to resolve conflict
4. Remove conflict markers
5. Add and commit

### Accidentally Committed Large File
```bash
git rm --cached large_file.fits
git commit -m "Remove large file"
echo "*.fits" >> .gitignore
```

### Need to Change Last Commit Message
```bash
git commit --amend -m "New message"
```

## VS Code Git Integration

VS Code has excellent Git support built-in:

1. **Source Control Panel**: Click branch icon in sidebar
2. **Stage Changes**: Click + next to files
3. **Commit**: Type message, press Ctrl+Enter
4. **Push/Pull**: Click sync icon

But learn command line first—it's more powerful and works everywhere!

## Practice Exercise

Let's practice the complete workflow:

1. **Create a test repository**:
```bash
mkdir git_practice
cd git_practice
git init
```

2. **Create a Python file**:
```python
# save as hello.py
def greet(name):
    return f"Hello, {name}!"

print(greet("ASTR 596"))
```

3. **Make your first commit**:
```bash
git add hello.py
git commit -m "Add greeting function"
```

4. **Make changes**:
```python
# Add to hello.py
def farewell(name):
    return f"Goodbye, {name}!"
```

5. **Commit changes**:
```bash
git add hello.py
git commit -m "Add farewell function"
```

6. **View history**:
```bash
git log --oneline
```

Congratulations! You're using version control!

## Resources

- **Pro Git Book** (free): https://git-scm.com/book
- **GitHub's Git Tutorial**: https://try.github.io
- **Atlassian Git Tutorial**: https://www.atlassian.com/git/tutorials
- **Oh Shit, Git!?!**: https://ohshitgit.com (solutions to common mistakes)

## Next Steps

1. Practice with the exercise above
2. Set up SSH keys for GitHub (optional but convenient)
3. Start Project 1 using GitHub Classroom
4. Commit your work frequently!

Remember: Git has a learning curve, but it's worth it. Every professional programmer and scientist uses version control. You're learning an essential skill that you'll use throughout your career!