# Getting Started

**Your Foundation for Computational Astrophysics**

Before we model the universe, we need to set up our computational laboratory. This module establishes the professional development environment and essential skills you'll use throughout ASTR 596 and your research career.

:::{tip} Module Philosophy: Tools Shape Thinking

The difference between a computational scientist and someone who codes isn't knowledge of Python syntax—it's mastery of the ecosystem. Version control, environment management, and command-line fluency aren't just "nice to have" skills; they fundamentally change how you approach problems, collaborate, and think about reproducible science.
:::

## Module Overview

::::{grid} 1 1 2 3

:::{grid-item-card} 🔧 **Software Setup**
:link: 01-software-setup-guide
:link-type: doc

Professional Python environment with Miniforge, scientific libraries, and VS Code configuration
:::

:::{grid-item-card} 📝 **Version Control** 
:link: 02-git-intro-guide
:link-type: doc

Git fundamentals, GitHub workflow, and collaboration patterns for scientific computing
:::

:::{grid-item-card} 💻 **Command Line**
:link: 03-cli-intro-guide
:link-type: doc

Terminal mastery for file management, automation, and remote computing
:::

::::

## Why These Tools Matter

```{mermaid}
graph LR
    A[Local Development] --> B[Version Control]
    B --> C[Collaboration]
    C --> D[Remote Computing]
    D --> E[Production Research]
    
    A1[VS Code + Conda] --> A
    B1[Git + GitHub] --> B
    C1[GitHub Classroom] --> C
    D1[SSH + HPC] --> D
    E1[Publications] --> E
    
    style A fill:#e3f2fd
    style E fill:#f3e5f5
```

## Learning Path

:::{important} 🎯 Module Goals

After completing this module, you will:

✅ **Have** a fully configured development environment isolated from system Python  
✅ **Understand** why conda environments prevent the "it works on my machine" problem  
✅ **Use** Git for every project, even personal ones  
✅ **Navigate** the terminal faster than any GUI  
✅ **Debug** setup issues independently using error messages  
✅ **Appreciate** why every computational scientist relies on these tools
:::

## Time Investment

::::{grid} 2 2 2 2

:::{grid-item}
**Initial Setup**: 2-3 hours

- Software installation
- Configuration
- Testing
:::

:::{grid-item}
**Skill Building**: 2-3 hours  

- Practice exercises
- Troubleshooting
- Workflow development
:::

::::

## Common Challenges (and Solutions)

:::{warning} ⚠️ Heads Up: Expected Hurdles

**Challenge 1: "Command not found" errors**  
*Solution*: Usually a PATH issue. Check if you restarted your terminal after installation.

**Challenge 2: "Module not found" in Python**  
*Solution*: Wrong environment activated. Always check for `(astr596)` in your prompt.

**Challenge 3: Git says "failed to push"**  
*Solution*: Someone else pushed first. Pull, resolve any conflicts, then push.

**Challenge 4: Feeling overwhelmed by the terminal**  
*Solution*: Normal! Use our quick reference card. Muscle memory develops within 2 weeks.
:::

## Your First Week Checklist

:::{tip} 📋 Setup Verification

### Complete these milestones in order:*

#### **Day 1: Environment Setup**

- [ ] Miniforge installed and working
- [ ] VS Code configured with Python extension
- [ ] Test script runs successfully
- [ ] AI assistants disabled (course requirement)

#### **Day 2: Version Control**

- [ ] Git configured with your identity
- [ ] GitHub account created with student benefits
- [ ] First repository created and pushed
- [ ] GitHub Classroom access confirmed

#### **Day 3: Command Line Comfort**

- [ ] Navigate directories without clicking
- [ ] Create Python project structure via terminal
- [ ] Run Python scripts with different parameters
- [ ] Use grep to search through code files

#### **Before Project 1**

- [ ] Clone assignment from GitHub Classroom
- [ ] Make commits with meaningful messages
- [ ] Push changes to GitHub
- [ ] Verify submission appears online
:::

## Quick Command Reference

:::{list-table} Essential Commands You'll Use Daily
:header-rows: 1
:widths: 30 70

* - Command
  - What It Does
* - `conda activate astr596`
  - Enter your course environment
* - `git status`
  - Check what's changed in your repository
* - `git add . && git commit -m "message"`
  - Save your work with version control
* - `git push`
  - Submit to GitHub (and assignments)
* - `python script.py`
  - Run your Python code
* - `ls -la`
  - See all files in current directory
* - `cd project_folder/`
  - Navigate to your project
* - `grep -n "TODO" *.py`
  - Find all TODO comments in Python files
:::

## Getting Help

:::{note} 🆘 Support Resources

**Setup Issues**: 

- Error messages are your friends—read them carefully
- Google the exact error message (in quotes)
- Check our troubleshooting guides in each chapter
- Post on course Slack with OS, command, and full error

**Conceptual Questions**:

- Why do we need virtual environments?
- When should I make a Git commit?
- How do I know which terminal commands to use?
→ These are great office hours questions!

**Remember**: Everyone struggles with setup. It's not you — it's genuinely complex. But once it's working, it just works.
:::

## Module Completion

:::{important} 🎉 Success Indicators

You're ready to move on to Python Fundamentals when you can:

1. **Open terminal and navigate** to any folder without using Finder/Explorer
2. **Activate your conda environment** and verify the right Python is running
3. **Create a Git repository**, make commits, and push to GitHub
4. **Run Python scripts** with command-line arguments
5. **Debug basic issues** using error messages and documentation

If you can do these five things, you have the foundation for everything else in this course.
:::

## What's Next?

After completing this module, you'll begin the Python Fundamentals module where you'll:
- Write your first object-oriented stellar evolution model
- Learn Python from first principles (no prior experience assumed)
- Apply these tools in real scientific computing contexts
- Start building your portfolio of computational projects

---

:::{tip} How to Succeed in This Module

**Don't Rush**: Setup problems compound. Better to spend an extra hour now than fight your environment all semester.

**Practice Daily**: Spend 10 minutes each day using the terminal. Muscle memory develops quickly with consistent practice.

**Commit Often**: Make Git commits after every small success. You can't over-commit, but you can under-commit.

**Read Error Messages**: They usually tell you exactly what's wrong. Copy the full error when asking for help.

**Keep Notes**: Document what worked for your specific system. You'll need these notes again for future projects.

**Remember**: These tools aren't just course requirements—they're essential skills for any scientific research career. The more you invest in mastering them now, the more prepared you'll be for real research challenges ahead.
:::

Ready to begin? Start with [Chapter 1: Software Setup](01-software-setup-guide) →