```yaml
# See docs at: https://mystmd.org/guide/frontmatter
version: 1
project:
  id: 76c4875f-ba4c-4a35-bc58-52304ee2705e
  title: "ASTR 596: Modeling the Universe"
  github: https://github.com/astrobytes-edu/astr596-modeling-universe
  toc:
    - file: index
      title: ASTR 596 Home
    - part: Course Information
      chapters:
        - file: course-info/syllabus
          title: Syllabus & Policies
        - file: course-info/schedule
          title: Schedule & Timeline
        - file: course-info/ai-guidelines-final
          title: AI Usage Guidelines
        - file: course-info/learning-guide-final
          title: Learning Guide
        - file: course-info/course-overview-final
          title: Course Overview
        - file: course-info/why-astr596-different-final
          title: Why This Course is Different
    - part: Python
      chapters:
        - file: python/index
          title: Why Python?
        - file: python/chapter1_environment
          title: Setting Up Your Environment
        - file: python/chapter2_fundamentals
          title: Fundamentals & Control Flow
        - file: python/chapter3_functions
          title: Functions & Modules
        - file: python/chapter4_data_structures
          title: Data Structures
        - file: python/chapter5_object_oriented_programming
          title: Object-Oriented Programming
        - file: python/chapter6_error_handling
          title: Error Handling
        - file: python/chapter7_optimization
          title: "NumPy: Performance & Optimization"
    - part: Python Libraries
      chapters:
        - file: python/01_numpy
          title: "NumPy: Numerical Python"
        - file: python/02_matplotlib
          title: "Matplotlib: Visualization with Python"
        - file: python/03_scipy
          title: "SciPy: Scientific Python"
        - file: python/04_pandas
          title: "Pandas: Data Analysis Library"
    - part: Computational Methods
      chapters:
        - file: computational-methods/index
          title: Methods Overview
        - file: computational-methods/python-fundamentals/index
          title: Python Foundations
        - file: computational-methods/numerical-methods/index
          title: Numerical Methods
        - file: computational-methods/machine-learning/index
          title: Machine Learning
        - file: computational-methods/modern-frameworks/index
          title: Modern Frameworks
    - part: Astrophysics Applications
      chapters:
        - file: astrophysics/index
          title: Applications Overview
        - file: astrophysics/stellar-physics/index
          title: Stellar Physics
        - file: astrophysics/gravitational-dynamics/index
          title: Gravitational Dynamics
        - file: astrophysics/radiative-transfer/index
          title: Radiative Transfer
        - file: astrophysics/advanced_topics/index
          title: Advanced Topics
    - part: Short Projects
      chapters:
        - file: short-projects/index
          title: Projects Overview
        - file: short-projects/0_project_submission_guide
          title: Submission Guidelines
        - file: short-projects/1_project1_description
          title: "Project 1: Python/OOP/Stellar Physics"
        - file: short-projects/2_project2_description
          title: "Project 2: N-Body Dynamics & Monte Carlo"
        - file: short-projects/3_project3_description
          title: "Project 3: Linear Regression & ML"
    - part: Final Project
      chapters:
        - file: final-project/index
          title: Project Overview
        - file: final-project/final_project_guide
          title: Detailed Guide & Requirements
    - part: Reference Materials
      chapters:
        - file: reference/index
          title: Reference Overview
        - file: reference/software-setup-guide
          title: Software Setup Guide
        - file: reference/cli-intro-guide
          title: CLI Intro Guide
        - file: reference/git-intro-guide
          title: Git Intro Guide
        - file: reference/cli-advanced-guide
          title: CLI Advanced Guide (Optional)
site:
  template: book-theme
  title: "ASTR 596: Modeling the Universe"
  options:
    logo_text: "✨ASTR 596✨"
    custom_css: "_static/custom.css"
```