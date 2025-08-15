```yaml
# See docs at: https://mystmd.org/guide/frontmatter
version: 1
project:
  id: 76c4875f-ba4c-4a35-bc58-52304ee2705e
  title: "ASTR 596: Modeling the Universe"
  keywords: [computational astrophysics, python, machine learning, monte carlo, neural networks, radiative transfer, bayesian inference, JAX]
  authors:
    - name: Anna Rosen
      email: alrosen@sdsu.edu
      affiliation: San Diego State University
  license: CC-BY-4.0
  subject: Computational Astrophysics
  venue:
    title: ASTR 596 - Fall 2025
    url: https://www.sdsu.edu
  github: https://github.com/astrobytes-edu/astr596-modeling-universe
  
  # Math rendering improvements
  math:
    mathjax3:
      tex:
        packages:
          '[+]': ['ams', 'physics']
        macros:
          # Common macros for the course
          vec: ['\boldsymbol{#1}', 1]
          del: '\nabla'
          avg: ['\langle #1 \rangle', 1]
          norm: ['\left\lVert #1 \right\rVert', 1]
  
  # MyST markdown extensions
  myst:
    enable_extensions:
      - amsmath
      - colon_fence
      - deflist
      - dollarmath
      - html_admonition
      - linkify
      - replacements
      - smartquotes
      - substitution
      - tasklist
  
  # Code formatting
  settings:
    code_style: github
    code_line_numbers: true
    code_copy: true
  
  toc:
    - file: index
      title: ASTR 596 Home
    - title: Course Information
      children:
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
    - title: Python
      children:
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
    - title: Python Libraries
      children:
        - file: python/01_numpy
          title: "NumPy: Numerical Python"
        - file: python/02_matplotlib
          title: "Matplotlib: Visualization with Python"
        - file: python/03_scipy
          title: "SciPy: Scientific Python"
        - file: python/04_pandas
          title: "Pandas: Data Analysis Library"
    - title: Computational Methods
      children:
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
    - title: Astrophysics Applications
      children:
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
    - title: Short Projects
      children:
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
    - title: Final Project
      children:
        - file: final-project/index
          title: Project Overview
        - file: final-project/final_project_guide
          title: Detailed Guide & Requirements
    - title: Reference Materials
      children:
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
    
    # Footer customization
    footer:
      content: "ASTR 596: Modeling the Universe | Fall 2025 | Dr. Anna Rosen | San Diego State University"
    
    # Code execution and interactivity
    launch_buttons:
      notebook_interface: jupyterlab
      binderhub_url: https://mybinder.org
      colab_url: https://colab.research.google.com
      thebe: true
    
    # Download options for students
    downloads:
      - pdf
      - ipynb
      - md
    
    # Repository integration
    repository:
      url: https://github.com/astrobytes-edu/astr596-modeling-universe
      path_to_book: ""
      branch: main
    use_repository_button: true
    use_issues_button: true
    use_edit_page_button: true
    
    # Announcement banner (update as needed for deadlines/announcements)
    # Uncomment and modify when you have announcements
    announcement: "🚨 Project 1 due Monday 11:59 PM! Office hours Thursday 11am."
    
    # Search configuration
    search:
      enable: true
      limit_results: 20
```