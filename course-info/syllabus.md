# ASTR 596: Modeling the Universe

**Fall 2025 - San Diego State University**  

**Fridays 11:00 AM - 1:40 PM | PA 215**

## Instructor Information

- **Dr. Anna Rosen**
- **Office:** Physics 239
- **Hacking Hours:** TBD in class (also available by appointment)
- **Email:** <alrosen@sdsu.edu>

## Course Information

- **Prerequisites:** Physics 196; MATH 254 or 342A; or equivalent with instructor permission, or graduate standing
- **Meeting Time:** Fridays 11:00 AM - 1:40 PM
- **Format:** ~30-45min lecture review + 2h hands-on project work
- **Location:** PA 215
- **Course Website:** <www.anna-rosen.com>
- **Platforms:** Canvas, Slack, GitHub Classroom
- **Expectations:** Students must come prepared having completed assigned JupyterBook readings

### Class Meeting Structure

**Pre-Class Preparation (Required):**

- Complete assigned JupyterBook chapter readings
- Review project requirements if new project assigned
- Prepare questions on material and implementation challenges

**Friday Class Sessions:**

- **11:00-11:45 AM:** Interactive review of week's concepts, Q&A on readings, clarification of project requirements
- **11:45 AM-1:40 PM:** Hands-on project work with pair programming, implementation support, and peer collaboration

Students are expected to arrive prepared, having engaged with the material. The lecture review assumes familiarity with the JupyterBook content and focuses on clarification, deeper insights, and connecting theory to implementation.

## Course Description

This course provides a hands-on introduction to the practice and theory of scientific computing, with an emphasis on numerical methods and machine learning, applied to astrophysical problems. Beginning with Python programming fundamentals and object-oriented design, the course progresses through sophisticated numerical methods including N-body dynamics, Monte Carlo radiative transfer, Bayesian inference, Gaussian processes, and culminates with neural networks. Students will implement all algorithms from first principles ("glass box" approach) before transitioning to modern frameworks (JAX ecosystem, e.g., Equinox, Flax, Optax). The course emphasizes professional software development practices, responsible AI integration, and preparation for computational research careers.

**Important Note:** I'm not testing your physics knowledge. I'm teaching you to be excellent scientific programmers. Astrophysics is our playground, not our examination subject. All necessary equations and scientific background will be provided — *your* task is to understand the scientific concepts, implement them correctly, and connect the computation to the physics. You'll learn the science through building it, without the stress of physics exams.

## Course Philosophy & Approach

This course provides comprehensive coverage of essential computational methods in astrophysics through a "glass box" approach—you'll build algorithms from first principles to understand both how and why they work. Rather than spending an entire semester on theoretical proofs for a single method, we spend 2-3 weeks per topic developing deep, practical understanding through implementation and experimentation. This is not a "cookbook" course—you will understand the mathematical foundations, physical intuition, computational implementation, and practical limitations of every algorithm you build.

By implementing methods from scratch while understanding their foundations, you'll develop both theoretical insight and practical skills. This course is your launching pad for computational research, equipping you to self-study advanced topics, implement papers, and build custom solutions when existing tools fall short.

**See the Course Learning Guide for expanded philosophy, detailed learning strategies, and additional resources.**

## Course Learning Outcomes

Upon successful completion of this course, students will be able to:

1. **Implement numerical schemes** for solving scientific problems using Python, employing advanced programming paradigms including object-oriented programming (OOP).

2. **Develop professional software practices** including modular algorithm design, meaningful documentation, version control, testing, and effective code structuring.

3. **Master key numerical techniques** including numerical integration, root finding, model fitting, and solving ordinary and partial differential equations, understanding both how and why these methods work.

4. **Apply Monte Carlo methods** to complex astrophysical problems including radiative transfer and Bayesian inference, implementing MCMC from scratch.

5. **Build neural networks from fundamentals** implementing backpropagation, gradient descent, and modern architectures without relying on libraries.

6. **Utilize modern computational frameworks** translating implementations to the JAX ecosystem (automatic differentiation, GPU acceleration, and differentiable programming).

7. **Integrate AI tools strategically** through a scaffolded three-phase approach while maintaining deep understanding and critical evaluation skills.

8. **Simulate advanced astrophysical phenomena** including N-body dynamics, stellar physics, and radiative processes with proper physics.

9. **Communicate computational methods and scientific results effectively** through written reports, code documentation, and oral presentations.

10. **Think computationally about physics** developing intuition for numerical stability, convergence, and the connection between physical and computational constraints.

## Required and Recommended Materials

### Textbooks (Free Online Resources)

- ASTR 596 Course JupyterBook (<www.anna-rosen.com>)
- VanderPlas, *Python Data Science Handbook* (<https://jakevdp.github.io/PythonDataScienceHandbook/>)
- Ting, *Statistical Machine Learning for Astronomy* (<https://arxiv.org/abs/2506.12230>)

### Software Requirements

- Python 3.10+ with scientific stack (`NumPy`, `SciPy`, `matplotlib`, `pandas`, `jax`, `equinox`, `flax`, `optax`)
- Git and GitHub account (<https://github.com/>)
- Jupyter Lab/Notebooks (**Project 1 only** - afterwards, all code in .py scripts)
- IDE (Integrated Development Environment, [VS Code](<https://code.visualstudio.com/>) recommended) with **ALL AI assistants disabled for entire semester**.
- Terminal/command line access (introduced in class and course materials)

**Recommended Python Installation Method:** Install Python and all required packages using `Miniforge`/`Conda` for simplified dependency management. To do this:

- Download Miniforge: <https://conda-forge.org/download/> or install from source <https://github.com/conda-forge/miniforge>

`Miniforge` provides conda package manager with conda-forge as the default channel and ensures reproducible environments across different operating systems

### Computational Resources

- **SDSU Instructional Cluster (Verne):** <https://sdsu-research-ci.github.io/instructionalcluster>
- **GitHub Classroom:** <https://sdsu-research-ci.github.io/github/classroom>
- All students will have access to high-performance computing resources for intensive computations.

## Grading Information

### Assessment Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Projects 1-6** | 50% | 8.33% each, due biweekly on Mondays |
| **Growth Memos** | 10% | 6 reflections at 1.67% each, linked to Projects 1-6 |
| **Technical Growth Synthesis** | 5% | Cumulative reflection due Dec 11 |
| **Final Project** | 25% | JAX implementation with research component due Dec 18|
| **Participation & Engagement** | 10% | Pre-class preparation, active contribution, collaboration |

**Technical Growth Synthesis (5%):** A final cumulative reflection (2-3 pages) due December 11 that synthesizes your computational learning journey across all projects. Unlike individual growth memos which are sequential, this document should demonstrate how your computational thinking, problem-solving approaches, and programming skills evolved throughout the entire course. Submitting this before the final project allows you to recognize your growth before tackling the culminating challenge.

**Participation & Engagement (10%):** Active participation requires coming to each class having completed the assigned readings and initial project attempts. Students are expected to contribute meaningfully to discussions, ask clarifying questions, answer peers' questions when able, engage productively in pair programming and peer review, and help create a collaborative learning environment. Preparation is non-negotiable: unprepared students cannot contribute effectively to our small group's collective learning.

### Grading Scale

- A: 93-100% (Outstanding) | A-: 90-92%
- B+: 87-89% | B: 83-86% (Praiseworthy) | B-: 80-82%
- C+: 77-79% | C: 73-76% (Average) | C-: 70-72%
- D+: 67-69% | D: 63-66% (Minimally Passing) | D-: 60-62%
- F: Below 60% (Failure)

**Note:** Final grade distribution and any curving will be at the instructor's discretion based on overall class performance and demonstrated effort.

## Growth Memos

Six reflective memos (1-2 pages each) submitted as PDFs to *Canvas* on Wednesdays following project completion. These memos focus on your development as a computational scientist, not just the technical content. You'll reflect on your problem-solving strategies, debugging approaches, metacognitive awareness, skill development, and evolution as a programmer. A standard set of prompts will guide your reflection on how you're learning to think computationally, overcome challenges, and develop professional practices. This timing ensures reflection while the experience is fresh.

## Course Schedule

### Projects Overview

Projects are carefully scaffolded to build upon each other—later projects may require importing and extending code from earlier ones. Each project deepens your computational toolkit while reinforcing previous concepts. Projects are assigned on Mondays (posted to GitHub Classroom) and due the following Monday at 11:59 PM. This schedule allows students to review requirements before Friday's class, where we'll work on implementation together.

| Project | Assigned | Due Date | Topic |
|---------|----------|----------|-------|
| **Project 1** | Aug 25 (Mon) | Sept 8 (Mon) | Python/OOP/Stellar Physics Basics |
| **Project 2** | Sept 8 (Mon) | Sept 22 (Mon) | ODE Integration + N-Body Dynamics + Monte Carlo Sampling |
| **Project 3** | Sept 22 (Mon) | Oct 6 (Mon) | Regression/ML Fundamentals |
| **Project 4** | Oct 6 (Mon) | Oct 20 (Mon) | Monte Carlo Radiative Transfer |
| **Project 5** | Oct 20 (Mon) | Nov 3 (Mon) | Bayesian/MCMC |
| **Project 6** | Nov 3 (Mon) | Nov 17 (Mon) | Gaussian Processes |
| **Final Project** | Nov 17 (Mon) | Dec 18 (Thu) | Neural Networks (From Scratch + JAX ecosystem) |

## Project Submission Requirements

### Development Environment Policies

**Jupyter Notebooks:** Allowed ONLY for Project 1. Starting with Project 2, all code must be written as modular Python scripts (.py files). This requirement develops essential professional skills including code reusability, proper testing, terminal proficiency, and version control practices.

**IDE AI Assistants:** Must remain disabled for the ENTIRE semester. This includes GitHub Copilot, Cursor AI, VS Code AI suggestions, and any AI-powered autocomplete. Developing programming "muscle memory" requires actively typing code without AI intervention.

### Each Project Submission Must Include

1. **Code Components:**
   - Modular Python scripts with clear organization.
   - Requirements.txt file listing dependencies.
   - Proper imports and function definitions.
   - No Jupyter notebooks after Project 1.

2. **Project Memo (Markdown .md format):**
   - Summary of methodology and approach.
   - Key results with embedded and readable plots with descriptive captions.
   - Understanding and interpretation of results, demonstrating conceptual grasp.
   - Challenges encountered and solutions.
   - Computational performance observations.
   - Informal tone, typically 2-5 pages.

3. **Documentation:**
   - README.md with installation and usage instructions.
   - Docstrings for all functions and classes.
   - Inline comments for complex algorithms.
   - Example usage scripts.

4. **GitHub Classroom Requirements:**
   - Regular commits demonstrating progress.
   - Meaningful commit messages.
   - Final push by Monday 11:59 PM deadline.
   - Proper .gitignore file.

**See the Project Submission Guide for detailed requirements, grading rubric, and examples.**

### Final Project: Neural Network Synthesis

The final project leverages your existing codebase to tackle a new, more advanced scientific question using neural networks. You'll select one of your previous implementations (P1-P6), refactor it to JAX, then apply neural network methods to solve a related but distinct problem that extends beyond the original project scope. This approach lets you build on familiar code while exploring how NNs enable new scientific investigations that would be difficult with classical methods alone.

**Requirements:**

- Select and refactor previous code to JAX ecosystem.
- Define a new scientific question that extends the original project.
- Implement NN solution from scratch + use JAX tools (Equinox/Flax/Optax).
- Demonstrate why NNs enable this new investigation.
- 8-12 page research report including plots and references.
- 10-minute presentation during finals week.

See detailed Final Project Guide for inspiration, project ideas, and requirements.

## Course Policies

### Attendance Policy

Attendance at Friday sessions is essential as they combine lecture and hands-on lab work. There are no recordings available. While attendance is not explicitly tracked, participation grades require active engagement in class, which is impossible without being present. Two absences are permitted without penalty. Additional absences may impact your participation grade and project success.

### Late Work Policy

- One no-questions-asked 2-day extension per semester.
- Must be requested at least 24 hours before the Monday 11:59 PM deadline.
- Extensions requested with less than 24 hours notice will not be granted and standard late penalties apply.
- Additional extensions only for documented emergencies.
- 10% penalty per day after grace period.
- Final project extensions strongly discouraged.
- Early submissions are encouraged—you may submit anytime after the project is assigned.

### Collaboration Policy

Pair programming is encouraged during lab sessions. While you may discuss approaches with classmates, all submitted code must be individually written and understood. Acknowledge all collaborators in your submissions. You must be able to explain every line of code you submit.

### Pair Programming Logistics

With our small class size, pairs will be assigned randomly and rotate each week to ensure everyone works with different partners. If we have an odd number due to absence, one group will have three members or someone will work independently.

**Equal Contribution Policy:** Pair programming is a privilege, not a right. While you may collaborate, discuss ideas, and look at each other's code during class, every student must submit their own independently written code. Copying or submitting identical code will result in strict disciplinary action. The goal is mutual support and learning — help each other understand concepts, debug issues, discuss ideas, and think through problems, but write your own implementation. This privilege will be revoked for students who abuse it.

### AI Usage Policy (Three-Phase Scaffolded Approach)

#### Phase 1: Foundation Building (Weeks 1-4)

- Minimal AI usage, Python documentation-first approach.
- 20-30 minute struggle rule before seeking help.
- AI only for debugging after genuine effort.

#### Phase 2: Strategic Integration (Weeks 5-8)

- Use AI for clarification after consulting documentation.
- Verify all AI suggestions against official sources.
- Begin developing prompt engineering skills.

#### Phase 3: Professional Practice (Weeks 9-16)

- Full AI integration with critical evaluation.
- Use AI as research accelerator.
- Maintain deep understanding requirement.

All AI usage must be cited in code comments and project memos. **See the detailed AI Usage Policy & Guide for specific examples and best practices.**

## Growth Mindset

This course operates on the principle that intelligence and computational abilities are not fixed traits but skills that can be developed through dedication and practice. Your current ability is just your starting point.

**Our Classroom Culture:**

- Struggle is necessary for growth - bugs and errors are where learning happens.
- Questions are celebrated, especially "basic" ones.
- Helping peers solidifies your own understanding.
- Every student can master this material with effort and support.

The specific tools you learn today will evolve, but your ability to learn, adapt, and think computationally will define your career. This course builds that meta-skill: learning how to learn complex computational methods.

## Red Flags: When You're Struggling

Recognize these warning signs and seek help immediately:

- **Can't explain your code** - If you can't describe what each line does, you don't understand it.
- **Copy-pasting without understanding** - Using code you found without knowing why it works.
- **Avoiding error messages** - Clearing outputs instead of reading and understanding errors.
- **Not testing incrementally** - Writing 100+ lines before running anything.
- **Skipping the readings** - Trying to start projects without completing JupyterBook chapters.
- **AI dependency** - Can't write basic functions without AI assistance.
- **Isolation** - Not asking questions in class or on Slack when confused.

If you notice these patterns, please reach out immediately. These are fixable problems, but only if addressed early.

**For comprehensive learning strategies, debugging tips, and additional resources, see the Course Learning Guide.**

## Resources

For comprehensive documentation, debugging strategies, and learning resources, see the **Course Learning Guide** on the course website.

## Academic Integrity

All work submitted must be your own, including code, written analysis, and project documentation. Plagiarism includes submitting code you cannot explain or understand, copying written work, or using content from online sources, AI tools, or other students without proper attribution and understanding.

To ensure academic integrity and support your learning, I may ask you to explain any aspect of your submitted work during office hours or class discussions. This is a normal part of the learning process and helps verify that you truly understand the concepts and methods you've implemented.

Violations will be reported to the Center for Student Rights and Responsibilities and may result in course failure. See SDSU's Academic Integrity Policy for details.

## Accommodations

Students with disabilities who may need accommodations should make an appointment with Student Disability Services (<sds@sdsu.edu>, 619-594-6473). Please consult with SDS within the first two weeks of class.

## Land Acknowledgement

We acknowledge that SDSU sits on the traditional territory of the Kumeyaay Nation. We honor their continued connection to this region and recognize their continuing presence.

## Important Dates

| Date | Item | Submission Method |
|------|------|-------------------|
| Sept 8 (Mon) | Project 1 Due | GitHub Classroom |
| Sept 10 (Wed) | Growth Memo 1 Due | Canvas PDF |
| Sept 22 (Mon) | Project 2 Due | GitHub Classroom |
| Sept 24 (Wed) | Growth Memo 2 Due | Canvas PDF |
| Oct 6 (Mon) | Project 3 Due | GitHub Classroom |
| Oct 8 (Wed) | Growth Memo 3 Due | Canvas PDF |
| Oct 20 (Mon) | Project 4 Due | GitHub Classroom |
| Oct 22 (Wed) | Growth Memo 4 Due | Canvas PDF |
| Nov 3 (Mon) | Project 5 Due | GitHub Classroom |
| Nov 5 (Wed) | Growth Memo 5 Due | Canvas PDF |
| Nov 17 (Mon) | Project 6 Due | GitHub Classroom |
| Nov 19 (Wed) | Growth Memo 6 Due | Canvas PDF |
| Nov 21 (Fri) | Final Project Proposal Due | Canvas PDF |
| Dec 5 (Fri) | Final Project Progress Report | Canvas PDF |
| Dec 11 (Wed) | Technical Growth Synthesis Due | Canvas PDF |
| Dec 17 or 18 | Final Presentations | In-person (TBD) |
| Dec 18 (Thu) | Final Project Due | GitHub + Canvas |

## Frequently Asked Questions

**Q: I have a question about the course.**
A: Check the syllabus first. If it's answered here, my response will only be "Please see the syllabus." This document is our contract—you're responsible for knowing its contents.

**Q: Where can I find more details about [topic]?**
A: The syllabus contains all essential policies. Expanded guidance is in the supplementary documents on the course website.

**Q: I've never used the command line. Will I struggle?**
A: Week 1 covers basics. Use AI tutors for practice. You'll be comfortable by Week 3.

**Q: Can I use different libraries than specified?**
A: No for Projects 1-6 (glass box approach). Yes for final project extensions.

**Q: What if I can't attend a Friday session?**
A: Contact instructor ASAP. Review recordings if available, complete work independently, check in via Slack.

**Q: How much time should I spend on each project?**
A: Expect 12-18 hours per 2-week project outside of class. Start early, work incrementally.

**Q: Can I use AI to write my code?**
A: No. Follow the three-phase policy. AI is for learning, not replacing your thinking.

**Q: When can I use AI tools like Claude or ChatGPT?**
A: You can use AI from day one to understand concepts, clarify theory, and reinforce learning. Just don't use it to write your code. Even in Phase 3, use AI sparingly for coding—primarily for debugging help, finding documentation, or understanding error messages, not for generating solutions.

**Q: What if I finish a project early?**
A: Great! Submit it early, explore suggested extensions, help peers, or start reading ahead for the next project.

**Q: What if my code works but is inefficient?**
A: Working code is the first goal. We'll discuss optimization in class and extensions.

**Q: Do I need to memorize equations?**
A: No. All physics equations will be provided. Focus on understanding what they mean physically and can be solved computationally.

**Q: Can I work ahead on projects?**
A: Yes! The JupyterBook provides the foundation. But wait for the official assignment for full requirements.

**Q: What programming background do I need?**
A: Basic Python. We'll build from there.

## Your Responsibility

This syllabus constitutes our course contract. You are responsible for reading and understanding all policies stated here. Questions answered in this document will receive the response: "It's in the syllabus."

## Additional Course Documents

Detailed guides expanding on these policies are available on the course website:

- **Course Learning Guide** - Expanded philosophy, learning strategies, debugging tips, resources
- **Project Submission Guide** - Detailed requirements, rubric, and examples
- **AI Usage Policy & Guide** - Specific examples and best practices
- **Final Project Guide** - Project ideas, requirements, and inspiration
