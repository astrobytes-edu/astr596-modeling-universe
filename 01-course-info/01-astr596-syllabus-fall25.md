---
title: Syllabus (Fall 2025)
subtitle: "ASTR 596: Modeling the Universe"
exports:
  - format: pdf
---
**Fall 2025 - San Diego State University (SDSU)**  
**Fridays 11:00 AM - 1:40 PM | PS 256**

## Instructor Information

- **Dr. Anna Rosen**
- **Office:** Physics 239
- **Student "Hacking" Hours:** Wednesdays 1-2 PM (P-239) or by appointment
- **Email:** <alrosen@sdsu.edu>

## Course Information

- **Prerequisites:** Physics 196; MATH 254 or 342A; or equivalent with instructor permission, or graduate standing
- **Meeting Time:** Fridays 11:00 AM - 1:40 PM
- **Format:** ~30-45min lecture review + 2h hands-on project work
- **Location:** PS-256
- **Course Website:** <https://astrobytes-edu.github.io/astr596-modeling-universe/>
(*All course material and information will be posted here.*)
- **Virtual Tutor:** [✨ASTR 596: Modeling the Universe Tutor✨ *(Custom GPT)*](https://chatgpt.com/g/g-68aabb9278d08191926feb3f5512686c-astr-596-modeling-the-universe-tutor)
- **Platforms:** Canvas, Slack, GitHub Classroom

### Class Meeting Structure

**Pre-Class Preparation *(Required)*:**

- Complete assigned readings on course website.
- Review project requirements if new project assigned.
- Prepare questions on reading material and assigned project implementation challenges.

**Friday Class Sessions:**

- **11:00-11:40 AM:** Interactive review of the week's concepts, Q&A on required course readings, clarification of project requirements.
- **11:40 AM-1:40 PM:** Hands-on project work with pair programming, implementation support, and peer collaboration.

## Course Description

This course provides a hands-on introduction to the practice and theory of scientific computing, with an emphasis on numerical methods and machine learning, applied to astrophysical problems. Beginning with Python programming fundamentals and object-oriented design, the course progresses through sophisticated numerical methods including N-body dynamics, Monte Carlo radiative transfer (MCRT), Bayesian inference and MCMC, Gaussian processes, and culminates with neural networks. Students will implement all algorithms from first principles ("glass box" approach) before transitioning to modern frameworks (JAX ecosystem). The course emphasizes professional software development practices, responsible AI integration, and preparation for computational research and industry careers.

**Important Note:** I'm not testing your astrophysics knowledge. All necessary equations and scientific background will be provided. Your task is to learn and understand the scientific concepts, implement them correctly in Python, and connect the computation to the astrophysics. The focus is on computational thinking and implementation skills that transfer to any research domain.

**For an expanded description of the course philosophy and approach, see:**

- [Why ASTR 596 is Designed This Way](06-why-astr596-is-different) 
- [Understanding Your Learning Journey in ASTR 596](05-astr596-course-overview)

## Course Learning Outcomes

Upon successful completion of this course, students will be able to:

1. **Implement numerical schemes** for solving scientific problems using Python, employing advanced programming paradigms including object-oriented programming (OOP).
2. **Develop professional software practices** including modular algorithm design, meaningful documentation, version control, testing, and effective code structuring.
3. **Master key numerical techniques** including numerical integration, Monte Carlo methods, model fitting, and solving ordinary differential equations.
4. **Apply Monte Carlo methods** to complex astrophysical problems including radiative transfer and Bayesian inference.
5. **Build neural networks from fundamentals** implementing backpropagation and gradient descent without relying on libraries.
6. **Utilize modern computational frameworks** translating implementations to the JAX ecosystem.
7. **Integrate AI tools strategically** through a scaffolded three-phase approach while maintaining deep understanding.
8. **Simulate advanced astrophysical phenomena** including N-body dynamics, stellar physics, and radiative processes.
9. **Communicate computational methods and scientific results effectively** through written reports and code documentation.
10. **Think computationally about astrophysics** developing intuition for numerical stability and convergence.

### Where Outcomes Are Assessed

- LO1 (implement schemes) → Projects 1–6, Final Project
- LO2 (professional practices) → repo hygiene, tests, Growth Memos
- LO3 (numerical techniques) → Projects 2–4
- LO4 (Monte Carlo methods) → Projects 2, 3, 5
- LO5 (build NNs) → Final Project
- LO6 (JAX migration) → Final Project
- LO7 (strategic AI) → Growth Memos + code comments
- LO8 (advanced phenomena) → Projects 2–5 (N-body/RT/Bayesian)
- LO9 (communication) → Research Memos + Final Research Report + Technical Growth Synthesis
- LO10 (numerical intuition) → Growth Memos + Technical Growth Synthesis + validation analyses + project extensions

## Materials

### Textbooks (Free Online Resources)

- **Required:** Rosen (2025), [www.astrobytes-edu.github.io/astr596-modeling-universe](https://astrobytes-edu.github.io/astr596-modeling-universe/), official ASTR 596 Course Website (powered by [MyST Markdown](https://mystmd.org/))

- Linge & Langtangen (2020), [Programming for Computations - Python (2nd Edition)](https://library.oapen.org/bitstream/id/35449b29-cca0-4d71-8553-2d49609b75fd/1007055.pdf), Springer Open

- Mehta et. al. (2018), [A High-Bias, Low-Variance Introduction to Machine Learning for Physicists](https://doi-org.libproxy.sdsu.edu/10.1016/j.physrep.2019.03.001), [arXiv e-print](https://arxiv.org/abs/1803.08823)

- Deisenroth, Faisal, & Ong (2020), [Mathematics for Machine Learning](https://mml-book.github.io/), Cambridge University Press

- Ting (2025), [Statistical Machine Learning for Astronomy *(arXiv preprint)*](https://arxiv.org/abs/2506.12230)

### Software Requirements

- Python 3.10+ with scientific stack
- Git and GitHub account
- Jupyter Lab/Notebooks (Project 1 only)
- IDE (VS Code recommended) with **ALL AI assistants and tab completion disabled**
- Terminal/command line access

**For detailed setup instructions, see:** [Software Installation Guide](../02-getting-started/02-software-setup)

### Computational Resources

- **SDSU Instructional Cluster (Verne):** <https://sdsu-research-ci.github.io/instructionalcluster>
  
Access to SDSU's Jupyter Hub provided to enrolled students, however students are strongly encouraged to setup their own Python environment and IDE workspace on their own machines. Detailed instructions are included in the [Getting Started](../02-getting-started/index.md) module.

- **ASTR 596 GitHub Classroom:** <https://classroom.github.com/classrooms/226159624-astr-596-modeling-the-universe>

All projects distributed and submitted here. Your last repository push by the deadline will count as your submitted assignment. For more information please see: <https://sdsu-research-ci.github.io/github/students>.

## Grading Information

### Assessment Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Short Projects 1-5** | 50% | Due roughly every 2-3 weeks on Mondays at 11:59 PM PT via GitHub Classroom (includes Growth Memos). |
| **Growth Memos** | 10% | Integrated with each project submission (.md or PDF in project repo) |
| **Final Project** | 25% | JAX implementation with research component. Due Thu Dec 18, 11:59 PM PT. |
| **Technical Growth Synthesis** | 5% | Comprehensive self-reflection growth portfolio integrated with Final Project submission (.md or PDF in project repo). |
| **Participation & Engagement** | 10% | Active contribution in lectures, lab activities, and peer collaboration (includes Project 0 completion). |

**Initial Course Reflection & Setup (Project 0)** must be completed on Canvas by Thursday, August 28 at 11:59 PM PT. This assignment ensures all students have reviewed the course policies and logistics, set up their Python development environment with a Git account tied to their SDSU email address, and completed an initial reflection/survey on their learning goals and prior Python experience. This ensures students come to the first class prepared for hands-on Python activities. **Note:** Project 0 is the only assignment submitted via Canvas and counts toward your Participation grade.

### Short Projects & Growth Memos (60% of course grade combined)

Each short project submission includes both technical implementation and a growth memo reflection. Together these components total 60% of your course grade (50% projects + 10% memos).

### Short Project Components

- **Core Implementation** - Working solution with correct physics/algorithms.
- **Research Memo** - 2-3 pages text plus figures (LaTeX/PDF required).
- **Code Quality** - Documentation, organization, Git practices.
- **Validation** - Testing against known solutions or benchmarks.
- **Student-Led Exploration** - Self-directed investigation extending beyond base requirements (**required for graduate students**, optional but encouraged for undergraduates).

### Growth Memo Components (1-2 pages, submit as .md or PDF in project repo)

*An informal reflection on your learning journey and skill development progression - write authentically about:*

- Technical skills developed.
- Challenges encountered and solutions found.
- Connection to course concepts.
- AI usage and verification process.
- Next learning goals.
- Any other insights, surprises, or moments that shaped your understanding.

**Note:** The relative weight between project implementation and growth memo may vary by project based on learning objectives and complexity. Early projects may emphasize reflection more heavily to develop metacognitive skills, while later projects may weigh technical implementation more strongly as mastery develops.

### Research Memo Format

- Single-spaced, submit as PDF in project repo.
- 2-3 pages of text (not counting figures/references).
- Include literature citations, methods, results, and discussion.

**For detailed guidelines for the short projects, see:** [ASTR 596 Project Submission Guide](../08-short-projects/00-project-submission-guide)

### Final Project Components (25% of course grade)

| Component | Weight | Description |
|-----------|--------|-------------|
| **Code Package** | 55% | Professional package structure with `__init__.py`, tests, and documentation. |
| **Written Report** | 30% | Scientific writeup with scientific background and motivation, methods, results, and conclusions. |
| **Presentation** | 15% | 12-minute total: research project, technical growth summary, and Q&A. |

### Code Package Requirements

- Proper modular package structure with `__init__.py` files.
- Unit tests with >70% coverage.
- Documentation (docstrings, README.md, and Jupyter notebook tutorial).
- Clean Git history showing iterative development.
- Requirements.txt/environment.yml.

### Presentation Structure (12 minutes total)

- 7 minutes on research project (motivation, methods, results, conclusions).
- 3 minutes on technical growth journey (key learning moments, skill evolution).
- 2 minutes for Q&A.

### Technical Growth Synthesis (5% of course grade)

### Portfolio Components (3-5 pages excluding figures, submit with Final Project)

*An informal reflective narrative documenting your computational journey through the entire course - write authentically about your evolution as a computational scientist:*

- Synthesis of learning across all projects.
- Technical growth through code evolution (compare early vs. final code samples with explanations).
- Development of computational thinking skills.
- Student-led explorations' impact on your research approach and interests (what you gained, challenges, preparation for independent work).
- Shift in learning mindset (did you find yourself exploring beyond requirements out of curiosity rather than obligation).
- AI's impact on your learning journey (did it accelerate understanding, push you beyond requirements, teach you why domain expertise matters).
- Assessment of the AI scaffolding design (effectiveness of the structured approach, what worked/didn't, suggestions for improvement).
- Key moments, breakthroughs, or realizations that defined your growth.

### Participation & Engagement (10% of course grade)

| Level | Observable weekly behaviors |
|------|------------------------------|
| 10/10 | Prepared (specific Qs on readings), active in pair work and discussions, helps answer peer questions, posts helpful tips, provides constructive feedback |
| 8/10  | Prepared, steady collaboration, contributes to discussions, occasional Q&A |
| 6/10  | Minimal prep, limited engagement in discussions, passive participation |
| 4/10  | Unprepared, disengaged, minimal discussion/participation, slows partner progress |
| 2/10  | Rarely attends or participates, unprepared when present, no discussion contribution |
| 0/10  | Habitually absent, no engagement |

**Note:** Participation includes: class discussions, asking questions, answering instructor/peer questions, pair programming engagement, Slack contributions, peer code review, and completion of Project 0.

**Important:** This rubric serves as a guideline. Students may be strong in some areas (e.g., excellent pair programming) while needing improvement in others (e.g., class discussion). Your grade reflects overall engagement. Disruptive behaviors (habitual tardiness, excessive phone use, off-topic conversations) negatively impact participation grades regardless of other contributions.

### Major Assignment Due Dates

| Assignment | Due Date |
|------------|----------|
| Initial Course Reflection & Software Setup (Project 0)* | Thursday, Aug 28, 11:59 PM PT |
| Project 1: Python Fundamentals, OOP & Classes, `numpy`/`matplotlib`, <br> & Stellar Physics (with Growth Memo 1) | Monday, Sept 8, 11:59 PM PT |
| Project 2: ODE Integration & N-Body Dynamics, <br> Monte Carlo Sampling (with Growth Memo 2) | Monday, Sept 22, 11:59 PM PT |
| Project 3: Monte Carlo Radiative Transfer (with Growth Memo 3) | Monday, Oct 13, 11:59 PM PT |
| Project 4: Bayesian/MCMC (with Growth Memo 4) | Monday, Nov 3, 11:59 PM PT |
| Project 5: Gaussian Processes (with Growth Memo 5) | Monday, Nov 24, 11:59 PM PT |
| Final Project & Technical Growth Synthesis | Thursday, Dec 18, 11:59 PM PT |

*Counts toward Participation grade

### Grading Scale

Final letter grades will be calculated using the grading scale listed below. These percentages represent guaranteed thresholds - earning the stated percentage guarantees at least that grade. The instructor reserves the right to adjust borderline grades upward based on effort, improvement, and engagement.

- A: 93-100% | A-: 90-92%
- B+: 87-89% | B: 83-86% | B-: 80-82%
- C+: 77-79% | C: 73-76% | C-: 70-72%
- D+: 67-69% | D: 63-66% | D-: 60-62%
- F: Below 60%

## Course Policies

### Late Work Policy

- **One free extension per semester:** Request ≥24h before deadline → 2-day grace, no penalty.
- **Late submission penalty:** 10% per day (24 hours), maximum 3 days late (30% deduction).
- **After 3 days:** Not accepted without documented emergency.
- **Note:** Late policy does not apply to Project 0, which **must** be completed before first class.

### Regrade Requests

- Submit within 7 calendar days of grade release.
- ≤200-word written justification referencing rubric and assignment expectations criteria.
- I will re-evaluate the entire submitted assignment (score may go up/down/unchanged).

### Peer Collaboration & Pair Programming

- **Allowed:** discussing strategy, whiteboarding equations & algorithm design (pseudocode), sharing tests you wrote
- **Not allowed:** sharing solution code or copying any code blocks
- **Must:** credit collaborators and list contributions in README.md project document

### AI Usage Policy

This course uses a novel three-phase scaffolded approach to AI integration. **Note:** AI raises the bar for expertise. LLMs confidently generate plausible-looking code that can be subtly wrong, numerically unstable, or inefficient. They also hallucinate facts, misexplain concepts, and make mathematical errors. Our three-phase approach builds your AI literacy systematically – teaching you to understand deeply enough to catch these errors and know why they're wrong.

- **Universal rules (applies all semester):** Always try documentation first. When docs aren't enough for understanding concepts (e.g., complex Python functions, math derivations, astrophysics concepts), AI is encouraged for clarification. For code: verify all AI suggestions; disclose AI use in submissions. *If you can't explain a line of code, you can't submit it.*

- **Phase 1 — Foundations (Weeks 1–6):** No AI-generated first drafts of code. After 30 minutes of documented struggle, AI may be used for debugging/clarification only.

- **Phase 2 — Strategic Integration (Weeks 7–12):** Once a baseline solution works, AI may propose refactors, tests, docstrings, and performance ideas. Keep/Reject each suggestion with a 1–2 line rationale in comments.

- **Phase 3 — Professional Practice (Weeks 13–16):** AI allowed for acceleration/boilerplate. All non-trivial logic must be authored or rewritten from memory and justified.

**For complete AI usage guidelines and examples, see:** [AI Usage Policy & Learning Guide](03-astr596-ai-policy)

### Academic Integrity

The California State University system requires instructors to report all instances of academic misconduct to the [Center for Student Rights and Responsibilities](https://sacd.sdsu.edu/student-rights). Academic dishonesty will result in disciplinary review by the University and may lead to probation, suspension, or expulsion. Instructors may also, at their discretion, penalize student grades on any assignment discovered to have been produced in an academically dishonest manner such as cheating and plagiarism as described here: [SDSU Academic Integrity Policy](https://sacd.sdsu.edu/student-rights/academic-dishonesty).

In this course, submitted work must demonstrate your understanding. You are expected to:

- Follow the AI Usage Policy guidelines for your current phase (see above).
- Document all AI assistance according to course requirements.
- Be able to explain any aspect of your submitted code or analysis.
- Ensure all code you submit is either written by you or fully understood and verifiable.

**Collaboration is encouraged through pair programming and peer review, but each student must submit their own implementation and growth reflections.** Using another student's code, submitting AI-generated code you cannot explain, or misrepresenting AI assistance as your own work constitutes academic dishonesty.

## Fostering a Growth Mindset

A **growth mindset** is the belief that intelligence, abilities, and talents are malleable and can be developed through effort and persistence, not fixed traits you're born with. This mindset is key for succeeding in this course. In ASTR 596, you'll encounter challenging concepts that may initially seem overwhelming. **This is normal and expected: it means you're learning.** A growth mindset allows you to:

- View struggles as opportunities to improve, not signs of inadequacy.
- Understand that "I don't know" means "I don't know *yet*."
- Recognize that debugging and errors are how we learn, not failures.
- Embrace mistakes as essential learning moments.
- Celebrate progress over perfection.

Your capability will grow exponentially throughout this course. What feels overwhelming in Week 1 will feel manageable by Week 5, and by Week 15 those early challenges will seem trivial.

Our classroom cultivates a growth mindset by creating a supportive environment where:

- Questions are encouraged, not judged.
- Struggle is normalized and productive.
- Progress is measured against your past self, not others.
- Help-seeking is a sign of strength, not weakness.

Trust the process, embrace the challenge, and discover that you're capable of doing hard things.

## Diversity and Inclusivity Statement

I consider this classroom to be a place where you will be treated with respect, and I welcome individuals of all ages, backgrounds, beliefs, ethnicities, genders, gender identities, gender expressions, national origins, religious affliations, sexual orientations, ability, and other visible and non-visible differences. All members of this class are expected to contribute to a respectful, welcoming and inclusive environment for every other member of the class. If something is said in class by myself or others that made you uncomfortable, please contact me or submit anonymous feedback. Anonymous feedback can be placed in my department mailbox (located in the mail/copier room, PA-122A) or given to Hilary Scalisi, Department Coordinator in office # P-132, for delivery to me.

## Essential Student Information

For essential information about student academic success, please see the [SDSU Student Academic Success Handbook](https://docs.google.com/document/d/1rXNpNGs1K7nIxcS73o6R-fxZqPIWQwS9gHD7XpIqjhM/edit) and the [SDSU Student Success Hub](https://studentsuccess.sdsu.edu/success-hub). For graduate student resources please refer to [SDSU's Graduate Student Resources](https://grad.sdsu.edu/current-students).

Class rosters are provided to the instructor with the student’s legal name. Please let the instructor know if you prefer an alternate name and/or gender pronoun.

SDSU provides disability-related accommodations via [Student Disability Services](https://sds.sdsu.edu) (email: <sds@sdsu.edu>). Please allow 10-14 business days for processing.

## Land Acknowledgement

SDSU sits on Kumeyaay land. The Kumeyaay people have lived in this region for over 10,000 years and continue to live here today.

## Your Responsibility

This syllabus constitutes our course contract. You are responsible for reading and understanding all policies stated here.
