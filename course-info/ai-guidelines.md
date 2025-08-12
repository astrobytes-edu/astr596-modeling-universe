# ASTR 596: Artificial Intelligence Policy and Learning Framework

## Course Philosophy: AI as a Performance Amplifier, Not a Substitute

**The Reality**: AI tools are transforming scientific computing and research. As future astrophysicists, you will work in an AI-integrated environment where these tools are standard professional practice.

**The Challenge**: **Recent MIT research suggests that heavy AI reliance weakens cognitive development.** Students who used ChatGPT showed weaker neural connectivity and couldn't remember what they'd just "written." AI raises the bar for human capability—it amplifies what we expect you to accomplish and understand, but only if you use it strategically.

**Our Approach**: This course treats AI as a **performance amplifier** for competent practitioners, not as a substitute for learning. You must first develop core competencies through productive struggle, then learn to use AI strategically to enhance your capabilities.

**Research-backed approach**: Ting & O'Briain (2025) found that students who used structured AI integration with documentation requirements **actually decreased their AI dependence over time** while developing stronger problem-solving skills and AI literacy. This contradicts common fears about AI creating academic dependency.

**Safe to Struggle**: Learning requires intellectual risk-taking, exploration, and yes, making mistakes. **The struggle is where cognitive growth happens.** This policy is designed to support genuine learning while building your capacity to work effectively with AI throughout your career.

## How to Cite AI Usage in Your Code

### **In-Code Documentation**
When AI assists with your code, document it clearly:

```python
# AI-assisted: ChatGPT helped optimize this nested loop structure
# Verified against NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
# Original approach used nested for loops, AI suggested vectorization
def calculate_distances(positions):
    # Implementation here
    pass

# Debugging with Claude: Identified index out of bounds error in original implementation
# Solution verified by testing edge cases [0, 1, n-1, n]
def get_neighbors(grid, i, j):
    # Implementation here
    pass
```

### **In Project Memos**
Include a dedicated "AI Usage" section:
```markdown
## AI Usage Documentation

**Conceptual Understanding**: Used ChatGPT to clarify difference between Euler and RK4 stability 
regions after reading Numerical Recipes Ch. 17.1. Verified understanding by implementing both 
and comparing error accumulation.

**Debugging Assistance**: Claude helped identify memory leak in particle array allocation. 
Solution involved properly clearing arrays between iterations (lines 145-150).

**Not AI-Generated**: Core algorithm implementation, physics calculations, and analysis 
were completed independently following course materials.
```

---

## Scaffolded AI Integration Framework

### **Phase 1: Foundation Building (Weeks 1-4)**
**Dates: Aug 29 - Sept 19**  
**Rule**: **Struggle First, AI Second**

- **Primary Learning**: Use official documentation, textbooks, and manual implementation
- **The 20-Minute Rule**: Spend at least 20-30 minutes wrestling with problems using only your mind before consulting AI
- **AI Usage**: Limited to **debugging assistance only** after genuine independent effort
- **Documentation Requirement**: All AI interactions must include links to relevant official documentation
- **Rationale**: Building neural pathways and pattern recognition that AI cannot provide—this is where cognitive growth happens

**Example Scenario**: Learning matplotlib plotting
1. ✅ **Correct Approach**: Study matplotlib documentation → understand Figure/Axes hierarchy → attempt implementation → struggle with specific issues → document your thinking → use AI to debug specific errors
2. ❌ **Incorrect Approach**: Ask AI "how to make a scatter plot" without first attempting to understand the underlying concepts

---

### **Phase 2: Strategic Integration (Weeks 5-8)**
**Dates: Sept 20 - Oct 17**  
**Rule**: **Documentation-First AI Assistance**

- **Primary Learning**: Continue documentation-first approach
- **AI Enhancement**: AI can be used for efficiency **after** demonstrating documentation literacy
- **Verification Requirement**: Must cross-reference AI suggestions with official sources
- **Critical Evaluation**: Must explain why AI suggestions are appropriate/inappropriate

**Example Scenario**: Adding legends to plots
1. ✅ **Correct Approach**: "I need to add a legend. I'll check matplotlib.pyplot.legend documentation first to understand all options, then use AI to help me implement the specific styling I want."
2. ❌ **Incorrect Approach**: Ask AI "add legend to plot" and copy-paste the first solution without understanding alternatives or parameters

---

### **Phase 3: Professional Practice (Weeks 9-16)**
**Dates: Oct 18 - Dec 18**  
**Rule**: **AI as Productivity Tool for Competent Practitioners**

- **Competency Assumption**: You now have foundation knowledge to evaluate AI suggestions critically
- **Professional Usage**: AI can be used for acceleration, exploration, and complex problem-solving
- **Quality Standard**: AI-assisted work must meet higher standards than manual work
- **Documentation Standard**: Continue citing AI usage and verification sources

---

## Specific AI Usage Rules

### **AI Usage Documentation in Growth Memos**
Rather than weekly summaries, document your AI usage in each Growth Memo (due after each project):
1. **Significant AI Interactions**: Describe 2-3 substantial uses of AI tools for that project
2. **Documentation Verification**: For each interaction, link to the primary sources you consulted
3. **Learning Reflection**: What did you learn about effective AI usage? What didn't work?
4. **Phase Compliance**: How did your AI usage align with the current phase expectations?

### **Portfolio Approach (Part of Technical Growth Synthesis)**
Your final Technical Growth Synthesis (due Dec 11) should include an AI literacy portfolio section demonstrating:
- **Representative Examples**: 3-4 well-documented AI interactions from across the semester showing your growth
- **Evolution Analysis**: How your AI strategies evolved from Phase 1 to Phase 3
- **Success and Failure Analysis**: Specific examples where AI helped and where it hindered learning
- **Professional Readiness**: Reflection on how you'll use AI in future research/industry work

### **Prohibited AI Usage**
- **Direct Problem Solving**: Asking AI to solve entire assignment problems
- **Conceptual Shortcuts**: Using AI to bypass understanding fundamental concepts
- **Unverified Implementation**: Using AI code without understanding or verification
- **Documentation Replacement**: Using AI instead of reading official documentation

### **Encouraged AI Usage**
- **Socratic Dialogue**: "Don't solve this for me, but help me understand what questions I should be asking"
- **Debugging Assistance**: After independent troubleshooting attempts and documentation consultation
- **Code Optimization**: After implementing working solutions manually to understand the core concepts
- **Alternative Perspectives**: "Here's my approach and where I'm stuck. What's a different mental model I could use?"
- **Pattern Recognition**: "I'm working on problems A, B, and C. What underlying principle connects them?"

### **The Cognitive Ownership Principle**
- **Never copy-paste AI responses** without understanding and rewriting in your own style
- **The Quote Test**: After AI assistance, close the conversation and summarize key insights from memory
- **The Teaching Test**: Be able to explain AI suggestions to a classmate in your own words

---

## Assessment Integration

### **Growth Memo AI Reflection Component**
Each Growth Memo (submitted Wednesdays after project completion) should include an AI usage section (~1 paragraph) addressing:
- **Memo 1 (Sept 10 - After Project 1)**: How did the 20-minute struggle rule affect your learning? What did you gain from avoiding immediate AI assistance?
- **Memo 2 (Sept 24 - After Project 2)**: Describe your debugging process. How did documentation-first help before using AI?
- **Memo 3 (Oct 8 - After Project 3)**: Compare solving problems with and without AI. What patterns do you notice?
- **Memo 4 (Oct 22 - After Project 4)**: How has AI helped or hindered your understanding of Monte Carlo methods?
- **Memo 5 (Nov 5 - After Project 5)**: Describe a time when AI gave incorrect advice. How did you identify and correct it?
- **Memo 6 (Nov 19 - After Project 6)**: How has your AI strategy evolved since Week 1? What's your professional approach going forward?

### **Understanding Verification**
Throughout the course, expect informal check-ins where I may:
- **Ask about your code**: "Walk me through why you chose this approach"
- **Discuss design decisions**: "What alternatives did you consider?"
- **Explore edge cases**: "What happens if we change this parameter?"
- **Connect concepts**: "How does this relate to what we learned in Project 2?"

These conversations help ensure you understand your implementations and can think critically about your code. This is supportive, not punitive—it's about reinforcing learning and catching confusion early.

---

## The Professional Reality

### **Why These Rules Matter**
1. **Industry Expectations**: Professional developers must understand their tools deeply enough to debug, optimize, and adapt AI suggestions
2. **Research Demands**: Scientific computing requires understanding trade-offs, limitations, and edge cases that AI may not address
3. **Career Longevity**: Practitioners who understand fundamentals adapt to new tools; those who don't become obsolete
4. **Quality Assurance**: In research, you must be able to verify and defend every aspect of your computational work

### **The Performance Standard**
- **Without AI**: You should be capable of solving problems manually, consulting documentation, and implementing solutions from first principles
- **With AI**: You should accomplish significantly more complex problems, higher quality implementations, and deeper exploration of edge cases
- **Integration Goal**: AI amplifies your capabilities without replacing your critical thinking

---

## Practical Examples

### **Acceptable AI Workflow: Implementing MCMC**
1. **Foundation**: Read course materials on MCMC theory and Metropolis-Hastings algorithm
2. **Documentation**: Study NumPy random module documentation for sampling functions
3. **Manual Implementation**: Code basic Metropolis-Hastings from mathematical principles
4. **AI Enhancement**: "I've implemented basic MCMC. Can you help me optimize the proposal distribution tuning based on my current acceptance rate calculation?"
5. **Verification**: Cross-reference AI suggestions with statistical computing literature

### **Unacceptable AI Workflow: Implementing MCMC**
1. **Shortcut**: "Write me a Metropolis-Hastings algorithm for parameter estimation"
2. **Copy-Paste**: Use AI code without understanding acceptance criteria or proposal distributions
3. **Skip Theory**: Never consult course materials or statistical references

---

## Academic Integrity and AI Misuse

### **AI Policy as Academic Privilege**
This AI integration policy is a **privilege** that allows you to use cutting-edge tools while learning. This privilege comes with strict responsibility for honest, ethical usage. Abuse of this policy violates both course standards and SDSU's academic integrity policies.

### **AI Misuse and Plagiarism**
The following constitute academic dishonesty under SDSU policies:

**Plagiarism Through AI**:
- Submitting AI-generated work without proper citation
- Using AI to complete assignments without demonstrating personal understanding
- Claiming AI-assisted work as entirely your own creation
- Copying AI responses without verification or comprehension

**AI Misuse Examples**:
- Having AI solve entire homework problems without independent work
- Using AI to write reports/memos without engaging with the material personally
- Submitting code you cannot explain or debug independently
- Violating phase-specific AI restrictions (e.g., using AI for primary learning in Phases 1-2)

### **SDSU Academic Integrity Alignment**
This course's AI policy operates within SDSU's academic integrity framework:
- **Honesty**: All AI usage must be transparent and documented
- **Trust**: Students are trusted to follow scaffolding guidelines; violations break this trust
- **Fairness**: Equal access to AI tools with equal responsibility for learning outcomes
- **Respect**: Respect for the learning process and instructor expectations
- **Responsibility**: Personal accountability for understanding and competency development

### **Consequences for AI Policy Violations**

**Minor Violations** (First offense, limited scope):
- **Course Level**: Assignment resubmission with 20% grade reduction
- **Learning Intervention**: Mandatory office hours meeting to discuss AI usage strategy
- **Documentation**: Incident recorded in course records

**Major Violations** (Extensive AI misuse, academic dishonesty):
- **Course Level**: Zero credit for assignment, no resubmission opportunity
- **SDSU Policy**: Reported to Office of Student Rights and Responsibilities per SDSU Student Conduct Code
- **Academic Record**: May result in formal academic integrity violation on transcript
- **Course Standing**: May result in course failure depending on violation severity

**Severe/Repeated Violations**:
- **Immediate Consequences**: Course failure regardless of other performance
- **University Action**: Full academic integrity investigation per SDSU procedures
- **AI Privilege Revocation**: Complete prohibition from AI tool usage for remainder of course
- **Academic Record**: Formal notation per university policies

### **Reporting and Investigation Process**
1. **Detection**: AI misuse identified through code analysis, plagiarism detection, or student inability to explain submitted work
2. **Documentation**: Detailed record of violation with evidence
3. **Student Conference**: Mandatory meeting to discuss violation and consequences
4. **SDSU Reporting**: Violations reported to appropriate university offices per SDSU Student Conduct Code Section 41301
5. **Appeal Process**: Students may appeal through standard SDSU academic grievance procedures

### **Why We Take This Seriously**
- **Professional Preparation**: Research careers require absolute integrity in computational work
- **Learning Outcomes**: AI dependency prevents mastery of essential skills
- **Fairness**: Students following the policy deserve recognition for their legitimate effort
- **Institutional Trust**: Graduate programs and employers rely on SDSU's academic standards

## Learning Support and Community

### **Office Hours Philosophy**
Office hours are designed for **collaborative problem-solving**, not answer-giving:
- Bring specific questions about your thinking process
- Come prepared to explain what you've tried
- Expect guided questions that help you discover solutions
- AI strategy consultations are always welcome

### **Peer Learning Integration**
- **Pair Programming**: Share AI strategies with your partner
- **Weekly Check-ins**: Brief class discussions about effective AI usage
- **Success Sharing**: Celebrate interesting discoveries and productive failures
- **Community Guidelines**: Support each other's intellectual risk-taking

---

## Course-Specific AI Tools

### **Recommended Tools**
- **ChatGPT/Claude**: For conceptual explanations and debugging assistance (use in browser/app, not IDE)
- **Perplexity**: For research with automatic source citation
- **NotebookLM**: For creating study guides from course materials
- **Note**: GitHub Copilot and all IDE AI assistants remain disabled throughout the semester per syllabus policy

### **Tool Selection Strategy**
- **Domain-Specific Tools**: Prefer specialized tools for astronomy/physics questions
- **General Tools**: Use for programming syntax and debugging
- **Multiple Sources**: Cross-reference between different AI tools and documentation
- **Human Verification**: Always verify AI responses with instructor/peers/documentation

---

## Bottom Line

**AI is a powerful tool that amplifies human intelligence and capability. It does not replace the need for deep understanding, critical thinking, or foundational knowledge. Students who master both fundamental competencies AND strategic AI usage will thrive. Those who attempt to substitute AI for learning will fail to meet the elevated standards that AI-augmented practice demands.**

**Your goal is not just to complete assignments, but to become a computational astrophysicist capable of tackling novel research problems with both traditional methods and cutting-edge AI tools. This requires mastering both domains.**