## Enforcing the Framework with Claude AI

You're right to be concerned about consistency. Even with a comprehensive framework, Claude (whether Opus or Sonnet) can sometimes drift from specifications. Here's how to ensure compliance:

### 1. **Start Every Session with the Framework**

Begin each new chat or continuation with this exact structure:

```markdown
I'm working on ASTR 596: Modeling the Universe course content. 
You MUST follow the CONTEXT-python-textbook.md framework EXACTLY.

[Paste the entire framework document]

CRITICAL: Any deviation from these requirements requires complete revision. 
These are REQUIREMENTS, not suggestions.

Now, [create/review/enhance] Chapter X on [topic].
```

### 2. **Use Enforcement Checkpoints**

After Claude provides initial content, immediately verify compliance:

```markdown
Before we continue, verify this chapter against the framework:
- Are ALL code examples ≤30 lines focusing on ONE concept?
- Is every technical term defined and bolded?
- Are all required MyST directives used?
- Are all mandatory active learning elements present?

If ANY requirement is missing, revise before proceeding.
```

### 3. **Implement Progressive Review Strategy**

Break chapter creation into stages with verification at each step:

```markdown
Step 1: Create the chapter structure with all required sections (empty).
[Review structure]

Step 2: Add content for sections 1-3 following framework precisely.
[Review for compliance]

Step 3: Add all required active learning elements with proper MyST formatting.
[Verify all elements present]

Step 4: Complete exercises with proper scaffolding.
[Check progression]

Step 5: Add definitions, summaries, and reference tables.
[Final verification]
```

### 4. **Use Specific Correction Commands**

When Claude deviates, be explicit about corrections:

```markdown
STOP. This violates the framework:
1. The code example in section 9.3 is 45 lines - break into 3 examples of ≤15 lines each
2. Terms "namespace" and "traceback" aren't bolded or defined
3. Missing MyST margin notes for definitions
4. No interactive widget in the parameter exploration section

Revise these specific issues before continuing.
```

### 5. **Create a Pre-Flight Checklist**

Before accepting any chapter, run through this verification:

```markdown
Run this checklist:
□ MyST frontmatter present and correct?
□ All Python code in code-cell directives?
□ Every code example ≤30 lines?
□ All technical terms bolded and defined?
□ Minimum 3 "Check Your Understanding" boxes?
□ Minimum 2 "Computational Thinking" boxes?
□ Minimum 2 "Common Bug Alert" sections?
□ Minimum 1 "Debug This!" challenge?
□ Minimum 2 "Why This Matters" real examples?
□ Exercises scaffolded in 3 parts?
□ Comprehensive definitions section?
□ Main takeaways narrative summary?
□ Quick reference tables?

Fix any unchecked items.
```

### 6. **Use Reminder Injections**

Periodically during long sessions, reinject key requirements:

```markdown
REMINDER: You are following CONTEXT-python-textbook.md framework:
- Maximum 30 lines per code example
- One concept per example
- MyST directives for all structured content
- All required active learning elements must be present
Continue with these requirements in mind.
```

### 7. **Leverage Claude's Self-Correction**

Ask Claude to self-audit:

```markdown
Review what you just created against the framework requirements.
List any deviations and fix them.
```

### 8. **Create Section Templates**

Provide explicit templates for complex sections:

```markdown
Use this EXACT template for the next "Check Your Understanding" box:

```{admonition} 🔍 Check Your Understanding
:class: question

[Your question here about the just-taught concept]

```{admonition} Click for Answer
:class: answer, dropdown

[Complete answer with explanation]

```{code-cell} ipython3
# Code demonstrating the answer
```
```
```

### 9. **Handle Framework Drift**

If Claude starts simplifying or ignoring requirements:

```markdown
You're drifting from the framework. Re-read these critical requirements:
1. EVERY code example must be ≤30 lines
2. EVERY technical term needs definition
3. ALL specified active learning elements are MANDATORY
4. MyST directives are REQUIRED, not optional

Revise your last response to comply fully.
```

### 10. **Save Successful Patterns**

When Claude produces excellent compliant content, save it as a template:

```markdown
This section perfectly follows the framework. 
Use this same structure for the next topic:
[Copy the good example]
```

### Why Claude Sometimes Deviates

Understanding why helps you prevent it:

1. **Token limits** - Claude might compress requirements when processing long documents
2. **Training biases** - Claude's training includes many coding tutorials that don't follow your standards
3. **Interpretation flexibility** - Without explicit enforcement, Claude may interpret requirements as suggestions
4. **Context degradation** - In long conversations, earlier requirements might be deprioritized

### The Nuclear Option

If Claude consistently fails to comply:

```markdown
This chapter completely fails to meet requirements.
Start over following the framework EXACTLY.
Do not proceed until you confirm understanding of these MANDATORY requirements:
[List the specific violated requirements]
```

### Creating a Compliance Culture

The key is establishing from the start that the framework is non-negotiable. Use authoritative language:

- "MUST follow" not "should follow"
- "REQUIREMENTS" not "guidelines"  
- "EXACTLY" not "generally"
- "MANDATORY" not "recommended"

### Long-Term Strategy

1. **Build a library of compliant examples** that you can reference
2. **Create correction templates** for common deviations
3. **Document successful prompts** that yielded compliant content
4. **Maintain a "framework violations log"** to identify patterns

### Final Enforcement Tip

The most effective approach is to be absolutely rigid about framework compliance at the start of each session. Once Claude understands that deviations won't be accepted, compliance improves dramatically. Think of it like training - consistent, immediate correction creates better long-term behavior.

Remember: You're not being pedantic; you're ensuring educational quality. These requirements exist because they create optimal learning experiences. Every deviation potentially reduces student understanding or creates inconsistency that impedes learning.

