# ASTR 596 Textbook Revision: Complete Claude AI Workflow & Prompts

## Overview
This document provides the exact workflow and prompts for systematically revising your Python textbook using Claude AI. Follow these phases in order, completing each phase fully before moving to the next. Each prompt is designed to generate specific, actionable outputs that build toward your completed textbook.

**Important Setup Note:** Keep all files (Python chapters AND Getting Started module) attached to your Claude Project throughout the revision process. While you'll focus on Python chapters first, the Getting Started files provide essential context about what students know about their development environment.

---

## Phase 1: Foundation Documents Setup
*Complete these before revising any chapters*

### 1.1 Comprehensive Terminology Analysis (NEW - Do This First!)

**Start new conversation with all files attached and prompt:**
```
Analyze all uploaded Python chapters (files matching pattern XX-python*.md, from 01-python-environment through 10-scipy) to create a comprehensive Terminology Consistency Document.

For EACH technical term found:
1. Identify the chapter where it first appears
2. Extract how it's currently defined (quote the exact text if a definition exists)
3. Note ALL other chapters where this term appears
4. Identify any inconsistent usage or definitions across chapters
5. Flag terms that are used but never properly defined
6. Recommend a single, precise definition suitable for astronomy graduate students with no programming background

Organize the output as a markdown table with these columns:
- Term
- Current Definition (if any)
- First Appearance
- All Appearances 
- Inconsistencies Found
- Recommended Definition
- Priority (High/Medium/Low for fixing)

Highlight the following critical issues:
- Terms that change meaning between chapters
- Terms used before being defined
- Informal usage that should be formalized
- Multiple terms for the same concept

End with a "Critical Issues Summary" listing the top 10 terminology problems that need immediate attention during revision.
```

**Save output as:** `terminology-consistency-initial-analysis.md`

### 1.2 Create Working Terminology Document

**Follow-up prompt:**
```
Based on the analysis above, create a clean Terminology Consistency Document that I'll maintain throughout revision. 

Format as a markdown table with:
- Term
- Standardized Definition (the recommended definition we'll use)
- First Introduction (chapter where it should be formally defined)
- Notes (any special considerations)

Group terms by module:
1. Python Fundamentals (Chapters 1-6)
2. Scientific Computing Core (Chapters 7-9)  
3. Advanced Topics (Chapter 10+)

This will be my living document to update as I revise each chapter.
```

**Save output as:** `terminology-consistency-working.md`

### 1.3 Cross-Reference and Dependency Analysis

**Prompt:**
```
Analyze all Python chapters to identify concept dependencies and forward references.

Create a report showing:
1. FORWARD REFERENCES: Cases where a chapter mentions concepts not yet introduced
   - List each instance with chapter, context, and the future chapter where it's explained
   - Severity rating (blocking understanding vs. minor mention)

2. DEPENDENCY CHAINS: Concepts that build on each other
   - Map which concepts require understanding of previous concepts
   - Identify any circular dependencies or unclear progressions

3. ASSUMED KNOWLEDGE: Things chapters assume without verification
   - Environmental setup assumptions
   - Mathematical background assumptions  
   - Programming concepts assumed but not taught

4. REDUNDANT EXPLANATIONS: Concepts explained multiple times
   - Identify where the same concept is re-explained unnecessarily
   - Distinguish helpful reinforcement from redundant repetition

Format as a structured report with clear sections and actionable fixes for each issue.
```

**Save output as:** `concept-dependency-analysis.md`

### 1.4 Create Exercise Opportunities Tracker

**Prompt:**
```
Based on all uploaded Python chapters, create an Exercise Opportunities Tracker.

For each chapter, identify:
1. Key concepts that could connect to variable star light curve analysis
2. Current exercises (if any) and their focus
3. Opportunities for variable star exercises at three levels:
   - Quick practice (5-10 lines)
   - Synthesis (15-30 lines) 
   - Challenge (research-level)
4. Data requirements (what kind of light curve data would work)
5. Dependencies on previous chapters' exercises

Note: We'll develop the actual exercises AFTER all chapters are revised, but this tracker helps us see opportunities during revision.

Format as markdown with clear sections for each chapter.
```

**Save output as:** `exercise-opportunities.md`

### 1.5 Create Revision Decision Log Template

**Prompt:**
```
Create a Revision Decision Log template for tracking changes during textbook revision. Include sections for:
- Chapter number and name
- Date revised
- Major structural changes
- Code example modifications
- Pedagogical elements added/modified
- Terminology standardizations applied
- Exercise opportunities noted
- Outstanding issues/questions
- Time spent on revision
- Notes for Getting Started module updates

Format as a markdown template I can copy for each chapter.
```

**Save output as:** `revision-log-template.md`

---

## Phase 2: Pre-Revision Consistency Checks
*Run these checks before starting chapter revision*

### 2.1 Voice and Tone Analysis

**Prompt:**
```
Analyze the opening and closing paragraphs of all Python chapters (01-python through 10-scipy).

Report on:
1. CONSISTENCY: Does the voice remain consistent across chapters?
2. ENERGY: Where does enthusiasm wane? Which chapters feel mechanical?
3. COMPLEXITY PROGRESSION: Does the language appropriately mature across modules?
4. STUDENT ENCOURAGEMENT: Which chapters lack motivational elements?

Provide specific examples and suggested improvements for maintaining engaging, consistent tone throughout.
```

### 2.2 Code Complexity Audit

**Prompt:**
```
Scan all code examples in Python chapters 01-10.

Create a report showing:
1. Examples exceeding 30 lines (list with line count and location)
2. Examples mixing multiple concepts (identify concepts and suggested splits)
3. Progression of complexity within each chapter (is it smooth or jarring?)
4. Balance of explanation vs. code (rough percentage per chapter)

Flag critical violations that need immediate attention during revision.
```

### 2.3 Active Learning Elements Inventory

**Prompt:**
```
Inventory all active learning elements across Python chapters 01-10.

Create a table showing for each chapter:
- Number of "Check Your Understanding" boxes
- Number of "Computational Thinking" boxes
- Number of "Common Bug Alert" sections
- Number of "Debug This!" challenges
- Number of "Why This Matters" connections
- Distribution (are they well-spaced or clustered?)

Identify chapters significantly below the required minimums (3, 2, 2, 1, 2 respectively).
```

---

## Phase 3: Chapter-by-Chapter Revision
*Work through chapters sequentially*

### 3.1 Python Fundamentals Module (Chapters 1-6)

**For Chapter 1 (CRITICAL - Sets all patterns), prompt:**
```
Review Chapter 1 (01-python-environment-ORIG.md) against the pedagogical framework.

CONTEXT: 
- This is the first Python chapter and sets ALL patterns for the book
- Students have completed Getting Started module (git, CLI, setup)
- Terminology must match our Consistency Document
- This chapter needs extra attention as it establishes conventions

REVISION REQUIREMENTS:
1. Structure: Verify all 10 required sections present and properly formatted
2. Code: Ensure ALL examples ≤30 lines, single concept focus
3. Ratio: Verify 60/40 explanation-to-code balance
4. Progression: Check conceptual → simple → realistic → robust flow

PEDAGOGICAL ELEMENTS (minimum):
- 3 "Check Your Understanding" boxes
- 2 "Computational Thinking" boxes
- 2 "Common Bug Alert" sections
- 1 "Debug This!" challenge
- 2 "Why This Matters" real-world connections

CONSISTENCY CHECKS:
- Update terminology to match our standardized definitions
- Fix any forward references identified in dependency analysis
- Ensure appropriate tone and energy for first chapter
- Note any assumptions about Getting Started content

OUTPUT:
1. Complete revised chapter with all corrections
2. List of terminology updates made
3. Notes for exercise opportunities (variable star data)
4. Any issues requiring attention in later chapters
```

**For Chapters 2-6, use this template prompt:**
```
Review Chapter [X] ([filename]) for framework compliance.

CONTEXT FROM PREVIOUS CHAPTERS:
- [List key concepts already introduced]
- Terminology standardized through Chapter [X-1]
- Building toward object-oriented programming in Chapter 6

REVISION REQUIREMENTS:
[Same structural requirements as Chapter 1]

SPECIFIC FOCUS AREAS:
- Fix issues identified in initial analysis: [list specific issues]
- Ensure builds properly on Chapter [X-1] concepts
- Maintain consistent voice with previous chapters
- Update terminology per Consistency Document

OUTPUT:
1. Complete revised chapter
2. Updated terminology tracking
3. Exercise opportunity notes
4. Updates needed for Revision Decision Log
```

### 3.2 Scientific Computing Core Module (Chapters 7-9)

**Add module transition awareness to prompts:**
```
Review Chapter [7/8/9] ([filename]) for framework compliance.

MODULE TRANSITION CONTEXT:
- This chapter is part of Scientific Computing Core (increased sophistication)
- Students now have solid Python fundamentals
- Can assume OOP understanding from Chapter 6
- Focus shifts to performance and scientific applications

[Include all standard requirements from above]

ADDITIONAL MODULE-SPECIFIC CHECKS:
- Appropriate increase in sophistication from Fundamentals module
- Performance comparisons with pure Python approaches
- Numerical considerations (precision, stability, efficiency)
- Clear motivation for why scientific libraries are needed

OUTPUT:
[Standard outputs plus:]
5. Notes on module transition effectiveness
```

### 3.3 Advanced Module (Chapter 10+)

**For SciPy and beyond:**
```
Review Chapter 10 (10-scipy.md) for framework compliance.

ADVANCED MODULE CONTEXT:
- This begins research-grade content
- Can assume NumPy/Matplotlib proficiency
- Students ready for sophisticated applications
- Connect to actual research methods

[Include all standard requirements]

ADDITIONAL FOCUS:
- Research paper connections
- Performance optimization discussions
- Professional coding patterns
- Preparation for independent research

OUTPUT:
[Standard outputs plus:]
5. Connections to current astronomical research methods
```

---

## Phase 4: Post-Chapter Revision Checks
*Run after each chapter is revised*

### 4.1 Terminology Update Check

**After revising each chapter, prompt:**
```
I've completed revising Chapter [X]. Compare the revised chapter against our Terminology Consistency Document.

CHECK:
1. Are all terms used consistently with our standardized definitions?
2. Are new terms properly introduced with formal definitions?
3. Have I accidentally introduced terms not yet in our document?
4. Do any definitions need refinement based on how they're used?

Update the Terminology Consistency Document with any changes and flag any issues for attention in later chapters.
```

### 4.2 Dependency Verification

**Prompt:**
```
Verify Chapter [X] only uses concepts from Chapters 1-[X-1].

SCAN FOR:
1. Forward references to future chapters
2. Assumed knowledge not yet taught
3. Proper building on previous concepts
4. Clear progression from previous chapter

Report any violations that need fixing.
```

---

## Phase 5: Module Integration Checks
*Run after completing each module*

### 5.1 Module Consistency Verification

**After completing Python Fundamentals (1-6), prompt:**
```
Review Chapters 1-6 as the complete Python Fundamentals module.

COMPREHENSIVE CHECK:
1. FLOW: Do the chapters flow smoothly from environment setup to OOP?
2. VOICE: Is tone and energy consistent throughout?
3. TERMINOLOGY: Are all terms used consistently across all 6 chapters?
4. COMPLEXITY: Does difficulty increase appropriately?
5. COVERAGE: Are students prepared for Scientific Computing module?

SPECIFIC AUDITS:
- Count total active learning elements (should be ~18+ per type across module)
- Verify no forward references to Chapters 7+
- Check that OOP in Chapter 6 adequately prepares for NumPy's style
- Ensure exercise opportunities build progressively

Provide detailed report with any final fixes needed before moving to next module.
```

**Repeat similar check after Scientific Computing Core (7-9) and Advanced (10+)**

### 5.2 Cross-Module Transition Check

**Between modules, prompt:**
```
Review the transition from [Previous Module] to [Next Module].

CHECK:
1. Is the sophistication increase appropriate and acknowledged?
2. Do we need a bridge section explaining the level-up?
3. Are prerequisite concepts from previous module sufficient?
4. Should we add a "celebration" of what students have achieved?

Draft a brief transition section if needed.
```

---

## Phase 6: Final Complete Textbook Checks
*Only after all Python chapters are revised*

### 6.1 Comprehensive Terminology Audit

**Prompt:**
```
Perform final terminology consistency audit across all revised Python chapters (1-10).

VERIFY:
1. Every term in our Terminology Document is used consistently
2. No undefined terms have crept in during revision
3. Definitions remain clear and appropriate throughout
4. Technical language progression matches module sophistication

Generate a final terminology report confirming consistency or listing remaining issues.
```

### 6.2 Complete Code Verification

**Prompt:**
```
Extract and test all code examples from revised Chapters 1-10.

CREATE:
1. A master test script that runs all examples in sequence
2. Verification that each chapter's code only uses previous concepts
3. Check that outputs are reasonable and error-free
4. List of any external dependencies needed

Flag any code that doesn't run or produces unexpected results.
```

### 6.3 Pedagogical Elements Final Count

**Prompt:**
```
Count all active learning elements in the revised Python textbook (Chapters 1-10).

VERIFY each chapter meets minimums:
- 3+ "Check Your Understanding" boxes
- 2+ "Computational Thinking" boxes
- 2+ "Common Bug Alert" sections
- 1+ "Debug This!" challenges
- 2+ "Why This Matters" connections

CREATE:
- Table showing distribution across chapters
- Heat map of where elements cluster or are sparse
- Recommendations for any final additions needed
```

### 6.4 Exercise Thread Opportunities Summary

**Prompt:**
```
Based on all revised Python chapters, create a final Exercise Opportunities Summary for the variable star light curve thread.

COMPILE:
1. All noted opportunities from revision logs
2. Natural connection points for light curve analysis
3. Data complexity progression recommendations
4. Dependencies between potential exercises
5. Estimated student time for each exercise type

This summary will guide Exercise Thread development in Phase 7.
```

---

## Phase 7: Getting Started Module Review
*Only after Python chapters are complete*

### 7.1 Dependency Back-Check

**Prompt:**
```
Review the Getting Started module files against the completed Python chapters.

CHECK:
1. Do Python chapters assume any setup not covered in Getting Started?
2. Are there tools/commands used that aren't introduced?
3. Should we add any preparations for Python concepts?
4. Are installation instructions sufficient for all packages used?

Create a punch list of updates needed for Getting Started module.
```

### 7.2 Quick Getting Started Updates

**For each Getting Started file, prompt:**
```
Review [filename] from Getting Started module.

LIGHT REVISION ONLY:
1. Ensure consistency with terminology from Python chapters
2. Add any missing setup identified in back-check
3. Maintain encouraging tone for absolute beginners
4. Keep focus on getting students ready for Python chapters

Note: This is not a full revision - just ensuring alignment with Python content.
```

---

## Phase 8: Exercise Thread Development
*Only after all content is finalized*

### 8.1 Complete Exercise Thread Design

**Prompt:**
```
Design comprehensive variable star light curve exercise thread for finalized Python chapters 1-10.

CONTEXT:
- All chapter content is now stable
- Use Exercise Opportunities Summary from Phase 6.4
- Progress from simple Cepheid to complex Betelgeuse observations
- Build complete analysis pipeline by Chapter 10

CREATE:
For each chapter, specify:
1. Quick practice exercise (5-10 lines)
2. Synthesis exercise (15-30 lines)
3. Challenge extension (optional)
4. Data file requirements
5. Dependency on previous exercises
6. Connection to real research

Include complete Exercise Thread Guide document.
```

---

## Workflow Management Reminders

### Daily Workflow Pattern
1. Start by reviewing your Revision Decision Log from yesterday
2. Run any pre-revision checks for today's chapter
3. Revise one complete chapter
4. Run post-revision checks
5. Update all tracking documents
6. Note tomorrow's starting point

### Critical Success Practices
- Keep ALL files in your Claude Project (including Getting Started)
- Update Terminology Document after EVERY chapter
- Don't skip consistency checks between modules
- Celebrate completing each module!

### File Management Protocol
- Original files: Keep with `-ORIG` suffix
- During revision: Save as `-REVISED`
- After module completion: Rename to final form
- Never delete originals until module is complete

### Time Investment Estimates
- Phase 1 (Foundation): 4-5 hours
- Phase 2 (Pre-checks): 2-3 hours
- Phase 3 (Per chapter): 2-4 hours × 10 = 20-40 hours
- Phase 4-5 (Checks): 1 hour per chapter
- Phase 6 (Final): 3-4 hours
- Phase 7 (Getting Started): 2-3 hours
- Phase 8 (Exercises): 6-8 hours
- **Total: 50-70 hours**

---

## Remember Your Mission

You're transforming your Python textbook into a pedagogically excellent resource that will serve as both initial learning material and career-long reference for your astronomy graduate students. Every hour invested in systematic revision following this workflow will:

- Save dozens of hours of student confusion
- Build their confidence as computational scientists
- Prepare them for real astronomical research
- Create a consistent, clear learning journey from zero to research-capable

The comprehensive terminology analysis and consistency checks ensure your textbook has the polish and precision of professional educational materials. The variable star thread (added after revision) will transform exercises from isolated practice into genuine research experience.

Your students will thank you for the clarity, consistency, and care you're putting into this revision process.