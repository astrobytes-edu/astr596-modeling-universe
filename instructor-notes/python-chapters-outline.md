# Python Fundamentals Course Outline - ASTR 596

## Overview
This outline presents a comprehensive path for teaching Python to graduate students in astronomy who need to quickly become computationally proficient. The approach builds from Python as an interactive calculator to sophisticated scientific programming, emphasizing practical skills needed for research computing.

## Chapter 2: Python as a Calculator & Basic Data Types

### 2.1 Getting Started with Interactive Python
- Using Python interactively (REPL vs scripts)
- Python as a calculator: basic arithmetic operations (+, -, *, /, //, %, **)
- Order of operations (PEMDAS) and the critical importance of parentheses
- Using parentheses for readability in complex expressions
- Breaking complex equations into intermediate variables for clarity
- Understanding the assignment operator (=) and what it really does
- Variable naming conventions and style (PEP 8 basics)

### 2.2 Numeric Types and Operations
- How Python stores numbers internally (objects vs primitive types)
- Integers (int): arbitrary precision and when that matters
- Floating-point numbers (float): IEEE 754 representation and its implications
- Understanding floating-point precision and roundoff errors
- Why 0.1 + 0.2 != 0.3 and how to handle it (epsilon comparisons)
- Machine epsilon and numerical stability considerations
- Complex numbers: when they appear in physics calculations
- Type promotion and mixing numeric types
- Common gotchas: integer division in Python 2 vs 3, floating-point comparisons
- The math module: sin, cos, log, sqrt, pi, e, and other essentials
- Scientific notation and very large/small numbers
- Tips for numerical computing: avoiding catastrophic cancellation, maintaining precision

### 2.3 Strings and Text Manipulation
- Creating strings with single, double, and triple quotes
- String concatenation and repetition
- String indexing and slicing (introduction to Python's slice notation)
- Common string methods: upper(), lower(), strip(), split(), join()
- String formatting evolution: %, .format(), and why f-strings are superior
- F-strings in depth: expressions, formatting specifications, debugging with =
- Escape characters and raw strings
- Converting between strings and numbers with error handling

### 2.4 Type System and Conversions
- Checking types with type() and isinstance()
- Type conversion functions: int(), float(), str(), bool()
- When conversions fail and how to handle it
- Understanding None and its uses
- Boolean type and truthiness in Python
- The importance of type awareness in scientific computing

### 2.5 Basic Input and Output
- The print() function and its parameters
- Getting user input with input()
- Formatting output for readability
- Simple file reading and writing (preview for later chapter)

## Chapter 3: Control Flow & Logic

### 3.1 Algorithm Design and Pseudocode
- Programming as teaching the computer to make decisions
- The importance of planning before coding
- Introduction to pseudocode and why it matters
- Common pseudocode conventions and structures
- Breaking problems into smaller steps
- Flowcharts for visual thinkers
- Examples: designing a magnitude-to-flux converter step by step

### 3.2 Boolean Logic and Comparisons
- Comparison operators: ==, !=, <, >, <=, >=
- Logical operators: and, or, not
- Bitwise operators: &, |, ^, ~, <<, >> and their uses in scientific computing
- Understanding the difference between logical and bitwise operations
- Chained comparisons (a < b < c)
- Truth tables and evaluating complex conditions
- Common pitfalls: = vs ==, operator precedence
- Short-circuit evaluation and its implications

### 3.3 Conditional Statements
- Planning decision trees before coding
- The if statement and code blocks (indentation matters!)
- elif chains for multiple conditions
- else clauses and default behavior
- Nested conditionals and when to avoid them
- Conditional expressions (ternary operator): x if condition else y
- Common patterns: guard clauses, early returns
- Pseudocode to Python: translating decision logic

### 3.4 The for Loop
- Designing loop logic with pseudocode first
- Iterating over ranges with range()
- Understanding range(start, stop, step)
- Iterating over sequences (strings, lists)
- The enumerate() function for getting indices
- The zip() function for parallel iteration
- Loop else clause (rarely used but good to know)
- When to use for loops vs other approaches

### 3.5 The while Loop
- Planning termination conditions before coding
- While loop syntax and use cases
- Infinite loops and how to avoid them
- Break and continue statements
- While loops with else clauses
- Common patterns: event loops, convergence checking
- When to choose while over for
- Debugging loops with print statements and counters

### 3.6 List Comprehensions and Generator Expressions
- Thinking declaratively vs imperatively
- Basic list comprehension syntax
- Adding conditions to list comprehensions
- Nested list comprehensions (and when they become unreadable)
- Dictionary and set comprehensions
- Generator expressions and memory efficiency
- When to use comprehensions vs explicit loops
- Performance considerations

### 3.7 Operator Precedence Reference
- Complete operator precedence table for Python
- Parentheses for clarity and debugging
- Common precedence mistakes and how to avoid them
- Bitwise vs logical operator precedence
- Best practices: when in doubt, use parentheses

## Chapter 4: Data Structures

### 4.1 Lists In-Depth
- Creating lists: literals, list(), range conversion
- List indexing and slicing (negative indices, stride)
- Modifying lists: append(), extend(), insert(), remove(), pop()
- List arithmetic: concatenation (+) and repetition (*)
- Sorting: sort() vs sorted(), key functions, reverse
- Searching: in operator, index(), count()
- List copying: shallow vs deep copies
- Lists as stacks and queues
- Performance characteristics of list operations

### 4.2 Tuples: Immutable Sequences
- Creating tuples and the singleton tuple syntax
- When and why to use tuples over lists
- Tuple unpacking and multiple assignment
- Named tuples for self-documenting code
- Tuples as dictionary keys
- Converting between lists and tuples

### 4.3 Dictionaries: Key-Value Mapping
- Creating dictionaries: literals, dict(), comprehensions
- Accessing, adding, and modifying values
- Dictionary methods: keys(), values(), items(), get(), pop()
- Iterating over dictionaries
- Dictionary merge and update (Python 3.9+ operators)
- Using dictionaries for counting and grouping
- defaultdict and Counter from collections
- Performance characteristics and hash tables

### 4.4 Sets: Unique Collections
- Creating sets and the empty set gotcha
- Set operations: union, intersection, difference, symmetric difference
- Set methods vs operators
- Modifying sets: add(), remove(), discard()
- Using sets for membership testing
- frozenset for immutable sets
- Common patterns: removing duplicates, finding common elements

### 4.5 Choosing the Right Data Structure
- Decision tree for data structure selection
- Memory and performance comparisons
- Nested data structures for complex data
- Converting between different structures
- Real-world examples from astronomical data processing

## Chapter 5: Functions & Modules

### 5.1 Defining Functions
- Function syntax and the def statement
- Parameters vs arguments
- Return values and the return statement
- Returning multiple values with tuples
- Function documentation with docstrings
- Type hints and their benefits
- Side effects vs pure functions

### 5.2 Function Arguments In-Depth
- Positional and keyword arguments
- Default parameter values and mutable default gotcha
- Variable-length arguments: *args and **kwargs
- Keyword-only arguments (after *)
- Argument unpacking with * and **
- Best practices for function signatures

### 5.3 Scope and Namespaces
- Local vs global scope
- The LEGB rule (Local, Enclosing, Global, Built-in)
- The global and nonlocal keywords
- Closures and nested functions
- When scope issues cause bugs
- Best practices for managing scope

### 5.4 Functional Programming Elements
- Lambda functions and their uses
- Map, filter, and reduce
- Functions as first-class objects
- Higher-order functions
- Partial functions with functools
- When functional style helps and when it hurts

### 5.5 Modules and Packages
- Importing modules: import vs from...import
- Import aliasing with as
- The import system and sys.path
- Creating your own modules
- Package structure and __init__.py
- Relative vs absolute imports
- Common standard library modules
- Installing third-party packages

## Chapter 6: NumPy & Scientific Computing

### 6.1 Why NumPy: Beyond "It's Faster"
- Memory layout: Python lists vs NumPy arrays
- Vectorization and CPU efficiency
- Broadcasting: the mental model that makes it click
- When to switch from pure Python to NumPy
- NumPy's dtype system and memory usage

### 6.2 Array Creation and Manipulation
- Creating arrays: array(), zeros(), ones(), arange(), linspace()
- Array properties: shape, dtype, size, ndim
- Reshaping and resizing arrays
- Stacking and splitting arrays
- Array indexing and slicing
- Advanced indexing: boolean and fancy indexing

### 6.3 Universal Functions and Vectorization
- Arithmetic operations on arrays
- Mathematical functions: sin, cos, exp, log
- Aggregation functions: sum, mean, std, min, max
- Axis parameter and reduction operations
- Writing vectorized code vs loops
- Performance comparison examples

### 6.4 Broadcasting Rules
- Understanding broadcasting with visual examples
- The broadcasting algorithm
- Common broadcasting patterns
- Debugging broadcasting errors
- When broadcasting helps and when it confuses

### 6.5 Random Numbers and Scientific Applications
- Random number generation: old vs new API
- Setting seeds for reproducibility
- Common distributions and their uses
- Monte Carlo simulations basics
- Random sampling and shuffling
- Statistical functions in NumPy

### 6.6 Memory Views and Copies
- Views vs copies in NumPy
- When operations return views
- The copy() method and when to use it
- Memory efficiency considerations
- Common bugs from unexpected views

## Chapter 7: Files, Errors & Debugging

### 7.1 File Input/Output
- Opening and closing files properly
- Context managers and the with statement
- Reading files: read(), readline(), readlines()
- Writing files: write() and writelines()
- Text vs binary modes
- CSV files with the csv module
- JSON for structured data

### 7.2 Path Handling with pathlib
- The Path object and why it's better than strings
- Creating paths in a platform-independent way
- Checking if files/directories exist
- Creating and removing directories
- Iterating over directory contents
- Finding files with glob patterns

### 7.3 Exception Handling
- Common exception types and what they mean
- try/except blocks and catching specific exceptions
- The else and finally clauses
- Raising exceptions with raise
- Creating custom exception classes
- When to catch exceptions vs let them propagate
- Best practices for error messages

### 7.4 Debugging Strategies
- Reading error messages and tracebacks
- Using print() effectively for debugging
- The Python debugger (pdb) basics
- Common debugging patterns
- Logging vs print statements
- Testing and assertions
- Debugging in Jupyter vs scripts

### 7.5 Working with Scientific Data Formats
- Text files with NumPy: loadtxt() and savetxt()
- Binary files with NumPy
- FITS files with astropy (brief introduction)
- HDF5 for large datasets (brief introduction)
- Pickle for Python objects (and its limitations)

## Chapter 8: Object-Oriented Programming Essentials

### 8.1 Classes and Objects
- Understanding objects and classes conceptually
- Defining classes with the class statement
- The __init__ method and initialization
- Instance attributes vs class attributes
- Methods and the self parameter
- Creating and using instances

### 8.2 Methods and Properties
- Instance methods, class methods, and static methods
- The @property decorator for computed attributes
- Setters and getters Python-style
- Private attributes and name mangling
- Method chaining and fluent interfaces

### 8.3 Inheritance and Composition
- Basic inheritance and the super() function
- Method overriding
- Multiple inheritance and method resolution order
- When to use inheritance vs composition
- Abstract base classes (brief introduction)

### 8.4 Special Methods
- String representation: __str__ vs __repr__
- Comparison methods: __eq__, __lt__, etc.
- Container emulation: __len__, __getitem__, __contains__
- Numeric emulation: __add__, __mul__, etc.
- Context managers: __enter__ and __exit__

### 8.5 When to Use OOP
- OOP vs functional vs procedural approaches
- Design principles: encapsulation, abstraction
- Common patterns in scientific computing
- Classes for data organization
- Classes for simulation and modeling
- Avoiding over-engineering

## Pedagogical Notes

### Learning Path
The chapters are designed to build on each other, with each new concept reinforcing previous ones. Students should be able to write simple but useful programs after Chapter 3, work with real data after Chapter 4, organize code professionally after Chapter 5, and handle scientific computing after Chapter 6.

### Algorithm Design Thread Throughout Chapters
Starting in Chapter 3, we introduce pseudocode and planning. This thread continues throughout:
- Chapter 4: Planning data structure choices before implementing
- Chapter 5: Designing function interfaces and module structure before coding
- Chapter 6: Planning array operations and avoiding unnecessary copies
- Chapter 7: Designing error handling strategies upfront
- Chapter 8: Planning class hierarchies and relationships

Each major programming task should follow the pattern:
1. Understand the problem
2. Write pseudocode or draw diagrams
3. Identify data structures and algorithms needed
4. Implement in Python
5. Test and debug
6. Optimize if necessary

### Exercise Philosophy
Each chapter should include exercises that:
1. Start with simple concept reinforcement
2. Build to combining multiple concepts
3. End with a practical mini-project
4. Include common bugs to debug
5. Connect to astronomical applications where natural
6. Require pseudocode or planning documents for complex problems

### Assessment Milestones
- After Chapter 3: Students can write basic data analysis scripts with proper planning
- After Chapter 5: Students can organize code into reusable modules with clear design
- After Chapter 6: Students can process scientific data efficiently with planned algorithms
- After Chapter 8: Students can design and implement complete solutions with documentation

### Time Allocation (Suggested)
- Chapters 2-3: Week 1 (Foundation)
- Chapters 4-5: Week 2 (Data and Organization)
- Chapter 6: Week 3 (Scientific Computing)
- Chapters 7-8: Week 4 (Professional Development)

This allows Project 1 to begin after basic competency (end of Week 1) while more advanced topics are covered in parallel.