# Terminology Consistency Working Document

## Python Fundamentals (Chapters 1-6)

| Term | Standardized Definition | First Introduction | Notes |
|------|------------------------|-------------------|-------|
| **Python** | A high-level programming language designed for readability and simplicity, widely used in scientific computing. | Ch01 | Distinguish from IPython and Jupyter in same section |
| **IPython** | An enhanced interactive Python shell that provides features like tab completion, magic commands, and better debugging. | Ch01 | Always clarify it's a way to use Python, not a separate language |
| **Jupyter** | A web-based notebook interface that allows mixing code, text, and visualizations in a single document. | Ch01 | Explain it uses IPython as its kernel |
| **Environment** | A self-contained Python installation with its own packages and dependencies, isolated from other Python installations. | Ch01 | Specifically "conda environment" when referring to conda |
| **Module** | A single Python file (.py) containing Python definitions and statements that can be imported and reused. | Ch01 | Example: `math` is a module, `math.py` is the file |
| **Package** | A directory containing multiple Python modules and an `__init__.py` file, organizing related code hierarchically. | Ch01 | Example: `numpy` is a package containing many modules |
| **Import** | The process of loading code from a module or package to use in your current program. | Ch01 | Distinguish the verb (import) from the noun (module) |
| **Script** | A Python file intended to be run directly from the command line, containing a sequence of commands. | Ch01 | Contrast with modules (imported) and notebooks (interactive) |
| **Variable** | A name that refers to a value stored in memory. | Ch02 | Emphasize variables don't contain values, they refer to them |
| **Float/Floating-point** | A number with a decimal point, represented in computer memory using scientific notation with limited precision. | Ch02 | Use "floating-point number" formally, "float" as shorthand |
| **Integer** | A whole number without a decimal point, represented exactly in computer memory. | Ch02 | Contrast with float's precision limitations |
| **String** | A sequence of characters (text) enclosed in quotes. | Ch02 | Immutable sequence type |
| **Type** | The category of data that determines what operations can be performed on it (int, float, str, etc.). | Ch02 | Use `type()` function to check |
| **Expression** | A combination of values, variables, and operators that Python evaluates to produce a result. | Ch02 | Example: `2 + 3 * x` |
| **Statement** | A complete instruction that Python can execute. | Ch02 | Example: `x = 5` or `print(x)` |
| **Iteration** | The process of repeatedly executing code for each item in a sequence. | Ch03 | The action performed by loops |
| **Iterator** | An object that can be traversed through its elements one at a time. | Ch03 | What a `for` loop uses behind the scenes |
| **Condition** | An expression that evaluates to True or False, used to control program flow. | Ch03 | Used in if statements and while loops |
| **Loop** | A control structure that repeats a block of code multiple times. | Ch03 | Two types: `for` (definite) and `while` (indefinite) |
| **List** | Python's built-in mutable sequence type that can hold items of different types. | Ch04 | Ordered, changeable, allows duplicates |
| **Tuple** | An immutable sequence type that cannot be changed after creation. | Ch04 | Use for data that shouldn't change |
| **Dictionary** | A mutable mapping type that stores key-value pairs with O(1) average lookup time. | Ch04 | Unordered (before 3.7) or insertion-ordered (3.7+) |
| **Set** | A mutable collection of unique, unordered elements with O(1) membership testing. | Ch04 | Use for removing duplicates or fast membership tests |
| **Mutable** | Objects whose state can be modified after creation (lists, dictionaries, sets). | Ch04 | Can be changed in-place |
| **Immutable** | Objects whose state cannot be modified after creation (tuples, strings, numbers). | Ch04 | Must create new object to "change" |
| **Index** | The numerical position of an element in a sequence, starting from 0. | Ch04 | Negative indices count from the end |
| **Slice/Slicing** | Extracting a portion of a sequence using `[start:stop:step]` notation. | Ch04 | Creates a new object (usually) |
| **Function** | A reusable block of code that performs a specific task, taking inputs and returning outputs. | Ch05 | Defined with `def` keyword |
| **Parameter** | A variable in a function definition that receives a value when the function is called. | Ch05 | In `def f(x):`, `x` is a parameter |
| **Argument** | The actual value passed to a function when calling it. | Ch05 | In `f(5)`, `5` is an argument |
| **Return Value** | The result that a function sends back to the code that called it. | Ch05 | Using `return` statement |
| **Scope** | The region of a program where a variable is accessible, following LEGB rule. | Ch05 | Local, Enclosing, Global, Built-in |
| **Namespace** | A container that holds a set of identifiers and their associated objects, preventing naming conflicts. | Ch05 | Each module has its own namespace |
| **Class** | A blueprint for creating objects that bundles data (attributes) and behavior (methods). | Ch06 | Defined with `class` keyword |
| **Object** | A specific instance created from a class, containing its own data. | Ch06 | Everything in Python is an object |
| **Instance** | A specific object created from a class (synonym for object, emphasizing its origin). | Ch06 | Use when emphasizing "instance of a class" |
| **Method** | A function defined within a class that operates on instances of that class. | Ch06 | Always has `self` as first parameter |
| **Attribute** | A variable bound to an object or class, accessed using dot notation. | Ch06 | Example: `obj.attribute` |
| **Constructor** | The special method `__init__` that initializes new instances of a class. | Ch06 | Called automatically when creating objects |
| **Inheritance** | A mechanism where a class derives properties and methods from another class. | Ch06 | "is-a" relationship |
| **self** | The first parameter of instance methods, referring to the instance being operated on. | Ch06 | Python's way of passing the object to its methods |

## Scientific Computing Core (Chapters 7-9)

| Term | Standardized Definition | First Introduction | Notes |
|------|------------------------|-------------------|-------|
| **NumPy** | A library providing efficient arrays and mathematical functions for scientific computing in Python. | Ch07 | Always capitalize as "NumPy" not "numpy" |
| **Array** | NumPy's homogeneous, fixed-type data structure optimized for numerical computation. | Ch07 | Contrast with Python lists |
| **ndarray** | The formal name for NumPy's N-dimensional array type. | Ch07 | Usually just called "array" |
| **Vectorization** | Operating on entire arrays at once using compiled operations rather than element-by-element Python loops. | Ch07 | Key to NumPy's speed |
| **Broadcasting** | NumPy's mechanism for performing operations on arrays of different shapes by automatically expanding dimensions. | Ch07 | Follows specific rules |
| **dtype** | The data type of elements in a NumPy array (e.g., float64, int32). | Ch07 | Determines memory usage and precision |
| **Shape** | The dimensions of an array as a tuple (e.g., (3, 4) for a 3×4 matrix). | Ch07 | Access with `array.shape` |
| **Axis** | A particular dimension of a multidimensional array. | Ch07 | axis=0 is rows, axis=1 is columns for 2D |
| **View** | An array that shares memory with another array, created by basic slicing. | Ch07 | Changes to view affect original |
| **Copy** | An independent array with its own memory, created by fancy indexing or `.copy()`. | Ch07 | Changes don't affect original |
| **In-place Operation** | An operation that modifies data directly without creating a copy. | Ch07 | Example: `arr += 1` modifies arr directly |
| **Fancy Indexing** | Using arrays or lists as indices to select multiple non-contiguous elements. | Ch07 | Always creates a copy |
| **Matplotlib** | The primary plotting library for creating static, animated, and interactive visualizations in Python. | Ch08 | Import as `import matplotlib.pyplot as plt` |
| **Figure** | The overall container for all plot elements in Matplotlib. | Ch08 | Can contain multiple subplots |
| **Axes** | A single plot area within a figure, containing the actual data visualization. | Ch08 | Not plural of axis! |
| **Subplot** | One of multiple axes arranged in a grid within a single figure. | Ch08 | Created with `plt.subplot()` or `fig.add_subplot()` |
| **Exception** | An event during execution that disrupts normal program flow. | Ch09 | Base class for all errors |
| **Error** | A type of exception indicating a problem in the code. | Ch09 | Subclass of Exception |
| **Try/Except** | Control structure for handling exceptions gracefully without crashing. | Ch09 | Basic error handling pattern |
| **Traceback** | The detailed report Python provides when an exception occurs, showing the call stack. | Ch09 | Read from bottom up |
| **Debugging** | The process of finding and fixing errors in code. | Ch09 | Systematic process, not random changes |
| **Assertion** | A statement that checks if a condition is true, raising AssertionError if not. | Ch09 | Use for documenting assumptions |
| **Test/Testing** | Code written to verify that other code works correctly. | Ch09 | Prevents bugs when changing code |
| **Logging** | Recording program events to track execution and diagnose problems. | Ch09 | Better than print for real programs |

## Advanced Topics (Chapter 10+)

| Term | Standardized Definition | First Introduction | Notes |
|------|------------------------|-------------------|-------|
| **SciPy** | A library built on NumPy providing algorithms for optimization, integration, interpolation, and more. | Ch10+ | Extends NumPy for specialized tasks |
| **Optimization** | Finding the minimum or maximum of a function. | Ch10+ | Core to many scientific problems |
| **Integration** | Numerical computation of definite integrals. | Ch10+ | Various algorithms for different cases |
| **Interpolation** | Estimating values between known data points. | Ch10+ | Linear, spline, etc. |
| **JAX** | A library providing automatic differentiation and JIT compilation for NumPy-like code. | Ch10+ | Requires functional programming style |
| **JIT** | Just-In-Time compilation that converts Python functions to optimized machine code. | Ch10+ | Makes Python code much faster |
| **Automatic Differentiation** | Computing exact derivatives of functions defined in code. | Ch10+ | Not symbolic or numerical differentiation |
| **Functional Programming** | A programming paradigm using pure functions without side effects. | Ch10+ | Required for JAX |
| **Pure Function** | A function that always returns the same output for the same input and has no side effects. | Ch10+ | No global variables or mutations |

---

## Usage Guidelines

### How to Introduce Terms

1. **First Use in Text:**
   - Bold the term: **module**
   - Provide immediate definition in the same sentence or next sentence
   - Add margin definition using MyST syntax if term will be used frequently
   - Example: "A **module** is a single Python file containing code that can be imported and reused."

2. **Formal Definition Box:**
   - Use MyST admonition for critical terms that need emphasis
   ```markdown
   ```{admonition} Definition: Module
   :class: definition
   A module is a single Python file (.py) containing Python definitions and statements that can be imported and reused in other programs.
   ```
   ```

3. **Glossary Entry:**
   - Include in chapter-end definitions section
   - Alphabetical order
   - Cross-reference related terms

### When to Reinforce Definitions

- **After Complex Examples:** When showing advanced usage, remind readers of the basic definition
- **Start of New Chapters:** Brief reminder if term is central to new content
- **In Debugging Sections:** When misunderstanding the term might cause errors
- **In Exercises:** Include terminology review in "Check Your Understanding" boxes

### Handling Synonyms and Related Terms

1. **Parameters vs Arguments:**
   - Define both clearly when introducing functions
   - Use correctly and consistently throughout
   - Add note: "Remember: parameters in definition, arguments in call"

2. **Float vs Floating-point:**
   - Use "floating-point number" in formal definitions
   - "float" is acceptable as shorthand after introduction
   - Be consistent within each chapter

3. **Object vs Instance:**
   - Use "instance" when emphasizing "instance of a specific class"
   - Use "object" for general discussion
   - Note their synonymous nature when introducing OOP

4. **Method vs Function:**
   - Never use "method" before Chapter 6
   - Always use "function" for standalone code blocks
   - After Chapter 6, be precise: "method" only for functions in classes

### Cross-References

- When a term depends on understanding another term, explicitly reference it
- Example: "A **method** is a function (defined in Chapter 5) that belongs to a class."
- Use MyST cross-references: `{ref}`function-definition`` to link to previous definitions

### Consistency Checks

Before finalizing each chapter:
1. Search for all technical terms
2. Verify they match standardized definitions
3. Ensure terms are defined before first use
4. Check that examples use terminology consistently
5. Confirm margin definitions match glossary entries

### Special Considerations

- **For Astronomy Context:** When possible, use astronomy-related examples
- **For No-Programming Background:** Avoid programming jargon in definitions; use plain language
- **For Career-Long Reference:** Include both conceptual understanding and practical usage