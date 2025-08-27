---
title: "Chapter 1: Introduction to the Command Line Interface (CLI)"
subtitle: "ASTR 596: Modeling the Universe"
exports:
  - format: pdf
---

## Before You Begin: Safety Rules

:::{important} 🛡️ Terminal Safety First

The terminal is powerful but unforgiving. Follow these rules ALWAYS:

1. **pwd before rm** - Always know where you are before deleting
2. **ls before rm** - Always see what you'll delete
3. **Use Tab completion** - Reduces dangerous typos
4. **Keep backups** - The terminal has NO undo button
5. **Think twice, type once** - Especially with delete commands

**Remember**: One wrong character can delete everything. There is no recycle bin.
:::

## Why Use the Terminal?

### The GUI Limitation

When you use Finder (macOS) or File Explorer (Windows), you're limited to what the designers decided to show you. Want to:

- Rename 1000 files at once? Good luck clicking each one.
- Find all Python files modified in the last week? No easy way.
- Run your code on a supercomputer? There's no GUI there.
- Process data on a remote server? You need the terminal.

### The CLI Superpower

The command line gives you:

- **Automation**: Do repetitive tasks in seconds, not hours
- **Remote access**: Control computers anywhere in the world
- **Power**: Access to thousands of tools not available in GUIs
- **Speed**: Keyboard is faster than mouse for many tasks
- **Reproducibility**: Save and share exact commands you ran
- **Professional necessity**: Every computational scientist uses it

**Real example**: Renaming simulation outputs

```bash
# GUI way: Click each file, rename manually (30 minutes for 100 files)

# CLI way: One command, 2 seconds (we'll learn how later)
for i in *.dat; do mv "$i" "simulation_${i}"; done
```

## Understanding the Terminal (2-minute conceptual foundation)

### What's What?

- **Terminal**: The application you open (like Terminal.app on Mac)
- **Shell**: The program that interprets your commands (bash, zsh, etc.)
- **Command Line**: Where you type commands
- **Prompt**: Shows you're ready for input (usually ends with $ or >)

### How Commands Work

```
command -options arguments
   ↑        ↑        ↑
  what    how to   what to
  to do    do it    do it on
```

*Example:* `ls -la /home` means "list (ls) with long format and all files (-la) in the /home directory"

## Getting Started: Opening the Terminal

### macOS

- Press `Cmd + Space`, type "Terminal", press Enter
- Or: Applications → Utilities → Terminal

### Linux

- Press `Ctrl + Alt + T`
- Or: Look for "Terminal" in applications

### Windows

- **Best option**: Use "Git Bash" (installed with Git)
- Alternative: Windows Terminal or PowerShell
- **Avoid**: Command Prompt (cmd.exe) - uses different commands

```{warning} ⚠️ Windows Users: Important Differences

If using Git Bash on Windows:
- Some commands work differently (we'll note these)
- No `man` command (use `--help` instead)
- Path separators: Use `/` not `\`
- Your home is `/c/Users/YourName` not `C:\Users\YourName`
```

## Your First Three Commands

Master these three before anything else. Seriously, practice just these for 10 minutes.

### 1. Where Am I? (`pwd`)

```bash
pwd    # Print Working Directory
```

**Try it**: Type `pwd` and press Enter

**You should see something like**:

- macOS/Linux: `/Users/yourname` or `/home/yourname`
- Windows Git Bash: `/c/Users/yourname`

### 2. What's Here? (`ls`)

```bash
ls      # List files and folders
```

**Try it**: Type `ls` and press Enter

You'll see files and folders in your current location.

### 3. Move Around (`cd`)

```bash
cd Desktop    # Change Directory to Desktop
cd ..         # Go up one level
cd ~          # Go to your home directory
```

**Try it**: 
1. Type `cd Desktop` (or any folder you see from `ls`)
2. Type `pwd` to confirm you moved
3. Type `cd ..` to go back

:::{tip} ✅ Quick Practice #1

Can you:

- [ ] Find out where you are? (pwd)
- [ ] See what's in your current folder? (ls)
- [ ] Move to a different folder and back? (cd)

If yes, you've mastered the basics! If no, practice for 5 more minutes.
:::

## File System Navigation

### Understanding Paths

Your computer's files are organized in a tree:

```
/                    # Root (top level)
├── home/           
│   └── yourname/    # Your home directory (~)
│       ├── Desktop/
│       ├── Documents/
│       └── astr596/
│           ├── project1/
│           └── project2/
```

**Two types of paths**:

- **Absolute**: Full path from root `/home/yourname/Desktop`
- **Relative**: Path from where you are `Desktop` or `../Documents`

### Navigation Shortcuts

```bash
~     # Your home directory
.     # Current directory  
..    # Parent directory (one up)
-     # Previous directory
```

### Enhanced ls Commands
Now let's add options to `ls`:

```bash
ls -a          # Show ALL files (including hidden)
ls -l          # Long format (details)
ls -la         # Combine: all files, detailed
ls -lh         # Human-readable sizes (KB, MB)
ls *.py        # List only Python files
```

**Example output of `ls -lh`**:

```bash
total 28K
-rw-r--r-- 1 user group 2.4K Nov 15 14:23 main.py
-rw-r--r-- 1 user group  15K Nov 15 14:20 nbody.py
drwxr-xr-x 2 user group 4.0K Nov 14 10:15 data/
```

This shows: permissions, owner, size, date, name

### Pro Navigation with Tab Completion

:::{important} 🎯 THE MOST IMPORTANT TIP

**Tab is your best friend!** Start typing, press Tab to autocomplete:

```bash
cd Desk[TAB]        # Autocompletes to Desktop/
cd ~/astr[TAB]/pr[TAB]  # Autocompletes full path
```

Tab prevents typos and saves typing. Use it CONSTANTLY.
:::

## File and Directory Operations

### Creating Directories

```bash
mkdir project3                   # Make one directory
mkdir -p data/raw/2024          # Make nested directories
mkdir results plots analysis    # Make multiple at once
```

### Creating Files

```bash
touch README.md              # Create empty file
touch script.py module.py    # Create multiple files
echo "# Project 1" > README.md   # Create file with content
```

### Copying Files

```bash
cp file1.py file2.py             # Copy file
cp file1.py backup/              # Copy to directory
cp -r project1/ project1_backup/ # Copy entire directory (-r = recursive)
```

:::{tip} 💡 Always Backup Before Dangerous Operations

Before modifying important files:
```bash
cp important.py important.py.backup
```

Now you can recover if something goes wrong!
:::

### Moving and Renaming

```bash
mv oldname.py newname.py    # Rename file
mv file.py ../              # Move up one directory
mv *.dat data/             # Move all .dat files
```

### ⚠️ Removing Files and Directories

:::{danger} 🔥 EXTREME DANGER ZONE: rm Commands

**THE rm COMMAND IS PERMANENT. NO UNDO. NO RECYCLE BIN.**

**NEVER EVER use these**:
- `rm -rf /` = DELETE ENTIRE COMPUTER
- `rm -rf ~` = DELETE ALL YOUR FILES
- `rm -rf *` = DELETE EVERYTHING IN CURRENT FOLDER

**What -rf means**:
- `-r` = recursive (delete folders and everything inside)
- `-f` = force (no confirmation, even for important files)

**Safe practices**:
1. Use `rm -i` for interactive mode (asks confirmation)
2. Use `ls` first to see what you'll delete
3. Use `pwd` to confirm you're in the right place
4. Start with `rm` (no flags) for single files
:::

Safe deletion examples:

```bash
rm file.py                # Remove single file
rm -i *.tmp              # Remove with confirmation
rm -r empty_folder/      # Remove empty directory
```

## Viewing Files

### Quick Views

```bash
cat file.py          # Show entire file
head file.py         # Show first 10 lines
head -n 20 file.py   # Show first 20 lines  
tail file.py         # Show last 10 lines
tail -f output.log   # Follow file updates (great for logs)
less bigfile.txt     # Page through file (q to quit)
```

### Searching in Files

```bash
grep "import" *.py       # Find "import" in Python files
grep -n "error" log.txt  # Show with line numbers
grep -r "TODO" .         # Search all files recursively
grep -i "warning" log    # Case-insensitive search
```

## Platform Compatibility Reference

:::{note} 🖥️ Command Differences Across Platforms

| Command | macOS/Linux | Git Bash | Windows CMD | Alternative |
|---------|-------------|----------|-------------|------------|
| `ls`    | ✅ Works    | ✅ Works | ❌ | Use `dir` |
| `pwd`   | ✅ Works    | ✅ Works | ❌ | Use `cd` (no args) |
| `cat`   | ✅ Works    | ✅ Works | ❌ | Use `type` |
| `rm`    | ✅ Works    | ✅ Works | ❌ | Use `del` |
| `man`   | ✅ Works    | ❌ | ❌ | Use `--help` |
| `grep`  | ✅ Works    | ✅ Works | ❌ | Use `findstr` |
| `ps aux`| ✅ Works    | ⚠️ Limited | ❌ | Use `tasklist` |

**Tip**: When in doubt, use `command --help` to see options
:::

## Working with Python

### Running Python Scripts

```bash
python script.py                 # Run script
python -m module                 # Run module
python -c "print('hello')"      # Run one line
python                          # Interactive mode (exit() to quit)
ipython                         # Better interactive mode
```

### Managing Your Environment

```bash
conda activate astr596    # Enter course environment
conda deactivate         # Exit environment
which python            # Check which Python is active
conda list              # See installed packages
```

## When Things Go Wrong

:::{tip} 🔧 Understanding Error Messages
:class: dropdown

**"command not found"**

- Typo in command name
- Command not installed  
- Not in PATH
- **Fix**: Check spelling, install missing tool

**"permission denied"**

- Need admin/sudo rights
- File is protected
- Wrong ownership
- **Fix**: Use sudo (carefully!) or check file permissions

**"no such file or directory"**

- Wrong path or filename
- In wrong directory
- File doesn't exist
- **Fix**: Check with `pwd` and `ls`, verify path

**"syntax error near unexpected token"**

- Special character not escaped
- Quote mismatch
- Wrong shell syntax
- **Fix**: Check quotes and special characters

**Process running forever**

- Press `Ctrl+C` to stop
- If frozen, try `Ctrl+Z` then `kill %1`
```

## Input/Output Redirection

### Saving Output

```bash
python script.py > output.txt    # Save output to file
python script.py >> output.txt   # Append to file
python script.py 2> errors.txt   # Save errors only
python script.py &> all.txt      # Save everything
```

### Pipes (Combining Commands)

```bash
ls -la | grep ".py"              # List files, filter Python
cat data.txt | sort | uniq       # Sort and remove duplicates
history | grep "git"             # Find git commands
```

## Useful Shortcuts and Tips

### Keyboard Shortcuts

```
Tab         # AUTOCOMPLETE (use constantly!)
↑/↓         # Previous/next command
Ctrl+C      # Stop current command
Ctrl+L      # Clear screen
Ctrl+A      # Go to line beginning
Ctrl+E      # Go to line end
Ctrl+R      # Search command history
Ctrl+D      # Exit/logout
```

### Wildcards (Pattern Matching)

```bash
*           # Any characters
?           # Single character
[abc]       # Any of a, b, c
[0-9]       # Any digit

# Examples:
ls *.py                 # All Python files
ls data_?.txt          # data_1.txt, data_2.txt, etc.
ls img_[0-9][0-9].png  # img_00.png through img_99.png
```

## Quick Reference Card

### Essential Daily Commands

```bash
# Navigation
pwd                 # Where am I?
ls -la              # What's here? (all files, detailed)
cd folder/          # Enter folder
cd ..               # Go up one level
cd ~                # Go home

# Files & Directories
mkdir project       # Create directory
touch file.py       # Create empty file
cp source dest      # Copy
mv old new          # Move/rename
rm file             # Delete (CAREFUL!)

# Viewing Files
cat file            # Show entire file
head -20 file       # Show first 20 lines
tail -f log         # Follow log file
grep "text" file    # Search in file

# Python & Course
python script.py            # Run Python
conda activate astr596      # Enter environment
git status                  # Check git
git add . && git commit -m "msg" && git push  # Submit work

# Getting Help
command --help      # See command options (not 'man' on Windows)
```

## Practice Exercises

### Exercise 1: Navigation Basics (10 min)

```bash
# 1. Find where you are
pwd

# 2. Go to your home directory
cd ~

# 3. Create a practice folder
mkdir cli_practice
cd cli_practice

# 4. Create some files
touch data1.txt data2.txt script.py

# 5. List what you created
ls -la

# 6. Go back home
cd ..
```

### Exercise 2: File Management (15 min)

1. Create a project structure:

```bash
mkdir -p project/{src,data,docs}
```

2. Create files in each folder
3. Copy a file between folders
4. Rename a file
5. Safely delete a test file

### Exercise 3: Real Task - Organize Files (20 min)

```bash
# Create messy folder
mkdir messy && cd messy
touch file1.py file2.py data1.txt data2.txt image1.png image2.png

# Now organize them
mkdir {code,data,images}
mv *.py code/
mv *.txt data/
mv *.png images/

# Verify organization
ls -la */
```

## Practical Examples for ASTR 596

### Setting Up a New Project

```bash
# Create full project structure
mkdir -p project2/{src,data,outputs,docs}
cd project2
touch README.md requirements.txt
touch src/{main.py,stellar.py,utils.py}
echo "# Project 2: N-Body Simulation" > README.md
```

### Running and Logging Simulations

```bash
# Run with timing
time python nbody_sim.py

# Run with output capture
python nbody_sim.py > output.log 2>&1

# Run multiple parameters
for n in 100 500 1000; do
    python nbody.py --particles=$n > results_n$n.txt
done
```

### Data Processing Pipeline

```bash
# Process all data files
for file in data/*.txt; do
    python process.py "$file" > "processed/$(basename $file)"
done

# Check results
grep "converged" processed/*.txt | wc -l
```

## Emergency Recovery

:::{tip} 🆘 "I'm Lost, Help!"

If you're confused or lost:

1. **Where am I?** → `pwd`
2. **Go home** → `cd ~`
3. **See what's here** → `ls -la`
4. **Stop running process** → `Ctrl+C`
5. **Clear messy screen** → `Ctrl+L` or `clear`
6. **Exit and start over** → `exit` and reopen terminal

Remember: Closing and reopening terminal resets everything!
:::

## Next Steps

**Congratulations!** You now know the essential CLI commands for this course.

**Your learning path**:

1. ✅ Master the first three commands (pwd, ls, cd)
2. ✅ Practice file operations carefully
3. ✅ Use Tab completion religiously
4. → Continue to course projects

**Optional Advanced Topics**: Curious about remote computing (SSH), long-running jobs (screen/tmux), or shell scripting? Those are useful for research but not needed for ASTR 596.

**Remember**:

- **Tab is your friend**: Saves typing and prevents errors
- **pwd before rm**: Always know where you are
- **Be paranoid with rm**: No undo in terminal!
- **Practice daily**: 10 minutes/day for two weeks = mastery

The command line may seem "old school" and scary at first, but it's just typing commands instead of clicking. Within two weeks of daily use, it'll feel natural. You've got this!
