# Introduction to the Command Line Interface (CLI)

## Quick Reference Card (TL;DR)

### Essential Commands You'll Use Every Day
```bash
# Navigation
pwd                 # Where am I?
ls -la              # What's here? (including hidden files)
cd folder/          # Go into folder
cd ..               # Go up one level
cd ~                # Go home

# Files & Directories
mkdir project       # Make directory
touch file.py       # Create empty file
cp source dest      # Copy
mv old new          # Move/rename
rm file             # Delete (CAREFUL - no undo!)
rm -r folder/       # Delete folder

# Viewing Files
cat file            # Show entire file
head file           # Show first 10 lines
tail file           # Show last 10 lines
less file           # Page through file (q to quit)
grep "text" file    # Search for text

# Python & Course Work
python script.py    # Run Python script
conda activate astr596  # Activate course environment
git status          # Check git status
git add .           # Stage all changes
git commit -m "msg" # Commit with message
git push            # Push to GitHub

# Useful Shortcuts
Tab                 # Autocomplete (USE THIS!)
↑/↓                 # Previous/next command
Ctrl+C              # Stop current command
Ctrl+L              # Clear screen
history             # Show command history
```

## Why Use the Terminal Instead of Clicking Around?

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

# CLI way: One command, 2 seconds
for i in *.dat; do mv "$i" "simulation_${i}"; done
```

## Getting Started: Opening the Terminal

### macOS
- Press `Cmd + Space`, type "Terminal", press Enter
- Or: Applications → Utilities → Terminal

### Linux
- Press `Ctrl + Alt + T`
- Or: Look for "Terminal" in applications

### Windows
- Use "Git Bash" (installed with Git)
- Or: Windows Terminal, or WSL (Windows Subsystem for Linux)
- **Avoid**: Command Prompt (cmd.exe) - it uses different commands

## Essential Concepts

### What is the Terminal?
A text-based interface to your computer. You type commands, computer executes them, shows results.

### File System Structure
Your computer's files are organized in a tree:
```
/                    # Root (Linux/Mac)
├── home/           
│   └── yourname/    # Your home directory (~)
│       ├── Desktop/
│       ├── Documents/
│       └── astr596/
│           ├── project1/
│           └── project2/
```

### Current Working Directory
You're always "somewhere" in the file system. The terminal shows where with the **prompt**:
```bash
yourname@computer:~/astr596/project1$
# This means you're in the project1 folder
```

## Core Navigation Commands

### Where Am I?
```bash
pwd    # Print Working Directory
```
**Example output**: `/home/yourname/astr596/project1`

### What's Here?
```bash
ls                # List files
ls -l             # Long format (shows permissions, size, date)
ls -la            # Include hidden files (start with .)
ls -lh            # Human-readable sizes (KB, MB, GB)
ls *.py           # List only Python files
```

**Example**:
```bash
$ ls -lh
total 28K
-rw-r--r-- 1 user group 2.4K Nov 15 14:23 main.py
-rw-r--r-- 1 user group  15K Nov 15 14:20 nbody.py
drwxr-xr-x 2 user group 4.0K Nov 14 10:15 data/
```

### Moving Around
```bash
cd project1              # Change to project1 directory
cd ..                    # Go up one level
cd ../..                 # Go up two levels
cd ~                     # Go to home directory
cd ~/astr596/project2    # Go to specific path
cd -                     # Go back to previous directory
```

**Pro tip**: Use Tab completion!
```bash
cd ast[TAB]         # Autocompletes to astr596/
cd ~/astr[TAB]/pr[TAB]  # Tab complete works with paths
```

## File and Directory Operations

### Creating Directories
```bash
mkdir project3                    # Make directory
mkdir -p data/raw/2024           # Make nested directories
mkdir results plots analysis     # Make multiple directories
```

**Example**: Organizing a project
```bash
mkdir -p project/{src,data,outputs,docs}
# Creates: project/src, project/data, project/outputs, project/docs
```

### Creating Files
```bash
touch README.md              # Create empty file
touch script.py module.py    # Create multiple files
echo "# Project 1" > README.md   # Create file with content
```

### Copying Files and Directories
```bash
cp file1.py file2.py             # Copy file
cp file1.py backup/              # Copy to directory
cp -r project1/ project1_backup/ # Copy entire directory
cp *.py scripts/                 # Copy all Python files
```

**Example**: Backing up your work
```bash
cp -r project1/ project1_backup_$(date +%Y%m%d)
# Creates: project1_backup_20241115
```

### Moving and Renaming
```bash
mv oldname.py newname.py         # Rename file
mv file.py ../                   # Move up one directory
mv *.dat data/                   # Move all .dat files
mv project1 project1_old         # Rename directory
```

**Example**: Organizing scattered files
```bash
mv *.py src/
mv *.png plots/
mv *.txt docs/
```

### Removing Files and Directories
```bash
rm file.py                       # Remove file
rm -i file.py                    # Ask before removing
rm -r directory/                 # Remove directory and contents
rm -rf directory/                # Force remove (CAREFUL!)
rm *.pyc                         # Remove all .pyc files
```

**⚠️ WARNING**: There's no trash/recycle bin! Deleted = gone forever!

## Viewing and Editing Files

### Quick Views
```bash
cat file.py              # Display entire file
head file.py             # Show first 10 lines
head -n 20 file.py       # Show first 20 lines
tail file.py             # Show last 10 lines
tail -f output.log       # Follow file as it updates (great for logs)
less bigfile.txt         # Page through file (q to quit)
```

### Searching in Files
```bash
grep "import" *.py       # Find "import" in all Python files
grep -n "error" log.txt  # Show line numbers
grep -r "TODO" .         # Search recursively in all files
grep -i "warning" *.log  # Case-insensitive search
```

**Example**: Finding all your TODO comments
```bash
grep -rn "TODO\|FIXME" --include="*.py" .
```

### Word Count and File Info
```bash
wc file.txt              # Lines, words, characters
wc -l *.py              # Count lines in all Python files
file mystery.dat        # Determine file type
du -sh project1/        # Directory size (human-readable)
```

## Working with Python

### Running Scripts
```bash
python script.py                 # Run Python script
python -m module                 # Run module
python -c "print('hello')"      # Run one-line command
python                          # Interactive Python (exit() to quit)
ipython                         # Better interactive Python
```

### Python Virtual Environments
```bash
conda activate astr596           # Activate environment
conda deactivate                # Deactivate
which python                    # Check which Python you're using
pip list                        # List installed packages
```

## Input/Output Redirection

### Output Redirection
```bash
python script.py > output.txt    # Save output to file
python script.py >> output.txt   # Append to file
python script.py 2> errors.txt   # Save errors to file
python script.py &> all.txt      # Save everything to file
```

**Example**: Saving simulation results
```bash
python nbody.py > results.txt 2> errors.log
```

### Pipes (Combining Commands)
```bash
ls -la | grep ".py"              # List files, filter for Python
cat data.txt | sort | uniq       # Sort and remove duplicates
history | grep "git"              # Find git commands in history
ps aux | grep python              # Find running Python processes
```

## Useful Productivity Commands

### Command History
```bash
history                          # Show command history
!123                            # Run command #123 from history
!!                              # Run last command
!py                             # Run last command starting with "py"
```

### Finding Files
```bash
find . -name "*.py"              # Find all Python files
find . -name "*test*"            # Find files with "test" in name
find . -mtime -7                 # Files modified in last 7 days
find . -size +10M                # Files larger than 10MB
```

**Example**: Finding lost work
```bash
find ~ -name "*stellar*" -mtime -3
# Find files with "stellar" modified in last 3 days
```

### Process Management
```bash
Ctrl+C                           # Stop current command
Ctrl+Z                           # Suspend current command
jobs                            # List suspended jobs
fg                              # Resume suspended job
python long_sim.py &            # Run in background
ps                              # Show your processes
kill 12345                      # Stop process with ID 12345
```

## Environment Variables

### Viewing and Setting
```bash
echo $PATH                      # View PATH variable
echo $HOME                      # Your home directory
export DATADIR=/path/to/data   # Set variable
echo $DATADIR                   # Use variable
```

### Using in Commands
```bash
cd $HOME/astr596
cp data.txt $DATADIR/
python script.py --input=$DATADIR/input.txt
```

## Useful Shortcuts and Tips

### Keyboard Shortcuts
```
Tab         # Autocomplete
↑/↓         # Previous/next command
Ctrl+A      # Go to line beginning
Ctrl+E      # Go to line end
Ctrl+L      # Clear screen
Ctrl+R      # Search command history
Ctrl+D      # Exit/logout
```

### Wildcards (Globbing)
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

### Command Aliases
Add to `~/.bashrc` (Linux) or `~/.zshrc` (Mac):
```bash
alias ll='ls -lh'
alias py='python'
alias jup='jupyter lab'
alias gs='git status'
```

## Practical Examples for ASTR 596

### Setting Up a New Project
```bash
# Create project structure
mkdir -p project2/{src,data,outputs,docs}
cd project2
touch README.md requirements.txt
touch src/{main.py,stellar.py,utils.py}
echo "# Project 2: N-Body Simulation" > README.md
```

### Running and Logging Simulations
```bash
# Run simulation with timing
time python nbody_sim.py

# Run with output logging
python nbody_sim.py > output.log 2>&1

# Run multiple parameter sets
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

### Backing Up Your Work
```bash
# Quick backup
cp -r project2/ ~/backups/project2_$(date +%Y%m%d_%H%M%S)

# Compress for submission
tar -czf project2_submission.tar.gz project2/

# Extract compressed file
tar -xzf project2_submission.tar.gz
```

## Practice Exercises

1. **Navigation Challenge**: 
   - Navigate to your home directory
   - Create a folder structure for a project
   - Move between directories using relative and absolute paths

2. **File Management**:
   - Create 10 test files
   - Rename them all at once
   - Organize them into subdirectories

3. **Data Processing**:
   - Generate a file with random numbers
   - Use grep to find specific patterns
   - Count lines, sort, and find unique values

4. **Automation**:
   - Write a command to backup your project
   - Create an alias for a commonly used command
   - Run a Python script with different parameters

## What's Next?

**Congratulations!** You now know all the CLI commands needed for this course. Practice these commands and they'll become second nature within a few weeks.

**Optional**: Curious about remote computing (SSH), long-running jobs (screen/tmux), or shell scripting? Check out our [Advanced CLI Guide](cli-advanced.md) for topics useful in research and HPC work. But don't worry - you won't need these for ASTR 596!

**Remember**:
- **Tab is your friend**: Autocomplete saves typing and prevents errors
- **Up arrow**: Recall previous commands  
- **Be careful with rm**: No undo!
- **Practice makes perfect**: The more you use it, the more natural it becomes

The command line seems intimidating at first, but within a few weeks it'll become second nature. You'll wonder how you ever lived without it!