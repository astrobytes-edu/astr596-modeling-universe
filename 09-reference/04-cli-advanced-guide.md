# Advanced CLI Guide (Optional)

## When You'll Need This

This guide covers advanced topics that aren't required for ASTR 596 but will be invaluable for:
- **Research computing** on HPC clusters (like SDSU's Verne)
- **Remote work** on servers or cloud computing
- **Automation** of complex data processing pipelines
- **Professional development** as a computational scientist

Feel free to reference this as needed - you don't need to master everything at once!

## Text Processing with awk and sed

### awk - Pattern Processing
```bash
# Extract specific columns from data
awk '{print $3}' data.txt              # Print 3rd column
awk '{print $1, $3}' data.txt          # Print columns 1 and 3
awk '{print $NF}' data.txt             # Print last column

# Conditional processing
awk '$3 > 100' data.txt                # Print lines where column 3 > 100
awk '$1 == "STAR"' catalog.txt         # Lines where first column is "STAR"

# Calculate sum/average
awk '{sum+=$2} END {print sum}' data.txt           # Sum column 2
awk '{sum+=$2; n++} END {print sum/n}' data.txt    # Average of column 2

# Field separator
awk -F',' '{print $2}' data.csv        # Use comma as separator (CSV)
```

**Astronomy example**: Processing catalog data
```bash
# Extract RA, Dec, and magnitude from catalog
awk '$5 < 15 {print $2, $3, $5}' star_catalog.txt > bright_stars.txt
```

### sed - Stream Editor
```bash
# Find and replace
sed 's/old/new/' file.txt              # Replace first occurrence per line
sed 's/old/new/g' file.txt             # Replace all occurrences
sed -i 's/old/new/g' file.txt          # Edit file in place

# Delete lines
sed '1d' file.txt                      # Delete first line
sed '$d' file.txt                      # Delete last line
sed '/pattern/d' file.txt              # Delete lines containing pattern

# Insert/append text
sed '1i\Header line' file.txt          # Insert at beginning
sed '$a\Footer line' file.txt          # Append at end
```

**Example**: Cleaning data files
```bash
# Remove comment lines and blank lines from data
sed '/^#/d; /^$/d' raw_data.txt > clean_data.txt
```

## Advanced find with exec

### Executing Commands on Found Files
```bash
# Find and perform action
find . -name "*.pyc" -exec rm {} \;    # Delete all .pyc files
find . -name "*.py" -exec wc -l {} \;  # Count lines in Python files

# Find and copy
find . -name "*.fits" -exec cp {} /backup/ \;

# Find with multiple conditions
find . -name "*.txt" -size +1M -mtime -7   # .txt files > 1MB modified in last week

# Find and rename
find . -name "*.dat" -exec bash -c 'mv "$0" "${0%.dat}.txt"' {} \;
```

**Astronomy example**: Processing observation files
```bash
# Find all FITS files and create thumbnails
find . -name "*.fits" -exec python make_thumbnail.py {} \;

# Archive old observations
find ./observations -name "*.fits" -mtime +365 -exec gzip {} \;
```

## SSH and Remote Computing

### Basic SSH Connection
```bash
# Connect to remote server
ssh username@server.sdsu.edu

# Connect with specific port
ssh -p 2222 username@server.edu

# Connect with verbose output (debugging)
ssh -v username@server.edu
```

### SSH Key Authentication (No More Passwords!)
```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -C "your_email@sdsu.edu"

# Copy public key to server
ssh-copy-id username@server.edu

# Now connect without password
ssh username@server.edu
```

### File Transfer with SCP and rsync
```bash
# Copy file to remote
scp local_file.py username@server:~/remote_dir/

# Copy from remote
scp username@server:~/results.txt ./

# Copy entire directory
scp -r project/ username@server:~/projects/

# rsync (better for large transfers, resumable)
rsync -avz local_dir/ username@server:~/remote_dir/
rsync -avz --progress large_file.fits username@server:~/
```

### SSH Config File
Create `~/.ssh/config`:
```
Host verne
    HostName verne.sdsu.edu
    User your_username
    Port 22

Host compute
    HostName compute.cluster.edu
    User astro_user
    IdentityFile ~/.ssh/id_ed25519
```

Now simply: `ssh verne` or `ssh compute`

## Screen and tmux for Persistent Sessions

### Why Use Screen/tmux?
- Run long simulations that continue after you disconnect
- Multiple terminal windows in one SSH session
- Resume work exactly where you left off

### Screen Basics
```bash
# Start new screen session
screen -S simulation

# Detach from screen (leaves it running)
Ctrl+A then D

# List active screens
screen -ls

# Reattach to screen
screen -r simulation

# Kill a screen session
screen -X -S simulation quit

# Commands within screen
Ctrl+A then C    # Create new window
Ctrl+A then N    # Next window
Ctrl+A then P    # Previous window
Ctrl+A then K    # Kill current window
```

### tmux Basics (More Modern Alternative)
```bash
# Start new tmux session
tmux new -s simulation

# Detach from tmux
Ctrl+B then D

# List sessions
tmux ls

# Reattach to tmux
tmux attach -t simulation

# Commands within tmux
Ctrl+B then C    # Create new window
Ctrl+B then N    # Next window
Ctrl+B then %    # Split vertically
Ctrl+B then "    # Split horizontally
```

**Example**: Running overnight simulation
```bash
ssh verne
screen -S nbody_run
python long_simulation.py --particles=1000000
# Ctrl+A then D to detach
# Log out, go home
# Next day:
ssh verne
screen -r nbody_run  # Simulation still running!
```

## Shell Scripting

### Basic Script Structure
```bash
#!/bin/bash
# This is a comment

# Variables
NAME="simulation"
N_PARTICLES=1000
OUTPUT_DIR="./results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Loop
for i in {1..10}; do
    echo "Running iteration $i"
    python simulate.py --n=$N_PARTICLES --seed=$i > $OUTPUT_DIR/run_$i.txt
done

# Conditional
if [ -f "$OUTPUT_DIR/run_1.txt" ]; then
    echo "First run completed successfully"
else
    echo "Error: First run failed"
fi
```

### Making Scripts Executable
```bash
chmod +x script.sh      # Make executable
./script.sh            # Run script
```

### Useful Script Patterns
```bash
# Process command line arguments
#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

INPUT_FILE=$1
echo "Processing $INPUT_FILE"

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found"
    exit 1
fi

# Array of values
MASSES=(0.5 1.0 2.0 5.0 10.0)
for mass in "${MASSES[@]}"; do
    python stellar_evolution.py --mass=$mass
done

# Read configuration file
source config.sh

# Parallel execution
for file in *.dat; do
    python process.py "$file" &
done
wait  # Wait for all background jobs to finish
```

## Performance Monitoring

### System Resources
```bash
# CPU and memory usage
top                     # Interactive process viewer
htop                    # Better top (if installed)

# Memory information
free -h                 # Memory usage summary
cat /proc/meminfo      # Detailed memory info

# Disk usage
df -h                   # Filesystem usage
du -sh *               # Size of each item in current directory
du -sh * | sort -hr    # Sorted by size

# Network
netstat -tuln          # Open network connections
ss -tuln               # Modern replacement for netstat
```

### Process Information
```bash
# Your processes
ps aux | grep $USER

# CPU usage by process
ps aux --sort=-%cpu | head

# Memory usage by process
ps aux --sort=-%mem | head

# Process tree
pstree -p

# Kill processes
killall python         # Kill all Python processes
kill -9 PID           # Force kill specific process
```

### Monitoring Files and I/O
```bash
# Watch file size grow
watch -n 1 'ls -lh output.dat'

# Monitor open files
lsof | grep python

# I/O statistics
iotop                  # Requires sudo

# File system activity
watch -n 1 'df -h'
```

## Job Management on HPC Clusters

### SLURM Basics (Common on HPC)
```bash
# Submit job
sbatch job_script.sh

# Check job status
squeue -u $USER

# Cancel job
scancel JOB_ID

# Interactive session
srun --pty bash
```

### Example SLURM Script
```bash
#!/bin/bash
#SBATCH --job-name=nbody_sim
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load python/3.11
conda activate astr596

python nbody_simulation.py --n=1000000
```

## Advanced Data Manipulation

### Combining Multiple Files
```bash
# Concatenate files
cat file1.txt file2.txt > combined.txt

# Merge sorted files
sort file1.txt > sorted1.txt
sort file2.txt > sorted2.txt
comm -12 sorted1.txt sorted2.txt  # Common lines

# Join files on common column
join -t',' -1 1 -2 1 file1.csv file2.csv
```

### Data Extraction and Formatting
```bash
# Extract columns from CSV
cut -d',' -f2,4 data.csv

# Transpose rows and columns
awk '{ for (i=1; i<=NF; i++) a[NR,i] = $i } 
     END { for (i=1; i<=NF; i++) 
           { for (j=1; j<=NR; j++) 
             printf "%s ", a[j,i]; 
             print "" }}' file.txt

# Format numbers
printf "%.2f\n" 3.14159
```

## Environment Customization

### Bash Configuration (~/.bashrc or ~/.bash_profile)
```bash
# Custom prompt
export PS1="\u@\h:\w\$ "

# Add to PATH
export PATH="$HOME/bin:$PATH"

# Aliases
alias ll='ls -lah'
alias gs='git status'
alias activate='conda activate astr596'

# Functions
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# History settings
export HISTSIZE=10000
export HISTFILESIZE=20000
export HISTCONTROL=ignoredups
```

### Useful Environment Variables
```bash
export EDITOR=vim                      # Default editor
export PYTHONPATH="$HOME/lib/python"  # Python module path
export OMP_NUM_THREADS=8              # OpenMP threads
```

## Tips and Tricks

### Command Line Efficiency
```bash
# Run previous command with sudo
sudo !!

# Fix typo in previous command
^old^new

# Run command ignoring aliases
\ls

# Time a command
time python script.py

# Run command at specific time
at 2am tomorrow
python long_simulation.py
Ctrl+D

# Repeat command every N seconds
watch -n 2 'ps aux | grep python'
```

### File Compression
```bash
# Compress
gzip file.txt              # Creates file.txt.gz
tar -czf archive.tar.gz directory/   # Compress directory

# Decompress
gunzip file.txt.gz
tar -xzf archive.tar.gz

# View compressed files without extracting
zcat file.txt.gz
zless file.txt.gz
```

## Quick Reference for Research Computing

```bash
# Remote work
ssh user@server            # Connect
scp file user@server:~/    # Copy file
screen -S name            # Persistent session

# Data processing
awk '{print $2}' file     # Extract column
sed 's/old/new/g' file    # Replace text
find . -name "*.py"       # Find files

# Monitoring
top                       # System resources
ps aux | grep python      # Check processes
du -sh *                  # Directory sizes

# Automation
for i in *.dat; do        # Loop over files
    python process.py $i
done
```

## Final Thoughts

These advanced topics will become relevant as you:
- Work with larger datasets
- Use HPC resources
- Collaborate on remote servers
- Automate complex workflows

Don't feel pressured to learn everything at once. Bookmark this guide and return to it when you encounter situations that need these tools. The best way to learn is by solving real problems!

**Remember**: Even experienced users look up syntax. The skill is knowing what tools exist and when to use them.