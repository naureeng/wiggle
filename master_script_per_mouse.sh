#!/bin/bash
#SBATCH --mem=4GB               # Set memory limit (adjust as needed)
#SBATCH --ntasks=1              # Number of tasks (adjust as needed)
#SBATCH --cpus-per-task=1       # Number of CPUs per task (adjust as needed)

# Get the subject name from the argument
subject=$1

# Activate conda environment
module load miniconda
conda activate /nfs/nhome/live/naureeng/anaconda3/envs/iblenv2

cd /nfs/nhome/live/naureeng/int-brain-lab/ibl-false-start/wiggle-main

# Run the analysis (replace with your actual command)
echo "Running analysis for subject: $subject"
python3 -W ignore master_script_per_mouse.py --subject "$subject"  
