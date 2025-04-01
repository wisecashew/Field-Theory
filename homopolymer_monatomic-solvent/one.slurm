#!/bin/bash
#
#SBATCH --job-name=monatomic-solvent+polymer-ONE-SIMULATION
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --mail-type=all
#SBATCH --gres=gpu:1
#SBATCH --mail-user=satyend@princeton.edu

# The modules I need to run my job
module purge

set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG

# echo an error message before exiting
trap 'echo "\"${last_command}\" command failed with exit code $?." ' EXIT

echo "The numeric id associated with this job is ${SLURM_ARRAY_JOB_ID}."
echo "The numeric task id associated with this job is ${SLURM_ARRAY_TASK_ID}."

module load anaconda3/2024.6
conda activate FTS


cd SIMULATIONS/SINGLE_CHAIN/TEMPERATURE_1
srun --gres=gpu:1 --ntasks=1 --nodes=1 python ../../../main.py --input-file inp.field > field.out 2>&1

