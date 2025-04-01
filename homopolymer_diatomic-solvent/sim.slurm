#!/bin/bash
#
#SBATCH --job-name=diatomic-solvent+polymer_bulk
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --time=01:10:00
#SBATCH --mail-type=all
#SBATCH --array=0-4
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

declare -A my_sl_array
my_sl_array[0]="0.01"
my_sl_array[1]="0.1"
my_sl_array[2]="1"
my_sl_array[3]="10"
my_sl_array[4]="100"

temp=${my_sl_array[${SLURM_ARRAY_TASK_ID}]}
cd SIMULATIONS/BULK/TEMPERATURE_${temp}
srun --gres=gpu:1 --ntasks=1 --nodes=1 python ../../../main.py --input-file bulk.field > field.out 2>&1

