#!/bin/bash
#
#SBATCH --job-name=homopolymer_repulsive
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --time=01:10:00
#SBATCH --mail-type=all
#SBATCH --array=0-8
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
my_sl_array[1]="0.05"
my_sl_array[2]="0.1"
my_sl_array[3]="0.5"
my_sl_array[4]="1"
my_sl_array[5]="5"
my_sl_array[6]="10"
my_sl_array[7]="50"
my_sl_array[8]="100"

temp=${my_sl_array[${SLURM_ARRAY_TASK_ID}]}
cd SIMULATIONS/SINGLE_CHAIN/REPULSIVE_KP-1/TEMPERATURE_${temp}
srun --gres=gpu:1 --ntasks=1 --nodes=1 python ../../../../main.py --input-file inp.field > field.out 2>&1

