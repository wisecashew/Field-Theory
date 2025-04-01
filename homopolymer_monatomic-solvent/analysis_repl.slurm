#!/bin/bash
#
#SBATCH --job-name=analysis_bulk-viz
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --time=00:10:00
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

addr="SIMULATIONS/BULK"
# srun --gres=gpu:1 --ntasks=1 --nodes=1 python rg.py          --state-pkl fields.pkl --dir-path ${addr} --img repl_rg-KP-1 > repl_rg_img.out 2>&1
# echo "Found rg."
srun --gres=gpu:1 --ntasks=1 --nodes=1 python fieldviz.py --state-pkl state.pkl --dir-path ${addr} > density_repl_img.out 2>&1

