#!/bin/bash
#
#SBATCH --job-name=diatomic-solvent+polymer
#SBATCH --ntasks=1
#SBATCH --mem=50GB
#SBATCH --time=00:05:00
#SBATCH --mail-type=all
#SBATCH --gres=gpu:1
#SBATCH --mail-user=satyend@princeton.edu

# The modules I need to run my job
module purge
# module load rh/devtoolset/7 

set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG

# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?." ' EXIT

module load anaconda3/2024.6
conda activate FTS
addr="SIMULATIONS/PRESSURE_5/"
srun --gres=gpu:1 --ntasks=1 --nodes=1 python rg.py --state-pkl state.pkl --dir-path ${addr} --img my_rg > look.out 2>&1

