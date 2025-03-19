#!/bin/bash
#
#SBATCH --job-name=monatomic_fluid
#SBATCH --ntasks=1
#SBATCH --mem=50GB
#SBATCH --time=01:10:00
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

srun --gres=gpu:1 --ntasks=1 --nodes=1 python main.py > dump.out 2>&1

# wait
