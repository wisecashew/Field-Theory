#!/bin/bash
#
#SBATCH --job-name=monatomic_fluid-restart
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

srun --gres=gpu:1 --ntasks=1 --nodes=1 python restart.py \
--field-pkl field_rho-1_u0-100.pkl > dump-restart_rho-1_u0-100.out 2>&1

# srun --gres=gpu:1 --ntasks=1 --nodes=1 python main.py \
# --field-pkl field_rho-2_u0-100.pkl --operator-dump operator_rho-2_u0-100.dat \
# --rho 2 > dump_rho-2_u0-100.out 2>&1

# srun --gres=gpu:1 --ntasks=1 --nodes=1 python main.py \
# --field-pkl field_rho-1_u0-100.pkl --operator-dump operator_rho-1_u0-100.dat \
# --rho 1 > dump_rho-1_u0-100.out 2>&1

# srun --gres=gpu:1 --ntasks=1 --nodes=1 python main.py \
# --field-pkl field_rho-0.1_u0-100.pkl --operator-dump operator_rho-0.1_u0-100.dat \
# --rho 0.1 > dump_rho-0.1_u0-100.out 2>&1
# wait
