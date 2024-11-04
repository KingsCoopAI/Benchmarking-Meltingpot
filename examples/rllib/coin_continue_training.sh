#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/logs/%j.out
#SBATCH --job-name=test_cuda
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=35
#SBATCH --mem=100G
#SBATCH --time=2-00:00

source ~/.bashrc

module load cuda


/scratch/prj/inf_du/ziyan/benchmarking_meltingpot/bk_conda/bin/python \
    /scratch/prj/inf_du/ziyan/benchmarking_meltingpot/Benchmarking-Meltingpot/examples/rllib/self_play_train.py \
    --use_wandb 1 \
    --num-cpus 100 \
    --num-workers 25 \
    --env-name 'coins' \
    --continue_training \
    --continue_training_path /scratch/prj/inf_du/ziyan/benchmarking_meltingpot/Benchmarking-Meltingpot/examples/rllib/checkpoints/coins/PPO_run_2/PPO_meltingpot_29688_00000_0_2024-10-30_13-59-21/checkpoint_000370