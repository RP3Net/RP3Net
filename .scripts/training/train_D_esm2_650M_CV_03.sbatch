#!/bin/env bash

#SBATCH --time=3-
#SBATCH --gpus=a100:4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --signal=SIGUSR1@90
#SBATCH -J "train_D_esm2_650M_CV_03_01"
#SBATCH -o "/homes/evgeny/nobackup/log/%x_%j.out"
#SBATCH -e "/homes/evgeny/nobackup/log/%x_%j.err"

# NB: set this on the command line
# SBATCH --ntasks-per-node=1

. ~/.bashrc
. ~/code/protein_production/scripts/ai/env_ebi.sh
micromamba activate ProtMan
mkdir -p ~/nobackup/pp_jobs/rp3_training/train_D_esm2_650M_CV_03_01
cd ~/nobackup/pp_jobs/rp3_training/train_D_esm2_650M_CV_03_01
rm -rf *
# wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O ~/micromamba/envs/ProtMan/bin/yq && chmod u+x ~/micromamba/envs/ProtMan/bin/yq
LOGFILE=`realpath ./fit.log` ROOT_DIR=`realpath .` yq '
    .data.init_args.hypers.validation_slice = "CV_03" |
    .data.init_args.hypers.max_seq_len = 900 |
    .logfile = strenv(LOGFILE) |
    .trainer.default_root_dir = strenv(ROOT_DIR) |
    .trainer.num_nodes = env(SLURM_NNODES) |
    .trainer.accelerator = "gpu" |
    .wandb.project = "rp3_training" |
    .wandb.run = "train_D_esm2_650M_CV_03_01"
' ~/code/RP3Net/config/trainer_d.yml > train_D_esm2_650M_CV_03_01.yml
srun rp3_train fit -c train_D_esm2_650M_CV_03_01.yml
