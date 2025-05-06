#!/bin/env bash

#SBATCH --time=1:00:00
#SBATCH --gres=gpu
#SBATCH --mem=16G
#SBATCH --signal=SIGUSR1@90
#SBATCH -J "train_A_esm2_650M_CV_03"
#SBATCH -o "/homes/evgeny/nobackup/log/%x_%j.out"
#SBATCH -e "/homes/evgeny/nobackup/log/%x_%j.err"

. ~/.bashrc
micromamba activate ProtMan
mkdir -p ~/nobackup/pp_jobs/rp3_training/train_A_esm2_650M_CV_03
cd ~/nobackup/pp_jobs/rp3_training/train_A_esm2_650M_CV_03
rm -rf *
# wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O ~/micromamba/envs/ProtMan/bin/yq && chmod u+x ~/micromamba/envs/ProtMan/bin/yq
LOGFILE=`realpath ./fit.log` ROOT_DIR=`realpath .` yq '
    .data.init_args.hypers.validation_slice = "CV_03" |
    .data.init_args.hypers.training_batch_size = 32 |
    .data.init_args.hypers.val_test_batch_size = 32 |
    .logfile = strenv(LOGFILE) |
    .trainer.default_root_dir = strenv(ROOT_DIR) |
    .trainer.accelerator = "gpu"
' ~/code/RP3Net/config/trainer_a.yml > train_A_esm2_650M_CV_03.yml
srun rp3_train fit -c train_A_esm2_650M_CV_03.yml
