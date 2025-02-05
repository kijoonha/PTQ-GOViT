#!/bin/bash

#SBATCH --job-name=FQ-ViT                               # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                            # Using 4 gpu
#SBATCH --time=0-12:00:00                     # 1 hour timelimit
#SBATCH --mem=40000MB                         # Using 20GB CPU Memory
#SBATCH --partition=P2                        # Using "b" partition 
#SBATCH --cpus-per-task=4                     # Using 4 maximum processor
#SBATCH --output=/home/s3/joonhaki/Video-Swin-Transformer_fqvit_backup/work_dirs/test_quant_log2/slurm-%j.out     # 표준 출력 파일



source ${HOME}/.bashrc
source ${HOME}/miniconda3/bin/activate
conda activate swin

srun python test_quant.py swin_base /shared/erc/lab02/ILSVRC12 --quant --ptf --lis --outlier --quant-method minmax