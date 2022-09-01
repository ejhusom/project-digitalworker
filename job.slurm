#!/bin/bash
#SBATCH --output /home/erikj/output/dvcrepro-%j.out
#SBATCH --job-name project-digitalworker-dvc-repro
#SBATCH --partition sintef
#SBATCH --ntasks 1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --time 00-08:00:00

. "/opt/miniconda3/etc/profile.d/conda.sh"

conda activate tf
srun ./dvcrepro
