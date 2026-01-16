#!/bin/bash
#SBATCH --job-name=GCC-net_train
#SBATCH --partition=gpu-2080ti
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --time=24:00:00
#SBATCH --mail-user=esteban.dreaudariz@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:3

# Aller dans le dossier du projet
cd ~/gccnet


PYTHON_VERSION=3.8.18
ENVIRONMENT_NAME="gccnet_env"

module load Anaconda3
# You need this to be able to use the 'conda' command. I am not satisfied with this solution, I'll try to find a way so that you won't have to set it in the future.
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh


# if the environment does not exist, create it
if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENVIRONMENT_NAME}

# Install python packages, either individually or using a requirements.txt file.
# pip install scipy matplotlib
# or
pip install -r requirements.txt


srun python tools/train.py configs/autoassign/autoassign_custom_dataset.py
