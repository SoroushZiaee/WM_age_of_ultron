#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter.out
#SBATCH --error=jupyter.err


#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G


#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

echo "Start Installing and setup env"
source /home/soroush1/projects/def-kohitij/soroush1/WM_age_of_ultron/bash/prepare_env/setup_env_node.sh

module list

pip freeze

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

echo "Installing requirements"
pip install --no-index -r requirements.txt

echo "Env has been set up"

pip freeze

# python /home/soroush1/projects/def-kohitij/soroush1/WM_age_of_ultron/scripts/prepare_server/datamodule_imagenet_preparation.py

/home/soroush1/projects/def-kohitij/soroush1/WM_age_of_ultron/bash/notebooks/lab.sh
