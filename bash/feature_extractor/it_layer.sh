#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --output=feature_extraction_%j.out
#SBATCH --error=feature_extraction_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=soroush1@yorku.ca

# Define the path to your feature extraction script
SCRIPT_PATH="/home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/test/extract_features/it_layer.py"

echo "Start Installing and setup env"
source /home/soroush1/projects/def-kohitij/soroush1/vitalab-trianing-clean-code/bash/prepare_env/setup_env_node.sh

module list
pip freeze
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
echo "Installing requirements"
pip install --no-index -r requirements.txt

echo "Env has been set up"
pip freeze

echo "Running feature extraction job"

# Run the feature extraction script
python $SCRIPT_PATH

echo "Feature extraction job completed"