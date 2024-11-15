# This script runs the multi-modal LLaVa-Next Script on cluster utilizing GPUs

#!/bin/bash
#SBATCH --job-name=palg_multimodal_gpu_jobl2
#SBATCH --output=output/visuals_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=i.q
#SBATCH --gres=gpu:l40s:2

source ~/LLaVaNV_env/bin/activate

#module load cuda/10.2             # Load CUDA or other GPU-related modules
#Ask GPT how to check how to check nvcc --version on slurm? and then ask your system to get the correct version for yourself for nvcc and cuda
module --ignore_cache load nvhpc-hpcx-cuda12/24.5

# Print the GPU being used
echo "SLURM job running on GPU(s):"
nvidia-smi --list-gpus

# Alternatively, print the CUDA_VISIBLE_DEVICES environment variable (useful in multi-GPU environments)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run the script
python TestScripts/Video_LLaVa.py
