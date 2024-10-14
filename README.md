# Multimodal_LLM_LLaVaNext_Project

#Create Environment <br />
python3 -m venv LLaVaNV_env      ##LLaVaNV_env is the name of the environment

#Activate the environment
source LLaVaNV_env/bin/activate 

#Install the following requirements in linux terminal <br />
pip install -r requirements.txt <br />
bash OtherRequirements.sh <br />

# First and foremost Covert .MOV video files to .mp4. This conversion depends on the Multimodal LLM (MLLMs) LLaVa-Next we used. It could be different for other MLLMs.<br />
Run ConvertVideos_MOV_to_mp4.py

#We used slurm for this project as we required HPC GPU clusters so we use sbatch <br />

