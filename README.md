# Multimodal_LLM_LLaVaNext_Project

### Description
This project leverages LLaVA Next, a multimodal large language model (MLLM) for visual understanding tasks. The goal is to explore and demonstrate the model's capabilities in interpreting visual information, handling multimodal inputs, and generating detailed outputs. Key applications include object recognition, scene description, and image-text alignment.

### About the Project
LLaVA Next offers state-of-the-art capabilities for multimodal understanding, combining visual encoding with natural language processing to support a range of tasks. This repository includes scripts and functions for loading and processing videos and querying the model, and obtaining visual interpretations.

### Key Features
Object and Scene Recognition: Identifies objects and scenes in images with contextual descriptions.
High-Resolution Support: Processes high-resolution images for detailed output.
Multimodal Querying: Combines image and text inputs to generate a cohesive understanding.

### Table of Contents
About the Project
Key Features
Installation and Setup
Usage Guide
Examples
Architecture Overview
Contributing
License
Contact

### Step 1: Create Environment <br />
python3 -m venv LLaVaNV_env      ##LLaVaNV_env is the name of the environment

### Step 2: Activate the environment
source LLaVaNV_env/bin/activate 

### Step 3: Install the following requirements in linux terminal <br />
pip install -r requirements.txt <br />
bash OtherRequirements.sh <br />

### Step 4: First and foremost Covert .MOV video files to .mp4. 
This conversion depends on the Multimodal LLM (MLLMs) LLaVa-Next we used. It could be different for other MLLMs.<br />
Run ConvertVideos_MOV_to_mp4.py

### Optional Steps (as per the GPU resource using)
We used slurm for this project as we required HPC GPU clusters so we use sbatch <br />


