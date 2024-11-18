# Multimodal_LLM_LLaVaNext_Project

### Description
This project leverages LLaVA Next, a multimodal large language model (MLLM) for visual understanding tasks. The goal is to explore and demonstrate the model's capabilities in interpreting visual information, handling multimodal inputs, and generating detailed outputs. Key applications include object recognition, scene description, and image-text alignment.
* To get a quick overview on MLLMs see [source 1](https://github.com/pallavig702/MultiModal-Knowledge-Base/blob/main/README.md) and [source 2](https://medium.com/@tenyks_blogger/multimodal-large-language-models-mllms-transforming-computer-vision-76d3c5dd267f)

### About the Project
LLaVA Next offers state-of-the-art capabilities for multimodal understanding, combining visual encoding with natural language processing to support a range of tasks. This repository includes scripts and functions for loading and processing videos and querying the model, and obtaining visual interpretations.

### Key Features
- Object and Scene Recognition: Identifies objects and scenes in images with contextual descriptions.  <br />
- High-Resolution Support: Processes high-resolution images for detailed output.  <br />
- Multimodal Querying: Combines image and text inputs to generate a cohesive understanding.  <br />

### Installation and Setup
To use this project, ensure that you have Python 3.8+ and follow the steps below:
#### Step 1: Clone the Repository
git clone https://github.com/yourusername/llava-next-visual.git <br />
cd llava-next-visual

#### Step 2: Create Environment <br />
python3 -m venv LLaVaNV_env      ##LLaVaNV_env is the name of the environment

#### Step 2: Activate the environment
source LLaVaNV_env/bin/activate 

#### Step 3: Install the following dependencies/requirements in linux terminal <br />
pip install -r requirements.txt <br />
bash OtherRequirements.sh <br />

#### Step 4: First and foremost Covert .MOV video files to .mp4. 
This conversion depends on the Multimodal LLM (MLLMs) LLaVa-Next we used. It could be different for other MLLMs.<br />
Run ConvertVideos_MOV_to_mp4.py

#### Step 5: Run the script to extract video understing from converted mp4.
python <ADD name of the script>
To check every step check same steps in jupyter notebook.

![test](https://github.com/pallavig702/Multimodal_LLM_LLaVaNext_Project/blob/main/Images/Flow_of_script.png)
#### Optional Steps (as per the GPU resource using)
We used slurm for this project as we required HPC GPU clusters so we use sbatch. We used 2 Nvidia L40s. <br />
sbatch sbatch_gpu_resource.sh


