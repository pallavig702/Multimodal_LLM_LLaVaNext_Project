#This script is a sophisticated machine learning pipeline to process video data alongside textual prompts using a pre-trained model (LLaVA-NeXT-Video-7B-hf).

############################################################################################
############################## STEP 1: LOAD THE MODEL ######################################
############################################################################################
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor #for pre-trained model and processor.
import torch #for tensor-based computation.
import csv
import av #video decoding.
import numpy as np

# Model configuration
# Explanation: Configures the model to load in 4-bit quantization (reducing memory footprint) and uses 16-bit floating point precision for computations.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Model and Processor Initialization:
# Processor: Prepares inputs for the LLaVA-NeXT model.
# Model: Loads the video-conditioned text generation model with pre-trained weights.

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    device_map='auto'
)

############################################################################################
####################### STEP 2: LOAD TEXT PROMPTS AND INSTRUCTIONS  ############################
############################################################################################

# Function to read text prompts from a file and format them as structured conversations for the model.
def load_text_prompts(filename,instruction):
    with open(filename, 'r') as file:
        # Each line represents a text prompt
        lines = [line.strip() for line in file if line.strip()]
    
    # Convert each line into a conversation format for the model
    conversations = [
        [{
            "role": "user",
            "instruction": instruction,
            "content": [
                {"type": "text", "text": line},
                {"type": "video"}
            ]
        }] for line in lines
    ]
    return conversations

# Load all prompts from 'prompts.txt alongwith instrcution'
instructions = "Observe the patient and objects in the home environment shown in the video. Answer each question based strictly on visible evidence, avoiding assumptions. Be concise, and respond with 'Yes', 'No', or 'Insufficient information' only."

# Load prompts function call
conversations = load_text_prompts('/home/pgupt60/TestScripts/Prompts/FallRisk_SimplePrompts.txt', instructions)

############################################################################################
############################## STEP 3: PREPARING THE VIDEO INPUTS ##########################
############################################################################################

############################## Video Decoding BEGINS #######################################
# Extract uniformly sampled frames (24 in total) from a video using PyAV.
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.
    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# Video processing
filename = '/home/pgupt60/scripts/CPU_ConvertedVideos/Scenario2_Ipad1_05.mp4'

# Step1: Video Processing: Opens the video file
container = av.open(filename)

# Step2: Video Processing: Sample uniformly 24 frames from the video
total_frames = container.streams.video[0].frames

# Step3: Video Processing: Determines frame indices to sample uniformly across the video's length.
indices = np.arange(0, total_frames, total_frames / 24).astype(int)

# Step4: Video Processing: Converts frames into NumPy arrays in RGB format.
clip_patient = read_video_pyav(container, indices)
############################## Video Decoding ENDS #######################################

##################### Code Block: Prompt and Video Processing begins #####################
# Process each text-based prompt conversation from the file
generated_texts = []
prompts = []
for i, conversation in enumerate(conversations):
    # Apply template and generate prompt for each conversation line   
    prompt_var = processor.apply_chat_template(conversation, add_generation_prompt=True)
    prompts.append(prompt_var)

# Create a list of video frames for each prompt
videos = [clip_patient] * len(prompts)

# Process inputs for all prompts
inputs = processor(
    prompts,
    videos=videos,
    padding=True,
    return_tensors="pt"
).to(model.device)
##################### Code Block: Prompt and Video Processing ends #####################

# Model generation with updated generate_kwargs
generate_kwargs = {"max_new_tokens": 1000, "do_sample": True, "top_p": 0.9, "temperature": 0.5}
output = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(output, skip_special_tokens=True)

# Clean up and store generated text
processed_text = [text.replace("\\n", "\n") for text in generated_text]
generated_texts.extend(processed_text)

# Format the generated processed text.
cleaned_data = []

for item in processed_text:
    # Split the text at "ASSISTANT:" to separate question and answer
    if "ASSISTANT:" in item:
        question, answer = item.split("ASSISTANT:", 1)
        # Remove "USER:" from the question part
        question = question.replace("USER:", "").strip()
        answer = answer.strip()  # Clean up any extra whitespace in the answer
        # Append to cleaned_data list
        cleaned_data.append([question, answer])

# Save to a CSV file
filename = 'output_questions_answers.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Question", "Answer"])
    # Write each question-answer pair
    writer.writerows(cleaned_data)

print(f"Output saved to {filename}")


# Save all generated texts to output file
filename2 = 'output_mllm'
with open(filename2, "w") as file:
    file.write("\n".join(generated_texts))

print(generated_texts)
