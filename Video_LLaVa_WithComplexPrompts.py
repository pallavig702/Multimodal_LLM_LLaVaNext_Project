############################################################################################
##################################### LOAD THE MODEL #######################################
############################################################################################
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch
import av
import numpy as np

# Model configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    device_map='auto'
)

############################################################################################
##################################### LOAD TEXT PROMPTS ####################################
############################################################################################

# Function to read multi-line prompts from a file and format them as conversations
def load_multiline_text_prompts(filename):
    with open(filename, 'r') as file:
        # Read the file and split prompts by blank lines
        content = file.read().strip().split("\n\n")
        
    # Format each multi-line prompt into a conversation structure
    conversations = []
    for block in content:
        # Split each block into lines and remove any empty lines
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        
        # Each conversation starts with the first line as an instruction, followed by questions
        conversation_content = [{"type": "text", "text": line} for line in lines]
        conversation_content.append({"type": "video"})  # Add video type at the end
        
        # Structure the conversation
        conversation = [{
            "role": "user",
            "content": conversation_content
        }]
        conversations.append(conversation)
    
    return conversations

# Load all multi-line prompts from 'prompts.txt'
conversations = load_multiline_text_prompts('/home/pgupt60/prompts3_structured.txt')

############################################################################################
##################################### PREPARING THE INPUTS #################################
############################################################################################

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
container = av.open(filename)

# Sample uniformly 24 frames from the video
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 24).astype(int)
clip_patient = read_video_pyav(container, indices)

# Process each conversation prompt and prepare inputs
generated_texts = []
prompts = []
for conversation in conversations:
    # Apply template and generate prompt for each conversation
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

# Model generation with updated generate_kwargs
generate_kwargs = {"max_new_tokens": 1000, "do_sample": True, "top_p": 0.9, "temperature": 0.5}
output = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(output, skip_special_tokens=True)

# Clean up and store generated text
processed_text = [text.replace("\\n", "\n") for text in generated_text]
generated_texts.extend(processed_text)

# Save all generated texts to output file
filename2 = 'output_mllm'
with open(filename2, "w") as file:
    file.write("\n\n".join(generated_texts))

print(generated_texts)

