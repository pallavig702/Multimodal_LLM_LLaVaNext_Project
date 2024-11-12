# Import necessary libraries and modules
import os
import torch
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import av
import numpy as np
import gc

# Set environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure and load the model with mixed precision (FP16)
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

# Prepare video input with memory management
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

filename = '/home/pgupt60/scripts/CPU_ConvertedVideos/Scenario2_Ipad1_05.mp4'
container = av.open(filename)

# Sample uniformly 24 frames from the video (adjust for longer videos)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 24).astype(int)
clip_patient = read_video_pyav(container, indices)

# Load prompts and add instruction
def load_conversations_with_instruction(file_path, instruction):
    """
    Load prompts from a text file and pair them with provided instructions,
    formatting the data into a structured list of dictionaries.
    """
    conversations = {}

    # Read prompts from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterate over each prompt and structure it as a conversation
    for idx, line in enumerate(lines):
        prompt_text = line.strip()
        if prompt_text:
            if idx == 0:
                conversation_key = "conversation"
            else:
                conversation_key = f"conversation_{idx - 1}"

            conversations[conversation_key] = [
                {
                    "role": "user",
                    "instruction": instruction,
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "video"}
                    ]
                }
            ]

    return conversations

file_path = '/home/pgupt60/prompts.txt'
instruction_text = "Please review the video carefully and provide the requested list of objects in detail without inference."

all_conversations = load_conversations_with_instruction(file_path, instruction_text)

# Apply chat template and handle prompts with mixed precision to save memory
prompts = []
for conversation_key, conversation in all_conversations.items():
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    prompts.append(prompt)

# Prepare videos and use the same video frames if required
videos = [clip_patient] * len(prompts)

# Enable mixed precision for memory efficiency
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()  # For scaled training

# Tokenize and prepare inputs
inputs = processor(prompts, videos=videos, padding=True, return_tensors="pt").to(model.device)
generate_kwargs = {"max_new_tokens": 1000, "do_sample": True, "top_p": 0.9, "temperature": 0.9}

# Generate responses with memory management
with autocast():
    output = model.generate(**inputs, **generate_kwargs)

# Decode and clear cached memory
generated_text = processor.batch_decode(output, skip_special_tokens=True)
processed_text = [text.replace("\\n", "\n") for text in generated_text]

# Save generated responses to file
filename2 = 'output_mllm'
with open(filename2, "w") as file:
    file.write("\n".join(processed_text))

# Free up GPU memory
torch.cuda.empty_cache()
gc.collect()

print(processed_text)
