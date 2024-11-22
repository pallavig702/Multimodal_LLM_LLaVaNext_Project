# Install the required package
# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

from llava.model.builder import load_pretrained_model  # Load pretrained model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # Utilities for processing images
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN, IGNORE_INDEX
)  # Constants used in the model
from llava.conversation import conv_templates, SeparatorStyle  # Conversation utilities
from PIL import Image  # Image processing library
import requests  # HTTP requests library
import copy  # To create deep copies of objects
import torch  # PyTorch for handling tensors and models
import warnings  # Suppress warnings
from decord import VideoReader, cpu  # Decord for video processing
import numpy as np  # Numerical operations

# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")

# Function to load and process video frames
def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """
    Load video frames from the specified path, sample frames based on fps, 
    and optionally force uniform sampling.

    Args:
        video_path (str): Path to the video file.
        max_frames_num (int): Maximum number of frames to sample.
        fps (int): Frames per second to sample.
        force_sample (bool): Whether to force uniform frame sampling.

    Returns:
        tuple: Sampled frames (numpy array), frame times (list), and video duration (float).
    """
    # If no frames are required, return a placeholder array
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    
    # Load video using Decord
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)  # Total number of frames in the video
    video_time = total_frame_num / vr.get_avg_fps()  # Calculate video duration in seconds

    # Calculate frame interval based on fps
    fps = max(1, round(vr.get_avg_fps() / fps))  # Ensure at least 1 frame per interval
    frame_idx = [i for i in range(0, len(vr), fps)]  # Frame indices to sample
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]  # Timestamps of sampled frames

    # If too many frames are sampled or forced sampling is requested
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num  # Limit frames to the maximum allowed
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()  # Update frame indices
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]  # Update frame times

    # Format frame times as a readable string
    frame_time = ", ".join([f"{i:.2f}s" for i in frame_time])
    
    # Extract sampled frames as a numpy array
    sampled_frames = vr.get_batch(frame_idx).asnumpy()
    
    return sampled_frames, frame_time, video_time

# Load model and associated resources
pretrained = "lmms-lab/LLaVA-Video-72B-Qwen2"  # Pretrained model ID
model_name = "llava_qwen"  # Model name
device = "cuda"  # Use GPU for faster computation
device_map = "auto"  # Automatically map model layers to GPU/CPU

# Load the pretrained model, tokenizer, and image processor
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map
)
model.eval()  # Set the model to evaluation mode

# Specify the path to the video file
video_path = "XXXX"  # Replace this with the actual video path
max_frames_num = 64  # Maximum number of frames to process

# Process the video to extract frames
video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)

# Preprocess video frames using the image processor
video_tensor = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
video_tensor = [video_tensor]  # Wrap the tensor in a list

# Define the conversation template and instructions
conv_template = "qwen_1_5"  # Specify the conversation template
'''
time_instruction = (
    f"The video lasts for {video_time:.2f} seconds, and {video_tensor[0].shape[0]} frames "
    f"are uniformly sampled from it. These frames are located at {frame_time}. "
    "Please answer the following questions related to this video."
)
'''
# Define the question to be asked about the video
question = DEFAULT_IMAGE_TOKEN + f"\nPlease describe this video in detail."

# Initialize the conversation
conv = copy.deepcopy(conv_templates[conv_template])  # Get a fresh copy of the conversation template
conv.append_message(conv.roles[0], question)  # Add the user's question
conv.append_message(conv.roles[1], None)  # Placeholder for the model's response
prompt_question = conv.get_prompt()  # Generate the full conversation prompt

# Tokenize the input question, adding image tokens
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

# Generate the model's response
cont = model.generate(
    input_ids,
    images=video_tensor,  # Pass the video frames as input
    modalities=["video"],  # Specify that the input is video
    do_sample=False,  # Use deterministic decoding
    temperature=0,  # No randomness in generation
    max_new_tokens=4096,  # Maximum length of the response
)

# Decode and print the model's output
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)
