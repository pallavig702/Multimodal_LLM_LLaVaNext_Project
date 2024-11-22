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
    frame_idx = [i for i in range(0, len(vr
