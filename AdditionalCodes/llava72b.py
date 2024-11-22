import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from decord import VideoReader, cpu
from torchvision import transforms
import numpy as np
import copy

# Preprocess Video
def preprocess_video(video_path, max_frames=64, target_size=(224, 224)):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()

    if frames.shape[-1] != 3:
        frames = np.repeat(frames[..., None], 3, axis=-1)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    processed_frames = torch.stack([transform(frame) for frame in frames])
    return processed_frames.unsqueeze(0)

# Load Model
pretrained_model_path = "lmms-lab/LLaVA-NeXT-Video-72B-Qwen2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained_model_path,
    None,
    model_name="llava_qwen",
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto"
)
model.eval()

# Ensure Vision Tower is Materialized and on Correct Device
vision_tower = model.get_vision_tower()
if vision_tower is not None:
    # Initialize meta tensors if needed
    for name, param in vision_tower.named_parameters():
        if param.device.type == "meta":
            print(f"Materializing parameter: {name}")
            param.data = torch.empty(param.size(), device=device, dtype=param.dtype)
    vision_tower.to(device)
    print(f"Vision Tower moved to: {device}")

# Preprocess Video
video_path = "/home/pgupt60/scripts/CPU_ConvertedVideos/Scenario2_Ipad1_05.mp4"
video_tensor = preprocess_video(video_path, max_frames=64, target_size=(224, 224))
video_tensor = video_tensor.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)

# Prepare Input IDs
conv_template = "qwen_1_5"
conv = copy.deepcopy(conv_templates[conv_template])
time_instruction = f"The video has been processed into 64 frames. Please describe this video."
conv.append_message(conv.roles[0], time_instruction)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, 0, return_tensors="pt").unsqueeze(0).to(device)

# Debug Inputs
print(f"Input IDs Shape: {input_ids.shape}, Device: {input_ids.device}")
print(f"Video Tensor Shape: {video_tensor.shape}, Device: {video_tensor.device}")

# Generate Output
image_sizes = [(224, 224)] * video_tensor.shape[1]

output = model.generate(
    input_ids=input_ids,
    images=[video_tensor],
    modalities=["video"],
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096
)

# Decode Output
result = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
print("Model Output:")
print(result)

