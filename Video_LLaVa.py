

############################################################################################
##################################### LOAD THE MODEL #######################################
############################################################################################
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch

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
##################################### PREPARING THE INPUTS #################################
############################################################################################
import av
import numpy as np

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


# sample uniformly 24 frames from the video (we can sample more for longer videos)
total_frames = container.streams.video[0].frames
#indices = np.arange(0, total_frames, total_frames / 8).astype(int)
indices = np.arange(0, total_frames, total_frames / 24).astype(int)
clip_patient = read_video_pyav(container, indices)


conversation = [
      {
          "role": "user",
          "content": [
        {"type": "text", "text": "Give me the list of the objects you see in the environment in the video. In addition, also tell me a list of objects you see on the wall and floor and things like windows, doors, glass doors, curtains, lamps, wheelchair, sofa, bed, clinical devices, table, counter tops, sinks, electronic devices etc. Be concise and just a list is needed. No inference required. Avoid Repetition."},
              {"type": "video"},
              ],
      },
]
conversation_0 = [
      {
          "role": "user",
          "content": [
             {"type": "text", "text":"Start Fresh. This video shows a patient in their home receiving a visit from a home health nurse. Give me the list of objects you see in the environment in the video like devices, clinical devices,  objects, electronic appliances etc.  Be concise and just a list is needed. No inference required. Don't repeat the objects"},
              {"type": "video"},
              ],
      },
]


conversation_1 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Give me the list of things related to patient that you see in the video, like clothes, color of clothes, shoes, hairs. Be concise and just list is needed. No inference required. Avoid Repetition."},
              {"type": "video"},
              ],
      },
]
conversation_2 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Observe the list of objects in the home environment around the patient and give give two word answers to each of the question mentioned below, based on what you see in the video. Be concise"},
              {"type": "text", "text":"1. Is the living area soiled? Are there any insects?"},
              {"type": "text", "text":"2. Is there mold?"},
              {"type": "text", "text":"3. Are there excessive pets?"},
              {"type": "text", "text":"4. Is the residence structurally unsound?"},
              {"type": "text", "text":"5. Are there any stairs?"},
              {"type": "text", "text":"6. Are there inadequate or obstructed entry or exits?"},
              {"type": "text", "text":"7. Is the living space cluttered with furniture?"}, 
                {"type": "text", "text":"8. Is the floor cluttered with unkempt or unorganised objects?"},
                {"type": "text", "text":"9. Are there too many objects scattered in the surrounding of patient?"},
                {"type": "text", "text":"10. Are there rugs or mats on the floors?"},
                {"type": "text", "text":"11. Is this a crowded living space?"},
                {"type": "text", "text":"12. Are there any exposed wiring?"},
                {"type": "text", "text":"13. Are there any structural barriers or lots of furniture that can cause tripping?"},
                {"type": "text", "text":"14. Do you see cigarettes around?"},
                {"type": "text", "text":"15. Is the home well lit?"},
              {"type": "video"},
              ],
      },
]
conversation_3 = [
      {
          "role": "user",
          "content": [
    {"type": "text", "text": "Observe the activities of the patient and give two word answers to each of question mentioned below, based on what you see in the video. Be concise."},
    {"type": "text", "text": "1. Does the patient show signs of sadness, hopelessness or low self esteem?"},
    {"type": "text", "text": "2. Is the patient showing loss of interest in activities, involvement in activities or self care?"},
    {"type": "text", "text": "3. Is the patient irritated, agitated or aggressive?"},
    {"type": "text", "text": "4. Does the patient show active mobility?"},
    {"type": "text", "text": "5. Does the patient get up from chair?"},
    {"type": "text", "text": "6. Does the patient do exercise or movements?"},
    {"type": "text", "text": "7. Is the patient properly dressed or unkempt?"},
              {"type": "video"},
              ],
      },
]


prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
prompt_0 = processor.apply_chat_template(conversation_0, add_generation_prompt=True)
prompt_1 = processor.apply_chat_template(conversation_1, add_generation_prompt=True)
prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)
prompt_3 = processor.apply_chat_template(conversation_3, add_generation_prompt=True)
prompt_4 = processor.apply_chat_template(conversation_3_a, add_generation_prompt=True)

############################################################################################
#we still need to call the processor to tokenize the prompt and get pixel_values for videos
############################################################################################
inputs = processor([prompt, prompt_0,prompt_1,prompt_2,prompt_3], videos=[clip_patient, clip_patient, clip_patient,clip_patient, clip_patient], padding=True, return_tensors="pt").to(model.device)
generate_kwargs = {"max_new_tokens": 1000, "do_sample": True, "top_p": 0.9}

output = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(output, skip_special_tokens=True,temperature=0.5)

processed_text = [text.replace("\\n", "\n") for text in generated_text]

filename2='output_mllm'
# Join the list into a single string with newlines and write it to the file
with open(filename2, "w") as file:
    file.write("\n".join(processed_text))

print(processed_text)

