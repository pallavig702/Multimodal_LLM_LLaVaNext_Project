

#Load the model

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

#Preparing the video inputs
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

#video_path_1 = "sample_data/test.mp4"
#container = av.open("/content/drive/My Drive/MaximTopazLab/MLLMs/Martin_Shared_Files/Videos/test.mp4")

#container = av.open("/home/pgupt60/scripts/test.mp4")
filename = '/home/pgupt60/scripts/CPU_ConvertedVideos/Scenario2_Ipad1_05.mp4'
container = av.open(filename)
#container = av.open("sample_data/01.MOV")


# sample uniformly 8 frames from the video (we can sample more for longer videos)
total_frames = container.streams.video[0].frames
#indices = np.arange(0, total_frames, total_frames / 8).astype(int)
indices = np.arange(0, total_frames, total_frames / 24).astype(int)
clip_patient = read_video_pyav(container, indices)

'''
#prepare the propmt and generate
conversation = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Start fresh. This is a video of a patient at their home space having a home health nurse visitat the house. Describe the scene in this video keeping the instruction in mind."},
              {"type": "video"},
              ],
      },
]
conversation_0 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Start fresh. This is a video of a patient at their home space having a home health nurse visitat the house.Ignore race of humans in this video. Describe the scene in this video keeping the instruction in mind."},
              {"type": "video"},
              ],
      },
]
conversation_x = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text":"Start Fresh. This video shows a patient at home during a visit from a home health nurse. The focus is on the interaction between the nurse and the patient, as the nurse attends to the patient’s needs, potentially performing health checks or engaging in conversation. The patient’s facial expression reflects their emotional state, which may appear happy, sad, or confused, providing context to their cognitive state. The home environment is visible, with elements of the living space contributing to the overall setting of personalized care within the patient's own surroundings. The interaction highlights the home health care process and its connection to the patient's everyday life. Describe this video keeping the instructions in mind"},
              {"type": "video"},
              ],
      },
]


conversation_2 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse. Where is the patient and how does the patient's surroundings look like in this video? Well organized or cluttered? Elucidate."},
              {"type": "video"},
              ],
      },
]
conversation_3 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse. what do you see in the home environment?"},
              {"type": "video"},
              ],
      },
]
conversation_4 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse. Is there any clinical or medical equipments used in the video?"},
              {"type": "video"},
              ],
      },
]
conversation_5 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse. Does the patient looks happy, sad, no expression, confused?"},
              {"type": "video"},
              ],
      },
]
conversation_6 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse. Does the patient look well groomed or unkempt in her appearance?"},
              {"type": "video"},
              ],
      },
]
conversation_7 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse Is the room where patient is sitting looks organized or unkempt?"},
              {"type": "video"},
              ],
      },
]
conversation_8 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse. Is the room well lit or is it dark and dingy? Are there any chances of mould growth"},
              {"type": "video"},
              ],
      },
]


conversation_9 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse? Is the patient's mobility shaky?"},
              {"type": "video"},
              ],
      },
]

conversation_10 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Is the patient stable or shows any movements or mobility? Describe the events"},
              {"type": "video"},
              ],
      },
]

conversation_11 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Is the patient interacting approriately without too much stress or thinking?"},
              {"type": "video"},
              ],
      },
]
conversation = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Start Fresh. This video shows a patient in their home receiving a visit from a home health nurse. Describe the visuals in the home environment in this video. What devices, clinical devices (like vitals measurement devices),  objects, electronic appliances do you see on the floor, wall and in the environment in this video"},
              {"type": "video"},
              ],
      },
]

#person attributes: "type": "text", "text": "
#environment:"Give me the list of the objects you see in the environment in the video. In addition also tell me list of objects you see on wall and floor and things like windows, doors, curtains etc. Are there any clinical devices in the environment? List them too. Be concise and just list is needed. No inference required. Avoid Repetition."},
'''


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
conversation_env = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Observe the list of objects in the home environment around the patient and give two word answers to each of the question mentioned below, based on what you see in the video. Be concise"},
              {"type": "text", "text":"1. Is there a table?"},
              {"type": "text", "text":"2. Is there a fixed-rail armrest?"},
              {"type": "text", "text":"3. Are there any cushions or pillows??"},
              {"type": "text", "text":"4. Is there any blanket?"},
              {"type": "text", "text":"5. Are there any crutches?"},
              {"type": "text", "text":"6. Is there any wheel chair?"},
              {"type": "text", "text":"7. Is there any book?"},
              {"type": "text", "text":"8. Is there any rug on the floor? If yes, how many rugs are there?"}, 
                {"type": "text", "text":"9. Can drinks and food be observed in the video?"},
                {"type": "text", "text":"10. Is there any door at enterance or exit of the room??"},
                {"type": "text", "text":"11. Are there any windows in the room?"},
                {"type": "text", "text":"12. Are there any curtains in the room?"},
                {"type": "text", "text":"13. Is there any trash can in the room?"},
                {"type": "text", "text":"14 Is there any trash on the floor?"},
                {"type": "text", "text":"15. Are there any sofa? How many sofa seats are there?"},
                {"type": "text", "text":"16. Are there any chairs? How many chairs are there?"},
                {"type": "text", "text":"17. Is this a kitchen?"},
                {"type": "text", "text":"18. Is this a living room?"},
                {"type": "text", "text":"19. Is there a laptop in the room?"},
                {"type": "text", "text":"20. Is there any artwork on the wall?"},
                {"type": "text", "text":"21. Is there any cart?"},
                {"type":"video"},
                ],
          },
      ]

conversation_person = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Observe the list of things about the patient and give two word answers to each of the question mentioned below, based on what you see in the video. Be concise"},
              {"type": "text", "text":"1. Does the patient use crutches in the video?"},
              {"type": "text", "text":"2. Is the person sitting on a wheel chair?"},
              {"type": "text", "text":"3. Is the person young or old?"},
              {"type": "text", "text":"4. Patient have long or short hairs?"}, 
                {"type": "text", "text":"5. Is the patient wearing dark color clothers??"},
                {"type": "text", "text":"6. Is the patient seen drinking or eating?"},
                {"type": "text", "text":"7. Is the patient having a conversation?"},
                {"type":"video"},
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
'''

conversation_3 = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse. Assess the patient's functional abilities, focusing on tasks such as eating, speaking, interacting with others, personal care, exercising and completing activities of daily living."},
              {"type": "video"},
              ],
      },
]
conversation_3_a = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse. Based on your observation, assess the patient's functional abilities in key areas, including:"},
            {"type": "text", "text": "1. Eating: How well the patient manages food intake."},
            {"type": "text", "text": "2. Speaking: Ability to communicate clearly."},
            {"type": "text", "text": "3. Interaction: Social engagement with the nurse or others present."},
            {"type": "text", "text": "4. Personal care: Managing grooming, hygiene, and dressing."},
            {"type": "text", "text": "5. Activities of daily living (ADLs): Ability to perform routine tasks like moving around or maintaining the environment."},
            {"type": "video"}
        ],
    },
]
'''

{"type": "text", "text": "Observe the activities of the patient and give two word answers or ‘yes’ or ‘no’, to the list of inferences mentioned below, based on what you see in the video:"},
{"type": "text", "text": "1.Nothing extra.Does the patient show signs of sadness, hopelessness or low self esteem?"},
{"type": "text", "text": "2. Is the patient showing loss of interest in activities, involvement in activities or self care?"},
{"type": "text", "text": "3. Is the patient irritated, agitated or aggressive?"},
{"type": "text", "text": "4. Does the patient show active mobility?"}
{"type": "text", "text": "5. Does the patient get up from chair or walk?"}
{"type": "text", "text": "6. Is the patient properly dressed or unkempt?"}
conversation_3_a = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "This video shows a patient in their home receiving a visit from a home health nurse. What do you see in the home environment in this video? What devices, clinical devices (like vitals measurement devices),  objects, electronic appliances do you see on the floor or on the wall or in the home environment in this video?"},
            {"type": "video"}
        ],
    },
]





prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
prompt_0 = processor.apply_chat_template(conversation_env, add_generation_prompt=True)
prompt_1 = processor.apply_chat_template(conversation_person, add_generation_prompt=True)
prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)
prompt_3 = processor.apply_chat_template(conversation_3, add_generation_prompt=True)
prompt_4 = processor.apply_chat_template(conversation_3_a, add_generation_prompt=True)
'''
prompt_5 = processor.apply_chat_template(conversation_5, add_generation_prompt=True)
prompt_6 = processor.apply_chat_template(conversation_6, add_generation_prompt=True)
prompt_7 = processor.apply_chat_template(conversation_7, add_generation_prompt=True)
prompt_8 = processor.apply_chat_template(conversation_8, add_generation_prompt=True)
prompt_9 = processor.apply_chat_template(conversation_9, add_generation_prompt=True)
prompt_10 = processor.apply_chat_template(conversation_10, add_generation_prompt=True)
prompt_11 = processor.apply_chat_template(conversation_11, add_generation_prompt=True)
'''
# we still need to call the processor to tokenize the prompt and get pixel_values for videos
#############inputs = processor([prompt,prompt_4, prompt_0,prompt_1,prompt_2,prompt_3], videos=[clip_patient,clip_patient,clip_patient,clip_patient,clip_patient,clip_patient], padding=True, return_tensors="pt").to(model.device)
#inputs = processor([prompt,prompt_2,prompt_3, prompt_4,prompt_5,prompt_6,prompt_7,prompt_8,prompt_9,prompt_10,prompt_11], videos=[clip_patient,clip_patient,clip_patient,clip_patient,clip_patient,clip_patient,clip_patient,clip_patient,clip_patient,clip_patient,clip_patient], padding=True, return_tensors="pt").to(model.device)

inputs = processor([prompt_0,prompt_1,prompt_2,prompt_3], videos=[clip_patient, clip_patient,clip_patient, clip_patient], padding=True, return_tensors="pt").to(model.device)
generate_kwargs = {"max_new_tokens": 1000, "do_sample": True, "top_p": 0.9}

output = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(output, skip_special_tokens=True,temperature=0.5)

processed_text = [text.replace("\\n", "\n") for text in generated_text]

filename2='output_mllm'
# Join the list into a single string with newlines and write it to the file
with open(filename2, "w") as file:
    file.write("\n".join(processed_text))


print(processed_text)

