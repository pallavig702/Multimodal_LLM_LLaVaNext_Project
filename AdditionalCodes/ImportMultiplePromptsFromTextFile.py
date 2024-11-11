import json

# Path to the text file with prompts
file_path = 'prompts.txt'  # Replace with your file path if needed

# Initialize the conversation structure
conversation = []

# Sample instructions for different aspects of analysis
instructions = [
    "Identify any potential hazards in the environment that could impact patient safety.",
    "Observe the floor condition. Is it cluttered, slippery, or uneven? How might this affect the patient’s mobility?",
    "Describe the patient’s gait. Are there signs of unsteadiness, shuffling, or limping?",
    "Watch for any difficulty in navigating obstacles, such as stepping over thresholds or moving around furniture.",
    "Observe any signs of balance issues. Does the patient sway or stumble while walking?",
    "Identify behaviors suggesting cognitive impairment, such as confusion in navigating or difficulty recognizing familiar objects or pathways.",
    "Based on observations, what recommendations can improve the patient’s mobility and safety?"
]

# Read the prompts from the text file and format them
with open(file_path, 'r') as file:
    for line, instruction in zip(file, instructions):
        prompt_text = line.strip()
        if prompt_text:  # Skip empty lines
            conversation.append({
                "instruction": instruction,  # Add specific instructions for the prompt
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "video"}
                ],
                "response": [
                    {"type": "text", "text": "<placeholder for response>"},
                    {"type": "video"}
                ]
            })

# Print the formatted conversation to verify
print(json.dumps({"conversation": conversation}, indent=4))
