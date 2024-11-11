import json

def load_prompts_with_instructions(file_path, instruction):
    """
    Load prompts from a text file and pair them with provided instructions,
    formatting the data into a structured JSON format.

    Args:
        file_path (str): The path to the text file containing prompts.
        instructions (list): A list of instructions, each paired with a prompt.

    Returns:
        dict: A JSON-like dictionary containing the conversation structure.
    """
    conversation = []

    # Read the prompts from the text file and format them with instructions
    with open(file_path, 'r') as file:
        for line in file:
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

    return {"conversation": conversation}


# Define the file path and instructions
file_path = 'prompts.txt'  # Replace with your file path if needed
instruction = "Based on observations, what recommendations can improve the patient’s mobility and safety?"

# Call the function and print the result
conversation_data = load_prompts_with_instructions(file_path, instruction)
print(json.dumps(conversation_data, indent=4))
