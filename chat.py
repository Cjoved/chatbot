import random
import json
import torch
import spacy
from data_model import NeuralNet
from process import bag_of_words, tokenize
from validate_response import date_validator, starttime_validator, endtime_validator,convert_to_military_date,convert_to_military_startime,convert_to_military_endtime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from JSON file
with open('data.json', 'r') as data_file:
    data = json.load(data_file)

# Load trained model
model_file = "data.pth"
model_data = torch.load(model_file)

# Initialize neural network model
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Bot parameters
bot_name = "Belle"
fallback_responses = [
    "I'm sorry, I didn't quite catch that.",
    "Could you please rephrase that?",
    "I'm not sure I understand. Can you provide more context?"
]

# Main conversation loop
print(f"Welcome to {bot_name}! (Type 'quit', 'exit', or 'bye' to stop the conversation)")

def process_user_input(user_input, nlp, model, tags, data, bot_name, device='cpu', all_words=None, fallback_responses=None, confidence_threshold=0.75):
   
    user_input = user_input.lower()

    if user_input in {"quit", "exit", "bye"}:
        return f"{bot_name}: Goodbye! Have a great day."

    # Tokenize user input using spaCy
    doc = nlp(user_input)
    user_tokens = [token.text for token in doc]

    # Bag of words
    user_bow = bag_of_words(user_tokens, all_words)
    user_bow = torch.from_numpy(user_bow).unsqueeze(0).to(device)

    # Pass user input through the model
    output = model(user_bow)
    _, predicted_idx = torch.max(output, dim=1)
    predicted_tag = tags[predicted_idx.item()]

    probs = torch.softmax(output, dim=1)
    predicted_prob = probs[0][predicted_idx.item()].item()

    if predicted_prob > confidence_threshold:
        for intent in data['data']:
            if predicted_tag == intent["tag"]:
                random.shuffle(intent["response"])
                response = random.choice(intent['response']).strip('[]')
                print(f"{bot_name}: {response}")

                # Check if the intent requires further user input
                if intent["tag"] == "Add":
                    context = {}
                    # Prompt for event
                    for response in data['data'][4]['response']:
                        print(f"{bot_name}: {response}")
                    user_input = input("Event: ")
                    context["Event: "] = user_input

                    # Prompt for Date
                    for response in data['data'][5]['response']:
                        print(f"{bot_name}: {response}")
                    user_input = get_valid_date("Add Date: ")
                    convert_date = convert_to_military_date(user_input)
                    context["Date: "] = user_input

                    # Prompt for start time
                    user_input = get_valid_time("Add Start Time: ")
                    convert_startTime = convert_to_military_startime(user_input)
                    context["Start Time:"] = user_input

                    # Prompt for end time
                    user_input = get_valid_time("Add End Time: ")
                    convert_endtime = convert_to_military_endtime(user_input)
                    context["End Time"] = user_input

                    # Prompt for location
                    for response in data['data'][8]['response']:
                        print(f"{bot_name}: {response}")
                    user_input = input("Location: ")
                    context["Location"] = user_input

                    print(f"{bot_name}: Details added successfully: {context}")
                      #check if its converted
                    print(f"Converted Date:{convert_date} Converted Startime: {convert_startTime} Converted End Time: {convert_endtime}")

                # For Update response
                elif intent["tag"] == "Update":
                    context_update= {}

                    user_input = input("Update Event:")
                    context_update["New Event: "] = user_input

                    for response in data['data'][11]['response']:
                        print(f"{bot_name}:{response}")
                        user_input = get_valid_date("New Date: ")
                        convert_date = convert_to_military_date(user_input)
                        context_update["New Date: "] = user_input

                        user_input = get_valid_time("Update Start Time: ")
                        convert_startTime = convert_to_military_startime(user_input)
                        context_update["New Start Time: "] = user_input

                        user_input = get_valid_time("Update End Time: ")
                        convert_endtime = convert_to_military_endtime(user_input)
                        context_update["New End Time:  "] = user_input
                    
                    for response in data['data'][14]['response']:
                        print(f"{bot_name}:{response}")
                        user_input = input("Update Location: ")
                        context_update["Updated Location: "] = user_input

                    print(f"{bot_name}: Details updated successfully: {context_update}")

                    #check if its converted
                    print(f"Converted Date:{convert_date} Converted Startime: {convert_startTime} Converted End Time: {convert_endtime}")
                
                # For Delete Response
                elif intent["tag"] == "Delete":
                    context_delete= {}

                    user_input = input("You: Delete Event: ")
                    context_delete["Deleted Event: "] = user_input

                    for response in data['data'][17]['response']:
                        print(f'{bot_name}:{response}')
                        user_input = get_valid_date("Delete Date: ")
                        convert_date = convert_to_military_date(user_input)
                        context_delete["Deleted Date: "] = user_input

                        print(convert_date)

                    for response in data['data'][18]['response']:
                        print(f'{bot_name}: {response}')                    

    # If no intent is matched, print a random response from the fallback responses
    else:
        print(f"{bot_name}: {random.choice(fallback_responses)}")

    return ""

# This function is for validating the time format. reloop if the user input is not valid
def get_valid_time(prompt):
    while True:
        user_input = input(prompt)
        if starttime_validator(user_input) or endtime_validator(user_input):
            return user_input
        else:
            print("Invalid time format. Please enter a valid time.")

# This function is for validating the date format. reloop if the user input is not valid
def get_valid_date(prompt):
    while True:
        user_input = input(prompt)
        if date_validator(user_input):
            return user_input
        else:
            print("Invalid date format. Please enter a valid date")
        

# This is for checking only. For testing purposes. Comment this if you want 
while True:
    user_input = input("You: ")
    if user_input.lower() in {"quit", "exit", "bye"}:
        print(f"{bot_name}: Goodbye! Have a great day.")
        break
    # calling the process function
    response = process_user_input(user_input, nlp, model, tags, data, bot_name, device, all_words, fallback_responses)
