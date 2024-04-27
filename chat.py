import random
import json
import torch
import spacy
from data_model import NeuralNet
from process import bag_of_words, tokenize

# Load English language model
nlp = spacy.load("en_core_web_sm")

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

# Bot parameters
bot_name = "Belle"
fallback_responses = [
    "I'm sorry, I didn't quite catch that.",
    "Could you please rephrase that?",
    "I'm not sure I understand. Can you provide more context?"
]

# Main conversation loop
print(f"Welcome to {bot_name}! (Type 'quit', 'exit', or 'bye' to stop the conversation)")
while True:
    user_input = input("You: ").lower()

    if user_input in {"quit", "exit", "bye"}:
        break
    
    
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

    confidence_threshold = 0.75

    # Check if prediction confidence is above threshold
    if predicted_prob > confidence_threshold:
        for intent in data['data']:
            if predicted_tag == intent["tag"]:
                random.shuffle(intent["response"])
                response = random.choice(intent['response']).strip('[]')
                print(f"{bot_name}: {response}")

                # Check if the intent requires further user input
                if intent["tag"] == "Add":
                    context = {}
                    # Prompt for category
                    user_input = input("You: Category: ")
                    context["Category: "] = user_input

                    # Prompt for event
                    for response in data['data'][4]['response']:
                        print(f"{bot_name}: {response}")
                    user_input = input("You: Event: ")
                    context["Event: "] = user_input

                    # Prompt for Date
                    for response in data['data'][5]['response']:
                        print(f"{bot_name}: {response}")
                    user_input = input("You: Date:")
                    context["Date: "] = user_input

                    # Prompt for start time
                    for response in data['data'][6]['response']:
                        print(f"{bot_name}: {response}")
                    user_input = input("You: Start Time: ")
                    context["Start Time:"] = user_input

                    # Prompt for end time
                    for response in data['data'][7]['response']:
                        print(f"{bot_name}: {response}")
                    user_input = input("You: End Time: ")
                    context["End Time"] = user_input

                    # Prompt for location
                    for response in data['data'][8]['response']:
                        print(f"{bot_name}: {response}")
                    user_input = input("You: Location: ")
                    context["Location"] = user_input

                    print(f"{bot_name}: Details added successfully: {context}")


                    #For Update response
                if intent["tag"] == "Update":
                    context_update= {}

                    user_input = input("You: Update Event:")
                    context_update["New Event: "] = user_input

                    for response in data['data'][11]['response']:
                        print(f"{bot_name}:{response}")
                        user_input = input("You: Update Date: ")
                        context_update["New Date: "] = user_input

                    for response in data['data'][12]['response']:
                        print(f"{bot_name}:{response}")
                        user_input = input("You: Update Start Time:: ")
                        context_update["New Start Time: "] = user_input

                    for response in data['data'][13]['response']:
                        print(f"{bot_name}:{response}")
                        user_input = input("You: Update End Time: ")
                        context_update["New End Time:  "] = user_input
                    
                    for response in data['data'][14]['response']:
                        print(f"{bot_name}:{response}")
                        user_input = input("You: Update Location: ")
                        context_update["New Location: "] = user_input

                    print(f"{bot_name}: Details updated successfully: {context_update}")
                
                    #For Delete Response
                if intent["tag"] == "Delete":
                    context_delete= {}

                    user_input = input("You: Delete Event: ")
                    context_delete["Deleted Event: "] = user_input

                    for response in data['data'][17]['response']:
                        print(f'{bot_name}:{response}')
                        user_input = input("You: Delete Date:")
                        context_delete["Deleted Date: "] = user_input

                    for response in data['data'][18]['response']:
                        print(f'{bot_name}: {response}')                    

    # If no intent is matched, print a random response from the fallback responses
    else:
        print(f"{bot_name}: {random.choice(fallback_responses)}")

#Quit
print(f"{bot_name}: Goodbye! Have a great day.")
