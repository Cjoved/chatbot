import random
import json
import torch
import spacy
from data_model import NeuralNet
from process import bag_of_words
from validate_response import date_validator, time_validator, location_validator, convert_to_military_time

# Initialize global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fallback_responses = [
    "I'm sorry, I didn't quite catch that.",
    "Could you please rephrase that?",
    "I'm not sure I understand. Can you provide more context?"
]


fallback_responses_time = [
    "I'm sorry, I'm having trouble understanding the time you provided. Could you please rephrase it or provide more context?",
    "It seems like there might be an issue with the time format. Could you double-check and provide the time again?",
    "Apologies, but I couldn't quite grasp the time you mentioned. Could you clarify or provide additional details?"
]

fallback_responses_date = [
    "I'm sorry, it looks like there's a misunderstanding with the date you provided. Could you please verify it or offer more information?",
    "It appears that there might be a problem with the date format. Could you please check and provide the date again?",
    "Apologies, I'm having difficulty understanding the date mentioned. Could you clarify or provide additional details?"
]


confidence_threshold = 0.75  # Global confidence threshold
model_file = "data.pth"
data_file = "data.json"

# Load data from JSON file
with open('data.json', 'r') as data_file:
    data = json.load(data_file)
# Load trained model
model_data = torch.load(model_file)
input_size, hidden_size, output_size = model_data["input_size"], model_data["hidden_size"], model_data["output_size"]
all_words, tags, model_state = model_data['all_words'], model_data['tags'], model_data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

def receive_input(message):
    response = predict_intent(message)
    return response

# Predict intent from user input
def predict_intent(message):
    doc = nlp(message)
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
        return handle_intent(predicted_tag,message)
    else:
        return {"message": random.choice(fallback_responses)}

# Generate response based on predicted intent
def generate_response(predicted_tag):
   # response = handle_intent(predicted_tag,message)

    intent_matched = False
    for intent in data['data']:
        if predicted_tag == intent["tag"]:
            intent_matched = True
            random.shuffle(intent["response"]) 
            
            #return f"{random.choice(intent['response']).strip('[]')}"
            message = random.choice(intent['response']).strip('[]')
            return {"message": message}
            
    if not intent_matched:
        return {random.choice(fallback_responses)}

def handle_intent(intent, message):
    if intent == "Add":
        return generate_response(intent)
    elif intent == "Add_Event":
        return generate_response(intent)
    elif intent == "Add_Date":
        text = message.strip()
        validation_result = date_validator(text)
        if validation_result[1]:
            message= generate_response(intent)
            convert_date = validation_result[0]
            return {"Date": convert_date, "message": message}
        else:                  
            return {"message": random.choice(fallback_responses_date)}
    elif intent == "Add_StartTime":
        text = message.strip()
        validation_result = time_validator(text)
        if validation_result[0]:
            message= generate_response(intent)
            convert_st = convert_to_military_time(validation_result[1])
            return {"Start Time": convert_st, "message": message}
        else:
             return {"message": random.choice(fallback_responses_time)}
    elif intent =="Add_EndTime":
        text = message.strip()
        validation_result = time_validator(text)
        if validation_result[0]:
            message= generate_response(intent)
            convert_et = convert_to_military_time(validation_result[1])
            return {"End Time": convert_et, "message": message}          
        else:
            return {"message": random.choice(fallback_responses_time)}
    elif intent == "Add_Location":
        location = message.strip()
        if location_validator(location):
           result = generate_response(intent)
           result["Location"] = location
           return result
        else:
            return {"message": random.choice(fallback_responses)}
        

    elif intent == "Update":
        return generate_response(intent)
    elif intent == "Update_Event":
      
      return generate_response(intent)
    elif intent == "Update_Date":
        text = message.strip()
        validation_result = date_validator(text)
        if validation_result[1]:
            message= generate_response(intent)
            convert_date = validation_result[0]
            return {"Date": convert_date, "message": message}
        else:
            return {"message": random.choice(fallback_responses_date)}
        
    elif intent == "Update_StartTime":
        text = message.strip()
        validation_result = time_validator(text)
        if validation_result[0]:
            message= generate_response(intent)
            convert_st = convert_to_military_time(validation_result[1])
            return {"Start Time": convert_st, "message": message}
        else:
             return {"message": random.choice(fallback_responses_time)}
    elif intent == "Update_EndTime":
        text = message.strip()
        validation_result = time_validator(text)
        if validation_result[0]:
            message= generate_response(intent)
            convert_et = convert_to_military_time(validation_result[1])
            return {"End Time": convert_et, "message": message}
        else:
            return {"message": random.choice(fallback_responses_time)}
    elif intent== "Update_location":
        update_location = message.strip()
        if location_validator(update_location):
            {"message" : data["data"][14]["response"], "Location": update_location}
        else:
            return{"message":""}
        
##### Still Thinking about thiss
    elif intent == "Delete":
        return generate_response(intent)
    elif intent == "Delete_Date":
        text = message.strip()
        validation_result = date_validator(text)
        if validation_result[1]:
            message= generate_response(intent)
            convert_date = validation_result[0]
            return {"Date": convert_date, "message": message}
        else:
            return {"message": random.choice(fallback_responses_date)}

    return message

############################################################################################
# MESSAGE
###########################################################################################
# Pede bang pagsama samahin na yung patterns for date, end, start at loc, at event?
# Kasi malilito lang ang model kasi halos pare pareho naman ang patterns nun, 
# maiidentify naman natin kung update, add at delete dahil nauuna naman yung pagtatanong natin ng ganun
# Yung sa response di ba pede pagsama samahin na lang din yung responses for update, add, delete?
# Lalagyan nalng siguro ng key sa response para malaman kung ano dun yung ireresponse pag yung 
# aim ni user if for update, add or delete
##############################################################################################
text = ''
while text != 'exit':
    text = input("My Message: ")
    response = receive_input(text)
    print("Response:" + str(response))
