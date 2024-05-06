from dateutil.parser import parse
from dateparser.search import search_dates
from datetime import datetime as dt
import spacy
import date_spacy
import re


time_format = "%H:%M:%S"
date_format = "%Y-%m-%d"
# Load your desired spaCy model
nlp = spacy.blank('en')

# Add the component to the pipeline
nlp.add_pipe('find_dates')

def date_validator(text):
    try:
        doc = nlp(text)
        #date=parse(date)
        # regex to confirm if the text has a number
        # kung walng number ang text, raise exception
        if re.search(r'\d', text):


            date_extracted = None
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    print(f"Text: {ent.text} -> Parsed Date: {ent._.date}")
                    date_extracted = ent._.date
                    date_formatted = date_extracted.strftime(date_format)
                    return date_formatted, True
        
                if date_extracted is None:
                    result = search_dates(text)
                    return result[0][1].strftime(date_format), False
        
        else:
            return Exception
    except Exception as e:
        print(f"Error occured: {e}")
        return None, False
    
def time_validator(text):
    try:
        starttime_nm = re.search(r"(?i)\d{1,2}:\d{2}(?:\s*([Pp][Mm]|[Aa][Mm]))?", text)#editable
        starttime_sc = re.search(r'(1[0-2]|0?[1-9])\s*(?:AM|PM)', text) # 10 PM
        starttime_oc = re.search(r'(1[0-2]|0?[1-9]) o\'clock', text)
        starttime = None
        if starttime_nm is not None:
            starttime = parse(starttime_nm.group(0))
        elif starttime_sc is not None:
            starttime = parse(starttime_sc.group(0))
        elif starttime_oc is not None:
            # convert format to a format that can parse
            number = starttime_oc.group(0).split(" ")
            if 'morning' or 'dawn' in text.lower():
                time = number[0] + " AM"
                starttime = parse(time)
            elif 'afternoon' or 'evening' in text:
                time = number[0] + " PM"
                starttime = parse(time)
        
        return True,str(starttime)
    except Exception as e:
        print(f"Error occured: {e}")
        return False, None
    
    
def location_validator(text):
    if len(text)<=50:
        return True
    else:
        return False

def convert_to_military_time(time_str):
  # Parse the time string
    parsed_time = parse(time_str)
    
    # Extract hour and minute components
    hour = parsed_time.hour
    minute = parsed_time.minute
    
    # Check if it's PM and not 12 PM
    if "PM" in time_str.upper() and hour != 12:
        military_hour = hour + 12
    elif hour == 12 and "AM" in time_str.upper():
        military_hour = 0  # 12:00 AM
    else:
        military_hour = hour
    
    # Format the military time string
    military_time_str = f"{military_hour:02d}:{minute:02d}"
    
    return military_time_str

    

# FOR TESTING
# print(date_validator("Let's meet sometime next Thursday"))
# print(time_validator("The event ends at 3 PM sharp."))