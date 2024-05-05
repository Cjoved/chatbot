
from dateutil.parser import parse
import datetime

def date_validator(date):

    try:
        date=parse(date)

        return True

    except:

        return False
    
def starttime_validator(starttime):
    try:
        starttime = parse(starttime)
        return True
    except:
        return False
    
def endtime_validator(endtime):
    try:
        endtime = parse(endtime)
        return True
    except:
        return False
def location_validator(location):
    if len(location)<=30:
        return True
    else:
        return False
    

def convert_to_military_startime(time_str):
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

def convert_to_military_endtime(time_str):
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
    
def convert_to_military_date(date_str):
    # Parse the date string
    parsed_date = parse(date_str)
    
    # Format the date in military format (YYYY-MM-DD)
    military_date_str = parsed_date.strftime('%Y-%m-%d')
    
    return military_date_str