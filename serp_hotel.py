from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv("SERP_API_KEY")
def serp_hotel(check_in:str,check_out:str,place:str,adults:int):
    params = {
        "engine": "google_hotels",
        "q":place,
        "check_in_date":check_in,
        "check_out_date": check_out,
        "adults": adults,
        "currency": "USD",
        "gl": "us",
        "hl": "en",
        "api_key": api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    hotels = results.get("properties", [])

    normalized:list[dict]=[]
    for hotel in hotels :
        normalized.append(
            {
                "hotel_name":hotel.get('name'),
                "description":hotel.get("description"),
                "hotel_claass":hotel.get("hotel_class"),
                "overal_rating":hotel.get("overall_rating"),
                "location_rating":hotel.get("location_rating"),
                "nearby_places":hotel.get("nearby_places")

            }
        )
    return normalized    
print(serp_hotel("2026-06-07","2026-06-10","dubai",3))