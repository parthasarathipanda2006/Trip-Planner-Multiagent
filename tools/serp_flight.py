from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv("SERP_API_KEY")
def serp_flight(Departure_id:str,Arrival_id:str,Outbound_date:str):
    params = {
        "engine": "google_flights",
        "departure_id": Departure_id,        # Paris Charles de Gaulle (IATA)
        "arrival_id": Arrival_id,          # New York JFK
        "outbound_date": Outbound_date,
        "type":2,  # remove for one-way
        "currency": "USD",
        "hl": "en",
        "api_key":api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    flights = results.get("best_flights", []) + results.get("other_flights", [])
    Normalized:list[dict]=[]

    for f in flights:
        Normalized.append(
            {
                "total_price":f.get("price") or "" ,
                "total_duration":f.get("total_duration") or "",
                "layovers":len(f.get("layovers",[])) or None,
                "ticket extensions":f.get("extensions",[]) or f.get('flight_extension',[]) or[],
                "departure_airport":f.get('flights')[0].get("departure_airport").get("name"),
                "arrival_airport":f.get('flights')[-1].get("arrival_airport").get("name"),
                "departure_time":f.get('flights')[0].get("departure_airport").get("time"),
                "arrival_time":f.get('flights')[-1].get("arrival_airport").get("time"),
                "segments":[
                    {
                        "travel_class":flight.get("travel_class"),
                        "flight_name":flight.get('airplane'),
                        "airline":flight.get('airline'),
                        "flight_extension":flight.get("extensions",[]) or flight.get('flight_extension',[]) or []
                    }
                    for flight in f.get('flights')
                ]

            }
        )
    return Normalized[:10]