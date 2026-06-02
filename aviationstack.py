from dotenv import load_dotenv
import os
import requests
load_dotenv()

def aviation(origin:str,dest:str):
    api_key=os.getenv("AVIATIONSTACK_API_KEY")
    url=f"http://api.aviationstack.com/v1/flights?access_key={api_key}&dep_iata={origin.upper().strip()}&arr_iata={dest.upper().strip()}&flight_status=active"
    DATA=[]
    try:
        r=requests.get(url).json()["data"]
        
        for data in r:
            DATA.append(
                    {
                        'flight_date':data['flight_date'],
                        'departure_time':data['departure']['scheduled'],
                        'departure_airport':data['departure']['airport'],
                        'arrival_airport':data['arrival']['airport'],
                        'arrival_time':data['arrival']['scheduled'],
                        'airline':data['airline']['name']
                    }
                )
        
        return DATA
    except Exception as e:
        return e