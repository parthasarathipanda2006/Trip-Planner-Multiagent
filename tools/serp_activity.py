from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv("SERP_API_KEY")

def serp_activity(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key
    }

    search=GoogleSearch(params)
    results=search.get_dict()
    sights=results.get('top_sights')
    normalized:list[dict]=[]
    for s in sights.get('sights',[]) :
        normalized.append(
            {
                "place_name":s.get('title'),
                "description":s.get('description'),
                "rating":s.get('rating'),
                "reviews":s.get('reviews'),
                "price":s.get('price'),
            }
        )
    return normalized
#print(serp_activity("top 10 tourist attractions in nepal"))