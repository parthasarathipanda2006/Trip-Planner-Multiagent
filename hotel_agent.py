from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage
from serp_hotel import serp_hotel
import os 

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
class hotel_state(TypedDict):
    place:str
    checkin_date:str
    checkout_date:str
    adults:int
    preferences:str
    hotel_recommendation:list[dict]
    final_recommend:str

hotel_prompt="""
            You are an expert hotel recommendation assistant.

            You are given:

            1. A list of hotel options.
            2. User preferences.

            Each hotel has the following structure:

            {
                "hotel_name": str
                    Name of the hotel.

                "total_rate": str | number
                    Total hotel cost for the selected stay.

                "description": str | list[str]
                    Hotel description, amenities, facilities, and notable features.

                "hotel_claass": int | float
                    Hotel class/star rating.

                "overal_rating": float
                    Overall guest rating.

                "location_rating": float
                    Location rating.

                "nearby_places": list[str]
                    Attractions, landmarks, shopping areas, beaches, business districts, airports, or places near the hotel.
            }

            Your task:

            1. Analyze all hotels.
            2. Compare them against the user's preferences.
            3. Rank the hotels from best to worst.
            4. Recommend at most 5 hotels.
            5. Never recommend the same hotel twice.
            6. Never invent hotel information.
            7. Use only the provided hotel data.
            8. If a preference cannot be verified from the provided data, explicitly mention it.
            9. Prefer hotels that satisfy more user preferences.
            10. Consider hotel cost, ratings, location, nearby places, and amenities when ranking.

            Preference handling:

            - Luxury → prioritize higher hotel_claass and overal_rating.
            - Budget/Cheap → prioritize lower total_rate.
            - Best value → balance total_rate with ratings.
            - Highly rated → prioritize overal_rating.
            - Good location → prioritize location_rating.
            - Tourist attractions → prioritize matching nearby_places.
            - Shopping → prioritize hotels near shopping areas.
            - Beach → prioritize hotels near beaches.
            - Family-friendly → use description and amenities.
            - Business travel → prioritize good location, amenities, and ratings.
            - Airport access → prioritize hotels with airports in nearby_places.

            Output Rules:

            - Return AT MOST 5 recommendations.
            - If fewer than 5 hotels satisfy the preferences, return only the matching hotels.
            - Do not repeat hotels.
            - Do not explain your ranking process.

            For each recommendation provide:

            1. Hotel Name
            2. Total Cost
            3. Overall Rating
            4. Location Rating
            5. Hotel Class
            6. Nearby Places
            7. Why it matches the user's preferences
            """
def hotel_node(state:hotel_state):
    hotel_list=serp_hotel(state["checkin_date"],state["checkout_date"],state["place"],state["adults"])
    response=llm.invoke(
        [
            SystemMessage(
                content=(hotel_prompt)
            ),
            HumanMessage(
                content=(
                    f"Hotel Data: {hotel_list}"
                    f"my preferences: {state["preferences"]}"
                )
            )
        ]
    )
    return {"final_recommend":response.content,"hotel_recommendtions":hotel_list}

graph=StateGraph(hotel_state)
graph.add_node("hotel",hotel_node)

graph.add_edge(START,"hotel")
graph.add_edge("hotel",END)

hotel_agent=graph.compile()

print(hotel_agent.invoke(
    {
        "place":"singapore",
        "checkin_date":"2026-06-08",
        "checkout_date":"2026-06-15",
        "adults":2,
        "preferences":"5 star hotel with swimming pool"
    }
).get('final_recommend'))