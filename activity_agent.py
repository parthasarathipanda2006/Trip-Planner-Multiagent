from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage
from serp_activity import serp_activity
import os 

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
class activity_state(TypedDict):
    place:str
    days:int
    preferences:list[str]
    recommendation:str
activity_prompt="""
                You are an expert travel itinerary planner.

                You are given:

                1. Destination city.
                2. Number of days of the trip.
                3. User activity preferences.
                4. A list of tourist attractions.

                Each attraction has the following structure:

                {
                    "place_name": str
                        Name of the attraction.

                    "description": str
                        Description of the attraction.

                    "rating": float
                        Attraction rating.

                    "reviews": int
                        Number of reviews.

                    "price": str | number | None
                        Entry fee or approximate cost if available.
                }

                Your task:

                1. Create a detailed day-by-day itinerary.
                2. Use ONLY the attractions provided in the attraction data.
                3. Match attractions to the user's preferences as closely as possible.
                4. Prefer attractions with higher ratings and more reviews when multiple options are suitable.
                5. Distribute attractions logically across the trip.
                6. Avoid repeating attractions.
                7. Balance busy and relaxed days.
                8. Group nearby or similar attractions on the same day whenever possible.
                9. Never invent attractions that are not present in the provided data.
                10. If there are not enough attractions for all days, leave free time suggestions instead of inventing places.

                Preference handling:

                - Adventure → prioritize adventure and outdoor attractions.
                - History → prioritize museums, monuments, and cultural sites.
                - Food → prioritize food markets and culinary experiences.
                - Shopping → prioritize shopping districts and malls.
                - Nature → prioritize parks, beaches, gardens, and scenic locations.
                - Nightlife → prioritize evening activities and entertainment districts.
                - Family → prioritize family-friendly attractions.
                - Luxury → prioritize premium experiences and high-end attractions.
                Output format:

                Destination: <destination>

                Day 1
                Morning:
                - Attraction Name
                - Why it was selected

                Afternoon:
                - Attraction Name
                - Why it was selected

                Evening:
                - Attraction Name
                - Why it was selected

                Estimated Cost:
                - Attraction 1: ...
                - Attraction 2: ...

                Day 2
                ...

                After the itinerary, provide:

                Trip Summary:
                - Total attractions planned
                - Highest-rated attraction
                - Approximate total attraction cost (if price data is available)
                - Special notes based on user preferences

                """
def activity_node(state:activity_state):

    tourist_places=serp_activity(f"top 10 tourist attractions in {state["place"]}")

    response=llm.invoke(
        [
            SystemMessage(content=activity_prompt),
            HumanMessage(
                content=(
                    f"destination: {state["place"]}"
                    f"trip length: {state["days"]}"
                    f"preferences:{state["preferences"]}"
                    f"Tourist Attraction Data:{tourist_places}"
                )
            )
        ]
    )
    return {"recommendation":response.content}
graph=StateGraph(activity_state)
graph.add_node("activity",activity_node)

graph.add_edge(START,"activity")
graph.add_edge("activity",END)

activity_agent=graph.compile()

print(activity_agent.invoke({"place":"dubai","days":4,"preferences":["swmming"]})["recommendation"])
