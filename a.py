from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage
#from aviationstack import aviation
import os
from langchain_tavily import TavilySearch

#=========================================MODEL=================================================================================
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
hotel_prompt="""
                You are a travel prompt engineer optimizing search queries for the Tavily Search API. Your task is to balance a traveler's Arrival Airport, Hotel Preferences, and Interests/Activities into a highly efficient search string.
                your prompt shuld be very context rich for tavily search.                            
                ---
                 user inputs will be:
                - Arrival Airport,
                - Hotel Preferences,
                - Interests & Activities
                ---
                ### BALANCING SEARCH PRIORITIES:
                - **Primary Focus:** Target areas that match the user's **Hotel Preferences** and offer easy access to their **Interests & Activities**. 
                - **Secondary Focus:** Keep the **Arrival Airport** in mind as a secondary factor. The hotel can be distant from the airport if it means a better match for activities, but the location should still remain practically accessible from the airport.
                ---
                ### INSTRUCTIONS FOR GENERATING THE TAVILY QUERY:
                1. **Location Targeting:** Identify the best neighborhoods that blend the user's activities and hotel preferences, while keeping an eye on reasonable transit or distance from the arrival airport.
                2. **Keyword Optimization:** Combine core keywords from the hotel preferences (e.g., "boutique," "beachfront," "historic") and activities (e.g., "near museums," "nightlife district").
                3. **Tavily Syntax Efficiency:** Omit conversational filler. Create a clean, dense, high-intent search string, incorporating the current year "2026" for up-to-date results.
                ---

                ### OUTPUT FORMAT:
                return just a sting containg the Query.

            """
Query=llm.invoke(
        [
            SystemMessage(
                content=(
                    hotel_prompt
                )
            ),
            HumanMessage(
                content=(
                    f"Arrival Airport :Narita International Airport, Tokyo "
                    f"Hotel Preferences:Traditional Ryokan style or quiet boutique hotel, budget around $200-$300/night, features an onsen (hot spring) or luxury bath "
                    f"Interests & Activities:Exploring historic temples, traditional tea ceremonies, and trying local street food markets. "
                )
            )
        ]
    )
tool = TavilySearch(max_results=5)
results = tool.invoke({"query": Query.content})
print(results["results"])
for result in results["results"]:
    print(result)
    print("\n")
