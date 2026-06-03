from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage
from serp_flight import serp_flight
from tavily import _tavily_search
import os 


load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


class Iata(BaseModel):
    origin_iata:str
    destination_iata:str

class flight_state(TypedDict):
    origin_place:str
    destination_place:str
    outbound_date:str
    preference:list[str]
    iata:Iata
    response:str

def IATA(state:flight_state):

    origin_place=state["origin_place"]
    destination_place=state["destination_place"]

    raw_data=_tavily_search(f"IATA Code of nearest airports of {origin_place} and {destination_place} ",4)
    system_prompt="""
                    You are a travel assistant that extracts airport information from search data.
                    Given raw search results and two place names (origin and destination), identify the nearest airport for each and return ONLY their IATA codes.
                    Respond strictly in this JSON format and nothing else:
                    {
                    "origin_iata": "XXX",
                    "destination_iata": "XXX"
                    }
                """

    response=llm.with_structured_output(Iata).invoke(
        [
            SystemMessage(
                content=(
                    system_prompt
                )
            ),
            HumanMessage(
                content=(
                    f"""
                    Origin: {origin_place}
                    Destination: {destination_place}

                    Search Results:
                    {raw_data}

                    What are the IATA codes of the nearest airports to the origin and destination?
                    """
                )
            )
        ]
    )
    print(response)
    return {"iata":response}

def flight_data(state:flight_state):
    
    flight_data=serp_flight(state["iata"].origin_iata,state["iata"].destination_iata,state["outbound_date"])
    
    print(f"\n {flight_data} \n")


    airplane_prompt = f"""
                    You are a flight recommendation assistant.

                    You are given a list of normalized flight options.

                    between a journy from origin to destintion there may or may not be layovers so multipleflights.

                    Each flight object has the following schema:

                    {{
                        "total_price": int
                            Total ticket price.

                        "total_duration": int
                            Total journey duration in minutes.

                        "layovers": int | None
                            Number of layovers.
                            None means a direct flight.

                        "ticket extensions": list[str]
                            Ticket rules and benefits 

                        "departure_airport": str
                            Origin airport name.

                        "arrival_airport": str
                            Destination airport name.

                        "departure_time": str
                            Departure datetime.

                        "arrival_time": str
                            Arrival datetime.

                        "segment": list[dict]
                            Individual flight segments that make up the journey.

                            Each segment contains:

                            {{
                                "travel_class": str
                                    Cabin class.

                                "flight_name": str
                                    Aircraft type.

                                "airline": str
                                    Airline operating this segment.

                                "flight_extension": list[str]
                                    Onboard amenities and characteristics 
                            }}
                    }}

                    User preferences may include:
                    - preferred airlines
                    - refundable tickets
                    - fewer layovers
                    - shortest duration
                    - cheapest price
                    - onboard amenities

                    Ranking Rules:
                    1. Never invent information.
                    2. Never assume a ticket is refundable unless it appears in "ticket extensions".
                    3. Never recommend the same flight twice.
                    4. Return at most 5 recommendations.
                    5. If fewer than 5 flights match the user's preferences, return only the matching flights.
                    6. Explain why each flight was selected.
                    7. If a user preference cannot be verified from the provided data, explicitly state that.

                    Flight Data:
                    {flight_data}
                """

    new_res=llm.invoke(
        [
            SystemMessage(
                content=(
                    airplane_prompt
                )
            ),
            HumanMessage(
                content=(
                    f"""
                    user_preference:{state["preference"]}
                    Recommend the top 5 most suitable flights based on my preferences and dates."""
                )
            )
        ]
    )
    return {"response":new_res.content}
graph=StateGraph(flight_state)
graph.add_node("iata",IATA)
graph.add_node("flight",flight_data)

graph.add_edge(START,"iata")
graph.add_edge("iata","flight")
graph.add_edge("flight",END)

flight_agent=graph.compile()

print(flight_agent.invoke({"origin_place":"bombay","destination_place":"dubai","outbound_date":"2026-06-08","preference":["indigo","refund"]})["response"])
