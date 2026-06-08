from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage
from tools.serp_flight import serp_flight
from tools.tavily import _tavily_search
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
    flight_recommendation: str
    flight_lists:list[dict]

system_prompt="""
                    You are a travel assistant that extracts airport information from search data.
                    Given raw search results and two place names (origin and destination), identify the nearest airport for each and return ONLY their IATA codes.
                    Respond strictly in this JSON format and nothing else:
                    {
                    "origin_iata": "XXX",
                    "destination_iata": "XXX"
                    }
                """

def IATA(state:flight_state):

    origin_place=state["origin_place"]
    destination_place=state["destination_place"]

    raw_data=_tavily_search(f"IATA Code of nearest airports of {origin_place} and {destination_place} ",4)
 
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

def flight_list(state:flight_state):

    flight_data=serp_flight(state["iata"].origin_iata,state["iata"].destination_iata,state["outbound_date"])
    return {"flight_lists":flight_data}

airplane_prompt = """
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
                    4. Return at most 5 recommendations, if user ask more recommendation give accordingly.
                    5. If fewer than 5 flights strongly match the user's preferences,
                       fill the remaining recommendations with flights that best approximate the user's preferences and clearly note any trade-offs.
                    6. Explain why each flight was selected.
                    7. If a user preference cannot be verified from the provided data, explicitly state that.
                    8. Make the response easy for travelers to understand. Clearly explain the full journey, 
                       including layovers and individual flight segments, using natural language. Emphasize the most relevant details for the user's preferences and summarize any important trade-offs (e.g., lower price vs. longer duration, direct flight vs. layovers).
                    Flight Data:
                    {flight_data}
                """

def flight_data(state:flight_state):
    

    new_res=llm.invoke(
        [
            SystemMessage(
                content=(
                    airplane_prompt.format(
                        flight_data=state["flight_lists"]
                    )
                )
            ),
            HumanMessage(
                content=(
                    f"""
                    user_preference:{state["preference"]}
                    """
                )
            )
        ]
    )
    return {"flight_recommendation":new_res.content}

def condition(state:flight_state):
    if state["is_needed"]:
        return "flight"
    else: return END

graph=StateGraph(flight_state)
graph.add_node("iata",IATA)
graph.add_node("list",flight_list)
graph.add_node("flight",flight_data)

graph.add_edge(START,"iata")
graph.add_edge("iata","list")
graph.add_edge("list","flight")
graph.add_edge("flight",END)

flight_agent=graph.compile()

#print(flight_agent.invoke({"origin_place":"BOM","destination_place":"DEL","outbound_date":'2026-10-07',"preference":["indigo","refund"]})["flight_recommendation"])
