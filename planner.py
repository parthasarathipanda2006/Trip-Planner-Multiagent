from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage
from langgraph.graph.message import add_messages
import os
from prompt_file import (
    router_prompt,
    planner_prompt,
    flight_prompt,
    hotel_prompt,
    activity_prompt
)

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


class Trip_profile(TypedDict):
    origin: str
    destination:str
    departure_date:str
    return_date: str
    travelers:Optional[int]
    budget:Optional[str]
    hotel_preference:Optional[str]
    intrests: list[str]
    currency: Optional[str]

class Trip_plan(TypedDict):
    flight_recommendations:list
    selected_flight:str
    hotel_recommendations:list
    selected_hotel:str
    activity_recommendations:list[str]
    selected_activities:list[str]

class Planner(TypedDict):
    next_action:str
    missing_info:str

class STATE(TypedDict):
    message_history:Annotated[list,add_messages]
    trip_profile:Trip_profile
    trip_plan:Trip_plan
    planner:Planner


def router(state:STATE):
    message=state["message_history"]
    recent_messages=message[-1].content
    response=llm.with_structured_output(Trip_profile).invoke(
        [
            SystemMessage(
                content=(router_prompt)
            ),
            {"role":"user","message":recent_messages}
        ]
    )
    return {"trip_profile":response}

def planner_agent(state:STATE):
    messages=state["message_history"]

def flight_agent(state:STATE):
    pass
def hotel_agent(state:STATE):
    pass
def activity_agent(state:STATE):
    pass
def router(state:STATE):
    action=state["planner"].next_action
    if action == "CALL_FLIGHT_AGENT":
        return "flight_agent"

    elif action == "CALL_HOTEL_AGENT":
        return "hotel_agent"

    elif action == "CALL_ACTIVITY_AGENT":
        return "activity_agent"
    
    elif action == "ASK_USER":
        return END
graph=StateGraph(STATE)

graph.add_node("planner",planner_agent)
graph.add_node("flight",flight_agent)
graph.add_node("hotel",hotel_agent)
graph.add_node("activity",activity_agent)

graph.add_edge(START,"planner")
graph.add_conditional_edges(
    "planner",
    router,
    {
        "flight": "flight_agent",
        "hotel": "hotel_agent",
        "activity": "activity_agent",
        "ask_user": END
    }
)
