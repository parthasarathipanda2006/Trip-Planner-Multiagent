from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

class Trip_profile(BaseModel):
    origin: str
    destination:str
    departure_date:str
    return_date: str
    travelers:Optional[int]
    budget:Optional[str]
    hotel_preference:Optional[str]
    intrests: list[str]
    currency: Optional[str]
class Preferences(BaseModel):
    hotel_preference:list[str]
    intrests:list[str]
    dietery_preference:list[str]
class Trip_plan(BaseModel):
    flight_recommendations:list
    selected_flight:str
    hotel_recommendations:list
    selected_hotel:str
    activity_recommendations:list[str]
    selected_activities:list[str]
class Planner(BaseModel):
    next_action:str
    missing_info:str

class STATE(TypedDict):
    trip_profile:Trip_profile
    preferences:Preferences
    trip_plan:Trip_plan
    planner:Planner

def planner_agent(state:STATE):
    pass
def flight_agent(state:STATE):
    pass
def hotel_agent(state:STATE):
    pass
def activity_agent(state:STATE);
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
