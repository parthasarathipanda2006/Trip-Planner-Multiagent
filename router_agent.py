from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage,AIMessage
from serp_hotel import serp_hotel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os 
from flight_agent import flight_agent
from activity_agent import activity_agent
from hotel_agent import hotel_agent

load_dotenv()


conn=sqlite3.connect(database="chatbot.db",check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
class input_schema(BaseModel):
    origin:Optional[str]=Field(
        default=None,
        description="Origin city or town from where the user starts the trip"
    )
    destination:Optional[str]=Field(
        default=None,
        description="Destination city or place the user wants to visit"
    )
    departure_date:Optional[str]=Field(
        default=None,
        description="Trip start date in YYYY-MM-DD format"
    )
    days:Optional[int]=Field(
        default=None,
        description="Total trip duration in days"
    )
    adults:Optional[int]=Field(
        default=1,
        description="no of adults going to trip"
    )
    flight_preference:Optional[list[str]]=Field(
        default=None,
        description="Flight preferences such as airline, refundable ticket, direct flight, budget, etc."
    )
    hotel_preference:Optional[list[str]]=Field(
        default=None,
        description="Hotel preferences such as 5 star, pool, spa, beach, luxury, budget, etc."
    )
    activity_preferences:Optional[list[str]]=Field(
        default=None,
        description="Activity preferences such as adventure, shopping, history, nightlife, food, nature, etc."
    )



class ParentState(TypedDict):
    input_state:input_schema
    message_hist:Annotated[list[BaseMessage],add_messages]
    flight_recommendation:list[str]
    hotel_recommendation:list[str]
    activity_recommendation:list[str]

chat_prompt="""
            You are the Travel Chat Agent.

            You are the only agent that talks directly with the user.

            You have access to:

            1. Complete conversation history.
            2. Current trip state.
            3. Latest user message.

            Your job is to:

            * Understand the user's intent.
            * Extract information from the conversation.
            * Determine what information is still missing.if it is none it is missing.
            * Ask concise follow-up questions when necessary.
            * Never perform flight searches, hotel searches, activity planning, or itinerary generation yourself.
            * Gather complete requirements and then allow specialized agents to continue.

            You MUST use the entire conversation history when interpreting the user's latest message.

            For example:

            User: Plan a trip to Dubai.
            Assistant: How many days will you stay?
            User: 4

            Interpret "4" as the answer to the previous question and update the trip duration accordingly.

            Never assume the latest user message contains enough context by itself.

            Required Information

            FLIGHT SEARCH

            Required:

            * origin
            * destination
            * departure_date

            Optional:

            * adults
            * flight_preferences

            HOTEL SEARCH

            Required:

            * destination
            * check_in_date
            * check_out_date OR days
            * adults

            Optional:

            * hotel_preferences

            ACTIVITY PLANNING

            Required:

            * destination
            * days

            Optional:

            * activity_preferences

            FULL TRIP PLANNING

            Required:

            * origin
            * destination
            * departure_date
            * days

            Optional:

            * adults
            * flight_preferences
            * hotel_preferences
            * activity_preferences

            Rules

            1. Never ask for information that already exists in the state.
            2. Use conversation history before asking questions.
            3. Ask only for missing required information.
            4. Ask one question at a time whenever possible.
            5. Be concise.
            6. If all required information is available, inform the user that planning can proceed.
            7. Do not ask for optional preferences unless all required information has already been collected.
            8. If a user's answer is clearly responding to a previous question, update the appropriate field instead of asking for clarification.
            9. If multiple required fields are missing, prioritize the most important one.
            10. If the user changes a previously supplied value, use the newest value.

            Examples

            Conversation:

            User: Find me a flight to Dubai.

            Current State:
            {{
            "origin": null,
            "destination": "Dubai",
            "departure_date": null
            }}

            Response:
            What city will you be departing from?

            Conversation:
            
            User: Find me a flight to Dubai.
            Assistant: What city will you be departing from?
            User: Mumbai

            Updated State:
            {{
            "origin": "Mumbai",
            "destination": "Dubai",
            "departure_date": null
            }}

            Response:
            What date would you like to depart?

            Conversation:

            User: Plan a trip from Mumbai to Dubai.
            Assistant: How many days would you like to stay?
            User: 4

            Updated State:
            {{
            "origin": "Mumbai",
            "destination": "Dubai",
            "days": 4
            }}

            Response:
            What date would you like to start your trip?

            Conversation:

            User: Find hotels in Dubai.
            Assistant: What is your check-in date?
            User: June 8, 2026
            Assistant: How many days will you stay?
            User: 5

            Updated State:
            {{
            "destination": "Dubai",
            "check_in_date": "2026-06-08",
            "days":5
            }}

            Response:
            How many adults will be staying?

            Current State:
            {state}

            Conversation History:
            {messages}

 
            Analyze the entire conversation, determine what information is missing, and respond naturally to the user.

            """
def chat_agent(state:ParentState):

    response=llm.invoke(
            [
                SystemMessage(
                    content=chat_prompt.format(
                            state=state["input_state"],
                            messages=state["message_hist"]
                        )
                ),
                HumanMessage(
                    content=(
                        f"{state["message_hist"][-1]}"
                    )
                )
            ]
        )
    return {"message_hist":[AIMessage(response.content)]}

router_prompt="""
                You are a Travel Information Extraction Agent.

                You are NOT a chatbot.

                Your job is to analyze the conversation history and the current trip state, then update the trip state with any information that can be confidently extracted.

                Rules:

                1. Read the entire conversation history.
                2. Use the latest user message in the context of the previous conversation.
                3. Extract travel information if it is clearly provided.
                4. Update only the fields that can be confidently determined.
                5. Never overwrite existing fields with null.
                6. Never remove existing information.
                7. If the user provides a newer value for a field, replace the old value.
                8. If a field cannot be determined, keep its current value.
                9. Return the complete input_state with updated values or newly added.
                10. Do not ask questions.
                11. Do not explain your reasoning.
                12. Do not generate conversational responses.

                Possible fields:

                {{
                "origin": str ,
                "destination": str,
                "departure_date": str,
                "days": int ,
                "adults": int ,
                "flight_preference": list[str] ,
                "hotel_preference": list[str] ,
                "activity_preference": list[str]
                }}

                Current State:

                {state}

                Conversation History:

                {messages}

                Return the complete state .
                """
def router(state:ParentState):

    response=llm.with_structured_output(input_schema).invoke(
        [SystemMessage(
            content=router_prompt.format(
                    state=state["input_state"],
                    messages=state["message_hist"]
                )
            )
        ]
        )
    return {"input_state":response}
   

def flight(state:ParentState):

    response=flight_agent.invoke(
        {
            "origin_place":state["input_state"].origin,
            "destination_place":state["input_state"].destination,
            "outbound_date":state["input_state"].departure_date,
            "preference":state["input_state"].flight_preference
        }
    )
    return {"flight_recommendation":response}

def activity(state:ParentState):
    
    response=activity_agent.invoke(
        {
            "place":state["input_state"].destination,
            "days":state["input_state"].days,
            "preferences":state["input_state"].activity_preferences
        }
    )
    return {"activity_recommendation":response}

def hotel_agent(state:ParentState):
    response=hotel_agent.invoke(
        {
            "place":state["input_state"].destination,
            "checkin_date":state["input_state"].departure_date,
            "checkout_date":"2026-06-15",
            "adults":state["input_state"].days,
            "preferences":state["input_state"].hotel_preference
        }
    )
    return {"hotel_recommendation":response}

graph=StateGraph(ParentState)

graph.add_node("chat",chat_agent)
graph.add_node("router",router)

graph.add_edge(START,"chat")
graph.add_edge("chat","router")
graph.add_edge("router",END)

agent=graph.compile(checkpointer=checkpointer)
config = {
    "configurable": {
        "thread_id": "user_1"
    }
}
print(agent.invoke({"input_state": input_schema(),
        "message_hist": [
            HumanMessage(content="i will prefer 5 star hotel and economy class flight")
        ]},config=config))




