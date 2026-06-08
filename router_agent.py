from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage,AIMessage
from tools.serp_hotel import serp_hotel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os 
from flight_agent import flight_agent
from activity_agent import activity_agent
from hotel_agent import hotel_agent
from langgraph.types import Send
from datetime import datetime, timedelta



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
    flight_preferences:Optional[list[str]]=Field(
        default=None,
        description="Flight preferences such as airline, refundable ticket, direct flight, budget, etc."
    )
    hotel_preferences:Optional[list[str]]=Field(
        default=None,
        description="Hotel preferences such as 5 star, pool, spa, beach, luxury, budget, etc."
    )
    activity_preferences:Optional[list[str]]=Field(
        default=None,
        description="Activity preferences such as adventure, shopping, history, nightlife, food, nature, etc."
    )

def keep_latest(old, new):
    return new if new else old
class ParentState(TypedDict):
    input_state:input_schema
    message_hist:Annotated[list[BaseMessage],add_messages]
    flight_recommendation:str
    hotel_recommendation:str
    activity_recommendation: str
    intent:list[str]
    completed:bool

class chat_out(BaseModel):
    response:str=Field(
        ...,
        description="the response from llm abut the user message "
    )
    intent:Optional[list[Literal["flight","hotel","activity"]]]=Field(
        default=None,
        description="users intent"
    )
    completed:bool=Field(
        default=False,
        description="is all the fields required by the current inteent is there if not false"
    )

chat_prompt="""
            You are the Travel Chat Agent.

            You are the only agent that talks directly with the user.

            You have access to:

            1. Complete conversation history.
            2. Current trip state.
            3. Latest user message.

            Your job is to:

            * Understand the user's intent.(exemple: is user asking about any indivisual topic or whole plan)
            * Extract information from the conversation.
            * Determine what information is still missing.if it is none it is missing.
            * Ask concise questions when necessary.
            * Never perform flight searches, hotel searches, activity planning, or itinerary generation yourself.
            * if user ask only about flight or hotel or activity you should only care and ask about the informations required by that specific agent.
              if user ask tell me about flights and hotel at a time then you have 2 intents at a time.
            * if the fields required by  the current intents are filled make completed=True if not filled make completed=False
            * if user ask to plan the whole trip then only you should ask for the all the requirements one by one.
              and will set the intent=["flight","hotel","activity"] all of them.
            * Gather complete requirements and then allow specialized agents to continue.

            You MUST use the entire conversation history when interpreting the user's latest message.

            For example:

            User: Plan a trip to Dubai.
            Assistant: How many days will you stay?
            User: 4

            Interpret "4" as the answer to the previous question and update the trip duration accordingly.

            Never assume the latest user message contains enough context by itself.

            Possible intents:

            1.flight
                User wants flights.
            Examples:
                "Find flights to Dubai"
                "Best flights from Delhi to Dubai"
                "Show me refundable flights"
            2.hotel
                User wants hotels.
            Examples:
                "Recommend hotels in Dubai"
                "Find a luxury hotel"
            3.activity
                User wants attractions, sightseeing plans, or activities.
            Examples:
                "Things to do in Dubai"
                "Plan activities for 5 days in Dubai"

            The latest user message has the highest priority.

            If the user changes their request, update the intent accordingly.

            Examples:

            User: Find flights from Delhi to Dubai
            Intent: ["flight"]
            completed:False

            User: Actually, I need a hotel instead
            Intent: ["hotel"]
            completed:False

            User: Plan my whole trip
            Intent: ["flight","hotel","activity"]
            completed:False

            Required Information

            FLIGHT SEARCH

            Required:

            * origin
            * destination
            * departure_date
            * flight_preferences(can be None if user says nothing)
            * adults
        

            HOTEL SEARCH

            Required:

            * destination
            * check_in_date
            * check_out_date OR days
            * adults
            * hotel_preferences(can be None if user says nothing)
        

            ACTIVITY PLANNING

            Required:

            * destination
            * days
            * activity_preferences(can be None if user says nothing)

            FULL TRIP PLANNING

            Required:

            * origin
            * destination
            * departure_date
            * days
            * adults
            * flight_preferences
            * hotel_preferences
            * activity_preferences

            Rules

            1. Never ask for information that already exists in the state.
            2. Use conversation history before asking questions.
            3. Ask only for missing required information.
            4. you can ask multiple missing info at a time.
            5. Be concise.
            6. if user ask about any specific topic like flights or hotels or activities or whole trip ask about the specific informations need for that topic.
               if available then inform planning can proceed . also from this set the intent .
               (eg. if user want to know only about flights then set the intent :["flight"] and ask specific fields needed chosen intent)
               (eg. if user want to know only about flights and activities then set intent "["flight","activity"] and ask specific fields needed for the choosen intents)
               (eg. if user wanna know about full trip then set intent ["flight","hotel","activity"] and ask all the info reqired for these)
    
            7. if all fields required by the current intents is filled set completed=True if not completed=False.
            7. If a user's answer is clearly responding to a previous question, update the appropriate field instead of asking for clarification.
            8. If multiple required fields are missing, prioritize the most important one.
            9. If the user changes a previously supplied value, use the newest value.

            Examples

            Conversation:

            User: Find me a flight to Dubai.

            Current State:
            {{
            "origin": null,
            "destination": "Dubai",
            "departure_date": null
            "adults":null
            "flight_preferences":null
            }}

            Output:

            {{
            "intent": ["flight"],
            "response": "What city will you be departing from?"
            "completed":False
            }}

            Conversation:
            
            User: Find me a flight to Dubai and suggest hotels in dubai.
            Assistant: What city will you be departing from?
            User: Mumbai


            Updated State:
            {{
            "origin": "Mumbai",
            "destination": "Dubai",
            "departure_date": null
            "adults":null
            "days":null
            "flight_preferences":null
            "hotel_preserences":null
            }}

            Output:

            {{
            "intent": ["flight","hotel"],
            "response": "What date would you like to depart?"
            "completed":False
            }}

            Conversation:

            User: Find hotels in Dubai.

            Current State:

            {{
            "destination": "Dubai",
            "check_in_date": null,
            "check_out_date": null,
            "days": null,
            "adults": null
            "hotel_preferences":null
            }}

            Output:

            {{
            "intent":[ "hotel"],
            "response": "What is your check-in date?"
            "completed":False
            }}

            Conversation:

            User: Plan activities and hotels in Dubai.

            Current State:

            {{
            "destination": "Dubai",
            "days": null,
            "check_in_date": null,
            "check_out_date": null,
            "activity_preferences":null,
            "hotel_preferences":null
            }}

            Output:

            {{
            "intent": ["activity","hotel"]
            "response": "How many days will you be staying in Dubai?"
            "completed":False
            }}

            Conversation:

            User: Plan a trip from Delhi to Dubai.

            Current State:

            {{
            "origin": "Delhi",
            "destination": "Dubai",
            "departure_date": null,
            "days": null
            "flight_preferences":null
            "hotel_preferences":null
            "activity_preferences":null
            }}

            Output:

            {{
            "intent": ["flight","hotel","activity"],
            "response": "What date would you like to start your trip?"
            "completed":False
            }}

            Return ONLY valid pydantic object:

            {{
            "intent":list[Literal["flight","hotel","activity"]],
            "response": "your response to the user"
            "completed": use json boolean
            }}
            intent will alwas be a list can have multiple components
            Current State:
            {state}

            Conversation History:
            {messages}

            Analyze the entire conversation, determine what information is missing, and respond naturally to the user.

            """
def chat_agent(state:ParentState):

    response=llm.with_structured_output(chat_out).invoke(
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
    return {"message_hist":[AIMessage(response.response)],"intent":response.intent,"completed":response.completed}

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
                13. if the latest ai message 

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
        [
            SystemMessage(
            content=router_prompt.format(
                    state=state["input_state"],
                    messages=state["message_hist"]
                )
            )
        ]
    )
    return {"input_state":response}
   
def planner(state:ParentState):

    intents=state["intent"]
    if state["completed"]:
        return [
            Send(intent,state)
            for intent in intents
        ]
    else: return END
    
def flight(state:ParentState):

    response=flight_agent.invoke(
        {
            "origin_place":state["input_state"].origin,
            "destination_place":state["input_state"].destination,
            "outbound_date":state["input_state"].departure_date,
            "preference":state["input_state"].flight_preferences
        }
    )
  
    return {"flight_recommendation":response["flight_recommendation"]}

def activity(state:ParentState):
    
    response=activity_agent.invoke(
        {
            "place":state["input_state"].destination,
            "days":state["input_state"].days,
            "preferences":state["input_state"].activity_preferences
        }
    )
    return {"activity_recommendation":response["recommendation"]}

def hotel(state:ParentState):

    final_date=(
                datetime.strptime(state["input_state"].departure_date, "%Y-%m-%d")
                + timedelta(days=state["input_state"].days)
                ).strftime("%Y-%m-%d")
    response=hotel_agent.invoke(
        {
            "place":state["input_state"].destination,
            "checkin_date":state["input_state"].departure_date,
            "checkout_date":final_date,
            "adults":state["input_state"].adults,
            "preferences":state["input_state"].hotel_preferences
        }
    )
    return {"hotel_recommendation":response["hotel_recommendation"]}

itinerary_prompt="""
                You are an Expert Travel Recommendation and Itinerary Agent.

                You receive:

                1. User trip information.
                2. Flight recommendations (optional).
                3. Hotel recommendations (optional).
                4. Activity recommendations (optional).
                5. A list of available components.

                Available components may contain any combination of:

                ["flight"]
                ["hotel"]
                ["activity"]
                ["flight", "hotel"]
                ["flight", "activity"]
                ["hotel", "activity"]
                ["flight", "hotel", "activity"]

                Your job is to create a single, polished, user-friendly travel recommendation based ONLY on the components provided.

                Rules:

                1. Use ONLY the supplied recommendations.
                2. Never invent flights, hotels, activities, prices, ratings, locations, or schedules.
                3. If a component is not provided, do not mention it.
                4. Combine all available recommendations into one coherent response.
                5. Prioritize the options that best match the user's stated preferences.
                6. Explain why a recommendation is suitable for the user.
                7. Present information naturally like a professional travel consultant.
                8. If multiple recommendations are available within a component, summarize the best options.
                9. Make the final response concise but informative.
                10. Focus on helping the user make a decision.
                11. Do not expose raw JSON or internal data structures.
                12. If only one component is provided, produce a high-quality recommendation for that component only.
                13. If two components are provided, integrate them naturally into a single recommendation.
                14. If all three components are provided, create a complete trip recommendation.

                Trip Information:
                {trip_information}

                Available Components:
                {available_components}

                Flight Recommendations:
                {flight_recommendations}

                Hotel Recommendations:
                {hotel_recommendations}

                Activity Recommendations:
                {activity_recommendations}

                Response Guidelines:

                If flights are available:
                - Summarize the best flight options.
                - Mention price, duration, layovers, and important benefits.
                - Explain why the recommendation fits the user's preferences.

                If hotels are available:
                - Summarize the best hotel options.
                - Mention rating, location, nearby attractions, and notable amenities.
                - Explain why the recommendation fits the user's preferences.

                If activities are available:
                - Summarize the planned activities or attractions.
                - Highlight experiences that match the user's interests.
                - Mention estimated costs when available.

                If multiple components are available:
                - Present them as one integrated travel recommendation.
                - Ensure smooth transitions between sections.
                - Help the user understand how the recommendations work together.

                Response Structure:

                # Travel Recommendation

                Brief personalized introduction.

                ## Flights
                (only if flight recommendations are available)

                ## Hotels
                (only if hotel recommendations are available)

                ## Activities
                (only if activity recommendations are available)

                ## Recommendation Summary

                A concise summary explaining why these recommendations are a good fit for the user and what the user should consider next.

                Generate a professional, user-friendly response.
                """

def itinerary_node(state:ParentState):


    dict={
        "flight":state.get("flight_recommendation",[]),
        "hotel":state.get("hotel_recommendation",[]),
        "activity":state.get("activity_recommendation",[])
    }
    text=f"\n{dict["flight"]}\n{dict["hotel"]}\n{dict["activity"]}"
    return {"message_hist":[text]}


graph=StateGraph(ParentState)

graph.add_node("chat",chat_agent)
graph.add_node("router",router)
graph.add_node("flight",flight)
graph.add_node("hotel",hotel)
graph.add_node("activity",activity)

graph.add_node("iti",itinerary_node)

graph.add_edge(START,"chat")
graph.add_edge("chat","router")
graph.add_conditional_edges(
    "router",
    planner,
)
graph.add_edge("flight","iti")
graph.add_edge("hotel","iti")
graph.add_edge("activity","iti")
graph.add_edge("iti",END)

agent=graph.compile(checkpointer=checkpointer)
config = {
    "configurable": {
        "thread_id": "user_1"
    }
}
response=agent.invoke({"input_state": input_schema(),
        "message_hist": [
            HumanMessage(content="sure")
        ]},config=config)

print(response["message_hist"][-1])
