from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage
from aviationstack import aviation
from tavily import _tavily_search
import os

#=========================================MODEL=================================================================================
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


class input_schema(BaseModel):
    origin: str
    destination:str
    departure_date:str
    return_date: str
    travelers:Optional[int]
    budget:Optional[str]
    hotel_preference:Optional[str]
    intrests: list[str]
    currency: Optional[str]

class flights(BaseModel):
    flight_date:str
    departure_time:str
    arrival_time:str
    departure_airport:str
    arrival_airport:str
    airline:str

class Recommended(BaseModel):

    budget_analysis:str
    flight_list:list[flights]
    justifications:str

class state(TypedDict):
    user_input:str
    input:input_schema
    flight_info:Recommended
    hotel_info:list[dict]
    query:str


router_prompt = """
                You are a travel planning assistant. Your job is to extract and structure 
                travel information from the user's natural language input into a strict JSON format matching the schema.

                ### EXTRACTION & TRANSLATION RULES:

                1. **origin (CRITICAL)**: Identify the city the user is traveling FROM, and translate it into its official **3-letter IATA airport code** in UPPERCASE. 
                - *Example*: "Mumbai" -> "BOM", "Delhi" -> "DEL", "New York" -> "JFK".

                2. **destination (CRITICAL)**: Identify the city the user is traveling TO, and translate it into its official **3-letter IATA airport code** in UPPERCASE.
                - *Example*: "Dubai" -> "DXB", "London" -> "LHR", "Singapore" -> "SIN".

                3. **departure_date**: Convert to YYYY-MM-DD format.

                4. **return_date**: Convert to YYYY-MM-DD format.
                - If the user specifies a duration (e.g., "5 days" or "1 week"), calculate the return date by adding those days to the departure_date.

                5. **travelers**: Integer count of people.
                - "solo" / "myself" -> 1
                - "couple" / "with my wife" / "me and my husband" -> 2
                - Default to 1 if completely unmentioned.

                6. **budget**: Keep as a readable descriptive string combining the amount and currency.
                - *Example*: "3 lakhs" -> "300000 INR", "under $4000" -> "under 4000 USD".

                7. **hotel_preference**: Map descriptive words to star ratings:
                - "cheap/budget" -> "2-star"
                - "decent/moderate" -> "3-star"
                - "good/comfortable" -> "4-star"
                - "luxury/premium/five star" -> "5-star"
                - Default to "3-star" if unmentioned.

                8. **interests**: Always a list of strings representing activities. Default to ["sightseeing"] if empty.

                9. **currency**: Standard 3-letter currency code based on user location/budget mention. Default to "INR".

                ### STRICT RULES:
                - Never leave any field null or empty. Use defaults if information is missing.
                - Do not output anything except the structured schema data.

                Now extract, translate, and structure the following user input:
                {user_message}
                """
flight_agent_template = """
                You are an Elite AI Travel Concierge. Your task is to analyze the user's trip details and a list of available flights for that trip, then recommend the absolute three best departure  flights based on comfort, schedule, and budget constraints.

                ### 1. CONTEXT & BUDGET ALLOCATION LOGIC
                The user has provided a `budget` parameter. CRITICAL: This is their TOTAL trip budget (including hotels, food, transport, and activities), NOT just their flight budget. 
                Before selecting flights, you must:
                - Analyze the total budget and the number of `travelers`.
                - Logically estimate a reasonable flight allowance. (Rule of thumb: deprature trip flights typically consume 15% to 20% of a total trip budget, depending on whether it is a luxury or budget trip). 
                - Ensure the  cost of the departing flights for ALL travelers fits within this estimated flight allowance.

                ### 2. SELECTION CRITERIA
                Evaluate the provided flight data based on:
                - **Budget Fit:** Does it align with the estimated flight allowance? (If actual flight prices are provided, use them. If not, use airline reputation and tier to estimate affordability).
                - **Comfort & Schedule:** Prioritize flights with reasonable `departure_time` and `arrival_time` (e.g., avoid 3:00 AM departures or red-eye flights unless it is a strict budget trip).
                - **Airline Quality:** Consider the reputation of the `airline` for comfort and reliability.
                - **Accuracy:** The flights must exactly match the user's `origin`, `destination`.

                ### 4. REQUIRED OUTPUT FORMAT
                Provide your response in the following structured format:

                1. **Budget Analysis:** Briefly explain your deduction of the flight budget. (e.g., "Total budget is $X for Y travelers. Allocating ~30% means we have roughly $Z to spend on round-trip flights.")
                2. **Recommended Outbound Flight:** Recommend top 3 flights if avilable.Provide the flight date, airline, departure airport,arrival airports,deprature time and arivl time.
                4. **Justification:** Explain exactly why you chose these 3 flights over the others, referencing comfort, timing, and how it perfectly fits the smart budget allocation.
                """
hotel_prompt="""
                You are a travel prompt engineer optimizing search queries for the Tavily Search API. Your task is to balance a traveler's Arrival Airport, Hotel Preferences, and Interests/Activities into a highly efficient search string.
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
def router(state:state):

    user_input=state["user_input"]
    response=llm.with_structured_output(input_schema).invoke(
        [
            SystemMessage(
                content=(
                    router_prompt.format(user_message=user_input)
                )
            )
        ]
    )
    return {"input":response}
def flight_agent(state:state):
    user_input=state["input"]
    flightdata=aviation(user_input.origin,user_input.destination)
    response=llm.with_structured_output(Recommended).invoke(
        [
            SystemMessage(
                content=(
                    flight_agent_template
                ),
            ),
            HumanMessage(
                content=(
                    f"my travelling detils: {user_input}\navailabe flight data {flightdata}"
                )
            )
        ]
    )
    return {"flight_info":response}
def hotel_agent(state:state):
    user_input=state["input"]
    Query=llm.invoke(
        [
            SystemMessage(
                content=(
                    hotel_prompt
                )
            ),
            HumanMessage(
                content=(
                    f"Arrival Airport : {user_input.destination}"
                    f"Hotel Preferences: {user_input. hotel_preference}"
                    f"Interests & Activities: {user_input.intrests}"
                )
            )
        ]
    )
    hotel_info=_tavily_search(Query.content,5)
    return {"hotel_info":hotel_info,"query":Query.content}

def itinerry(state:state):
    


graph=StateGraph(state)
graph.add_node("router",router)
graph.add_node("flight_agent",flight_agent)
graph.add_node("hotel_agent",hotel_agent)
graph.add_edge(START,"router")
graph.add_edge("router","flight_agent")
graph.add_edge("flight_agent","hotel_agent")

agent=graph.compile()
response=agent.invoke({"user_input":"Plan a solo trip from Mumbai to Dubai for 5 days from August 10 2025, luxury stay, budget 3 lakhs, interested in shopping and beaches"})
print("\nRESULT\n")
print(response["flight_info"])
print("\n")
print(response["hotel_info"])
print("\n")
print(response["query"])

