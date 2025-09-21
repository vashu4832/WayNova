import streamlit as st
import os
import json
import requests
import uuid
from datetime import datetime

from dotenv import load_dotenv
from amadeus import Client, ResponseError

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# -------------------------------------------------------------------
# 🔹 1. FIXED DATE (today = 21-Sept-2025)
# -------------------------------------------------------------------
TODAY_DATE = datetime(2025, 9, 21)

# -------------------------------------------------------------------
# 🔹 2. LOAD ENVIRONMENT VARIABLES
# -------------------------------------------------------------------
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
amadeus = Client(
    client_id=os.getenv("AMADEUS_CLIENT_ID"),
    client_secret=os.getenv("AMADEUS_CLIENT_SECRET")
)
google_places_api_key = os.getenv("GOOGLE_PLACES_API_KEY")
openweather_api_key = os.getenv("OPENWEATHER_API_KEY")

if not openai_api_key or not google_places_api_key or not openweather_api_key:
    st.error("Missing API Keys. Please check your .env file.")
    st.stop()

# -------------------------------------------------------------------
# 🔹 3. API FUNCTIONS
# -------------------------------------------------------------------

def search_flights(origin, destination, departure_date, adults=1, max_results=3):
    """Search flights using Amadeus API"""
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            adults=adults,
            max=max_results
        )
        flights = []
        for idx, offer in enumerate(response.data[:max_results]):
            itinerary = offer["itineraries"][0]["segments"]
            seg = itinerary[0]
            dep = seg["departure"]["at"]
            arr = seg["arrival"]["at"]
            price = offer["price"]["total"]
            flights.append(
                f"{origin} {dep} → {destination} {arr} | 💲 {price}"
            )
        return "\n".join(flights)
    except ResponseError as e:
        return f"Amadeus API error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

def find_hotels(location):
    """Find hotels using Google Places API"""
    geocode_url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={location}&inputtype=textquery&fields=geometry&key={google_places_api_key}"
    geo_resp = requests.get(geocode_url).json()
    if not geo_resp.get("candidates"):
        return "❌ Location not found."
    loc = geo_resp["candidates"][0]["geometry"]["location"]
    lat_lon = f"{loc['lat']},{loc['lng']}"

    places_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat_lon}&radius=3000&type=hotel&key={google_places_api_key}"
    places_resp = requests.get(places_url).json()

    hotels = []
    for place in places_resp.get("results", [])[:5]:
        hotels.append(
            f"{place.get('name')} ⭐ {place.get('rating','N/A')} — {place.get('vicinity','N/A')}"
        )
    return "\n".join(hotels)

def get_weather(city_name, target_date=None):
    """Get weather forecast from OpenWeather"""
    if not target_date:
        target_date = TODAY_DATE.strftime("%Y-%m-%d")

    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={openweather_api_key}&units=metric"
    resp = requests.get(url).json()

    if "list" not in resp:
        return f"Weather API error: {resp.get('message','unknown error')}"

    for entry in resp["list"]:
        dt_txt = entry["dt_txt"].split(" ")[0]
        if dt_txt == target_date:
            temp = entry["main"]["temp"]
            desc = entry["weather"][0]["description"]
            return f"🌤 {city_name} on {target_date}: {temp}°C, {desc}"

    return f"No forecast available for {city_name} on {target_date}"

def book_flight(flight_details: str):
    """
    Dummy booking for flights.
    flight_details = "BOM, GOI, 2025-09-25"
    """
    txn_id = str(uuid.uuid4())[:8]
    booking = {
        "type": "flight",
        "details": flight_details,
        "amount": "₹5000",
        "status": "confirmed",
        "transaction_id": txn_id
    }
    return f"✅ Flight booked!\nDetails: {flight_details}\nPayment ID: {txn_id}"

def book_hotel(hotel_name: str, checkin: str, checkout: str):
    """
    Dummy booking for hotels.
    """
    txn_id = str(uuid.uuid4())[:8]
    booking = {
        "type": "hotel",
        "hotel": hotel_name,
        "checkin": checkin,
        "checkout": checkout,
        "amount": "₹3000",
        "status": "confirmed",
        "transaction_id": txn_id
    }
    return f"🏨 Hotel booked!\nHotel: {hotel_name}\nStay: {checkin} → {checkout}\nPayment ID: {txn_id}"

# -------------------------------------------------------------------
# 🔹 4. DEFINE TOOLS FOR AGENT
# -------------------------------------------------------------------
tools = [
    Tool(
        name="SearchFlights",
        func=lambda s: search_flights(*[item.strip() for item in s.split(',')]),
        description="Search flights. Input: 'BOM, GOI, 2025-09-22'"
    ),
    Tool(
        name="FindHotels",
        func=find_hotels,
        description="Find hotels in a location. Input: 'Goa, India'"
    ),
    Tool(
        name="GetWeather",
        func=lambda city: get_weather(city),
        description="Get weather forecast for today (21-09-2025) or a given date. Input: 'Goa'"
    ),
    Tool(
        name="BookFlight",
        func=book_flight,
        description="Book a flight with dummy payment. Input: 'BOM, GOI, 2025-09-25'"
    ),
    Tool(
        name="BookHotel",
        func=lambda s: book_hotel(*[item.strip() for item in s.split(',')]),
        description="Book a hotel with dummy payment. Input: 'HotelName, 2025-09-25, 2025-09-30'"
    )
]

# -------------------------------------------------------------------
# 🔹 5. INITIALIZE LLM AGENT WITH MEMORY
# -------------------------------------------------------------------
def initialize_travel_agent():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=openai_api_key,
        temperature=0.7
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=10,
        return_messages=True,
        output_key="output"
    )
    chat_history_placeholder = MessagesPlaceholder(variable_name="chat_history")

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors="Sorry, I didn’t understand. Can you rephrase?",
        agent_kwargs={"extra_prompt_messages": [chat_history_placeholder]}
    )
    return agent

# -------------------------------------------------------------------
# 🔹 6. STREAMLIT UI
# -------------------------------------------------------------------
st.set_page_config(page_title="✈️ AI Travel Planner", page_icon="🌴", layout="wide")
st.title("🌴 Smart AI Travel Planner")
st.caption(f"Your AI assistant. Today’s date is fixed as {TODAY_DATE.strftime('%d-%b-%Y')}")

if "agent" not in st.session_state:
    st.session_state.agent = initialize_travel_agent()
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I am your AI Travel Planner. How can I help you?")
    ]

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant", avatar="✈️").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user", avatar="👤").write(msg.content)

if prompt := st.chat_input("Ask me about flights, hotels, or weather..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user", avatar="👤").write(prompt)

    with st.chat_message("assistant", avatar="✈️"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.messages
                })
                output = response["output"]
                st.session_state.messages.append(AIMessage(content=output))
                st.write(output)
            except Exception as e:
                error_message = f"😕 Error: {e}"
                st.error(error_message)
                st.session_state.messages.append(AIMessage(content=error_message))
