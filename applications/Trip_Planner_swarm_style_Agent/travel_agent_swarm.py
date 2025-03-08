import datetime
import pandas as pd
import os.path
import os
import asyncio
from typing import Callable, Any, Dict
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

# Set environment variables before importing torch-related libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import OPENAI_API_KEY, OPENAI_MODEL, REDIS_URL, DB_PATH
from database import db_manager
from services import search_service
from redis_langgraph import RedisSaver

# Initialize database with data only if tables don't exist
def initialize_data():
    """Initialize database with flight and hotel data if not already initialized"""
    try:
        # Check if tables exist by attempting a search
        db_manager.search_flights("test query", limit=1)
        db_manager.search_hotels("test query", limit=1)
        print("Database tables already exist, skipping initialization")
    except Exception as e:
        print("Initializing database tables...")
        try:
            flight_data = pd.read_csv("final_flight_data.csv", index_col=0)
            hotel_data = pd.read_csv("final_hotels_dataN.csv", index_col=0) 
            db_manager.initialize_tables(flight_data, hotel_data)
            print("Database initialization completed successfully")
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

# Create handoff tools
transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant that can search for and book hotels.",
)

transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant that can search for and book flights.",
)

# Define agent prompt
def make_prompt(base_system_prompt: str) -> Callable:
    def prompt(state: dict, config: RunnableConfig) -> list:
        system_prompt = (
            base_system_prompt
            + f"\nToday is: {datetime.datetime.now()}"
        )
        return [{"role": "system", "content": system_prompt}] + state["messages"]
    return prompt

# Initialize the LLM
model = ChatOpenAI(model=OPENAI_MODEL)

# Define flight assistant
flight_assistant = create_react_agent(
    model,
    [search_service.search_flights, transfer_to_hotel_assistant],
    prompt=make_prompt("""You are a flight booking assistant for domestic Indian flights only. You help users find flights within India using the search_flights tool.
    When users ask about hotels, transfer them to the hotel assistant.
    
    Important constraints and behaviors:
    1. You ONLY handle domestic flights within India
    2. You have deep knowledge of Indian geography and can identify Indian cities vs international locations
    3. For international flight requests:
       - Politely explain you only handle domestic Indian flights
       - Suggest they try a different service for international travel
    4. For non-travel queries:
       - Politely redirect them to ask about Indian domestic flights or hotels
    
    The search_flights tool returns structured data with flight details. For each flight, you'll receive:
    - Basic info (flight number, airline, origin, destination, departure time, price)
    - Detailed breakdown of the flight including route, timing, price/seats, and class information
    
    When handling flight queries:
    1. Use your knowledge to verify both cities are valid Indian destinations
    2. If you're unsure about a location being in India, ask the user for clarification
    3. Present flight information in a clear, organized manner:
       - Highlight key details like timing, price, and special features
       - Point out relevant matches to user preferences
    4. Always ask if they'd like to:
       - Proceed with booking
       - See more options
       - Try different search criteria
    
    Remember to:
    - Transfer hotel-related queries to the hotel assistant
    - Use your knowledge to validate Indian locations
    - Maintain a helpful and informative tone"""),
    name="flight_assistant",
)

# Define hotel assistant
hotel_assistant = create_react_agent(
    model,
    [search_service.search_hotels, transfer_to_flight_assistant],
    prompt=make_prompt("""You are a hotel booking assistant for hotels in India only. You help users find hotels using the search_hotels tool.
    When users ask about flights, transfer them to the flight assistant.

    Important constraints and behaviors:
    1. You ONLY handle hotels within India
    2. You have deep knowledge of Indian geography and can identify Indian cities vs international locations
    3. For international hotel requests:
       - Politely explain you only handle Indian hotels
       - Suggest they try a different service for international accommodations
    4. For non-travel queries:
       - Politely redirect them to ask about Indian hotels or flights

    When handling hotel queries:
    1. Use your knowledge to verify the requested city is in India
    2. If you're unsure about a location being in India, ask the user for clarification
    3. For valid Indian cities:
       - Search for exact matches in the requested city
       - If no hotels found, suggest nearby popular Indian cities
    4. Present hotel information clearly, highlighting:
       - Hotel name and rating
       - Location and neighborhood
       - Price per night
       - Key amenities
       - Guest ratings
    5. Filter and validate results:
       - Only show hotels in the requested city
       - Ensure all locations are within India
    6. If limited options are found:
       - Clearly state the limitation
       - Suggest alternative nearby Indian cities
    
    Always ask if they'd like to:
    - See specific hotel details
    - Search with different criteria
    - Try nearby cities
    - Proceed with booking

    Remember to:
    - Transfer flight-related queries to the flight assistant
    - Use your knowledge to validate Indian locations
    - Maintain a helpful and informative tone"""),
    name="hotel_assistant",
)

# Create and compile the swarm
class SwarmPlanner:
    def __init__(self):
        # Initialize database if needed
        initialize_data()
        
        try:
            # Initialize Redis saver
            print("Initializing Redis checkpointer...")
            self.redis_saver = RedisSaver.from_conn_url(REDIS_URL)
            self.checkpointer = self.redis_saver.__enter__()
            print("Successfully connected to Redis")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            print("Falling back to InMemorySaver")
            self.checkpointer = InMemorySaver()
        
        # Create and compile the swarm
        self.builder = create_swarm(
            [flight_assistant, hotel_assistant],
            default_active_agent="flight_assistant"
        )
        self.planner = self.builder.compile(checkpointer=self.checkpointer)
    
    def stream(self, input_data: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Any:
        """Stream responses from the planner"""
        try:
            return self.planner.stream(input_data, config, **kwargs)
        except Exception as e:
            print(f"Error in streaming: {e}")
            raise
    
    def __del__(self):
        # Clean up Redis connection if it was used
        if hasattr(self, 'redis_saver'):
            try:
                self.redis_saver.__exit__(None, None, None)
            except:
                pass

# Create the planner instance
def create_planner():
    try:
        planner_instance = SwarmPlanner()
        return planner_instance.planner
    except Exception as e:
        print(f"Error creating planner: {e}")
        raise

planner_instance = SwarmPlanner()
app_planner = planner_instance.planner

if __name__ == "__main__":    
    try:
        # Example usage
        config = {"configurable": {"thread_id": "500", "user_id": "2"}}
        result = app_planner.invoke({
            "messages": [{
                "role": "user",
                "content": "please help me with nearby mumbai hotels"
            }]
        }, config)
        print("Flight Query Result:", result)
    except Exception as e:
        print(f"Error running example: {e}")


