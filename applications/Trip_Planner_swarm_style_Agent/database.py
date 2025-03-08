import lancedb
import pandas as pd
from typing import List, Dict, Any
from config import DB_PATH
from models import FlightModel, HotelModel

class DatabaseManager:
    def __init__(self):
        self.db = lancedb.connect(DB_PATH)
        self.flight_table = None
        self.hotel_table = None

    def initialize_tables(self, flight_data: pd.DataFrame, hotel_data: pd.DataFrame) -> None:
        """Initialize database tables with data"""
        # Reset index for both dataframes
        flight_data = flight_data.reset_index().rename(columns={'index': 'id'})
        hotel_data = hotel_data.reset_index().rename(columns={'index': 'id'})

        # Create and populate flight table
        self.flight_table = self.db.create_table(
            "hotel_recommandation1s",
            schema=FlightModel,
            mode="overwrite"
        )
        self.flight_table.create_fts_index(
            ["Airline", "Flight_details"],
            use_tantivy=True,
            replace=True
        )
        self.flight_table.add(data=flight_data)

        # Create and populate hotel table
        self.hotel_table = self.db.create_table(
            "hotel_recommendations",
            schema=HotelModel,
            mode="overwrite"
        )
        self.hotel_table.create_fts_index(
            ["City", "all_hotel_details"],
            use_tantivy=True,
            replace=True
        )
        self.hotel_table.add(data=hotel_data)

    def search_flights(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search flights based on query"""
        if not self.flight_table:
            raise ValueError("Flight table not initialized")
        
        return self.flight_table.search(query).limit(limit).select([
            "Flight_Number",
            "Airline",
            "Origin_City",
            "Destination_City",
            "Departure_Time",
            "Price",
            "Flight_details"
        ]).to_list()

    def search_hotels(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search hotels based on query"""
        if not self.hotel_table:
            raise ValueError("Hotel table not initialized")
        
        return self.hotel_table.search(query).limit(limit).select([
            "Name",
            "City",
            "Neighborhood",
            "Stars",
            "Price_Per_Night",
            "Rating",
            "Amenities",
            "all_hotel_details"
        ]).to_list()

# Create singleton instance
db_manager = DatabaseManager() 