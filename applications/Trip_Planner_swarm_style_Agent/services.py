from typing import Dict, Any, List
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, SEARCH_LIMIT
from database import db_manager
from models import FlightSearchParams, HotelSearchParams
import ast

class SearchService:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _validate_travel_query(self, query: str) -> bool:
        """Validate if the query is travel-related"""
        travel_keywords = {'flight', 'hotel', 'book', 'travel', 'stay', 'trip', 
                         'accommodation', 'room', 'airport', 'destination', 
                         'journey', 'vacation', 'holiday', 'tour', 'booking'}
        query_words = set(query.lower().split())
        return bool(query_words.intersection(travel_keywords))

    def _format_flight_details(self, flight: Dict[str, Any]) -> Dict[str, Any]:
        """Format flight details for presentation"""
        try:
            details = flight['Flight_details'].split('|')
            return {
                "flight_number": flight['Flight_Number'],
                "airline": flight.get('Airline', 'N/A'),
                "origin": flight.get('Origin_City', 'N/A'),
                "destination": flight.get('Destination_City', 'N/A'),
                "departure": flight.get('Departure_Time', 'N/A'),
                "price": flight.get('Price', 'N/A'),
                "details": {
                    "flight_info": details[0].strip(),
                    "route": details[1].strip(),
                    "timing": details[2].strip(),
                    "price_seats": details[3].strip(),
                    "class": details[4].strip() if len(details) > 4 else "N/A"
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "raw_details": flight.get('Flight_details', 'N/A')
            }

    def _format_hotel_details(self, hotel: Dict[str, Any]) -> Dict[str, Any]:
        """Format hotel details for presentation"""
        return {
            "name": hotel["Name"],
            "city": hotel["City"],
            "neighborhood": hotel["Neighborhood"],
            "stars": hotel["Stars"],
            "price_per_night": hotel["Price_Per_Night"],
            "rating": hotel["Rating"],
            "amenities": hotel["Amenities"],
            "details": hotel["all_hotel_details"]
        }

    def search_flights(self, query: str) -> Dict[str, Any]:
        """Search flights using LanceDB with improved formatting"""
        try:
            # First validate if query is travel-related
            if not self._validate_travel_query(query):
                return {
                    "status": "error",
                    "error": "Please ask questions related to flight or hotel bookings.",
                    "total_results": 0,
                    "flights": []
                }

            # Get structured parameters from GPT
            completion = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"""Query String: {query}

            Now from the query string above extract these following entities for flight search:
            1. Origin City/Airport: Extract the departure city or airport
            2. Destination City/Airport: Extract the destination city or airport
            3. Travel Date/Time (if mentioned): Extract any date or time preferences
            4. Class Preference (if mentioned): Extract class preference (Economy, Business, etc.)
            5. Airline Preference (if mentioned): Extract specific airline if mentioned
            6. Price Range (if mentioned): Extract any price preferences or budget constraints
            7. Other Preferences: Extract any other specific requirements

            Return ONLY a Python dictionary with these keys. If any value is not mentioned in the query, set it as None. Do not include any additional text or explanation."""
                            }]
            )
            
            # Parse the GPT response
            try:
                params = ast.literal_eval(completion.choices[0].message.content)
            except:
                params = {"query": query}
            
            # Search flights using the parameters
            results = db_manager.search_flights(str(params), SEARCH_LIMIT)
            
            # Format the results
            return {
                "status": "success",
                "total_results": len(results),
                "flights": [self._format_flight_details(flight) for flight in results]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "total_results": 0,
                "flights": []
            }

    def search_hotels(self, query: str) -> Dict[str, Any]:
        """Search hotels using LanceDB with improved formatting"""
        try:
            # First validate if query is travel-related
            if not self._validate_travel_query(query):
                return {
                    "status": "error",
                    "error": "Please ask questions related to flight or hotel bookings.",
                    "total_results": 0,
                    "hotels": []
                }

            # Get structured parameters from GPT
            completion = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"""Query String: {query}

            Now from the query string above extract these following entities for hotel search:
            1. City: Extract the city where the user wants to stay
            2. Neighborhood (if mentioned): Extract specific area or neighborhood preference
            3. Price Range (if mentioned): Extract budget constraints or price preferences per night
            4. Star Rating (if mentioned): Extract preferred hotel star rating
            5. Amenities (if mentioned): Extract specific amenities requirements
            6. Guest Rating (if mentioned): Extract minimum rating requirements
            7. Special Requirements (if mentioned): Extract any other specific needs

            Return ONLY a Python dictionary with these keys. If any value is not mentioned in the query, set it as None. Do not include any additional text or explanation."""
                            }]
            )
            
            # Parse the GPT response
            try:
                params = ast.literal_eval(completion.choices[0].message.content)
            except:
                params = {"query": query}
            
            # Search hotels using the parameters
            results = db_manager.search_hotels(str(params), SEARCH_LIMIT)
            
            # Filter out irrelevant results
            filtered_results = []
            target_city = params.get('city', '').lower() if isinstance(params, dict) else None
            
            for hotel in results:
                # Only include hotels in the requested city
                if target_city and hotel['City'].lower() != target_city:
                    continue
                filtered_results.append(hotel)
            
            # Format the results
            return {
                "status": "success",
                "total_results": len(filtered_results),
                "hotels": [self._format_hotel_details(hotel) for hotel in filtered_results]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "total_results": 0,
                "hotels": []
            }

# Create singleton instance
search_service = SearchService() 