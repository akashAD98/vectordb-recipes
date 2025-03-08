from typing import Optional
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from lancedb.embeddings import EmbeddingFunctionRegistry

# Initialize registry
registry = EmbeddingFunctionRegistry.get_instance()

def get_embedding_function():
    # Lazy load the embedding function when needed
    return registry.get("sentence-transformers").create(device="cpu")

class FlightModel(LanceModel):
    id: str
    Flight_Number: str
    Airline: str
    Origin_City: str
    Origin_Airport: str
    Destination_City: str
    Destination_Airport: str
    Departure_Time: str
    Arrival_Time: str
    Duration: str
    Price: str
    Available_Seats: str
    Class: str
    Flight_details: str
    
    def __init__(self, **data):
        super().__init__(**data)
        embedding_func = get_embedding_function()
        self.vector: Vector = embedding_func.VectorField()
        for field in self.__fields__:
            if field != "vector":
                setattr(self.__class__, field, embedding_func.SourceField())

class HotelModel(LanceModel):
    id: str
    ID: str
    Name: str
    City: str
    Neighborhood: str
    Address: str
    Stars: str
    Price_Per_Night: str
    Amenities: str
    Rating: str
    all_hotel_details: str
    
    def __init__(self, **data):
        super().__init__(**data)
        embedding_func = get_embedding_function()
        self.vector: Vector = embedding_func.VectorField()
        for field in self.__fields__:
            if field != "vector":
                setattr(self.__class__, field, embedding_func.SourceField())

# Define data classes for search parameters
class FlightSearchParams:
    def __init__(
        self,
        origin_city_airport: Optional[str] = None,
        destination_city_airport: Optional[str] = None,
        travel_date_time: Optional[str] = None,
        class_preference: Optional[str] = None,
        airline_preference: Optional[str] = None,
        price_range: Optional[str] = None,
        other_preferences: Optional[str] = None
    ):
        self.origin_city_airport = origin_city_airport
        self.destination_city_airport = destination_city_airport
        self.travel_date_time = travel_date_time
        self.class_preference = class_preference
        self.airline_preference = airline_preference
        self.price_range = price_range
        self.other_preferences = other_preferences

class HotelSearchParams:
    def __init__(
        self,
        city: Optional[str] = None,
        neighborhood: Optional[str] = None,
        price_range: Optional[str] = None,
        star_rating: Optional[str] = None,
        amenities: Optional[str] = None,
        guest_rating: Optional[str] = None,
        special_requirements: Optional[str] = None
    ):
        self.city = city
        self.neighborhood = neighborhood
        self.price_range = price_range
        self.star_rating = star_rating
        self.amenities = amenities
        self.guest_rating = guest_rating
        self.special_requirements = special_requirements 