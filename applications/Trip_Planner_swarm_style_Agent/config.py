import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"
REDIS_URL = 'redis://localhost:6379'
# Database Configuration
DB_PATH = "/tmp/plannertrip"

# Search Configuration
SEARCH_LIMIT = 3

# Model Configuration
EMBEDDING_DEVICE = "cpu" 