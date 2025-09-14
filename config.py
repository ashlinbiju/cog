import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///hotel_recommendations.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-string'
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    
    # Google APIs Configuration
    GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY') or 'your-google-maps-api-key'
    GOOGLE_PLACES_API_KEY = os.environ.get('GOOGLE_PLACES_API_KEY') or 'your-google-places-api-key'
    
    # Google API URLs
    GOOGLE_MAPS_BASE_URL = 'https://maps.googleapis.com/maps/api'
    GOOGLE_PLACES_BASE_URL = 'https://maps.googleapis.com/maps/api/place'
    
    # External API Configuration (fallback)
    EXTERNAL_HOTEL_API = os.environ.get('EXTERNAL_HOTEL_API') or 'https://api.example.com'
    API_KEY = os.environ.get('API_KEY') or 'your-api-key'
    
    # Sentiment Analysis Configuration
    SENTIMENT_THRESHOLD_POSITIVE = 0.1
    SENTIMENT_THRESHOLD_NEGATIVE = -0.1
    
    # Recommendation Configuration
    MIN_REVIEWS_FOR_RECOMMENDATION = 3
    RECOMMENDATION_COUNT = 10
    SIMILARITY_THRESHOLD = 0.1
    
    # Google API Settings
    GOOGLE_API_TIMEOUT = 30
    MAX_HOTELS_PER_REQUEST = 20
    DEFAULT_SEARCH_RADIUS = 5000  # 5km radius
    
    # User session settings
    REQUIRE_LOGIN_FOR_RECOMMENDATIONS = True
    ALLOW_GUEST_BROWSING = False

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or 'sqlite:///hotel_dev.db'

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}