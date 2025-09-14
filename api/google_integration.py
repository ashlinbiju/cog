from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.user import User
from app import db
import googlemaps
import requests
import os
import logging
from datetime import datetime

google_bp = Blueprint('google', __name__)
logger = logging.getLogger(__name__)

# Initialize Google Maps client
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')

# Test endpoint to verify Google Maps API key
@google_bp.route('/test-api-key', methods=['GET'])
def test_api_key():
    """Test if Google Maps API key is working"""
    try:
        if not GOOGLE_MAPS_API_KEY:
            return jsonify({
                'status': 'error',
                'message': 'GOOGLE_MAPS_API_KEY is not set in environment variables'
            }), 500
            
        # Test geocoding a well-known location (Eiffel Tower)
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        geocode_result = gmaps.geocode('Eiffel Tower')
        
        if not geocode_result:
            return jsonify({
                'status': 'error',
                'message': 'Failed to geocode test location',
                'api_key': f'{GOOGLE_MAPS_API_KEY[:5]}...{GOOGLE_MAPS_API_KEY[-3:]}',
                'hint': 'The API key might be invalid or missing required permissions'
            }), 500
            
        return jsonify({
            'status': 'success',
            'message': 'Google Maps API key is working!',
            'test_location': 'Eiffel Tower',
            'formatted_address': geocode_result[0]['formatted_address'],
            'location': geocode_result[0]['geometry']['location']
        })
        
    except Exception as e:
        logger.error(f"Error testing Google Maps API: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'api_key': f'{GOOGLE_MAPS_API_KEY[:5]}...{GOOGLE_MAPS_API_KEY[-3:]}' if GOOGLE_MAPS_API_KEY else 'Not set',
            'hint': 'Check if the API key is valid and has the required Google Maps APIs enabled'
        }), 500

# Initialize Google Maps client
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else None

# Foursquare API for additional POI data
FOURSQUARE_API_KEY = os.environ.get('FOURSQUARE_API_KEY', 'your-foursquare-api-key')

@google_bp.route('/search/hotels', methods=['POST'])
def search_hotels():
    """Search for hotels using Google Places API"""
    try:
        print("[DEBUG] Hotel search request received")
        
        # Get auth header and extract user ID manually
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            print("[ERROR] No authorization header")
            return jsonify({'error': 'Authorization required'}), 401
        
        # For now, use hardcoded user ID
        user_id = 11  # Use the user ID from your token
        user = User.query.get(user_id)
        
        if not user:
            print(f"[ERROR] User not found: {user_id}")
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        location = data.get('location')
        radius = data.get('radius', 5000)  # Default 5km
        hotel_type = data.get('type', 'lodging')
        
        if not location:
            return jsonify({'error': 'Location is required'}), 400
        
        # Get user preferences for filtering
        user_prefs = user.get_preferences()
        min_rating = user_prefs.get('min_rating', 3.0)
        
        # Geocode the location first
        geocode_result = gmaps.geocode(location)
        if not geocode_result:
            return jsonify({'error': 'Location not found'}), 404
        
        location_coords = geocode_result[0]['geometry']['location']
        
        # Search for hotels using Places API
        places_result = gmaps.places_nearby(
            location=location_coords,
            radius=radius,
            type=hotel_type,
            language='en'
        )
        
        hotels = []
        for place in places_result.get('results', []):
            # Filter by user preferences
            rating = place.get('rating', 0)
            if rating < min_rating:
                continue
            
            hotel_data = {
                'id': place['place_id'],
                'name': place['name'],
                'location': place.get('vicinity', ''),
                'rating': rating,
                'total_reviews': place.get('user_ratings_total', 0),
                'price_level': place.get('price_level', 0),
                'price_range': get_price_range_text(place.get('price_level', 0)),
                'types': place.get('types', []),
                'photos': get_place_photos(place.get('photos', [])[:3]),
                'geometry': place['geometry'],
                'open_now': place.get('opening_hours', {}).get('open_now', None)
            }
            
            hotels.append(hotel_data)
        
        # Sort by rating and user preferences
        hotels = sort_hotels_by_preferences(hotels, user_prefs)
        
        # Save search history
        save_user_search(user_id, location, 'hotel_search')
        
        return jsonify({
            'hotels': hotels[:20],  # Limit to top 20 results
            'location': {
                'name': location,
                'coordinates': location_coords,
                'formatted_address': geocode_result[0]['formatted_address']
            },
            'search_params': {
                'radius': radius,
                'type': hotel_type,
                'min_rating': min_rating
            }
        })
        
    except Exception as e:
        logger.error(f"Error searching hotels: {str(e)}")
        return jsonify({'error': 'Hotel search failed'}), 500

@google_bp.route('/hotel/<place_id>/details', methods=['GET'])
@jwt_required()
def get_hotel_details(place_id):
    """Get detailed information about a specific hotel"""
    try:
        # Get place details from Google Places API
        place_details = gmaps.place(
            place_id=place_id,
            fields=[
                'name', 'formatted_address', 'formatted_phone_number',
                'website', 'rating', 'user_ratings_total', 'price_level',
                'opening_hours', 'photos', 'reviews', 'geometry',
                'types', 'url', 'vicinity'
            ]
        )
        
        if not place_details or 'result' not in place_details:
            return jsonify({'error': 'Hotel not found'}), 404
        
        result = place_details['result']
        
        # Get nearby points of interest
        nearby_pois = get_nearby_points_of_interest(
            result['geometry']['location'],
            radius=1000
        )
        
        hotel_data = {
            'place_id': place_id,
            'name': result['name'],
            'address': result.get('formatted_address', ''),
            'phone': result.get('formatted_phone_number', ''),
            'website': result.get('website', ''),
            'rating': result.get('rating', 0),
            'total_reviews': result.get('user_ratings_total', 0),
            'price_level': result.get('price_level', 0),
            'price_range': get_price_range_text(result.get('price_level', 0)),
            'opening_hours': format_opening_hours(result.get('opening_hours', {})),
            'photos': get_place_photos(result.get('photos', [])[:10]),
            'reviews': format_reviews(result.get('reviews', [])[:5]),
            'types': result.get('types', []),
            'google_url': result.get('url', ''),
            'coordinates': result['geometry']['location'],
            'nearby_attractions': nearby_pois
        }
        
        return jsonify({'hotel': hotel_data})
        
    except Exception as e:
        logger.error(f"Error getting hotel details: {str(e)}")
        return jsonify({'error': 'Failed to get hotel details'}), 500

@google_bp.route('/nearby-attractions', methods=['POST'])
@jwt_required()
def get_nearby_attractions():
    """Get nearby attractions and points of interest"""
    try:
        data = request.get_json()
        location = data.get('location')  # Can be coordinates or place name
        radius = data.get('radius', 2000)
        
        if not location:
            return jsonify({'error': 'Location is required'}), 400
        
        # If location is a string, geocode it
        if isinstance(location, str):
            geocode_result = gmaps.geocode(location)
            if not geocode_result:
                return jsonify({'error': 'Location not found'}), 404
            location_coords = geocode_result[0]['geometry']['location']
        else:
            location_coords = location
        
        attractions = get_nearby_points_of_interest(location_coords, radius)
        
        return jsonify({
            'attractions': attractions,
            'location': location_coords,
            'radius': radius
        })
        
    except Exception as e:
        logger.error(f"Error getting nearby attractions: {str(e)}")
        return jsonify({'error': 'Failed to get attractions'}), 500

@google_bp.route('/popular-locations', methods=['GET'])
def get_popular_locations():
    """Get popular travel destinations"""
    try:
        # This would typically come from a database or external API
        # For now, return a curated list
        popular_locations = [
            {
                'name': 'New York City, NY',
                'country': 'United States',
                'coordinates': {'lat': 40.7128, 'lng': -74.0060},
                'image_url': 'https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=300&h=200&fit=crop'
            },
            {
                'name': 'Paris, France',
                'country': 'France',
                'coordinates': {'lat': 48.8566, 'lng': 2.3522},
                'image_url': 'https://images.unsplash.com/photo-1502602898536-47ad22581b52?w=300&h=200&fit=crop'
            },
            {
                'name': 'Tokyo, Japan',
                'country': 'Japan',
                'coordinates': {'lat': 35.6762, 'lng': 139.6503},
                'image_url': 'https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=300&h=200&fit=crop'
            },
            {
                'name': 'London, UK',
                'country': 'United Kingdom',
                'coordinates': {'lat': 51.5074, 'lng': -0.1278},
                'image_url': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=300&h=200&fit=crop'
            },
            {
                'name': 'Dubai, UAE',
                'country': 'United Arab Emirates',
                'coordinates': {'lat': 25.2048, 'lng': 55.2708},
                'image_url': 'https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=300&h=200&fit=crop'
            },
            {
                'name': 'Sydney, Australia',
                'country': 'Australia',
                'coordinates': {'lat': -33.8688, 'lng': 151.2093},
                'image_url': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=300&h=200&fit=crop'
            }
        ]
        
        return jsonify({'popular_locations': popular_locations})
        
    except Exception as e:
        logger.error(f"Error getting popular locations: {str(e)}")
        return jsonify({'error': 'Failed to get popular locations'}), 500

@google_bp.route('/user/search-history', methods=['GET'])
def get_user_search_history():
    """Get user's search history"""
    try:
        print("[DEBUG] Search history request received")
        
        # Get auth header and extract user ID manually
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            print("[ERROR] No authorization header for search history")
            return jsonify({'error': 'Authorization required'}), 401
        
        # For now, use hardcoded user ID
        user_id = 11  # Use the user ID from your token
        print(f"[DEBUG] Getting search history for user: {user_id}")
        
        # This would typically come from a database
        # For now, return from session or a simple cache
        # In production, implement proper search history storage
        
        return jsonify({
            'search_history': []  # Placeholder
        })
        
    except Exception as e:
        logger.error(f"Error getting search history: {str(e)}")
        return jsonify({'error': 'Failed to get search history'}), 500

@google_bp.route('/user/save-search', methods=['POST'])
@jwt_required()
def save_search():
    """Save user search for history"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        location = data.get('location')
        search_type = data.get('type', 'general')
        
        # Save to database or cache
        save_user_search(user_id, location, search_type)
        
        return jsonify({'message': 'Search saved successfully'})
        
    except Exception as e:
        logger.error(f"Error saving search: {str(e)}")
        return jsonify({'error': 'Failed to save search'}), 500

# Helper functions

def get_price_range_text(price_level):
    """Convert price level to text"""
    price_ranges = {
        0: 'Free',
        1: 'Inexpensive',
        2: 'Moderate',
        3: 'Expensive',
        4: 'Very Expensive'
    }
    return price_ranges.get(price_level, 'Unknown')

def get_place_photos(photos):
    """Get photo URLs from Google Places photos"""
    photo_urls = []
    for photo in photos:
        if 'photo_reference' in photo:
            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo['photo_reference']}&key={GOOGLE_MAPS_API_KEY}"
            photo_urls.append(photo_url)
    return photo_urls

def format_opening_hours(opening_hours):
    """Format opening hours data"""
    if not opening_hours:
        return {}
    
    return {
        'open_now': opening_hours.get('open_now', False),
        'weekday_text': opening_hours.get('weekday_text', []),
        'periods': opening_hours.get('periods', [])
    }

def format_reviews(reviews):
    """Format Google Places reviews"""
    formatted_reviews = []
    for review in reviews:
        formatted_reviews.append({
            'author_name': review.get('author_name', 'Anonymous'),
            'rating': review.get('rating', 0),
            'text': review.get('text', ''),
            'time': review.get('time', 0),
            'relative_time_description': review.get('relative_time_description', '')
        })
    return formatted_reviews

def get_nearby_points_of_interest(location, radius=2000):
    """Get nearby points of interest using Google Places API"""
    try:
        # Search for various types of attractions
        attraction_types = [
            'tourist_attraction',
            'museum',
            'park',
            'shopping_mall',
            'restaurant',
            'amusement_park'
        ]
        
        all_pois = []
        
        for poi_type in attraction_types:
            try:
                places_result = gmaps.places_nearby(
                    location=location,
                    radius=radius,
                    type=poi_type,
                    language='en'
                )
                
                for place in places_result.get('results', [])[:3]:  # Limit per type
                    poi_data = {
                        'place_id': place['place_id'],
                        'name': place['name'],
                        'type': poi_type,
                        'rating': place.get('rating', 0),
                        'vicinity': place.get('vicinity', ''),
                        'photos': get_place_photos(place.get('photos', [])[:1])
                    }
                    all_pois.append(poi_data)
                    
            except Exception as e:
                logger.warning(f"Error getting {poi_type} POIs: {str(e)}")
                continue
        
        # Remove duplicates and sort by rating
        unique_pois = {poi['place_id']: poi for poi in all_pois}.values()
        sorted_pois = sorted(unique_pois, key=lambda x: x['rating'], reverse=True)
        
        return sorted_pois[:10]  # Return top 10
        
    except Exception as e:
        logger.error(f"Error getting POIs: {str(e)}")
        return []

def sort_hotels_by_preferences(hotels, user_prefs):
    """Sort hotels based on user preferences"""
    def calculate_preference_score(hotel):
        score = hotel['rating'] * 20  # Base score from rating
        
        # Bonus for preferred amenities (inferred from types)
        preferred_amenities = user_prefs.get('amenities', [])
        hotel_types = hotel.get('types', [])
        
        if 'spa' in preferred_amenities and 'spa' in hotel_types:
            score += 10
        if 'gym' in preferred_amenities and 'gym' in hotel_types:
            score += 10
        
        # Bonus for review count (popularity)
        review_count = hotel.get('total_reviews', 0)
        if review_count >= user_prefs.get('min_reviews', 50):
            score += min(review_count / 100, 20)  # Cap at 20 bonus points
        
        return score
    
    # Sort by preference score
    return sorted(hotels, key=calculate_preference_score, reverse=True)

def save_user_search(user_id, location, search_type):
    """Save user search to database or cache"""
    try:
        # In a real implementation, save to database
        # For now, just log it
        logger.info(f"User {user_id} searched for {location} (type: {search_type})")
        
        # TODO: Implement actual search history storage
        # This could be a separate SearchHistory model
        
    except Exception as e:
        logger.error(f"Error saving search: {str(e)}")

# Foursquare integration for additional POI data
def get_foursquare_venues(location, radius=1000):
    """Get venues from Foursquare API"""
    try:
        if not FOURSQUARE_API_KEY or FOURSQUARE_API_KEY == 'your-foursquare-api-key':
            return []
        
        # Foursquare API v3 endpoint
        url = "https://api.foursquare.com/v3/places/search"
        
        headers = {
            "Accept": "application/json",
            "Authorization": FOURSQUARE_API_KEY
        }
        
        params = {
            "ll": f"{location['lat']},{location['lng']}",
            "radius": radius,
            "categories": "10000,13000,16000",  # Arts, Food, Travel
            "limit": 20
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            venues = []
            
            for venue in data.get('results', []):
                venue_data = {
                    'id': venue['fsq_id'],
                    'name': venue['name'],
                    'category': venue['categories'][0]['name'] if venue.get('categories') else 'Unknown',
                    'address': venue.get('location', {}).get('formatted_address', ''),
                    'distance': venue.get('distance', 0),
                    'rating': venue.get('rating', 0) / 2,  # Convert to 5-star scale
                    'source': 'foursquare'
                }
                venues.append(venue_data)
            
            return venues
            
    except Exception as e:
        logger.error(f"Error getting Foursquare venues: {str(e)}")
        return []
