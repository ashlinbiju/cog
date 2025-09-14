from flask import Blueprint, jsonify
import os
import googlemaps
from datetime import datetime
import requests

api_status_bp = Blueprint('api_status', __name__)

@api_status_bp.route('/status', methods=['GET'])
def check_api_status():
    """Check the status of all APIs used in the system"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'apis': {}
    }
    
    # Check Google Maps API
    google_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    if google_api_key:
        try:
            gmaps = googlemaps.Client(key=google_api_key)
            # Test geocoding
            test_result = gmaps.geocode('New York')
            if test_result:
                status['apis']['google_maps'] = {
                    'status': 'working',
                    'message': 'Geocoding API working',
                    'test_result': f"Found {len(test_result)} results for 'New York'"
                }
            else:
                status['apis']['google_maps'] = {
                    'status': 'error',
                    'message': 'Geocoding returned no results'
                }
        except Exception as e:
            status['apis']['google_maps'] = {
                'status': 'error',
                'message': f'Google Maps API Error: {str(e)}'
            }
    else:
        status['apis']['google_maps'] = {
            'status': 'not_configured',
            'message': 'Google Maps API key not found in environment'
        }
    
    # Check Google Places API
    if google_api_key:
        try:
            gmaps = googlemaps.Client(key=google_api_key)
            # Test places nearby
            places_result = gmaps.places_nearby(
                location={'lat': 40.7128, 'lng': -74.0060},  # NYC coordinates
                radius=1000,
                type='lodging'
            )
            if places_result.get('results'):
                status['apis']['google_places'] = {
                    'status': 'working',
                    'message': 'Places API working',
                    'test_result': f"Found {len(places_result['results'])} hotels near NYC"
                }
            else:
                status['apis']['google_places'] = {
                    'status': 'limited',
                    'message': 'Places API responding but no results'
                }
        except Exception as e:
            status['apis']['google_places'] = {
                'status': 'error',
                'message': f'Google Places API Error: {str(e)}'
            }
    else:
        status['apis']['google_places'] = {
            'status': 'not_configured',
            'message': 'Google Maps API key not found'
        }
    
    # Check Database connectivity
    try:
        from models.user import User
        user_count = User.query.count()
        status['apis']['database'] = {
            'status': 'working',
            'message': f'Database connected, {user_count} users found'
        }
    except Exception as e:
        status['apis']['database'] = {
            'status': 'error',
            'message': f'Database Error: {str(e)}'
        }
    
    # Check Smart Recommendations API
    try:
        from models.hotel import Hotel
        hotel_count = Hotel.query.count()
        status['apis']['smart_recommendations'] = {
            'status': 'working',
            'message': f'Smart recommendations ready, {hotel_count} hotels in database'
        }
    except Exception as e:
        status['apis']['smart_recommendations'] = {
            'status': 'error',
            'message': f'Smart Recommendations Error: {str(e)}'
        }
    
    # Overall system status
    working_apis = sum(1 for api in status['apis'].values() if api['status'] == 'working')
    total_apis = len(status['apis'])
    
    status['overall'] = {
        'working_apis': working_apis,
        'total_apis': total_apis,
        'health_percentage': round((working_apis / total_apis) * 100, 1)
    }
    
    return jsonify(status)

@api_status_bp.route('/google-api-test', methods=['GET'])
def test_google_api():
    """Detailed Google API testing"""
    google_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    
    if not google_api_key:
        return jsonify({
            'error': 'Google Maps API key not configured',
            'solution': 'Add GOOGLE_MAPS_API_KEY to your .env file'
        }), 400
    
    results = {
        'api_key': f"{google_api_key[:10]}...{google_api_key[-5:]}" if len(google_api_key) > 15 else "Short key",
        'tests': {}
    }
    
    try:
        gmaps = googlemaps.Client(key=google_api_key)
        
        # Test 1: Geocoding API
        try:
            geocode_result = gmaps.geocode('New York, NY')
            results['tests']['geocoding'] = {
                'status': 'success' if geocode_result else 'no_results',
                'message': f"Found {len(geocode_result)} results" if geocode_result else "No results"
            }
        except Exception as e:
            results['tests']['geocoding'] = {
                'status': 'error',
                'message': str(e)
            }
        
        # Test 2: Places Nearby API
        try:
            places_result = gmaps.places_nearby(
                location={'lat': 40.7128, 'lng': -74.0060},
                radius=5000,
                type='lodging'
            )
            results['tests']['places_nearby'] = {
                'status': 'success' if places_result.get('results') else 'no_results',
                'message': f"Found {len(places_result.get('results', []))} places"
            }
        except Exception as e:
            results['tests']['places_nearby'] = {
                'status': 'error',
                'message': str(e)
            }
        
        # Test 3: Place Details API
        if results['tests'].get('places_nearby', {}).get('status') == 'success':
            try:
                first_place = places_result['results'][0]
                place_details = gmaps.place(
                    place_id=first_place['place_id'],
                    fields=['name', 'rating', 'formatted_address']
                )
                results['tests']['place_details'] = {
                    'status': 'success',
                    'message': f"Got details for {place_details['result'].get('name', 'Unknown')}"
                }
            except Exception as e:
                results['tests']['place_details'] = {
                    'status': 'error',
                    'message': str(e)
                }
        
    except Exception as e:
        results['client_error'] = str(e)
    
    return jsonify(results)
