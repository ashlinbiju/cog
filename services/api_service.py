import requests
import json
import logging
from typing import Dict, List, Optional
from config import Config
from models.hotel import Hotel
from models.review import Review
from app import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExternalAPIService:
    def __init__(self):
        self.base_url = Config.EXTERNAL_HOTEL_API
        self.api_key = Config.API_KEY
        self.timeout = 30
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'HotelRecommendationSystem/1.0'
        })
        
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'X-API-Key': self.api_key
            })
    
    def fetch_hotels_from_api(self, location: str = None, limit: int = 50) -> List[Dict]:
        """Fetch hotels from external API"""
        try:
            endpoint = f"{self.base_url}/hotels"
            params = {'limit': limit}
            
            if location:
                params['location'] = location
            
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            hotels = data.get('hotels', []) if isinstance(data, dict) else data
            
            logger.info(f"Fetched {len(hotels)} hotels from API")
            return hotels
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return []
    
    def fetch_hotel_details(self, hotel_id: str) -> Optional[Dict]:
        """Fetch detailed information for a specific hotel"""
        try:
            endpoint = f"{self.base_url}/hotels/{hotel_id}"
            response = self.session.get(endpoint, timeout=self.timeout)
            response.raise_for_status()
            
            hotel_data = response.json()
            logger.info(f"Fetched details for hotel ID: {hotel_id}")
            return hotel_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch hotel details: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error parsing hotel details: {str(e)}")
            return None
    
    def fetch_hotel_reviews(self, hotel_id: str, limit: int = 100) -> List[Dict]:
        """Fetch reviews for a specific hotel from external API"""
        try:
            endpoint = f"{self.base_url}/hotels/{hotel_id}/reviews"
            params = {'limit': limit}
            
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            reviews = data.get('reviews', []) if isinstance(data, dict) else data
            
            logger.info(f"Fetched {len(reviews)} reviews for hotel {hotel_id}")
            return reviews
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch reviews: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error parsing reviews: {str(e)}")
            return []
    
    def search_hotels_by_criteria(self, criteria: Dict) -> List[Dict]:
        """Search hotels using multiple criteria"""
        try:
            endpoint = f"{self.base_url}/hotels/search"
            
            # Prepare search parameters
            params = {}
            
            if criteria.get('location'):
                params['location'] = criteria['location']
            if criteria.get('check_in'):
                params['check_in'] = criteria['check_in']
            if criteria.get('check_out'):
                params['check_out'] = criteria['check_out']
            if criteria.get('guests'):
                params['guests'] = criteria['guests']
            if criteria.get('price_min'):
                params['price_min'] = criteria['price_min']
            if criteria.get('price_max'):
                params['price_max'] = criteria['price_max']
            if criteria.get('amenities'):
                params['amenities'] = ','.join(criteria['amenities'])
            if criteria.get('rating_min'):
                params['rating_min'] = criteria['rating_min']
            
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            hotels = data.get('hotels', []) if isinstance(data, dict) else data
            
            logger.info(f"Found {len(hotels)} hotels matching criteria")
            return hotels
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Hotel search failed: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error in hotel search: {str(e)}")
            return []
    
    def sync_hotels_with_database(self, location: str = None) -> int:
        """Sync external API hotels with local database"""
        try:
            api_hotels = self.fetch_hotels_from_api(location)
            synced_count = 0
            
            for api_hotel in api_hotels:
                try:
                    # Check if hotel already exists
                    existing_hotel = Hotel.query.filter_by(
                        external_id=str(api_hotel.get('id', ''))
                    ).first()
                    
                    if existing_hotel:
                        # Update existing hotel
                        self._update_hotel_from_api(existing_hotel, api_hotel)
                    else:
                        # Create new hotel
                        new_hotel = self._create_hotel_from_api(api_hotel)
                        if new_hotel:
                            db.session.add(new_hotel)
                    
                    synced_count += 1
                    
                except Exception as e:
                    logger.error(f"Error syncing hotel {api_hotel.get('id', 'unknown')}: {str(e)}")
                    continue
            
            db.session.commit()
            logger.info(f"Successfully synced {synced_count} hotels")
            return synced_count
            
        except Exception as e:
            logger.error(f"Error syncing hotels: {str(e)}")
            db.session.rollback()
            return 0
    
    def _create_hotel_from_api(self, api_hotel: Dict) -> Optional[Hotel]:
        """Create Hotel object from API data"""
        try:
            hotel = Hotel(
                name=api_hotel.get('name', ''),
                location=api_hotel.get('location', ''),
                description=api_hotel.get('description', ''),
                price_range=self._map_price_range(api_hotel.get('price_range')),
                rating=float(api_hotel.get('rating', 0)),
                total_reviews=int(api_hotel.get('review_count', 0)),
                external_id=str(api_hotel.get('id', '')),
                api_source='external_api',
                star_rating=int(api_hotel.get('stars', 0)),
                category=api_hotel.get('category', 'hotel')
            )
            
            # Set amenities
            amenities = api_hotel.get('amenities', [])
            hotel.set_amenities(amenities if isinstance(amenities, list) else [])
            
            # Set contact info
            contact = {
                'phone': api_hotel.get('phone', ''),
                'email': api_hotel.get('email', ''),
                'website': api_hotel.get('website', '')
            }
            hotel.set_contact_info(contact)
            
            return hotel
            
        except Exception as e:
            logger.error(f"Error creating hotel from API data: {str(e)}")
            return None
    
    def _update_hotel_from_api(self, hotel: Hotel, api_hotel: Dict):
        """Update existing hotel with API data"""
        try:
            hotel.name = api_hotel.get('name', hotel.name)
            hotel.location = api_hotel.get('location', hotel.location)
            hotel.description = api_hotel.get('description', hotel.description)
            hotel.rating = float(api_hotel.get('rating', hotel.rating))
            hotel.total_reviews = int(api_hotel.get('review_count', hotel.total_reviews))
            hotel.star_rating = int(api_hotel.get('stars', hotel.star_rating or 0))
            hotel.category = api_hotel.get('category', hotel.category)
            
            # Update amenities
            amenities = api_hotel.get('amenities', [])
            if isinstance(amenities, list):
                hotel.set_amenities(amenities)
            
            # Update contact info
            existing_contact = hotel.get_contact_info()
            contact = {
                'phone': api_hotel.get('phone', existing_contact.get('phone', '')),
                'email': api_hotel.get('email', existing_contact.get('email', '')),
                'website': api_hotel.get('website', existing_contact.get('website', ''))
            }
            hotel.set_contact_info(contact)
            
        except Exception as e:
            logger.error(f"Error updating hotel: {str(e)}")
    
    def _map_price_range(self, api_price_range) -> str:
        """Map API price range to internal format"""
        if not api_price_range:
            return 'medium'
        
        price_mapping = {
            '$': 'budget',
            '$$': 'low',
            '$$$': 'medium',
            '$$$$': 'high',
            '$$$$$': 'luxury',
            'budget': 'budget',
            'low': 'low',
            'medium': 'medium',
            'high': 'high',
            'luxury': 'luxury'
        }
        
        return price_mapping.get(str(api_price_range).lower(), 'medium')
    
    def sync_hotel_reviews(self, hotel_id: int, external_hotel_id: str) -> int:
        """Sync reviews for a specific hotel"""
        try:
            api_reviews = self.fetch_hotel_reviews(external_hotel_id)
            synced_count = 0
            
            for api_review in api_reviews:
                try:
                    # Here you would need to map API users to your local users
                    # For now, we'll skip creating reviews from API
                    # as it requires user authentication and mapping
                    pass
                    
                except Exception as e:
                    logger.error(f"Error syncing review: {str(e)}")
                    continue
            
            logger.info(f"Synced {synced_count} reviews for hotel {hotel_id}")
            return synced_count
            
        except Exception as e:
            logger.error(f"Error syncing reviews: {str(e)}")
            return 0
    
    def get_hotel_availability(self, hotel_id: str, check_in: str, check_out: str, guests: int = 2) -> Dict:
        """Check hotel availability and pricing"""
        try:
            endpoint = f"{self.base_url}/hotels/{hotel_id}/availability"
            params = {
                'check_in': check_in,
                'check_out': check_out,
                'guests': guests
            }
            
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            availability_data = response.json()
            return availability_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch availability: {str(e)}")
            return {'available': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"Error checking availability: {str(e)}")
            return {'available': False, 'error': str(e)}
    
    def get_location_suggestions(self, query: str) -> List[Dict]:
        """Get location suggestions for autocomplete"""
        try:
            endpoint = f"{self.base_url}/locations/suggest"
            params = {'query': query, 'limit': 10}
            
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            suggestions = data.get('suggestions', []) if isinstance(data, dict) else data
            
            return suggestions
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch location suggestions: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error getting location suggestions: {str(e)}")
            return []
    
    def validate_api_connection(self) -> bool:
        """Test if API connection is working"""
        try:
            endpoint = f"{self.base_url}/health"
            response = self.session.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                logger.info("API connection successful")
                return True
            else:
                logger.warning(f"API health check returned status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API connection failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error validating API connection: {str(e)}")
            return False

# Global API service instance
api_service = ExternalAPIService()