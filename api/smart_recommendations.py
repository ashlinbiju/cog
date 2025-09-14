from flask import Blueprint, request, jsonify
from models.user import User
from models.hotel import Hotel
from models.review import Review
from app import db
import googlemaps
import os
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from datetime import datetime, timedelta

smart_recommendations_bp = Blueprint('smart_recommendations', __name__)
logger = logging.getLogger(__name__)

# Initialize Google Maps client with free tier optimization
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
gmaps = None

if GOOGLE_MAPS_API_KEY:
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        # Test the API key with a simple request
        test_result = gmaps.geocode("New York")
        print(f"[INFO] Google Maps API initialized successfully")
    except Exception as e:
        print(f"[WARNING] Google Maps API failed to initialize: {str(e)}")
        print("[INFO] System will use fallback recommendations")
        gmaps = None
else:
    print("[INFO] No Google Maps API key found, using fallback system")

@smart_recommendations_bp.route('/personalized/<int:user_id>', methods=['GET'])
def get_personalized_recommendations(user_id):
    """Get personalized hotel recommendations using collaborative filtering and user preferences"""
    try:
        print(f"[DEBUG] Getting personalized recommendations for user {user_id}")
        
        # Get auth header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            print("[ERROR] No authorization header")
            return jsonify({'error': 'Authorization required'}), 401
        
        user = User.query.get(user_id)
        if not user:
            print(f"[ERROR] User not found: {user_id}")
            return jsonify({'error': 'User not found'}), 404
        
        # Get user preferences
        preferences = user.get_preferences()
        print(f"[DEBUG] User preferences: {preferences}")
        
        # Get query parameters
        location = request.args.get('location', user.location)
        limit = min(int(request.args.get('limit', 10)), 20)
        price_range = request.args.get('price_range', '')
        min_rating = request.args.get('min_rating', '')
        sort_by = request.args.get('sort_by', 'relevance')
        
        # Step 1: Collaborative Filtering
        collaborative_recommendations = get_collaborative_filtering_recommendations(user_id, limit)
        
        # Step 2: Content-based filtering using preferences
        content_recommendations = get_content_based_recommendations(user_id, preferences, location, limit)
        
        # Step 3: Google Maps API integration for nearby hotels
        google_recommendations = []
        if location and gmaps:
            google_recommendations = get_google_maps_recommendations(location, preferences, limit)
        
        # Step 4: Combine recommendations using hybrid approach
        final_recommendations = combine_recommendations(
            collaborative_recommendations, 
            content_recommendations, 
            google_recommendations, 
            preferences, 
            limit
        )
        
        # If no recommendations found, get fallback hotels
        if not final_recommendations:
            print("[DEBUG] No recommendations found, using fallback hotels")
            final_recommendations = get_fallback_hotels(location, limit)
        
        # Apply additional filters
        if price_range:
            filtered = [r for r in final_recommendations 
                       if r.get('hotel', {}).get('price_range') == price_range]
            if filtered:  # Only apply filter if it doesn't remove all results
                final_recommendations = filtered
        
        if min_rating:
            min_rating_float = float(min_rating)
            filtered = [r for r in final_recommendations 
                       if r.get('hotel', {}).get('rating', 0) >= min_rating_float]
            if filtered:  # Only apply filter if it doesn't remove all results
                final_recommendations = filtered
        
        # Apply sorting
        if sort_by == 'price_low':
            final_recommendations.sort(key=lambda x: x.get('hotel', {}).get('price', 999999))
        elif sort_by == 'price_high':
            final_recommendations.sort(key=lambda x: x.get('hotel', {}).get('price', 0), reverse=True)
        elif sort_by == 'rating':
            final_recommendations.sort(key=lambda x: x.get('hotel', {}).get('rating', 0), reverse=True)
        
        # Ensure we always return at least some recommendations
        if not final_recommendations:
            print("[DEBUG] Still no recommendations after filtering, using basic fallback")
            final_recommendations = get_fallback_hotels(location, limit)
        
        print(f"[DEBUG] Final recommendations count after filters: {len(final_recommendations)}")
        return jsonify({
            'recommendations': final_recommendations,
            'total': len(final_recommendations),
            'user_preferences': preferences,
            'applied_filters': {
                'location': location,
                'price_range': price_range,
                'min_rating': min_rating,
                'sort_by': sort_by
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {str(e)}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

def get_collaborative_filtering_recommendations(user_id, limit):
    """Get recommendations using collaborative filtering"""
    try:
        print(f"[DEBUG] Getting collaborative filtering recommendations for user {user_id}")
        
        # Create user-item matrix
        reviews = Review.query.all()
        if not reviews:
            print("[DEBUG] No reviews found for collaborative filtering")
            return []
        
        # Build user-item matrix
        user_item_data = []
        for review in reviews:
            user_item_data.append({
                'user_id': review.user_id,
                'hotel_id': review.hotel_id,
                'rating': review.rating
            })
        
        df = pd.DataFrame(user_item_data)
        user_item_matrix = df.pivot_table(index='user_id', columns='hotel_id', values='rating', fill_value=0)
        
        if user_id not in user_item_matrix.index:
            print(f"[DEBUG] User {user_id} not in user-item matrix")
            return []
        
        # Calculate user similarity
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
        
        # Find similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]  # Top 5 similar users
        
        recommendations = []
        user_rated_hotels = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
        
        for similar_user_id, similarity_score in similar_users.items():
            if similarity_score > 0.1:  # Minimum similarity threshold
                similar_user_hotels = user_item_matrix.loc[similar_user_id][user_item_matrix.loc[similar_user_id] > 0]
                
                for hotel_id, rating in similar_user_hotels.items():
                    if hotel_id not in user_rated_hotels and rating >= 4.0:
                        hotel = Hotel.query.get(hotel_id)
                        if hotel and hotel.is_active:
                            recommendations.append({
                                'hotel_id': hotel_id,
                                'hotel': hotel.to_dict(),
                                'predicted_rating': rating * similarity_score,
                                'similarity_score': similarity_score,
                                'method': 'collaborative_filtering'
                            })
        
        # Sort by predicted rating and return top recommendations
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:limit]
        
    except Exception as e:
        logger.error(f"Error in collaborative filtering: {str(e)}")
        return []

def get_content_based_recommendations(user_id, preferences, location, limit):
    """Get recommendations based on user preferences and hotel content"""
    try:
        print(f"[DEBUG] Getting content-based recommendations for user {user_id}")
        
        # Get all active hotels
        hotels = Hotel.query.filter_by(is_active=True).all()
        if not hotels:
            print("[DEBUG] No active hotels found in database")
            return []
        
        print(f"[DEBUG] Found {len(hotels)} active hotels")
        recommendations = []
        
        for hotel in hotels:
            score = 1  # Base score for all hotels
            reasons = ["Available hotel"]
            
            # Location preference
            if location and location.lower() in hotel.location.lower():
                score += 3
                reasons.append(f"Located in {location}")
            
            # Budget preference
            if 'budget_category' in preferences:
                budget_prefs = preferences['budget_category']
                if isinstance(budget_prefs, list) and hotel.price_range in budget_prefs:
                    score += 2
                    reasons.append(f"Matches {hotel.price_range} budget")
            
            # Amenities preference
            if 'amenities' in preferences and hotel.amenities:
                preferred_amenities = preferences['amenities']
                if isinstance(preferred_amenities, list):
                    hotel_amenities = hotel.amenities if isinstance(hotel.amenities, list) else []
                    common_amenities = set(preferred_amenities) & set(hotel_amenities)
                    if common_amenities:
                        score += len(common_amenities) * 0.5
                        reasons.append(f"Has amenities: {', '.join(common_amenities)}")
            
            # Rating preference - be more lenient
            min_rating = float(preferences.get('min_rating', 3.0))  # Default to 3.0 if not specified
            if hotel.rating >= min_rating:
                score += 2
                reasons.append(f"Rating {hotel.rating} meets minimum {min_rating}")
            elif hotel.rating >= 3.0:  # Still include decent hotels
                score += 1
                reasons.append(f"Good rating: {hotel.rating}")
            
            # Review count preference - be more lenient
            min_reviews = int(preferences.get('min_reviews', 10))  # Lower default
            if hotel.total_reviews >= min_reviews:
                score += 1
                reasons.append(f"Has {hotel.total_reviews} reviews")
            elif hotel.total_reviews >= 5:  # Still include hotels with some reviews
                score += 0.5
                reasons.append(f"Has {hotel.total_reviews} reviews")
            
            # Travel purpose
            if 'travel_purpose' in preferences:
                travel_purposes = preferences['travel_purpose']
                if isinstance(travel_purposes, list):
                    # Simple keyword matching with hotel description/category
                    for purpose in travel_purposes:
                        if hotel.description and purpose.lower() in hotel.description.lower():
                            score += 1
                            reasons.append(f"Suitable for {purpose}")
            
            # Always include hotels with at least base score
            recommendations.append({
                'hotel_id': hotel.id,
                'hotel': hotel.to_dict(),
                'content_score': score,
                'reasons': reasons,
                'method': 'content_based'
            })
        
        # Sort by content score
        recommendations.sort(key=lambda x: x['content_score'], reverse=True)
        result = recommendations[:limit]
        print(f"[DEBUG] Returning {len(result)} content-based recommendations")
        return result
        
    except Exception as e:
        logger.error(f"Error in content-based recommendations: {str(e)}")
        return []

def get_google_maps_recommendations(location, preferences, limit):
    """Get hotel recommendations using Google Maps API"""
    try:
        print(f"[DEBUG] Getting Google Maps recommendations for location: {location}")
        
        if not gmaps or not GOOGLE_MAPS_API_KEY:
            print("[DEBUG] Google Maps client not initialized or API key missing")
            return get_fallback_hotels(location, limit)
        
        # First try to geocode the location
        try:
            geocode_result = gmaps.geocode(location)
            if not geocode_result:
                print(f"[DEBUG] Could not geocode location: {location}")
                return get_fallback_hotels(location, limit)
            
            lat_lng = geocode_result[0]['geometry']['location']
        except Exception as e:
            print(f"[DEBUG] Geocoding failed: {str(e)}")
            return get_fallback_hotels(location, limit)
        
        # Search for hotels near the location
        try:
            places_result = gmaps.places_nearby(
                location=lat_lng,
                radius=5000,  # 5km radius
                type='lodging',
                language='en'
            )
        except Exception as e:
            print(f"[DEBUG] Places API failed: {str(e)}")
            return get_fallback_hotels(location, limit)
        
        recommendations = []
        
        for place in places_result.get('results', [])[:limit]:
            # Get place details
            place_details = gmaps.place(
                place_id=place['place_id'],
                fields=['name', 'formatted_address', 'rating', 'user_ratings_total', 'price_level', 'photos', 'types']
            )
            
            details = place_details.get('result', {})
            
            # Get hotel image from Google Places photos
            hotel_image = None
            if 'photos' in details and details['photos']:
                photo_reference = details['photos'][0].get('photo_reference')
                if photo_reference:
                    hotel_image = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=800&photoreference={photo_reference}&key={GOOGLE_MAPS_API_KEY}"
            
            # Calculate relevance score based on preferences
            relevance_score = 0
            
            # Rating preference
            if 'min_rating' in preferences and 'rating' in details:
                if details['rating'] >= float(preferences.get('min_rating', 0)):
                    relevance_score += 2
            
            # Review count preference
            if 'min_reviews' in preferences and 'user_ratings_total' in details:
                if details['user_ratings_total'] >= int(preferences.get('min_reviews', 0)):
                    relevance_score += 1
            
            # Budget preference (Google's price_level: 0-4 scale)
            if 'budget_category' in preferences and 'price_level' in details:
                budget_prefs = preferences['budget_category']
                price_level = details['price_level']
                
                # Map Google's price levels to budget categories
                price_mapping = {
                    0: 'budget',
                    1: 'budget', 
                    2: 'mid-range',
                    3: 'luxury',
                    4: 'luxury'
                }
                
                if isinstance(budget_prefs, list) and price_mapping.get(price_level) in budget_prefs:
                    relevance_score += 2
            
            # Map Google price level to price range
            price_range = 'mid-range'
            price = None
            if 'price_level' in details:
                price_level = details['price_level']
                if price_level <= 1:
                    price_range = 'budget'
                    price = 89
                elif price_level == 2:
                    price_range = 'mid-range'
                    price = 159
                elif price_level == 3:
                    price_range = 'luxury'
                    price = 299
                else:
                    price_range = 'premium'
                    price = 449

            recommendations.append({
                'hotel_id': f"google_{place['place_id']}",
                'hotel': {
                    'name': details.get('name', 'Unknown'),
                    'location': details.get('formatted_address', ''),
                    'rating': details.get('rating', 0),
                    'total_reviews': details.get('user_ratings_total', 0),
                    'price_level': details.get('price_level'),
                    'price_range': price_range,
                    'price': price,
                    'image': hotel_image,
                    'google_place_id': place['place_id'],
                    'source': 'google_maps'
                },
                'relevance_score': relevance_score,
                'method': 'google_maps'
            })
        
        # Sort by relevance score
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting Google Maps recommendations: {str(e)}")
        return get_fallback_hotels(location, limit)

def get_fallback_hotels(location, limit):
    """Get fallback hotel data when Google Maps API fails"""
    fallback_hotels = [
        {
            'hotel_id': 'fallback_1',
            'hotel': {
                'name': f'Grand Hotel {location}' if location else 'Grand Hotel',
                'location': location or 'City Center',
                'rating': 4.2,
                'total_reviews': 156,
                'price_range': 'mid-range',
                'price': 159,
                'image': 'https://images.unsplash.com/photo-1566073771259-6a8506099945?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
                'source': 'fallback'
            },
            'relevance_score': 3.0,
            'method': 'fallback'
        },
        {
            'hotel_id': 'fallback_2',
            'hotel': {
                'name': f'Luxury Resort {location}' if location else 'Luxury Resort',
                'location': location or 'Downtown',
                'rating': 4.6,
                'total_reviews': 203,
                'price_range': 'luxury',
                'price': 299,
                'image': 'https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
                'source': 'fallback'
            },
            'relevance_score': 4.0,
            'method': 'fallback'
        },
        {
            'hotel_id': 'fallback_3',
            'hotel': {
                'name': f'Budget Inn {location}' if location else 'Budget Inn',
                'location': location or 'Near Airport',
                'rating': 3.8,
                'total_reviews': 89,
                'price_range': 'budget',
                'price': 89,
                'image': 'https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
                'source': 'fallback'
            },
            'relevance_score': 2.5,
            'method': 'fallback'
        }
    ]
    
    return fallback_hotels[:limit]

def combine_recommendations(collaborative, content_based, google_maps, preferences, limit):
    """Combine recommendations from different sources using weighted scoring"""
    try:
        print("[DEBUG] Combining recommendations from all sources")
        
        all_recommendations = []
        
        # Weight different recommendation methods
        weights = {
            'collaborative_filtering': 0.4,
            'content_based': 0.4,
            'google_maps': 0.2
        }
        
        # Process collaborative filtering recommendations
        for rec in collaborative:
            rec['final_score'] = rec.get('predicted_rating', 0) * weights['collaborative_filtering']
            all_recommendations.append(rec)
        
        # Process content-based recommendations
        for rec in content_based:
            rec['final_score'] = rec.get('content_score', 0) * weights['content_based']
            all_recommendations.append(rec)
        
        # Process Google Maps recommendations
        for rec in google_maps:
            rec['final_score'] = rec.get('relevance_score', 0) * weights['google_maps']
            all_recommendations.append(rec)
        
        # Remove duplicates (same hotel from different sources)
        unique_recommendations = {}
        for rec in all_recommendations:
            hotel_id = rec['hotel_id']
            if hotel_id not in unique_recommendations or rec['final_score'] > unique_recommendations[hotel_id]['final_score']:
                unique_recommendations[hotel_id] = rec
        
        # Convert back to list and sort by final score
        final_recommendations = list(unique_recommendations.values())
        final_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Add ranking and format response
        for i, rec in enumerate(final_recommendations[:limit]):
            rec['rank'] = i + 1
            rec['final_score'] = round(rec['final_score'], 2)
        
        return final_recommendations[:limit]
        
    except Exception as e:
        logger.error(f"Error combining recommendations: {str(e)}")
        return []

@smart_recommendations_bp.route('/trending', methods=['GET'])
def get_trending_hotels():
    """Get trending hotels based on recent activity and ratings"""
    try:
        print("[DEBUG] Getting trending hotels")
        
        location = request.args.get('location')
        limit = min(int(request.args.get('limit', 10)), 20)
        
        # Get hotels with recent reviews (last 30 days) and high ratings
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        trending_query = """
            SELECT h.id, h.name, h.location, h.rating, h.total_reviews,
                   COUNT(r.id) as recent_reviews,
                   AVG(r.rating) as recent_avg_rating
            FROM hotels h
            LEFT JOIN reviews r ON h.id = r.hotel_id AND r.created_at >= :thirty_days_ago
            WHERE h.is_active = 1 AND h.total_reviews >= 3
        """
        
        params = {'thirty_days_ago': thirty_days_ago}
        
        if location:
            trending_query += " AND h.location LIKE :location"
            params['location'] = f'%{location}%'
        
        trending_query += """
            GROUP BY h.id
            ORDER BY recent_reviews DESC, recent_avg_rating DESC, h.rating DESC
            LIMIT :limit
        """
        params['limit'] = limit
        
        result = db.session.execute(trending_query, params)
        
        trending_hotels = []
        for row in result:
            hotel = Hotel.query.get(row.id)
            if hotel:
                hotel_data = hotel.to_dict()
                hotel_data['trending_info'] = {
                    'recent_reviews': int(row.recent_reviews) if row.recent_reviews else 0,
                    'recent_avg_rating': round(float(row.recent_avg_rating), 2) if row.recent_avg_rating else 0
                }
                trending_hotels.append(hotel_data)
        
        return jsonify({
            'trending_hotels': trending_hotels,
            'location_filter': location,
            'total_found': len(trending_hotels)
        })
        
    except Exception as e:
        logger.error(f"Error getting trending hotels: {str(e)}")
        return jsonify({'error': 'Failed to get trending hotels'}), 500

@smart_recommendations_bp.route('/nearby', methods=['POST'])
def get_nearby_hotels():
    """Get hotels near a specific location using Google Maps"""
    try:
        print("[DEBUG] Getting nearby hotels")
        
        data = request.get_json()
        location = data.get('location')
        radius = data.get('radius', 5000)
        
        if not location:
            return jsonify({'error': 'Location is required'}), 400
        
        if not gmaps:
            return jsonify({'error': 'Google Maps API not configured'}), 500
        
        # Search for nearby hotels
        places_result = gmaps.places_nearby(
            location=location,
            radius=radius,
            type='lodging',
            language='en'
        )
        
        nearby_hotels = []
        
        for place in places_result.get('results', []):
            # Get detailed information
            place_details = gmaps.place(
                place_id=place['place_id'],
                fields=['name', 'formatted_address', 'rating', 'user_ratings_total', 'price_level', 'opening_hours', 'photos']
            )
            
            details = place_details.get('result', {})
            
            nearby_hotels.append({
                'name': details.get('name', 'Unknown'),
                'address': details.get('formatted_address', ''),
                'rating': details.get('rating', 0),
                'total_reviews': details.get('user_ratings_total', 0),
                'price_level': details.get('price_level'),
                'place_id': place['place_id'],
                'location': place['geometry']['location'],
                'is_open': details.get('opening_hours', {}).get('open_now', None)
            })
        
        return jsonify({
            'nearby_hotels': nearby_hotels,
            'search_location': location,
            'radius': radius,
            'total_found': len(nearby_hotels)
        })
        
    except Exception as e:
        logger.error(f"Error getting nearby hotels: {str(e)}")
        return jsonify({'error': 'Failed to get nearby hotels'}), 500
