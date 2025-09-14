from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.hotel import Hotel
from models.review import Review
from models.user import User
from services.sentiment_analyzer import SentimentAnalyzer
from services.api_service import api_service
from app import db
from datetime import datetime
import logging

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

@api_bp.route('/hotels', methods=['GET'])
def get_hotels():
    """Get all hotels with optional filters"""
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        location = request.args.get('location')
        price_range = request.args.get('price_range')
        min_rating = request.args.get('min_rating', type=float)
        query = request.args.get('query')
        amenities = request.args.getlist('amenities')
        
        # Search hotels
        if any([query, location, price_range, min_rating, amenities]):
            hotels = Hotel.search_hotels(
                query=query,
                location=location,
                price_range=price_range,
                min_rating=min_rating,
                amenities=amenities
            )
            
            # Paginate results
            start = (page - 1) * per_page
            end = start + per_page
            paginated_hotels = hotels[start:end]
            
            return jsonify({
                'hotels': [hotel.to_dict() for hotel in paginated_hotels],
                'total': len(hotels),
                'page': page,
                'per_page': per_page,
                'pages': (len(hotels) - 1) // per_page + 1 if hotels else 0
            })
        else:
            # Get all hotels with pagination
            hotels_query = Hotel.query.filter_by(is_active=True).order_by(Hotel.rating.desc())
            paginated = hotels_query.paginate(
                page=page,
                per_page=per_page,
                error_out=False
            )
            
            return jsonify({
                'hotels': [hotel.to_dict() for hotel in paginated.items],
                'total': paginated.total,
                'page': page,
                'per_page': per_page,
                'pages': paginated.pages
            })
            
    except Exception as e:
        logger.error(f"Error getting hotels: {str(e)}")
        return jsonify({'error': 'Failed to retrieve hotels'}), 500

@api_bp.route('/hotels/<int:hotel_id>', methods=['GET'])
def get_hotel(hotel_id):
    """Get specific hotel details"""
    try:
        hotel = Hotel.query.get_or_404(hotel_id)
        
        if not hotel.is_active:
            return jsonify({'error': 'Hotel not found'}), 404
        
        # Include recent reviews in response
        hotel_data = hotel.to_dict(include_reviews=True)
        
        return jsonify({'hotel': hotel_data})
        
    except Exception as e:
        logger.error(f"Error getting hotel {hotel_id}: {str(e)}")
        return jsonify({'error': 'Failed to retrieve hotel'}), 500

@api_bp.route('/hotels/<int:hotel_id>/reviews', methods=['GET'])
def get_hotel_reviews(hotel_id):
    """Get reviews for a specific hotel"""
    try:
        hotel = Hotel.query.get_or_404(hotel_id)
        
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        min_rating = request.args.get('min_rating', type=int)
        sentiment_filter = request.args.get('sentiment')
        
        # Get reviews with filters
        reviews = Review.get_reviews_by_hotel(
            hotel_id, 
            min_rating=min_rating, 
            sentiment_filter=sentiment_filter
        )
        
        # Paginate results
        start = (page - 1) * per_page
        end = start + per_page
        paginated_reviews = reviews[start:end]
        
        return jsonify({
            'reviews': [review.to_dict(include_user=True) for review in paginated_reviews],
            'total': len(reviews),
            'page': page,
            'per_page': per_page,
            'pages': (len(reviews) - 1) // per_page + 1 if reviews else 0,
            'hotel': {
                'id': hotel.id,
                'name': hotel.name,
                'location': hotel.location
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting reviews for hotel {hotel_id}: {str(e)}")
        return jsonify({'error': 'Failed to retrieve reviews'}), 500

@api_bp.route('/reviews', methods=['POST'])
@jwt_required()
def create_review():
    """Create a new review"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['hotel_id', 'rating', 'comment']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        hotel_id = data['hotel_id']
        rating = data['rating']
        comment = data['comment']
        
        # Validate data
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({'error': 'Rating must be an integer between 1 and 5'}), 400
        
        # Check if hotel exists
        hotel = Hotel.query.get(hotel_id)
        if not hotel or not hotel.is_active:
            return jsonify({'error': 'Hotel not found'}), 404
        
        # Check if user already reviewed this hotel
        existing_review = Review.query.filter_by(user_id=user_id, hotel_id=hotel_id).first()
        if existing_review:
            return jsonify({'error': 'You have already reviewed this hotel'}), 400
        
        # Create new review
        review = Review(
            user_id=user_id,
            hotel_id=hotel_id,
            rating=rating,
            comment=comment.strip()
        )
        
        db.session.add(review)
        db.session.commit()
        
        # Update hotel rating
        hotel.update_rating()
        
        return jsonify({
            'message': 'Review created successfully',
            'review': review.to_dict(include_user=True, include_hotel=True)
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating review: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create review'}), 500

@api_bp.route('/reviews/<int:review_id>', methods=['PUT'])
@jwt_required()
def update_review(review_id):
    """Update an existing review"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Get review
        review = Review.query.get_or_404(review_id)
        
        # Check if user owns this review
        if review.user_id != user_id:
            return jsonify({'error': 'Unauthorized to update this review'}), 403
        
        # Update fields
        if 'rating' in data:
            rating = data['rating']
            if not isinstance(rating, int) or rating < 1 or rating > 5:
                return jsonify({'error': 'Rating must be an integer between 1 and 5'}), 400
            review.rating = rating
        
        if 'comment' in data:
            review.comment = data['comment'].strip()
            # Re-analyze sentiment
            review.analyze_sentiment()
        
        review.updated_at = datetime.utcnow()
        db.session.commit()
        
        # Update hotel rating
        review.hotel.update_rating()
        
        return jsonify({
            'message': 'Review updated successfully',
            'review': review.to_dict(include_user=True, include_hotel=True)
        })
        
    except Exception as e:
        logger.error(f"Error updating review {review_id}: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update review'}), 500

@api_bp.route('/reviews/<int:review_id>', methods=['DELETE'])
@jwt_required()
def delete_review(review_id):
    """Delete a review"""
    try:
        user_id = get_jwt_identity()
        
        # Get review
        review = Review.query.get_or_404(review_id)
        
        # Check if user owns this review
        if review.user_id != user_id:
            return jsonify({'error': 'Unauthorized to delete this review'}), 403
        
        hotel = review.hotel
        db.session.delete(review)
        db.session.commit()
        
        # Update hotel rating
        hotel.update_rating()
        
        return jsonify({'message': 'Review deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting review {review_id}: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete review'}), 500

@api_bp.route('/reviews/<int:review_id>/helpful', methods=['POST'])
@jwt_required()
def mark_review_helpful(review_id):
    """Mark a review as helpful"""
    try:
        review = Review.query.get_or_404(review_id)
        review.update_helpful_count(increment=True)
        
        return jsonify({
            'message': 'Review marked as helpful',
            'helpful_count': review.helpful_count
        })
        
    except Exception as e:
        logger.error(f"Error marking review {review_id} as helpful: {str(e)}")
        return jsonify({'error': 'Failed to mark review as helpful'}), 500

@api_bp.route('/sentiment/<int:hotel_id>', methods=['GET'])
def get_hotel_sentiment(hotel_id):
    """Get sentiment analysis for a hotel"""
    try:
        hotel = Hotel.query.get_or_404(hotel_id)
        
        # Get reviews for sentiment analysis
        reviews = Review.query.filter_by(hotel_id=hotel_id).all()
        
        if not reviews:
            return jsonify({
                'hotel_id': hotel_id,
                'sentiment_summary': {
                    'average_sentiment': 0.0,
                    'total_reviews': 0,
                    'positive_reviews': 0,
                    'negative_reviews': 0,
                    'neutral_reviews': 0
                }
            })
        
        # Analyze sentiment
        analyzer = SentimentAnalyzer()
        sentiment_summary = analyzer.get_hotel_sentiment_summary(reviews)
        
        # Get aspect-based sentiment for recent reviews
        recent_reviews = reviews[-10:] if len(reviews) > 10 else reviews
        aspect_sentiments = {}
        
        for review in recent_reviews:
            if review.comment:
                aspects = analyzer.get_aspect_based_sentiment(review.comment)
                for aspect, sentiment in aspects.items():
                    if aspect not in aspect_sentiments:
                        aspect_sentiments[aspect] = []
                    aspect_sentiments[aspect].append(sentiment['score'])
        
        # Average aspect sentiments
        avg_aspect_sentiments = {}
        for aspect, scores in aspect_sentiments.items():
            avg_aspect_sentiments[aspect] = {
                'average_score': round(sum(scores) / len(scores), 3),
                'review_count': len(scores)
            }
        
        return jsonify({
            'hotel_id': hotel_id,
            'hotel_name': hotel.name,
            'sentiment_summary': sentiment_summary,
            'aspect_sentiments': avg_aspect_sentiments
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment for hotel {hotel_id}: {str(e)}")
        return jsonify({'error': 'Failed to analyze sentiment'}), 500

@api_bp.route('/users', methods=['POST'])
def create_user():
    """Create a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        username = data['username']
        email = data['email']
        password = data['password']
        
        # Check if user already exists
        if User.get_user_by_username(username):
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.get_user_by_email(email):
            return jsonify({'error': 'Email already exists'}), 400
        
        # Create user
        user = User.create_user(
            username=username,
            email=email,
            password=password,
            age=data.get('age'),
            location=data.get('location')
        )
        
        # Set preferences if provided
        if 'preferences' in data:
            user.set_preferences(data['preferences'])
            db.session.commit()
        
        return jsonify({
            'message': 'User created successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create user'}), 500

@api_bp.route('/users/<int:user_id>', methods=['GET'])
@jwt_required()
def get_user(user_id):
    """Get user information"""
    try:
        current_user_id = get_jwt_identity()
        
        # Users can only view their own profile (or implement admin check)
        if current_user_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        user = User.query.get_or_404(user_id)
        
        if not user.is_active:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({'user': user.to_dict()})
        
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to retrieve user'}), 500

@api_bp.route('/users/<int:user_id>/reviews', methods=['GET'])
@jwt_required()
def get_user_reviews(user_id):
    """Get reviews by a specific user"""
    try:
        current_user_id = get_jwt_identity()
        
        # Users can only view their own reviews (or implement admin check)
        if current_user_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        user = User.query.get_or_404(user_id)
        
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        # Get user reviews
        reviews = Review.get_reviews_by_user(user_id)
        
        # Paginate results
        start = (page - 1) * per_page
        end = start + per_page
        paginated_reviews = reviews[start:end]
        
        return jsonify({
            'reviews': [review.to_dict(include_hotel=True) for review in paginated_reviews],
            'total': len(reviews),
            'page': page,
            'per_page': per_page,
            'pages': (len(reviews) - 1) // per_page + 1 if reviews else 0,
            'user': {
                'id': user.id,
                'username': user.username
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting reviews for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to retrieve user reviews'}), 500

@api_bp.route('/sync/hotels', methods=['POST'])
def sync_external_hotels():
    """Sync hotels from external API"""
    try:
        data = request.get_json() or {}
        location = data.get('location')
        
        # Sync hotels
        synced_count = api_service.sync_hotels_with_database(location)
        
        return jsonify({
            'message': f'Successfully synced {synced_count} hotels',
            'synced_count': synced_count
        })
        
    except Exception as e:
        logger.error(f"Error syncing external hotels: {str(e)}")
        return jsonify({'error': 'Failed to sync hotels'}), 500

@api_bp.route('/search/locations', methods=['GET'])
def search_locations():
    """Get location suggestions for autocomplete"""
    try:
        query = request.args.get('q', '')
        
        if not query or len(query) < 2:
            return jsonify({'suggestions': []})
        
        # Get suggestions from external API
        suggestions = api_service.get_location_suggestions(query)
        
        # Also get suggestions from local database
        local_hotels = Hotel.query.filter(
            Hotel.location.ilike(f'%{query}%'),
            Hotel.is_active == True
        ).limit(5).all()
        
        local_suggestions = [{'name': hotel.location, 'type': 'city'} for hotel in local_hotels]
        
        # Combine and deduplicate
        all_suggestions = suggestions + local_suggestions
        unique_suggestions = []
        seen = set()
        
        for suggestion in all_suggestions:
            name = suggestion.get('name', '')
            if name and name not in seen:
                seen.add(name)
                unique_suggestions.append(suggestion)
        
        return jsonify({'suggestions': unique_suggestions[:10]})
        
    except Exception as e:
        logger.error(f"Error searching locations: {str(e)}")
        return jsonify({'error': 'Failed to search locations'}), 500

@api_bp.route('/stats/reviews', methods=['GET'])
def get_review_stats():
    """Get overall review statistics"""
    try:
        stats = Review.get_sentiment_statistics()
        
        # Additional stats
        total_hotels = Hotel.query.filter_by(is_active=True).count()
        total_users = User.query.filter_by(is_active=True).count()
        recent_reviews = Review.get_recent_reviews(10)
        
        return jsonify({
            'sentiment_stats': stats,
            'total_hotels': total_hotels,
            'total_users': total_users,
            'recent_reviews': [review.to_dict(include_user=True, include_hotel=True) for review in recent_reviews]
        })
        
    except Exception as e:
        logger.error(f"Error getting review stats: {str(e)}")
        return jsonify({'error': 'Failed to get statistics'}), 500