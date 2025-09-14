from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.collaborative_filtering import CollaborativeFilteringRecommendation
from services.sentiment_analyzer import SentimentAnalyzer
from models.user import User
from models.hotel import Hotel
from models.review import Review
import logging

recommendations_bp = Blueprint('recommendations', __name__)
logger = logging.getLogger(__name__)

# Global recommendation engine instance
recommendation_engine = CollaborativeFilteringRecommendation()

@recommendations_bp.route('/<int:user_id>', methods=['GET'])
@jwt_required()
def get_user_recommendations(user_id):
    """Get personalized recommendations for a user"""
    try:
        current_user_id = get_jwt_identity()
        
        # Users can only get their own recommendations (or implement admin check)
        if current_user_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Check if user exists
        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify({'error': 'User not found'}), 404
        
        # Get query parameters
        method = request.args.get('method', 'hybrid')
        limit = min(request.args.get('limit', 10, type=int), 50)
        
        # Validate method
        valid_methods = ['user_based', 'item_based', 'svd', 'hybrid', 'content_based', 'popular']
        if method not in valid_methods:
            method = 'hybrid'
        
        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(
            user_id=user_id,
            method=method,
            n_recommendations=limit
        )
        
        # Enhance recommendations with additional data
        enhanced_recommendations = []
        for rec in recommendations:
            hotel = Hotel.query.get(rec['hotel_id'])
            if hotel:
                enhanced_rec = rec.copy()
                enhanced_rec.update({
                    'hotel': hotel.to_dict(),
                    'sentiment_summary': hotel.get_sentiment_summary(),
                    'recent_reviews': [review.to_dict() for review in hotel.get_recent_reviews(3)]
                })
                enhanced_recommendations.append(enhanced_rec)
        
        return jsonify({
            'user_id': user_id,
            'method': method,
            'recommendations': enhanced_recommendations,
            'total_recommendations': len(enhanced_recommendations),
            'user_info': {
                'username': user.username,
                'location': user.location,
                'preferences': user.get_preferences()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@recommendations_bp.route('/retrain', methods=['POST'])
@jwt_required()
def retrain_models():
    """Retrain recommendation models"""
    try:
        # In production, this should be restricted to admin users
        # and probably run as a background task
        
        success = recommendation_engine.train_models()
        
        if success:
            return jsonify({
                'message': 'Models retrained successfully',
                'is_trained': recommendation_engine.is_trained
            })
        else:
            return jsonify({
                'error': 'Failed to retrain models',
                'is_trained': recommendation_engine.is_trained
            }), 500
        
    except Exception as e:
        logger.error(f"Error retraining models: {str(e)}")
        return jsonify({'error': 'Failed to retrain models'}), 500

@recommendations_bp.route('/similar-hotels/<int:hotel_id>', methods=['GET'])
def get_similar_hotels(hotel_id):
    """Get hotels similar to a given hotel"""
    try:
        hotel = Hotel.query.get(hotel_id)
        if not hotel or not hotel.is_active:
            return jsonify({'error': 'Hotel not found'}), 404
        
        limit = min(request.args.get('limit', 10, type=int), 20)
        
        # Train models if not already trained
        if not recommendation_engine.is_trained:
            recommendation_engine.train_models()
        
        similar_hotels = []
        
        if recommendation_engine.is_trained and recommendation_engine.item_similarity_df is not None:
            try:
                # Get item similarity scores for this hotel
                if hotel_id in recommendation_engine.item_similarity_df.index:
                    similarities = recommendation_engine.item_similarity_df.loc[hotel_id]
                    # Sort by similarity (excluding self)
                    similar_items = similarities.drop(hotel_id).sort_values(ascending=False)
                    
                    for similar_hotel_id, similarity_score in similar_items.head(limit).items():
                        similar_hotel = Hotel.query.get(similar_hotel_id)
                        if similar_hotel and similar_hotel.is_active:
                            similar_hotels.append({
                                'hotel': similar_hotel.to_dict(),
                                'similarity_score': round(similarity_score, 3),
                                'method': 'collaborative_filtering'
                            })
            except Exception as e:
                logger.warning(f"Collaborative filtering failed for hotel {hotel_id}: {str(e)}")
        
        # Fallback: Content-based similarity
        if len(similar_hotels) < limit:
            content_similar = _get_content_based_similar_hotels(hotel, limit - len(similar_hotels))
            similar_hotels.extend(content_similar)
        
        return jsonify({
            'hotel_id': hotel_id,
            'hotel_name': hotel.name,
            'similar_hotels': similar_hotels,
            'total_found': len(similar_hotels)
        })
        
    except Exception as e:
        logger.error(f"Error getting similar hotels for {hotel_id}: {str(e)}")
        return jsonify({'error': 'Failed to get similar hotels'}), 500

def _get_content_based_similar_hotels(target_hotel, limit):
    """Get content-based similar hotels"""
    try:
        # Find hotels with similar features
        candidate_hotels = Hotel.query.filter(
            Hotel.id != target_hotel.id,
            Hotel.is_active == True
        ).all()
        
        hotel_scores = []
        target_amenities = set(target_hotel.get_amenities())
        
        for hotel in candidate_hotels:
            score = 0
            
            # Location similarity (same city/state)
            if target_hotel.location.split(',')[0].strip() in hotel.location:
                score += 3
            
            # Price range similarity
            if hotel.price_range == target_hotel.price_range:
                score += 2
            
            # Amenity similarity
            hotel_amenities = set(hotel.get_amenities())
            common_amenities = target_amenities & hotel_amenities
            score += len(common_amenities) * 0.5
            
            # Rating similarity
            rating_diff = abs(hotel.rating - target_hotel.rating)
            if rating_diff <= 0.5:
                score += 2
            elif rating_diff <= 1.0:
                score += 1
            
            # Category similarity
            if hotel.category == target_hotel.category:
                score += 1
            
            if score > 0:
                hotel_scores.append({
                    'hotel': hotel.to_dict(),
                    'similarity_score': round(score, 2),
                    'method': 'content_based'
                })
        
        # Sort by score and return top results
        hotel_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        return hotel_scores[:limit]
        
    except Exception as e:
        logger.error(f"Error in content-based similarity: {str(e)}")
        return []

@recommendations_bp.route('/trending', methods=['GET'])
def get_trending_hotels():
    """Get trending/popular hotels"""
    try:
        limit = min(request.args.get('limit', 10, type=int), 20)
        location = request.args.get('location')
        
        # Get hotels with recent reviews and high ratings
        base_query = Hotel.query.filter_by(is_active=True).filter(
            Hotel.total_reviews >= 3
        )
        
        if location:
            base_query = base_query.filter(Hotel.location.ilike(f'%{location}%'))
        
        # Get trending hotels (high recent activity)
        trending_hotels = base_query.order_by(
            Hotel.rating.desc(),
            Hotel.total_reviews.desc()
        ).limit(limit).all()
        
        # Enhance with sentiment analysis
        trending_data = []
        for hotel in trending_hotels:
            hotel_data = hotel.to_dict()
            hotel_data['sentiment_summary'] = hotel.get_sentiment_summary()
            hotel_data['recent_reviews'] = [
                review.to_dict(include_user=True) for review in hotel.get_recent_reviews(3)
            ]
            trending_data.append(hotel_data)
        
        return jsonify({
            'trending_hotels': trending_data,
            'location_filter': location,
            'total_found': len(trending_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting trending hotels: {str(e)}")
        return jsonify({'error': 'Failed to get trending hotels'}), 500

@recommendations_bp.route('/evaluate/<int:user_id>', methods=['POST'])
@jwt_required()
def evaluate_recommendations(user_id):
    """Evaluate recommendation quality for a user"""
    try:
        current_user_id = get_jwt_identity()
        
        # Users can only evaluate their own recommendations
        if current_user_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json() or {}
        method = data.get('method', 'hybrid')
        
        # Evaluate recommendations
        if not recommendation_engine.is_trained:
            recommendation_engine.train_models()
        
        evaluation_results = recommendation_engine.evaluate_recommendations(user_id, method)
        
        if evaluation_results is None:
            return jsonify({
                'error': 'Insufficient data for evaluation',
                'user_id': user_id
            }), 400
        
        return jsonify({
            'user_id': user_id,
            'method': method,
            'evaluation_results': evaluation_results
        })
        
    except Exception as e:
        logger.error(f"Error evaluating recommendations for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to evaluate recommendations'}), 500

@recommendations_bp.route('/by-sentiment', methods=['GET'])
def get_recommendations_by_sentiment():
    """Get hotel recommendations based on sentiment analysis"""
    try:
        sentiment_type = request.args.get('sentiment', 'positive')  # positive, negative, neutral
        limit = min(request.args.get('limit', 10, type=int), 20)
        location = request.args.get('location')
        
        # Validate sentiment type
        if sentiment_type not in ['positive', 'negative', 'neutral']:
            sentiment_type = 'positive'
        
        # Get hotels with reviews of the specified sentiment
        base_query = """
            SELECT h.id, h.name, h.location, h.rating, h.total_reviews,
                   AVG(r.sentiment_score) as avg_sentiment,
                   COUNT(CASE WHEN r.sentiment_label = :sentiment_type THEN 1 END) as sentiment_count
            FROM hotels h
            JOIN reviews r ON h.id = r.hotel_id
            WHERE h.is_active = 1
        """
        
        params = {'sentiment_type': sentiment_type}
        
        if location:
            base_query += " AND h.location LIKE :location"
            params['location'] = f'%{location}%'
        
        base_query += """
            GROUP BY h.id
            HAVING COUNT(r.id) >= 3 AND sentiment_count > 0
            ORDER BY avg_sentiment DESC, sentiment_count DESC
            LIMIT :limit
        """
        params['limit'] = limit
        
        from app import db
        result = db.session.execute(base_query, params)
        
        recommendations = []
        for row in result:
            hotel = Hotel.query.get(row.id)
            if hotel:
                hotel_data = hotel.to_dict()
                hotel_data['sentiment_info'] = {
                    'average_sentiment': round(float(row.avg_sentiment), 3),
                    'sentiment_count': int(row.sentiment_count),
                    'sentiment_type': sentiment_type
                }
                recommendations.append(hotel_data)
        
        return jsonify({
            'recommendations': recommendations,
            'sentiment_filter': sentiment_type,
            'location_filter': location,
            'total_found': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment-based recommendations: {str(e)}")
        return jsonify({'error': 'Failed to get sentiment-based recommendations'}), 500

@recommendations_bp.route('/cold-start/<int:user_id>', methods=['GET'])
@jwt_required()
def get_cold_start_recommendations(user_id):
    """Get recommendations for new users (cold start problem)"""
    try:
        current_user_id = get_jwt_identity()
        
        if current_user_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify({'error': 'User not found'}), 404
        
        limit = min(request.args.get('limit', 10, type=int), 20)
        
        # Check if user is indeed a cold start case
        user_review_count = Review.query.filter_by(user_id=user_id).count()
        
        recommendations = []
        
        if user_review_count == 0:
            # True cold start - use popular hotels and user preferences
            user_preferences = user.get_preferences()
            
            base_query = Hotel.query.filter_by(is_active=True).filter(
                Hotel.total_reviews >= 5
            )
            
            # Filter by user location if available
            if user.location:
                location_hotels = base_query.filter(
                    Hotel.location.ilike(f'%{user.location}%')
                ).order_by(Hotel.rating.desc()).limit(limit // 2).all()
                recommendations.extend(location_hotels)
            
            # Filter by preferences
            if 'price_range' in user_preferences:
                price_hotels = base_query.filter(
                    Hotel.price_range == user_preferences['price_range']
                ).order_by(Hotel.rating.desc()).limit(limit // 2).all()
                recommendations.extend(price_hotels)
            
            # Fill remaining slots with top-rated hotels
            remaining_slots = limit - len(recommendations)
            if remaining_slots > 0:
                top_hotels = Hotel.get_top_rated_hotels(remaining_slots * 2)
                for hotel in top_hotels:
                    if hotel not in recommendations:
                        recommendations.append(hotel)
                        if len(recommendations) >= limit:
                            break
        
        else:
            # User has some reviews - use content-based recommendations
            content_recs = recommendation_engine.get_content_based_recommendations(
                user_id, limit
            )
            recommendations = [Hotel.query.get(rec['hotel_id']) for rec in content_recs]
            recommendations = [h for h in recommendations if h is not None]
        
        # Format response
        formatted_recommendations = []
        for i, hotel in enumerate(recommendations[:limit]):
            formatted_recommendations.append({
                'rank': i + 1,
                'hotel': hotel.to_dict(),
                'reason': 'Popular in your area' if user.location and user.location.lower() in hotel.location.lower() else 'Highly rated',
                'method': 'cold_start'
            })
        
        return jsonify({
            'user_id': user_id,
            'is_cold_start': user_review_count == 0,
            'user_review_count': user_review_count,
            'recommendations': formatted_recommendations,
            'total_recommendations': len(formatted_recommendations)
        })
        
    except Exception as e:
        logger.error(f"Error getting cold start recommendations for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to get cold start recommendations'}), 500

@recommendations_bp.route('/stats', methods=['GET'])
def get_recommendation_stats():
    """Get recommendation system statistics"""
    try:
        # Calculate various statistics
        total_users = User.query.filter_by(is_active=True).count()
        total_hotels = Hotel.query.filter_by(is_active=True).count()
        total_reviews = Review.query.count()
        
        # Users with reviews (not cold start)
        users_with_reviews = db.session.query(Review.user_id).distinct().count()
        cold_start_users = total_users - users_with_reviews
        
        # Average ratings
        avg_hotel_rating = db.session.query(db.func.avg(Hotel.rating)).scalar() or 0
        avg_review_rating = db.session.query(db.func.avg(Review.rating)).scalar() or 0
        
        # Sentiment distribution
        sentiment_stats = Review.get_sentiment_statistics()
        
        # Model training status
        model_status = {
            'is_trained': recommendation_engine.is_trained,
            'user_item_matrix_shape': None,
            'total_interactions': None
        }
        
        if recommendation_engine.is_trained and recommendation_engine.user_item_matrix is not None:
            model_status['user_item_matrix_shape'] = list(recommendation_engine.user_item_matrix.shape)
            model_status['total_interactions'] = int((recommendation_engine.user_item_matrix > 0).sum().sum())
        
        return jsonify({
            'system_stats': {
                'total_users': total_users,
                'total_hotels': total_hotels,
                'total_reviews': total_reviews,
                'users_with_reviews': users_with_reviews,
                'cold_start_users': cold_start_users,
                'avg_hotel_rating': round(avg_hotel_rating, 2),
                'avg_review_rating': round(avg_review_rating, 2)
            },
            'sentiment_stats': sentiment_stats,
            'model_status': model_status
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendation stats: {str(e)}")
        return jsonify({'error': 'Failed to get recommendation statistics'}), 500