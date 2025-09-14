import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from app import db
from models.user import User
from models.hotel import Hotel
from models.review import Review
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborativeFilteringRecommendation:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.svd_model = None
        self.user_knn_model = None
        self.item_knn_model = None
        self.is_trained = False
        
        # Hyperparameters
        self.n_components = 50  # For SVD
        self.n_neighbors = 10   # For KNN
        self.min_reviews_per_user = 2
        self.min_reviews_per_hotel = 2
    
    def load_data(self):
        """Load user-hotel ratings data from database"""
        try:
            # Query all reviews with user and hotel information
            reviews = db.session.query(Review, User, Hotel).join(
                User, Review.user_id == User.id
            ).join(
                Hotel, Review.hotel_id == Hotel.id
            ).filter(User.is_active == True, Hotel.is_active == True).all()
            
            # Create DataFrame
            data = []
            for review, user, hotel in reviews:
                data.append({
                    'user_id': user.id,
                    'hotel_id': hotel.id,
                    'rating': review.rating,
                    'username': user.username,
                    'hotel_name': hotel.name,
                    'user_location': user.location,
                    'hotel_location': hotel.location,
                    'sentiment_score': review.sentiment_score or 0.0
                })
            
            self.ratings_df = pd.DataFrame(data)
            
            if self.ratings_df.empty:
                logger.warning("No ratings data found")
                return False
            
            logger.info(f"Loaded {len(self.ratings_df)} ratings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def create_user_item_matrix(self):
        """Create user-item rating matrix"""
        try:
            # Filter users and hotels with minimum reviews
            user_counts = self.ratings_df['user_id'].value_counts()
            hotel_counts = self.ratings_df['hotel_id'].value_counts()
            
            valid_users = user_counts[user_counts >= self.min_reviews_per_user].index
            valid_hotels = hotel_counts[hotel_counts >= self.min_reviews_per_hotel].index
            
            # Filter DataFrame
            filtered_df = self.ratings_df[
                (self.ratings_df['user_id'].isin(valid_users)) &
                (self.ratings_df['hotel_id'].isin(valid_hotels))
            ]
            
            if filtered_df.empty:
                logger.warning("No data after filtering")
                return False
            
            # Create pivot table (user-item matrix)
            self.user_item_matrix = filtered_df.pivot_table(
                index='user_id',
                columns='hotel_id',
                values='rating',
                fill_value=0
            )
            
            logger.info(f"Created user-item matrix: {self.user_item_matrix.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating user-item matrix: {str(e)}")
            return False
    
    def compute_user_similarity(self):
        """Compute user-based similarity matrix"""
        try:
            # Convert to numpy array for efficiency
            user_ratings = self.user_item_matrix.values
            
            # Compute cosine similarity between users
            self.user_similarity_matrix = cosine_similarity(user_ratings)
            
            # Convert back to DataFrame for easier indexing
            self.user_similarity_df = pd.DataFrame(
                self.user_similarity_matrix,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.index
            )
            
            logger.info("Computed user similarity matrix")
            return True
            
        except Exception as e:
            logger.error(f"Error computing user similarity: {str(e)}")
            return False
    
    def compute_item_similarity(self):
        """Compute item-based similarity matrix"""
        try:
            # Transpose matrix for item-item similarity
            item_ratings = self.user_item_matrix.T.values
            
            # Compute cosine similarity between items (hotels)
            self.item_similarity_matrix = cosine_similarity(item_ratings)
            
            # Convert back to DataFrame
            self.item_similarity_df = pd.DataFrame(
                self.item_similarity_matrix,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
            
            logger.info("Computed item similarity matrix")
            return True
            
        except Exception as e:
            logger.error(f"Error computing item similarity: {str(e)}")
            return False
    
    def train_svd_model(self):
        """Train SVD model for matrix factorization"""
        try:
            # Prepare data for SVD
            user_ratings = self.user_item_matrix.values
            
            # Initialize and fit SVD
            self.svd_model = TruncatedSVD(
                n_components=min(self.n_components, min(user_ratings.shape) - 1),
                random_state=42
            )
            
            self.user_factors = self.svd_model.fit_transform(user_ratings)
            self.item_factors = self.svd_model.components_.T
            
            logger.info(f"Trained SVD model with {self.svd_model.n_components} components")
            return True
            
        except Exception as e:
            logger.error(f"Error training SVD model: {str(e)}")
            return False
    
    def train_knn_models(self):
        """Train KNN models for user and item based recommendations"""
        try:
            # User-based KNN
            self.user_knn_model = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, len(self.user_item_matrix)),
                metric='cosine'
            )
            self.user_knn_model.fit(self.user_item_matrix.values)
            
            # Item-based KNN
            self.item_knn_model = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, len(self.user_item_matrix.columns)),
                metric='cosine'
            )
            self.item_knn_model.fit(self.user_item_matrix.T.values)
            
            logger.info("Trained KNN models")
            return True
            
        except Exception as e:
            logger.error(f"Error training KNN models: {str(e)}")
            return False
    
    def train_models(self):
        """Train all recommendation models"""
        logger.info("Starting model training...")
        
        if not self.load_data():
            return False
        
        if not self.create_user_item_matrix():
            return False
        
        if not self.compute_user_similarity():
            return False
        
        if not self.compute_item_similarity():
            return False
        
        if not self.train_svd_model():
            return False
        
        if not self.train_knn_models():
            return False
        
        self.is_trained = True
        logger.info("Model training completed successfully")
        return True
    
    def get_user_based_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using user-based collaborative filtering"""
        if not self.is_trained:
            if not self.train_models():
                return []
        
        try:
            if user_id not in self.user_item_matrix.index:
                # Handle cold start problem with popular hotels
                return self.get_popular_hotels(n_recommendations)
            
            # Find similar users
            user_similarities = self.user_similarity_df.loc[user_id]
            similar_users = user_similarities.sort_values(ascending=False)[1:self.n_neighbors+1]
            
            # Get user's ratings
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_hotels = user_ratings[user_ratings == 0].index
            
            # Calculate predicted ratings for unrated hotels
            hotel_scores = {}
            
            for hotel_id in unrated_hotels:
                numerator = 0
                denominator = 0
                
                for similar_user_id, similarity in similar_users.items():
                    if similarity > 0 and self.user_item_matrix.loc[similar_user_id, hotel_id] > 0:
                        rating = self.user_item_matrix.loc[similar_user_id, hotel_id]
                        numerator += similarity * rating
                        denominator += abs(similarity)
                
                if denominator > 0:
                    predicted_rating = numerator / denominator
                    hotel_scores[hotel_id] = predicted_rating
            
            # Sort and return top recommendations
            sorted_hotels = sorted(hotel_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = []
            
            for hotel_id, score in sorted_hotels[:n_recommendations]:
                hotel = Hotel.query.get(hotel_id)
                if hotel:
                    recommendations.append({
                        'hotel_id': hotel_id,
                        'predicted_rating': round(score, 2),
                        'hotel_name': hotel.name,
                        'hotel_location': hotel.location,
                        'actual_rating': hotel.rating,
                        'method': 'user_based'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in user-based recommendations: {str(e)}")
            return []
    
    def get_item_based_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using item-based collaborative filtering"""
        if not self.is_trained:
            if not self.train_models():
                return []
        
        try:
            if user_id not in self.user_item_matrix.index:
                return self.get_popular_hotels(n_recommendations)
            
            # Get user's ratings
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_hotels = user_ratings[user_ratings > 0].index
            unrated_hotels = user_ratings[user_ratings == 0].index
            
            # Calculate predicted ratings for unrated hotels
            hotel_scores = {}
            
            for hotel_id in unrated_hotels:
                numerator = 0
                denominator = 0
                
                # Find similar hotels that user has rated
                hotel_similarities = self.item_similarity_df.loc[hotel_id]
                
                for rated_hotel_id in rated_hotels:
                    similarity = hotel_similarities[rated_hotel_id]
                    if similarity > 0:
                        user_rating = user_ratings[rated_hotel_id]
                        numerator += similarity * user_rating
                        denominator += abs(similarity)
                
                if denominator > 0:
                    predicted_rating = numerator / denominator
                    hotel_scores[hotel_id] = predicted_rating
            
            # Sort and return top recommendations
            sorted_hotels = sorted(hotel_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = []
            
            for hotel_id, score in sorted_hotels[:n_recommendations]:
                hotel = Hotel.query.get(hotel_id)
                if hotel:
                    recommendations.append({
                        'hotel_id': hotel_id,
                        'predicted_rating': round(score, 2),
                        'hotel_name': hotel.name,
                        'hotel_location': hotel.location,
                        'actual_rating': hotel.rating,
                        'method': 'item_based'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in item-based recommendations: {str(e)}")
            return []
    
    def get_svd_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using SVD matrix factorization"""
        if not self.is_trained:
            if not self.train_models():
                return []
        
        try:
            if user_id not in self.user_item_matrix.index:
                return self.get_popular_hotels(n_recommendations)
            
            # Get user index
            user_idx = list(self.user_item_matrix.index).index(user_id)
            
            # Predict ratings for all hotels
            user_factor = self.user_factors[user_idx]
            predicted_ratings = np.dot(user_factor, self.item_factors.T)
            
            # Get user's actual ratings
            user_ratings = self.user_item_matrix.loc[user_id]
            
            # Only recommend unrated hotels
            hotel_scores = {}
            for idx, hotel_id in enumerate(self.user_item_matrix.columns):
                if user_ratings[hotel_id] == 0:  # Unrated
                    hotel_scores[hotel_id] = predicted_ratings[idx]
            
            # Sort and return top recommendations
            sorted_hotels = sorted(hotel_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = []
            
            for hotel_id, score in sorted_hotels[:n_recommendations]:
                hotel = Hotel.query.get(hotel_id)
                if hotel:
                    recommendations.append({
                        'hotel_id': hotel_id,
                        'predicted_rating': round(score, 2),
                        'hotel_name': hotel.name,
                        'hotel_location': hotel.location,
                        'actual_rating': hotel.rating,
                        'method': 'svd'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in SVD recommendations: {str(e)}")
            return []
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=10, weights=None):
        """Get hybrid recommendations combining multiple methods"""
        if weights is None:
            weights = {'user_based': 0.3, 'item_based': 0.3, 'svd': 0.4}
        
        try:
            # Get recommendations from different methods
            user_recs = self.get_user_based_recommendations(user_id, n_recommendations * 2)
            item_recs = self.get_item_based_recommendations(user_id, n_recommendations * 2)
            svd_recs = self.get_svd_recommendations(user_id, n_recommendations * 2)
            
            # Combine scores
            combined_scores = {}
            
            # User-based scores
            for rec in user_recs:
                hotel_id = rec['hotel_id']
                score = rec['predicted_rating']
                combined_scores[hotel_id] = combined_scores.get(hotel_id, 0) + score * weights['user_based']
            
            # Item-based scores
            for rec in item_recs:
                hotel_id = rec['hotel_id']
                score = rec['predicted_rating']
                combined_scores[hotel_id] = combined_scores.get(hotel_id, 0) + score * weights['item_based']
            
            # SVD scores
            for rec in svd_recs:
                hotel_id = rec['hotel_id']
                score = rec['predicted_rating']
                combined_scores[hotel_id] = combined_scores.get(hotel_id, 0) + score * weights['svd']
            
            # Sort and create final recommendations
            sorted_hotels = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = []
            
            for hotel_id, score in sorted_hotels[:n_recommendations]:
                hotel = Hotel.query.get(hotel_id)
                if hotel:
                    recommendations.append({
                        'hotel_id': hotel_id,
                        'predicted_rating': round(score, 2),
                        'hotel_name': hotel.name,
                        'hotel_location': hotel.location,
                        'actual_rating': hotel.rating,
                        'method': 'hybrid'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {str(e)}")
            return []
    
    def get_popular_hotels(self, n_recommendations=10):
        """Get popular hotels as fallback for cold start problem"""
        try:
            popular_hotels = Hotel.query.filter_by(is_active=True).filter(
                Hotel.total_reviews >= 3
            ).order_by(Hotel.rating.desc(), Hotel.total_reviews.desc()).limit(n_recommendations).all()
            
            recommendations = []
            for hotel in popular_hotels:
                recommendations.append({
                    'hotel_id': hotel.id,
                    'predicted_rating': hotel.rating,
                    'hotel_name': hotel.name,
                    'hotel_location': hotel.location,
                    'actual_rating': hotel.rating,
                    'method': 'popular'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting popular hotels: {str(e)}")
            return []
    
    def get_content_based_recommendations(self, user_id, n_recommendations=10):
        """Get content-based recommendations using user preferences and hotel features"""
        try:
            user = User.query.get(user_id)
            if not user:
                return []
            
            user_preferences = user.get_preferences()
            user_reviews = Review.query.filter_by(user_id=user_id).all()
            
            # Handle cold start problem - use preferences if no reviews
            if not user_reviews:
                return self.get_preference_based_recommendations(user_id, n_recommendations)
            
            # Analyze user's preferred hotel features
            preferred_locations = []
            preferred_price_ranges = []
            high_rated_amenities = []
            for review in user_reviews:
                if review.rating >= 4:  # Consider only high-rated hotels
                    hotel = review.hotel
                    preferred_locations.append(hotel.location)
                    preferred_price_ranges.append(hotel.price_range)
                    high_rated_amenities.extend(hotel.get_amenities())
            
            # Find hotels with similar features
            candidate_hotels = Hotel.query.filter_by(is_active=True).all()
            hotel_scores = {}
            
            for hotel in candidate_hotels:
                # Skip if user already reviewed this hotel
                if any(r.hotel_id == hotel.id for r in user_reviews):
                    continue
                
                score = 0
                
                # Location similarity
                if any(loc in hotel.location for loc in preferred_locations):
                    score += 2
                
                # Price range similarity
                if hotel.price_range in preferred_price_ranges:
                    score += 1.5
                
                # Amenity similarity
                hotel_amenities = hotel.get_amenities()
                common_amenities = set(high_rated_amenities) & set(hotel_amenities)
                score += len(common_amenities) * 0.5
                
                # Hotel rating boost
                score += hotel.rating * 0.3
                
                # User preference matching
                if 'location' in user_preferences:
                    if user_preferences['location'].lower() in hotel.location.lower():
                        score += 1
                
                if 'price_range' in user_preferences:
                    if user_preferences['price_range'] == hotel.price_range:
                        score += 1
                
                hotel_scores[hotel.id] = score
            
            # Sort and return top recommendations
            sorted_hotels = sorted(hotel_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = []
            
            for hotel_id, score in sorted_hotels[:n_recommendations]:
                hotel = Hotel.query.get(hotel_id)
                if hotel and score > 0:
                    recommendations.append({
                        'hotel_id': hotel_id,
                        'predicted_rating': min(5.0, hotel.rating + (score * 0.1)),
                        'hotel_name': hotel.name,
                        'hotel_location': hotel.location,
                        'actual_rating': hotel.rating,
                        'content_score': round(score, 2),
                        'method': 'content_based'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {str(e)}")
            return []
    
    def get_preference_based_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations based purely on user preferences (cold start solution)"""
        try:
            user = User.query.get(user_id)
            if not user:
                return []
            
            user_preferences = user.get_preferences()
            
            # Get all active hotels
            candidate_hotels = Hotel.query.filter_by(is_active=True).all()
            hotel_scores = {}
            
            for hotel in candidate_hotels:
                score = 0
                
                # Base score from hotel rating and reviews
                score += hotel.rating * 2
                score += min(hotel.total_reviews / 100, 2)  # Cap review bonus at 2 points
                
                # Budget preference matching
                budget_category = user_preferences.get('budget_category', [])
                if budget_category:
                    hotel_price_level = getattr(hotel, 'price_level', 2)  # Default to mid-range
                    if 'budget' in budget_category and hotel_price_level <= 1:
                        score += 3
                    elif 'mid-range' in budget_category and hotel_price_level == 2:
                        score += 3
                    elif 'luxury' in budget_category and hotel_price_level == 3:
                        score += 3
                    elif 'ultra-luxury' in budget_category and hotel_price_level >= 4:
                        score += 3
                
                # Travel purpose matching
                travel_purpose = user_preferences.get('travel_purpose', [])
                hotel_types = getattr(hotel, 'hotel_type', '').lower()
                
                if 'business' in travel_purpose and ('business' in hotel_types or 'conference' in hotel_types):
                    score += 2
                elif 'leisure' in travel_purpose and ('resort' in hotel_types or 'vacation' in hotel_types):
                    score += 2
                elif 'family' in travel_purpose and ('family' in hotel_types or 'kid' in hotel_types):
                    score += 2
                elif 'romantic' in travel_purpose and ('romantic' in hotel_types or 'couples' in hotel_types):
                    score += 2
                
                # Location preference matching
                location_prefs = user_preferences.get('location_preferences', [])
                hotel_location = hotel.location.lower()
                
                for pref in location_prefs:
                    if pref == 'city-center' and ('downtown' in hotel_location or 'center' in hotel_location):
                        score += 2
                    elif pref == 'beach' and ('beach' in hotel_location or 'oceanfront' in hotel_location):
                        score += 2
                    elif pref == 'airport' and ('airport' in hotel_location):
                        score += 2
                    elif pref == 'quiet' and ('quiet' in hotel_location or 'peaceful' in hotel_location):
                        score += 2
                
                # Amenity preferences
                preferred_amenities = user_preferences.get('amenities', [])
                hotel_amenities = hotel.get_amenities() if hasattr(hotel, 'get_amenities') else []
                
                for amenity in preferred_amenities:
                    if amenity in hotel_amenities:
                        score += 1
                
                # Rating threshold
                min_rating = user_preferences.get('min_rating', 3.0)
                if hotel.rating < min_rating:
                    score *= 0.5  # Penalize hotels below preferred rating
                
                # Review count threshold
                min_reviews = user_preferences.get('min_reviews', 10)
                if hotel.total_reviews < min_reviews:
                    score *= 0.7  # Penalize hotels with few reviews
                
                hotel_scores[hotel.id] = score
            
            # Sort and return top recommendations
            sorted_hotels = sorted(hotel_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = []
            
            for hotel_id, score in sorted_hotels[:n_recommendations]:
                hotel = Hotel.query.get(hotel_id)
                if hotel and score > 0:
                    recommendations.append({
                        'hotel_id': hotel_id,
                        'predicted_rating': min(5.0, hotel.rating + (score * 0.05)),
                        'hotel_name': hotel.name,
                        'hotel_location': hotel.location,
                        'actual_rating': hotel.rating,
                        'preference_score': round(score, 2),
                        'method': 'preference_based'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in preference-based recommendations: {str(e)}")
            return self.get_popular_hotels(n_recommendations)
    
    def get_recommendations(self, user_id, method='hybrid', n_recommendations=10):
        """Get recommendations using specified method"""
        methods = {
            'user_based': self.get_user_based_recommendations,
            'item_based': self.get_item_based_recommendations,
            'svd': self.get_svd_recommendations,
            'hybrid': self.get_hybrid_recommendations,
            'content_based': self.get_content_based_recommendations,
            'popular': self.get_popular_hotels
        }
        
        if method not in methods:
            method = 'hybrid'
        
        if method == 'popular':
            return methods[method](n_recommendations)
        else:
            return methods[method](user_id, n_recommendations)
    
    def evaluate_recommendations(self, test_user_id, method='hybrid'):
        """Evaluate recommendation quality for a test user"""
        if test_user_id not in self.user_item_matrix.index:
            return None
        
        # Get user's actual ratings
        actual_ratings = self.user_item_matrix.loc[test_user_id]
        rated_hotels = actual_ratings[actual_ratings > 0]
        
        if len(rated_hotels) < 2:
            return None
        
        # Hide some ratings for testing
        test_hotels = rated_hotels.sample(min(3, len(rated_hotels)))
        
        # Get recommendations
        recommendations = self.get_recommendations(test_user_id, method, len(test_hotels) * 2)
        
        # Calculate metrics
        recommended_hotels = [r['hotel_id'] for r in recommendations]
        hits = len(set(test_hotels.index) & set(recommended_hotels))
        
        precision = hits / len(recommended_hotels) if recommended_hotels else 0
        recall = hits / len(test_hotels) if test_hotels else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'hits': hits,
            'total_recommendations': len(recommended_hotels),
            'total_relevant': len(test_hotels)
        }