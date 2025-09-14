"""
Data processing utilities for hotel recommendation system.
This module handles data cleaning, preprocessing, and transformation tasks.
"""

import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data processing tasks for the hotel recommendation system."""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and periods
        text = re.sub(r'[^\w\s.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text."""
        if not text:
            return []
        
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if word not in self.stop_words and len(word) >= min_length
        ]
        
        return list(set(keywords))  # Remove duplicates
    
    def normalize_ratings(self, ratings: List[float], target_scale: tuple = (1, 5)) -> List[float]:
        """Normalize ratings to a target scale."""
        if not ratings:
            return []
        
        ratings_array = np.array(ratings)
        min_rating = np.min(ratings_array)
        max_rating = np.max(ratings_array)
        
        if min_rating == max_rating:
            return [target_scale[1]] * len(ratings)
        
        # Normalize to 0-1 first
        normalized = (ratings_array - min_rating) / (max_rating - min_rating)
        
        # Scale to target range
        target_min, target_max = target_scale
        scaled = normalized * (target_max - target_min) + target_min
        
        return scaled.tolist()
    
    def process_amenities(self, amenities_list: List[str]) -> Dict[str, Any]:
        """Process and categorize hotel amenities."""
        if not amenities_list:
            return {'categories': {}, 'total': 0}
        
        amenity_categories = {
            'connectivity': ['wifi', 'internet', 'wireless'],
            'fitness': ['gym', 'fitness', 'pool', 'spa', 'sauna'],
            'dining': ['restaurant', 'bar', 'breakfast', 'room service', 'kitchen'],
            'business': ['business center', 'meeting rooms', 'conference'],
            'convenience': ['parking', 'concierge', 'laundry', 'elevator'],
            'entertainment': ['tv', 'cable', 'movies', 'games'],
            'comfort': ['air conditioning', 'heating', 'balcony', 'view']
        }
        
        categorized = {category: [] for category in amenity_categories}
        uncategorized = []
        
        for amenity in amenities_list:
            amenity_lower = amenity.lower()
            categorized_flag = False
            
            for category, keywords in amenity_categories.items():
                if any(keyword in amenity_lower for keyword in keywords):
                    categorized[category].append(amenity)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                uncategorized.append(amenity)
        
        return {
            'categories': categorized,
            'uncategorized': uncategorized,
            'total': len(amenities_list)
        }
    
    def calculate_similarity_score(self, item1: Dict, item2: Dict, 
                                 weights: Dict[str, float] = None) -> float:
        """Calculate similarity score between two items (hotels)."""
        if weights is None:
            weights = {
                'location': 0.3,
                'price_range': 0.2,
                'amenities': 0.3,
                'rating': 0.2
            }
        
        similarity_score = 0.0
        
        # Location similarity (simple string matching)
        if item1.get('location') and item2.get('location'):
            location1_words = set(self.clean_text(item1['location']).split())
            location2_words = set(self.clean_text(item2['location']).split())
            location_similarity = len(location1_words & location2_words) / max(len(location1_words | location2_words), 1)
            similarity_score += location_similarity * weights.get('location', 0)
        
        # Price range similarity
        if item1.get('price_range') == item2.get('price_range'):
            similarity_score += weights.get('price_range', 0)
        
        # Amenities similarity
        amenities1 = set(item1.get('amenities', []))
        amenities2 = set(item2.get('amenities', []))
        if amenities1 or amenities2:
            amenity_similarity = len(amenities1 & amenities2) / max(len(amenities1 | amenities2), 1)
            similarity_score += amenity_similarity * weights.get('amenities', 0)
        
        # Rating similarity (inverse of difference)
        rating1 = item1.get('rating', 0)
        rating2 = item2.get('rating', 0)
        if rating1 and rating2:
            rating_diff = abs(rating1 - rating2)
            rating_similarity = max(0, 1 - rating_diff / 5)  # Assuming 5-point scale
            similarity_score += rating_similarity * weights.get('rating', 0)
        
        return min(similarity_score, 1.0)  # Cap at 1.0
    
    def process_user_preferences(self, user_data: Dict) -> Dict[str, Any]:
        """Process and standardize user preference data."""
        preferences = user_data.get('preferences', {})
        
        processed = {
            'location_preference': preferences.get('location', '').lower(),
            'price_preference': preferences.get('price_range', 'medium'),
            'amenity_preferences': [
                amenity.lower() for amenity in preferences.get('amenities', [])
            ],
            'category_preference': preferences.get('category', 'any').lower()
        }
        
        # Add derived preferences
        if user_data.get('age'):
            age = user_data['age']
            if age < 30:
                processed['inferred_preferences'] = ['social', 'budget-friendly', 'technology']
            elif age < 50:
                processed['inferred_preferences'] = ['business', 'family-friendly', 'convenience']
            else:
                processed['inferred_preferences'] = ['luxury', 'comfort', 'service']
        else:
            processed['inferred_preferences'] = []
        
        return processed
    
    def create_feature_vector(self, hotel_data: Dict) -> np.ndarray:
        """Create a numerical feature vector for a hotel."""
        features = []
        
        # Rating (normalized to 0-1)
        rating = hotel_data.get('rating', 0) / 5.0
        features.append(rating)
        
        # Price range (encoded)
        price_map = {'budget': 0.2, 'low': 0.4, 'medium': 0.6, 'high': 0.8, 'luxury': 1.0}
        price_feature = price_map.get(hotel_data.get('price_range', 'medium'), 0.6)
        features.append(price_feature)
        
        # Number of amenities (normalized)
        amenity_count = len(hotel_data.get('amenities', []))
        amenity_feature = min(amenity_count / 20, 1.0)  # Assume max 20 amenities
        features.append(amenity_feature)
        
        # Review count (log-normalized)
        review_count = hotel_data.get('total_reviews', 0)
        review_feature = np.log(review_count + 1) / 10  # Log transform and normalize
        features.append(min(review_feature, 1.0))
        
        # Star rating (if available)
        star_rating = hotel_data.get('star_rating', 3) / 5.0
        features.append(star_rating)
        
        return np.array(features)
    
    def batch_process_hotels(self, hotels_data: List[Dict]) -> pd.DataFrame:
        """Process a batch of hotel data into a structured DataFrame."""
        processed_data = []
        
        for hotel in hotels_data:
            try:
                processed_hotel = {
                    'id': hotel.get('id'),
                    'name': hotel.get('name', ''),
                    'location': hotel.get('location', ''),
                    'rating': float(hotel.get('rating', 0)),
                    'price_range': hotel.get('price_range', 'medium'),
                    'total_reviews': int(hotel.get('total_reviews', 0)),
                    'amenities': hotel.get('amenities', []),
                    'amenity_count': len(hotel.get('amenities', [])),
                    'feature_vector': self.create_feature_vector(hotel).tolist(),
                    'processed_amenities': self.process_amenities(hotel.get('amenities', []))
                }
                
                # Add text features
                description = hotel.get('description', '')
                processed_hotel['description_keywords'] = self.extract_keywords(description)
                processed_hotel['description_length'] = len(description)
                
                processed_data.append(processed_hotel)
                
            except Exception as e:
                logger.error(f"Error processing hotel {hotel.get('id', 'unknown')}: {str(e)}")
                continue
        
        return pd.DataFrame(processed_data)
    
    def detect_outliers(self, data: List[float], method: str = 'iqr') -> List[int]:
        """Detect outliers in numerical data."""
        if not data or len(data) < 4:
            return []
        
        data_array = np.array(data)
        outlier_indices = []
        
        if method == 'iqr':
            q1 = np.percentile(data_array, 25)
            q3 = np.percentile(data_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_indices = [
                i for i, value in enumerate(data_array)
                if value < lower_bound or value > upper_bound
            ]
        
        elif method == 'zscore':
            z_scores = np.abs((data_array - np.mean(data_array)) / np.std(data_array))
            outlier_indices = [i for i, z in enumerate(z_scores) if z > 2.5]
        
        return outlier_indices
    
    def aggregate_reviews_by_time(self, reviews_data: List[Dict], 
                                time_window: str = 'month') -> Dict[str, Any]:
        """Aggregate review data by time periods."""
        if not reviews_data:
            return {}
        
        df = pd.DataFrame(reviews_data)
        
        if 'created_at' not in df.columns:
            return {}
        
        df['created_at'] = pd.to_datetime(df['created_at'])
        df.set_index('created_at', inplace=True)
        
        # Resample by time window
        if time_window == 'day':
            freq = 'D'
        elif time_window == 'week':
            freq = 'W'
        elif time_window == 'month':
            freq = 'M'
        else:
            freq = 'M'
        
        aggregated = df.resample(freq).agg({
            'rating': ['mean', 'count', 'std'],
            'sentiment_score': ['mean', 'std']
        }).fillna(0)
        
        return aggregated.to_dict()

# Global data processor instance
data_processor = DataProcessor()