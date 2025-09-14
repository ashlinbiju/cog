from app import db
from datetime import datetime

class Review(db.Model):
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    hotel_id = db.Column(db.Integer, db.ForeignKey('hotels.id'), nullable=False, index=True)
    rating = db.Column(db.Integer, nullable=False, index=True)  # 1-5 stars
    comment = db.Column(db.Text)
    sentiment_score = db.Column(db.Float, index=True)  # -1 to 1 (negative to positive)
    sentiment_label = db.Column(db.String(20), index=True)  # positive, negative, neutral
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_verified = db.Column(db.Boolean, default=False)  # For verified reviews
    helpful_count = db.Column(db.Integer, default=0)  # How many found this review helpful
    
    # Additional sentiment analysis fields
    compound_score = db.Column(db.Float)  # VADER compound score
    positive_score = db.Column(db.Float)  # VADER positive score
    negative_score = db.Column(db.Float)  # VADER negative score
    neutral_score = db.Column(db.Float)  # VADER neutral score
    
    # Constraints
    __table_args__ = (
        db.CheckConstraint('rating >= 1 AND rating <= 5', name='rating_range'),
        db.UniqueConstraint('user_id', 'hotel_id', name='unique_user_hotel_review'),
    )
    
    def __init__(self, **kwargs):
        super(Review, self).__init__(**kwargs)
        # Analyze sentiment when review is created
        if self.comment and not self.sentiment_score:
            self.analyze_sentiment()
    
    def analyze_sentiment(self):
        """Analyze sentiment of the review comment"""
        if not self.comment:
            return
        
        try:
            from services.sentiment_analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            
            # Get sentiment analysis
            result = analyzer.analyze_text(self.comment)
            
            self.sentiment_score = result.get('textblob_polarity', 0.0)
            self.compound_score = result.get('vader_compound', 0.0)
            self.positive_score = result.get('vader_positive', 0.0)
            self.negative_score = result.get('vader_negative', 0.0)
            self.neutral_score = result.get('vader_neutral', 0.0)
            
            # Determine sentiment label
            if self.sentiment_score > 0.1:
                self.sentiment_label = 'positive'
            elif self.sentiment_score < -0.1:
                self.sentiment_label = 'negative'
            else:
                self.sentiment_label = 'neutral'
                
        except ImportError:
            # Fallback if sentiment analyzer is not available
            self.sentiment_score = 0.0
            self.sentiment_label = 'neutral'
    
    def get_sentiment_emoji(self):
        """Get emoji representation of sentiment"""
        if self.sentiment_label == 'positive':
            return '😊'
        elif self.sentiment_label == 'negative':
            return '😞'
        else:
            return '😐'
    
    def get_rating_stars(self):
        """Get star representation of rating"""
        return '⭐' * self.rating + '☆' * (5 - self.rating)
    
    def is_helpful(self):
        """Check if review has significant helpful votes"""
        return self.helpful_count >= 3
    
    def update_helpful_count(self, increment=True):
        """Update helpful count"""
        if increment:
            self.helpful_count += 1
        else:
            self.helpful_count = max(0, self.helpful_count - 1)
        
        self.updated_at = datetime.utcnow()
        db.session.commit()
    
    def to_dict(self, include_user=False, include_hotel=False):
        """Convert review to dictionary"""
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'hotel_id': self.hotel_id,
            'rating': self.rating,
            'comment': self.comment,
            'sentiment_score': round(self.sentiment_score, 3) if self.sentiment_score else None,
            'sentiment_label': self.sentiment_label,
            'sentiment_emoji': self.get_sentiment_emoji(),
            'rating_stars': self.get_rating_stars(),
            'compound_score': round(self.compound_score, 3) if self.compound_score else None,
            'positive_score': round(self.positive_score, 3) if self.positive_score else None,
            'negative_score': round(self.negative_score, 3) if self.negative_score else None,
            'neutral_score': round(self.neutral_score, 3) if self.neutral_score else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_verified': self.is_verified,
            'helpful_count': self.helpful_count,
            'is_helpful': self.is_helpful()
        }
        
        if include_user and self.user:
            data['user'] = {
                'id': self.user.id,
                'username': self.user.username,
                'location': self.user.location
            }
        
        if include_hotel and self.hotel:
            data['hotel'] = {
                'id': self.hotel.id,
                'name': self.hotel.name,
                'location': self.hotel.location
            }
        
        return data
    
    @staticmethod
    def get_reviews_by_hotel(hotel_id, limit=None, min_rating=None, sentiment_filter=None):
        """Get reviews for a specific hotel with filters"""
        query = Review.query.filter_by(hotel_id=hotel_id)
        
        if min_rating:
            query = query.filter(Review.rating >= min_rating)
        
        if sentiment_filter:
            query = query.filter(Review.sentiment_label == sentiment_filter)
        
        query = query.order_by(db.desc(Review.created_at))
        
        if limit:
            return query.limit(limit).all()
        
        return query.all()
    
    @staticmethod
    def get_reviews_by_user(user_id, limit=None):
        """Get reviews by a specific user"""
        query = Review.query.filter_by(user_id=user_id).order_by(db.desc(Review.created_at))
        
        if limit:
            return query.limit(limit).all()
        
        return query.all()
    
    @staticmethod
    def get_recent_reviews(limit=10):
        """Get most recent reviews across all hotels"""
        return Review.query.order_by(db.desc(Review.created_at)).limit(limit).all()
    
    @staticmethod
    def get_sentiment_statistics():
        """Get overall sentiment statistics"""
        total_reviews = Review.query.count()
        positive_reviews = Review.query.filter(Review.sentiment_label == 'positive').count()
        negative_reviews = Review.query.filter(Review.sentiment_label == 'negative').count()
        neutral_reviews = Review.query.filter(Review.sentiment_label == 'neutral').count()
        
        if total_reviews == 0:
            return {
                'total': 0,
                'positive_percentage': 0,
                'negative_percentage': 0,
                'neutral_percentage': 0
            }
        
        return {
            'total': total_reviews,
            'positive_count': positive_reviews,
            'negative_count': negative_reviews,
            'neutral_count': neutral_reviews,
            'positive_percentage': round((positive_reviews / total_reviews) * 100, 1),
            'negative_percentage': round((negative_reviews / total_reviews) * 100, 1),
            'neutral_percentage': round((neutral_reviews / total_reviews) * 100, 1)
        }
    
    def __repr__(self):
        return f'<Review User:{self.user_id} Hotel:{self.hotel_id} Rating:{self.rating}>'