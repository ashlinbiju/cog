from app import db
from datetime import datetime
import json
from sqlalchemy import func

class Hotel(db.Model):
    __tablename__ = 'hotels'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, index=True)
    location = db.Column(db.String(200), nullable=False, index=True)
    description = db.Column(db.Text)
    amenities = db.Column(db.Text)  # JSON string for amenities list
    price_range = db.Column(db.String(20), index=True)  # low, medium, high, luxury
    rating = db.Column(db.Float, default=0.0, index=True)
    total_reviews = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True, index=True)
    
    # External API fields
    external_id = db.Column(db.String(100), unique=True, index=True)
    api_source = db.Column(db.String(50))  # Source of the hotel data
    
    # Additional fields for better recommendations
    category = db.Column(db.String(50))  # business, leisure, luxury, budget
    star_rating = db.Column(db.Integer)  # 1-5 stars
    contact_info = db.Column(db.Text)  # JSON for phone, email, website
    
    # Relationships
    reviews = db.relationship('Review', backref='hotel', lazy='dynamic', cascade='all, delete-orphan')
    
    def __init__(self, **kwargs):
        super(Hotel, self).__init__(**kwargs)
        if self.amenities is None:
            self.amenities = json.dumps([])
        if self.contact_info is None:
            self.contact_info = json.dumps({})
    
    def get_amenities(self):
        """Get amenities as list"""
        try:
            return json.loads(self.amenities) if self.amenities else []
        except json.JSONDecodeError:
            return []
    
    def set_amenities(self, amenities_list):
        """Set amenities from list"""
        self.amenities = json.dumps(amenities_list)
    
    def get_contact_info(self):
        """Get contact info as dictionary"""
        try:
            return json.loads(self.contact_info) if self.contact_info else {}
        except json.JSONDecodeError:
            return {}
    
    def set_contact_info(self, contact_dict):
        """Set contact info from dictionary"""
        self.contact_info = json.dumps(contact_dict)
    
    def update_rating(self):
        """Update hotel rating based on reviews"""
        reviews = self.reviews.filter_by().all()
        if reviews:
            self.rating = sum(review.rating for review in reviews) / len(reviews)
            self.total_reviews = len(reviews)
        else:
            self.rating = 0.0
            self.total_reviews = 0
        
        self.updated_at = datetime.utcnow()
        db.session.commit()
    
    def get_sentiment_summary(self):
        """Get sentiment analysis summary for hotel"""
        from models.review import Review
        
        reviews = self.reviews.filter(Review.sentiment_score.isnot(None)).all()
        if not reviews:
            return {
                'average_sentiment': 0.0,
                'positive_count': 0,
                'neutral_count': 0,
                'negative_count': 0,
                'total_reviews': 0
            }
        
        sentiments = [review.sentiment_score for review in reviews]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        return {
            'average_sentiment': round(avg_sentiment, 3),
            'positive_count': positive_count,
            'neutral_count': neutral_count,
            'negative_count': negative_count,
            'total_reviews': len(reviews)
        }
    
    def get_recent_reviews(self, limit=5):
        """Get recent reviews for hotel"""
        return self.reviews.order_by(db.desc('created_at')).limit(limit).all()
    
    def get_price_category(self):
        """Get price category with more detail"""
        price_ranges = {
            'budget': {'min': 0, 'max': 100, 'description': 'Budget-friendly'},
            'low': {'min': 50, 'max': 150, 'description': 'Affordable'},
            'medium': {'min': 100, 'max': 300, 'description': 'Mid-range'},
            'high': {'min': 250, 'max': 500, 'description': 'Premium'},
            'luxury': {'min': 400, 'max': 1000, 'description': 'Luxury'}
        }
        return price_ranges.get(self.price_range, {'description': 'Not specified'})
    
    def to_dict(self, include_reviews=False):
        """Convert hotel to dictionary"""
        data = {
            'id': self.id,
            'name': self.name,
            'location': self.location,
            'description': self.description,
            'amenities': self.get_amenities(),
            'price_range': self.price_range,
            'price_info': self.get_price_category(),
            'rating': round(self.rating, 2),
            'total_reviews': self.total_reviews,
            'star_rating': self.star_rating,
            'category': self.category,
            'contact_info': self.get_contact_info(),
            'sentiment_summary': self.get_sentiment_summary(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_active': self.is_active,
            'external_id': self.external_id,
            'api_source': self.api_source
        }
        
        if include_reviews:
            data['recent_reviews'] = [review.to_dict() for review in self.get_recent_reviews()]
        
        return data
    
    @staticmethod
    def search_hotels(query=None, location=None, price_range=None, min_rating=None, amenities=None):
        """Search hotels with various filters"""
        hotels_query = Hotel.query.filter_by(is_active=True)
        
        if query:
            hotels_query = hotels_query.filter(
                db.or_(
                    Hotel.name.ilike(f'%{query}%'),
                    Hotel.description.ilike(f'%{query}%')
                )
            )
        
        if location:
            hotels_query = hotels_query.filter(Hotel.location.ilike(f'%{location}%'))
        
        if price_range:
            hotels_query = hotels_query.filter(Hotel.price_range == price_range)
        
        if min_rating:
            hotels_query = hotels_query.filter(Hotel.rating >= min_rating)
        
        if amenities:
            for amenity in amenities:
                hotels_query = hotels_query.filter(Hotel.amenities.like(f'%{amenity}%'))
        
        return hotels_query.order_by(db.desc(Hotel.rating)).all()
    
    @staticmethod
    def get_top_rated_hotels(limit=10):
        """Get top rated hotels"""
        return Hotel.query.filter_by(is_active=True).filter(Hotel.total_reviews >= 3).order_by(db.desc(Hotel.rating)).limit(limit).all()
    
    @staticmethod
    def get_hotels_by_location(location, limit=20):
        """Get hotels by location"""
        return Hotel.query.filter_by(is_active=True).filter(Hotel.location.ilike(f'%{location}%')).order_by(db.desc(Hotel.rating)).limit(limit).all()
    
    def __repr__(self):
        return f'<Hotel {self.name}>'