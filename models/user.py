from app import db
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    age = db.Column(db.Integer)
    location = db.Column(db.String(100))
    preferences = db.Column(db.Text)  # JSON string for user preferences
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    onboarding_complete = db.Column(db.Boolean, default=False)
    
    # Relationships
    reviews = db.relationship('Review', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        if self.preferences is None:
            self.preferences = json.dumps({})
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_preferences(self):
        """Get user preferences as dictionary"""
        try:
            return json.loads(self.preferences) if self.preferences else self.get_default_preferences()
        except json.JSONDecodeError:
            return self.get_default_preferences()
    
    def set_preferences(self, preferences_dict):
        """Set user preferences from dictionary"""
        self.preferences = json.dumps(preferences_dict)
    
    def get_default_preferences(self):
        """Get default preferences for cold start problem"""
        return {
            'travel_purpose': [],
            'budget_range': 300,
            'budget_category': [],
            'amenities': [],
            'location_preferences': [],
            'min_rating': 4.0,
            'min_reviews': 50,
            'review_aspects': [],
            'onboarding_completed': False
        }
    
    def is_new_user(self):
        """Check if user needs onboarding (cold start problem)"""
        prefs = self.get_preferences()
        return not prefs.get('onboarding_completed', False)
    
    def complete_onboarding(self):
        """Mark onboarding as completed"""
        prefs = self.get_preferences()
        prefs['onboarding_completed'] = True
        self.set_preferences(prefs)
    
    def get_preference_vector(self):
        """Get user preferences as a feature vector for ML models"""
        prefs = self.get_preferences()
        
        # Create feature vector for collaborative filtering
        vector = {
            'budget_range': prefs.get('budget_range', 300),
            'min_rating': prefs.get('min_rating', 4.0),
            'min_reviews': prefs.get('min_reviews', 50),
            'travel_purpose_business': 1 if 'business' in prefs.get('travel_purpose', []) else 0,
            'travel_purpose_leisure': 1 if 'leisure' in prefs.get('travel_purpose', []) else 0,
            'travel_purpose_family': 1 if 'family' in prefs.get('travel_purpose', []) else 0,
            'travel_purpose_romantic': 1 if 'romantic' in prefs.get('travel_purpose', []) else 0,
            'amenity_wifi': 1 if 'wifi' in prefs.get('amenities', []) else 0,
            'amenity_pool': 1 if 'pool' in prefs.get('amenities', []) else 0,
            'amenity_gym': 1 if 'gym' in prefs.get('amenities', []) else 0,
            'amenity_spa': 1 if 'spa' in prefs.get('amenities', []) else 0,
            'location_city_center': 1 if 'city-center' in prefs.get('location_preferences', []) else 0,
            'location_beach': 1 if 'beach' in prefs.get('location_preferences', []) else 0,
            'location_airport': 1 if 'airport' in prefs.get('location_preferences', []) else 0,
        }
        
        return vector
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def get_review_count(self):
        """Get total number of reviews by this user"""
        return self.reviews.count()
    
    def get_average_rating(self):
        """Get average rating given by this user"""
        reviews = self.reviews.all()
        if not reviews:
            return 0.0
        return sum(review.rating for review in reviews) / len(reviews)
    
    def has_reviewed_hotel(self, hotel_id):
        """Check if user has reviewed a specific hotel"""
        return self.reviews.filter_by(hotel_id=hotel_id).first() is not None
    
    def to_dict(self, include_sensitive=False):
        """Convert user to dictionary"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'age': self.age,
            'location': self.location,
            'preferences': self.get_preferences(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'onboarding_complete': self.onboarding_complete,
            'review_count': self.get_review_count(),
            'average_rating': round(self.get_average_rating(), 2)
        }
        
        if include_sensitive:
            data['password_hash'] = self.password_hash
        
        return data
    
    @staticmethod
    def create_user(username, email, password, **kwargs):
        """Create a new user"""
        user = User(username=username, email=email, **kwargs)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user
    
    @staticmethod
    def get_user_by_username(username):
        """Get user by username"""
        return User.query.filter_by(username=username, is_active=True).first()
    
    @staticmethod
    def get_user_by_email(email):
        """Get user by email"""
        return User.query.filter_by(email=email, is_active=True).first()
    
    def __repr__(self):
        return f'<User {self.username}>'