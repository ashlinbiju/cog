from app import db
import json
import os
from datetime import datetime

def init_sample_data():
    """Initialize database with sample data"""
    from models.user import User
    from models.hotel import Hotel
    from models.review import Review
    
    # Check if data already exists
    if User.query.first() is not None:
        print("Sample data already exists. Skipping initialization.")
        return
    
    # Load sample data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Load and create users
    try:
        with open(os.path.join(data_dir, 'sample_users.json'), 'r') as f:
            users_data = json.load(f)
            
        for user_data in users_data:
            user = User(
                username=user_data['username'],
                email=user_data['email'],
                age=user_data.get('age'),
                location=user_data.get('location'),
                preferences=json.dumps(user_data.get('preferences', {}))
            )
            user.set_password(user_data.get('password', 'password123'))
            db.session.add(user)
        
        db.session.commit()
        print(f"Created {len(users_data)} users")
    except FileNotFoundError:
        print("Sample users file not found, creating default users...")
        create_default_users()
    
    # Load and create hotels
    try:
        with open(os.path.join(data_dir, 'sample_hotels.json'), 'r') as f:
            hotels_data = json.load(f)
            
        for hotel_data in hotels_data:
            hotel = Hotel(
                name=hotel_data['name'],
                location=hotel_data['location'],
                description=hotel_data.get('description', ''),
                amenities=json.dumps(hotel_data.get('amenities', [])),
                price_range=hotel_data.get('price_range', 'medium'),
                rating=hotel_data.get('rating', 0.0),
                total_reviews=hotel_data.get('total_reviews', 0)
            )
            db.session.add(hotel)
        
        db.session.commit()
        print(f"Created {len(hotels_data)} hotels")
    except FileNotFoundError:
        print("Sample hotels file not found, creating default hotels...")
        create_default_hotels()
    
    # Load and create reviews
    try:
        with open(os.path.join(data_dir, 'sample_reviews.json'), 'r') as f:
            reviews_data = json.load(f)
            
        for review_data in reviews_data:
            review = Review(
                user_id=review_data['user_id'],
                hotel_id=review_data['hotel_id'],
                rating=review_data['rating'],
                comment=review_data.get('comment', ''),
                sentiment_score=review_data.get('sentiment_score', 0.0),
                created_at=datetime.now()
            )
            db.session.add(review)
        
        db.session.commit()
        print(f"Created {len(reviews_data)} reviews")
    except FileNotFoundError:
        print("Sample reviews file not found, creating default reviews...")
        create_default_reviews()

def create_default_users():
    """Create default users if sample file doesn't exist"""
    from models.user import User
    
    default_users = [
        {'username': 'john_doe', 'email': 'john@example.com', 'age': 28, 'location': 'New York'},
        {'username': 'jane_smith', 'email': 'jane@example.com', 'age': 35, 'location': 'California'},
        {'username': 'bob_wilson', 'email': 'bob@example.com', 'age': 42, 'location': 'Texas'},
        {'username': 'alice_brown', 'email': 'alice@example.com', 'age': 31, 'location': 'Florida'},
        {'username': 'charlie_davis', 'email': 'charlie@example.com', 'age': 26, 'location': 'Illinois'}
    ]
    
    for user_data in default_users:
        user = User(
            username=user_data['username'],
            email=user_data['email'],
            age=user_data['age'],
            location=user_data['location']
        )
        user.set_password('password123')
        db.session.add(user)
    
    db.session.commit()

def create_default_hotels():
    """Create default hotels if sample file doesn't exist"""
    from models.hotel import Hotel
    
    default_hotels = [
        {
            'name': 'Grand Plaza Hotel',
            'location': 'New York, NY',
            'description': 'Luxury hotel in the heart of Manhattan',
            'price_range': 'high',
            'rating': 4.5
        },
        {
            'name': 'Sunset Beach Resort',
            'location': 'Miami, FL',
            'description': 'Beautiful beachfront resort with ocean views',
            'price_range': 'high',
            'rating': 4.3
        },
        {
            'name': 'Downtown Business Hotel',
            'location': 'Chicago, IL',
            'description': 'Modern hotel perfect for business travelers',
            'price_range': 'medium',
            'rating': 4.1
        },
        {
            'name': 'Mountain View Lodge',
            'location': 'Denver, CO',
            'description': 'Cozy lodge with stunning mountain views',
            'price_range': 'medium',
            'rating': 4.2
        },
        {
            'name': 'Budget Inn Express',
            'location': 'Austin, TX',
            'description': 'Clean and affordable accommodation',
            'price_range': 'low',
            'rating': 3.8
        }
    ]
    
    for hotel_data in default_hotels:
        hotel = Hotel(
            name=hotel_data['name'],
            location=hotel_data['location'],
            description=hotel_data['description'],
            price_range=hotel_data['price_range'],
            rating=hotel_data['rating']
        )
        db.session.add(hotel)
    
    db.session.commit()

def create_default_reviews():
    """Create default reviews if sample file doesn't exist"""
    from models.review import Review
    
    default_reviews = [
        {'user_id': 1, 'hotel_id': 1, 'rating': 5, 'comment': 'Excellent service and beautiful rooms!'},
        {'user_id': 2, 'hotel_id': 1, 'rating': 4, 'comment': 'Great location, staff was helpful.'},
        {'user_id': 3, 'hotel_id': 2, 'rating': 5, 'comment': 'Amazing beach view and clean facilities.'},
        {'user_id': 1, 'hotel_id': 2, 'rating': 4, 'comment': 'Perfect for vacation, would recommend.'},
        {'user_id': 4, 'hotel_id': 3, 'rating': 4, 'comment': 'Good for business trips, convenient location.'},
        {'user_id': 5, 'hotel_id': 4, 'rating': 5, 'comment': 'Beautiful mountain views and cozy atmosphere.'},
        {'user_id': 2, 'hotel_id': 5, 'rating': 3, 'comment': 'Basic but clean, good value for money.'}
    ]
    
    for review_data in default_reviews:
        review = Review(
            user_id=review_data['user_id'],
            hotel_id=review_data['hotel_id'],
            rating=review_data['rating'],
            comment=review_data['comment'],
            created_at=datetime.now()
        )
        db.session.add(review)
    
    db.session.commit()