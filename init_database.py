#!/usr/bin/env python3
"""
Database initialization script for Hotel Recommendation System
Creates sample data for testing the cold start problem and recommendation algorithms
"""

import os
import sys
from datetime import datetime, timedelta
import random
import json

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from models.user import User
from models.hotel import Hotel
from models.review import Review

def create_sample_hotels():
    """Create sample hotels with diverse characteristics"""
    
    hotels_data = [
        {
            'name': 'Grand Plaza Hotel',
            'location': 'Downtown New York, NY',
            'description': 'Luxury business hotel in the heart of Manhattan',
            'rating': 4.5,
            'total_reviews': 1250,
            'price_range': 'Expensive',
            'price_level': 3,
            'hotel_type': 'business luxury',
            'amenities': ['wifi', 'gym', 'spa', 'restaurant', 'business-center', 'parking'],
            'latitude': 40.7589,
            'longitude': -73.9851
        },
        {
            'name': 'Oceanview Resort & Spa',
            'location': 'Miami Beach, FL',
            'description': 'Beachfront resort perfect for leisure and romantic getaways',
            'rating': 4.7,
            'total_reviews': 890,
            'price_range': 'Very Expensive',
            'price_level': 4,
            'hotel_type': 'resort leisure romantic',
            'amenities': ['wifi', 'pool', 'spa', 'restaurant', 'beach-access'],
            'latitude': 25.7907,
            'longitude': -80.1300
        },
        {
            'name': 'Family Fun Lodge',
            'location': 'Orlando, FL',
            'description': 'Family-friendly hotel near theme parks',
            'rating': 4.2,
            'total_reviews': 2100,
            'price_range': 'Moderate',
            'price_level': 2,
            'hotel_type': 'family vacation',
            'amenities': ['wifi', 'pool', 'restaurant', 'parking', 'kids-club'],
            'latitude': 28.3772,
            'longitude': -81.5707
        },
        {
            'name': 'Budget Inn Express',
            'location': 'Austin, TX',
            'description': 'Clean and affordable accommodation for budget travelers',
            'rating': 3.8,
            'total_reviews': 450,
            'price_range': 'Inexpensive',
            'price_level': 1,
            'hotel_type': 'budget business',
            'amenities': ['wifi', 'parking', 'breakfast'],
            'latitude': 30.2672,
            'longitude': -97.7431
        },
        {
            'name': 'Mountain View Retreat',
            'location': 'Aspen, CO',
            'description': 'Peaceful mountain resort for wellness and adventure',
            'rating': 4.6,
            'total_reviews': 320,
            'price_range': 'Very Expensive',
            'price_level': 4,
            'hotel_type': 'wellness adventure quiet',
            'amenities': ['wifi', 'spa', 'gym', 'restaurant', 'hiking-trails'],
            'latitude': 39.1911,
            'longitude': -106.8175
        },
        {
            'name': 'Airport Business Center',
            'location': 'Near LAX, Los Angeles, CA',
            'description': 'Convenient business hotel with airport shuttle',
            'rating': 4.0,
            'total_reviews': 680,
            'price_range': 'Moderate',
            'price_level': 2,
            'hotel_type': 'business airport',
            'amenities': ['wifi', 'gym', 'business-center', 'airport-shuttle', 'restaurant'],
            'latitude': 33.9425,
            'longitude': -118.4081
        },
        {
            'name': 'Historic Downtown Boutique',
            'location': 'Savannah, GA',
            'description': 'Charming boutique hotel in historic district',
            'rating': 4.4,
            'total_reviews': 275,
            'price_range': 'Expensive',
            'price_level': 3,
            'hotel_type': 'boutique romantic quiet',
            'amenities': ['wifi', 'restaurant', 'historic-charm', 'courtyard'],
            'latitude': 32.0835,
            'longitude': -81.0998
        },
        {
            'name': 'City Center Suites',
            'location': 'Downtown Chicago, IL',
            'description': 'Modern suites in the heart of the city',
            'rating': 4.3,
            'total_reviews': 950,
            'price_range': 'Expensive',
            'price_level': 3,
            'hotel_type': 'business city-center',
            'amenities': ['wifi', 'gym', 'restaurant', 'business-center', 'suites'],
            'latitude': 41.8781,
            'longitude': -87.6298
        },
        {
            'name': 'Beachside Budget Hotel',
            'location': 'Virginia Beach, VA',
            'description': 'Affordable beachfront accommodation',
            'rating': 3.9,
            'total_reviews': 520,
            'price_range': 'Inexpensive',
            'price_level': 1,
            'hotel_type': 'budget beach family',
            'amenities': ['wifi', 'pool', 'beach-access', 'parking'],
            'latitude': 36.8529,
            'longitude': -75.9780
        },
        {
            'name': 'Luxury Spa Resort',
            'location': 'Napa Valley, CA',
            'description': 'Ultra-luxury wellness resort with world-class spa',
            'rating': 4.9,
            'total_reviews': 180,
            'price_range': 'Very Expensive',
            'price_level': 4,
            'hotel_type': 'luxury wellness romantic quiet',
            'amenities': ['wifi', 'spa', 'restaurant', 'wine-tasting', 'wellness-center'],
            'latitude': 38.2975,
            'longitude': -122.2869
        }
    ]
    
    hotels = []
    for hotel_data in hotels_data:
        # Convert amenities list to JSON string
        amenities_json = json.dumps(hotel_data.pop('amenities'))
        
        hotel = Hotel(
            **hotel_data,
            amenities=amenities_json,
            is_active=True,
            created_at=datetime.utcnow() - timedelta(days=random.randint(30, 365))
        )
        
        db.session.add(hotel)
        hotels.append(hotel)
    
    db.session.commit()
    print(f"Created {len(hotels)} sample hotels")
    return hotels

def create_sample_users():
    """Create sample users with different preferences"""
    
    users_data = [
        {
            'username': 'business_traveler',
            'email': 'john.doe@business.com',
            'age': 35,
            'location': 'New York, NY',
            'preferences': {
                'travel_purpose': ['business'],
                'budget_range': 400,
                'budget_category': ['luxury'],
                'amenities': ['wifi', 'gym', 'business-center', 'restaurant'],
                'location_preferences': ['city-center', 'airport'],
                'min_rating': 4.0,
                'min_reviews': 100,
                'review_aspects': ['service', 'location', 'cleanliness'],
                'onboarding_completed': True
            }
        },
        {
            'username': 'family_vacationer',
            'email': 'sarah.smith@family.com',
            'age': 42,
            'location': 'Chicago, IL',
            'preferences': {
                'travel_purpose': ['family'],
                'budget_range': 250,
                'budget_category': ['mid-range'],
                'amenities': ['wifi', 'pool', 'restaurant', 'parking'],
                'location_preferences': ['tourist-area', 'quiet'],
                'min_rating': 4.2,
                'min_reviews': 200,
                'review_aspects': ['cleanliness', 'value', 'service'],
                'onboarding_completed': True
            }
        },
        {
            'username': 'budget_backpacker',
            'email': 'mike.jones@travel.com',
            'age': 24,
            'location': 'Austin, TX',
            'preferences': {
                'travel_purpose': ['leisure', 'adventure'],
                'budget_range': 100,
                'budget_category': ['budget'],
                'amenities': ['wifi', 'parking'],
                'location_preferences': ['transport', 'city-center'],
                'min_rating': 3.5,
                'min_reviews': 50,
                'review_aspects': ['value', 'location'],
                'onboarding_completed': True
            }
        },
        {
            'username': 'romantic_couple',
            'email': 'emma.wilson@romance.com',
            'age': 29,
            'location': 'San Francisco, CA',
            'preferences': {
                'travel_purpose': ['romantic'],
                'budget_range': 500,
                'budget_category': ['luxury'],
                'amenities': ['spa', 'restaurant', 'wifi'],
                'location_preferences': ['beach', 'quiet'],
                'min_rating': 4.5,
                'min_reviews': 100,
                'review_aspects': ['service', 'cleanliness'],
                'onboarding_completed': True
            }
        },
        {
            'username': 'wellness_seeker',
            'email': 'alex.green@wellness.com',
            'age': 38,
            'location': 'Denver, CO',
            'preferences': {
                'travel_purpose': ['wellness'],
                'budget_range': 350,
                'budget_category': ['mid-range', 'luxury'],
                'amenities': ['spa', 'gym', 'wifi', 'wellness-center'],
                'location_preferences': ['quiet', 'beach'],
                'min_rating': 4.3,
                'min_reviews': 75,
                'review_aspects': ['service', 'cleanliness'],
                'onboarding_completed': True
            }
        },
        {
            'username': 'new_user_coldstart',
            'email': 'newbie@test.com',
            'age': 30,
            'location': 'Seattle, WA',
            'preferences': {
                'onboarding_completed': False
            }
        }
    ]
    
    users = []
    for user_data in users_data:
        preferences = user_data.pop('preferences', {})
        
        user = User(
            **user_data,
            is_active=True,
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 180))
        )
        user.set_password('password123')  # Default password for testing
        user.set_preferences(preferences)
        
        db.session.add(user)
        users.append(user)
    
    db.session.commit()
    print(f"Created {len(users)} sample users")
    return users

def create_sample_reviews(users, hotels):
    """Create sample reviews to populate the recommendation system"""
    
    reviews = []
    
    # Business traveler reviews
    business_user = next(u for u in users if u.username == 'business_traveler')
    business_hotels = [h for h in hotels if 'business' in h.hotel_type]
    
    for hotel in business_hotels[:4]:
        rating = random.uniform(4.0, 5.0)
        review = Review(
            user_id=business_user.id,
            hotel_id=hotel.id,
            rating=rating,
            review_text=f"Great business hotel with excellent facilities. Perfect for work trips.",
            sentiment_score=0.8,
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 90))
        )
        reviews.append(review)
    
    # Family vacationer reviews
    family_user = next(u for u in users if u.username == 'family_vacationer')
    family_hotels = [h for h in hotels if 'family' in h.hotel_type or h.price_level <= 2]
    
    for hotel in family_hotels[:3]:
        rating = random.uniform(4.0, 4.8)
        review = Review(
            user_id=family_user.id,
            hotel_id=hotel.id,
            rating=rating,
            review_text=f"Perfect for families! Kids loved the amenities and location was great.",
            sentiment_score=0.9,
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 60))
        )
        reviews.append(review)
    
    # Budget traveler reviews
    budget_user = next(u for u in users if u.username == 'budget_backpacker')
    budget_hotels = [h for h in hotels if h.price_level <= 2]
    
    for hotel in budget_hotels[:3]:
        rating = random.uniform(3.5, 4.2)
        review = Review(
            user_id=budget_user.id,
            hotel_id=hotel.id,
            rating=rating,
            review_text=f"Good value for money. Basic but clean and comfortable.",
            sentiment_score=0.6,
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 45))
        )
        reviews.append(review)
    
    # Romantic couple reviews
    romantic_user = next(u for u in users if u.username == 'romantic_couple')
    romantic_hotels = [h for h in hotels if 'romantic' in h.hotel_type or 'spa' in h.amenities]
    
    for hotel in romantic_hotels[:3]:
        rating = random.uniform(4.3, 5.0)
        review = Review(
            user_id=romantic_user.id,
            hotel_id=hotel.id,
            rating=rating,
            review_text=f"Absolutely romantic! Perfect for couples with amazing spa services.",
            sentiment_score=0.95,
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 30))
        )
        reviews.append(review)
    
    # Wellness seeker reviews
    wellness_user = next(u for u in users if u.username == 'wellness_seeker')
    wellness_hotels = [h for h in hotels if 'wellness' in h.hotel_type or 'spa' in h.amenities]
    
    for hotel in wellness_hotels[:2]:
        rating = random.uniform(4.2, 4.9)
        review = Review(
            user_id=wellness_user.id,
            hotel_id=hotel.id,
            rating=rating,
            review_text=f"Excellent wellness facilities. Very relaxing and rejuvenating experience.",
            sentiment_score=0.85,
            created_at=datetime.utcnow() - timedelta(days=random.randint(1, 20))
        )
        reviews.append(review)
    
    # Add some cross-preference reviews for better collaborative filtering
    for _ in range(10):
        user = random.choice(users[:-1])  # Exclude the new user
        hotel = random.choice(hotels)
        
        # Check if user already reviewed this hotel
        existing = any(r for r in reviews if r.user_id == user.id and r.hotel_id == hotel.id)
        if not existing:
            rating = random.uniform(3.0, 5.0)
            sentiment = 0.3 + (rating - 3.0) * 0.35  # Correlate sentiment with rating
            
            review = Review(
                user_id=user.id,
                hotel_id=hotel.id,
                rating=rating,
                review_text=f"Had a good stay at this hotel. Would recommend to others.",
                sentiment_score=sentiment,
                created_at=datetime.utcnow() - timedelta(days=random.randint(1, 120))
            )
            reviews.append(review)
    
    # Add all reviews to database
    for review in reviews:
        db.session.add(review)
    
    db.session.commit()
    print(f"Created {len(reviews)} sample reviews")
    return reviews

def initialize_database():
    """Initialize database with sample data"""
    
    print("Initializing Hotel Recommendation System Database...")
    
    # Create all tables
    db.create_all()
    print("Database tables created")
    
    # Check if data already exists
    if Hotel.query.first():
        print("Database already contains data. Skipping initialization.")
        return
    
    # Create sample data
    hotels = create_sample_hotels()
    users = create_sample_users()
    reviews = create_sample_reviews(users, hotels)
    
    print("\n" + "="*50)
    print("DATABASE INITIALIZATION COMPLETE!")
    print("="*50)
    print(f"✅ Created {len(hotels)} hotels")
    print(f"✅ Created {len(users)} users")
    print(f"✅ Created {len(reviews)} reviews")
    print("\nTest Users (password: 'password123'):")
    print("-" * 40)
    for user in users:
        onboarding_status = "✅ Completed" if user.get_preferences().get('onboarding_completed', False) else "❌ Needs Onboarding"
        print(f"• {user.username} ({user.email}) - {onboarding_status}")
    
    print("\nYou can now:")
    print("1. Test the login system with any of the above users")
    print("2. Test cold start recommendations with 'new_user_coldstart'")
    print("3. Test the onboarding flow by registering a new user")
    print("4. Test collaborative filtering with existing users")
    
    return hotels, users, reviews

if __name__ == '__main__':
    app = create_app()
    
    with app.app_context():
        try:
            result = initialize_database()
            if result:
                hotels, users, reviews = result
                print(f"\n🚀 Database successfully initialized! Start the server with: python run.py")
            else:
                print("\nDatabase was already initialized. No new data was added.")
                print("🚀 You can start the server with: python run.py")

        except Exception as e:
            print(f"❌ An unexpected error occurred during database initialization: {str(e)}")
            import traceback
            traceback.print_exc()
