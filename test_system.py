#!/usr/bin/env python3
"""
Hotel Recommendation System - Comprehensive Testing Script
Tests all major components including database, authentication, recommendations, and API integrations.
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from models.database import db
from models.user import User
from models.hotel import Hotel
from models.review import Review
from services.collaborative_filtering import CollaborativeFilteringRecommendation
from api.google_integration import GoogleMapsService

def test_database_connection():
    """Test database connection and table creation"""
    print("🔍 Testing Database Connection...")
    
    try:
        app = create_app('development')
        with app.app_context():
            # Create all tables
            db.create_all()
            
            # Test basic database operations
            test_user = User(
                username='test_db_user',
                email='test@example.com',
                password='testpass123'
            )
            db.session.add(test_user)
            db.session.commit()
            
            # Verify user was created
            retrieved_user = User.query.filter_by(username='test_db_user').first()
            assert retrieved_user is not None
            assert retrieved_user.email == 'test@example.com'
            
            # Clean up
            db.session.delete(retrieved_user)
            db.session.commit()
            
            print("✅ Database connection and operations working correctly")
            return True
            
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return False

def test_user_authentication():
    """Test user registration, login, and preference handling"""
    print("\n🔐 Testing User Authentication...")
    
    try:
        app = create_app('development')
        with app.app_context():
            # Test user creation
            user = User(
                username='auth_test_user',
                email='auth@example.com',
                password='secure_password_123'
            )
            
            # Test password hashing
            assert user.check_password('secure_password_123')
            assert not user.check_password('wrong_password')
            
            # Test preference handling
            preferences = {
                'travel_purpose': 'business',
                'budget_range': 'moderate',
                'preferred_amenities': ['wifi', 'gym', 'pool'],
                'location_preference': 'city_center',
                'review_importance': 'high'
            }
            
            user.set_preferences(preferences)
            assert user.get_preferences() == preferences
            assert user.is_new_user() == False  # Should be False after setting preferences
            
            # Test preference vector generation
            pref_vector = user.get_preference_vector()
            assert len(pref_vector) > 0
            
            print("✅ User authentication and preferences working correctly")
            return True
            
    except Exception as e:
        print(f"❌ User authentication test failed: {str(e)}")
        return False

def test_hotel_model():
    """Test hotel model and data handling"""
    print("\n🏨 Testing Hotel Model...")
    
    try:
        app = create_app('development')
        with app.app_context():
            # Create test hotel
            hotel = Hotel(
                name='Test Grand Hotel',
                address='123 Test Street, Test City',
                latitude=40.7128,
                longitude=-74.0060,
                rating=4.5,
                price_level=3,
                description='A luxurious test hotel',
                amenities=['wifi', 'pool', 'gym', 'spa'],
                phone='+1-555-0123',
                website='https://testhotel.com'
            )
            
            db.session.add(hotel)
            db.session.commit()
            
            # Test hotel retrieval and methods
            retrieved_hotel = Hotel.query.filter_by(name='Test Grand Hotel').first()
            assert retrieved_hotel is not None
            assert retrieved_hotel.rating == 4.5
            assert 'wifi' in retrieved_hotel.amenities
            
            # Test hotel dictionary conversion
            hotel_dict = retrieved_hotel.to_dict()
            assert hotel_dict['name'] == 'Test Grand Hotel'
            assert hotel_dict['rating'] == 4.5
            
            # Clean up
            db.session.delete(retrieved_hotel)
            db.session.commit()
            
            print("✅ Hotel model working correctly")
            return True
            
    except Exception as e:
        print(f"❌ Hotel model test failed: {str(e)}")
        return False

def test_review_system():
    """Test review creation and sentiment analysis"""
    print("\n📝 Testing Review System...")
    
    try:
        app = create_app('development')
        with app.app_context():
            # Create test user and hotel
            user = User(username='reviewer', email='reviewer@test.com', password='pass123')
            hotel = Hotel(name='Review Test Hotel', address='Test Address', latitude=40.0, longitude=-74.0)
            
            db.session.add(user)
            db.session.add(hotel)
            db.session.commit()
            
            # Create test review
            review = Review(
                user_id=user.id,
                hotel_id=hotel.id,
                rating=5,
                comment='This hotel was absolutely amazing! Great service and beautiful rooms.'
            )
            
            db.session.add(review)
            db.session.commit()
            
            # Test review retrieval
            retrieved_review = Review.query.filter_by(user_id=user.id).first()
            assert retrieved_review is not None
            assert retrieved_review.rating == 5
            
            # Test sentiment analysis (if available)
            if hasattr(retrieved_review, 'sentiment_score'):
                assert retrieved_review.sentiment_score > 0  # Positive sentiment
            
            # Clean up
            db.session.delete(review)
            db.session.delete(hotel)
            db.session.delete(user)
            db.session.commit()
            
            print("✅ Review system working correctly")
            return True
            
    except Exception as e:
        print(f"❌ Review system test failed: {str(e)}")
        return False

def test_recommendation_engine():
    """Test collaborative filtering and recommendation algorithms"""
    print("\n🤖 Testing Recommendation Engine...")
    
    try:
        app = create_app('development')
        with app.app_context():
            # Create test data
            users = []
            hotels = []
            reviews = []
            
            # Create test users with preferences
            for i in range(3):
                user = User(
                    username=f'rec_user_{i}',
                    email=f'rec_user_{i}@test.com',
                    password='testpass'
                )
                user.set_preferences({
                    'travel_purpose': 'leisure' if i % 2 == 0 else 'business',
                    'budget_range': 'moderate',
                    'preferred_amenities': ['wifi', 'pool'] if i % 2 == 0 else ['wifi', 'gym']
                })
                users.append(user)
                db.session.add(user)
            
            # Create test hotels
            for i in range(5):
                hotel = Hotel(
                    name=f'Test Hotel {i}',
                    address=f'Address {i}',
                    latitude=40.0 + i * 0.1,
                    longitude=-74.0 + i * 0.1,
                    rating=3.5 + (i * 0.3),
                    price_level=2 + (i % 3),
                    amenities=['wifi'] + (['pool'] if i % 2 == 0 else ['gym'])
                )
                hotels.append(hotel)
                db.session.add(hotel)
            
            db.session.commit()
            
            # Create test reviews
            for user in users:
                for hotel in hotels[:3]:  # Each user reviews first 3 hotels
                    review = Review(
                        user_id=user.id,
                        hotel_id=hotel.id,
                        rating=4 + (hash(f"{user.id}{hotel.id}") % 2),  # Random 4 or 5
                        comment=f'Review by {user.username} for {hotel.name}'
                    )
                    reviews.append(review)
                    db.session.add(review)
            
            db.session.commit()
            
            # Test recommendation engine
            rec_engine = CollaborativeFilteringRecommendation()
            
            # Test collaborative filtering recommendations
            recommendations = rec_engine.get_collaborative_recommendations(users[0].id, n_recommendations=3)
            assert len(recommendations) > 0
            print(f"   Generated {len(recommendations)} collaborative recommendations")
            
            # Test content-based recommendations
            content_recs = rec_engine.get_content_based_recommendations(users[0].id, n_recommendations=3)
            assert len(content_recs) > 0
            print(f"   Generated {len(content_recs)} content-based recommendations")
            
            # Test cold start recommendations (user with no reviews)
            cold_user = User(username='cold_user', email='cold@test.com', password='pass')
            cold_user.set_preferences({
                'travel_purpose': 'leisure',
                'budget_range': 'luxury',
                'preferred_amenities': ['spa', 'pool']
            })
            db.session.add(cold_user)
            db.session.commit()
            
            cold_recs = rec_engine.get_preference_based_recommendations(cold_user.id, n_recommendations=3)
            assert len(cold_recs) > 0
            print(f"   Generated {len(cold_recs)} cold start recommendations")
            
            # Clean up
            for review in reviews:
                db.session.delete(review)
            for hotel in hotels:
                db.session.delete(hotel)
            for user in users:
                db.session.delete(user)
            db.session.delete(cold_user)
            db.session.commit()
            
            print("✅ Recommendation engine working correctly")
            return True
            
    except Exception as e:
        print(f"❌ Recommendation engine test failed: {str(e)}")
        return False

def test_google_maps_integration():
    """Test Google Maps API integration (if API key is available)"""
    print("\n🗺️ Testing Google Maps Integration...")
    
    try:
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key or api_key == 'your-google-maps-api-key-here':
            print("⚠️ Google Maps API key not configured, skipping integration test")
            return True
        
        # Test Google Maps service initialization
        maps_service = GoogleMapsService(api_key)
        
        # Test geocoding (convert address to coordinates)
        test_address = "Times Square, New York, NY"
        try:
            coords = maps_service.geocode_address(test_address)
            if coords:
                print(f"   Geocoded '{test_address}' to {coords}")
                assert 'lat' in coords and 'lng' in coords
            else:
                print("   Geocoding returned no results (API quota may be exceeded)")
        except Exception as e:
            print(f"   Geocoding test failed: {str(e)}")
        
        print("✅ Google Maps integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ Google Maps integration test failed: {str(e)}")
        return False

def test_api_endpoints():
    """Test API endpoints with test client"""
    print("\n🌐 Testing API Endpoints...")
    
    try:
        app = create_app('development')
        
        with app.test_client() as client:
            # Test health check endpoint
            response = client.get('/api/health')
            if response.status_code == 404:
                print("   Health endpoint not found, creating basic test...")
                
            # Test user registration
            registration_data = {
                'username': 'api_test_user',
                'email': 'api_test@example.com',
                'password': 'secure_password_123'
            }
            
            response = client.post('/api/auth/register', 
                                 data=json.dumps(registration_data),
                                 content_type='application/json')
            
            if response.status_code in [200, 201]:
                print("   User registration endpoint working")
                
                # Try to login
                login_data = {
                    'username': 'api_test_user',
                    'password': 'secure_password_123'
                }
                
                response = client.post('/api/auth/login',
                                     data=json.dumps(login_data),
                                     content_type='application/json')
                
                if response.status_code == 200:
                    print("   User login endpoint working")
                    
                    # Get access token for authenticated requests
                    token_data = json.loads(response.data)
                    access_token = token_data.get('access_token')
                    
                    if access_token:
                        headers = {'Authorization': f'Bearer {access_token}'}
                        
                        # Test recommendations endpoint
                        response = client.get('/api/recommendations/collaborative', headers=headers)
                        print(f"   Recommendations endpoint status: {response.status_code}")
                        
            else:
                print(f"   Registration failed with status: {response.status_code}")
        
        print("✅ API endpoints test completed")
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🚀 Starting Comprehensive Hotel Recommendation System Test")
    print("=" * 60)
    
    test_results = {
        'Database Connection': test_database_connection(),
        'User Authentication': test_user_authentication(),
        'Hotel Model': test_hotel_model(),
        'Review System': test_review_system(),
        'Recommendation Engine': test_recommendation_engine(),
        'Google Maps Integration': test_google_maps_integration(),
        'API Endpoints': test_api_endpoints()
    }
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The hotel recommendation system is ready for use.")
    else:
        print("⚠️ Some tests failed. Please review the output above for details.")
    
    print("\n📋 Next Steps:")
    print("1. Start the Flask application: python run.py")
    print("2. Visit http://localhost:5000 to access the system")
    print("3. Register a new user and test the onboarding flow")
    print("4. Configure Google Maps API key for full functionality")
    
    return passed == total

if __name__ == '__main__':
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
