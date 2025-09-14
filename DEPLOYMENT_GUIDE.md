# Hotel Recommendation System - Deployment & Testing Guide

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Navigate to project directory
cd "c:\Users\farha\Downloads\cognizant hackathon\cognizant hackathon"

# Install dependencies
pip install -r requirements.txt

# Initialize database
python init_database.py

# Start the application
python run.py
```

### 2. Access the Application
- Open your browser and go to: `http://localhost:5000`
- The application will start on port 5000 by default

## 🔧 Configuration

### Environment Variables (.env)
The system is pre-configured with:
- ✅ Flask secret keys
- ✅ SQLite database
- ✅ Google Maps API key (configured)
- ⚠️ Foursquare API (optional - needs your key)

### Google Maps Integration
The system includes a Google Maps API key. For production use:
1. Get your own API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Enable these APIs:
   - Maps JavaScript API
   - Places API
   - Geocoding API
3. Replace the key in `.env` file

## 🧪 Testing the System

### Manual Testing Checklist

#### 1. User Registration & Onboarding
- [ ] Go to `/login` and click "Register"
- [ ] Create a new account
- [ ] Complete the onboarding flow (5 steps)
- [ ] Verify preferences are saved

#### 2. Cold Start Recommendations
- [ ] After onboarding, check dashboard recommendations
- [ ] Verify recommendations match your preferences
- [ ] Test different preference combinations

#### 3. Hotel Search & Maps
- [ ] Use the search bar on dashboard
- [ ] Test location-based search
- [ ] Verify Google Maps integration works
- [ ] Check nearby attractions loading

#### 4. Hotel Details Page
- [ ] Click on any hotel from search results
- [ ] Verify map shows hotel location
- [ ] Check nearby attractions display
- [ ] Test review submission (if logged in)

#### 5. Recommendation Engine
- [ ] Submit reviews for hotels
- [ ] Check if recommendations improve
- [ ] Test collaborative filtering

### Automated Testing
Run the comprehensive test suite:
```bash
python test_system.py
```

## 📊 System Features

### ✅ Implemented Features
1. **User Authentication & JWT**
   - Registration, login, logout
   - Password hashing with Werkzeug
   - JWT token management

2. **Cold Start Problem Solution**
   - 5-step onboarding process
   - Preference collection and storage
   - Preference-based recommendations

3. **Recommendation Engine**
   - Collaborative filtering using SVD and KNN
   - Content-based filtering with TF-IDF
   - Hybrid recommendations
   - Sentiment analysis with TextBlob and VADER

4. **Google Maps Integration**
   - Hotel location visualization
   - Nearby attractions via Places API
   - Interactive maps with markers
   - Location-based hotel search

5. **Modern UI/UX**
   - Responsive Bootstrap design
   - Interactive onboarding flow
   - Real-time search and filtering
   - Mobile-friendly interface

6. **Database & Models**
   - SQLite database with SQLAlchemy ORM
   - User, Hotel, Review models
   - Preference storage as JSON
   - Sample data initialization

### 🔄 API Endpoints

#### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/profile` - Get user profile
- `PUT /api/auth/profile` - Update user profile

#### Recommendations
- `GET /api/recommendations/collaborative` - Collaborative filtering
- `GET /api/recommendations/content-based` - Content-based recommendations
- `GET /api/recommendations/hybrid` - Hybrid recommendations

#### Hotels
- `GET /api/hotels` - List hotels with filters
- `GET /api/hotels/<id>` - Get hotel details
- `POST /api/hotels/<id>/reviews` - Submit hotel review

#### Google Integration
- `GET /api/google/search-hotels` - Search hotels by location
- `GET /api/google/hotel-details/<place_id>` - Get Google Places details
- `GET /api/google/nearby-attractions` - Get nearby points of interest

## 🛠️ Troubleshooting

### Common Issues

1. **Database Not Found**
   ```bash
   # Run database initialization
   python init_database.py
   ```

2. **Google Maps Not Loading**
   - Check if API key is valid in `.env`
   - Ensure APIs are enabled in Google Cloud Console
   - Check browser console for errors

3. **Recommendations Not Working**
   - Ensure sample data is loaded
   - Check if users have reviews/preferences
   - Verify collaborative filtering has enough data

4. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

### Development Mode
For development with auto-reload:
```bash
# Set environment variable
set FLASK_ENV=development
python run.py
```

## 📈 Performance Optimization

### Database Optimization
- Indexes on frequently queried fields
- Efficient query patterns in SQLAlchemy
- Connection pooling for production

### Recommendation Engine
- Caching of similarity matrices
- Batch processing for large datasets
- Asynchronous recommendation updates

### Frontend Optimization
- Lazy loading of maps and images
- Debounced search inputs
- Progressive enhancement

## 🚀 Production Deployment

### Environment Setup
1. Set `FLASK_ENV=production`
2. Use PostgreSQL instead of SQLite
3. Configure proper secret keys
4. Set up SSL/HTTPS
5. Use a production WSGI server (Gunicorn)

### Security Considerations
- JWT token expiration and rotation
- Input validation and sanitization
- Rate limiting on API endpoints
- CORS configuration for production domains

## 📝 Sample Data

The system includes comprehensive sample data:
- **5 Hotels** with varied amenities and locations
- **3 Users** with different preferences
- **15 Reviews** with sentiment analysis
- **Cold start user** for testing onboarding

### Test Users
1. **john_doe** (password: password123)
   - Business traveler preferences
   - Has review history

2. **jane_smith** (password: password123)
   - Leisure traveler preferences
   - Has review history

3. **cold_start_user** (password: password123)
   - New user with no reviews
   - Tests cold start recommendations

## 🎯 Success Metrics

### Key Performance Indicators
1. **User Engagement**
   - Onboarding completion rate
   - Time spent on recommendations
   - Hotel detail page views

2. **Recommendation Quality**
   - Click-through rate on recommendations
   - User feedback on suggestions
   - Booking conversion (if integrated)

3. **System Performance**
   - API response times
   - Database query performance
   - Map loading times

## 🔮 Future Enhancements

### Planned Features
1. **Advanced ML Models**
   - Deep learning recommendations
   - Real-time personalization
   - A/B testing framework

2. **Enhanced Integrations**
   - Booking system integration
   - Social media sharing
   - Email notifications

3. **Analytics Dashboard**
   - User behavior tracking
   - Recommendation performance metrics
   - Business intelligence reports

## 📞 Support

For issues or questions:
1. Check this deployment guide
2. Review the test results from `test_system.py`
3. Check application logs for error details
4. Verify environment configuration

---

**Happy Testing! 🎉**

The Hotel Recommendation System is ready for comprehensive testing and deployment.
