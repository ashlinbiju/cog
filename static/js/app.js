// Hotel Recommendation System - Frontend JavaScript (Fixed)

class HotelRecommendationApp {
    constructor() {
        this.apiBase = '/api';
        this.authToken = this.getFromStorage('authToken');
        this.currentUser = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAuthStatus();
        this.loadInitialData();
    }

    // Safe localStorage access
    getFromStorage(key) {
        try {
            return localStorage.getItem(key);
        } catch (e) {
            console.warn('LocalStorage not available:', e);
            return null;
        }
    }

    setToStorage(key, value) {
        try {
            localStorage.setItem(key, value);
        } catch (e) {
            console.warn('LocalStorage not available:', e);
        }
    }

    removeFromStorage(key) {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.warn('LocalStorage not available:', e);
        }
    }

    setupEventListeners() {
        // Login form
        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        }

        // Register form
        const registerForm = document.getElementById('registerForm');
        if (registerForm) {
            registerForm.addEventListener('submit', (e) => this.handleRegister(e));
        }

        // Show register form
        const showRegisterBtn = document.getElementById('showRegisterForm');
        if (showRegisterBtn) {
            showRegisterBtn.addEventListener('click', () => this.showRegisterForm());
        }

        // Show login form
        const showLoginBtn = document.getElementById('showLoginForm');
        if (showLoginBtn) {
            showLoginBtn.addEventListener('click', () => this.showLoginForm());
        }

        // Login link
        const loginLink = document.getElementById('loginLink');
        if (loginLink) {
            loginLink.addEventListener('click', (e) => {
                e.preventDefault();
                this.showModal('loginModal');
            });
        }

        // Logout
        const logoutLink = document.getElementById('logoutLink');
        if (logoutLink) {
            logoutLink.addEventListener('click', (e) => this.handleLogout(e));
        }

        // Hotel search
        const searchForm = document.getElementById('hotelSearchForm');
        if (searchForm) {
            searchForm.addEventListener('submit', (e) => this.handleHotelSearch(e));
        }

        // Recommendation method buttons
        const methodButtons = document.querySelectorAll('[data-method]');
        methodButtons.forEach(btn => {
            btn.addEventListener('click', (e) => this.changeRecommendationMethod(e));
        });

        // Review form
        const reviewForm = document.getElementById('reviewForm');
        if (reviewForm) {
            reviewForm.addEventListener('submit', (e) => this.handleReviewSubmit(e));
        }

        // Rating stars
        this.setupRatingStars();
    }

    setupRatingStars() {
        const stars = document.querySelectorAll('.rating-stars .star');
        stars.forEach((star, index) => {
            star.addEventListener('click', () => {
                const rating = index + 1;
                const ratingInput = document.getElementById('reviewRating');
                if (ratingInput) {
                    ratingInput.value = rating;
                }
                
                // Update visual feedback
                stars.forEach((s, i) => {
                    if (i < rating) {
                        s.classList.add('active');
                    } else {
                        s.classList.remove('active');
                    }
                });
            });

            star.addEventListener('mouseover', () => {
                const rating = index + 1;
                stars.forEach((s, i) => {
                    if (i < rating) {
                        s.style.opacity = '1';
                    } else {
                        s.style.opacity = '0.3';
                    }
                });
            });
        });

        // Reset on mouse leave
        const ratingContainer = document.querySelector('.rating-stars');
        if (ratingContainer) {
            ratingContainer.addEventListener('mouseleave', () => {
                const currentRating = document.getElementById('reviewRating')?.value || 0;
                stars.forEach((s, i) => {
                    if (i < currentRating) {
                        s.style.opacity = '1';
                        s.classList.add('active');
                    } else {
                        s.style.opacity = '0.3';
                        s.classList.remove('active');
                    }
                });
            });
        }
    }

    async checkAuthStatus() {
        if (this.authToken) {
            try {
                const response = await this.apiCall('/auth/verify-token', 'POST');
                if (response.valid) {
                    this.currentUser = { id: response.user_id, username: response.username };
                    this.updateUIForLoggedInUser();
                } else {
                    this.handleLogout();
                }
            } catch (error) {
                console.error('Auth check failed:', error);
                this.handleLogout();
            }
        }
    }

    updateUIForLoggedInUser() {
        const loginLink = document.getElementById('loginLink');
        const logoutLink = document.getElementById('logoutLink');
        
        if (loginLink) loginLink.classList.add('d-none');
        if (logoutLink) {
            logoutLink.classList.remove('d-none');
            logoutLink.textContent = `Logout (${this.currentUser.username})`;
        }

        // Show recommendations content if on recommendations page
        const recommendationsContent = document.getElementById('recommendationsContent');
        const loginRequired = document.getElementById('loginRequired');
        
        if (recommendationsContent && loginRequired) {
            loginRequired.classList.add('d-none');
            recommendationsContent.classList.remove('d-none');
            this.loadUserRecommendations();
        }
    }

    async handleLogin(e) {
        e.preventDefault();
        const username = document.getElementById('loginUsername')?.value;
        const password = document.getElementById('loginPassword')?.value;

        if (!username || !password) {
            this.showAlert('Please enter username and password', 'danger');
            return;
        }

        try {
            const response = await this.apiCall('/auth/login', 'POST', {
                username,
                password
            });

            this.authToken = response.access_token;
            this.setToStorage('authToken', this.authToken);
            this.currentUser = response.user;
            
            this.updateUIForLoggedInUser();
            this.closeModal('loginModal');
            this.showAlert('Login successful!', 'success');

            // Clear form
            document.getElementById('loginForm')?.reset();

        } catch (error) {
            this.showAlert(error.message || 'Login failed', 'danger');
        }
    }

    async handleRegister(e) {
        e.preventDefault();
        const username = document.getElementById('registerUsername')?.value;
        const email = document.getElementById('registerEmail')?.value;
        const password = document.getElementById('registerPassword')?.value;
        const age = document.getElementById('registerAge')?.value;
        const location = document.getElementById('registerLocation')?.value;

        if (!username || !email || !password) {
            this.showAlert('Please fill in required fields', 'danger');
            return;
        }

        try {
            const requestBody = {
                username,
                email,
                password
            };

            if (age) requestBody.age = parseInt(age);
            if (location) requestBody.location = location;

            const response = await this.apiCall('/auth/register', 'POST', requestBody);

            this.authToken = response.access_token;
            this.setToStorage('authToken', this.authToken);
            this.currentUser = response.user;
            
            this.updateUIForLoggedInUser();
            this.closeModal('loginModal');
            this.showAlert('Registration successful!', 'success');

            // Clear form
            document.getElementById('registerForm')?.reset();

        } catch (error) {
            this.showAlert(error.message || 'Registration failed', 'danger');
        }
    }

    handleLogout(e) {
        if (e) e.preventDefault();
        
        this.authToken = null;
        this.currentUser = null;
        this.removeFromStorage('authToken');

        const loginLink = document.getElementById('loginLink');
        const logoutLink = document.getElementById('logoutLink');
        
        if (loginLink) loginLink.classList.remove('d-none');
        if (logoutLink) logoutLink.classList.add('d-none');

        // Hide recommendations content if on recommendations page
        const recommendationsContent = document.getElementById('recommendationsContent');
        const loginRequired = document.getElementById('loginRequired');
        
        if (recommendationsContent && loginRequired) {
            loginRequired.classList.remove('d-none');
            recommendationsContent.classList.add('d-none');
        }

        this.showAlert('Logged out successfully', 'info');
    }

    async handleHotelSearch(e) {
        e.preventDefault();
        
        const location = document.getElementById('searchLocation')?.value || '';
        const priceRange = document.getElementById('searchPriceRange')?.value || '';
        const minRating = document.getElementById('searchMinRating')?.value || '';

        const params = new URLSearchParams();
        if (location) params.append('location', location);
        if (priceRange) params.append('price_range', priceRange);
        if (minRating) params.append('min_rating', minRating);

        try {
            const queryString = params.toString();
            const endpoint = queryString ? `/hotels?${queryString}` : '/hotels';
            const response = await this.apiCall(endpoint);
            this.displayHotels(response.hotels, 'Search Results');
        } catch (error) {
            this.showAlert('Search failed: ' + error.message, 'danger');
        }
    }

    async loadInitialData() {
        // Load featured hotels
        try {
            const response = await this.apiCall('/hotels?per_page=6');
            this.displayHotels(response.hotels, 'Featured Hotels');
        } catch (error) {
            console.error('Failed to load featured hotels:', error);
        }

        // Load system stats
        this.loadSystemStats();
        
        // Load trending hotels
        this.loadTrendingHotels();
    }

    async loadSystemStats() {
        try {
            const response = await this.apiCall('/stats/reviews');
            this.displaySystemStats(response);
        } catch (error) {
            console.error('Failed to load system stats:', error);
            const container = document.getElementById('systemStats');
            if (container) {
                container.innerHTML = '<p class="text-muted">Unable to load statistics</p>';
            }
        }
    }

    async loadTrendingHotels() {
        try {
            const response = await this.apiCall('/recommendations/trending?limit=5');
            this.displayTrendingHotels(response.trending_hotels);
        } catch (error) {
            console.error('Failed to load trending hotels:', error);
            const container = document.getElementById('trendingHotels');
            if (container) {
                container.innerHTML = '<p class="text-muted">Unable to load trending hotels</p>';
            }
        }
    }

    async loadUserRecommendations(method = 'hybrid') {
        if (!this.currentUser) return;

        try {
            const response = await this.apiCall(`/recommendations/${this.currentUser.id}?method=${method}`);
            this.displayRecommendations(response.recommendations, method);
            this.displayUserProfile(response.user_info);
            this.loadUserReviews();
        } catch (error) {
            this.showAlert('Failed to load recommendations: ' + error.message, 'danger');
        }
    }

    async loadUserReviews() {
        if (!this.currentUser) return;

        try {
            const response = await this.apiCall(`/users/${this.currentUser.id}/reviews?per_page=5`);
            this.displayUserReviews(response.reviews);
        } catch (error) {
            console.error('Failed to load user reviews:', error);
        }
    }

    async changeRecommendationMethod(e) {
        const method = e.target.dataset.method;
        
        // Update active button
        document.querySelectorAll('[data-method]').forEach(btn => {
            btn.classList.remove('active');
        });
        e.target.classList.add('active');

        // Load recommendations with new method
        await this.loadUserRecommendations(method);
    }

    displayHotels(hotels, title) {
        const container = document.getElementById('hotelsContainer');
        const titleElement = document.querySelector('#hotelsList h3');
        
        if (titleElement) titleElement.textContent = title;
        
        if (!container) return;

        if (!hotels || hotels.length === 0) {
            container.innerHTML = '<div class="col-12"><p class="text-muted">No hotels found.</p></div>';
            return;
        }

        container.innerHTML = hotels.map(hotel => `
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="card hotel-card fade-in">
                    <div class="card-body">
                        <h5 class="card-title">${this.escapeHtml(hotel.name)}</h5>
                        <p class="card-text text-muted small">${this.escapeHtml(hotel.location)}</p>
                        <div class="mb-2">
                            <span class="hotel-rating">${'⭐'.repeat(Math.floor(hotel.rating || 0))} ${(hotel.rating || 0).toFixed(1)}</span>
                            <span class="text-muted small">(${hotel.total_reviews || 0} reviews)</span>
                        </div>
                        <p class="card-text small">${this.escapeHtml((hotel.description || '').substring(0, 100))}${hotel.description && hotel.description.length > 100 ? '...' : ''}</p>
                        <div class="mb-2">
                            ${(hotel.amenities || []).slice(0, 3).map(amenity => 
                                `<span class="amenity-tag">${this.escapeHtml(amenity)}</span>`
                            ).join('')}
                            ${hotel.amenities && hotel.amenities.length > 3 ? `<span class="amenity-tag">+${hotel.amenities.length - 3} more</span>` : ''}
                        </div>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="hotel-price">${this.escapeHtml((hotel.price_info && hotel.price_info.description) || hotel.price_range || 'N/A')}</span>
                            <a href="/hotel/${hotel.id}" class="btn btn-primary btn-sm">View Details</a>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    displayRecommendations(recommendations, method) {
        const container = document.getElementById('recommendationsList');
        
        if (!container) return;

        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No recommendations available. Try reviewing some hotels first!</div>';
            return;
        }

        container.innerHTML = recommendations.map((rec, index) => `
            <div class="card recommendation-card mb-3 fade-in">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <h5 class="card-title mb-0">${index + 1}. ${this.escapeHtml(rec.hotel_name || rec.hotel.name)}</h5>
                        <div>
                            <span class="recommendation-score">${(rec.predicted_rating || 0).toFixed(1)}</span>
                            <span class="method-badge ms-2">${this.escapeHtml(rec.method || method)}</span>
                        </div>
                    </div>
                    <p class="text-muted small mb-2">${this.escapeHtml(rec.hotel_location || (rec.hotel && rec.hotel.location) || '')}</p>
                    <div class="mb-2">
                        <span class="star-rating">${'⭐'.repeat(Math.floor(rec.actual_rating || 0))} ${(rec.actual_rating || 0).toFixed(1)}</span>
                        <span class="text-muted small">(${rec.hotel && rec.hotel.total_reviews || 0} reviews)</span>
                    </div>
                    ${rec.hotel && rec.hotel.description ? `
                        <p class="card-text">${this.escapeHtml(rec.hotel.description.substring(0, 150))}...</p>
                    ` : ''}
                    
                    <a href="/hotel/${rec.hotel_id}" class="btn btn-primary">View Details</a>
                </div>
            </div>
        `).join('');
    }

    displaySystemStats(stats) {
        const container = document.getElementById('systemStats');
        
        if (!container) return;

        container.innerHTML = `
            <div class="stat-item">
                <div class="d-flex justify-content-between">
                    <span>Total Hotels:</span>
                    <span class="stat-value">${stats.total_hotels || 0}</span>
                </div>
            </div>
            <div class="stat-item">
                <div class="d-flex justify-content-between">
                    <span>Total Reviews:</span>
                    <span class="stat-value">${stats.sentiment_stats && stats.sentiment_stats.total || 0}</span>
                </div>
            </div>
            <div class="stat-item">
                <div class="d-flex justify-content-between">
                    <span>Positive Reviews:</span>
                    <span class="stat-value sentiment-positive">${stats.sentiment_stats && stats.sentiment_stats.positive_percentage || 0}%</span>
                </div>
            </div>
            <div class="stat-item">
                <div class="d-flex justify-content-between">
                    <span>Active Users:</span>
                    <span class="stat-value">${stats.total_users || 0}</span>
                </div>
            </div>
        `;
    }

    displayTrendingHotels(hotels) {
        const container = document.getElementById('trendingHotels');
        
        if (!container) return;

        if (!hotels || hotels.length === 0) {
            container.innerHTML = '<p class="text-muted">No trending hotels available.</p>';
            return;
        }

        container.innerHTML = hotels.map(hotel => `
            <div class="trending-hotel">
                <h6 class="mb-1">${this.escapeHtml(hotel.name)}</h6>
                <p class="text-muted small mb-1">${this.escapeHtml(hotel.location)}</p>
                <div class="d-flex justify-content-between align-items-center">
                    <span class="star-rating small">${'⭐'.repeat(Math.floor(hotel.rating || 0))} ${(hotel.rating || 0).toFixed(1)}</span>
                    <a href="/hotel/${hotel.id}" class="btn btn-outline-primary btn-sm">View</a>
                </div>
            </div>
        `).join('');
    }

    displayUserProfile(userInfo) {
        const container = document.getElementById('userProfile');
        
        if (!container || !userInfo) return;

        container.innerHTML = `
            <p><strong>Username:</strong> ${this.escapeHtml(userInfo.username)}</p>
            ${userInfo.location ? `<p><strong>Location:</strong> ${this.escapeHtml(userInfo.location)}</p>` : ''}
            <p><strong>Preferences:</strong></p>
            <ul class="small">
                ${Object.entries(userInfo.preferences || {}).map(([key, value]) => 
                    `<li>${this.escapeHtml(key)}: ${this.escapeHtml(String(value))}</li>`
                ).join('')}
            </ul>
        `;
    }

    displayUserReviews(reviews) {
        const container = document.getElementById('userReviews');
        
        if (!container) return;

        if (!reviews || reviews.length === 0) {
            container.innerHTML = '<p class="text-muted">No reviews yet.</p>';
            return;
        }

        container.innerHTML = reviews.map(review => `
            <div class="small mb-2 pb-2 border-bottom">
                <strong>${this.escapeHtml(review.hotel && review.hotel.name || 'Hotel')}</strong><br>
                <span class="star-rating">${'⭐'.repeat(review.rating || 0)}</span><br>
                <small class="text-muted">${new Date(review.created_at).toLocaleDateString()}</small>
            </div>
        `).join('');
    }

    async handleReviewSubmit(e) {
        e.preventDefault();
        
        if (!this.currentUser) {
            this.showModal('loginModal');
            return;
        }

        const rating = document.getElementById('reviewRating')?.value;
        const comment = document.getElementById('reviewComment')?.value;
        const hotelId = this.getCurrentHotelId();

        if (!rating || !comment) {
            this.showAlert('Please provide both rating and comment', 'danger');
            return;
        }

        try {
            await this.apiCall('/reviews', 'POST', {
                hotel_id: parseInt(hotelId),
                rating: parseInt(rating),
                comment
            });

            this.closeModal('reviewModal');
            this.showAlert('Review submitted successfully!', 'success');
            
            // Clear form
            document.getElementById('reviewForm')?.reset();
            document.querySelectorAll('.rating-stars .star').forEach(star => {
                star.classList.remove('active');
            });
            
            // Reload hotel details to show new review
            setTimeout(() => {
                location.reload();
            }, 1000);

        } catch (error) {
            this.showAlert('Failed to submit review: ' + error.message, 'danger');
        }
    }

    getCurrentHotelId() {
        const pathParts = window.location.pathname.split('/');
        return pathParts[pathParts.length - 1];
    }

    async loadHotelDetails(hotelId) {
        try {
            const hotelResponse = await this.apiCall(`/hotels/${hotelId}`);
            const reviewsResponse = await this.apiCall(`/hotels/${hotelId}/reviews`);
            
            let sentimentResponse = null;
            try {
                sentimentResponse = await this.apiCall(`/sentiment/${hotelId}`);
            } catch (e) {
                console.warn('Sentiment analysis not available:', e);
            }

            this.displayHotelDetails(hotelResponse.hotel, reviewsResponse.reviews, sentimentResponse);

        } catch (error) {
            const container = document.getElementById('hotelDetails');
            if (container) {
                container.innerHTML = `
                    <div class="alert alert-danger">Failed to load hotel details: ${error.message}</div>
                `;
            }
        }
    }

    displayHotelDetails(hotel, reviews, sentiment) {
        const container = document.getElementById('hotelDetails');
        
        if (!container) return;

        container.innerHTML = `
            <div class="row">
                <div class="col-md-8">
                    <div class="card mb-4">
                        <div class="card-body">
                            <h1>${this.escapeHtml(hotel.name)}</h1>
                            <p class="text-muted">${this.escapeHtml(hotel.location)}</p>
                            <div class="mb-3">
                                <span class="hotel-rating h4">${'⭐'.repeat(Math.floor(hotel.rating || 0))} ${(hotel.rating || 0).toFixed(1)}</span>
                                <span class="text-muted">(${hotel.total_reviews || 0} reviews)</span>
                            </div>
                            <p class="lead">${this.escapeHtml(hotel.description || '')}</p>
                            
                            <h5>Amenities</h5>
                            <div class="mb-3">
                                ${(hotel.amenities || []).map(amenity => `<span class="amenity-tag">${this.escapeHtml(amenity)}</span>`).join('')}
                            </div>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <p><strong>Price Range:</strong> ${this.escapeHtml((hotel.price_info && hotel.price_info.description) || hotel.price_range || 'N/A')}</p>
                                    <p><strong>Category:</strong> ${this.escapeHtml(hotel.category || 'N/A')}</p>
                                </div>
                                <div class="col-md-6">
                                    ${hotel.star_rating ? `<p><strong>Star Rating:</strong> ${'⭐'.repeat(hotel.star_rating)}</p>` : ''}
                                    <button class="btn btn-primary" onclick="app.showReviewModal()">Write a Review</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header">
                            <h5>Reviews</h5>
                        </div>
                        <div class="card-body">
                            ${reviews && reviews.length > 0 ? reviews.map(review => `
                                <div class="review-card card mb-3">
                                    <div class="card-body">
                                        <div class="d-flex justify-content-between align-items-start mb-2">
                                            <h6>${this.escapeHtml(review.user && review.user.username || 'Anonymous')}</h6>
                                            <div>
                                                <span class="star-rating">${'⭐'.repeat(review.rating || 0)}</span>
                                                ${review.sentiment_emoji ? `
                                                    <span class="ms-2">${review.sentiment_emoji}</span>
                                                ` : ''}
                                            </div>
                                        </div>
                                        <p class="mb-1">${this.escapeHtml(review.comment || '')}</p>
                                        <small class="text-muted">${new Date(review.created_at).toLocaleDateString()}</small>
                                    </div>
                                </div>
                            `).join('') : '<p class="text-muted">No reviews available.</p>'}
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    ${sentiment ? `
                        <div class="card mb-3">
                            <div class="card-header">
                                <h5>Sentiment Analysis</h5>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <div class="d-flex justify-content-between">
                                        <span>Positive:</span>
                                        <span class="sentiment-positive">${sentiment.sentiment_summary.positive_reviews} (${((sentiment.sentiment_summary.positive_reviews / sentiment.sentiment_summary.total_reviews) * 100).toFixed(1)}%)</span>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <span>Negative:</span>
                                        <span class="sentiment-negative">${sentiment.sentiment_summary.negative_reviews} (${((sentiment.sentiment_summary.negative_reviews / sentiment.sentiment_summary.total_reviews) * 100).toFixed(1)}%)</span>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <span>Neutral:</span>
                                        <span class="sentiment-neutral">${sentiment.sentiment_summary.neutral_reviews} (${((sentiment.sentiment_summary.neutral_reviews / sentiment.sentiment_summary.total_reviews) * 100).toFixed(1)}%)</span>
                                    </div>
                                </div>
                                <p><strong>Overall Sentiment:</strong> 
                                    <span class="sentiment-${sentiment.sentiment_summary.average_sentiment > 0.1 ? 'positive' : 
                                        sentiment.sentiment_summary.average_sentiment < -0.1 ? 'negative' : 'neutral'}">
                                        ${sentiment.sentiment_summary.average_sentiment > 0.1 ? 'Positive' : 
                                          sentiment.sentiment_summary.average_sentiment < -0.1 ? 'Negative' : 'Neutral'}
                                    </span>
                                </p>
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    showReviewModal() {
        if (!this.currentUser) {
            this.showModal('loginModal');
            return;
        }
        this.showModal('reviewModal');
    }

    // Utility methods
    async apiCall(endpoint, method = 'GET', body = null) {
        const config = {
            method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        if (this.authToken) {
            config.headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        if (body) {
            config.body = JSON.stringify(body);
        }

        const response = await fetch(this.apiBase + endpoint, config);
        
        if (!response.ok) {
            let errorMessage = 'Request failed';
            try {
                const errorData = await response.json();
                errorMessage = errorData.error || errorMessage;
            } catch (e) {
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }

        return await response.json();
    }

    showAlert(message, type = 'info') {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());

        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${this.escapeHtml(message)}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        const container = document.querySelector('.container');
        if (container) {
            container.insertAdjacentHTML('afterbegin', alertHtml);
            
            setTimeout(() => {
                const alert = container.querySelector('.alert');
                if (alert) {
                    alert.remove();
                }
            }, 5000);
        }
    }

    showModal(modalId) {
        const modalElement = document.getElementById(modalId);
        if (modalElement && typeof bootstrap !== 'undefined') {
            const modal = new bootstrap.Modal(modalElement);
            modal.show();
        }
    }

    closeModal(modalId) {
        const modalElement = document.getElementById(modalId);
        if (modalElement && typeof bootstrap !== 'undefined') {
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) {
                modal.hide();
            }
        }
    }

    showLoginForm() {
        const loginForm = document.getElementById('loginForm');
        const registerForm = document.getElementById('registerForm');
        const loginModalLabel = document.getElementById('loginModalLabel');
        
        if (loginForm) loginForm.classList.remove('d-none');
        if (registerForm) registerForm.classList.add('d-none');
        if (loginModalLabel) loginModalLabel.textContent = 'Login';
    }

    showRegisterForm() {
        const loginForm = document.getElementById('loginForm');
        const registerForm = document.getElementById('registerForm');
        const loginModalLabel = document.getElementById('loginModalLabel');
        
        if (loginForm) loginForm.classList.add('d-none');
        if (registerForm) registerForm.classList.remove('d-none');
        if (loginModalLabel) loginModalLabel.textContent = 'Register';
    }

    escapeHtml(text) {
        if (!text) return '';
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.toString().replace(/[&<>"']/g, (m) => map[m]);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.app = new HotelRecommendationApp();
        
        // Load hotel details if on hotel details page
        const path = window.location.pathname;
        if (path.startsWith('/hotel/')) {
            const hotelId = path.split('/')[2];
            if (hotelId && !isNaN(hotelId)) {
                app.loadHotelDetails(hotelId);
            }
        }
    } catch (error) {
        console.error('Failed to initialize app:', error);
    }
});