from flask import Flask, render_template, redirect, url_for, session, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from functools import wraps
from config import config
import os

# Initialize extensions
db = SQLAlchemy()
jwt = JWTManager()

def create_app(config_name=None):
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions with app
    db.init_app(app)
    jwt.init_app(app)
    CORS(app)
    
    # JWT error handlers
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return {'error': 'Token has expired'}, 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return {'error': 'Invalid token'}, 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return {'error': 'Authorization token required'}, 401
    
    # JWT token blacklist checker
    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        from api.auth import blacklisted_tokens
        jti = jwt_payload['jti']
        return jti in blacklisted_tokens
    
    # Import models to ensure they're registered
    from models.user import User
    from models.hotel import Hotel
    from models.review import Review
    
    # Register blueprints
    from api.auth import auth_bp
    from api.recommendations import recommendations_bp
    from api.google_integration import google_bp
    from api.smart_recommendations import smart_recommendations_bp
    from api.api_status import api_status_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(recommendations_bp, url_prefix='/api/recommendations')
    app.register_blueprint(google_bp, url_prefix='/api/google')
    app.register_blueprint(smart_recommendations_bp, url_prefix='/api/smart')
    app.register_blueprint(api_status_bp, url_prefix='/api/status')
    
    # Login required decorator
    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login_page', next=request.url))
            return f(*args, **kwargs)
        return decorated_function

    # Main routes
    @app.route('/')
    def index():
        if 'user_id' in session:
            return redirect(url_for('onboarding' if 'onboarding_complete' not in session else 'recommendations_page'))
        return redirect(url_for('login_page'))
    
    @app.route('/login', methods=['GET', 'POST'])
    def login_page():
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('login.html')
    
    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('login_page'))
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        return render_template('dashboard.html')
    
    @app.route('/onboarding')
    @login_required
    def onboarding():
        return render_template('onboarding.html')
    
    @app.route('/recommendations')
    @login_required
    def recommendations_page():
        if 'onboarding_complete' not in session:
            return redirect(url_for('onboarding'))
        return render_template('recommendations.html')
    
    @app.route('/hotel/<int:hotel_id>')
    @login_required
    def hotel_details(hotel_id):
        return render_template('hotel_details.html', hotel_id=hotel_id)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    
    return app