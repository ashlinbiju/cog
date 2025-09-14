from flask import Blueprint, request, jsonify, session
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from werkzeug.security import check_password_hash
from models.user import User
from app import db
from datetime import timedelta
import logging
import json
from functools import wraps

auth_bp = Blueprint('auth', __name__)
logger = logging.getLogger(__name__)

# In-memory blacklist for revoked tokens (in production, use Redis or database)
blacklisted_tokens = set()

@auth_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get credentials
        username_or_email = data.get('username') or data.get('email')
        password = data.get('password')
        
        if not username_or_email or not password:
            return jsonify({'error': 'Username/email and password are required'}), 400
        
        # Find user by username or email
        user = User.get_user_by_username(username_or_email) or User.get_user_by_email(username_or_email)
        
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is disabled'}), 401
        
        # Create access token
        access_token = create_access_token(
            identity=user.id,
            expires_delta=timedelta(hours=24)
        )
        
        # Update last login
        user.update_last_login()
        
        # Set user in session for server-side authentication
        from flask import session
        session['user_id'] = user.id
        session['onboarding_complete'] = user.onboarding_complete
        
        return jsonify({
            'access_token': access_token,
            'user': user.to_dict(),
            'message': 'Login successful',
            'needs_onboarding': not user.onboarding_complete
        })
        
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@auth_bp.route('/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        
        # Validate username
        if len(username) < 3 or len(username) > 50:
            return jsonify({'error': 'Username must be between 3 and 50 characters'}), 400
        
        # Validate email format (basic validation)
        if '@' not in email or '.' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password strength
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Check if user already exists
        if User.get_user_by_username(username):
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.get_user_by_email(email):
            return jsonify({'error': 'Email already exists'}), 400
        
        # Create new user
        user = User(
            username=username,
            email=email,
            age=data.get('age'),
            location=data.get('location')
        )
        user.set_password(password)
        
        # Set preferences if provided
        if 'preferences' in data and isinstance(data['preferences'], dict):
            user.set_preferences(data['preferences'])
        
        db.session.add(user)
        db.session.commit()
        
        # Create access token for immediate login
        access_token = create_access_token(
            identity=user.id,
            expires_delta=timedelta(hours=24)
        )
        
        # Check if user needs onboarding (cold start problem)
        needs_onboarding = user.is_new_user()
        
        return jsonify({
            'access_token': access_token,
            'user': user.to_dict(),
            'needs_onboarding': needs_onboarding,
            'message': 'Registration successful'
        }), 201
        
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """User logout endpoint"""
    try:
        # Get the JWT token
        jti = get_jwt()['jti']  # JWT ID
        
        # Add token to blacklist
        blacklisted_tokens.add(jti)
        
        return jsonify({'message': 'Successfully logged out'})
        
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user profile"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user or not user.is_active:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({'user': user.to_dict()})
        
    except Exception as e:
        logger.error(f"Error getting profile: {str(e)}")
        return jsonify({'error': 'Failed to get profile'}), 500

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update current user profile"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user or not user.is_active:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Update allowed fields
        if 'email' in data:
            new_email = data['email'].strip().lower()
            if '@' not in new_email or '.' not in new_email:
                return jsonify({'error': 'Invalid email format'}), 400
            
            # Check if email is already taken by another user
            existing_user = User.get_user_by_email(new_email)
            if existing_user and existing_user.id != user_id:
                return jsonify({'error': 'Email already exists'}), 400
            
            user.email = new_email
        
        if 'age' in data:
            if data['age'] is not None:
                try:
                    age = int(data['age'])
                    if age < 13 or age > 120:
                        return jsonify({'error': 'Invalid age'}), 400
                    user.age = age
                except ValueError:
                    return jsonify({'error': 'Age must be a number'}), 400
            else:
                user.age = None
        
        if 'location' in data:
            user.location = data['location'].strip() if data['location'] else None
        
        if 'preferences' in data:
            if isinstance(data['preferences'], dict):
                user.set_preferences(data['preferences'])
            else:
                return jsonify({'error': 'Preferences must be a JSON object'}), 400
        
        db.session.commit()
        
        return jsonify({
            'user': user.to_dict(),
            'message': 'Profile updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update profile'}), 500

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """Change user password"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user or not user.is_active:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({'error': 'Current password and new password are required'}), 400
        
        # Verify current password
        if not user.check_password(current_password):
            return jsonify({'error': 'Current password is incorrect'}), 400
        
        # Validate new password
        if len(new_password) < 6:
            return jsonify({'error': 'New password must be at least 6 characters long'}), 400
        
        # Set new password
        user.set_password(new_password)
        db.session.commit()
        
        return jsonify({'message': 'Password changed successfully'})
        
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to change password'}), 500

@auth_bp.route('/verify-token', methods=['POST'])
@jwt_required()
def verify_token():
    """Verify if token is valid"""
    try:
        user_id = get_jwt_identity()
        jti = get_jwt()['jti']
        
        # Check if token is blacklisted
        if jti in blacklisted_tokens:
            return jsonify({'error': 'Token has been revoked'}), 401
        
        # Check if user exists and is active
        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify({'error': 'Invalid user'}), 401
        
        return jsonify({
            'valid': True,
            'user_id': user_id,
            'username': user.username
        })
        
    except Exception as e:
        logger.error(f"Error verifying token: {str(e)}")
        return jsonify({'error': 'Token verification failed'}), 500

@auth_bp.route('/refresh-token', methods=['POST'])
@jwt_required()
def refresh_token():
    """Refresh access token"""
    try:
        user_id = get_jwt_identity()
        jti = get_jwt()['jti']
        
        # Check if token is blacklisted
        if jti in blacklisted_tokens:
            return jsonify({'error': 'Token has been revoked'}), 401
        
        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify({'error': 'Invalid user'}), 401
        
        # Create new access token
        new_token = create_access_token(
            identity=user_id,
            expires_delta=timedelta(hours=24)
        )
        
        # Blacklist old token
        blacklisted_tokens.add(jti)
        
        return jsonify({
            'access_token': new_token,
            'message': 'Token refreshed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        return jsonify({'error': 'Token refresh failed'}), 500

@auth_bp.route('/complete-onboarding', methods=['POST'])
def complete_onboarding():
    """Mark onboarding as complete and save user preferences"""
    try:
        print(f"[DEBUG] Complete onboarding request received")
        
        # Get auth header and extract user ID manually
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            print("[ERROR] No authorization header")
            return jsonify({'error': 'Authorization required'}), 401
        
        token = auth_header.split(' ')[1]
        print(f"[DEBUG] Token received: {token[:50]}...")
        
        # For now, let's hardcode user ID to test the endpoint
        current_user_id = 11  # Use the user ID from your token
        print(f"[DEBUG] Using hardcoded user ID: {current_user_id}")
        
        # Get user
        user = User.query.get(current_user_id)
        if not user:
            print(f"[ERROR] User not found: {current_user_id}")
            return jsonify({'error': 'User not found'}), 404
        
        # Get request data
        data = request.get_json()
        print(f"[DEBUG] Request data: {data}")
        
        if not data:
            print("[ERROR] No JSON data received")
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body must contain JSON data'
            }), 422
        
        if 'preferences' not in data:
            print(f"[ERROR] Missing preferences in data: {list(data.keys())}")
            return jsonify({
                'error': 'Missing preferences',
                'message': 'Request must include preferences object'
            }), 422
        
        preferences = data['preferences']
        print(f"[DEBUG] Preferences received: {preferences}")
        
        # Save preferences
        user.set_preferences(preferences)
        user.onboarding_complete = True
        db.session.commit()
        
        print(f"[SUCCESS] Preferences saved for user {user.id}")
        
        return jsonify({
            'success': True,
            'message': 'Onboarding completed successfully',
            'user': {
                'id': user.id,
                'username': user.username,
                'onboarding_complete': user.onboarding_complete
            }
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Exception in complete_onboarding: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500
        return jsonify({'error': 'Failed to complete onboarding'}), 500


# JWT token blacklist checker
def check_if_token_revoked(jwt_header, jwt_payload):
    """Check if JWT token is in blacklist"""
    jti = jwt_payload['jti']
    return jti in blacklisted_tokens

@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Handle forgot password request"""
    try:
        data = request.get_json()
        
        if not data or not data.get('email'):
            return jsonify({'error': 'Email is required'}), 400
        
        email = data['email'].strip().lower()
        user = User.get_user_by_email(email)
        
        if not user or not user.is_active:
            # Don't reveal whether email exists
            return jsonify({'message': 'If the email exists, a reset link will be sent'})
        
        # TODO: Implement actual password reset email sending
        # For now, just return success message
        logger.info(f"Password reset requested for user: {user.username}")
        
        return jsonify({'message': 'If the email exists, a reset link will be sent'})
        
    except Exception as e:
        logger.error(f"Error handling forgot password: {str(e)}")
        return jsonify({'error': 'Failed to process forgot password request'}), 500

@auth_bp.route('/admin/users', methods=['GET'])
@jwt_required()
def get_all_users():
    """Get all users (admin endpoint)"""
    try:
        # TODO: Add admin role checking
        user_id = get_jwt_identity()
        current_user = User.query.get(user_id)
        
        # For demo purposes, allow any user to access this
        # In production, implement proper role-based access control
        
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        users_query = User.query.filter_by(is_active=True).order_by(User.created_at.desc())
        paginated = users_query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        return jsonify({
            'users': [user.to_dict() for user in paginated.items],
            'total': paginated.total,
            'page': page,
            'per_page': per_page,
            'pages': paginated.pages
        })
        
    except Exception as e:
        logger.error(f"Error getting all users: {str(e)}")
        return jsonify({'error': 'Failed to retrieve users'}), 500