# API package initialization
from .routes import api_bp
from .auth import auth_bp
from .recommendations import recommendations_bp

__all__ = ['api_bp', 'auth_bp', 'recommendations_bp']