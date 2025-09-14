"""
Helper utilities for the hotel recommendation system.
This module contains common utility functions used across the application.
"""

import re
import json
import hashlib
import secrets
import string
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

def generate_random_string(length: int = 32, include_special: bool = False) -> str:
    """Generate a random string for tokens, passwords, etc."""
    characters = string.ascii_letters + string.digits
    if include_special:
        characters += "!@#$%^&*"
    
    return ''.join(secrets.choice(characters) for _ in range(length))

def hash_string(text: str, salt: str = None) -> str:
    """Create a SHA-256 hash of a string with optional salt."""
    if salt is None:
        salt = generate_random_string(16)
    
    combined = f"{text}{salt}"
    return hashlib.sha256(combined.encode()).hexdigest()

def validate_email(email: str) -> bool:
    """Validate email address format."""
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_url(url: str) -> bool:
    """Validate URL format."""
    if not url or not isinstance(url, str):
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing invalid characters."""
    if not filename:
        return "untitled"
    
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)  # Replace spaces with underscores
    sanitized = sanitized.strip('._')  # Remove leading/trailing dots and underscores
    
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    
    return sanitized or "untitled"

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency amount with proper symbols."""
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CAD': 'C$',
        'AUD': 'A$'
    }
    
    symbol = currency_symbols.get(currency, '$')
    
    if amount >= 1000000:
        return f"{symbol}{amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"{symbol}{amount/1000:.1f}K"
    else:
        return f"{symbol}{amount:.2f}"

def calculate_age(birth_date: datetime) -> int:
    """Calculate age from birth date."""
    if not isinstance(birth_date, datetime):
        return 0
    
    today = datetime.now()
    age = today.year - birth_date.year
    
    # Adjust if birthday hasn't occurred this year
    if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
        age -= 1
    
    return max(0, age)

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with optional suffix."""
    if not text or len(text) <= max_length:
        return text or ""
    
    return text[:max_length - len(suffix)] + suffix

def clean_phone_number(phone: str) -> str:
    """Clean and format phone number."""
    if not phone:
        return ""
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Format US phone numbers
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    
    return digits

def convert_to_timezone(dt: datetime, target_timezone: str = 'UTC') -> datetime:
    """Convert datetime to specified timezone."""
    if not isinstance(dt, datetime):
        return dt
    
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # For simplicity, just handle UTC conversion
        if target_timezone == 'UTC':
            return dt.astimezone(timezone.utc)
        
        return dt
    except Exception as e:
        logger.error(f"Timezone conversion error: {e}")
        return dt

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback."""
    if not json_string:
        return default
    
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parsing failed: {e}")
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely convert object to JSON string."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization failed: {e}")
        return default

def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula."""
    try:
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        
        return r * c
    except Exception as e:
        logger.error(f"Distance calculation error: {e}")
        return 0.0

def paginate_list(items: List, page: int = 1, per_page: int = 10) -> Dict[str, Any]:
    """Paginate a list of items."""
    if not items:
        return {
            'items': [],
            'total': 0,
            'page': page,
            'per_page': per_page,
            'pages': 0,
            'has_prev': False,
            'has_next': False
        }
    
    total = len(items)
    pages = (total - 1) // per_page + 1
    
    start = (page - 1) * per_page
    end = start + per_page
    
    return {
        'items': items[start:end],
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': pages,
        'has_prev': page > 1,
        'has_next': page < pages
    }

def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

def get_nested_value(data: Dict, path: str, default: Any = None) -> Any:
    """Get nested dictionary value using dot notation path."""
    try:
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    except Exception:
        return default

def set_nested_value(data: Dict, path: str, value: Any) -> Dict:
    """Set nested dictionary value using dot notation path."""
    keys = path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return data

def flatten_dict(data: Dict, prefix: str = '', separator: str = '.') -> Dict:
    """Flatten nested dictionary with dot notation keys."""
    flattened = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, separator))
        else:
            flattened[new_key] = value
    
    return flattened

def chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size."""
    if chunk_size <= 0:
        return [items]
    
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def calculate_percentage(part: float, total: float, decimal_places: int = 1) -> float:
    """Calculate percentage with specified decimal places."""
    if total == 0:
        return 0.0
    
    percentage = (part / total) * 100
    return round(percentage, decimal_places)

def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    if not filename or '.' not in filename:
        return ""
    
    return filename.rsplit('.', 1)[1].lower()

def is_valid_file_type(filename: str, allowed_types: List[str]) -> bool:
    """Check if file type is allowed."""
    extension = get_file_extension(filename)
    return extension in [t.lower() for t in allowed_types]

def generate_slug(text: str, max_length: int = 50) -> str:
    """Generate URL-friendly slug from text."""
    if not text:
        return ""
    
    # Convert to lowercase and replace spaces with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    slug = slug.strip('-')
    
    # Limit length
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')
    
    return slug

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure."""
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    import time
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts failed. Last error: {e}")
        
        raise last_exception
    
    return wrapper

def get_client_ip(request) -> str:
    """Get client IP address from request."""
    if not request:
        return "unknown"
    
    # Check for forwarded headers
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    
    forwarded = request.headers.get('X-Forwarded')
    if forwarded:
        return forwarded.split(',')[0].strip()
    
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    return request.remote_addr or "unknown"