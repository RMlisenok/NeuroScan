from functools import wraps
from flask import abort
from flask_login import current_user


def admin_required(f):
    """Доступ только для админов."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return wrapper


def doctor_or_admin_required(f):
    """Доступ для врачей и админов."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            abort(403)
        if not current_user.has_role("doctor", "admin"):
            abort(403)
        return f(*args, **kwargs)
    return wrapper


def doctor_required(f):
    """Доступ только для врачей."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_doctor:
            abort(403)
        return f(*args, **kwargs)
    return wrapper