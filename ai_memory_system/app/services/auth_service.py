import hashlib
import hmac
import secrets
import threading

# In-memory stores keep this implementation simple for local development.
_USERS = {}
_SESSIONS = {}
_LOCK = threading.Lock()


def _hash_password(password, salt):
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return pwd_hash.hex()


def _encode_password(password):
    salt = secrets.token_bytes(16)
    pwd_hash = _hash_password(password, salt)
    return f"{salt.hex()}${pwd_hash}"


def _verify_password(password, encoded_password):
    salt_hex, expected_hash = encoded_password.split("$", 1)
    salt = bytes.fromhex(salt_hex)
    actual_hash = _hash_password(password, salt)
    return hmac.compare_digest(actual_hash, expected_hash)


def signup_user(username, password):
    with _LOCK:
        if username in _USERS:
            raise ValueError("Username already exists")
        _USERS[username] = _encode_password(password)


def login_user(username, password):
    with _LOCK:
        encoded_password = _USERS.get(username)
        if not encoded_password:
            return None
        if not _verify_password(password, encoded_password):
            return None

        token = secrets.token_urlsafe(32)
        _SESSIONS[token] = username
        return token


def get_user_from_token(token):
    with _LOCK:
        return _SESSIONS.get(token)
