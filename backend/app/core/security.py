# backend/app/core/security.py
import hashlib
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import settings

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY", "a_very_secret_key_for_jwt_should_be_in_env") # Replace with a strong, env-based key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 # Token validity period

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password using passlib."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain password using passlib."""
    return pwd_context.hash(password)

# --- JWT Token Handling ---

class TokenPayload(BaseModel):
    sub: Optional[str] = None # Subject (usually username or user ID)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[TokenPayload]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None # Or raise credentials_exception
        token_data = TokenPayload(sub=username)
    except JWTError:
        return None # Or raise credentials_exception
    return token_data

# --- Simple Hashlib (as used in original app1.py, kept for reference if needed) ---
def verify_password_simple_sha256(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a simple SHA256 hash."""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def get_password_hash_simple_sha256(password: str) -> str:
    """Hashes a plain password using simple SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

