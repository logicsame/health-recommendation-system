# backend/app/models/user.py
from pydantic import BaseModel, EmailStr, Field
import datetime

class UserBase(BaseModel):
    email: EmailStr
    age: int = Field(..., gt=0, le=120)
    height: float = Field(..., gt=0) # in cm
    weight: float = Field(..., gt=0) # in kg
    gender: str
    activity_level: str
    health_conditions: str = "None"

class UserCreate(UserBase):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserProfile(UserBase):
    username: str
    bmi: float
    join_date: str # Store as string for simplicity, could use date

class UserInDB(UserProfile):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

