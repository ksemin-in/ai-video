"""
app/api/routes/auth.py — JWT login & user registration
"""
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token, get_current_user
from app.core.config import settings
from app.models import User

router = APIRouter()


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    role: str = "viewer"


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool


@router.post("/auth/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(select(User).where(User.email == form.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(
        {"sub": str(user.id), "email": user.email, "role": user.role}
    )
    return Token(access_token=token)


@router.post("/auth/register", response_model=UserOut, status_code=201)
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    existing = await db.execute(select(User).where(User.email == data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=data.email,
        full_name=data.full_name,
        hashed_pw=hash_password(data.password),
        role=data.role,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return UserOut(id=user.id, email=user.email, full_name=user.full_name, role=user.role, is_active=user.is_active)


@router.get("/auth/me", response_model=dict)
async def me(current_user=Depends(get_current_user)):
    return current_user
