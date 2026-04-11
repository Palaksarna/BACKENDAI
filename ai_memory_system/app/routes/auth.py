from fastapi import APIRouter, HTTPException, status

from ..models.schema import (
    AuthTokenResponse,
    LoginRequest,
    MessageResponse,
    SignupRequest,
)
from ..services.auth_service import login_user, signup_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest):
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    try:
        signup_user(req.username, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"message": "Signup successful"}


@router.post("/login", response_model=AuthTokenResponse)
def login(req: LoginRequest):
    token = login_user(req.username, req.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"access_token": token, "token_type": "bearer"}
