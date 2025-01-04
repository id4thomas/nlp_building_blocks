from fastapi import APIRouter

from config import get_settings

settings = get_settings()

router = APIRouter(prefix="/paper")