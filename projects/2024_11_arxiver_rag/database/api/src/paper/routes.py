from fastapi import APIRouter

router = APIRouter(prefix="/paper")

from paper.interface.controllers.information import (
    router as information_router
)
from paper.interface.controllers.status import (
    router as status_router
)

requirement_router.include_router(information_router)
requirement_router.include_router(status_router)
