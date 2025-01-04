from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from config import get_settings

from core.dto.response import ApiResponse
from core.errors import NotFoundException
from core.logger import logger

from paper.routes import router as paper_router

settings = get_settings()

app = FastAPI()
app.include_router(paper_router)

if settings.app_env == "dev":
    @app.on_event("startup")
    async def print_routes():
        for route in app.router.routes:
            logger.info(f"Route: {route.path}, Methods: {route.methods}")


@app.exception_handler(Exception)
async def unknown_exception_handler(request: Request, exc: Exception):
    base_response = ApiResponse[str](code=1000, message="Unknown Error", data=str(exc))
    logger.error(f"[Error] {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=base_response.model_dump(),
    )

@app.exception_handler(NotFoundException)
async def unknown_exception_handler(request: Request, exc: NotFoundException):
    base_response = ApiResponse[str](code=1001, message="Not Found", data=str(exc))
    logger.info(f"[NotFound] {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=base_response.model_dump(),
    )


@app.exception_handler(SQLAlchemyError)
async def unknown_sql_alchemy_exception(request: Request, exc: SQLAlchemyError):
    base_response = ApiResponse[str](
        code=1100, message="Unknown SQLAlchemy Error", data=str(exc)
    )
    logger.error(f"[SQLAlchemy Error] {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=base_response.model_dump(),
    )

@app.exception_handler(IntegrityError)
async def integrity_error_exception_handler(request: Request, exc: IntegrityError):
    base_response = ApiResponse[str](code=1101, message="SQLAlchemy Integrity Error", data=str(exc))
    logger.error(f"[SQLAlchemy IntegrityError] {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=base_response.model_dump(),
    )