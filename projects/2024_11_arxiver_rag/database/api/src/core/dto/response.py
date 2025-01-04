from datetime import datetime
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int = 0
    message: str = "success"
    data: Optional[T] = None


class BaseResponse(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.strftime("%Y-%m-%d %H:%M:%S")}


class BaseSearchResponse(BaseModel, Generic[T]):
    total_count: int
    items: List[T]