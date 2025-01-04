from datetime import datetime

# from dataclasses import dataclass
from pydantic import BaseModel


class BaseData(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime