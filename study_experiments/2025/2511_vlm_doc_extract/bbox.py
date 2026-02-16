from pydantic import BaseModel

class Bbox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    
    class Config:
        extra="forbid"