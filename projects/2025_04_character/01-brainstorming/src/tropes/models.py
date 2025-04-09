from pydantic import BaseModel

class Trope(BaseModel):
    trope_id: str
    name: str
    explanation: str