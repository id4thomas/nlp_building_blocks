from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field

class ArxivPaperSection(BaseModel):
    header: Literal["h2", "h3", "h4", "h5", "h6", "p"]
    title:str = Field("", description="Section title")
    text: str = Field("", description = "Section contents")
    children: List["ArxivPaperSection"] = Field(list(), description="child sections")

class ArxivPaperMetadata(BaseModel):
    authors: str
    published_date: datetime
    link: str

class ArxivPaper(BaseModel):
    id: str = Field(...)
    title: str = Field(..., description="paper title")
    abstract: str = Field(..., description="paper abstract")
    sections: List[ArxivPaperSection] = Field(..., description="sections of the paper")
    metadata: ArxivPaperMetadata = Field(None, description="paper metadata")