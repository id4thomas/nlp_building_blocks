from pydantic import Field
from typing import List, Optional, TYPE_CHECKING

from core.data import BaseData

from paper.properties import (
    INFORMATION_FIELD_DESCRIPTIONS,
    STATUS_FIELD_DESCRIPTIONS
)

class PaperInformation(BaseData):
    paper_id: str = Field(..., description=INFORMATION_FIELD_DESCRIPTIONS["paper_id"])
    created_at: datetime = Field(..., description=INFORMATION_FIELD_DESCRIPTIONS["published_date"])
    title: str = Field(..., description=INFORMATION_FIELD_DESCRIPTIONS["title"])
    authors: str = Field(..., description=INFORMATION_FIELD_DESCRIPTIONS["authors"])
    link: str = Field(..., description=INFORMATION_FIELD_DESCRIPTIONS["link"])

class PaperStatus(BaseData):
    paper_information_id: int = Field(..., description=STATUS_FIELD_DESCRIPTIONS["paper_information_id"])
    file_extension: str = Field("pdf", description=STATUS_FIELD_DESCRIPTIONS["file_extension"])
    parse_status: str = Field("PENDING", description=STATUS_FIELD_DESCRIPTIONS["parse_status"])
    extract_status: str = Field("PENDING", description=STATUS_FIELD_DESCRIPTIONS["extract_status"])
    split_status: str = Field("PENDING", description=STATUS_FIELD_DESCRIPTIONS["split_status"])
    embed_status: str = Field("PENDING", description=STATUS_FIELD_DESCRIPTIONS["embed_status"])
