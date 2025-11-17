from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class EntityLabel(str, Enum):
    PS="PS"
    LC="LC"
    OG="OG"
    DT="DT"
    TI="TI"
    QT="QT"

class Entity(BaseModel):
    value: str = Field(..., description="엔티티 표면 문자열")
    label: EntityLabel = Field(..., description="엔티티 라벨 (PS/LC/OG/DT/TI/QT)")
    # sentence: str = Field(..., description="엔티티가 등장한 문장")

class NerResult(BaseModel):
    entities: List[Entity] = Field(..., description="추출된 엔티티 리스트")
    tagged_text: str = Field(..., description="인라인 태그가 삽입된 전체 텍스트")