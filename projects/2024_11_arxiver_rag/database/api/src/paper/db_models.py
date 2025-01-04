from datetime import datetime
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db_models import BaseMixin
from core.database import Base


class PaperInformation(BaseMixin, Base):
    __tablename__ = "paper_information"
    
    paper_id: Mapped[str] = mapped_column(Text, nullable=False)
    published_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[str] = mapped_column(Text, nullable=False)
    link: Mapped[str] = mapped_column(Text, nullable=False)
    
class PaperStatus(BaseMixin, Base):
    __tablename__ = "paper_status"
    
    paper_information_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("paper_information.id"), nullable=False
    )
    file_extension: Mapped[str] = mapped_column(Text, nullable=False)
    parse_status: Mapped[str] = mapped_column(Text, nullable=False)
    extract_status: Mapped[str] = mapped_column(Text, nullable=False)
    split_status: Mapped[str] = mapped_column(Text, nullable=False)
    embed_status: Mapped[str] = mapped_column(Text, nullable=False)