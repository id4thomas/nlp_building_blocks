from datetime import datetime
from sqlalchemy import DateTime, Integer, func, inspect
from sqlalchemy.orm import Mapped, mapped_column


class BaseMixin:
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in inspect(self).attrs.keys()}