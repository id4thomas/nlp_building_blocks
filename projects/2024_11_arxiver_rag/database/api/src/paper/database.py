from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import get_settings

settings = get_settings()

async_engine = create_async_engine(
    url=settings.postgres_url, echo=settings.app_env == "dev"
)

AsyncSessionLocal: sessionmaker[AsyncSession] = sessionmaker(
    autocommit=False, autoflush=False, bind=async_engine, class_=AsyncSession
)

Base = declarative_base()


# Dependency for getting the async DB session
async def get_info_db():
    async with AsyncSessionLocal() as session:
        yield session