"""
精简版数据库模块

提供SQLAlchemy 2.0+异步API支持models，兼容SQLite(开发)和PostgreSQL(生产)
包含Article和Report数据模型，支持完整的CRUD操作
"""

import asyncio
import logging
import os
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator, Optional, Dict, Any
from urllib.parse import urlparse

from sqlalchemy import (
    create_engine, MetaData, Index, CheckConstraint, text,
    String, Text, Integer
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine, AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.dialects.sqlite import JSON
import uuid

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 自定义异常类
class DatabaseError(Exception):
    """数据库操作异常基类"""
    pass

class ValidationError(DatabaseError):
    """数据验证异常"""
    pass

class ConstraintError(DatabaseError):
    """约束违反异常"""
    pass

# 枚举类型
class ReportType(str, Enum):
    """报告类型枚举"""
    SINGLE = "single"
    PAIRWISE = "pairwise"

# 数据库配置
class DatabaseSettings(BaseSettings):
    """数据库配置设置"""
    database_url: str = Field(default="sqlite+aiosqlite:///./app.db", description="数据库连接URL")
    echo: bool = Field(default=True, description="是否打印SQL语句(开发模式)")
    pool_size: int = Field(default=10, description="连接池大小")
    max_overflow: int = Field(default=20, description="最大溢出连接数")
    
    class Config:
        env_file = ".env"
        env_prefix = "DB_"

db_settings = DatabaseSettings()

# 数据库模型基类
class Base(DeclarativeBase):
    """数据库模型基类"""
    metadata = MetaData()

# Article模型
class Article(Base):
    """文章数据模型"""
    __tablename__ = "articles"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    author: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, unique=True, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    publish_time: Mapped[Optional[datetime]] = mapped_column(nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now(), nullable=False, index=True)
    
    __table_args__ = (
        Index("idx_article_title_author", "title", "author"),
        Index("idx_article_publish_time", "publish_time"),
    )
    
    def __repr__(self) -> str:
        return f"Article(id={self.id}, title='{self.title[:30]}...', author='{self.author}')"

# Report模型
class Report(Base):
    """检测报告数据模型"""
    __tablename__ = "reports"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    report_type: Mapped[ReportType] = mapped_column(String(20), nullable=False, index=True)
    results: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB if "postgresql" in db_settings.database_url else JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(default=func.now(), nullable=False, index=True)
    
    __table_args__ = (
        CheckConstraint("report_type IN ('single', 'pairwise')", name="ck_report_type"),
        Index("idx_report_user_type", "user_id", "report_type"),
    )
    
    def __repr__(self) -> str:
        return f"Report(id={self.id}, user_id={self.user_id}, type='{self.report_type}')"

# 创建数据库引擎
def create_database_engine() -> AsyncEngine:
    """创建数据库引擎"""
    database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")
    connect_args = {"check_same_thread": False} if "sqlite" in database_url else {}
    
    # SQLite不支持连接池参数
    if "sqlite" in database_url:
        return create_async_engine(
            database_url,
            echo=db_settings.echo,
            connect_args=connect_args,
            future=True,
        )
    else:
        return create_async_engine(
            database_url,
            echo=db_settings.echo,
            pool_size=db_settings.pool_size,
            max_overflow=db_settings.max_overflow,
            connect_args=connect_args,
            future=True,
        )

# 创建异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=create_database_engine(),
    class_=AsyncSession,
    expire_on_commit=False,
    future=True
)

# 数据库初始化
async def init_db() -> None:
    """异步初始化数据库表"""
    try:
        logger.info("初始化数据库...")
        engine = create_database_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # 测试连接
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        logger.info("数据库初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise DatabaseError(f"数据库初始化失败: {e}") from e

# 会话依赖生成器
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """异步会话依赖生成器"""
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        if "UNIQUE constraint failed" in str(e):
            raise ConstraintError(f"唯一约束违反: {e}") from e
        raise DatabaseError(f"数据库操作失败: {e}") from e
    finally:
        await session.close()

# 数据验证模型
class ArticleCreate(BaseModel):
    """创建文章的数据验证模型"""
    title: str = Field(..., min_length=1, max_length=255)
    author: Optional[str] = Field(None, max_length=100)
    source_url: Optional[str] = Field(None, max_length=512)
    content: str = Field(..., min_length=1)
    publish_time: Optional[datetime] = None
    
    @validator('source_url')
    def validate_source_url(cls, v):
        if v:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]) or result.scheme not in ['http', 'https']:
                raise ValueError('无效的URL格式')
        return v

class ArticleUpdate(BaseModel):
    """更新文章的数据验证模型"""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    author: Optional[str] = Field(None, max_length=100)
    source_url: Optional[str] = Field(None, max_length=512)
    content: Optional[str] = Field(None, min_length=1)
    publish_time: Optional[datetime] = None

class ReportCreate(BaseModel):
    """创建报告的数据验证模型"""
    user_id: int = Field(..., ge=1)
    report_type: ReportType
    results: Optional[Dict[str, Any]] = None

class ReportUpdate(BaseModel):
    """更新报告的数据验证模型"""
    user_id: Optional[int] = Field(None, ge=1)
    report_type: Optional[ReportType] = None
    results: Optional[Dict[str, Any]] = None

# CRUD操作函数
async def create_article(session: AsyncSession, article_data: ArticleCreate) -> Article:
    """创建文章"""
    try:
        article = Article(**article_data.dict())
        session.add(article)
        await session.flush()
        await session.refresh(article)
        logger.info(f"创建文章成功: {article.id}")
        return article
    except Exception as e:
        logger.error(f"创建文章失败: {e}")
        raise

async def get_article(session: AsyncSession, article_id: uuid.UUID) -> Optional[Article]:
    """获取文章"""
    return await session.get(Article, article_id)

async def update_article(session: AsyncSession, article_id: uuid.UUID, article_data: ArticleUpdate) -> Optional[Article]:
    """更新文章"""
    article = await session.get(Article, article_id)
    if not article:
        return None
    
    update_data = article_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(article, field, value)
    
    await session.flush()
    await session.refresh(article)
    return article

async def delete_article(session: AsyncSession, article_id: uuid.UUID) -> bool:
    """删除文章"""
    article = await session.get(Article, article_id)
    if not article:
        return False
    
    await session.delete(article)
    return True

async def create_report(session: AsyncSession, report_data: ReportCreate) -> Report:
    """创建报告"""
    try:
        report = Report(**report_data.dict())
        session.add(report)
        await session.flush()
        await session.refresh(report)
        logger.info(f"创建报告成功: {report.id}")
        return report
    except Exception as e:
        logger.error(f"创建报告失败: {e}")
        raise

async def get_report(session: AsyncSession, report_id: uuid.UUID) -> Optional[Report]:
    """获取报告"""
    return await session.get(Report, report_id)

async def update_report(session: AsyncSession, report_id: uuid.UUID, report_data: ReportUpdate) -> Optional[Report]:
    """更新报告"""
    report = await session.get(Report, report_id)
    if not report:
        return None
    
    update_data = report_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(report, field, value)
    
    await session.flush()
    await session.refresh(report)
    return report

async def delete_report(session: AsyncSession, report_id: uuid.UUID) -> bool:
    """删除报告"""
    report = await session.get(Report, report_id)
    if not report:
        return False
    
    await session.delete(report)
    return True

# 健康检查
async def check_database_health() -> Dict[str, Any]:
    """检查数据库健康状态"""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            return {"status": "healthy", "database_url": os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# 导出主要组件
__all__ = [
    "Base", "Article", "Report", "AsyncSessionLocal", "init_db", "get_db",
    "check_database_health", "ArticleCreate", "ArticleUpdate", "ReportCreate", "ReportUpdate",
    "create_article", "get_article", "update_article", "delete_article",
    "create_report", "get_report", "update_report", "delete_report",
    "ReportType", "DatabaseError", "ValidationError", "ConstraintError", "db_settings"
]

# 简单测试
if __name__ == "__main__":
    async def main():
        try:
            await init_db()
            health = await check_database_health()
            print(f"数据库状态: {health['status']}")
        except Exception as e:
            print(f"测试失败: {e}")
    
    asyncio.run(main())
