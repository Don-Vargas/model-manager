from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import Integer, Boolean, DateTime, ForeignKey, JSON, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Base(DeclarativeBase):
    pass


class RepositoryModel(Base):
    __tablename__ = "repositories"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # repo_id
    commit_sha: Mapped[str] = mapped_column(String(64), nullable=False)
    gated: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    units: Mapped[List["ModelUnitModel"]] = relationship(
        "ModelUnitModel", back_populates="repository", cascade="all, delete-orphan"
    )


class ModelUnitModel(Base):
    __tablename__ = "model_units"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # repo_id#unit_id
    repo_id: Mapped[str] = mapped_column(ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    unit_type: Mapped[str] = mapped_column(String(50), nullable=False)
    task_type: Mapped[str] = mapped_column(String(100), default="unknown")
    framework: Mapped[str] = mapped_column(String(100), default="unknown")
    precision: Mapped[str] = mapped_column(String(50), default="unknown")
    quantization: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    base_model_name_or_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    unit_metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    repository: Mapped["RepositoryModel"] = relationship("RepositoryModel", back_populates="units")
    components: Mapped[List["ComponentModel"]] = relationship(
        "ComponentModel", back_populates="unit", cascade="all, delete-orphan"
    )
    direct_artifacts: Mapped[List["ArtifactModel"]] = relationship(
        "ArtifactModel", back_populates="unit", cascade="all, delete-orphan"
    )


class ComponentModel(Base):
    __tablename__ = "components"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # unit_id#comp_name
    unit_id: Mapped[str] = mapped_column(ForeignKey("model_units.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    component_type: Mapped[str] = mapped_column(String(100), nullable=False)

    unit: Mapped["ModelUnitModel"] = relationship("ModelUnitModel", back_populates="components")
    artifacts: Mapped[List["ArtifactModel"]] = relationship(
        "ArtifactModel", back_populates="component", cascade="all, delete-orphan"
    )


class ArtifactModel(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    unit_id: Mapped[Optional[str]] = mapped_column(ForeignKey("model_units.id", ondelete="CASCADE"), nullable=True)
    component_id: Mapped[Optional[str]] = mapped_column(ForeignKey("components.id", ondelete="CASCADE"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(50), nullable=False)
    files: Mapped[list] = mapped_column(JSON, default=list)
    is_sharded: Mapped[bool] = mapped_column(Boolean, default=False)
    index_file: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # NEW
    status: Mapped[str] = mapped_column(String(50), default="discovered")
    minio_bucket: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    minio_prefix: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    unit: Mapped[Optional["ModelUnitModel"]] = relationship("ModelUnitModel", back_populates="direct_artifacts")
    component: Mapped[Optional["ComponentModel"]] = relationship("ComponentModel", back_populates="artifacts")
    