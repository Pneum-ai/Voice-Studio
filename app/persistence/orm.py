"""SQLAlchemy mappings for the fresh runtime schema."""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
  """Base class for all runtime mappings."""


class SchemaMetadataRecord(Base):
  """Persist schema-level version and migration metadata."""

  __tablename__ = "schema_metadata"

  key: Mapped[str] = mapped_column(String, primary_key=True)
  value: Mapped[str] = mapped_column(String, nullable=False)


class ProfileRecord(Base):
  """Persist one named voice profile."""

  __tablename__ = "profiles"
  __table_args__ = (
    UniqueConstraint("name_key", name="uq_profiles_name_key"),
    Index("ix_profiles_created_at", "created_at_ms"),
  )

  id: Mapped[str] = mapped_column(String, primary_key=True)
  name: Mapped[str] = mapped_column(String(100), nullable=False)
  name_key: Mapped[str] = mapped_column(String(100), nullable=False)
  created_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)
  updated_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)


class ProfileSampleRecord(Base):
  """Persist one normalized reference sample."""

  __tablename__ = "profile_samples"
  __table_args__ = (
    Index("ix_profile_samples_profile_created", "profile_id", "created_at_ms"),
    Index("ix_profile_samples_content_sha256", "content_sha256"),
  )

  id: Mapped[str] = mapped_column(String, primary_key=True)
  profile_id: Mapped[str] = mapped_column(
    String,
    ForeignKey("profiles.id", ondelete="CASCADE"),
    nullable=False,
  )
  audio_file: Mapped[str] = mapped_column(String, nullable=False)
  transcript: Mapped[str] = mapped_column(Text, nullable=False)
  style_controls_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
  duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
  sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
  content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
  created_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)


class GenerationRecord(Base):
  """Persist one saved generation and its complete request metadata."""

  __tablename__ = "generations"
  __table_args__ = (
    Index("ix_generations_created_at", "created_at_ms"),
    Index("ix_generations_model_created", "model", "created_at_ms"),
    Index("ix_generations_profile_created", "profile_id", "created_at_ms"),
  )

  id: Mapped[str] = mapped_column(String, primary_key=True)
  profile_id: Mapped[str | None] = mapped_column(
    String,
    ForeignKey("profiles.id", ondelete="SET NULL"),
    nullable=True,
  )
  profile_name: Mapped[str] = mapped_column(String(100), nullable=False)
  input_text: Mapped[str] = mapped_column(Text, nullable=False)
  language: Mapped[str] = mapped_column(String(16), nullable=False)
  model: Mapped[str] = mapped_column(String(100), nullable=False)
  seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
  instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
  audio_file: Mapped[str] = mapped_column(String, nullable=False)
  audio_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
  duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
  sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
  created_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)
