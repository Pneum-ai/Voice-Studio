"""FastAPI dependency accessors for explicit process services."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.container import Services


def get_services(request: Request) -> Services:
  """Return the initialized service graph attached during application setup."""
  services: Services = request.app.state.services
  return services


async def get_session(
  services: Annotated[Services, Depends(get_services)],
) -> AsyncIterator[AsyncSession]:
  """Yield one short-lived async database session for a request."""
  async with services.database.sessions() as session:
    yield session


ServiceDependency = Annotated[Services, Depends(get_services)]
SessionDependency = Annotated[AsyncSession, Depends(get_session)]
