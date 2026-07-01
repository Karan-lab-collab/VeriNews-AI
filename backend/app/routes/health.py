from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    message: str


@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check() -> HealthResponse:
    """
    Returns the operational status of the VeriNews AI backend.
    """
    return HealthResponse(
        status="ok",
        message="VeriNews AI backend is running.",
    )
