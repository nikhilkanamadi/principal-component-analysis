from fastapi import APIRouter
from app.schemas.models import MonitorResponse
from app.core import monitoring

router = APIRouter(prefix="/monitor", tags=["Monitoring"])


@router.get("", response_model=MonitorResponse)
def monitor(location: str = "New York"):
    report = monitoring.get_report(location)
    return MonitorResponse(**report)
