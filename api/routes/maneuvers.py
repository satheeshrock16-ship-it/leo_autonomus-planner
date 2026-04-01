from fastapi import APIRouter

from ..services.wrapper import compute_maneuver

router = APIRouter(prefix="/maneuver", tags=["maneuver"])

@router.post("/")
def create_maneuver():
    result = compute_maneuver()
    if "error" not in result:
        return {"maneuver": result}
    else:
        return result