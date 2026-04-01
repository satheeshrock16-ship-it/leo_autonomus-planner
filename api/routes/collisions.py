from fastapi import APIRouter

from ..services.wrapper import detect_collisions

router = APIRouter(prefix="/collisions", tags=["collisions"])

@router.get("/")
def get_collisions():
    result = detect_collisions()
    if isinstance(result, list):
        return {"collisions": result}
    else:
        return result