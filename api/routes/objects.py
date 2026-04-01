from fastapi import APIRouter
from backend_api import fetch_tle_data, generate_analysis

router = APIRouter()

@router.get("/objects")
def objects():
    print("ROUTE HIT")
    analysis = generate_analysis()
    objects_payload = fetch_tle_data()
    return {
        "analysis": analysis,
        "objects": objects_payload,
    }
