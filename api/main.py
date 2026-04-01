from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
from api.services.engine import run_engine

from .routes.objects import router as objects_router
from .routes.collisions import router as collisions_router
from .routes.maneuvers import router as maneuvers_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def start_engine():
    thread = threading.Thread(target=run_engine, daemon=True)
    thread.start()

app.include_router(objects_router)
app.include_router(collisions_router)
app.include_router(maneuvers_router)