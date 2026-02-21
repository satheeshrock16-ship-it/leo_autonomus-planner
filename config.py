"""Global configuration for the autonomous LEO collision avoidance prototype."""
from pathlib import Path
import os

from utils.config_loader import load_config

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SATELLITE_DATA_DIR = DATA_DIR / "satellite"
DEBRIS_DATA_DIR = DATA_DIR / "debris"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "visualization" / "artifacts"
RESULTS_DIR = ROOT / "results"

for path in [SATELLITE_DATA_DIR, DEBRIS_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, PLOTS_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

SPACE_TRACK_IDENTITY = os.getenv("SPACE_TRACK_IDENTITY", "")
SPACE_TRACK_PASSWORD = os.getenv("SPACE_TRACK_PASSWORD", "")
SPACE_TRACK_BASE_URL = "https://www.space-track.org"

EARTH_MU_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.137

CONFIG = load_config()
PROPAGATION_CONFIG = CONFIG["propagation"]
COVARIANCE_CONFIG = CONFIG["covariance"]
MANEUVER_CONFIG = CONFIG["maneuver"]
PERFORMANCE_CONFIG = CONFIG["performance"]
TCA_REFINEMENT_CONFIG = CONFIG["tca_refinement"]
SYNTHETIC_ML_CONFIG = CONFIG["synthetic_ml"]

# Decision settings
COLLISION_PROBABILITY_THRESHOLD = 0.10
SAFE_MISS_DISTANCE_KM = 2.0
SERIAL_PORT = os.getenv("THRUSTER_SERIAL_PORT", "/dev/ttyUSB0")
SERIAL_BAUDRATE = int(os.getenv("THRUSTER_SERIAL_BAUDRATE", "115200"))
