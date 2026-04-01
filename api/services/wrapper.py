from api.services.state import latest_data

def get_all_objects():
    print("\n===== SERVING FROM CACHE =====\n")
    return latest_data

def detect_collisions():
    print("SERVING FROM CACHE")
    return latest_data

def compute_maneuver(input_data=None):
    print("SERVING FROM CACHE")
    return latest_data