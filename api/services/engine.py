import time
from pipeline.run_full_pipeline import run_autonomous_cycle
from api.services.state import latest_data

def run_engine():
    while True:
        try:
            result = run_autonomous_cycle(fetch_live_data=False, benchmark_mode=False)
            latest_data.clear()
            latest_data.update(result)
            print("ENGINE UPDATED DATA")
        except Exception as e:
            print("ENGINE ERROR:", e)

        time.sleep(60)
