# Author: Mezred Mohamed Wassim
# Role: MLOps / Monitoring Engineer

import os

def trigger_retrain():
    os.makedirs("signals", exist_ok=True)
    open("signals/retrain.txt", "w").close()

def trigger_fallback():
    os.makedirs("signals", exist_ok=True)
    open("signals/fallback.txt", "w").close()
