
import os
import json
import shutil

print("STARTING MODEL DEPLOYMENT...")

with open("artifacts/metrics.json", "r") as f:
    metrics = json.load(f)

r2 = metrics["r2_score"]

print("Model R2 Score:", round(r2, 4))

if r2 >= 0.30:
    os.makedirs("deployed", exist_ok=True)

    shutil.copy(
        "artifacts/model.joblib",
        "deployed/model.joblib"
    )

    print("MODEL VALIDATED SUCCESSFULLY!")
    print("MODEL DEPLOYED SUCCESSFULLY!")

else:
    print("MODEL VALIDATION FAILED!")
    print("Deployment cancelled.")
