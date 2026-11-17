
import os
import json
import joblib

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("artifacts", exist_ok=True)

print("Loading dataset...")

data = load_diabetes()
X = data.data
y = data.target

print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

joblib.dump(model, "artifacts/model.joblib")

with open("artifacts/metrics.json", "w") as f:
    json.dump({"mse": float(mse), "r2_score": float(r2)}, f)

print("MSE:", mse)
print("R2 Score:", r2)
print("MODEL TRAINED AND SAVED SUCCESSFULLY!")
