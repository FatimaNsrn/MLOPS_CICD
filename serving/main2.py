from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# ------------------------
# 1️⃣ Set MLflow tracking URI
# ------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ------------------------
# 2️⃣ Load registered model
# ------------------------
MODEL_NAME = "profit_classifier_pipeline"
MODEL_STAGE_OR_ALIAS = "champion"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}@{MODEL_STAGE_OR_ALIAS}"
)

# ------------------------
# 3️⃣ FastAPI app
# ------------------------
app = FastAPI(title="Profit Classification Service")

# ---- Request schema ----
class PredictionRequest(BaseModel):
    product_id: int
    category: str
    brand: str
    store_id: int
    store_location: str
    promotion_type: str
    day_of_week: str
    day_of_year: int
    season: str
    base_price: float
    month: int
    discount_rate: float
    avg_units_sold_30d: float
    avg_customers_30d: float
    is_holiday: int

class PredictionResponse(BaseModel):
    predicted_class: str

# ---- Endpoint ----
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    df = pd.DataFrame([request.model_dump()])
    print(df) 
    print(df.dtypes)
    prediction = model.predict(df)[0][0]
    return {"predicted_class": prediction}

@app.get("/")
def root():
    return {"message": "Profit Classification Service is running!"}
