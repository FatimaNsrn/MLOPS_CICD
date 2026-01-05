from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import mlflow
from contextlib import asynccontextmanager

MODEL_URI = r"file:./training/mlruns/1/models/m-132965c0c7ba47158567fd33de7f03fc"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        print("🔄 Loading ML model...")
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Failed to load model:", e)
        model = None

    yield  # app runs here

    print("🛑 App shutdown")


app = FastAPI(
    title="Profit Classification Service",
    lifespan=lifespan
)


# ---------- SCHEMAS ----------
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
    discount_rate: float
    units_sold_avg: float
    avg_customers_30d: float
    is_holiday: int


class PredictionResponse(BaseModel):
    predicted_class: str


# ---------- ROUTES ----------
@app.get("/")
def root():
    return {
        "status": "running",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        return {"predicted_class": "model_not_loaded"}

    df = pd.DataFrame([request.model_dump()])
    prediction = model.predict(df)[0]

    return {"predicted_class": prediction}
