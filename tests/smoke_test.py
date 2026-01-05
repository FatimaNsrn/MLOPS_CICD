import requests
import time


def test_smoke_prediction():
    # wait for FastAPI to start
    time.sleep(5)

    payload = {
        "product_id": 1,
        "category": "electronics",
        "brand": "brand_a",
        "store_id": 10,
        "store_location": "istanbul",
        "promotion_type": "discount",
        "day_of_week": "monday",
        "day_of_year": 120,
        "season": "spring",
        "base_price": 100.0,
        "month": 4,
        "discount_rate": 0.1,
        "avg_units_sold_30d": 20.0,
        "avg_customers_30d": 50.0,
        "is_holiday": 0
    }

    response = requests.post(
        "http://localhost:8000/predict",
        json=payload
    )

    assert response.status_code == 200
    assert "predicted_class" in response.json()
