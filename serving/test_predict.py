import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "product_id": 123,
    "category": "Electronics",
    "brand": "BrandX",
    "store_id": 12,
    "store_location": "Istanbul",
    "promotion_type": "Discount",
    "day_of_week": "Monday",
    "day_of_year": 15,
    "season": "Winter",
    "base_price": 100.0,
    "month": 1,
    "discount_rate": 0.2,
    "avg_units_sold_30d": 50.0,
    "avg_customers_30d": 100.0,
    "is_holiday": 0
}

response = requests.post(url, json=data)

try:
    print(response.json())
except Exception as e:
    print("Error decoding JSON:", e)
    print("Response text:", response.text)
