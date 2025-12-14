#!/usr/bin/env python3
import requests
import json

# Test the predict endpoint
url = "http://localhost:8001/predict"
data = {
    "ilce": "Kadıköy",
    "oda_sayisi": 3,
    "salon_sayisi": 1,
    "brut_alan": 120,
    "net_alan": 100,
    "yas": 5,
    "kat": 3,
    "toplam_kat": 5,
    "isitma": "Merkezi",
    "banyo_sayisi": 1
}

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print("API Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Check if dataset prices are included
        if "comparison" in result and "dataset_prices" in result["comparison"]:
            print("\n✅ Dataset prices successfully included in response!")
            print(f"Number of dataset prices: {len(result['comparison']['dataset_prices'])}")
            print(f"Min price: {result['comparison']['dataset_price_min']}")
            print(f"Max price: {result['comparison']['dataset_price_max']}")
            print(f"Median price: {result['comparison']['dataset_price_median']}")
        else:
            print("\n❌ Dataset prices not found in response")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Connection error: {e}")