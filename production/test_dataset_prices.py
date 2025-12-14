#!/usr/bin/env python3
"""
API Dataset Fiyat Testi
Bu script API'nin dataset fiyatlarını döndürüp döndürmediğini test eder.
"""

import requests
import json
import sys
import os

# API URL
API_URL = "http://localhost:8002/predict"

# Test verisi
test_data = {
    "district": "Kadıköy",
    "net_m2": 100,
    "rooms": 3,
    "gross_m2": 120,
    "building_age": 5,
    "floor": 3,
    "num_floors": 5,
    "bathrooms": 1,
    "asking_price": 2000000  # Bu önemli! Karşılaştırma için gerekli
}

def test_api():
    try:
        print("🔄 API'ye bağlanılıyor...")
        response = requests.post(API_URL, json=test_data, timeout=10)

        if response.status_code == 200:
            result = response.json()

            print("✅ BAŞARILI! API yanıt verdi.")
            print("\n" + "="*50)
            print("🎯 TAHMİN SONUCU:")
            print(f"Tahmin edilen fiyat: {result['prediction']['predicted_price_formatted']}")

            if "comparison" in result and result["comparison"]:
                comp = result["comparison"]
                print("\n📊 KARŞILAŞTIRMA SONUCU:")
                print(f"Verdict: {comp['verdict']}")
                print(f"Benzer ev sayısı: {comp['similar_properties_count']}")

                print("\n💰 DATASET FİYATLARI:")
                if "dataset_prices" in comp:
                    print(f"✅ Dataset fiyatları BULUNDU! ({len(comp['dataset_prices'])} adet)")
                    print(f"En düşük fiyat: {comp['dataset_price_min']:,.0f} TL")
                    print(f"En yüksek fiyat: {comp['dataset_price_max']:,.0f} TL")
                    print(f"Ortanca fiyat: {comp['dataset_price_median']:,.0f} TL")

                    print("\nİlk 10 dataset fiyatı:")
                    for i, price in enumerate(comp['dataset_prices'][:10], 1):
                        print(f"  {i}. {price:,.0f} TL")
                else:
                    print("❌ Dataset fiyatları BULUNAMADI!")
                    print("Mevcut anahtarlar:", list(comp.keys()))
            else:
                print("❌ Karşılaştırma verisi yok!")
                print("Not: asking_price parametresi gereklidir.")

        else:
            print(f"❌ API hatası: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ Bağlantı hatası!")
        print("API'nin çalıştığından emin olun:")
        print("cd /Users/alikemal/Documents/GitHub/hachathon/production")
        print("PYTHONPATH=/Users/alikemal/Documents/GitHub/hachathon/production python3 -c \"from api import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8002)\"")

    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")

if __name__ == "__main__":
    print("🏠 Emlak API Dataset Fiyat Testi")
    print("="*50)
    test_api()