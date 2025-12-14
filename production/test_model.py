#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODEL TEST SCRIPT - Test verisi ile model performansını değerlendirme
======================================================================
Kullanım: python3 test_model.py <test_csv_path>
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_model():
    """Modeli yükle"""
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    base_models = model_data["base_models"]
    meta_model = model_data["meta_model"]
    feature_columns = model_data["feature_columns"]
    district_enc = model_data["district_encoding"]
    neighborhood_enc = model_data["neighborhood_encoding"]
    global_mean = model_data["global_mean"]

    return base_models, meta_model, feature_columns, district_enc, neighborhood_enc, global_mean

def preprocess_test_data(df):
    """Test verisini preprocess et"""
    print("🧹 Test verisi preprocessing...")

    # Price temizleme
    df['Price'] = df['Price'].str.replace(' TL', '').str.replace('.', '').astype(float)

    # Net m2
    df['Net_m2'] = pd.to_numeric(df['m² (Net)'], errors='coerce')

    # Gross m2 (opsiyonel)
    if 'm² (Gross)' in df.columns:
        df['Gross_m2'] = pd.to_numeric(df['m² (Gross)'], errors='coerce')
    else:
        df['Gross_m2'] = df['Net_m2'] * 1.15

    # Building Age mapping
    age_map = {
        '0': 0, '1-5 between': 3, '6-10 between': 8, '11-15 between': 13,
        '16-20 between': 18, '21-25 between': 23, '26-30 between': 28,
        '31  and more than': 35, '5-10 between': 7.5
    }
    if 'Building Age' in df.columns:
        df['Building_Age'] = df['Building Age'].replace(age_map)
        df['Building_Age'] = pd.to_numeric(df['Building_Age'], errors='coerce').fillna(10)
    else:
        df['Building_Age'] = 10

    # Rooms mapping
    room_map = {
        '1+0': 1, '1+1': 2, '2+0': 2, '2+1': 3, '2+2': 4,
        '3+1': 4, '3+2': 5, '4+1': 5, '4+2': 6, '4+3': 7,
        '5+1': 6, '5+2': 7, '5+3': 8, '5+4': 9,
        '6+1': 7, '6+2': 8, '6+3': 9, '7+1': 8, '7+2': 9,
        '8+1': 9, '8+2': 10, '8+3': 11, '8+4': 12,
        '9+1': 10, '9+2': 11, '10 and more': 12
    }
    if 'Number of rooms' in df.columns:
        df['Rooms'] = df['Number of rooms'].map(room_map)
    else:
        df['Rooms'] = 2  # default

    # Floor mapping
    floor_map = {
        'Ground floor': 0, 'Kot 1': -1, 'Kot 2': -2, 'Kot 3': -3,
        'High entrance': 1, 'Entrance floor': 0, 'Mezzanine': 0.5,
        'Basement': -1, 'Middle floor': 3, 'Top floor': 10
    }
    if 'Floor location' in df.columns:
        df['Floor'] = df['Floor location'].replace(floor_map)
        df['Floor'] = pd.to_numeric(df['Floor'], errors='coerce').fillna(2)
    else:
        df['Floor'] = 2

    # Number of floors
    if 'Number of floors' in df.columns:
        df['Num_Floors'] = pd.to_numeric(df['Number of floors'], errors='coerce').fillna(5)
    else:
        df['Num_Floors'] = 5

    # Bathrooms
    if 'Number of bathrooms' in df.columns:
        df['Bathrooms'] = pd.to_numeric(df['Number of bathrooms'], errors='coerce').fillna(1)
    else:
        df['Bathrooms'] = 1

    # Available for Loan
    if 'Available for Loan' in df.columns:
        def _avail_map(x):
            if pd.isna(x):
                return 0
            s = str(x).strip().lower()
            if s in ['1', '1.0', 'true', 'yes', 'y', 'evet', 'var', 'available']:
                return 1
            if s in ['0', '0.0', 'false', 'no', 'n', 'hayir', 'yok', 'not available', 'not_available']:
                return 0
            try:
                v = float(s)
                return 1 if v > 0 else 0
            except Exception:
                return 0

        df['Available_for_Loan'] = df['Available for Loan'].apply(_avail_map)
    else:
        df['Available_for_Loan'] = 0

    # Heating
    heating_candidates = ['Heating', 'Heating System', 'Heating Type', 'Isıtma', 'Isıtma Sistemi']
    heating_col = None
    for c in heating_candidates:
        if c in df.columns:
            heating_col = c
            break

    if heating_col:
        df['Heating_raw'] = df[heating_col].fillna('').astype(str).str.lower()
    else:
        df['Heating_raw'] = ''

    # Temizleme
    df = df.dropna(subset=['Price', 'Net_m2', 'Rooms', 'District'])
    df = df[(df['Price'] > 100000) & (df['Price'] < 10000000)]
    df = df[(df['Net_m2'] > 20) & (df['Net_m2'] < 600)]

    print(f"✅ Temizlenmiş test verisi: {len(df)} satır")
    return df

def engineer_features(df, district_enc, neighborhood_enc, global_mean):
    """Feature engineering"""
    # Target encodings
    df['District_enc'] = df['District'].map(district_enc).fillna(global_mean)
    if 'Neighborhood' in df.columns and neighborhood_enc:
        df['Neigh_enc'] = df['Neighborhood'].map(neighborhood_enc).fillna(global_mean)
    else:
        df['Neigh_enc'] = global_mean

    # Log transforms
    df['Log_m2'] = np.log1p(df['Net_m2'])
    df['Log_District'] = np.log1p(df['District_enc'])

    # Area polynomial
    df['m2_sq'] = df['Net_m2'] ** 2 / 10000

    # Key ratios
    df['m2_per_room'] = df['Net_m2'] / (df['Rooms'] + 0.1)
    df['Floor_ratio'] = df['Floor'] / (df['Num_Floors'] + 0.1)

    # Key interactions
    df['District_x_m2'] = df['District_enc'] * df['Net_m2'] / 1000000
    df['Age_x_m2'] = df['Building_Age'] * df['Net_m2'] / 1000

    # Age inverse
    df['Age_inv'] = 1 / (df['Building_Age'] + 1)

    # Luxury indicator
    luxury = ['Beşiktaş', 'Sarıyer', 'Kadıköy', 'Üsküdar', 'Şişli', 'Bakırköy']
    df['Is_Luxury'] = df['District'].isin(luxury).astype(int)
    df['Luxury_m2'] = df['Is_Luxury'] * df['Net_m2']

    # Budget indicator
    budget = ['Esenyurt', 'Bağcılar', 'Sultangazi', 'Esenler', 'Arnavutköy']
    df['Is_Budget'] = df['District'].isin(budget).astype(int)

    # New building
    df['Is_New'] = (df['Building_Age'] <= 5).astype(int)

    # Expected price
    df['Expected'] = df['District_enc'] * df['Net_m2'] / 100

    # Heating flags
    if 'Heating_raw' in df.columns:
        df['Heating_Natural_Gas'] = df['Heating_raw'].str.contains('doğalgaz|dogalgaz|natural|gas', regex=True).fillna(False).astype(int)
        df['Heating_Central'] = df['Heating_raw'].str.contains('merkezi|central|district', regex=True).fillna(False).astype(int)
        df['Heating_Electric'] = df['Heating_raw'].str.contains('elektrik|electric', regex=True).fillna(False).astype(int)
        df['Heating_Stove'] = df['Heating_raw'].str.contains('soba|stove|wood', regex=True).fillna(False).astype(int)
    else:
        df['Heating_Natural_Gas'] = 0
        df['Heating_Central'] = 0
        df['Heating_Electric'] = 0
        df['Heating_Stove'] = 0

    return df

def predict_single(base_models, meta_model, features, feature_columns):
    """Tek bir örnek için tahmin yap"""
    # Feature array oluştur
    feature_array = np.array([[features[col] for col in feature_columns]])

    # Base modellerden tahmin al
    base_predictions = []
    for model in base_models.values():
        pred = model.predict(feature_array)[0]
        base_predictions.append(pred)

    base_predictions = np.array(base_predictions).reshape(1, -1)

    # Meta model ile final tahmin
    final_pred = meta_model.predict(base_predictions)[0]

    return np.expm1(final_pred)  # Log'dan geri çevir

def main():
    if len(sys.argv) != 2:
        print("Kullanım: python3 test_model.py <test_csv_path>")
        print("Örnek: python3 test_model.py ../data/test_verisi.csv")
        sys.exit(1)

    test_csv_path = sys.argv[1]

    if not os.path.exists(test_csv_path):
        print(f"❌ Test dosyası bulunamadı: {test_csv_path}")
        sys.exit(1)

    print("=" * 80)
    print("🧪 MODEL TESTİ BAŞLATILIYOR")
    print("=" * 80)
    print(f"📂 Test dosyası: {test_csv_path}")

    try:
        # Modeli yükle
        print("\n📂 Model yükleniyor...")
        base_models, meta_model, feature_columns, district_enc, neighborhood_enc, global_mean = load_model()
        print("✅ Model yüklendi")

        # Test verisini yükle
        print("\n📂 Test verisi yükleniyor...")
        df_test = pd.read_csv(test_csv_path, sep=';', encoding='utf-8')
        print(f"✅ Ham test verisi: {len(df_test)} satır")

        # Preprocess
        df_test = preprocess_test_data(df_test)

        # Feature engineering
        print("\n🔧 Feature engineering...")
        df_test = engineer_features(df_test, district_enc, neighborhood_enc, global_mean)

        # NaN değerleri doldur
        df_test[feature_columns] = df_test[feature_columns].fillna(0)

        # Tahminler
        print("\n🎯 Tahminler yapılıyor...")
        predictions = []
        actual_prices = []

        for idx, row in df_test.iterrows():
            try:
                pred = predict_single(base_models, meta_model, row, feature_columns)
                predictions.append(pred)
                actual_prices.append(row['Price'])

                if (idx + 1) % 100 == 0:
                    print(f"  {idx + 1}/{len(df_test)} tamamlandı...")

            except Exception as e:
                print(f"❌ Satır {idx} işlenirken hata: {e}")
                continue

        predictions = np.array(predictions)
        actual_prices = np.array(actual_prices)

        print(f"✅ {len(predictions)} tahmin tamamlandı")

        # Metrikleri hesapla
        print("\n📊 METRİKLER")
        print("-" * 40)

        mae = mean_absolute_error(actual_prices, predictions)
        rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
        mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
        r2 = r2_score(actual_prices, predictions)

        print(f"MAE (Ortalama Mutlak Hata): ₺{mae:,.0f}")
        print(f"RMSE (Kök Ortalama Kare Hata): ₺{rmse:,.0f}")
        print(f"MAPE (Ortalama Yüzde Hata): %{mape:.2f}")
        print(f"R² Skoru: {r2:.4f}")
        # Detaylı sonuçlar
        print("\n📋 DETAYLI SONUÇLAR")
        print("-" * 40)

        errors = predictions - actual_prices
        print(f"En Küçük Hata: ₺{np.min(np.abs(errors)):,.0f}")
        print(f"En Büyük Hata: ₺{np.max(np.abs(errors)):,.0f}")
        print(f"Ortalama Hata: ₺{np.mean(errors):,.0f}")
        print(f"Medyan Mutlak Hata: ₺{np.median(np.abs(errors)):,.0f}")
        # İlk 10 sonucu göster
        print("\n📋 İLK 10 TAHMİN")
        print("-" * 80)
        print(f"{'#':<3} {'Gerçek':<10} {'Tahmin':<10} {'Hata':<10} {'%|Hata':<8}")
        print("-" * 80)

        for i in range(min(10, len(predictions))):
            error = predictions[i] - actual_prices[i]
            pct_error = (error / actual_prices[i]) * 100
            print(f"{i+1:<3} {actual_prices[i]:<10,.0f} {predictions[i]:<10,.0f} {error:<10,.0f} {pct_error:<8.1f}")

        # CSV olarak kaydet
        results_df = pd.DataFrame({
            'Gercek_Fiyat': actual_prices,
            'Tahmin': predictions,
            'Hata': errors,
            'Mutlak_Hata': np.abs(errors)
        })

        output_path = os.path.splitext(test_csv_path)[0] + '_sonuclar.csv'
        results_df.to_csv(output_path, index=False, sep=';')
        print(f"\n💾 Detaylı sonuçlar kaydedildi: {output_path}")

        print("\n" + "=" * 80)
        print("✅ MODEL TESTİ TAMAMLANDI")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()