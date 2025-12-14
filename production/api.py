#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production API - Ev Fiyat Tahmini ve Karsilastirma
==================================================
V6 Model ile fiyat tahmini yapar ve benzer evlerle karsilastirarak
iyi/orta/kotu tercih analizi sunar.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import pickle
import os
import uvicorn

# =============================================================================
# APP SETUP
# =============================================================================
app = FastAPI(
    title="Ev Fiyat Tahmini API",
    description="V6 Model - 22 ozellik ile fiyat tahmini ve tercih analizi",
    version="2.0.0"
)

# Tüm CORS'lara izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# LOAD MODEL
# =============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hackathon_train_set.csv")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

base_models = model_data["base_models"]
meta_model = model_data["meta_model"]
feature_columns = model_data["feature_columns"]
district_enc = model_data["district_encoding"]
neighborhood_enc = model_data["neighborhood_encoding"]
global_mean = model_data["global_mean"]

LUXURY_DISTRICTS = ['Beşiktaş', 'Sarıyer', 'Kadıköy', 'Üsküdar', 'Şişli', 'Bakırköy']
BUDGET_DISTRICTS = ['Esenyurt', 'Bağcılar', 'Sultangazi', 'Esenler', 'Arnavutköy']

# Dataset yukle (karsilastirma icin)
try:
    df_compare = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')
    df_compare['Price'] = df_compare['Price'].str.replace(' TL', '').str.replace('.', '').astype(float)
    df_compare['Net_m2'] = pd.to_numeric(df_compare['m² (Net)'], errors='coerce')
    
    room_map = {
        '1+0': 1, '1+1': 2, '2+0': 2, '2+1': 3, '2+2': 4,
        '3+1': 4, '3+2': 5, '4+1': 5, '4+2': 6, '4+3': 7,
        '5+1': 6, '5+2': 7, '5+3': 8, '5+4': 9,
        '6+1': 7, '6+2': 8, '6+3': 9, '7+1': 8, '7+2': 9,
        '8+1': 9, '8+2': 10, '8+3': 11, '8+4': 12,
        '9+1': 10, '9+2': 11, '10 and more': 12
    }
    df_compare['Rooms'] = df_compare['Number of rooms'].map(room_map)
    
    age_map = {
        '0': 0, '1-5 between': 3, '6-10 between': 8, '11-15 between': 13,
        '16-20 between': 18, '21-25 between': 23, '26-30 between': 28,
        '31  and more than': 35, '5-10 between': 7.5
    }
    df_compare['Building_Age'] = df_compare['Building Age'].replace(age_map)
    df_compare['Building_Age'] = pd.to_numeric(df_compare['Building_Age'], errors='coerce').fillna(10)
    
    df_compare = df_compare.dropna(subset=['Price', 'Net_m2', 'Rooms', 'District'])
    df_compare = df_compare[(df_compare['Price'] > 100000) & (df_compare['Price'] < 10000000)]
    df_compare = df_compare[(df_compare['Net_m2'] > 20) & (df_compare['Net_m2'] < 600)]
    
    HAS_COMPARE_DATA = True
    print(f"Karsilastirma verisi yuklendi: {len(df_compare)} kayit")
except Exception as e:
    HAS_COMPARE_DATA = False
    df_compare = None
    print(f"Karsilastirma verisi yuklenemedi: {e}")

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================
class PropertyInput(BaseModel):
    # ZORUNLU PARAMETRELER
    district: str = Field(..., description="🏠 ZORUNLU: İlçe adı (örn: Kadıköy, Beşiktaş, Esenyurt)")
    net_m2: float = Field(..., gt=20, lt=600, description="🏠 ZORUNLU: Net metrekare (20-600 arası)")
    rooms: int = Field(..., ge=1, le=12, description="🏠 ZORUNLU: Oda sayısı (1+1=2, 2+1=3, 3+1=4, 4+1=5 şeklinde)")

    # İSTEĞE BAĞLI PARAMETRELER (varsayılan değerlerle çalışır)
    neighborhood: Optional[str] = Field(None, description="📍 İSTEĞE BAĞLI: Mahalle adı (örn: Caferağa Mh., Ulus Mh.)")
    gross_m2: Optional[float] = Field(None, description="📐 İSTEĞE BAĞLI: Brüt metrekare (varsayılan: net_m2 × 1.15)")
    building_age: Optional[float] = Field(10, ge=0, le=50, description="🏢 İSTEĞE BAĞLI: Bina yaşı (varsayılan: 10 yıl)")
    floor: Optional[int] = Field(2, ge=-3, le=30, description="🏠 İSTEĞE BAĞLI: Bulunduğu kat (-3: bodrum, 0: zemin, varsayılan: 2)")
    num_floors: Optional[int] = Field(5, ge=1, le=50, description="🏢 İSTEĞE BAĞLI: Bina toplam kat sayısı (varsayılan: 5)")
    bathrooms: Optional[int] = Field(1, ge=1, le=5, description="🚿 İSTEĞE BAĞLI: Banyo sayısı (varsayılan: 1)")

    # KARŞILAŞTIRMA İÇİN
    asking_price: Optional[float] = Field(None, description="💰 İSTEĞE BAĞLI: İstenen fiyat (TL) - piyasa karşılaştırması için")
    purpose: Optional[str] = Field(None, description="Kullanım amacı: 'oturum' veya 'yatırım' (isteğe bağlı)")

class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    price_range_low: float
    price_range_high: float
    confidence: str

class ComparisonResult(BaseModel):
    verdict: str
    verdict_emoji: str
    verdict_description: str
    background_color: str  # Kart arka plan rengi
    asking_price: float
    predicted_price: float
    difference_percent: float
    similar_properties_count: int
    similar_avg_price: float
    percentile: float
    # Dataset gerçek fiyatları
    dataset_prices: List[float]  # Tüm benzer evlerin fiyatları
    dataset_price_min: float     # En düşük fiyat
    dataset_price_max: float     # En yüksek fiyat
    dataset_price_median: float  # Ortanca fiyat

class FullResponse(BaseModel):
    prediction: PredictionResponse
    comparison: Optional[ComparisonResult] = None
    input_features: dict

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def engineer_features(data: dict) -> np.ndarray:
    district = data['district']
    neighborhood = data.get('neighborhood', '')
    net_m2 = data['net_m2']
    gross_m2 = data.get('gross_m2') or net_m2 * 1.15
    rooms = data['rooms']
    building_age = data.get('building_age', 10)
    floor = data.get('floor', 2)
    num_floors = data.get('num_floors', 5)
    bathrooms = data.get('bathrooms', 1)
    
    district_value = district_enc.get(district, global_mean)
    neigh_value = neighborhood_enc.get(neighborhood, global_mean) if neighborhood else global_mean
    
    # Sıfır veya negatif değer kontrolü
    district_value = max(district_value, 1)  # Minimum 1
    neigh_value = max(neigh_value, 1)  # Minimum 1
    net_m2 = max(net_m2, 1)  # Minimum 1
    
    log_m2 = np.log1p(net_m2)
    log_district = np.log1p(district_value)
    m2_sq = (net_m2 ** 2) / 10000
    m2_per_room = net_m2 / (rooms + 0.1)
    floor_ratio = floor / (num_floors + 0.1)
    district_x_m2 = district_value * net_m2 / 1000000
    age_x_m2 = building_age * net_m2 / 1000
    age_inv = 1 / (building_age + 1)
    
    is_luxury = 1 if district in LUXURY_DISTRICTS else 0
    luxury_m2 = is_luxury * net_m2
    is_budget = 1 if district in BUDGET_DISTRICTS else 0
    is_new = 1 if building_age <= 5 else 0
    expected = district_value * net_m2 / 100
    
    features = {
        'Net_m2': net_m2,
        'Rooms': rooms,
        'Building_Age': building_age,
        'Floor': floor,
        'Num_Floors': num_floors,
        'Bathrooms': bathrooms,
        'Gross_m2': gross_m2,
        'District_enc': district_value,
        'Neigh_enc': neigh_value,
        'Log_District': log_district,
        'Log_m2': log_m2,
        'm2_sq': m2_sq,
        'm2_per_room': m2_per_room,
        'Floor_ratio': floor_ratio,
        'District_x_m2': district_x_m2,
        'Age_x_m2': age_x_m2,
        'Age_inv': age_inv,
        'Is_Luxury': is_luxury,
        'Luxury_m2': luxury_m2,
        'Is_Budget': is_budget,
        'Is_New': is_new,
        'Expected': expected
    }
    
    feature_array = [features.get(col, 0) for col in feature_columns]
    return np.array(feature_array).reshape(1, -1)

def predict_price(features: np.ndarray) -> tuple:
    base_preds = np.zeros((1, len(base_models)))
    for i, (name, model) in enumerate(base_models.items()):
        base_preds[0, i] = model.predict(features)[0]
    
    log_pred = meta_model.predict(base_preds)[0]
    
    # NaN/inf kontrolü
    if not np.isfinite(log_pred):
        log_pred = np.log1p(1000000)  # Varsayılan 1M TL için log değeri
    
    predicted_price = np.expm1(log_pred)
    
    # Tekrar NaN/inf kontrolü
    if not np.isfinite(predicted_price) or predicted_price <= 0:
        predicted_price = 1000000  # Varsayılan 1M TL
    
    price_low = predicted_price * 0.88
    price_high = predicted_price * 1.12
    
    return predicted_price, price_low, price_high

def compare_with_dataset(district: str, net_m2: float, rooms: int, building_age: float, asking_price: float, predicted_price: float = None, purpose: str = None) -> dict:
    if not HAS_COMPARE_DATA:
        return None
    
    m2_tol = 0.20
    age_tol = 10
    room_tol = 1
    
    similar = df_compare[
        (df_compare['District'] == district) &
        (df_compare['Net_m2'] >= net_m2 * (1 - m2_tol)) &
        (df_compare['Net_m2'] <= net_m2 * (1 + m2_tol)) &
        (df_compare['Rooms'] >= rooms - room_tol) &
        (df_compare['Rooms'] <= rooms + room_tol) &
        (df_compare['Building_Age'] >= building_age - age_tol) &
        (df_compare['Building_Age'] <= building_age + age_tol)
    ]
    
    if len(similar) < 5:
        similar = df_compare[
            (df_compare['District'] == district) &
            (df_compare['Net_m2'] >= net_m2 * (1 - m2_tol * 1.5)) &
            (df_compare['Net_m2'] <= net_m2 * (1 + m2_tol * 1.5))
        ]
    
    if len(similar) < 3:
        similar = df_compare[df_compare['District'] == district]
    
    if len(similar) == 0:
        return None
    
    avg_price = similar['Price'].mean()
    prices = similar['Price'].values
    percentile = (prices < asking_price).sum() / len(prices) * 100
    
    def format_verdict(verdict_code: str) -> str:
        """Verdict kodunu güzel yazıya çevir"""
        parts = verdict_code.split('_')
        return ' '.join(word.capitalize() for word in parts)
    
    def get_background_color(verdict_code: str) -> str:
        """Verdict'e göre kart arka plan rengini belirle"""
        if verdict_code.startswith("COK_IYI") or verdict_code.startswith("IYI"):
            return "#22c55e"  # Yeşil
        elif verdict_code.startswith("ORTA"):
            return "#eab308"  # Sarı
        elif verdict_code.startswith("KOTU"):
            return "#ef4444"  # Kırmızı
        else:
            return "#6b7280"  # Gri (varsayılan)
    
    # Model tahmini ile karşılaştırma
    if predicted_price is not None:
        diff_from_pred = ((asking_price - predicted_price) / predicted_price) * 100
        # Yatırım için ek analiz (örnek: amortisman)
        annual_rent = predicted_price * 0.04  # Yıllık kira tahmini (örnek)
        amortisman = asking_price / (annual_rent if annual_rent > 0 else 1)
        # 4 seviye karar mantığı
        if purpose == "yatırım":
            if diff_from_pred <= -20:
                verdict = "COK_IYI_YATIRIM"
                verdict_emoji = "💰"
                verdict_description = f"ÇOK İYİ YATIRIM: Fiyat model tahmininin %{abs(diff_from_pred):.1f} altında, amortisman {amortisman:.1f} yıl."
            elif -20 < diff_from_pred <= -10:
                verdict = "IYI_YATIRIM"
                verdict_emoji = "💸"
                verdict_description = f"İYİ YATIRIM: Fiyat model tahmininin %{abs(diff_from_pred):.1f} altında, amortisman {amortisman:.1f} yıl."
            elif -10 < diff_from_pred <= 10:
                verdict = "ORTA_YATIRIM"
                verdict_emoji = "🟡"
                verdict_description = f"ORTA: Fiyat model tahminine yakın (%{diff_from_pred:.1f}), amortisman {amortisman:.1f} yıl."
            else:
                verdict = "KOTU_YATIRIM"
                verdict_emoji = "⏳"
                verdict_description = f"KÖTÜ YATIRIM: Fiyat model tahmininin %{diff_from_pred:.1f} üstünde, amortisman {amortisman:.1f} yıl."
        elif purpose == "oturum":
            if diff_from_pred <= -20:
                verdict = "COK_IYI_OTURUM"
                verdict_emoji = "🌟"
                verdict_description = f"ÇOK İYİ FIRSAT (Oturum): Fiyat model tahmininin %{abs(diff_from_pred):.1f} altında."
            elif -20 < diff_from_pred <= -10:
                verdict = "IYI_OTURUM"
                verdict_emoji = "🏠"
                verdict_description = f"İYİ FIRSAT (Oturum): Fiyat model tahmininin %{abs(diff_from_pred):.1f} altında."
            elif -10 < diff_from_pred <= 10:
                verdict = "ORTA_OTURUM"
                verdict_emoji = "🟡"
                verdict_description = f"ORTA: Fiyat model tahminine yakın (%{diff_from_pred:.1f})."
            else:
                verdict = "KOTU_OTURUM"
                verdict_emoji = "🚫"
                verdict_description = f"KÖTÜ: Fiyat model tahmininin %{diff_from_pred:.1f} üstünde."
        else:
            if diff_from_pred <= -20:
                verdict = "COK_IYI_TERCIH"
                verdict_emoji = "🌟"
                verdict_description = f"ÇOK İYİ FIRSAT: Fiyat model tahmininin %{abs(diff_from_pred):.1f} altında."
            elif -20 < diff_from_pred <= -10:
                verdict = "IYI_TERCIH"
                verdict_emoji = "🟢"
                verdict_description = f"İYİ FIRSAT: Fiyat model tahmininin %{abs(diff_from_pred):.1f} altında."
            elif -10 < diff_from_pred <= 10:
                verdict = "ORTA_TERCIH"
                verdict_emoji = "🟡"
                verdict_description = f"ORTA: Fiyat model tahminine yakın (%{diff_from_pred:.1f})."
            else:
                verdict = "KOTU_TERCIH"
                verdict_emoji = "🔴"
                verdict_description = f"KÖTÜ: Fiyat model tahmininin %{diff_from_pred:.1f} üstünde."
        diff_percent = diff_from_pred
    else:
        diff_from_avg = ((asking_price - avg_price) / avg_price) * 100
        if asking_price <= avg_price * 0.90:
            verdict = "IYI_TERCIH"
            verdict_emoji = "🟢"
            verdict_description = f"Bu ev piyasa ortalamasının %{abs(diff_from_avg):.1f} altında! Çok iyi fırsat."
        elif asking_price <= avg_price * 1.05:
            verdict = "ORTA_TERCIH"
            verdict_emoji = "🟡"
            verdict_description = f"Bu ev piyasa ortalamasına yakın. Makul bir fiyat."
        elif asking_price <= avg_price * 1.15:
            verdict = "ORTA_TERCIH"
            verdict_emoji = "🟠"
            verdict_description = f"Bu ev piyasa ortalamasının %{diff_from_avg:.1f} üstünde. Pazarlık yapılabilir."
        else:
            verdict = "KOTU_TERCIH"
            verdict_emoji = "🔴"
            verdict_description = f"Bu ev piyasa ortalamasının %{diff_from_avg:.1f} üstünde. Pahalı görünüyor."
        diff_percent = diff_from_avg
    
    return {
        "verdict": format_verdict(verdict),
        "verdict_emoji": verdict_emoji,
        "verdict_description": verdict_description,
        "background_color": get_background_color(verdict),
        "similar_count": len(similar),
        "avg_price": avg_price,
        "percentile": percentile,
        "diff_percent": diff_percent,
        "dataset_prices": prices.tolist(),  # Tüm fiyatlar
        "dataset_price_min": float(prices.min()),
        "dataset_price_max": float(prices.max()),
        "dataset_price_median": float(np.median(prices))
    }

# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/")
async def root():
    return {
        "status": "online",
        "model": "V6 Minimal Features",
        "features": len(feature_columns),
        "base_models": list(base_models.keys()),
        "compare_data_available": HAS_COMPARE_DATA,
        "endpoints": {
            "/predict": "POST - Fiyat tahmini",
            "/quick-check": "POST - Hızlı fiyat kontrolü",
            "/dashboard": "POST - İlçe istatistikleri",
            "/trends": "POST - Fiyat trendleri",
            "/train-with-new-data": "POST - Model eğitimi",
            "/model-metrics": "GET - Model metrikleri ve özellikleri",
            "/districts": "GET - Mevcut ilceler",
            "/health": "GET - Saglik kontrolu"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.get("/districts")
async def get_districts():
    districts = list(district_enc.keys())
    return {
        "count": len(districts),
        "luxury": [d for d in LUXURY_DISTRICTS if d in districts],
        "budget": [d for d in BUDGET_DISTRICTS if d in districts],
        "all": sorted(districts)
    }

@app.post("/predict", response_model=FullResponse)
async def predict(prop: PropertyInput):
    if prop.district not in district_enc:
        available = list(district_enc.keys())[:10]
        raise HTTPException(
            status_code=400,
            detail=f"Bilinmeyen ilce: {prop.district}. Mevcut ilceler: {available}..."
        )
    
    input_data = {
        'district': prop.district,
        'neighborhood': prop.neighborhood,
        'net_m2': prop.net_m2,
        'gross_m2': prop.gross_m2,
        'rooms': prop.rooms,
        'building_age': prop.building_age,
        'floor': prop.floor,
        'num_floors': prop.num_floors,
        'bathrooms': prop.bathrooms
    }
    
    features = engineer_features(input_data)
    predicted_price, price_low, price_high = predict_price(features)
    
    price_range = (price_high - price_low) / predicted_price * 100
    if price_range < 20:
        confidence = "Yuksek"
    elif price_range < 30:
        confidence = "Orta"
    else:
        confidence = "Dusuk"
    
    prediction = PredictionResponse(
        predicted_price=round(predicted_price, 0),
        predicted_price_formatted=f"{predicted_price:,.0f} TL".replace(",", "."),
        price_range_low=round(price_low, 0),
        price_range_high=round(price_high, 0),
        confidence=confidence
    )
    
    comparison = None
    if prop.asking_price:
        comp_result = compare_with_dataset(
            district=prop.district,
            net_m2=prop.net_m2,
            rooms=prop.rooms,
            building_age=prop.building_age,
            asking_price=prop.asking_price,
            predicted_price=predicted_price,
            purpose=prop.purpose
        )
        if comp_result:
            comparison = ComparisonResult(
                verdict=comp_result["verdict"],
                verdict_emoji=comp_result["verdict_emoji"],
                verdict_description=comp_result["verdict_description"],
                background_color=comp_result["background_color"],
                asking_price=prop.asking_price,
                predicted_price=round(predicted_price, 0),
                difference_percent=round(comp_result["diff_percent"], 1),
                similar_properties_count=comp_result["similar_count"],
                similar_avg_price=round(comp_result["avg_price"], 0),
                percentile=round(comp_result["percentile"], 1),
                dataset_prices=comp_result["dataset_prices"],
                dataset_price_min=comp_result["dataset_price_min"],
                dataset_price_max=comp_result["dataset_price_max"],
                dataset_price_median=comp_result["dataset_price_median"]
            )
    return FullResponse(
        prediction=prediction,
        comparison=comparison,
        input_features=input_data
    )

@app.post("/quick-check")
async def quick_check(
    district: str,
    net_m2: float,
    rooms: int,
    asking_price: float,
    building_age: float = 10,
    purpose: Optional[str] = None
):
    if district not in district_enc:
        raise HTTPException(status_code=400, detail=f"Bilinmeyen ilce: {district}")
    
    input_data = {
        'district': district,
        'neighborhood': None,
        'net_m2': net_m2,
        'gross_m2': net_m2 * 1.15,
        'rooms': rooms,
        'building_age': building_age,
        'floor': 2,
        'num_floors': 5,
        'bathrooms': 1
    }
    
    features = engineer_features(input_data)
    predicted_price, _, _ = predict_price(features)
    
    comp = compare_with_dataset(district, net_m2, rooms, building_age, asking_price, purpose=purpose)
    
    result = {
        "predicted_price": f"{predicted_price:,.0f} TL".replace(",", "."),
        "asking_price": f"{asking_price:,.0f} TL".replace(",", "."),
        "difference": f"{((asking_price - predicted_price) / predicted_price * 100):+.1f}%"
    }
    
    if comp:
        result["verdict"] = f"{comp['verdict_emoji']} {comp['verdict']}"
        result["analysis"] = comp["verdict_description"]
        result["similar_count"] = comp["similar_count"]
    
    return result

# =====================
# DASHBOARD & TRENDS ENDPOINTLERİ
# =====================

from fastapi import Body

class DashboardStats(BaseModel):
    avgPrice: float
    medianPrice: float
    priceChange: float
    listings: int
    predictedPrice: float
    percentile: float
    minPrice: float
    maxPrice: float
    avgRooms: float
    avgArea: float
    totalListings: int
    activeListings: int

class PriceDistribution(BaseModel):
    q1: float
    q2: float
    q3: float

class DashboardResponse(BaseModel):
    district: str
    stats: DashboardStats
    priceDistribution: PriceDistribution

class PriceHistoryItem(BaseModel):
    date: str
    price: float

class CurrentStats(BaseModel):
    avgPrice: float
    medianPrice: float
    minPrice: float
    maxPrice: float
    listings: int

class TrendInfo(BaseModel):
    trend: float
    priceHistory: list[PriceHistoryItem]
    currentStats: CurrentStats

class TrendsResponse(BaseModel):
    district: str
    trendInfo: TrendInfo

@app.post("/dashboard", response_model=DashboardResponse)
async def dashboard(data: dict = Body(...)):
    district = data.get("district")
    if not district:
        raise HTTPException(status_code=400, detail="district parametresi zorunlu.")
    
    # District kontrolü
    if district not in district_enc:
        raise HTTPException(status_code=400, detail=f"Bilinmeyen ilce: {district}")
    
    df = df_compare[df_compare['District'] == district]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"{district} için veri bulunamadı.")
    prices = df['Price']
    avg_price = float(np.mean(prices))
    median_price = float(np.median(prices))
    min_price = float(np.min(prices))
    max_price = float(np.max(prices))
    q1 = float(np.percentile(prices, 25))
    q2 = float(np.percentile(prices, 50))
    q3 = float(np.percentile(prices, 75))
    avg_rooms = float(np.mean(df['Rooms'].dropna()))
    avg_area = float(np.mean(df['Net_m2'].dropna()))
    total_listings = int(len(df_compare))
    active_listings = int(len(df))
    listings = active_listings
    if len(df) > 50 and 'Adrtisement Date' in df.columns and not df['Adrtisement Date'].isnull().all():
        df_temp = df.copy()
        df_temp.loc[:, 'Adrtisement Date'] = pd.to_datetime(df_temp['Adrtisement Date'], format='%d/%m/%Y', errors='coerce')
        df_temp = df_temp.dropna(subset=['Adrtisement Date'])
        
        if len(df_temp) > 10:  # Yeterli veri varsa
            last_prices = df_temp.sort_values(by='Adrtisement Date', ascending=False)['Price'].head(50)
            last_mean = last_prices.mean()
            if not np.isnan(last_mean) and not np.isinf(last_mean):
                price_change = ((last_mean - avg_price) / avg_price) * 100
            else:
                price_change = 0.0
        else:
            price_change = 0.0
    else:
        price_change = 0.0
    sample = {
        'district': district,
        'neighborhood': df['Neighborhood'].mode().iloc[0] if not df['Neighborhood'].isnull().all() else None,
        'net_m2': avg_area,
        'gross_m2': avg_area * 1.15,
        'rooms': int(round(avg_rooms)),
        'building_age': float(df['Building_Age'].mean()),
        'floor': 2,
        'num_floors': 5,
        'bathrooms': 1
    }
    features = engineer_features(sample)
    predicted_price, _, _ = predict_price(features)
    
    # NaN/inf kontrolü
    if not np.isfinite(predicted_price):
        predicted_price = avg_price  # Varsayılan olarak ortalama fiyat kullan
    
    percentile = float((prices < predicted_price).sum() / len(prices) * 100)
    stats = DashboardStats(
        avgPrice=round(avg_price, 2),
        medianPrice=round(median_price, 2),
        priceChange=round(price_change, 2),
        listings=listings,
        predictedPrice=round(predicted_price, 2),
        percentile=round(percentile, 2),
        minPrice=round(min_price, 2),
        maxPrice=round(max_price, 2),
        avgRooms=round(avg_rooms, 2),
        avgArea=round(avg_area, 2),
        totalListings=total_listings,
        activeListings=active_listings
    )
    price_dist = PriceDistribution(q1=round(q1, 2), q2=round(q2, 2), q3=round(q3, 2))
    return DashboardResponse(district=district, stats=stats, priceDistribution=price_dist)

@app.post("/trends", response_model=TrendsResponse)
async def trends(data: dict = Body(...)):
    district = data.get("district")
    if not district:
        raise HTTPException(status_code=400, detail="district parametresi zorunlu.")
    
    # District kontrolü
    if district not in district_enc:
        raise HTTPException(status_code=400, detail=f"Bilinmeyen ilce: {district}")
    
    df = df_compare[df_compare['District'] == district]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"{district} için veri bulunamadı.")
    # Fiyat trendi (örnek: son 12 ay)
    if 'Adrtisement Date' in df.columns and not df['Adrtisement Date'].isnull().all():
        df_trend = df.copy()  # Copy oluştur
        df_trend.loc[:, 'Adrtisement Date'] = pd.to_datetime(df_trend['Adrtisement Date'], format='%d/%m/%Y', errors='coerce')
        df_trend = df_trend.dropna(subset=['Adrtisement Date'])
        
        if len(df_trend) > 5:  # Yeterli veri varsa
            df_trend = df_trend.sort_values(by='Adrtisement Date')
            df_trend.loc[:, 'Month'] = df_trend['Adrtisement Date'].dt.to_period('M')
            price_history = df_trend.groupby('Month')['Price'].mean().reset_index()
            price_history = price_history.dropna()  # NaN değerleri temizle
            
            if len(price_history) > 1:
                first_price = price_history['Price'].iloc[0]
                last_price = price_history['Price'].iloc[-1]
                if not (np.isnan(first_price) or np.isnan(last_price) or np.isinf(first_price) or np.isinf(last_price)) and first_price != 0:
                    trend = ((last_price - first_price) / first_price) * 100
                else:
                    trend = 0.0
                price_history_items = [PriceHistoryItem(date=str(row['Month']), price=round(row['Price'], 2)) for _, row in price_history.iterrows()]
            else:
                trend = 0.0
                price_history_items = []
        else:
            trend = 0.0
            price_history_items = []
    else:
        price_history_items = []
        trend = 0.0
    prices = df['Price']
    avg_price = float(np.mean(prices))
    median_price = float(np.median(prices))
    min_price = float(np.min(prices))
    max_price = float(np.max(prices))
    listings = int(len(df))
    current_stats = CurrentStats(
        avgPrice=round(avg_price, 2),
        medianPrice=round(median_price, 2),
        minPrice=round(min_price, 2),
        maxPrice=round(max_price, 2),
        listings=listings
    )
    trend_info = TrendInfo(trend=round(trend, 2), priceHistory=price_history_items, currentStats=current_stats)
    return TrendsResponse(district=district, trendInfo=trend_info)

# =====================
# MODEL TRAINING ENDPOINT
# =====================

class TrainingData(BaseModel):
    csv_data: str  # CSV string olarak gelecek
    target_column: str = "Price"
    test_size: float = 0.2

class TrainingResponse(BaseModel):
    status: str
    message: str
    training_results: dict
    model_updated: bool

@app.post("/train-with-new-data", response_model=TrainingResponse)
async def train_with_new_data(data: TrainingData):
    try:
        # CSV verisini DataFrame'e çevir
        from io import StringIO
        new_df = pd.read_csv(StringIO(data.csv_data), sep=';')
        
        # Temel temizlik
        new_df['Price'] = new_df['Price'].str.replace(' TL', '').str.replace('.', '').astype(float)
        new_df['Net_m2'] = pd.to_numeric(new_df['m² (Net)'], errors='coerce')
        
        # Oda mapping
        room_map = {
            '1+0': 1, '1+1': 2, '2+0': 2, '2+1': 3, '2+2': 4,
            '3+1': 4, '3+2': 5, '4+1': 5, '4+2': 6, '4+3': 7,
            '5+1': 6, '5+2': 7, '5+3': 8, '5+4': 9,
            '6+1': 7, '6+2': 8, '6+3': 9, '7+1': 8, '7+2': 9,
            '8+1': 9, '8+2': 10, '8+3': 11, '8+4': 12,
            '9+1': 10, '9+2': 11, '10 and more': 12
        }
        new_df['Rooms'] = new_df['Number of rooms'].map(room_map)
        
        # Yaş mapping
        age_map = {
            '0': 0, '1-5 between': 3, '6-10 between': 8, '11-15 between': 13,
            '16-20 between': 18, '21-25 between': 23, '26-30 between': 28,
            '31  and more than': 35, '5-10 between': 7.5
        }
        new_df['Building_Age'] = new_df['Building Age'].replace(age_map)
        new_df['Building_Age'] = pd.to_numeric(new_df['Building_Age'], errors='coerce').fillna(10)
        
        # Temizleme
        new_df = new_df.dropna(subset=['Price', 'Net_m2', 'Rooms', 'District'])
        new_df = new_df[(new_df['Price'] > 100000) & (new_df['Price'] < 10000000)]
        new_df = new_df[(new_df['Net_m2'] > 20) & (new_df['Net_m2'] < 600)]
        
        # Mevcut veriyle birleştir
        combined_df = pd.concat([df_compare, new_df], ignore_index=True)
        
        # Feature engineering
        combined_df['features'] = combined_df.apply(lambda row: engineer_features({
            'district': row['District'],
            'neighborhood': row['Neighborhood'],
            'net_m2': row['Net_m2'],
            'gross_m2': row['Net_m2'] * 1.15,
            'rooms': row['Rooms'],
            'building_age': row['Building_Age'],
            'floor': 2,
            'num_floors': 5,
            'bathrooms': 1
        }), axis=1)
        
        # X ve y hazırla
        X = np.vstack(combined_df['features'].values)
        y = combined_df['Price'].values
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data.test_size, random_state=42)
        
        # Base modelleri eğit
        trained_base_models = {}
        base_predictions_train = []
        base_predictions_test = []
        
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            trained_base_models[name] = model
            base_predictions_train.append(model.predict(X_train))
            base_predictions_test.append(model.predict(X_test))
        
        # Meta model için stacking
        X_meta_train = np.column_stack(base_predictions_train)
        X_meta_test = np.column_stack(base_predictions_test)
        
        # Meta model eğit
        meta_model.fit(X_meta_train, y_train)
        
        # Tahminler
        meta_pred_train = meta_model.predict(X_meta_train)
        meta_pred_test = meta_model.predict(X_meta_test)
        
        # Metrikler
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        train_mae = mean_absolute_error(y_train, meta_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, meta_pred_train))
        train_r2 = r2_score(y_train, meta_pred_train)
        
        test_mae = mean_absolute_error(y_test, meta_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, meta_pred_test))
        test_r2 = r2_score(y_test, meta_pred_test)
        
        # Model güncelle (isteğe bağlı)
        # Bu kısım comment out, çünkü production'da model güncellemek riskli
        # updated_model_data = {
        #     "base_models": trained_base_models,
        #     "meta_model": meta_model,
        #     "feature_columns": feature_columns,
        #     "district_encoding": district_enc,
        #     "neighborhood_encoding": neighborhood_enc,
        #     "global_mean": global_mean
        # }
        # with open(MODEL_PATH, "wb") as f:
        #     pickle.dump(updated_model_data, f)
        
        training_results = {
            "original_data_size": len(df_compare),
            "new_data_size": len(new_df),
            "combined_data_size": len(combined_df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_metrics": {
                "mae": round(train_mae, 2),
                "rmse": round(train_rmse, 2),
                "r2": round(train_r2, 4)
            },
            "test_metrics": {
                "mae": round(test_mae, 2),
                "rmse": round(test_rmse, 2),
                "r2": round(test_r2, 4)
            }
        }
        
        return TrainingResponse(
            status="success",
            message=f"Model yeni verilerle eğitildi. {len(new_df)} yeni kayıt eklendi.",
            training_results=training_results,
            model_updated=False  # Production'da model güncellenmiyor
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eğitim hatası: {str(e)}")

# =====================
# MODEL METRICS ENDPOINT
# =====================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
