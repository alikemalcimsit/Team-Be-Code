#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZED TRAINING V6 - Minimal Features, Maximum Performance
==============================================================
Target: R² > 0.85 with fewer features (~25) and minimal overfitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    ExtraTreesRegressor, 
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import pickle
import warnings
warnings.filterwarnings('ignore')

# XGBoost import
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

print("=" * 80)
print("🚀 OPTIMIZED TRAINING V6 - Minimal Features")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n📂 Loading data...")
df = pd.read_csv('../data/hackathon_train_set.csv', sep=';', encoding='utf-8')
print(f"✅ Loaded: {df.shape[0]} rows")

# =============================================================================
# 2. PREPROCESSING
# =============================================================================
print("\n🧹 Preprocessing...")

df['Price'] = df['Price'].str.replace(' TL', '').str.replace('.', '').astype(float)
df['Net_m2'] = pd.to_numeric(df['m² (Net)'], errors='coerce')
df['Gross_m2'] = pd.to_numeric(df['m² (Gross)'], errors='coerce')

age_map = {
    '0': 0, '1-5 between': 3, '6-10 between': 8, '11-15 between': 13,
    '16-20 between': 18, '21-25 between': 23, '26-30 between': 28,
    '31  and more than': 35, '5-10 between': 7.5
}
df['Building_Age'] = df['Building Age'].replace(age_map)
df['Building_Age'] = pd.to_numeric(df['Building_Age'], errors='coerce').fillna(10)
df['Num_Floors'] = pd.to_numeric(df['Number of floors'], errors='coerce').fillna(5)

room_map = {
    '1+0': 1, '1+1': 2, '2+0': 2, '2+1': 3, '2+2': 4,
    '3+1': 4, '3+2': 5, '4+1': 5, '4+2': 6, '4+3': 7,
    '5+1': 6, '5+2': 7, '5+3': 8, '5+4': 9,
    '6+1': 7, '6+2': 8, '6+3': 9, '7+1': 8, '7+2': 9,
    '8+1': 9, '8+2': 10, '8+3': 11, '8+4': 12,
    '9+1': 10, '9+2': 11, '10 and more': 12
}
df['Rooms'] = df['Number of rooms'].map(room_map)

floor_map = {
    'Ground floor': 0, 'Kot 1': -1, 'Kot 2': -2, 'Kot 3': -3,
    'High entrance': 1, 'Entrance floor': 0, 'Mezzanine': 0.5,
    'Basement': -1, 'Middle floor': 3, 'Top floor': 10
}
df['Floor'] = df['Floor location'].replace(floor_map)
df['Floor'] = pd.to_numeric(df['Floor'], errors='coerce').fillna(2)
df['Bathrooms'] = pd.to_numeric(df['Number of bathrooms'], errors='coerce').fillna(1)

# Available for Loan - normalize to 0/1 if column present
if 'Available for Loan' in df.columns:
    def _avail_map(x):
        if pd.isna(x):
            return 0
        s = str(x).strip().lower()
        if s in ['1', '1.0', 'true', 'yes', 'y', 'evet', 'var', 'available']:
            return 1
        if s in ['0', '0.0', 'false', 'no', 'n', 'hayir', 'yok', 'not available', 'not_available']:
            return 0
        # fallback: check if numeric
        try:
            v = float(s)
            return 1 if v > 0 else 0
        except Exception:
            return 0

    df['Available_for_Loan'] = df['Available for Loan'].apply(_avail_map)
else:
    df['Available_for_Loan'] = 0

# Clean
df = df.dropna(subset=['Price', 'Net_m2', 'Rooms', 'District'])
df = df[(df['Price'] > 100000) & (df['Price'] < 10000000)]
df = df[(df['Net_m2'] > 20) & (df['Net_m2'] < 600)]
print(f"✅ Clean data: {df.shape[0]} rows")

# =============================================================================
# Heating system normalization
# Detect possible heating-related column names and create a normalized text column
# =============================================================================
heating_candidates = ['Heating', 'Heating System', 'Heating Type', 'Isıtma', 'Isıtma Sistemi', 'Heating_System', 'HeatingType']
heating_col = None
for c in heating_candidates:
    if c in df.columns:
        heating_col = c
        break

if heating_col:
    df['Heating_raw'] = df[heating_col].fillna('').astype(str).str.lower()
else:
    # try to infer from other columns or set empty
    df['Heating_raw'] = ''


# =============================================================================
# 3. TARGET ENCODING
# =============================================================================
print("\n🔧 Target encoding...")

global_mean = df['Price'].mean()
smoothing = 50

# District encoding
district_stats = df.groupby('District')['Price'].agg(['mean', 'count'])
district_stats['enc'] = (district_stats['mean'] * district_stats['count'] + global_mean * smoothing) / (district_stats['count'] + smoothing)
district_enc = district_stats['enc'].to_dict()

# Neighborhood encoding  
if 'Neighborhood' in df.columns:
    neigh_stats = df.groupby('Neighborhood')['Price'].agg(['mean', 'count'])
    neigh_stats['enc'] = (neigh_stats['mean'] * neigh_stats['count'] + global_mean * smoothing) / (neigh_stats['count'] + smoothing)
    neigh_enc = neigh_stats['enc'].to_dict()
else:
    neigh_enc = {}

# =============================================================================
# 4. MINIMAL FEATURE ENGINEERING
# =============================================================================
print("\n🔧 Engineering features (minimal set)...")

def engineer_features(data, district_enc, neigh_enc, global_mean):
    df = data.copy()
    
    # Target encodings (en önemli)
    df['District_enc'] = df['District'].map(district_enc).fillna(global_mean)
    if 'Neighborhood' in df.columns and neigh_enc:
        df['Neigh_enc'] = df['Neighborhood'].map(neigh_enc).fillna(global_mean)
    else:
        df['Neigh_enc'] = global_mean
    
    # Log transforms
    df['Log_m2'] = np.log1p(df['Net_m2'])
    df['Log_District'] = np.log1p(df['District_enc'])
    
    # Area polynomial
    df['m2_sq'] = df['Net_m2'] ** 2 / 10000  # Normalize
    
    # Key ratios
    df['m2_per_room'] = df['Net_m2'] / (df['Rooms'] + 0.1)
    df['Floor_ratio'] = df['Floor'] / (df['Num_Floors'] + 0.1)
    
    # Key interactions
    df['District_x_m2'] = df['District_enc'] * df['Net_m2'] / 1000000
    df['Age_x_m2'] = df['Building_Age'] * df['Net_m2'] / 1000
    
    # Age inverse (new buildings more valuable)
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

    # Pass-through for available for loan (already normalized in preprocessing)
    if 'Available_for_Loan' in df.columns:
        df['Available_for_Loan'] = df['Available_for_Loan'].fillna(0).astype(int)

    # Heating flags (from normalized Heating_raw column created in preprocessing)
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

df = engineer_features(df, district_enc, neigh_enc, global_mean)

# =============================================================================
# 5. TRAIN-TEST SPLIT
# =============================================================================
print("\n📊 Splitting data...")

district_freq = df['District'].value_counts()
df['District_bin'] = df['District'].map(lambda x: x if district_freq[x] > 50 else 'Other')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['District_bin'])
print(f"✅ Train: {len(train_df)}, Test: {len(test_df)}")

# =============================================================================
# 6. FEATURE SELECTION (25 features max)
# =============================================================================
feature_columns = [
    # Core (7)
    'Net_m2', 'Rooms', 'Building_Age', 'Floor', 'Num_Floors', 'Bathrooms', 'Gross_m2',
    
    # Encodings (3)
    'District_enc', 'Neigh_enc', 'Log_District',
    
    # Transforms (2)
    'Log_m2', 'm2_sq',
    
    # Ratios (2)
    'm2_per_room', 'Floor_ratio',
    
    # Interactions (3)
    'District_x_m2', 'Age_x_m2', 'Age_inv',
    
    # Categories (4)
    'Is_Luxury', 'Luxury_m2', 'Is_Budget', 'Is_New',
    # Available for loan flag
    'Available_for_Loan',
    # Heating system flags
    'Heating_Natural_Gas', 'Heating_Central', 'Heating_Electric', 'Heating_Stove',
    
    # Expected (1)
    'Expected'
]

feature_columns = [c for c in feature_columns if c in df.columns]
print(f"✅ Using {len(feature_columns)} features")

# Fill NaN
train_df[feature_columns] = train_df[feature_columns].fillna(0)
test_df[feature_columns] = test_df[feature_columns].fillna(0)

X_train = train_df[feature_columns].values
X_test = test_df[feature_columns].values

y_train = np.log1p(train_df['Price'].values)
y_test = np.log1p(test_df['Price'].values)

# Sample weights
luxury_mask = train_df['Is_Luxury'].values == 1
sample_weights = np.ones(len(y_train))
sample_weights[luxury_mask] = 1.15

# =============================================================================
# 7. DEFINE MODELS
# =============================================================================
print("\n🎯 Defining models...")

base_models = {
    'HGB1': HistGradientBoostingRegressor(
        max_iter=400, max_depth=8, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=0.1, random_state=42
    ),
    'HGB2': HistGradientBoostingRegressor(
        max_iter=500, max_depth=10, learning_rate=0.04,
        min_samples_leaf=15, l2_regularization=0.08, random_state=43
    ),
    'GB': GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8, random_state=44
    ),
    'ET': ExtraTreesRegressor(
        n_estimators=300, max_depth=20, min_samples_leaf=5,
        max_features=0.7, random_state=45, n_jobs=-1
    ),
    'RF': RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_leaf=5,
        max_features=0.7, random_state=46, n_jobs=-1
    ),
}

if HAS_XGB:
    base_models['XGB1'] = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=47, n_jobs=-1, verbosity=0
    )
    base_models['XGB2'] = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.8,
        random_state=48, n_jobs=-1, verbosity=0
    )

print(f"✅ Models: {list(base_models.keys())}")

# =============================================================================
# 8. OOF STACKING
# =============================================================================
print("\n🎯 Building OOF predictions...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train), len(base_models)))
test_preds = np.zeros((len(X_test), len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    print(f"  {name}...", end=" ")
    
    oof = np.zeros(len(X_train))
    tst = np.zeros(len(X_test))
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]
        w_tr = sample_weights[train_idx]
        
        fold_model = model.__class__(**model.get_params())
        try:
            fold_model.fit(X_tr, y_tr, sample_weight=w_tr)
        except TypeError:
            fold_model.fit(X_tr, y_tr)
        
        oof[val_idx] = fold_model.predict(X_val)
        tst += fold_model.predict(X_test) / 5
    
    oof_preds[:, i] = oof
    test_preds[:, i] = tst
    print(f"R² = {r2_score(y_train, oof):.4f}")

# =============================================================================
# 9. META-MODEL
# =============================================================================
print("\n🎯 Tuning meta-model...")

best_r2 = -np.inf
best_meta = None
best_alpha = None

for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    meta = Ridge(alpha=alpha)
    scores = cross_val_score(meta, oof_preds, y_train, cv=5, scoring='r2')
    if scores.mean() > best_r2:
        best_r2 = scores.mean()
        best_meta = Ridge(alpha=alpha)
        best_alpha = alpha

print(f"✅ Best alpha: {best_alpha} (CV R²: {best_r2:.5f})")
best_meta.fit(oof_preds, y_train)

# =============================================================================
# 10. EVALUATION
# =============================================================================
train_pred = best_meta.predict(oof_preds)
test_pred = best_meta.predict(test_preds)

y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)
train_pred_actual = np.expm1(train_pred)
test_pred_actual = np.expm1(test_pred)

def calc_metrics(y_true, y_pred):
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'r2': r2_score(y_true, y_pred)
    }

train_m = calc_metrics(y_train_actual, train_pred_actual)
test_m = calc_metrics(y_test_actual, test_pred_actual)

# CV metrics
cv_r2s = []
for train_idx, val_idx in kf.split(X_train):
    val_pred = best_meta.predict(oof_preds[val_idx])
    cv_r2s.append(r2_score(np.expm1(y_train[val_idx]), np.expm1(val_pred)))
cv_consistency = np.std(cv_r2s) / np.mean(cv_r2s)

print("\n" + "="*80)
print("📊 FINAL RESULTS - V6 (Minimal Features)")
print("="*80)

print(f"\n🎯 Train: R²={train_m['r2']:.4f}, MAPE={train_m['mape']:.2f}%, RMSE={train_m['rmse']:,.0f} TL")
print(f"🎯 Test:  R²={test_m['r2']:.4f}, MAPE={test_m['mape']:.2f}%, RMSE={test_m['rmse']:,.0f} TL")
print(f"🎯 CV:    R²={np.mean(cv_r2s):.4f} (±{np.std(cv_r2s):.4f}), Consistency={cv_consistency:.4f}")

print(f"\n🔍 Overfitting Check:")
r2_gap = train_m['r2'] - test_m['r2']
print(f"   R² Gap: {r2_gap:.4f}")
if abs(r2_gap) < 0.02:
    print("   ✅ EXCELLENT - Minimal overfitting")
elif abs(r2_gap) < 0.05:
    print("   ✅ GOOD - Low overfitting")
else:
    print("   ⚠️ WARNING - Overfitting detected")

# =============================================================================
# 11. SAVE
# =============================================================================
print("\n🎯 Fitting final models...")

final_models = {}
for name, model in base_models.items():
    m = model.__class__(**model.get_params())
    try:
        m.fit(X_train, y_train, sample_weight=sample_weights)
    except TypeError:
        m.fit(X_train, y_train)
    final_models[name] = m

print("\n💾 Saving model...")

model_pkg = {
    'base_models': final_models,
    'meta_model': best_meta,
    'feature_columns': feature_columns,
    'district_encoding': district_enc,
    'neighborhood_encoding': neigh_enc,
    'global_mean': global_mean,
    'metrics': {
        'train_r2': train_m['r2'], 'test_r2': test_m['r2'],
        'train_mape': train_m['mape'], 'test_mape': test_m['mape'],
        'cv_r2_mean': np.mean(cv_r2s), 'cv_consistency': cv_consistency
    }
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_pkg, f)

print("✅ Model saved!")
print("\n" + "="*80)
print(f"📊 ÖZET: {len(feature_columns)} feature ile R²={test_m['r2']:.4f}")
print("="*80)
