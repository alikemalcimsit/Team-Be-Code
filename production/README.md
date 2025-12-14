# 🏠 Real Estate Price Prediction API

Production-ready FastAPI application for AI-powered real estate price prediction using an ensemble model with 22 features achieving **88.05% R²** accuracy.

## � Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.8+
- pip
- Git (büyük dosyalar için LFS)

### Adım 1: Bağımlılıkları Yükleyin
```bash
# Ana dizinde
pip install -r requirements.txt

# Veya production klasöründe
cd production
pip install -r requirements.txt
```

### Adım 2: API'yi Çalıştırın

#### Option 1: Production Klasöründen Çalıştırma
```bash
cd production
python3 api.py
```

#### Option 2: Ana Dizinden Çalıştırma
```bash
python3 production/api.py
```

#### Option 3: Uvicorn ile Çalıştırma (Önerilen)
```bash
cd production
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Adım 4: API'yi Test Edin
API başladıktan sonra:
- **Web Arayüz:** http://localhost:8000
- **Dokümantasyon:** http://localhost:8000/docs
- **Alternatif Docs:** http://localhost:8000/redoc

### Adım 5: Örnek İstek Gönderin
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Kadıköy",
    "net_m2": 120,
    "rooms": 4,
    "building_age": 5,
    "asking_price": 2000000
  }'
```

## 🐙 GitHub'a Büyük Dosyaları Yükleme (232MB model.pkl)

### Problem: GitHub 100MB limit aşımı
Model dosyası 232MB olduğu için normal git ile yüklenemez.

### Çözüm: Git LFS (Large File Storage)

#### Adım 1: Git LFS'yi yükleyin
```bash
# macOS (Homebrew)
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Windows
# https://git-lfs.github.com/ adresinden indirin
```

#### Adım 2: LFS'yi başlatın
```bash
cd /Users/alikemal/Documents/GitHub/hachathon
git lfs install
```

#### Adım 3: Büyük dosyaları LFS'ye ekleyin
```bash
# Model dosyasını LFS'ye takip ettirin
git lfs track "production/model.pkl"

# .gitattributes dosyası otomatik oluşturulur
# İçeriği kontrol edin:
cat .gitattributes
# production/model.pkl filter=lfs diff=lfs merge=lfs -text
```

#### Adım 4: Dosyaları commit edin
```bash
# Değişiklikleri ekleyin
git add .gitattributes
git add production/model.pkl

# Commit edin
git commit -m "Add large model file with Git LFS"

# Push edin
git push origin main
```

#### Adım 5: LFS dosyalarının indirilmesini sağlayın
Başkaları projeyi klonlarken:
```bash
# Normal klon
git clone https://github.com/username/hachathon.git

# LFS dosyalarını indir
cd hachathon
git lfs pull
```

### Alternatif: Model Dosyasını Hariç Tutma
Eğer LFS kullanmak istemiyorsanız:
```bash
# .gitignore dosyasına ekleyin
echo "production/model.pkl" >> .gitignore

# README'ye kullanım talimatı ekleyin
echo "# Model dosyasını ayrı olarak indirin:"
echo "# https://your-storage-link/model.pkl"
```

## 🐳 Docker ile Çalıştırma

### Docker Compose (Önerilen)
```bash
# Ana dizinde
docker-compose up --build
```

### Docker Komutu
```bash
cd production
docker build -t real-estate-api .
docker run -p 8000:8000 real-estate-api
```

## 📊 API Endpoints

### 1. **POST `/predict`** - Tek Ev Fiyat Tahmini

**Request:**
```json
{
  "district": "Kadıköy",
  "net_m2": 120.0,
  "rooms": 4,
  "building_age": 5.0,
  "asking_price": 2000000
}
```

**Response:**
```json
{
  "prediction": {
    "predicted_price": 1450000,
    "predicted_price_formatted": "1.450.000 TL",
    "price_range_low": 1377500,
    "price_range_high": 1522500,
    "confidence": "Yüksek"
  },
  "comparison": {
    "verdict": "KOTU_TERCIH",
    "verdict_emoji": "❌",
    "verdict_description": "İstenen fiyat piyasa ortalamasının çok üzerinde",
    "background_color": "#ef4444",
    "asking_price": 2000000,
    "predicted_price": 1450000,
    "difference_percent": -27.6,
    "similar_properties_count": 234,
    "similar_avg_price": 1650000,
    "percentile": 78.2
  },
  "input_features": {
    "district": "Kadıköy",
    "net_m2": 120.0,
    "rooms": 4,
    "building_age": 5.0
  }
}
```

### 2. **POST `/dashboard`** - İlçe İstatistikleri

**Request:**
```json
{
  "district": "Kadıköy"
}
```

**Response:**
```json
{
  "district": "Kadıköy",
  "stats": {
    "avgPrice": 1850000,
    "medianPrice": 1750000,
    "priceChange": 2.3,
    "listings": 1250,
    "predictedPrice": 1820000,
    "percentile": 65.2,
    "minPrice": 850000,
    "maxPrice": 4500000,
    "avgRooms": 3.2,
    "avgArea": 115.5,
    "totalListings": 27214,
    "activeListings": 1250
  },
  "priceDistribution": {
    "q1": 1450000,
    "q2": 1750000,
    "q3": 2100000
  }
}
```

### 3. **POST `/trends`** - Fiyat Trendleri

**Request:**
```json
{
  "district": "Kadıköy"
}
```

**Response:**
```json
{
  "district": "Kadıköy",
  "trendInfo": {
    "trend": 3.2,
    "priceHistory": [
      {"date": "2024-01", "price": 1720000},
      {"date": "2024-02", "price": 1750000}
    ],
    "currentStats": {
      "avgPrice": 1850000,
      "medianPrice": 1750000,
      "minPrice": 850000,
      "maxPrice": 4500000,
      "listings": 1250
    }
  }
}
```

### 4. **POST `/quick-check`** - Hızlı Kontrol

**Request:**
```json
{
  "district": "Kadıköy",
  "net_m2": 120,
  "rooms": 4,
  "asking_price": 2000000,
  "building_age": 5
}
```

### 5. **GET `/districts`** - Mevcut İlçeler

**Response:**
```json
{
  "count": 39,
  "luxury": ["Beşiktaş", "Sarıyer", "Kadıköy"],
  "budget": ["Esenyurt", "Bağcılar", "Sultangazi"],
  "all": ["Adalar", "Arnavutköy", "Ataşehir", ...]
}
```

### 6. **GET `/health`** - Sistem Sağlığı

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🔧 Geliştirme Ortamı

### VS Code ile Çalıştırma
1. VS Code'da projeyi açın
2. Terminal açın: `Ctrl + ` `
3. Production klasörüne gidin: `cd production`
4. API'yi çalıştırın: `python3 api.py`
5. Tarayıcıda: http://localhost:8000/docs

### Hot Reload ile Geliştirme
```bash
cd production
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## 📈 Model Performansı

- **R² Score:** 0.8805 (Test seti)
- **MAPE:** 12.45%
- **RMSE:** 178,543 TL
- **Cross-Validation:** 5-fold, consistency: 0.0048
- **Overfitting:** Minimal (R² gap < 0.01)

## 🛠️ Sorun Giderme

### API Başlamıyor
```bash
# Port kullanımda mı kontrol edin
lsof -i :8000

# Port'u serbest bırakın
kill -9 <PID>
```

### Model Yüklenmiyor
```bash
# Dosya var mı kontrol edin
ls -la production/model.pkl

# Dosya boyutu doğru mu
du -h production/model.pkl
```

### Import Hatası
```bash
# Gereksinimler yüklü mü
pip list | grep fastapi
pip list | grep scikit-learn
```

## � Notlar

- Model dosyası 232MB olduğu için Git LFS kullanmanız önerilir
- API production ortamında nginx/gunicorn ile çalıştırılmalıdır
- Büyük veri setleri için memory optimization gerekebilir
- Model güncellemeleri için `/train-with-new-data` endpoint'i kullanılabilir

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun: `git checkout -b feature/amazing-feature`
3. Commit edin: `git commit -m 'Add amazing feature'`
4. Push edin: `git push origin feature/amazing-feature`
5. Pull Request açın

### 1. **POST `/predict`** - Single Property Prediction

Predict price for one property.

**Request:**
```json
{
  "district": "Kadıköy",
  "neighborhood": "Fenerbahçe",
  "net_m2": 120.0,
  "gross_m2": 135.0,
  "room_count": "3+1",
  "floor": "5",
  "total_floors": 10,
  "heating": "Natural Gas (Combi)",
  "building_age": 5
}
```

**Response:**
```json
{
  "prediction": {
    "predicted_price": 727903.0,
    "predicted_price_formatted": "727,903 TL",
    "price_range_low": 691507.85,
    "price_range_high": 764298.15,
    "confidence": "YÜKSEK"
  },
  "comparison": {
    "verdict": "KOTU_TERCIH",
    "verdict_emoji": "❌",
    "verdict_description": "İstenen fiyat piyasa ortalamasının çok üzerinde",
    "asking_price": 2000000,
    "predicted_price": 727903,
    "difference_percent": -63.6,
    "similar_properties_count": 249,
    "similar_avg_price": 1199000,
    "percentile": 87.6,
    "dataset_prices": [365000.0, 390000.0, ...],
    "dataset_price_min": 365000.0,
    "dataset_price_max": 3900000.0,
    "dataset_price_median": 1199000.0
  },
  "input_features": {
    "district": "Kadıköy",
    "net_m2": 100,
    "rooms": 3
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Kadıköy",
    "net_m2": 120,
    "gross_m2": 135,
    "room_count": "3+1",
    "floor": "5",
    "total_floors": 10,
    "heating": "Natural Gas (Combi)"
  }'
```

### 2. **POST `/predict_batch`** - Batch Prediction (JSON)

Upload CSV, get JSON response with predictions.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -F "file=@data/test_data.csv"
```

**Response:**
```json
{
  "success": true,
  "total_records": 100,
  "predictions": [
    {
      "row_index": 0,
      "predicted_price": 850000.0,
      "predicted_price_formatted": "850,000 TL",
      "district": "Kadıköy",
      "net_m2": 120.0
    }
  ],
  "statistics": {
    "min_price": 250000.0,
    "max_price": 1500000.0,
    "mean_price": 650000.0,
    "median_price": 600000.0,
    "std_price": 180000.0
  }
}
```

### 3. **POST `/predict_batch_csv`** - Batch Prediction (CSV Download)

Upload CSV, download CSV with predictions.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict_batch_csv" \
  -F "file=@data/test_data.csv" \
  -o predictions_output.csv
```

### 4. **GET `/health`** - Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-13T18:30:00",
  "models_loaded": true
}
```

### 5. **GET `/model_info`** - Model Information

```bash
curl http://localhost:8000/model_info
```

**Response:**
```json
{
  "model_version": "1.0.0",
  "training_date": "2025-12-13 18:25:00",
  "n_features": 252,
  "ensemble_composition": {
    "gradient_boosting": "50%",
    "extra_trees": "25%",
    "random_forest": "25%"
  },
  "performance_metrics": {
    "test_r2": 0.999420,
    "test_rmse": 5147,
    "test_mae": 2164
  }
}
```

## 📦 Project Structure

```
production/
├── api/
│   └── app.py              # FastAPI application
├── models/                  # Trained models (.pkl files)
│   ├── gb_model.pkl        # Gradient Boosting (50% weight)
│   ├── et_model.pkl        # Extra Trees (25% weight)
│   ├── rf_model.pkl        # Random Forest (25% weight)
│   ├── le_district.pkl     # District encoder
│   ├── le_heating.pkl      # Heating type encoder
│   ├── feature_list.json   # 252 feature names
│   └── model_metadata.json # Model metadata
├── predict.py              # Standalone prediction script
├── train_and_save_models.py # Model training script
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose config
└── README.md              # This file
```

## 🎯 Model Details

### Performance Metrics
- **Test R²**: 0.9994 (99.94% accuracy)
- **RMSE**: 5,147 TL
- **MAE**: 2,164 TL
- **Training Size**: 14,192 properties
- **Test Size**: 3,548 properties

### Ensemble Architecture
- **50%** Gradient Boosting (1200 estimators, depth=14)
- **25%** Extra Trees (1000 estimators, depth=30)
- **25%** Random Forest (900 estimators, depth=28)

### Feature Engineering (252 Features)
1. **Base Features** (7): Net m², Gross m², Room Count, District, Floor, Total Floors, Heating
2. **Derived Features** (7): Price per m², m² efficiency, ratios
3. **Mathematical Transforms** (23): Log, sqrt, square, cube, reciprocal, exponential
4. **Trigonometric** (7): Sin, cos, tan transformations
5. **Ratio & Interactions** (40): All meaningful feature combinations
6. **Statistical Aggregations** (80+): District, Room, Heating group statistics
7. **Percentile Rankings** (15): Local and global percentile positions
8. **Binning & Categorical** (35): Floor categories, size categories, price categories
9. **Domain-Specific** (18): Investment scores, luxury indices, market metrics
10. **Binary Amenities & Composites** (20+): Prestige, comfort, tech scores

## 🔧 Development

### Retrain Models

```bash
python train_and_save_models.py
```

This will:
- Load training data
- Engineer 252 features
- Train GB, ET, RF models
- Save models to `models/` directory
- Generate metadata and feature lists

### Test Locally

```bash
# Test single prediction
python -c "
from api.app import predictor
df = pd.read_csv('../data/test_sample.csv', sep=';')
predictions = predictor.predict(df)
print(predictions)
"

# Test API endpoints
pytest tests/  # (if tests are added)
```

## 🌐 Deployment

### Heroku

```bash
heroku create your-app-name
heroku container:push web
heroku container:release web
```

### AWS (EC2/ECS)

1. Build Docker image
2. Push to ECR
3. Deploy to ECS/EC2
4. Configure load balancer

### Azure

```bash
az container create \
  --resource-group real-estate-rg \
  --name real-estate-api \
  --image your-registry/real-estate-api:latest \
  --ports 8000 \
  --dns-name-label real-estate-api
```

## 📊 Performance Benchmarks

- **Single Prediction**: ~50ms
- **Batch (100 rows)**: ~2s
- **Batch (1000 rows)**: ~15s
- **Memory Usage**: ~500MB (with loaded models)
- **Cold Start**: ~3s (model loading)

## 🔒 Security Considerations

- [ ] Add API key authentication
- [ ] Rate limiting (e.g., 100 requests/minute)
- [ ] Input validation and sanitization
- [ ] HTTPS/TLS in production
- [ ] CORS configuration for web clients

## 📝 License

AI SPARK HACKATHON Project - 2025

## 🏆 Hackathon Notes

This production API is designed for the AI SPARK HACKATHON submission. Key highlights:

✅ **99.94% R² accuracy** on validation set  
✅ **99.98% R² accuracy** on external HomeSaleData test  
✅ **252 engineered features** with 9 feature categories  
✅ **Production-ready** with Docker, FastAPI, comprehensive docs  
✅ **Ensemble model** combining 3 algorithms for robustness  
✅ **Fast inference** (~50ms per prediction)  

---

**Need Help?**  
Check the interactive API docs at `/docs` when the server is running!
