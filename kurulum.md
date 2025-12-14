# Kurulum Rehberi

## Gerekliliklerin Yüklenmesi

İlk olarak, projenin gerekliliklerini yüklemek için aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

## Model Eğitimi

Model eğitimi için production klasörüne gidin ve eğitim script'ini çalıştırın:

```bash
cd production
python3 train_v6_minimal.py
```

## Model Testi (Konsol Üzerinden)

Modeli test etmek için test script'ini kullanın. Test verinizi hazırladıktan sonra:

```bash
cd production
python3 test_model.py /path/to/your/test_data.csv
```

**Test verisi formatı:**

- CSV dosyası noktalı virgül (;) ile ayrılmış olmalı
- Gerekli sütunlar: `District`, `Price`, `m² (Net)`, `Number of rooms`
- Opsiyonel sütunlar: `Neighborhood`, `m² (Gross)`, `Building Age`, `Floor location`, `Number of floors`, `Number of bathrooms`, `Heating`, `Available for Loan`

**Örnek kullanım:**

```bash
python3 test_model.py ../data/test_verisi.csv
```

**Script aşağıdaki metrikleri gösterecek:**

- MAE (Ortalama Mutlak Hata)
- RMSE (Kök Ortalama Kare Hata)
- MAPE (Ortalama Yüzde Hata)
- R² Skoru

Sonuçlar ayrıca `test_verisi_sonuclar.csv` olarak kaydedilecektir.

## API'nin Başlatılması

Model eğitimi tamamlandıktan sonra, API'yi başlatmak için:

```bash
python3 api.py
```

API hazır hale gelecektir.

## Frontend'in Çalıştırılması

Frontend'i çalıştırmak için frontend klasörüne gidin ve geliştirme sunucusunu başlatın:

```bash
cd ../frontend
npm run dev
```

Artık uygulama kullanıma hazırdır.
