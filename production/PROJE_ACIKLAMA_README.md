# Ev Fiyat Tahmini API - Proje Açıklaması

## Proje Amacı
Bu proje, Türkiye'deki konut piyasasında, kullanıcıların girdiği ev özelliklerine göre makine öğrenmesiyle fiyat tahmini yapan ve piyasadaki benzer evlerle karşılaştırma sunan bir REST API geliştirmeyi amaçlar. Hedef, hem yatırımcıya hem de oturum amaçlı ev arayanlara hızlı, şeffaf ve veri temelli karar desteği sağlamaktır.

## Neden Bu Proje?
- **Şeffaflık:** Piyasa fiyatlarının ve model tahminlerinin kullanıcıya açıkça sunulması.
- **Hız:** Anında tahmin ve karşılaştırma ile kullanıcı deneyimini iyileştirmek.
- **Veri Odaklılık:** Gerçek ilan verileriyle eğitilmiş model sayesinde güvenilir sonuçlar.
- **Yatırım ve Oturum Analizi:** Kullanım amacına göre farklı değerlendirme ve öneriler.

## Kullanılan Teknolojiler ve Nedenleri
- **FastAPI:** Modern, hızlı ve otomatik dokümantasyon sunan bir Python web framework'ü. Yüksek performans ve kolay geliştirme için seçildi.
- **Pandas & NumPy:** Veri işleme ve istatistiksel analizler için.
- **scikit-learn:** Fiyat tahmin modeli (ensemble/stacking) ve temel makine öğrenmesi işlemleri için.
- **Pickle:** Eğitilmiş modellerin hızlıca yüklenmesi için.
- **Uvicorn:** API sunucusunu çalıştırmak için hızlı bir ASGI server.

## API Nasıl Çalışır?
1. **/predict:** Kullanıcıdan evin temel özellikleri alınır. Model, bu özelliklerle fiyat tahmini yapar. Eğer kullanıcı "isteyen fiyat" (asking_price) girerse, bu fiyatı piyasadaki benzer evlerle ve model tahminiyle karşılaştırır, uygunluk analizi döner.
2. **/dashboard:** Bir ilçedeki tüm ilanlardan istatistikler (ortalama, medyan, min, max fiyat, oda, alan, fiyat değişimi vs.) döner.
3. **/trends:** İlçedeki fiyatların zaman içindeki değişimini ve güncel istatistikleri verir.
4. **/train-with-new-data:** Yeni veriyle modeli yeniden eğitir (opsiyonel, prod ortamda model güncellenmez).
5. **/quick-check:** Hızlı fiyat ve piyasa karşılaştırması yapar.

## Model Nasıl Eğitildi?
- 27.000+ gerçek ilan verisiyle, 22 mühendislik özelliği kullanılarak stacking ensemble modeli eğitildi.
- İlçe ve mahalle encoding, metrekare, oda, yaş, lüks/bütçe bölge gibi domain bilgisiyle zenginleştirildi.
- Model, hem genel hem de bölgesel fiyat dağılımlarını dikkate alır.

## Neden Model Metrics API Kaldırıldı?
- Modelin performans metrikleri (R², MAE, RMSE) geliştirme ve validasyon aşamasında ölçülüp optimize edildi.
- Üretim ortamında, kullanıcıya modelin teknik detaylarını göstermek yerine, doğrudan tahmin ve karşılaştırma sunmak daha kullanıcı dostu ve güvenli.
- Gereksiz endpoint karmaşasını önlemek için kaldırıldı.

## Proje Dosya Yapısı
- `api.py`: Tüm endpoint ve iş mantığı burada.
- `model.pkl`: Eğitilmiş model ve encodingler.
- `data/`: Eğitim ve karşılaştırma için kullanılan veri setleri.
- `requirements.txt`: Gerekli Python paketleri.

## Geliştirme ve Test
- Tüm endpointler Postman ile test edildi.
- Pandas uyarıları ve performans sorunları giderildi.
- Kodda açıklamalar ve tip kontrolleri ile sürdürülebilirlik sağlandı.

## Sonuç
Bu API, konut piyasasında hızlı, güvenilir ve anlaşılır fiyat tahmini ile karar desteği sunar. Kod ve model, yeni veriyle kolayca güncellenebilir ve farklı şehirler için uyarlanabilir.
