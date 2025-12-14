import React, { useState } from 'react'
import Map from '../components/Map'

const istanbulDistricts = [
  'Adalar', 'Arnavutköy', 'Ataşehir', 'Avcılar', 'Bağcılar', 'Bahçelievler',
  'Bakırköy', 'Başakşehir', 'Bayrampaşa', 'Beşiktaş', 'Beykoz', 'Beylikdüzü',
  'Beyoğlu', 'Büyükçekmece', 'Çatalca', 'Çekmeköy', 'Esenler', 'Esenyurt',
  'Eyüpsultan', 'Fatih', 'Gaziosmanpaşa', 'Güngören', 'Kadıköy', 'Kağıthane',
  'Kartal', 'Küçükçekmece', 'Maltepe', 'Pendik', 'Sancaktepe', 'Sarıyer',
  'Silivri', 'Sultanbeyli', 'Sultangazi', 'Şile', 'Şişli', 'Tuzla', 'Ümraniye',
  'Üsküdar', 'Zeytinburnu'
]

export default function PredictPrice() {
  const [formData, setFormData] = useState({
    district: '',
    neighborhood: '',
    net_m2: '',
    gross_m2: '',
    rooms: '',
    building_age: '',
    floor: '',
    num_floors: '',
    bathrooms: '',
    available_for_loan: '',
    heating: ''
  })

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const formatNumber = (value) => {
    if (!value) return ''
    const numbers = value.replace(/\D/g, '')
    return numbers
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    // Build payload - only include provided fields
    const payload = {
      district: formData.district,
      net_m2: Number(formData.net_m2) || Number(formData.gross_m2),
      rooms: Number(formData.rooms) || 0
    }

    // Add optional fields only if provided
    if (formData.neighborhood) payload.neighborhood = formData.neighborhood
    if (formData.gross_m2) payload.gross_m2 = Number(formData.gross_m2)
    if (formData.building_age) payload.building_age = Number(formData.building_age)
    if (formData.floor) payload.floor = Number(formData.floor)
    if (formData.num_floors) payload.num_floors = Number(formData.num_floors)
    if (formData.bathrooms) payload.bathrooms = Number(formData.bathrooms)
    if (formData.available_for_loan) payload.available_for_loan = formData.available_for_loan === 'true'
    if (formData.heating) payload.heating = formData.heating

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
      console.error('API Error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">Fiyat Tahmini</h1>
          <p className="text-xl text-gray-600">Emlakınızın özelliklerini girerek AI destekli fiyat tahmini alın</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Form Section */}
          <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-8 shadow-xl border border-gray-200/50">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* İlçe - Required */}
              <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                  İlçe <span className="text-orange-500">*</span>
                </label>
                <select
                  value={formData.district}
                  onChange={(e) => setFormData({ ...formData, district: e.target.value })}
                  required
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900"
                >
                  <option value="">İlçe seçiniz...</option>
                  {istanbulDistricts.map(district => (
                    <option key={district} value={district}>{district}</option>
                  ))}
                </select>
              </div>

              {/* Mahalle - Optional */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Mahalle (Opsiyonel)</label>
                <input
                  type="text"
                  value={formData.neighborhood}
                  onChange={(e) => setFormData({ ...formData, neighborhood: e.target.value })}
                  placeholder="Örn: Çengelköy Mh."
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900 placeholder-gray-400"
                />
              </div>

              {/* Metrekare */}
              <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Net m² <span className="text-orange-500">*</span>
                </label>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <input
                      type="text"
                      value={formData.net_m2}
                      onChange={(e) => setFormData({ ...formData, net_m2: formatNumber(e.target.value) })}
                      placeholder="Net m²"
                      required
                      className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900 placeholder-gray-400"
                    />
                  </div>
                  <div>
                    <input
                      type="text"
                      value={formData.gross_m2}
                      onChange={(e) => setFormData({ ...formData, gross_m2: formatNumber(e.target.value) })}
                      placeholder="Brüt m² (Opsiyonel)"
                      className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900 placeholder-gray-400"
                    />
                  </div>
                </div>
              </div>

              {/* Oda Sayısı - Required */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Oda Sayısı <span className="text-orange-500">*</span>
                </label>
                <input
                  type="number"
                  value={formData.rooms}
                  onChange={(e) => setFormData({ ...formData, rooms: e.target.value })}
                  placeholder="Örn: 4 (3+1 ev için 4 girin)"
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900 placeholder-gray-400"
                  min="1"
                  step="1"
                />
                <p className="text-xs text-gray-500 mt-1">3+1 ev için 4, 2+1 ev için 3 gibi toplam oda sayısını girin</p>
              </div>

              {/* Bina Yaşı - Optional */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Bina Yaşı (Opsiyonel)</label>
                <input
                  type="text"
                  value={formData.building_age}
                  onChange={(e) => setFormData({ ...formData, building_age: formatNumber(e.target.value) })}
                  placeholder="Örn: 12"
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900 placeholder-gray-400"
                />
              </div>

              {/* Kat Bilgileri */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Bulunduğu Kat (Opsiyonel)</label>
                  <input
                    type="text"
                    value={formData.floor}
                    onChange={(e) => setFormData({ ...formData, floor: formatNumber(e.target.value) })}
                    placeholder="Örn: 3"
                    className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900 placeholder-gray-400"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Toplam Kat Sayısı (Opsiyonel)</label>
                  <input
                    type="text"
                    value={formData.num_floors}
                    onChange={(e) => setFormData({ ...formData, num_floors: formatNumber(e.target.value) })}
                    placeholder="Örn: 6"
                    className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900 placeholder-gray-400"
                  />
                </div>
              </div>

              {/* Banyo Sayısı - Optional */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Banyo Sayısı (Opsiyonel)</label>
                <input
                  type="text"
                  value={formData.bathrooms}
                  onChange={(e) => setFormData({ ...formData, bathrooms: formatNumber(e.target.value) })}
                  placeholder="Örn: 2"
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900 placeholder-gray-400"
                />
              </div>

              {/* Krediye Uygunluk - Optional */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Krediye Uygunluk (Opsiyonel)</label>
                <select
                  value={formData.available_for_loan}
                  onChange={(e) => setFormData({ ...formData, available_for_loan: e.target.value })}
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900"
                >
                  <option value="">Seçiniz</option>
                  <option value="true">Evet</option>
                  <option value="false">Hayır</option>
                </select>
              </div>

              {/* Isıtma Sistemi - Optional */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Isıtma Sistemi (Opsiyonel)</label>
                <select
                  value={formData.heating}
                  onChange={(e) => setFormData({ ...formData, heating: e.target.value })}
                  className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-gray-900"
                >
                  <option value="">Seçiniz</option>
                  <option value="Doğalgaz">Doğalgaz</option>
                  <option value="Merkezi Sistem">Merkezi Sistem</option>
                  <option value="Kombi">Kombi</option>
                  <option value="Soba">Soba</option>
                  <option value="Klima">Klima</option>
                  <option value="Yerden Isıtma">Yerden Isıtma</option>
                  <option value="Diğer">Diğer</option>
                </select>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading || !formData.district || !formData.net_m2 || !formData.rooms}
                className="w-full px-6 py-4 bg-gradient-to-r from-gray-900 to-gray-800 text-white rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed transform hover:-translate-y-1"
              >
                {loading ? 'Analiz Ediliyor...' : 'Fiyat Tahmini Al'}
              </button>
            </form>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {/* Map */}
            <Map district={formData.district} neighborhood={formData.neighborhood} height="500px" />

            {/* Results */}
            {loading && (
              <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-8 shadow-xl border border-gray-200/50 text-center">
                <div className="inline-block animate-spin rounded-full h-16 w-16 border-b-4 border-gray-900 mb-4"></div>
                <h3 className="text-xl font-semibold text-gray-900">Analiz Ediliyor...</h3>
              </div>
            )}

            {error && (
              <div className="bg-orange-50 border border-orange-200 rounded-3xl p-8 shadow-xl">
                <div className="text-orange-700 font-semibold mb-2">⚠️ Bağlantı Hatası</div>
                <div className="text-gray-700 mb-4">{error}</div>
                <div className="mt-4 text-sm text-gray-600">
                  Backend servisinizin çalıştığından ve <code className="bg-gray-100 px-2 py-1 rounded text-gray-800">http://localhost:8000</code> adresinde erişilebilir olduğundan emin olun.
                </div>
              </div>
            )}

            {result && result.prediction && (
              <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-8 shadow-xl border border-gray-200/50 space-y-6">
                <div className="text-center">
                  <div className="text-sm text-gray-600 mb-2">Tahmini Değer</div>
                  <div className="text-5xl font-bold text-gray-900 mb-2">
                    {result.prediction.predicted_price_formatted || 
                      `${result.prediction.predicted_price?.toLocaleString('tr-TR')} TL`}
                  </div>
                  <div className="text-sm text-gray-600">
                    Aralık: {result.prediction.price_range_low?.toLocaleString('tr-TR')} - {result.prediction.price_range_high?.toLocaleString('tr-TR')} TL
                  </div>
                  {result.prediction.confidence && (
                    <div className="mt-3 inline-block px-4 py-2 bg-gray-100 rounded-full">
                      <span className="text-sm font-semibold text-gray-700">Güven: {result.prediction.confidence}</span>
                    </div>
                  )}
                </div>

                {result.comparison && (
                  <div className="border-t border-gray-200 pt-6">
                    <div className="text-sm text-gray-600 mb-2">Benzer İlanlar</div>
                    <div className="text-lg font-semibold text-gray-900">
                      {result.comparison.similar_properties_count} benzer emlak analiz edildi
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

