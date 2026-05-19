import gradio as gr
import pandas as pd
import joblib

# 1. Tam Kapsamlı (20 Özellikli) Modelleri ve Sütunları Yükle
xgb_model = joblib.load('xgb_su_modeli_full.pkl')
kmeans_model = joblib.load('kmeans_su_modeli_full.pkl')
scaler = joblib.load('scaler_su_full.pkl')
sutun_isimleri = joblib.load('tam_sutun_isimleri.pkl') # CSV'deki tam sıra

kume_profilleri = {
    0: "Grup 0: Genel Karışık / Kalitesiz Su",
    1: "Grup 1: Biyolojik Tehlike Alanı! (Virüs/Bakteri)",
    2: "Grup 2: Görece Düşük Riskli (Ön Eleme Grubu)",
    3: "Grup 3: Ağır Metal Kokteyli! (Kimyasal Tehlike)"
}

# 2. Tahmin Fonksiyonu (*args kullanarak sınırsız/dinamik sayıda girdi alma)
def su_kalitesi_tahmin_et(*args):
    # Gelen değerleri sütun isimleriyle eşleştirip tek satırlık DataFrame yapıyoruz
    girdi_sozlugu = {sutun_isimleri[i]: [args[i]] for i in range(len(sutun_isimleri))}
    girdi_verisi = pd.DataFrame(girdi_sozlugu)
    
    # --- XGBoost Tahmini ---
    karar = xgb_model.predict(girdi_verisi)[0]
    eminlik_skoru = xgb_model.predict_proba(girdi_verisi)[0][1] * 100
    
    if karar == 1:
        xgb_cikti = f"✅ GÜVENLİ / İÇİLEBİLİR SU (Yapay Zeka Eminlik Oranı: %{eminlik_skoru:.2f})"
    else:
        xgb_cikti = f"☠️ TEHLİKELİ / ZEHİRLİ SU (İçilebilirlik İhtimal Puanı: %{eminlik_skoru:.2f})"
        
    # --- K-Means Profillemesi ---
    girdi_scaled = scaler.transform(girdi_verisi)
    kume_id = kmeans_model.predict(girdi_scaled)[0]
    kmeans_cikti = kume_profilleri.get(kume_id, "Bilinmeyen Profil")
    
    return xgb_cikti, kmeans_cikti

# 3. Gradio Arayüzü İçin Dinamik Girdi Listesi Oluşturma
# CSV'deki her sütun için arayüze otomatik bir "Sayı Girdi Kutusu" ekliyoruz.
arayuz_girdileri = []
for kolon in sutun_isimleri:
    arayuz_girdileri.append(gr.Number(value=0.0, label=kolon.capitalize()))

# 4. Arayüzü Başlat
arayuz = gr.Interface(
    fn=su_kalitesi_tahmin_et,
    inputs=arayuz_girdileri,
    outputs=[
        gr.Textbox(label="XGBoost Sınıflandırma Kararı"),
        gr.Textbox(label="K-Means Doğal Segmentasyon Profili")
    ],
    title="🔬 Akıllı Su Kalitesi Analiz Laboratuvarı (Tam Kapsamlı Model)",
    description="Bu arayüz, laboratuvar ölçümlerindeki 20 kimyasal ve biyolojik boyutun tamamını değerlendirerek en hassas ve gerçekçi analizi yapar."
)

if __name__ == "__main__":
    arayuz.launch()