import gradio as gr
import pandas as pd
import joblib

# 1. Kaydedilen Modelleri ve Ölçeklendiriciyi Yükle
xgb_model = joblib.load('xgb_su_modeli.pkl')
kmeans_model = joblib.load('kmeans_su_modeli.pkl')
scaler = joblib.load('scaler_su.pkl')

# 2. K=4 Laboratuvar Analizimize Göre Küme Profili Açıklamaları
kume_profilleri = {
    0: "Grup 0: Genel Karışık / Kalitesiz Su",
    1: "Grup 1: Klorlama Gerekli",
    2: "Grup 2: Görece İçilebilir (Riskli)",
    3: "Grup 3: Ağır Metal Riski - Kimyasal Arıtma Şart"
}

# 3. Tahmin Fonksiyonu
def su_kalitesi_tahmin_et(cadmium, aluminium, silver, perchlorate, uranium):
    # Girdileri yapay zekanın anlayacağı DataFrame yapısına getirme
    girdi_verisi = pd.DataFrame([{
        'cadmium': cadmium,
        'aluminium': aluminium,
        'silver': silver,
        'perchlorate': perchlorate,
        'uranium': uranium
    }])
    
    # --- XGBoost Tahmin Operasyonu ---
    karar = xgb_model.predict(girdi_verisi)[0]
    eminlik_skoru = xgb_model.predict_proba(girdi_verisi)[0][1] * 100
    
    if karar == 1:
        xgb_cikti = f"✅ GÜVENLİ / İÇİLEBİLİR SU (Yapay Zeka Eminlik Oranı: %{eminlik_skoru:.2f})"
    else:
        xgb_cikti = f"☠️ TEHLİKELİ / ZEHİRLİ SU (İçilebilirlik İhtimal Puanı: %{eminlik_skoru:.2f})"
        
    # --- K-Means Kümeleme Operasyonu ---
    # Girdiyi önce eğitildiği scaler ile dönüştürüyoruz
    girdi_scaled = scaler.transform(girdi_verisi)
    kume_id = kmeans_model.predict(girdi_scaled)[0]
    kmeans_cikti = kume_profilleri.get(kume_id, "Bilinmeyen Profil")
    
    return xgb_cikti, kmeans_cikti

# 4. Gradio Arayüz Tasarımı (Görsel ve Fonksiyonel Kontroller)
arayuz = gr.Interface(
    fn=su_kalitesi_tahmin_et,
    inputs=[
        gr.Slider(minimum=0.0, maximum=0.5, step=0.001, value=0.005, label="Kadmiyum (Cadmium)"),
        gr.Slider(minimum=0.0, maximum=10, step=0.01, value=0.10, label="Alüminyum (Aluminium)"),
        gr.Slider(minimum=0.0, maximum=0.6, step=0.01, value=0.02, label="Gümüş (Silver)"),
        gr.Slider(minimum=0.0, maximum=100, step=0.5, value=15.0, label="Perklorat (Perchlorate)"),
        gr.Slider(minimum=0.0, maximum=0.2, step=0.001, value=0.01, label="Uranyum (Uranium)")
    ],
    outputs=[
        gr.Textbox(label="XGBoost Sınıflandırma Kararı ve Güven Skoru"),
        gr.Textbox(label="K-Means Doğal Kimyasal Kümeleme Profili")
    ],
    title="🔬 Akıllı Su Kalitesi Analiz ve Profilleme Laboratuvarı",
    description="En kritik 5 kimyasal parametreyi değiştirerek suyun hem anlık güvenlik durumunu (XGBoost) test edebilir, hem de hiçbir etiket olmadan arka planda hangi doğal su profiline (K-Means) dahil olduğunu inceleyebilirsiniz."
)

# Arayüzü Başlatma
if __name__ == "__main__":
    arayuz.launch()