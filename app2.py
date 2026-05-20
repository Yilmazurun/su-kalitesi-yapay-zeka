import html

import gradio as gr
import joblib
import pandas as pd


# Tam kapsamli modeller ve egitimde kullanilan sutun sirasi.
xgb_model = joblib.load("xgb_su_modeli_full.pkl")
kmeans_model = joblib.load("kmeans_su_modeli_full.pkl")
scaler = joblib.load("scaler_su_full.pkl")
sutun_isimleri = joblib.load("tam_sutun_isimleri.pkl")


PARAMETRELER = {
    "aluminium": {
        "label": "Alüminyum",
        "unit": "mg/L",
        "min": 0.0,
        "max": 5.0,
        "step": 0.01,
        "value": 0.10,
    },
    "ammonia": {
        "label": "Amonyak",
        "unit": "mg/L",
        "min": 0.0,
        "max": 30.0,
        "step": 0.05,
        "value": 0.20,
    },
    "arsenic": {
        "label": "Arsenik",
        "unit": "mg/L",
        "min": 0.0,
        "max": 1.1,
        "step": 0.001,
        "value": 0.01,
    },
    "barium": {
        "label": "Baryum",
        "unit": "mg/L",
        "min": 0.0,
        "max": 5.0,
        "step": 0.01,
        "value": 0.20,
    },
    "cadmium": {
        "label": "Kadmiyum",
        "unit": "mg/L",
        "min": 0.0,
        "max": 0.15,
        "step": 0.001,
        "value": 0.005,
    },
    "chloramine": {
        "label": "Kloramin",
        "unit": "mg/L",
        "min": 0.0,
        "max": 9.0,
        "step": 0.01,
        "value": 1.00,
    },
    "chromium": {
        "label": "Krom",
        "unit": "mg/L",
        "min": 0.0,
        "max": 1.0,
        "step": 0.001,
        "value": 0.02,
    },
    "copper": {
        "label": "Bakır",
        "unit": "mg/L",
        "min": 0.0,
        "max": 2.0,
        "step": 0.01,
        "value": 0.05,
    },
    "flouride": {
        "label": "Florür",
        "unit": "mg/L",
        "min": 0.0,
        "max": 1.6,
        "step": 0.01,
        "value": 0.70,
    },
    "bacteria": {
        "label": "Bakteri indeksi",
        "unit": "0-1",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "value": 0.00,
    },
    "viruses": {
        "label": "Virüs indeksi",
        "unit": "0-1",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "value": 0.00,
    },
    "lead": {
        "label": "Kurşun",
        "unit": "mg/L",
        "min": 0.0,
        "max": 0.25,
        "step": 0.001,
        "value": 0.005,
    },
    "nitrates": {
        "label": "Nitrat",
        "unit": "mg/L",
        "min": 0.0,
        "max": 20.0,
        "step": 0.05,
        "value": 4.00,
    },
    "nitrites": {
        "label": "Nitrit",
        "unit": "mg/L",
        "min": 0.0,
        "max": 3.0,
        "step": 0.01,
        "value": 0.10,
    },
    "mercury": {
        "label": "Cıva",
        "unit": "mg/L",
        "min": 0.0,
        "max": 0.02,
        "step": 0.0001,
        "value": 0.001,
    },
    "perchlorate": {
        "label": "Perklorat",
        "unit": "mg/L",
        "min": 0.0,
        "max": 100.0,
        "step": 0.1,
        "value": 3.00,
    },
    "radium": {
        "label": "Radyum",
        "unit": "pCi/L",
        "min": 0.0,
        "max": 8.0,
        "step": 0.01,
        "value": 0.20,
    },
    "selenium": {
        "label": "Selenyum",
        "unit": "mg/L",
        "min": 0.0,
        "max": 0.15,
        "step": 0.001,
        "value": 0.01,
    },
    "silver": {
        "label": "Gümüş",
        "unit": "mg/L",
        "min": 0.0,
        "max": 0.6,
        "step": 0.01,
        "value": 0.02,
    },
    "uranium": {
        "label": "Uranyum",
        "unit": "mg/L",
        "min": 0.0,
        "max": 0.12,
        "step": 0.001,
        "value": 0.01,
    },
}

PARAMETRE_GRUPLARI = [
    (
        "Metaller",
        [
            "aluminium",
            "arsenic",
            "barium",
            "cadmium",
            "chromium",
            "copper",
            "lead",
            "mercury",
            "selenium",
            "silver",
            "uranium",
        ],
    ),
    ("Biyolojik", ["bacteria", "viruses", "chloramine"]),
    (
        "Mineral ve Anyonlar",
        ["ammonia", "flouride", "nitrates", "nitrites", "perchlorate", "radium"],
    ),
]

KUME_PROFILLERI = {
    0: {
        "baslik": "Genel karışık / kalite düşük",
        "metin": "Birden fazla parametrede kaliteyi düşüren karma bir profil.",
        "etiket": "Öncelik: geniş tarama",
        "renk": "amber",
    },
    1: {
        "baslik": "Biyolojik tehlike alanı",
        "metin": "Bakteri, virüs veya dezenfeksiyon göstergeleri öne çıkıyor.",
        "etiket": "Öncelik: biyolojik arıtma",
        "renk": "danger",
    },
    2: {
        "baslik": "Görece düşük riskli ön eleme",
        "metin": "Model bu örneği daha dengeli ve düşük riskli profile yakın buldu.",
        "etiket": "Öncelik: rutin izleme",
        "renk": "safe",
    },
    3: {
        "baslik": "Ağır metal kokteyli",
        "metin": "Metal kaynaklı kimyasal riskler bu segmentte belirginleşiyor.",
        "etiket": "Öncelik: kimyasal arıtma",
        "renk": "danger",
    },
}

CUSTOM_CSS = """
:root {
    --ink: #0f172a;
    --muted: #5b6b7a;
    --line: rgba(15, 118, 110, 0.16);
    --panel: rgba(255, 255, 255, 0.92);
    --panel-strong: #ffffff;
    --water: #0891b2;
    --water-deep: #075985;
    --mint: #14b8a6;
    --safe: #0f9f6e;
    --danger: #dc4c3f;
    --amber: #c78112;
}

.gradio-container {
    min-height: 100vh;
    color: var(--ink);
    background:
        linear-gradient(180deg, rgba(236, 253, 255, 0.96), rgba(247, 254, 252, 0.98)),
        repeating-linear-gradient(135deg, rgba(8, 145, 178, 0.08) 0 1px, transparent 1px 24px);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.gradio-container .contain {
    max-width: none;
}

#app-shell {
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 18px 40px;
}

#hero {
    position: relative;
    overflow: hidden;
    min-height: 210px;
    padding: 34px;
    border: 1px solid rgba(255, 255, 255, 0.34);
    border-radius: 8px;
    color: #f8fcff;
    background:
        linear-gradient(135deg, rgba(7, 89, 133, 0.96), rgba(15, 118, 110, 0.88)),
        repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.08) 0 1px, transparent 1px 18px);
    box-shadow: 0 24px 70px rgba(7, 89, 133, 0.22);
}

#hero::after {
    content: "";
    position: absolute;
    right: -6%;
    bottom: -38px;
    width: 70%;
    height: 120px;
    opacity: 0.45;
    background:
        radial-gradient(ellipse at 50% 100%, rgba(255, 255, 255, 0.42) 0, rgba(255, 255, 255, 0.18) 32%, transparent 34%),
        repeating-linear-gradient(170deg, rgba(255, 255, 255, 0.18) 0 2px, transparent 2px 16px);
}

#hero .eyebrow {
    margin: 0 0 12px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
    color: #bff4ff;
}

#hero h1 {
    position: relative;
    z-index: 1;
    max-width: 760px;
    margin: 0;
    font-size: 2.65rem;
    line-height: 1.05;
    letter-spacing: 0;
}

#hero p {
    position: relative;
    z-index: 1;
    max-width: 720px;
    margin: 16px 0 0;
    color: rgba(248, 252, 255, 0.84);
    font-size: 1rem;
    line-height: 1.65;
}

#metric-strip {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    min-height: 34px;
    padding: 7px 12px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 999px;
    color: #e8fbff;
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(10px);
    font-size: 0.88rem;
}

#workspace {
    align-items: stretch;
    gap: 18px;
    margin-top: 18px;
}

.surface {
    padding: 20px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel);
    box-shadow: 0 18px 55px rgba(7, 89, 133, 0.12);
}

.panel-heading {
    margin-bottom: 14px;
}

.panel-heading h2 {
    margin: 0;
    color: var(--ink);
    font-size: 1.15rem;
    line-height: 1.2;
    letter-spacing: 0;
}

.panel-heading p {
    margin: 6px 0 0;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.5;
}

.gradio-container .tabs {
    border: 0;
}

.gradio-container .tab-nav {
    gap: 8px;
    border-bottom: 1px solid var(--line);
}

.gradio-container button[role="tab"] {
    border-radius: 8px 8px 0 0;
    color: var(--muted);
    font-weight: 700;
}

.gradio-container button[role="tab"][aria-selected="true"] {
    color: var(--water-deep);
    border-bottom-color: var(--mint);
}

.gradio-container .block,
.gradio-container .form {
    border-color: var(--line);
    border-radius: 8px;
}

.gradio-container label,
.gradio-container .label-wrap {
    color: var(--ink);
    font-weight: 700;
}

.gradio-container input,
.gradio-container textarea {
    border-radius: 8px;
}

.gradio-container input[type="range"] {
    accent-color: var(--water);
}

#action-row {
    gap: 10px;
    margin-top: 16px;
}

#analyze-button {
    min-height: 44px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--water), var(--mint));
    border: 0;
    color: white;
    font-weight: 800;
}

#reset-button {
    min-height: 44px;
    border-radius: 8px;
    color: var(--water-deep);
    border: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.7);
    font-weight: 800;
}

.results-surface {
    position: sticky;
    top: 18px;
}

.result-card {
    margin-top: 14px;
    padding: 18px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel-strong);
}

.result-card.safe {
    border-left: 5px solid var(--safe);
}

.result-card.danger {
    border-left: 5px solid var(--danger);
}

.result-card.amber {
    border-left: 5px solid var(--amber);
}

.result-kicker {
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
}

.result-title {
    margin-top: 8px;
    color: var(--ink);
    font-size: 1.45rem;
    font-weight: 850;
    line-height: 1.2;
}

.result-text {
    margin-top: 8px;
    color: var(--muted);
    font-size: 0.95rem;
    line-height: 1.55;
}

.score-row {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    margin-top: 16px;
    color: var(--muted);
    font-weight: 700;
}

.score-row strong {
    color: var(--ink);
    white-space: nowrap;
}

.score-track {
    overflow: hidden;
    height: 10px;
    margin-top: 8px;
    border-radius: 999px;
    background: #dceff3;
}

.score-fill {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--water), var(--mint));
}

.danger .score-fill {
    background: linear-gradient(90deg, var(--danger), var(--amber));
}

.profile-tag {
    display: inline-flex;
    margin-top: 14px;
    padding: 7px 10px;
    border-radius: 999px;
    color: var(--water-deep);
    background: #e2f8fb;
    font-size: 0.86rem;
    font-weight: 800;
}

.danger .profile-tag {
    color: #8f2018;
    background: #ffe8e5;
}

.amber .profile-tag {
    color: #7a4b00;
    background: #fff3d7;
}

footer {
    display: none !important;
}

@media (max-width: 900px) {
    #workspace {
        flex-direction: column;
    }

    .results-surface {
        position: static;
    }
}

@media (max-width: 640px) {
    #app-shell {
        padding: 16px 12px 28px;
    }

    #hero {
        min-height: 0;
        padding: 24px 20px;
    }

    #hero h1 {
        font-size: 2rem;
    }

    .surface {
        padding: 16px;
    }

    .result-title {
        font-size: 1.25rem;
    }
}
"""


def varsayilan_degerler():
    return [PARAMETRELER[kolon]["value"] for kolon in sutun_isimleri]


def guven_skoru_hesapla(girdi_verisi):
    olasiliklar = xgb_model.predict_proba(girdi_verisi)[0]
    siniflar = list(getattr(xgb_model, "classes_", []))
    try:
        guvenli_indeks = siniflar.index(1)
    except ValueError:
        guvenli_indeks = -1
    return float(olasiliklar[guvenli_indeks] * 100)


def analiz_karti(karar, eminlik_skoru):
    guvenli = int(karar) == 1
    varyant = "safe" if guvenli else "danger"
    baslik = "Güvenli / içilebilir su" if guvenli else "Riskli / içilemez su"
    metin = (
        "Model, verilen ölçümlerin içilebilir profile yakın olduğunu görüyor."
        if guvenli
        else "Model, verilen ölçümlerde içilebilirlik açısından belirgin risk görüyor."
    )
    skor = max(0.0, min(100.0, eminlik_skoru))

    return f"""
    <div class="result-card {varyant}">
        <div class="result-kicker">XGBoost sınıflandırması</div>
        <div class="result-title">{html.escape(baslik)}</div>
        <div class="result-text">{html.escape(metin)}</div>
        <div class="score-row">
            <span>İçilebilirlik olasılığı</span>
            <strong>%{skor:.2f}</strong>
        </div>
        <div class="score-track">
            <span class="score-fill" style="width: {skor:.2f}%"></span>
        </div>
    </div>
    """


def profil_karti(kume_id):
    profil = KUME_PROFILLERI.get(
        int(kume_id),
        {
            "baslik": "Bilinmeyen profil",
            "metin": "Bu ölçüm kümesi kayıtlı profillerle eşleşmedi.",
            "etiket": "Manuel inceleme",
            "renk": "amber",
        },
    )

    return f"""
    <div class="result-card {html.escape(profil['renk'])}">
        <div class="result-kicker">K-Means doğal segmentasyon</div>
        <div class="result-title">Grup {int(kume_id)} · {html.escape(profil['baslik'])}</div>
        <div class="result-text">{html.escape(profil['metin'])}</div>
        <div class="profile-tag">{html.escape(profil['etiket'])}</div>
    </div>
    """


def su_kalitesi_tahmin_et(*args):
    if len(args) != len(sutun_isimleri):
        raise ValueError("Girdi sayısı modelin beklediği sütun sayısıyla eşleşmiyor.")

    girdi_sozlugu = {sutun_isimleri[i]: [args[i]] for i in range(len(sutun_isimleri))}
    girdi_verisi = pd.DataFrame(girdi_sozlugu)

    karar = xgb_model.predict(girdi_verisi)[0]
    eminlik_skoru = guven_skoru_hesapla(girdi_verisi)

    girdi_scaled = scaler.transform(girdi_verisi)
    kume_id = kmeans_model.predict(girdi_scaled)[0]

    return analiz_karti(karar, eminlik_skoru), profil_karti(kume_id)


def varsayilanlara_don():
    degerler = varsayilan_degerler()
    return (*degerler, *su_kalitesi_tahmin_et(*degerler))


def parametre_bileseni(kolon):
    meta = PARAMETRELER[kolon]
    return gr.Slider(
        minimum=meta["min"],
        maximum=meta["max"],
        step=meta["step"],
        value=meta["value"],
        label=f"{meta['label']} ({meta['unit']})",
    )


def ikili_satirlar(liste):
    for indeks in range(0, len(liste), 2):
        yield liste[indeks : indeks + 2]


tema = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="teal",
    neutral_hue="slate",
    radius_size="sm",
)

ilk_sonuclar = su_kalitesi_tahmin_et(*varsayilan_degerler())

with gr.Blocks(title="Akıllı Su Kalitesi Analiz Laboratuvarı") as arayuz:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(
            """
            <section id="hero">
                <p class="eyebrow">Su Kalitesi Yapay Zeka Laboratuvarı</p>
                <h1>Akıllı Su Kalitesi Analizi</h1>
                <p>Kimyasal, biyolojik ve mineral ölçümleri tek ekranda değerlendirip içilebilirlik kararını ve doğal su profilini üretir.</p>
                <div id="metric-strip">
                    <span class="metric-pill">20 laboratuvar parametresi</span>
                    <span class="metric-pill">XGBoost karar modeli</span>
                    <span class="metric-pill">K-Means profil analizi</span>
                </div>
            </section>
            """
        )

        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=7, elem_classes=["surface"]):
                gr.HTML(
                    """
                    <div class="panel-heading">
                        <h2>Ölçüm Değerleri</h2>
                        <p>Parametreler analiz türlerine göre ayrıldı.</p>
                    </div>
                    """
                )

                bilesenler = {}
                with gr.Tabs():
                    for grup_adi, kolonlar in PARAMETRE_GRUPLARI:
                        with gr.Tab(grup_adi):
                            for satir in ikili_satirlar(kolonlar):
                                with gr.Row():
                                    for kolon in satir:
                                        with gr.Column():
                                            bilesenler[kolon] = parametre_bileseni(kolon)

                arayuz_girdileri = [bilesenler[kolon] for kolon in sutun_isimleri]

                with gr.Row(elem_id="action-row"):
                    analiz_butonu = gr.Button(
                        "Analiz et", variant="primary", elem_id="analyze-button"
                    )
                    reset_butonu = gr.Button(
                        "Varsayılanlar", variant="secondary", elem_id="reset-button"
                    )

            with gr.Column(scale=5, elem_classes=["surface", "results-surface"]):
                gr.HTML(
                    """
                    <div class="panel-heading">
                        <h2>Sonuç Paneli</h2>
                        <p>Model çıktıları güven skoru ve doğal profil olarak ayrıştırılır.</p>
                    </div>
                    """
                )
                xgb_cikti = gr.HTML(value=ilk_sonuclar[0])
                kmeans_cikti = gr.HTML(value=ilk_sonuclar[1])

        analiz_butonu.click(
            fn=su_kalitesi_tahmin_et,
            inputs=arayuz_girdileri,
            outputs=[xgb_cikti, kmeans_cikti],
        )
        reset_butonu.click(
            fn=varsayilanlara_don,
            inputs=None,
            outputs=[*arayuz_girdileri, xgb_cikti, kmeans_cikti],
        )


if __name__ == "__main__":
    arayuz.launch(theme=tema, css=CUSTOM_CSS)
