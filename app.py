import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Kredi Kartı Dolandırıcılık Tespiti",
    page_icon="💳",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .result-normal {
        background: #064e3b; border: 2px solid #10b981;
        border-radius: 12px; padding: 1.2rem; text-align: center;
    }
    .result-fraud {
        background: #7f1d1d; border: 2px solid #ef4444;
        border-radius: 12px; padding: 1.2rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Gerçek örnekler ──────────────────────────────────────────────────────────
REAL_SAMPLES = {
    "🚨 #1 — Dolandırıcılık — 1.0€, saat 06:00": {
        "amount": 1.0, "time_hour": 6.0, "time_day": 0,
        "v4": 11.7862, "v10": -12.7448, "v12": -13.3779, "v14": -12.7843, "v17": -11.326,
        "label": 1
    },
    "🚨 #2 — Dolandırıcılık — 23.36€, saat 07:00": {
        "amount": 23.36, "time_hour": 7.0, "time_day": 0,
        "v4": 1.9219, "v10": -3.4657, "v12": -3.8267, "v14": -4.8529, "v17": -4.8356,
        "label": 1
    },
    "🚨 #3 — Dolandırıcılık — 1.0€, saat 03:00": {
        "amount": 1.0, "time_hour": 3.0, "time_day": 0,
        "v4": 10.3432, "v10": -11.8539, "v12": -17.2287, "v14": -18.4938, "v17": -14.4412,
        "label": 1
    },
    "🚨 #4 — Dolandırıcılık — 84.28€, saat 13:00": {
        "amount": 84.28, "time_hour": 13.0, "time_day": 1,
        "v4": 0.9874, "v10": -3.8411, "v12": -3.9069, "v14": -4.8982, "v17": -9.3343,
        "label": 1
    },
    "🚨 #5 — Dolandırıcılık — 254.76€, saat 21:00": {
        "amount": 254.76, "time_hour": 21.0, "time_day": 0,
        "v4": 4.2843, "v10": -6.8255, "v12": -6.9439, "v14": -6.947, "v17": -10.5643,
        "label": 1
    },
    "🚨 #6 — Dolandırıcılık — 1.0€, saat 02:00": {
        "amount": 1.0, "time_hour": 2.0, "time_day": 1,
        "v4": 6.5542, "v10": -14.5332, "v12": -15.8357, "v14": -11.567, "v17": -19.173,
        "label": 1
    },
    "🚨 #7 — Dolandırıcılık — 290.18€, saat 17:00": {
        "amount": 290.18, "time_hour": 17.0, "time_day": 0,
        "v4": 1.4012, "v10": -0.0914, "v12": -0.6138, "v14": 0.0161, "v17": -0.0573,
        "label": 1
    },
    "🚨 #8 — Dolandırıcılık — 1.0€, saat 02:00 (2)": {
        "amount": 1.0, "time_hour": 2.0, "time_day": 0,
        "v4": 4.0077, "v10": -4.625, "v12": -7.1482, "v14": -6.2103, "v17": -4.8303,
        "label": 1
    },
    "🚨 #9 — Dolandırıcılık — 59.68€, saat 11:00": {
        "amount": 59.68, "time_hour": 11.0, "time_day": 0,
        "v4": 8.6986, "v10": -12.6959, "v12": -11.9609, "v14": -9.8872, "v17": -21.7102,
        "label": 1
    },
    "🚨 #10 — Dolandırıcılık — 99.99€, saat 07:00": {
        "amount": 99.99, "time_hour": 7.0, "time_day": 0,
        "v4": 6.3201, "v10": -8.3728, "v12": -6.8728, "v14": -6.8266, "v17": -12.3753,
        "label": 1
    },
    "🚨 #11 — Dolandırıcılık — 99.99€, saat 07:00 (2)": {
        "amount": 99.99, "time_hour": 7.0, "time_day": 0,
        "v4": 6.4767, "v10": -7.8701, "v12": -5.6615, "v14": -5.0798, "v17": -11.275,
        "label": 1
    },
    "🚨 #12 — Dolandırıcılık — 0.83€, saat 15:00": {
        "amount": 0.83, "time_hour": 15.0, "time_day": 0,
        "v4": 5.1778, "v10": -8.3329, "v12": -8.7079, "v14": -9.5772, "v17": -12.5039,
        "label": 1
    },
    "🚨 #13 — Dolandırıcılık — 1.0€, saat 11:00": {
        "amount": 1.0, "time_hour": 11.0, "time_day": 1,
        "v4": 0.3947, "v10": -2.2977, "v12": -1.9444, "v14": -4.3463, "v17": -3.7465,
        "label": 1
    },
    "🚨 #14 — Dolandırıcılık — 112.45€, saat 14:00": {
        "amount": 112.45, "time_hour": 14.0, "time_day": 0,
        "v4": 2.238, "v10": -2.705, "v12": -4.3819, "v14": -2.9538, "v17": -5.5858,
        "label": 1
    },
    "🚨 #15 — Dolandırıcılık — 127.14€, saat 20:00": {
        "amount": 127.14, "time_hour": 20.0, "time_day": 1,
        "v4": 3.0444, "v10": -4.5335, "v12": -5.2672, "v14": -4.288, "v17": -8.7139,
        "label": 1
    },
    "✅ #16 — Normal — 2.58€, saat 22:00": {
        "amount": 2.58, "time_hour": 22.0, "time_day": 1,
        "v4": -0.7985, "v10": -0.2553, "v12": 0.9514, "v14": 0.1903, "v17": -0.8284,
        "label": 0
    },
    "✅ #17 — Normal — 11.04€, saat 17:00": {
        "amount": 11.04, "time_hour": 17.0, "time_day": 0,
        "v4": 0.7464, "v10": -0.0929, "v12": 0.9449, "v14": 0.1239, "v17": -0.2719,
        "label": 0
    },
    "✅ #18 — Normal — 1.78€, saat 11:00": {
        "amount": 1.78, "time_hour": 11.0, "time_day": 0,
        "v4": -0.1362, "v10": -0.5067, "v12": -0.4078, "v14": -0.4607, "v17": -0.291,
        "label": 0
    },
    "✅ #19 — Normal — 157.29€, saat 04:00": {
        "amount": 157.29, "time_hour": 4.0, "time_day": 1,
        "v4": 0.1917, "v10": -0.2805, "v12": -1.4615, "v14": 2.3689, "v17": 0.1279,
        "label": 0
    },
    "✅ #20 — Normal — 34.12€, saat 15:00": {
        "amount": 34.12, "time_hour": 15.0, "time_day": 0,
        "v4": 0.5841, "v10": 1.5966, "v12": 0.6041, "v14": -0.1726, "v17": -0.1379,
        "label": 0
    },
    "✅ #21 — Normal — 83.0€, saat 14:00": {
        "amount": 83.0, "time_hour": 14.0, "time_day": 0,
        "v4": -0.0681, "v10": -0.2406, "v12": 0.3594, "v14": 0.4331, "v17": -0.6502,
        "label": 0
    },
    "✅ #22 — Normal — 95.14€, saat 17:00": {
        "amount": 95.14, "time_hour": 17.0, "time_day": 0,
        "v4": 0.3354, "v10": -0.0587, "v12": 0.2397, "v14": 0.2111, "v17": -0.5083,
        "label": 0
    },
    "✅ #23 — Normal — 117.28€, saat 20:00": {
        "amount": 117.28, "time_hour": 20.0, "time_day": 0,
        "v4": 2.4153, "v10": 0.8946, "v12": -0.6934, "v14": 1.0615, "v17": -0.7653,
        "label": 0
    },
    "✅ #24 — Normal — 9.86€, saat 16:00": {
        "amount": 9.86, "time_hour": 16.0, "time_day": 0,
        "v4": -0.1642, "v10": -0.2429, "v12": -0.3509, "v14": 0.437, "v17": 0.1949,
        "label": 0
    },
    "✅ #25 — Normal — 25.2€, saat 18:00": {
        "amount": 25.2, "time_hour": 18.0, "time_day": 1,
        "v4": 0.2369, "v10": 1.3835, "v12": -0.9346, "v14": 0.2008, "v17": 0.4723,
        "label": 0
    },
    "✅ #26 — Normal — 5.99€, saat 22:00": {
        "amount": 5.99, "time_hour": 22.0, "time_day": 0,
        "v4": -0.0534, "v10": -0.487, "v12": -0.7325, "v14": -0.2834, "v17": -0.1359,
        "label": 0
    },
    "✅ #27 — Normal — 6.53€, saat 16:00": {
        "amount": 6.53, "time_hour": 16.0, "time_day": 0,
        "v4": 1.2077, "v10": -0.2181, "v12": 0.9663, "v14": -0.3822, "v17": -0.1714,
        "label": 0
    },
    "✅ #28 — Normal — 2.69€, saat 20:00": {
        "amount": 2.69, "time_hour": 20.0, "time_day": 1,
        "v4": -0.7313, "v10": -0.4051, "v12": 0.1991, "v14": 0.8125, "v17": -0.5207,
        "label": 0
    },
    "✅ #29 — Normal — 138.0€, saat 11:00": {
        "amount": 138.0, "time_hour": 11.0, "time_day": 1,
        "v4": 0.5406, "v10": 0.3721, "v12": -0.8656, "v14": -0.3348, "v17": -0.0757,
        "label": 0
    },
    "✅ #30 — Normal — 431.24€, saat 12:00": {
        "amount": 431.24, "time_hour": 12.0, "time_day": 1,
        "v4": -0.3519, "v10": 0.3379, "v12": 0.9309, "v14": -2.4412, "v17": 0.5717,
        "label": 0
    },
}



@st.cache_resource
def load_model():
    model  = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)

def predict(amount, time_hour, time_day, v4, v10, v12, v14, v17):
    features = np.zeros(31)
    features[3]  = v4
    features[9]  = v10
    features[11] = v12
    features[13] = v14
    features[16] = v17
    features[28] = np.log1p(amount)
    features[29] = time_hour
    features[30] = time_day
    X_scaled = scaler.transform(features.reshape(1, -1))
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    return pred, prob[1] * 100

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Dolandırıcılık Tespiti")
    st.markdown("---")
    page = st.radio("Sayfa", ["🔍 Tahmin Yap", "🧪 Gerçek Örnekler", "📊 Model Performansı", "ℹ️ Hakkında"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model:** Gradient Boosting")
    st.markdown("**PR-AUC:** 0.8754")
    st.markdown("**ROC-AUC:** 0.9703")
    st.markdown("**Recall:** %86.8")

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 1 — TAHMİN YAP
# ════════════════════════════════════════════════════════════════════════════
if page == "🔍 Tahmin Yap":
    st.title("🔍 İşlem Dolandırıcılık Tahmini")
    st.markdown("İşlem bilgilerini girin ve modelin tahminini görün.")

    if not model_loaded:
        st.error(f"Model yüklenemedi: {model_error}")
        st.stop()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Temel Bilgiler")
        amount    = st.number_input("İşlem Tutarı (€)", min_value=0.0, max_value=30000.0, value=50.0, step=0.01)
        time_hour = st.slider("İşlem Saati (0–23)", 0, 23, 14)
        time_day  = st.radio("Gün", [0, 1], format_func=lambda x: "1. Gün" if x == 0 else "2. Gün", horizontal=True)
    with col2:
        st.markdown("#### Önemli PCA Özellikleri")
        st.caption("Dolandırıcılık işlemlerinde V14 ve V17 genellikle çok negatif olur.")
        v14 = st.slider("V14", -20.0, 10.0, 0.0, 0.1)
        v17 = st.slider("V17", -25.0, 10.0, 0.0, 0.1)
        v12 = st.slider("V12", -20.0, 8.0,  0.0, 0.1)
        v10 = st.slider("V10", -20.0, 23.0, 0.0, 0.1)
        v4  = st.slider("V4",  -5.0,  17.0, 0.0, 0.1)

    st.markdown("---")
    if st.button("🚀 Tahmin Et", type="primary", use_container_width=True):
        pred, fraud_prob = predict(amount, time_hour, time_day, v4, v10, v12, v14, v17)
        normal_prob = 100 - fraud_prob

        r1, r2, r3 = st.columns(3)
        with r1:
            if pred == 0:
                st.markdown('<div class="result-normal"><div style="font-size:2rem">✅</div><div style="color:#6ee7b7;font-size:1.3rem;font-weight:700">NORMAL</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-fraud"><div style="font-size:2rem">🚨</div><div style="color:#fca5a5;font-size:1.3rem;font-weight:700">DOLANDIRICILIK</div></div>', unsafe_allow_html=True)
        with r2:
            st.metric("Dolandırıcılık Olasılığı", f"%{fraud_prob:.1f}")
        with r3:
            st.metric("Normal Olasılığı", f"%{normal_prob:.1f}")

        st.markdown("---")
        if fraud_prob < 10:
            st.success(f"✅ Düşük risk (%{fraud_prob:.1f}). İşlem büyük olasılıkla normaldir.")
        elif fraud_prob < 40:
            st.warning(f"⚠️ Orta risk (%{fraud_prob:.1f}). Manuel inceleme önerilir.")
        else:
            st.error(f"🚨 Yüksek risk (%{fraud_prob:.1f}). İşlem acilen incelenmelidir!")

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 2 — GERÇEK ÖRNEKLER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Gerçek Örnekler":
    st.title("🧪 Gerçek Veri Seti Örnekleri")
    st.markdown("Veri setinden alınan **gerçek işlemleri** test et. Modelin doğru tahmin edip etmediğini gör.")

    if not model_loaded:
        st.error(f"Model yüklenemedi: {model_error}")
        st.stop()

    st.markdown("---")

    secim = st.selectbox("Bir işlem seç:", list(REAL_SAMPLES.keys()))
    ornek = REAL_SAMPLES[secim]

    # Örnek detayları
    with st.expander("İşlem detayları", expanded=True):
        d1, d2, d3, d4, d5, d6, d7 = st.columns(7)
        d1.metric("Tutar", f"{ornek['amount']}€")
        d2.metric("Saat", f"{int(ornek['time_hour']):02d}:00")
        d3.metric("V14", f"{ornek['v14']:.2f}")
        d4.metric("V17", f"{ornek['v17']:.2f}")
        d5.metric("V12", f"{ornek['v12']:.2f}")
        d6.metric("V10", f"{ornek['v10']:.2f}")
        d7.metric("V4",  f"{ornek['v4']:.2f}")

    gercek = "🚨 Dolandırıcılık" if ornek["label"] == 1 else "✅ Normal"
    st.info(f"**Gerçek etiket:** {gercek}")

    if st.button("🤖 Modeli Çalıştır", type="primary", use_container_width=True):
        pred, fraud_prob = predict(
            ornek["amount"], ornek["time_hour"], ornek["time_day"],
            ornek["v4"], ornek["v10"], ornek["v12"], ornek["v14"], ornek["v17"]
        )
        normal_prob = 100 - fraud_prob

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        with r1:
            if pred == 0:
                st.markdown('<div class="result-normal"><div style="font-size:2rem">✅</div><div style="color:#6ee7b7;font-size:1.3rem;font-weight:700">NORMAL</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-fraud"><div style="font-size:2rem">🚨</div><div style="color:#fca5a5;font-size:1.3rem;font-weight:700">DOLANDIRICILIK</div></div>', unsafe_allow_html=True)
        with r2:
            st.metric("Dolandırıcılık Olasılığı", f"%{fraud_prob:.1f}")
        with r3:
            st.metric("Normal Olasılığı", f"%{normal_prob:.1f}")

        # Doğru mu yanlış mı?
        dogru = (pred == ornek["label"])
        if dogru:
            st.success("✅ Model doğru tahmin etti!")
        else:
            st.error("❌ Model yanlış tahmin etti. (Bu gerçek veri setinde de olabilir — %82 Recall)")

        # Neden dolandırıcılık / normal açıklaması
        st.markdown("---")
        st.markdown("#### 🔎 Neden bu karar?")
        if ornek["label"] == 1:
            st.markdown(f"""
            Bu işlem dolandırıcılık çünkü:
            - **V14 = {ornek['v14']:.2f}** → Dolandırıcılık işlemlerinde V14 genellikle **-5'in altında** olur
            - **V17 = {ornek['v17']:.2f}** → Dolandırıcılık işlemlerinde V17 genellikle **-4'ün altında** olur
            - **Tutar = {ornek['amount']}€** → Küçük tutarlar saldırganların radar altında kalmak için kullandığı bir yöntem
            """)
        else:
            st.markdown(f"""
            Bu işlem normal çünkü:
            - **V14 = {ornek['v14']:.2f}** → Normal işlemlerde V14 genellikle **-1 ile +3 arasında** kalır
            - **V17 = {ornek['v17']:.2f}** → Normal işlemlerde V17 genellikle **-1 ile +1 arasında** kalır
            - Özellik değerleri dolandırıcılık için tipik olan aşırı negatif değerler göstermiyor
            """)

    st.markdown("---")
    st.markdown("#### Tüm örneklerin özeti")
    tablo = []
    for isim, v in REAL_SAMPLES.items():
        pred, fp = predict(v["amount"], v["time_hour"], v["time_day"], v["v4"], v["v10"], v["v12"], v["v14"], v["v17"])
        dogru = "✅" if pred == v["label"] else "❌"
        tablo.append({
            "İşlem": isim,
            "Gerçek": "Dolandırıcılık" if v["label"] == 1 else "Normal",
            "Tahmin": "Dolandırıcılık" if pred == 1 else "Normal",
            "Risk %": f"%{fp:.1f}",
            "Sonuç": dogru
        })
    st.dataframe(pd.DataFrame(tablo), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 3 — MODEL PERFORMANSI
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performansı":
    st.title("📊 Model Performansı")

    st.markdown("### Final Model — Gradient Boosting (Test Seti)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("PR-AUC",   "0.8005")
    m2.metric("ROC-AUC",  "0.9645")
    m3.metric("F1 Macro", "0.7873")
    m4.metric("Recall",   "%82.1")
    m5.metric("Precision","%44.3")

    st.markdown("---")
    st.markdown("### Güçlü Rakip Karşılaştırması")
    df = pd.DataFrame({
        "Model": ["Gradient Boosting ⭐", "LightGBM", "RF (Tuned)", "XGBoost", "Logistic Regression", "Baseline (Dummy)"],
        "PR-AUC":  [0.8754, 0.8744, 0.8728, 0.8626, 0.8205, 0.0],
        "F1":      [0.8200, 0.9011, 0.9303, 0.8508, 0.5435, 0.50],
        "Recall":  [0.8684, 0.8553, 0.8553, 0.8684, 0.8947, 0.00],
        "ROC-AUC": [0.9703, 0.9690, 0.9736, 0.9691, 0.9764, 0.50],
    })
    st.dataframe(df.style.highlight_max(subset=["PR-AUC","F1","Recall","ROC-AUC"], color="#064e3b")
                   .format({"PR-AUC":"{:.4f}","F1":"{:.4f}","Recall":"{:.4f}","ROC-AUC":"{:.4f}"}),
                 use_container_width=True, hide_index=True)

    st.info("**Neden Accuracy kullanmıyoruz?** Veri setinin %99.83'ü normal. Her şeyi Normal tahmin eden Dummy model bile %99.83 Accuracy alıyor ama Recall = 0.")

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 4 — HAKKINDA
# ════════════════════════════════════════════════════════════════════════════
else:
    st.title("ℹ️ Proje Hakkında")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Proje Özeti
        Bir Avrupa bankasına ait gerçek kredi kartı işlem verisini kullanarak
        dolandırıcılığı tespit eden uçtan uca bir makine öğrenmesi sistemi.

        ### 📂 Veri Seti
        - **Kaynak:** Kaggle — ULB Machine Learning Group
        - **Boyut:** 284.807 işlem × 31 özellik
        - **Dolandırıcılık oranı:** %0.17 (492 işlem)

        ### ⚙️ Pipeline
        - Duplicate temizleme (1.081 satır)
        - RobustScaler + feature engineering
        - SMOTE (sadece train setinde)
        - Optuna Bayesian optimizasyon (30 trial)
        """)
    with col2:
        st.markdown("""
        ### 🔑 Neden V14 ve V17 önemli?
        | Özellik | Normal | Dolandırıcılık |
        |---------|--------|----------------|
        | V14 | -1 ile +3 arası | -5'in altı |
        | V17 | -1 ile +1 arası | -4'ün altı |
        | V12 | -1 ile +1 arası | -3'ün altı |

        Bu özellikler PCA ile anonimleştirilmiş — orijinal
        anlamları bilinmiyor ama dolandırıcılıkla güçlü korelasyon var.

        ### 📁 Dosyalar
        - `app.py` — Streamlit arayüzü
        - `model.pkl` — Gradient Boosting modeli
        - `scaler.pkl` — RobustScaler
        """)
