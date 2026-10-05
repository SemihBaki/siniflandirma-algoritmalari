"""
Sınıflandırma Algoritmaları Karşılaştırma Paneli
Veri seti: Breast Cancer Wisconsin (scikit-learn içinde hazır gelir)

Çalıştırmak için:  streamlit run app.py
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

st.set_page_config(page_title="Sınıflandırma Algoritmaları", page_icon="🔬", layout="wide")

# ----------------------------------------------------------------------------
# Sabitler
# ----------------------------------------------------------------------------
# load_breast_cancer içinde: 0 = malignant (kötü huylu), 1 = benign (iyi huylu)
SINIF_ADLARI = ["Kötü huylu", "İyi huylu"]
POZITIF_SINIF = 0  # Precision / Recall / F1 "kötü huylu" sınıfı için hesaplanır
METRIKLER = ["Accuracy", "Precision", "Recall", "F1"]

VARSAYILAN = {
    "Lojistik Regresyon": {"C": 1.0},
    "KNN": {"k": 5},
    "Karar Ağacı": {"max_depth": 0},  # 0 = sınırsız
    "Random Forest": {"n_estimators": 100},
    "Naive Bayes": {},
    "SVM": {"C": 1.0, "kernel": "rbf"},
}
ALGORITMALAR = list(VARSAYILAN)


# ----------------------------------------------------------------------------
# Veri ve model fonksiyonları
# ----------------------------------------------------------------------------
@st.cache_data
def veri_yukle():
    ds = load_breast_cancer(as_frame=True)
    return ds.data, ds.target


@st.cache_data
def bolme(test_orani, seed):
    X, y = veri_yukle()
    return train_test_split(X, y, test_size=test_orani, random_state=seed, stratify=y)


def model_olustur(ad, p, seed=42):
    if ad == "Lojistik Regresyon":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=2000))
    if ad == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=p["k"]))
    if ad == "Karar Ağacı":
        return DecisionTreeClassifier(max_depth=p["max_depth"] or None, random_state=seed)
    if ad == "Random Forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], random_state=seed)
    if ad == "Naive Bayes":
        return GaussianNB()
    if ad == "SVM":
        return make_pipeline(
            StandardScaler(),
            SVC(C=p["C"], kernel=p["kernel"], probability=True, random_state=seed),
        )
    raise ValueError(f"Bilinmeyen algoritma: {ad}")


def metrikler(y_gercek, tahmin):
    return {
        "Accuracy": accuracy_score(y_gercek, tahmin),
        "Precision": precision_score(y_gercek, tahmin, pos_label=POZITIF_SINIF),
        "Recall": recall_score(y_gercek, tahmin, pos_label=POZITIF_SINIF),
        "F1": f1_score(y_gercek, tahmin, pos_label=POZITIF_SINIF),
    }


@st.cache_data
def tum_modelleri_karsilastir(test_orani, seed):
    X_train, X_test, y_train, y_test = bolme(test_orani, seed)
    satirlar = []
    for ad in ALGORITMALAR:
        model = model_olustur(ad, VARSAYILAN[ad], seed)
        model.fit(X_train, y_train)
        satirlar.append({"Algoritma": ad, **metrikler(y_test, model.predict(X_test))})
    return pd.DataFrame(satirlar).sort_values("F1", ascending=False).reset_index(drop=True)


@st.cache_data
def capraz_dogrulama(seed):
    X, y = veri_yukle()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    skor = make_scorer(f1_score, pos_label=POZITIF_SINIF)
    satirlar = []
    for ad in ALGORITMALAR:
        model = model_olustur(ad, VARSAYILAN[ad], seed)
        puanlar = cross_val_score(model, X, y, cv=cv, scoring=skor)
        satirlar.append(
            {"Algoritma": ad, "F1 (ortalama)": puanlar.mean(), "F1 (std)": puanlar.std()}
        )
    return pd.DataFrame(satirlar).sort_values("F1 (ortalama)", ascending=False).reset_index(drop=True)


@st.cache_data
def onemli_ozellikler(n=6):
    X, y = veri_yukle()
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)
    onem = pd.Series(rf.feature_importances_, index=X.columns)
    return list(onem.nlargest(n).index)


def parametre_sec(ad):
    """Seçilen algoritmaya göre kenar çubuğunda ayar bileşenlerini gösterir."""
    p = dict(VARSAYILAN[ad])
    if ad == "Lojistik Regresyon":
        p["C"] = st.sidebar.select_slider(
            "C (düzenlileştirme gücünün tersi)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0
        )
    elif ad == "KNN":
        p["k"] = st.sidebar.slider("Komşu sayısı (k)", 1, 25, 5)
    elif ad == "Karar Ağacı":
        p["max_depth"] = st.sidebar.slider("Maksimum derinlik (0 = sınırsız)", 0, 15, 0)
    elif ad == "Random Forest":
        p["n_estimators"] = st.sidebar.slider("Ağaç sayısı", 10, 300, 100, step=10)
    elif ad == "SVM":
        p["C"] = st.sidebar.select_slider("C", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        p["kernel"] = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    else:
        st.sidebar.caption("Naive Bayes'te ayarlanacak önemli bir parametre yok.")
    return p


def sonucu_goster(model, satir, gercek=None):
    """Tek bir hasta için tahmin sonucunu ekrana yazar."""
    sinif = int(model.predict(satir)[0])
    if sinif == 0:
        st.error("Tahmin: **Kötü huylu**")
    else:
        st.success("Tahmin: **İyi huylu**")

    if hasattr(model, "predict_proba"):
        olasilik = model.predict_proba(satir)[0]  # sıra: [kötü huylu, iyi huylu]
        c1, c2 = st.columns(2)
        c1.metric("Kötü huylu olasılığı", f"%{olasilik[0] * 100:.1f}")
        c2.metric("İyi huylu olasılığı", f"%{olasilik[1] * 100:.1f}")

    if gercek is not None:
        dogru = int(gercek) == sinif
        durum = "✅ Model doğru bildi" if dogru else "❌ Model yanıldı"
        st.write(f"Gerçek sınıf: **{SINIF_ADLARI[int(gercek)]}** — {durum}")


# ----------------------------------------------------------------------------
# Sayfa başlığı ve kenar çubuğu
# ----------------------------------------------------------------------------
st.title("🔬 Sınıflandırma Algoritmaları Karşılaştırma Paneli")
st.caption(
    "Veri seti: Breast Cancer Wisconsin (569 hasta, 30 özellik). "
    "Amaç: bir tümörün kötü huylu mu iyi huylu mu olduğunu tahmin etmek."
)

X, y = veri_yukle()

st.sidebar.header("⚙️ Ayarlar")
secilen = st.sidebar.selectbox("Algoritma", ALGORITMALAR)
test_orani = st.sidebar.slider("Test seti oranı", 0.1, 0.5, 0.2, 0.05)
seed = int(st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42))
st.sidebar.subheader(f"{secilen} ayarları")
params = parametre_sec(secilen)
st.sidebar.divider()
st.sidebar.caption(
    "Precision, Recall ve F1 değerleri 'kötü huylu' sınıfı pozitif kabul edilerek hesaplanır."
)

X_train, X_test, y_train, y_test = bolme(test_orani, seed)
model = model_olustur(secilen, params, seed)
model.fit(X_train, y_train)
tahmin = model.predict(X_test)

sekme_veri, sekme_model, sekme_karsilastirma, sekme_tahmin = st.tabs(
    ["📊 Veri", "🎯 Model sonuçları", "⚖️ Karşılaştırma", "🩺 Tahmin et"]
)

# ----------------------------------------------------------------------------
# 1) Veri sekmesi
# ----------------------------------------------------------------------------
with sekme_veri:
    c1, c2, c3 = st.columns(3)
    c1.metric("Örnek sayısı", X.shape[0])
    c2.metric("Özellik sayısı", X.shape[1])
    c3.metric("Eğitim / Test", f"{len(X_train)} / {len(X_test)}")

    st.subheader("Sınıf dağılımı")
    dagilim = y.map(dict(enumerate(SINIF_ADLARI))).value_counts()
    st.bar_chart(dagilim)

    st.subheader("İlk 10 satır")
    st.dataframe(X.head(10))
    with st.expander("İstatistiksel özet"):
        st.dataframe(X.describe().T)

# ----------------------------------------------------------------------------
# 2) Seçilen modelin sonuçları
# ----------------------------------------------------------------------------
with sekme_model:
    st.subheader(f"{secilen} sonuçları")
    m = metrikler(y_test, tahmin)
    for sutun, (ad, deger) in zip(st.columns(4), m.items()):
        sutun.metric(ad, f"{deger:.3f}")

    sol, sag = st.columns(2)
    with sol:
        st.markdown("**Confusion matrix (test seti)**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, tahmin, display_labels=SINIF_ADLARI, cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        st.pyplot(fig)
        plt.close(fig)

    with sag:
        cm = confusion_matrix(y_test, tahmin, labels=[0, 1])
        st.markdown("**Nasıl okunur?**")
        st.write(f"- Kötü huylu olup doğru bilinen: **{cm[0, 0]}**")
        st.write(f"- Kötü huylu olup *kaçırılan*: **{cm[0, 1]}** (en tehlikeli hata)")
        st.write(f"- İyi huylu olup yanlışlıkla 'kötü' denen: **{cm[1, 0]}** (yanlış alarm)")
        st.write(f"- İyi huylu olup doğru bilinen: **{cm[1, 1]}**")
        st.info(
            "Bu problemde kötü huylu bir tümörü kaçırmak, yanlış alarm vermekten çok daha "
            "pahalıdır. Bu yüzden Accuracy yanında **Recall** değerine de bakmak gerekir."
        )

    if hasattr(model, "feature_importances_"):
        st.subheader("En önemli 10 özellik")
        onem = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
        st.bar_chart(onem)

    if secilen == "Karar Ağacı":
        with st.expander("🌳 Ağaç yapısı (ilk 3 seviye)"):
            fig, ax = plt.subplots(figsize=(16, 7))
            plot_tree(
                model,
                max_depth=3,
                feature_names=list(X.columns),
                class_names=SINIF_ADLARI,
                filled=True,
                fontsize=7,
                ax=ax,
            )
            st.pyplot(fig)
            plt.close(fig)

# ----------------------------------------------------------------------------
# 3) Tüm algoritmaların karşılaştırması
# ----------------------------------------------------------------------------
with sekme_karsilastirma:
    st.subheader("Tüm algoritmalar (varsayılan ayarlarla)")
    st.caption(
        f"Aynı veri bölünmesi kullanılır: test oranı %{int(test_orani * 100)}, "
        f"random state {seed}. Tablo F1 değerine göre sıralıdır."
    )
    df = tum_modelleri_karsilastir(test_orani, seed)
    st.dataframe(
        df.style.format({k: "{:.3f}" for k in METRIKLER}).highlight_max(
            subset=METRIKLER, color="#b7e4c7"
        ),
        hide_index=True,
    )
    st.bar_chart(df.set_index("Algoritma")[METRIKLER])

    en_iyi = df.iloc[0]
    st.success(f"Bu bölünmede en yüksek F1: **{en_iyi['Algoritma']}** ({en_iyi['F1']:.3f})")

    st.divider()
    st.markdown(
        "Tek bir train/test ayrımındaki küçük farklar şansa bağlı olabilir. "
        "Daha güvenilir bir karşılaştırma için çapraz doğrulama kullanılır."
    )
    if st.checkbox("5 katlı çapraz doğrulama ile karşılaştır"):
        cv_df = capraz_dogrulama(seed)
        st.dataframe(
            cv_df.style.format({"F1 (ortalama)": "{:.3f}", "F1 (std)": "{:.3f}"}),
            hide_index=True,
        )

# ----------------------------------------------------------------------------
# 4) Tahmin ekranı
# ----------------------------------------------------------------------------
with sekme_tahmin:
    st.subheader(f"Tahmin: {secilen}")
    st.caption("Kullanılan model, sol paneldeki ayarlarla eğitilmiş modeldir.")
    yontem = st.radio(
        "Giriş yöntemi",
        ["Test setinden bir hasta seç", "Değerleri elle gir"],
        horizontal=True,
    )

    if yontem == "Test setinden bir hasta seç":
        hasta = st.selectbox("Hasta no", list(X_test.index))
        satir = X_test.loc[[hasta]]
        with st.expander("Bu hastanın 30 özelliği"):
            st.dataframe(satir.T.rename(columns={hasta: "Değer"}))
        sonucu_goster(model, satir, gercek=y_test.loc[hasta])
    else:
        st.caption(
            "En önemli 6 özelliği değiştirebilirsin; diğer özellikler veri setinin ortalamasında sabit tutulur."
        )
        ozellikler = onemli_ozellikler(6)
        degerler = X.mean()
        sutunlar = st.columns(3)
        for i, ad in enumerate(ozellikler):
            alt, ust, ort = float(X[ad].min()), float(X[ad].max()), float(X[ad].mean())
            with sutunlar[i % 3]:
                degerler[ad] = st.slider(ad, alt, ust, ort, step=(ust - alt) / 200)
        satir = pd.DataFrame([degerler], columns=X.columns)
        sonucu_goster(model, satir)

    st.warning("Bu uygulama eğitim amaçlıdır, tıbbi teşhis aracı değildir.")
