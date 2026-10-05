# 🔬 Sınıflandırma Algoritmaları Karşılaştırma Paneli

Breast Cancer Wisconsin veri seti üzerinde 6 sınıflandırma algoritmasını karşılaştıran küçük bir Streamlit uygulaması.

**Algoritmalar:** Lojistik Regresyon, KNN, Karar Ağacı, Random Forest, Naive Bayes, SVM

## Özellikler

- **Veri:** sınıf dağılımı, ilk satırlar, istatistiksel özet
- **Model sonuçları:** Accuracy, Precision, Recall, F1, confusion matrix, özellik önemleri, karar ağacı görseli
- **Karşılaştırma:** 6 algoritma yan yana (tablo + grafik), isteğe bağlı 5 katlı çapraz doğrulama
- **Tahmin:** test setinden bir hasta seçerek veya değerleri elle girerek canlı tahmin

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud ile yayınlama

1. Bu klasörü GitHub'a yükle (`app.py` ve `requirements.txt` ana dizinde olsun).
2. https://share.streamlit.io adresine GitHub hesabınla gir.
3. "New app" de, repoyu seç, ana dosya olarak `app.py` yaz ve "Deploy" de.

## Not

Precision, Recall ve F1 değerleri "kötü huylu" sınıfı pozitif kabul edilerek hesaplanır.
Uygulama eğitim amaçlıdır, tıbbi teşhis aracı değildir.
