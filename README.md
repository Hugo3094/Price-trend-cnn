# Price Trend Prediction — MLP vs LSTM vs CNN

> **Problématique :** Dans quelle mesure le choix de la représentation des données (tabulaire, séquentielle, visuelle) et de l'architecture associée (MLP, RNN/LSTM/GRU, CNN) influence-t-il la capacité à prédire les rendements boursiers ?

Implémentation et extension du papier **"(Re-)Imag(in)ing Price Trends"** (Jiang, Kelly & Xiu, *Journal of Finance*, 2023).

---

## Résultats clés

| Modèle | Représentation | Accuracy | AUC | Sharpe H-L (EW) |
|--------|---------------|----------|-----|-----------------|
| MLP    | Tabulaire (image scale) | ~52% | ~0.54 | ~1.2 |
| GRU    | Séquentielle | ~53% | ~0.55 | ~1.5 |
| LSTM   | Séquentielle | ~54% | ~0.56 | ~1.8 |
| **CNN**| **Images OHLC** | **~56%** | **~0.59** | **~3.1** |

> *Résultats obtenus sur un sous-ensemble de tickers S&P500 (2000–2022), fenêtre 20 jours, horizon 5 jours.*

---

## Architecture du projet

```
price-trend-dl/
├── data/
│   └── fetch_data.py        # Téléchargement Yahoo Finance, normalisation, labels
├── imaging/
│   └── ohlc_chart.py        # Génération d'images OHLC (fidèle au papier)
├── models/
│   ├── mlp.py               # MLP avec BatchNorm + Dropout
│   ├── lstm.py              # LSTM / GRU / Attention-LSTM
│   └── cnn.py               # CNN 2D + Grad-CAM
├── training/
│   └── train.py             # Pipeline unifié (early stopping, scheduler)
├── evaluation/
│   └── metrics.py           # Sharpe ratio, déciles, comparaison des modèles
├── notebooks/
│   └── results.ipynb        # Analyse complète et visualisations
└── requirements.txt
```

---

## Représentation des données

### 1. Tabulaire (MLP)
Les données OHLCV sont aplaties en un vecteur fixe. La normalisation **image scale** (max High = 1, min Low = 0) est cruciale — elle surpasse la normalisation par rendements cumulés.

### 2. Séquentielle (LSTM / GRU)
Les données sont traitées comme une séquence temporelle `(window, n_features)`. Le modèle apprend les dépendances temporelles via les états cachés.

### 3. Visuelle — Images OHLC (CNN)

Chaque fenêtre de prix est encodée comme une image noir et blanc suivant exactement la spécification du papier :

- **Fond noir**, objets blancs
- **3 pixels par jour** : barre haute-basse | marque ouverture | marque clôture
- **Volume** : 1/5 inférieur de l'image
- **Moyenne mobile** : tracée pixel par pixel (algorithme de Bresenham)
- **Normalisation** : max High → haut de l'image, min Low → bas

| Fenêtre | Dimensions |
|---------|-----------|
| 5 jours | 32 × 15 px |
| 20 jours | 64 × 60 px |
| 60 jours | 96 × 180 px |

```
Exemple d'image 20 jours (agrandie) :

████████████████████████████████████████████████
█ ┤  ┤  ┤           │  ┤  ┤  ┤  ┤  ┤  ┤  ┤  █
█ ┼──┼──┼──┼──┼─────┼──┼──┼──┼──┼──┼──┼──┼──█
█ ┤  ┤  ┤  ┤  ┤     ┤  ┤  ┤  ┤  ┤  ┤  ┤  ┤  █
█─────────────────────────────────────────────█
█ ▌▌ ▌▌  ▌ ▌  ▌ ▌▌▌  ▌ ▌▌ ▌▌ ▌▌ ▌▌ ▌  ▌▌  ▌ █  ← Volume
████████████████████████████████████████████████
```

---

## Axes d'analyse

### 1. Impact de la représentation
Comparaison directe MLP vs LSTM vs CNN avec les **mêmes données** et la **même normalisation**.
Résultat principal : l'image scale est le facteur dominant ; le CNN ajoute une couche de non-linéarité spatiale qui capte les relations entre prix, volatilité et volume simultanément.

### 2. Interprétabilité — Grad-CAM
Visualisation des zones de l'image OHLC qui activent le CNN :

- Les jours récents (t-1, t-2) sont les plus influents
- La position du Close par rapport au range High-Low est un signal fort
- Le volume élevé renforce les signaux directionnels

### 3. Transfer Learning
Un modèle entraîné sur des données US (5 jours) est appliqué directement à des données CAC40 sans ré-entraînement. Résultat cohérent avec le papier : le transfert surpasse le ré-entraînement local sur les marchés de petite taille.

### 4. Robustesse
- Fenêtres temporelles : 5j, 20j, 60j
- Ajout / retrait du volume
- Bruit sur les images (robustesse au MaxPooling)

---

## Installation

```bash
git clone https://github.com/Hugo3094/price-trend-dl.git
cd price-trend-dl
pip install -r requirements.txt
```

---

## Utilisation rapide

```python
from data.fetch_data import download_ohlcv, make_multi_stock_dataset
from imaging.ohlc_chart import make_image_dataset
from models.cnn import build_cnn
from training.train import Trainer, set_seed

set_seed(42)

# Données
raw = download_ohlcv(start="2010-01-01", end="2022-12-31")

# Dataset images
img_ds = make_image_dataset(raw, window=20, horizon=5)

# Modèle
cnn = build_cnn(window=20)

# Entraînement
trainer = Trainer(cnn, model_type="cnn", save_dir="checkpoints/cnn")
history = trainer.fit(
    img_ds["X_train"], img_ds["y_train"],
    img_ds["X_val"],   img_ds["y_val"],
    epochs=50, batch_size=64,
)
```

---

## Référence

```bibtex
@article{jiang2023reimagining,
  title   = {(Re-)Imag(in)ing Price Trends},
  author  = {Jiang, Jingwen and Kelly, Bryan and Xiu, Dacheng},
  journal = {The Journal of Finance},
  volume  = {78},
  number  = {6},
  pages   = {3193--3249},
  year    = {2023},
  doi     = {10.1111/jofi.13268}
}
```

---

## Auteurs

Mathieu Lang -- Mateo Molinaro -- Hugo Lecointre
