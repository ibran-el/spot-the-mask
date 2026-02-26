# 😷 Face Mask Classifier

> Binary image classification system to detect whether a person in an image is wearing a face mask.  
> Built for the **CMU-Africa Data Science Club Zindi Competition**.  
> Final Log Loss: **0.014**

---

## 🏆 Results

| Version | Description | Log Loss |
|---------|-------------|----------|
| v0.1 | EfficientNet-B0 baseline | 0.040 |
| v0.2 | Fine-tuning + Test Time Augmentation | 0.039 |
| v0.3 | Ensemble with ResNet50 | 0.036 |
| v0.4 | Weighted Ensemble (0.3 / 0.7) | **0.014** |

---

## 🧱 Architecture

OOP modular pipeline with clean separation of concerns.

```
face-mask-classifier/
├── src/
│   ├── data_loader.py     # Dataset, transforms, train/val/test splits
│   ├── model.py           # EfficientNet-B0 + ResNet50 classifiers
│   ├── trainer.py         # Training loop, early stopping, checkpointing
│   ├── predictor.py       # Inference, TTA, Ensemble predictor
│   └── submitter.py       # Formats and saves submission CSV
├── data/
│   ├── images/            # All train + test images (flat folder)
│   ├── train_labels.csv
│   └── SampleSubmission.csv
├── models/                # Saved model checkpoints
├── notebooks/
│   └── pipeline.ipynb     # Orchestration notebook
├── submissions/           # Generated submission CSVs
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/yourusername/face-mask-classifier.git
cd face-mask-classifier

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**requirements.txt**
```
torch
torchvision
pandas
numpy
scikit-learn
pillow
tqdm
```

---

## 📂 Data Setup

1. Download `images.zip`, `train_labels.csv`, `SampleSubmission.csv` from Zindi
2. Unzip `images.zip` into `data/images/`
3. Place CSV files in `data/`

```
data/
├── images/
│   ├── abc123.jpg
│   ├── xyz456.jpg
│   └── ...
├── train_labels.csv
└── SampleSubmission.csv
```

---

## 🚀 Running the Pipeline

Open `notebooks/pipeline.ipynb` and run cells in order:

| Cell | Action |
|------|--------|
| 1 | Setup & device check |
| 2 | Load & split data |
| 3 | Train EfficientNet-B0 |
| 4 | Baseline submission |
| 5 | Fine-tune EfficientNet |
| 6 | EfficientNet + TTA submission |
| 7 | Train ResNet50 |
| 8 | Fine-tune ResNet50 |
| 9 | Weighted ensemble + TTA submission |

---

## 🔬 Key Techniques

**Transfer Learning**
- EfficientNet-B0 and ResNet50 pretrained on ImageNet
- Fresh classification head trained first, backbone unfrozen with lower LR (3e-5) to prevent catastrophic forgetting

**Test Time Augmentation (TTA)**
- 4 augmented views per image at inference (original, horizontal flip, center crop, rotation)
- Predictions averaged → more robust probability estimates

**Weighted Ensemble**
- EfficientNet weight: `0.3` | ResNet50 weight: `0.7`
- Weights reflect relative model performance (lower val loss = higher trust)

**Log Loss Optimization**
- `BCEWithLogitsLoss` for numerical stability during training
- Probability clipping to `[1e-6, 1-1e-6]` in submission to prevent infinite loss

---

## 📊 Model Details

| Model | Params | Val Loss | Ensemble Weight |
|-------|--------|----------|-----------------|
| EfficientNet-B0 | ~5.3M | 0.07 | 0.3 |
| ResNet50 | ~25.6M | < 0.07 | 0.7 |

---

## 💡 Possible Extensions

- [ ] EfficientNet-B3/B4 for more capacity
- [ ] 5-Fold cross-validation ensemble
- [ ] Albumentations augmentation pipeline
- [ ] Label smoothing for better probability calibration
- [ ] Grad-CAM visualization of model attention

---

## 📁 Git History

```
v0.1.0 — feat: baseline EfficientNet-B0 mask classifier with OOP pipeline
v0.2.0 — feat: fine-tuning + TTA inference
v0.3.0 — feat: ensemble EfficientNet-B0 + ResNet50 with TTA
v0.4.0 — feat: weighted ensemble 0.3/0.7, logloss=0.014
v1.0.0 — chore: final cleanup, competition complete
```

---

## 🏫 Competition

**Host:** CMU-Africa Data Science Club  
**Platform:** Zindi  
**Metric:** Log Loss  
**Dataset:** ~1,800 images (1,300 train / 509 test)