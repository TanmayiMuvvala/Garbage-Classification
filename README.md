# Garbage-Classification
# 🗑️ Garbage Classification using MobileNetV2 (Transfer Learning)

This project uses **MobileNetV2**, a powerful pre-trained CNN architecture, to classify garbage images into 6 categories:

- **cardboard**
- **glass**
- **metal**
- **paper**
- **plastic**
- **trash**

It aims to help automate waste sorting, which is important for efficient recycling and environmental sustainability.

---

## 📁 Dataset

The dataset used is the **TrashType Image Dataset**, which includes images of garbage items categorized into folders named after their respective classes.

The dataset was split into `train/` and `test/` folders using an 80:20 ratio, ensuring a balanced and reproducible split.

---

## 🧠 Model: MobileNetV2

We used **MobileNetV2** from TensorFlow Keras with the following steps:

1. **Base Model**: MobileNetV2 with `weights="imagenet"` and `include_top=False`
2. **Added Layers**:
   - Global Average Pooling
   - Dropout
   - Dense (ReLU) + Dropout
   - Output Layer (Softmax with 6 units)
3. **Freezing Base Layers**: Initially, all MobileNetV2 layers are frozen
4. **Fine-tuning**: Last 30 layers of the base model were unfrozen and retrained

---

## ⚙️ Training

- Initial Training: 10 epochs
- Fine-tuning: 10 more epochs with:
  - **Lower learning rate**: `1e-5`
  - **Label smoothing**: `0.1`
  - **Callbacks**:
    - `EarlyStopping` (to prevent overfitting)
    - `ModelCheckpoint` (to save best model)
    - `ReduceLROnPlateau` (for dynamic learning rate tuning)

---

## 📈 Results

- ✅ **Test Accuracy**: ~85.8%
- ✅ **Test Loss**: ~0.42
- Confusion Matrix and Classification Report included
- Training/Validation Accuracy and Loss graphs plotted

---

## 🧪 How to Use

1. Clone this repository or run the Jupyter Notebook.
2. Place the dataset in the proper structure (`train/`, `test/` under class folders).
3. Run the notebook to train and fine-tune the model.
4. Use `model.predict()` on new images.

---

## 🗂️ Folder Structure

```
TrashType_Image_Dataset/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── test/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

---

## 🧾 Requirements

- TensorFlow
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

---

## 📥 Model Files

- `best_mobilenet_model.keras` – Best model during initial training
- `fine_tuned_mobilenet.keras` – Final model after fine-tuning

---

## 📌 Future Work

- Integrate Gradio or Streamlit for live image classification
- Deploy as a web or mobile application
- Add data augmentation and more robust evaluation

---

## 🙋‍♀️ Created By

**Tanmayi Muvvala**  
B.Tech Student | AI & ML Enthusiast  
Inspired by a passion for technology and sustainability.

---

## 🌱 License

This project is open-source and free to use for academic and research purposes.
