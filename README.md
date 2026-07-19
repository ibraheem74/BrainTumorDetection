# 🧠 Brain Tumor Detection using Attention U-Net

A deep learning-based web application for **Brain Tumor Detection and Segmentation** from MRI images using the **Attention U-Net** architecture.

The application allows users to upload a Brain MRI image, predicts the tumor region, generates a segmentation mask, and displays the prediction result with a confidence score and tumor pixel count.

---

## ✨ Features

- 🧠 Brain MRI Tumor Detection
- 🎯 Tumor Segmentation using Attention U-Net
- 📤 Upload MRI Images through a Flask Web Interface
- 📊 Displays Prediction Confidence
- 📍 Tumor Pixel Count
- 🖼️ Segmentation Mask Generation
- 💻 Simple and User-Friendly Interface

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Flask
- OpenCV
- NumPy
- HTML
- CSS

---

## 🧠 Model Information

| Property | Value |
|----------|-------|
| Model | Attention U-Net |
| Input Size | 256 × 256 × 3 |
| Output | Binary Tumor Segmentation Mask |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy + Dice Loss |

---

# 📸 Application Screenshots

## Home Page

![Home Page](images/home.png)

---

## Prediction Result

![Prediction Result](images/prediction.png)

---

# 📂 Project Structure

```text
BrainTumorDetection/
│
├── app.py                  # Flask web application
├── train.py                # Train the Attention U-Net model
├── evaluate.py             # Evaluate trained model
├── predict.py              # Prediction script
├── plot_training.py        # Plot training graphs
│
├── models/
│   ├── attention_unet.py
│   ├── losses.py
│   └── metrics.py
│
├── utils/
│   ├── dataloader.py
│   └── preprocessing.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── uploads/
│   └── predictions/
│
├── images/
│   ├── home.png
│   └── prediction.png
│
├── saved_models/
│   └── best_attention_unet.keras (Generated after training)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📥 Dataset

This project uses the **LGG MRI Segmentation Dataset**, which contains Brain MRI images and corresponding tumor segmentation masks.

### Dataset Source

- **Kaggle Dataset:** <https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation>
- **Reference Notebook:** <https://www.kaggle.com/code/donottalk/lgg-mri-segmentation>

After downloading the dataset, organize it as follows:

```text
dataset/
├── train/
│   ├── images/
│   └── masks/
│
├── val/
│   ├── images/
│   └── masks/
│
└── test/
    ├── images/
    └── masks/
```

---

# 📥 Trained Model

The trained Attention U-Net model is **not included** in this repository because the model file exceeds GitHub's maximum file size limit (100 MB).

To use this project:

1. Download the dataset.
2. Train the model using `train.py`.
3. After training, place the generated model inside:

```text
saved_models/
└── best_attention_unet.keras
```

The Flask application automatically loads this model during prediction.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/ibraheem74/BrainTumorDetection.git
cd BrainTumorDetection
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Download the Dataset

Download the **LGG MRI Segmentation Dataset** from Kaggle and place it inside the project directory.

Example:

```text
dataset/
├── train/
├── val/
└── test/
```

---

## 4. Train the Model

Run the following command:

```bash
python train.py
```

After training, the model will be saved as:

```text
saved_models/
└── best_attention_unet.keras
```

---

## 5. Run the Flask Application

Start the Flask server:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

Upload a Brain MRI image and click **Predict**.

---

# 📊 Output

The application displays:

- Uploaded MRI Image
- Predicted Tumor Segmentation Mask
- Tumor Detection Result
- Confidence Score
- Tumor Pixel Count

---

# 🔮 Future Improvements

- Improve segmentation accuracy with additional training
- Support multiple MRI image formats
- Deploy the application on cloud platforms
- Add Grad-CAM visualization for explainable AI
- Develop a responsive mobile-friendly interface

---

# ⚠️ Important Notes

- The **dataset** is **not included** in this repository.
- The **trained model** (`best_attention_unet.keras`) is **not included** due to GitHub's file size limitation.
- Users must download the dataset and train the model before running the application.
- Ensure the trained model is placed inside the `saved_models/` directory before starting the Flask application.

---

# 👨‍💻 Author

**Ibraheem**

B.Tech Computer Science Engineering

Dr. M.G.R. Educational and Research Institute

**GitHub:** https://github.com/ibraheem74

---

# 📄 License

This project is developed for **educational and academic purposes**.

---

## ⭐ Support

If you found this project helpful, consider giving it a **⭐ Star** on GitHub.
