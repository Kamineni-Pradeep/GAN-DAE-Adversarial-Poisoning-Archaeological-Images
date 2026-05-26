# Mitigating Iterative Fast Gradient Adversarial Poisoning in Archaeological Image Datasets Using GANs and Denoising Autoencoders

> 📄 **Published at ICEARS2025 Conference** — December 2025

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Overview

Archaeological image datasets are increasingly used to train deep learning models for heritage classification and site analysis. However, these datasets are vulnerable to **adversarial poisoning attacks** — subtle pixel-level perturbations that cause models to misclassify images while remaining visually imperceptible to humans.

This project presents a defense system that:
- Generates adversarial examples using **Iterative Fast Gradient Sign Method (IFGSM)**
- Detects poisoned images using a **Denoising Autoencoder (DAE)**
- Validates detection using a **Generative Adversarial Network (GAN)** discriminator
- Provides a **Flask web application** for real-time image poisoning detection

---
<img width="300" height="234" alt="image" src="https://github.com/user-attachments/assets/be2004a3-cebc-4c40-b48f-52e994712a9c" /> <img width="300" height="234" alt="image" src="https://github.com/user-attachments/assets/ee9d187c-d86f-4373-90c1-6eb82d9fd916" /> <img width="300" height="234" alt="image" src="https://github.com/user-attachments/assets/102fe221-4686-46ba-81ed-36aba00c491e" />



## 🏗️ System Architecture

```
Input Images
     │
     ▼
┌─────────────────────┐
│  IFGSM Attack       │  ← Generates adversarial perturbations
│  (epsilon=0.1,      │
│   iterations=10)    │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Denoising          │     │  GAN Discriminator  │
│  Autoencoder (DAE)  │     │                     │
│  Reconstruction     │     │  Real/Fake Score    │
│  Error              │     │                     │
└─────────────────────┘     └─────────────────────┘
     │                              │
     └──────────────┬───────────────┘
                    ▼
          Poisoning Detection
          (Threshold-based)
                    │
                    ▼
            Flask Web App
          Results Dashboard
```

---

## 🔬 How It Works

### 1. Adversarial Attack — IFGSM
The **Iterative Fast Gradient Sign Method** generates adversarial examples by iteratively perturbing images in the direction that maximises reconstruction loss:

```
x_adv = x + ε × sign(∇ₓ L(DAE(x), x))
```
Repeated for `n` iterations with pixel values clipped to [0, 1].

### 2. Detection — Denoising Autoencoder
The DAE is trained to reconstruct clean images. Adversarially poisoned images produce a **higher reconstruction error** because their perturbations don't follow natural image patterns:
- Reconstruction error **> 0.05** → flagged as potentially poisoned

### 3. Validation — GAN Discriminator
The GAN discriminator distinguishes real from fake (adversarial) images:
- Discriminator score **< 0.5** → image flagged as adversarial

### 4. Final Decision
An image is classified as **poisoned** only when **both** conditions are met:
```
is_poisoned = (real_score < 0.5) AND (reconstruction_error > 0.05)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- NVIDIA GPU (recommended) with CUDA 12.x
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Kamineni-Pradeep/GAN-DAE-Adversarial-Poisoning-Archaeological-Images.git
cd GAN-DAE-Adversarial-Poisoning-Archaeological-Images

# Create virtual environment
python -m venv env
env\Scripts\activate        # Windows
source env/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Model Files
The pre-trained model files (`gan.h5`, `discriminator.h5`, `denoising_autoencoder.h5`) are not included in this repository due to their size. Place them in the `Models/` folder:

```
Models/
├── gan.h5
├── discriminator.h5
└── denoising_autoencoder.h5
```

### Run the Application

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 🖥️ Web Application

The Flask web app allows users to:
1. Upload one or multiple archaeological images
2. Automatically generate adversarial versions using IFGSM
3. View poisoning detection results with scores
4. See reconstruction error and discriminator scores per image

### Folder Structure
```
project/
├── app.py                  ← Main Flask application
├── requirements.txt        ← Python dependencies
├── .gitignore
├── Models/                 ← Pre-trained model files (not tracked)
│   ├── gan.h5
│   ├── discriminator.h5
│   └── denoising_autoencoder.h5
├── templates/
│   ├── index.html          ← Upload page
│   └── results.html        ← Results dashboard
└── static/
│   ├── uploads/            ← Uploaded original images
│   ├── adversarial/        ← orignial adversarial images
│   ├──adversarial_images   ← Generated adversarial images
│
├── Uploads                 ← images upload During Testing
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Poisoning Detection Threshold (Reconstruction Error) | 0.05 |
| Discriminator Real Score Threshold | 0.5 |
| IFGSM Epsilon | 0.1 |
| IFGSM Iterations | 10 |
| Image Resolution | 256 × 256 |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | TensorFlow 2.x / Keras |
| Adversarial Attack | IFGSM (custom implementation) |
| Defense Model 1 | Denoising Autoencoder (DAE) |
| Defense Model 2 | Generative Adversarial Network (GAN) |
| Web Framework | Flask |
| Image Processing | Pillow, NumPy |
| Frontend | HTML, CSS, Jinja2 |

---

## 📝 Publication

This project is based on research published at the **ICEARS2025 International Conference**:

> *Mitigating Iterative Fast Gradient Adversarial Poisoning in Archaeological Image Datasets Using Generative Adversarial Networks and Denoising Autoencoders*
> Pradeep Kamineni — ICEARS2025, December 2025

---

## 👤 Author

**Pradeep Kamineni**
MSc Artificial Intelligence — Brunel University London
B.Tech AI & Data Science — Lakireddy Bali Reddy College of Engineering (CGPA: 8.57)

📧 pradeepkamineni74@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/pradeep-kamineni-72792b24a)
🐙 [GitHub](https://github.com/Kamineni-Pradeep)

---

## 📄 Licence

This project is licensed under the MIT Licence — see the [LICENSE](LICENSE) file for details.
