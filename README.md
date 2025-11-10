<h1 align="center">🎓 Study Time Predictor</h1>

<p align="center">
  <em>Predict how much time students study daily — powered by Machine Learning & Flask</em>  
</p>

<p align="center">
  <a href="https://study-time-predictor.onrender.com" target="_blank">
    <img src="https://img.shields.io/badge/Live%20Demo-Visit%20Now-blue?style=for-the-badge&logo=google-chrome" alt="Live Demo" />
  </a>
  <img src="https://img.shields.io/badge/Machine%20Learning-LinearRegression-orange?style=for-the-badge&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Framework-Flask-green?style=for-the-badge&logo=flask" />
</p>

---

## 🧠 Overview

**Study Time Predictor** is a web application that estimates how much time a student studies daily based on their reading habits and screen-time behavior.  
It was developed as part of an academic project to help faculty understand student learning patterns and provide better mentorship.

---

## ✨ Features

✅ **Machine Learning-based Predictions** — Uses a regression model trained on real survey data  
✅ **Dataset Uploading** — Upload your own `.csv` file to retrain the model live  
✅ **Download Model** — Download the trained `model.pkl` file  
✅ **Modern UI** — Beautiful gradient background, responsive layout, and smooth animations  
✅ **Live Deployment** — Hosted on Render for free public access  

---

## 🌐 Live Demo

🎯 **Try it now:**  
👉 [https://study-time-predictor.onrender.com](https://study-time-predictor-zmsl.onrender.com)

---

## ⚙️ Tech Stack

| Component | Technology Used |
|------------|-----------------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Python (Flask Framework) |
| **Machine Learning** | scikit-learn, pandas, numpy |
| **Deployment** | Render (Free Web Service) |

---

## 🧩 How It Works

1. Loads dataset (`Test Data.csv` or your uploaded file)  
2. Preprocesses features — books read, genre, and screen time  
3. Trains a `LinearRegression` model using scikit-learn  
4. Exposes endpoints:
   - `/predict` → for predictions  
   - `/upload` → for dataset upload & retraining  
   - `/download_model` → to download your trained model  
5. Displays predictions with a clean UI and dynamic feedback

---

## 🚀 Run Locally

Clone the repository and install dependencies:

```bash
git clone https://github.com/iamsayakdas/Study-Time-Predictor.git
cd Study-Time-Predictor

| Name            | Role                                       |
| --------------- | ------------------------------------------ |
| **Sayak Das**   | Team Lead • Model Development • Deployment |
| **Sudip Ghosh** | Data Preprocessing • Testing               |
| **Sayak Ghosh** | Frontend Design • Documentation            |


🏫 Academic Project

This project was created as part of an academic coursework on Machine Learning Applications in Education, demonstrating integration of regression models with full-stack web deployment.

🪪 License & Copyright

© 2025 Sayak Das, Sudip Ghosh, Sayak Ghosh
All Rights Reserved.
This project is open-sourced for educational and research purposes under the MIT License
.

<p align="center"> <em>“Learn smart. Predict smarter.”</em> 🌟 </p> ```
pip install -r requirements.txt
python app.py
