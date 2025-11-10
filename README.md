<h1 align="center">🎓 Study Time Predictor</h1>

<p align="center">
  <em>AI-powered prediction of student study habits — built using Flask & Machine Learning</em>
</p>

<p align="center">
  <a href="https://study-time-predictor.onrender.com" target="_blank">
    <img src="https://img.shields.io/badge/Live%20Demo-Click%20Here-4CAF50?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Live Demo"/>
  </a>
  <img src="https://img.shields.io/badge/Framework-Flask-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-LinearRegression-FF9800?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
</p>

---

## 🧠 Overview

**Study Time Predictor** is a full-stack machine learning web application that predicts how long students study daily based on their reading habits, screen time, and interests.  
It features an elegant, animated frontend and a live Flask backend deployed online using Render.

---

## ✨ Key Features

- 🤖 **AI-Powered Regression Model** — Predicts study hours using `scikit-learn`.
- 📂 **Dataset Upload** — Upload custom CSV files and retrain the model live.
- 💾 **Download Trained Model** — Instantly get your personalized `model.pkl`.
- 🎨 **Modern Animated UI** — Gradient backgrounds, glassmorphism design, and smooth transitions.
- 🌐 **Deployed Web App** — Available online for real-time testing.

---

## 🌍 Live Demo

🎯 Try it yourself:  
👉 [https://study-time-predictor.onrender.com](https://study-time-predictor.onrender.com)

---

## 🧩 Tech Stack

| Layer | Technology |
|--------|-------------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Python, Flask |
| **Machine Learning** | scikit-learn, pandas, numpy |
| **Deployment** | Render (Free Hosting) |
| **Version Control** | Git + GitHub |

---

## ⚙️ How It Works

1. Loads and cleans the dataset (`Test Data.csv` or user-uploaded).  
2. Encodes features and trains a **Linear Regression** model.  
3. Exposes endpoints:
   - `/upload` → Upload CSV and retrain model  
   - `/predict` → Predict study time  
   - `/download_model` → Download trained model  
4. Renders predictions beautifully in a web interface.

---

## 🚀 Run Locally

```bash
# Clone this repository
git clone https://github.com/iamsayakdas/Study-Time-Predictor.git
cd Study-Time-Predictor

# Install dependencies
pip install -r requirements.txt


## 👨‍💻 Project Team

| Name | Role |
|------|------|
| **Sayak Das** | Project Lead • Model Development • Deployment |
| **Sudip Ghosh** | Data Processing • Testing & Validation |
| **Sayak Ghosh** | Frontend Design • Documentation |


## 🏫 Academic Details

📚 *Developed as part of an academic project on Machine Learning & Predictive Modeling.*  
The goal was to integrate a predictive regression model within a user-friendly web interface and deploy it live for real-world usage.


## 🪪 License & Copyright

© **2025 Sayak Das, Sudip Ghosh, Sayak Ghosh**  
All Rights Reserved.  

This project is licensed under the [MIT License](LICENSE).  
You are free to use and modify it for educational and research purposes, with proper credit to the authors.
# Run the Flask app
python app.py
