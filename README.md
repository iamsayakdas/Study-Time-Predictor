# 📘 Study Time Predictor

> 🔮 A machine learning–powered web app that predicts how much time a student studies daily, based on their reading habits and screen time.

---

## 🌐 Live Demo  
🎯 **Try it here:** [Study Time Predictor on Render](https://study-time-predictor.onrender.com)

---

## 🧠 Project Overview
Faculty often want to estimate how much each student studies daily for better mentoring.  
This project builds a **regression model** that predicts study time using features like:
- 📚 Number of books read per year  
- 🎭 Favorite book genre  
- 📺 Weekly screen time (movies/series)

The app uses **Flask** for the backend and a simple HTML/CSS frontend.  
It’s trained on sample survey data (`Test Data.csv`) and provides quick predictions in a web interface.

---

## 🧩 Tech Stack
| Component | Technology |
|------------|-------------|
| **Frontend** | HTML, CSS (via Flask templates) |
| **Backend** | Python, Flask |
| **Machine Learning** | scikit-learn, pandas, joblib |
| **Deployment** | Render (Free Web Service) |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/iamsayakdas/Study-Time-Predictor.git
cd Study-Time-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask app
python app.py
