import os
import re
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# -------------------------------------------------------
# ⚙️ Configuration
# -------------------------------------------------------
CSV_PATH = "Test Data.csv"  # Default dataset
MODEL_PATH = "model.pkl"
UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.secret_key = "supersecretkey"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------------
# 🔍 Column Detection (Explicit + Smart Fallback)
# -------------------------------------------------------
def detect_columns(df):
    """
    Detects key columns for Reads, Genre, Screen Time, and Target.
    Explicitly maps your dataset headers and provides fallbacks for new ones.
    """
    col_map = {"reads": None, "genre": None, "screen": None, "target": None}

    # Explicit mapping for your dataset
    for col in df.columns:
        name = col.strip().lower()
        if "reads books" in name:
            col_map["reads"] = col
        elif "book genre top 1" in name:
            col_map["genre"] = col
        elif "screen time movies" in name:
            col_map["screen"] = col
        elif "books read past year" in name:
            col_map["target"] = col

    # Fallback detection for unknown datasets
    for col in df.columns:
        n = col.strip().lower()
        if not col_map["reads"] and "read" in n:
            col_map["reads"] = col
        if not col_map["genre"] and ("genre" in n or "category" in n):
            col_map["genre"] = col
        if not col_map["screen"] and ("screen" in n or "hour" in n or "time" in n):
            col_map["screen"] = col
        if not col_map["target"] and ("target" in n or "books" in n or "past" in n):
            col_map["target"] = col

    # Final fallback — assign defaults if still missing
    for k, v in col_map.items():
        if v is None:
            col_map[k] = df.columns[0]

    return col_map


# -------------------------------------------------------
# 🧠 Data Preparation
# -------------------------------------------------------
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    col_map = detect_columns(df)

    reads_col = col_map["reads"]
    genre_col = col_map["genre"]
    screen_col = col_map["screen"]
    target_col = col_map["target"]

    df = df[[reads_col, genre_col, screen_col, target_col]].copy()

    # Clean and convert
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[screen_col] = pd.to_numeric(df[screen_col], errors="coerce").fillna(0)
    df[reads_col] = df[reads_col].fillna("No")
    df[genre_col] = df[genre_col].fillna("Unknown")

    df = df.dropna(subset=[target_col])

    # Encode categorical variables
    df = pd.get_dummies(df, columns=[reads_col, genre_col], drop_first=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y, df.columns.tolist()


# -------------------------------------------------------
# 🏋️‍♂️ Train & Save Model
# -------------------------------------------------------
def train_and_save_model(csv_path=CSV_PATH):
    X, y, cols = load_and_prepare_data(csv_path)
    if len(X) < 2:
        raise RuntimeError("Not enough training data to train model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_PATH)
    print("✅ Model retrained and saved successfully.")
    return model, X.columns.tolist()


# -------------------------------------------------------
# 📦 Load Model
# -------------------------------------------------------
def load_model():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data["model"], data["columns"]
    else:
        return train_and_save_model()


model, model_columns = load_model()


# -------------------------------------------------------
# 🧩 Prepare Input for Prediction
# -------------------------------------------------------
def prepare_input_dict(form):
    reads_input = form.get("reads", "No")
    genre_input = form.get("genre", "Unknown")
    try:
        screen_input = float(form.get("screen", 0))
    except:
        screen_input = 0.0

    input_dict = {c: 0 for c in model_columns}

    # Find screen time column
    screen_cols = [c for c in model_columns if re.search("screen|hour|time", c, re.I)]
    if screen_cols:
        input_dict[screen_cols[0]] = screen_input

    # One-hot encode reads & genre
    for c in model_columns:
        if reads_input.lower() in c.lower():
            input_dict[c] = 1
        if genre_input.lower() in c.lower():
            input_dict[c] = 1

    X_input = pd.DataFrame([input_dict], columns=model_columns)
    return X_input


# -------------------------------------------------------
# 🌐 Routes
# -------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    latest_csv = CSV_PATH
    uploads = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".csv")]
    if uploads:
        uploads.sort(
            key=lambda f: os.path.getmtime(os.path.join(UPLOAD_FOLDER, f)),
            reverse=True,
        )
        latest_csv = os.path.join(UPLOAD_FOLDER, uploads[0])

    detected_info = ""
    preview = None

    try:
        df = pd.read_csv(latest_csv)
        col_map = detect_columns(df)

        genre_col = col_map["genre"]
        reads_col = col_map["reads"]

        # Safely extract dropdown values
        genres = sorted(df[genre_col].dropna().astype(str).unique().tolist())
        reads_options = sorted(df[reads_col].dropna().astype(str).unique().tolist())

        detected_info = (
            f"📊 Detected → Reads: {col_map['reads']} | "
            f"Genre: {col_map['genre']} | "
            f"Screen: {col_map['screen']} | "
            f"Target: {col_map['target']}"
        )

        preview = df.head(5).to_html(classes="preview-table", index=False)
    except Exception as e:
        print("⚠️ Error loading dataset:", e)
        genres, reads_options = ["Unknown"], ["No"]
        detected_info = "⚠️ Could not detect column names. Using defaults."

    if "Unknown" not in genres:
        genres = ["Unknown"] + genres
    if "No" not in reads_options:
        reads_options = ["No"] + reads_options

    return render_template(
        "index.html",
        genres=genres,
        reads_options=reads_options,
        prediction=False,
        detected_info=detected_info,
        preview=preview,
    )


@app.route("/predict", methods=["POST"])
def predict():
    global model, model_columns
    X_input = prepare_input_dict(request.form)
    pred_books = float(model.predict(X_input)[0])
    pred_books = max(0.0, pred_books)
    est_daily_hours = round((pred_books * 8) / 365, 2)

    return render_template(
        "index.html",
        prediction=True,
        pred_books=round(pred_books, 2),
        est_daily_hours=est_daily_hours,
        genres=[],
        reads_options=[],
        detected_info="",
        preview=None,
    )


@app.route("/upload", methods=["POST"])
def upload_dataset():
    if "dataset" not in request.files:
        flash("⚠️ No file uploaded.")
        return redirect(url_for("index"))

    file = request.files["dataset"]
    if file.filename == "":
        flash("⚠️ Please select a file before uploading.")
        return redirect(url_for("index"))

    if not file.filename.lower().endswith(".csv"):
        flash("❌ Only CSV files are allowed.")
        return redirect(url_for("index"))

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        global model, model_columns
        model, model_columns = train_and_save_model(filepath)
        flash("✅ Dataset uploaded and model retrained successfully!")
    except Exception as e:
        flash(f"❌ Error processing dataset: {e}")

    return redirect(url_for("index"))


@app.route("/download_model")
def download_model():
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True)
    else:
        flash("⚠️ Model file not found.")
        return redirect(url_for("index"))


# -------------------------------------------------------
# 🚀 Run App
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
