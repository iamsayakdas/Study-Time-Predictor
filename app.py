import os
from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Paths
CSV_PATH = "Test Data.csv"  # Keep the original file in repo root
MODEL_PATH = "model.pkl"
UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.secret_key = "supersecretkey"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------------
# Utility: Load and prepare data
# -------------------------------------------------------
def load_and_prepare_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    # Define your dataset’s column names exactly
    screen_col = '  Screen Time Movies/series in hours per week  \n(Provide value between 0-40)'
    reads_col = '  Reads Books   '
    genre_col = '  Book Genre Top 1'
    target_col = '  Books read past year\nProvide in integer value between (0-50)  '

    selected_cols = [reads_col, genre_col, screen_col, target_col]
    df_selected = df[selected_cols].copy()

    # Clean and process data
    df_selected[target_col] = df_selected[target_col].astype(str).str.split('-').str[0]
    df_selected[target_col] = pd.to_numeric(df_selected[target_col], errors='coerce')

    df_selected[screen_col] = pd.to_numeric(df_selected[screen_col], errors='coerce')
    df_selected[screen_col] = df_selected[screen_col].clip(lower=0)
    df_selected[screen_col] = df_selected[screen_col].fillna(df_selected[screen_col].median())

    df_selected[reads_col] = df_selected[reads_col].fillna("No")
    df_selected[genre_col] = df_selected[genre_col].fillna("Unknown")

    # Encode categorical features
    categorical_cols = [reads_col, genre_col]
    df_processed = pd.get_dummies(df_selected, columns=categorical_cols, drop_first=True)

    # Drop NaN targets
    df_processed = df_processed.dropna(subset=[target_col])

    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col].astype(float)

    return X, y, df_processed.columns.tolist()


# -------------------------------------------------------
# Utility: Train and save model
# -------------------------------------------------------
def train_and_save_model(csv_path=CSV_PATH):
    X, y, cols = load_and_prepare_data(csv_path)
    if X.shape[0] < 2:
        raise RuntimeError("Not enough training data to train model.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_PATH)
    print("✅ Model trained and saved to", MODEL_PATH)
    return model, X.columns.tolist()


# -------------------------------------------------------
# Load model (or train if not exists)
# -------------------------------------------------------
def load_model():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data["model"], data["columns"]
    else:
        return train_and_save_model()


# Global model load
model, model_columns = load_model()


# -------------------------------------------------------
# Helper: Prepare input for prediction
# -------------------------------------------------------
def prepare_input_dict(form):
    reads_input = form.get("reads", "No")
    genre_input = form.get("genre", "Unknown")
    try:
        screen_input = float(form.get("screen", 0))
    except:
        screen_input = 0.0

    # Build feature vector
    input_dict = {c: 0 for c in model_columns}

    # Find screen time column
    screen_col_candidates = [c for c in model_columns if 'Screen Time' in c]
    if screen_col_candidates:
        input_dict[screen_col_candidates[0]] = screen_input

    # One-hot encodings
    reads_dummy = '  Reads Books   _' + str(reads_input)
    genre_dummy = '  Book Genre Top 1_' + str(genre_input)

    if reads_dummy in input_dict:
        input_dict[reads_dummy] = 1
    if genre_dummy in input_dict:
        input_dict[genre_dummy] = 1

    X_input = pd.DataFrame([input_dict], columns=model_columns)
    return X_input


# -------------------------------------------------------
# Routes
# -------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    df = pd.read_csv(CSV_PATH)
    genres = sorted(df['  Book Genre Top 1'].dropna().unique().tolist())
    reads_options = sorted(df['  Reads Books   '].dropna().unique().tolist())

    if 'Unknown' not in genres:
        genres = ['Unknown'] + genres
    if 'No' not in reads_options:
        reads_options = ['No'] + reads_options

    return render_template("index.html",
                           genres=genres,
                           reads_options=reads_options,
                           prediction=False)


@app.route("/predict", methods=["POST"])
def predict():
    global model, model_columns
    X_input = prepare_input_dict(request.form)
    pred_books = float(model.predict(X_input)[0])
    pred_books = max(0.0, pred_books)
    est_daily_hours = round((pred_books * 8) / 365, 2)

    return render_template("index.html",
                           prediction=True,
                           pred_books=round(pred_books, 2),
                           est_daily_hours=est_daily_hours,
                           genres=[],
                           reads_options=[])


@app.route("/upload", methods=["POST"])
def upload_dataset():
    """Upload CSV and retrain model"""
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
        from flask import send_file
        return send_file(MODEL_PATH, as_attachment=True)
    else:
        flash("⚠️ Model file not found.")
        return redirect(url_for("index"))


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
