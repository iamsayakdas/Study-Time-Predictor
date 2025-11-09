import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Paths
CSV_PATH = "Test Data.csv"  # keep the original file in repo root (or change path)
MODEL_PATH = "model.pkl"

app = Flask(__name__)

def load_and_prepare_data(csv_path=CSV_PATH):
    # Load raw CSV
    df = pd.read_csv(csv_path)

    # Columns used in the notebook
    screen_col = '  Screen Time Movies/series in hours per week  \n(Provide value between 0-40)'
    reads_col = '  Reads Books   '
    genre_col = '  Book Genre Top 1'
    target_col = '  Books read past year\nProvide in integer value between (0-50)  '

    selected_cols = [reads_col, genre_col, screen_col, target_col]
    df_selected = df[selected_cols].copy()

    # Clean target: convert ranges like "1-3" to lower bound, "No" to 0 etc.
    df_selected[target_col] = df_selected[target_col].astype(str).str.split('-').str[0]
    df_selected[target_col] = pd.to_numeric(df_selected[target_col], errors='coerce')

    # Clean screen time: remove non-numeric, coerce to numeric, fill NaN with median
    df_selected[screen_col] = pd.to_numeric(df_selected[screen_col], errors='coerce')
    df_selected[screen_col] = df_selected[screen_col].clip(lower=0)  # ensure non-negative
    df_selected[screen_col] = df_selected[screen_col].fillna(df_selected[screen_col].median())

    # For Reads Books categorical: fillna with 'No'
    df_selected[reads_col] = df_selected[reads_col].fillna("No")

    # For Book Genre fillna with 'Unknown'
    df_selected[genre_col] = df_selected[genre_col].fillna("Unknown")

    # One-hot encode categorical columns (drop_first=True to match notebook)
    categorical_cols = [reads_col, genre_col]
    df_processed = pd.get_dummies(df_selected, columns=categorical_cols, drop_first=True)

    # Drop rows where target is NA
    df_processed = df_processed.dropna(subset=[target_col])

    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col].astype(float)

    return X, y, df_processed.columns.tolist()

def train_and_save_model():
    X, y, columns = load_and_prepare_data()
    if X.shape[0] < 2:
        raise RuntimeError("Not enough training data to train model.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump({
        "model": model,
        "columns": X.columns.tolist()
    }, MODEL_PATH)
    print("Model trained and saved to", MODEL_PATH)
    return model, X.columns.tolist()

def load_model():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data['model'], data['columns']
    else:
        return train_and_save_model()

# Load model when starting app
model, model_columns = load_model()

@app.route("/", methods=["GET"])
def index():
    # Provide some helpful defaults/options for the front end
    # We'll extract unique genres and reads options from CSV so the UI can present them
    df = pd.read_csv(CSV_PATH)
    genres = sorted(df['  Book Genre Top 1'].dropna().unique().tolist())
    reads_options = sorted(df['  Reads Books   '].dropna().unique().tolist())
    # Ensure some defaults
    if 'Unknown' not in genres:
        genres = ['Unknown'] + genres
    if 'No' not in reads_options:
        reads_options = ['No'] + reads_options

    return render_template("index.html",
                           genres=genres,
                           reads_options=reads_options,
                           screen_col_label='Screen time (hours/week)',
                           reads_col_label='Reads books (select)',
                           genre_col_label='Book genre')

def prepare_input_dict(form):
    # Form contains:
    # 'reads' -> string like "Regularly" or "Sometimes" or "No"
    # 'genre' -> e.g. "Fiction"
    # 'screen' -> numeric (hours per week)
    reads_input = form.get("reads", "No")
    genre_input = form.get("genre", "Unknown")
    try:
        screen_input = float(form.get("screen", 0))
    except:
        screen_input = 0.0

    # Create a single-row dataframe with model columns
    input_dict = {c: 0 for c in model_columns}
    # set the numeric column (screen)
    # find the exact column name in the processed dataframe that corresponds to screen col
    # We assume the screen column name exists unchanged in model_columns:
    # (Based on preprocessing, it should).
    # Find any column that contains 'Screen Time' substring
    screen_col_candidates = [c for c in model_columns if 'Screen Time' in c]
    if len(screen_col_candidates) == 1:
        input_screen_col = screen_col_candidates[0]
    else:
        # fallback: choose first numeric-like column (not a dummy)
        numeric_cols = [c for c in model_columns if not any(prefix in c for prefix in ['  Reads Books', '  Book Genre Top 1'])]
        input_screen_col = numeric_cols[0] if numeric_cols else screen_col_candidates[0] if screen_col_candidates else None

    if input_screen_col:
        input_dict[input_screen_col] = screen_input

    # One-hot for reads: encoded columns are like '  Reads Books   _Regularly' etc.
    reads_col_prefix = '  Reads Books   _'
    reads_dummy = reads_col_prefix + str(reads_input)
    if reads_dummy in input_dict:
        input_dict[reads_dummy] = 1
    # If drop_first=True was used when training, the base category (e.g., 'No') may not have a dummy;
    # leaving all zeros corresponds to that base category.

    # One-hot for genre: '  Book Genre Top 1_<genre>'
    genre_prefix = '  Book Genre Top 1_'
    genre_dummy = genre_prefix + str(genre_input)
    if genre_dummy in input_dict:
        input_dict[genre_dummy] = 1

    # Return as DataFrame
    X_input = pd.DataFrame([input_dict], columns=model_columns)
    return X_input

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form or request.json or {}
    X_input = prepare_input_dict(form)
    pred_books = model.predict(X_input)[0]
    # Round and ensure non-negative
    pred_books = max(0.0, float(pred_books))
    pred_books_rounded = round(pred_books, 2)

    # Optionally convert predicted books per year to daily study hours estimate:
    # NOTE: this is a heuristic suggestion (assumption). We will offer a conservative conversion:
    # assume 1 book ~ 8 study hours (reading/studying time). This conversion is optional and can be changed.
    assumed_hours_per_book = 8
    est_total_hours_per_year = pred_books * assumed_hours_per_book
    est_daily_hours = est_total_hours_per_year / 365.0

    return render_template("index.html",
                           prediction=True,
                           pred_books=pred_books_rounded,
                           est_daily_hours=round(est_daily_hours, 2),
                           genres=[],
                           reads_options=[],
                           screen_col_label='Screen time (hours/week)',
                           reads_col_label='Reads books (select)',
                           genre_col_label='Book genre')

if __name__ == "__main__":
    # When running locally
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
