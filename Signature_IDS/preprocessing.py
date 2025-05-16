import pandas as pd
import os
import numpy as np
import gc
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

drive.mount('/content/drive')

# Load dataset from Google Drive
DATASET_PATH = "/content/drive/My Drive/UNSW_NB15_part1.parquet"
CACHE_PATH = "/content/drive/My Drive/preprocessed_data.joblib"

# Check if file exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}\nCheck if the file is correctly uploaded to Google Drive!")

def get_preprocessed_data(use_cache=False):
    if use_cache and os.path.exists(CACHE_PATH):
        print("Loading preprocessed data from cache...")
        return joblib.load(CACHE_PATH)

    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"Dataset loaded! Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    gc.collect()

    TARGET_COLUMN = "label"
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Error: Target column '{TARGET_COLUMN}' not found!")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Handle non-numeric columns
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        print(f"Non-numeric columns found: {non_numeric_cols}")
        print("Converting categorical columns using 'category' dtype...")
        for col in non_numeric_cols:
            X[col] = X[col].astype("category").cat.codes.astype(np.int16)
        print("All categorical columns converted successfully.")
    else:
        print("No categorical columns found.")

    X = X.astype(np.float32)
    y = y.astype("category")  # Memory optimization

    # Stratified Train-Test Split for multiclass detection
    print("Splitting full dataset into Train & Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Encode multi-class labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train).astype(np.uint8)
    y_test_encoded = label_encoder.transform(y_test).astype(np.uint8)

    # Show unique labels
    print("Unique labels in original 'label' column:\n", y.unique())

    gc.collect()

    print("Preprocessing complete.")
    joblib.dump(label_encoder, "/content/drive/My Drive/label_encoder.joblib")
    joblib.dump(X_train.columns.tolist(), "/content/drive/My Drive/feature_columns.joblib")
    joblib.dump(X_test.index, "/content/drive/My Drive/test_indexes.joblib")

    # Prepare final processed data
    processed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train_encoded,
        "y_test": y_test_encoded,
        "label_encoder": label_encoder
    }

    # Append IP and attack category info for visualization
    if "srcip" in df.columns and "attack_cat" in df.columns:
        processed_data["ip_test"] = df["srcip"].iloc[X_test.index].reset_index(drop=True)
        processed_data["attack_cat_test"] = df["attack_cat"].iloc[X_test.index].astype(str).reset_index(drop=True)
        processed_data["class_names"] = sorted(df["label"].astype("category").cat.categories.tolist())

    # Save to cache if needed
    if use_cache:
        joblib.dump(processed_data, CACHE_PATH)
        print("Preprocessed data cached for future use.")

    return processed_data

# Run preprocessing and store returned values
processed_data = get_preprocessed_data()
