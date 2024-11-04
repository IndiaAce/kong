import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle


def load_cleaned_data():
    # Load cleaned data from JSON and CSV files
    mitre_file = "data/processed_data/mitre_attack_cleaned.json"
    escu_file = "data/processed_data/splunk_escu_cleaned.json"
    splunk_docs_file = "data/processed_data/splunk_docs_cleaned.csv"

    # Output file sizes for debugging purposes
    print(f"MITRE ATT&CK file size: {os.path.getsize(mitre_file) / 1024:.2f} KB")
    print(f"Splunk ESCU file size: {os.path.getsize(escu_file) / 1024:.2f} KB")
    print(f"Splunk Docs file size: {os.path.getsize(splunk_docs_file) / 1024:.2f} KB")

    with open(mitre_file, "r", encoding="utf-8") as f:
        mitre_data = json.load(f)

    with open(escu_file, "r", encoding="utf-8") as f:
        escu_data = json.load(f)

    splunk_docs_data = pd.read_csv(splunk_docs_file)

    print(f"Loaded MITRE data: {len(mitre_data)} records")
    print(f"Loaded ESCU data: {len(escu_data)} records")
    print(f"Loaded Splunk Docs data: {len(splunk_docs_data)} records")

    return mitre_data, escu_data, splunk_docs_data


def merge_datasets(mitre_data, escu_data, splunk_docs_data):
    # Convert MITRE and ESCU data to DataFrames
    mitre_df = pd.DataFrame(mitre_data)
    escu_df = pd.DataFrame(escu_data)

    # Merge the data on common attributes if any (e.g., attack pattern names)
    # For simplicity, we're concatenating all datasets here
    combined_df = pd.concat([mitre_df, escu_df, splunk_docs_data], axis=0, ignore_index=True)
    
    # Handle missing values more gracefully by filling with placeholders
    combined_df.fillna("missing", inplace=True)

    print(f"Combined dataset size after merging: {len(combined_df)} records")
    print(f"Combined dataset columns: {combined_df.columns}")
    print(combined_df.head())

    return combined_df


def encode_and_tokenize_data(df):
    # Encode categorical variables
    label_encoder = LabelEncoder()
    if "name" in df.columns:
        df["name_encoded"] = label_encoder.fit_transform(df["name"])
    
    # Tokenize text data using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    if "description" in df.columns:
        # Check if there is enough valid content to tokenize
        non_empty_descriptions = df["description"].str.strip().replace('missing', np.nan).dropna()
        print(f"Number of non-empty descriptions: {len(non_empty_descriptions)}")
        if len(non_empty_descriptions) > 0:
            tfidf_matrix = tfidf_vectorizer.fit_transform(non_empty_descriptions)
            if tfidf_matrix.shape[1] > 0:  # Ensure there are valid tokens
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
                df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
            else:
                print("Warning: No valid tokens found in the description column.")
        else:
            print("Warning: The description column contains insufficient valid content for tokenization.")
    
    print(f"Dataset size after encoding and tokenization: {len(df)} records")
    print(f"Dataset columns after encoding and tokenization: {df.columns}")
    print(df.head())

    return df, label_encoder, tfidf_vectorizer


def split_data(df):
    # Define features and target
    if "name_encoded" in df.columns:
        X = df.drop(columns=["name", "description", "name_encoded"])
        y = df["name_encoded"]
    else:
        X = df.drop(columns=["name", "description"])
        y = df["name"]

    # Check if there are enough samples to split the data
    print(f"Number of samples available for splitting: {len(df)}")
    if len(df) < 2:
        raise ValueError("Not enough data to perform a train/test split. Please provide more data.")

    # Split the data into training, validation, and test sets
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        print(f"Training set size: {len(X_train)} records")
        print(f"Validation set size: {len(X_val)} records")
        print(f"Test set size: {len(X_test)} records")
    except ValueError as e:
        print(f"Error during train/test split: {e}")
        X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_prepared_data(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, tfidf_vectorizer):
    # Save the prepared data and encoders for future use
    data_dir = "data/prepared_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if X_train is not None:
        X_train.to_csv(os.path.join(data_dir, "X_train.csv"), index=False)
        X_val.to_csv(os.path.join(data_dir, "X_val.csv"), index=False)
        X_test.to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(data_dir, "y_train.csv"), index=False)
        y_val.to_csv(os.path.join(data_dir, "y_val.csv"), index=False)
        y_test.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)

        with open(os.path.join(data_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)

        with open(os.path.join(data_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(tfidf_vectorizer, f)

        print("Prepared data saved to data/prepared_data directory.")
    else:
        print("No data saved due to insufficient samples.")


if __name__ == "__main__":
    mitre_data, escu_data, splunk_docs_data = load_cleaned_data()
    combined_df = merge_datasets(mitre_data, escu_data, splunk_docs_data)
    processed_df, label_encoder, tfidf_vectorizer = encode_and_tokenize_data(combined_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_df)
    save_prepared_data(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, tfidf_vectorizer)