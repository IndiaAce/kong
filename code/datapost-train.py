import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV


def load_prepared_data():
    # Load the prepared datasets
    data_dir = "data/prepared_data"
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # Replace 'missing' with NaN for proper imputation
    X_train.replace('missing', np.nan, inplace=True)
    X_val.replace('missing', np.nan, inplace=True)
    X_test.replace('missing', np.nan, inplace=True)

    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Check if 'description' column exists and preserve it for transformer training
    if 'description' in X_train.columns:
        X_train_description = X_train['description'].copy()
        X_val_description = X_val['description'].copy()
        X_test_description = X_test['description'].copy()
    else:
        X_train_description = pd.Series([''] * len(X_train))
        X_val_description = pd.Series([''] * len(X_val))
        X_test_description = pd.Series([''] * len(X_test))

    # Convert complex data types to strings
    for col in X_train.columns:
        if isinstance(X_train[col][0], (list, dict)):
            X_train[col] = X_train[col].apply(lambda x: str(x))
            X_val[col] = X_val[col].apply(lambda x: str(x))
            X_test[col] = X_test[col].apply(lambda x: str(x))

    # One-hot encode categorical columns
    categorical_columns = X_train.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_columns]))
    X_val_encoded = pd.DataFrame(encoder.transform(X_val[categorical_columns]))
    X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_columns]))

    # Drop original categorical columns and concatenate encoded columns
    X_train.drop(columns=categorical_columns, inplace=True)
    X_val.drop(columns=categorical_columns, inplace=True)
    X_test.drop(columns=categorical_columns, inplace=True)
    X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_val.reset_index(drop=True), X_val_encoded.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)

    # Encode labels using LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = pd.Series(y_val).map(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1).values
    y_test = pd.Series(y_test).map(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1).values

    # Balance the dataset using RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_description, X_val_description, X_test_description, label_encoder


def train_random_forest(X_train, y_train, X_val, y_val):
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best model from GridSearch
    rf_model = grid_search.best_estimator_

    # Evaluate the model
    y_val_pred = rf_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print("Random Forest Validation Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_val, y_val_pred, zero_division=1))

    # Save the trained model
    with open("models/random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    return rf_model


def train_transformer(X_train_description, y_train, X_val_description, y_val):
    # Tokenizer and model initialization
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y_train)))

    # Prepare data for the transformer
    train_encodings = tokenizer(list(X_train_description), truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_encodings = tokenizer(list(X_val_description), truncation=True, padding=True, max_length=128, return_tensors="pt")

    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # Training loop (simplified)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(5):  # Increased epochs for better training
        optimizer.zero_grad()
        outputs = model(**train_encodings, labels=train_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1} - Loss: {loss.item()}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(**val_encodings)
        predictions = torch.argmax(outputs.logits, dim=-1)
        accuracy = accuracy_score(val_labels, predictions)
        print("Transformer Validation Accuracy:", accuracy)

    # Save the trained model
    model.save_pretrained("models/transformer_model")
    tokenizer.save_pretrained("models/transformer_tokenizer")

    return model, tokenizer


def evaluate_model(model, X_test, y_test, model_type="random_forest", X_test_description=None):
    # Evaluate the model using the test set
    if model_type == "random_forest":
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Random Forest Test Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
    elif model_type == "transformer" and X_test_description is not None:
        tokenizer = AutoTokenizer.from_pretrained("models/transformer_tokenizer")
        test_encodings = tokenizer(list(X_test_description), truncation=True, padding=True, max_length=128, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            outputs = model(**test_encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = accuracy_score(y_test, predictions)
            print("Transformer Test Accuracy:", accuracy)


def generate_splunk_detection_yaml(model, user_input, model_type="random_forest"):
    # Generate a Splunk detection rule based on user input
    detection = {}

    if model_type == "random_forest":
        prediction = model.predict([user_input])[0]
        detection["name"] = f"Detection for {user_input}"
        detection["description"] = f"Generated based on user input: {user_input}"
        detection["search"] = f"search index=* {user_input}"
        detection["tags"] = {"generated": True}
    elif model_type == "transformer":
        tokenizer = AutoTokenizer.from_pretrained("models/transformer_tokenizer")
        encodings = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            outputs = model(**encodings)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            detection["name"] = f"Transformer Detection for {user_input}"
            detection["description"] = f"Generated based on user input: {user_input}"
            detection["search"] = f"search index=* {user_input}"
            detection["tags"] = {"generated": True}

    # Save detection to YAML
    yaml_file = f"detections/detection_{prediction}.yml"
    with open(yaml_file, "w") as file:
        yaml.dump(detection, file, default_flow_style=False)
    print(f"Detection rule saved to {yaml_file}")


if __name__ == "__main__":
    # Load the prepared data
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_description, X_val_description, X_test_description, label_encoder = load_prepared_data()

    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # Train and evaluate Random Forest model
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    evaluate_model(rf_model, X_test, y_test, model_type="random_forest")

    # Train and evaluate Transformer model
    transformer_model, transformer_tokenizer = train_transformer(X_train_description, y_train, X_val_description, y_val)
    evaluate_model(transformer_model, X_test, y_test, model_type="transformer", X_test_description=X_test_description)

    # Generate Splunk detection rules based on user input
    sample_user_input = "malicious activity detection"
    generate_splunk_detection_yaml(rf_model, sample_user_input, model_type="random_forest")
    generate_splunk_detection_yaml(transformer_model, sample_user_input, model_type="transformer")
