from pymongo import MongoClient
import pandas as pd
import os

def preprocess_and_upload_data(file_path="students_mental_health.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # MongoDB connection
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Mental_Health"]
    collection = db["students_mental_health"]
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Define predictors to impute
    predictors = ["Sleep_Quality", "Physical_Activity", "Diet_Quality", "CGPA", "Gender", "Financial_Stress", "Substance_Use"]
    
    # Impute missing values
    for col in predictors:
        if col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    # Convert categorical columns to category type
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].astype('category')
    
    # Drop existing collection and upload fresh data
    collection.drop()
    records = df.to_dict(orient="records")
    collection.insert_many(records)
    
    print("Dataset uploaded to MongoDB successfully!")
    return collection

if __name__ == "__main__":
    preprocess_and_upload_data()