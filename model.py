from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import pandas as pd
from pymongo import MongoClient

def train_model():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Mental_Health"]
    collection = db["students_mental_health"]
    
    predictors = ["Sleep_Quality", "Physical_Activity", "Diet_Quality", "CGPA", "Gender", "Financial_Stress"]
    data = list(collection.find({}, {field: 1 for field in predictors + ["Stress_Level", "_id"]}))
    df = pd.DataFrame(data)
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
    
    # Prepare features
    X = pd.get_dummies(df[predictors], drop_first=True)
    y = (df["Stress_Level"] > 2).astype(int)  # Binary target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Logistic Regression
    lr_model = LogisticRegression(class_weight="balanced", max_iter=200)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_metrics = {
        "Accuracy": round(accuracy_score(y_test, lr_pred), 2),
        "Precision": round(precision_score(y_test, lr_pred), 2),
        "Recall": round(recall_score(y_test, lr_pred), 2),
        "F1 Score": round(f1_score(y_test, lr_pred), 2)
    }
    lr_importance = dict(zip(X.columns, lr_model.coef_[0]))
    lr_sorted_importance = {k: round(float(v), 2) for k, v in sorted(lr_importance.items(), key=lambda x: abs(x[1]), reverse=True)}
    print("Logistic Regression Performance:", lr_metrics)
    print("Logistic Regression Feature Importance:", lr_sorted_importance)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = {
        "Accuracy": round(accuracy_score(y_test, rf_pred), 2),
        "Precision": round(precision_score(y_test, rf_pred), 2),
        "Recall": round(recall_score(y_test, rf_pred), 2),
        "F1 Score": round(f1_score(y_test, rf_pred), 2)
    }
    rf_importance = dict(zip(X.columns, rf_model.feature_importances_))
    rf_sorted_importance = {k: round(float(v), 2) for k, v in sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)}
    print("Random Forest Performance:", rf_metrics)
    print("Random Forest Feature Importance:", rf_sorted_importance)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    df["Cluster"] = clusters
    
    # Update MongoDB in batches to avoid pipeline length error
    for i, row in df.iterrows():
        collection.update_one({"_id": row["_id"]}, {"$set": {"Cluster": int(row["Cluster"])}})
    print("K-Means Cluster Counts:", df["Cluster"].value_counts().to_dict())
    
    # Return Random Forest results (better performance)
    return rf_model, rf_metrics, rf_sorted_importance

if __name__ == "__main__":
    train_model()