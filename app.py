from flask import Flask, render_template, jsonify, request
import pymongo
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import numpy as np

app = Flask(__name__)

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Mental_Health"]
collection = db["students_mental_health"]

# Optimize with indexes (check if they exist first to avoid duplicate errors)
existing_indexes = collection.index_information()
for field in ["Stress_Level", "Course", "Sleep_Quality", "Anxiety_Score", "Depression_Score", 
              "Physical_Activity", "Diet_Quality", "Social_Support", "Gender", "Chronic_Illness", 
              "Counseling_Service_Use", "Relationship_Status", "Extracurricular_Involvement", 
              "Residence_Type", "Semester_Credit_Load", "Financial_Stress", "CGPA", "Age"]:
    index_name = f"{field}_1"
    if index_name not in existing_indexes:
        collection.create_index([(field, 1)])

# Load and preprocess data
data = pd.DataFrame(list(collection.find()))
if data.empty:
    raise ValueError("No data found in the 'Mental_Health' collection.")
if '_id' in data.columns:
    data = data.drop('_id', axis=1)

# Ensure required columns
required_columns = ['Stress_Level', 'Course', 'Sleep_Quality', 'Depression_Score', 'Anxiety_Score', 
                   'Physical_Activity', 'Diet_Quality', 'Social_Support', 'Semester_Credit_Load',
                   'Gender', 'Chronic_Illness', 'Counseling_Service_Use', 'Relationship_Status',
                   'Extracurricular_Involvement', 'Residence_Type', 'Financial_Stress', 'CGPA', 'Age']
missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Convert numeric columns to float to avoid type mismatches
numeric_cols = ['Stress_Level', 'Depression_Score', 'Anxiety_Score', 'Semester_Credit_Load', 
                'Financial_Stress', 'CGPA', 'Age']
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Prepare features and target
X = data.drop(['Stress_Level', 'Depression_Score', 'Anxiety_Score'], axis=1)
y = (data['Stress_Level'] > 2).astype(int)

# Impute missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

if numeric_cols.size > 0:
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
if categorical_cols.size > 0:
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# Convert categorical to dummy variables
X = pd.get_dummies(X)

# Split data and train model (ensure enough data for split)
if len(X) < 5:
    raise ValueError("Dataset too small for meaningful train-test split.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Metrics (handle zero-division cases)
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1 Score': f1_score(y_test, y_pred, zero_division=0)
}

# Feature importance
feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]

# Routes
@app.route('/')
def front():
    return render_template('front.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', metrics=metrics, feature_importance=top_features)

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')

@app.route('/add_data', methods=['POST'])
def add_data():
    new_student = request.json
    try:
        collection.insert_one(new_student)
        return jsonify({"message": "Data added successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Existing API Endpoints with Inferences
@app.route('/api/stress_by_course')
def stress_by_course():
    pipeline = [{"$group": {"_id": "$Course", "average_stress": {"$avg": "$Stress_Level"}}}, {"$sort": {"_id": 1}}]
    result = list(collection.aggregate(pipeline))
    inference = "Courses with higher stress may indicate heavier workloads or insufficient support."
    return jsonify({"data": [{"course": r["_id"], "average_stress": r["average_stress"]} for r in result], "inference": inference})

@app.route('/api/stress_by_sleep')
def stress_by_sleep():
    pipeline = [{"$group": {"_id": "$Sleep_Quality", "average_stress": {"$avg": "$Stress_Level"}}}, {"$sort": {"_id": 1}}]
    result = list(collection.aggregate(pipeline))
    inference = "Poor sleep quality correlates with higher stress, suggesting sleep interventions."
    return jsonify({"data": [{"sleep_quality": r["_id"], "average_stress": r["average_stress"]} for r in result], "inference": inference})

@app.route('/api/stress_distribution')
def stress_distribution():
    pipeline = [{"$group": {"_id": "$Stress_Level", "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}]
    result = list(collection.aggregate(pipeline))
    inference = "A skew toward higher stress levels suggests widespread mental health challenges."
    return jsonify({"data": [{"stress_level": r["_id"], "count": r["count"]} for r in result], "inference": inference})

@app.route('/api/high_stress_alerts')
def high_stress_alerts():
    result = list(collection.find({"Stress_Level": {"$gt": 4}}))
    inference = "Students with stress > 4 need urgent attention; high counts may indicate systemic issues."
    return jsonify({"data": [dict(doc, **{'_id': str(doc['_id'])}) for doc in result], "inference": inference})

@app.route('/api/anxiety_by_course')
def anxiety_by_course():
    pipeline = [{"$group": {"_id": "$Course", "average_anxiety": {"$avg": "$Anxiety_Score"}}}, {"$sort": {"_id": 1}}]
    result = list(collection.aggregate(pipeline))
    inference = "High-anxiety courses may require targeted mental health resources."
    return jsonify({"data": [{"course": r["_id"], "average_anxiety": r["average_anxiety"]} for r in result], "inference": inference})

@app.route('/api/depression_by_social_support')
def depression_by_social_support():
    pipeline = [{"$group": {"_id": "$Social_Support", "average_depression": {"$avg": "$Depression_Score"}}}, {"$sort": {"_id": 1}}]
    result = list(collection.aggregate(pipeline))
    inference = "Low social support links to higher depression, suggesting peer support initiatives."
    return jsonify({"data": [{"social_support": r["_id"], "average_depression": r["average_depression"]} for r in result], "inference": inference})

@app.route('/api/stress_by_activity')
def stress_by_activity():
    pipeline = [{"$group": {"_id": "$Physical_Activity", "average_stress": {"$avg": "$Stress_Level"}}}, {"$sort": {"_id": 1}}]
    result = list(collection.aggregate(pipeline))
    inference = "Low activity correlates with higher stress, promoting exercise may help."
    return jsonify({"data": [{"activity_level": r["_id"], "average_stress": r["average_stress"]} for r in result], "inference": inference})

@app.route('/api/stress_by_diet')
def stress_by_diet():
    pipeline = [{"$group": {"_id": "$Diet_Quality", "average_stress": {"$avg": "$Stress_Level"}}}, {"$sort": {"_id": 1}}]
    result = list(collection.aggregate(pipeline))
    inference = "Poor diet quality is associated with higher stress, indicating nutrition’s role."
    return jsonify({"data": [{"diet_quality": r["_id"], "average_stress": r["average_stress"]} for r in result], "inference": inference})

@app.route('/api/stress_by_credit_load')
def stress_by_credit_load():
    pipeline = [{"$group": {"_id": "$Semester_Credit_Load", "average_stress": {"$avg": "$Stress_Level"}}}, {"$sort": {"_id": 1}}]
    result = list(collection.aggregate(pipeline))
    inference = "Higher credit loads increase stress, suggesting academic load adjustments."
    return jsonify({"data": [{"credit_load": r["_id"], "average_stress": r["average_stress"]} for r in result], "inference": inference})

@app.route('/api/key_insights')
def key_insights():
    sleep_stress = next(collection.aggregate([{"$match": {"Sleep_Quality": "Poor"}}, {"$group": {"_id": None, "avg_stress": {"$avg": "$Stress_Level"}}}]), {"avg_stress": 0})["avg_stress"]
    high_stress_course = next(collection.aggregate([{"$group": {"_id": "$Course", "avg_stress": {"$avg": "$Stress_Level"}}}, {"$sort": {"avg_stress": -1}}, {"$limit": 1}]), {"_id": "Unknown", "avg_stress": 0})
    insights = {
        "Trends": [f"Poor sleep quality linked to avg stress of {sleep_stress:.2f}.", f"Course '{high_stress_course['_id']}' has highest stress (avg: {high_stress_course['avg_stress']:.2f})."],
        "Recommendations": ["Implement sleep education workshops.", f"Review '{high_stress_course['_id']}' workload."]
    }
    inference = "These insights highlight key areas for mental health interventions."
    return jsonify({"data": insights, "inference": inference})

@app.route('/api/data_distribution')
def data_distribution():
    fields = ['Stress_Level', 'Gender', 'Course', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality', 
              'Social_Support', 'Chronic_Illness', 'Counseling_Service_Use', 'Relationship_Status', 
              'Extracurricular_Involvement', 'Residence_Type']
    distributions = {}
    for field in fields:
        pipeline = [{"$group": {"_id": f"${field}", "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}]
        result = list(collection.aggregate(pipeline))
        distributions[field] = [{"category": str(r["_id"]), "count": r["count"]} for r in result]
    inference = "Skewed distributions may suggest sampling bias or dominant trends in the population."
    return jsonify({"data": distributions, "inference": inference})

@app.route('/api/data_balance')
def data_balance():
    fields = ['Stress_Level', 'Gender', 'Course', 'Sleep_Quality', 'Physical_Activity', 'Diet_Quality', 
              'Social_Support', 'Chronic_Illness', 'Counseling_Service_Use', 'Relationship_Status', 
              'Extracurricular_Involvement', 'Residence_Type']
    total_docs = collection.count_documents({})
    if total_docs == 0:
        return jsonify({"data": {}, "inference": "No data available for balance analysis."})
    balances = {}
    for field in fields:
        pipeline = [{"$group": {"_id": f"${field}", "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}]
        result = list(collection.aggregate(pipeline))
        balances[field] = [{"category": str(r["_id"]), "count": r["count"], "percentage": r["count"] / total_docs * 100} for r in result]
    inference = "Highly imbalanced data (e.g., <10% in a category) may affect model fairness or indicate underrepresentation."
    return jsonify({"data": balances, "inference": inference})

@app.route('/api/correlations')
def correlations():
    numeric_fields = ['Stress_Level', 'Depression_Score', 'Anxiety_Score', 'Financial_Stress', 'Semester_Credit_Load', 'CGPA', 'Age']
    df_numeric = data[numeric_fields].astype(float)
    corr_matrix = df_numeric.corr()
    correlations = []
    for i, var1 in enumerate(numeric_fields):
        for var2 in numeric_fields[i+1:]:
            corr_value = corr_matrix.loc[var1, var2]
            if pd.isna(corr_value):
                corr_value = 0.0
            correlations.append({"var1": var1, "var2": var2, "correlation": corr_value})
    inference = "Strong correlations (>0.5 or <-0.5) indicate significant relationships; weak ones (near 0) suggest independence."
    return jsonify({"data": correlations, "inference": inference})

if __name__ == '__main__':
    app.run(debug=True, port=5001)